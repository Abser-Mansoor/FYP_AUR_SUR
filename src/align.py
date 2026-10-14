"""5-point face alignment for the diagnostic alignment ablation.

Detector: insightface SCRFD det_10g.onnx (buffalo_l). Decoding follows
insightface python-package/insightface/model_zoo/scrfd.py (forward/detect/nms).
Warp: Umeyama similarity to the ArcFace 112x112 template (as face_align.norm_crop).
"""
import cv2
import numpy as np

ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class SCRFD:
    strides = (8, 16, 32)
    num_anchors = 2

    def __init__(self, model_file, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        import onnxruntime
        onnxruntime.preload_dlls()  # locate CUDA 13 / cuDNN DLLs from site-packages/nvidia
        so =onnxruntime.SessionOptions()
        so.log_severity_level = 3  # det_10g was exported at 640px; other sizes trigger harmless shape warnings
        self.sess = onnxruntime.InferenceSession(model_file, sess_options=so, providers=list(providers))
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        assert len(self.output_names) == 9, "expected det_10g with keypoints"
        self._centers = {}

    def _anchor_centers(self, h, w, stride):
        key = (h, w, stride)
        if key not in self._centers:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            self._centers[key] = np.stack([c] * self.num_anchors, axis=1).reshape(-1, 2)
        return self._centers[key]

    def detect(self, bgr, input_size, thresh=0.5, nms_thresh=0.4):
        """Return (dets[N,5], kps[N,5,2]) in original image coordinates."""
        W, H = input_size
        im_ratio = bgr.shape[0] / bgr.shape[1]
        if im_ratio > H / W:
            nh, nw = H, int(H / im_ratio)
        else:
            nw, nh = W, int(W * im_ratio)
        scale = nh / bgr.shape[0]
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (W, H), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(self.output_names, {self.input_name: blob})
        batched = outs[0].ndim == 3
        scores_l, boxes_l, kps_l = [], [], []
        for i, s in enumerate(self.strides):
            sc, bb, kp = outs[i], outs[i + 3] * s, outs[i + 6] * s
            if batched:
                sc, bb, kp = sc[0], bb[0], kp[0]
            c = self._anchor_centers(H // s, W // s, s)
            keep = np.where(sc.ravel() >= thresh)[0]
            boxes = np.stack([c[:, 0] - bb[:, 0], c[:, 1] - bb[:, 1], c[:, 0] + bb[:, 2], c[:, 1] + bb[:, 3]], -1)
            kps = np.stack([c[:, j % 2] + kp[:, j] for j in range(10)], -1).reshape(-1, 5, 2)
            scores_l.append(sc.ravel()[keep]); boxes_l.append(boxes[keep]); kps_l.append(kps[keep])
        scores = np.concatenate(scores_l)
        if scores.size == 0:
            return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)
        order = np.argsort(-scores, kind="stable")
        dets = np.hstack([np.concatenate(boxes_l) / scale, scores[:, None]])[order].astype(np.float32)
        kps = (np.concatenate(kps_l) / scale)[order]
        keep = _nms(dets, nms_thresh)
        return dets[keep], kps[keep]


def _nms(dets, thresh):
    x1, y1, x2, y2, s = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-s, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        w = np.maximum(0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        inter = w * h
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter) <= thresh]
    return keep


def umeyama(src, dst):
    """Least-squares similarity transform src->dst (2x3), as skimage SimilarityTransform."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / s.var(0).sum()
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]]).astype(np.float32)


def norm_crop(bgr, kps, size=112):
    M = umeyama(np.asarray(kps, np.float32), ARCFACE_DST * size / 112.0)
    return cv2.warpAffine(bgr, M, (size, size), borderValue=0.0)


def detect_landmarks(det, bgr, lowres_side=None, thresh=0.3):
    """Landmarks of the most central/largest face, or None.

    Low-res crops fill the whole frame, so they are upsampled to `lowres_side`
    and padded by 50% on each side so the detector sees a face with context.
    """
    if lowres_side:
        f = lowres_side / max(bgr.shape[:2])
        up = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        p = lowres_side // 2
        img = cv2.copyMakeBorder(up, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        side = img.shape[1] // 32 * 32 + 32
        dets, kps = det.detect(img, (side, side), thresh)
        if len(dets) == 0:
            return None
        kps = (kps - p) / f
        dets = dets.copy(); dets[:, :4] = (dets[:, :4] - p) / f
    else:
        dets, kps = det.detect(bgr, (640, 640), thresh)
        if len(dets) == 0:
            return None
    h, w = bgr.shape[:2]
    cx, cy = (dets[:, 0] + dets[:, 2]) / 2 - w / 2, (dets[:, 1] + dets[:, 3]) / 2 - h / 2
    area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    return kps[int(np.argmax(area - 2.0 * (cx ** 2 + cy ** 2)))]
