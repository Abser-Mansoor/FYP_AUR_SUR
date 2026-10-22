"""Face detectors with one interface and three inference modes (Phase 5 detector study).

    det = SCRFDDetector("models/detector/det_10g.onnx")        # or RetinaFaceDetector("models/detector/retinaface/Resnet50_Final.pth")
    dets, kps = det.detect(bgr, size=640)                       # dets[N,5] = x1,y1,x2,y2,score ; kps[N,5,2] ; original image coordinates
    dets, kps = detect_mode(det, bgr, "tiled")                  # modes: full640 | full1280 | tiled

`size` is the length of the longest side the detector sees (the image is resized to it, aspect kept). `tiled` cuts the frame into
2x2 tiles with 20% overlap, runs each tile at `size` 640 and merges with NMS 0.4 (SAHI-style). All detectors return every box above
`thresh` (default 0.02, as the WIDER FACE protocol expects); callers apply their own score threshold afterwards.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch  # noqa: F401  must load before onnxruntime creates a CUDA session: a CUDA 13 cuDNN loaded first breaks torch's CUDA 12 cuDNN

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.align import SCRFD, _nms  # noqa: E402

TILE_GRID, TILE_OVERLAP, TILE_SIZE, NMS = 2, 0.2, 640, 0.4
MODES = {"full640": ("full", 640), "full1280": ("full", 1280), "tiled": ("tiled", TILE_SIZE)}


class SCRFDDetector:
    name = "scrfd"

    def __init__(self, model_file, providers=("CUDAExecutionProvider", "CPUExecutionProvider"), thresh=0.02, nms=NMS):
        self.det = SCRFD(str(model_file), providers)
        self.thresh, self.nms = thresh, nms

    def detect(self, bgr, size=640, thresh=None):
        return self.det.detect(bgr, (size, size), self.thresh if thresh is None else thresh, self.nms)


class RetinaFaceDetector:
    """biubug6/Pytorch_Retinaface ResNet-50, fp32. Decode and NMS follow its detect.py (top-k 5000, keep-top-k 750)."""
    name = "retinaface"
    MEAN = (104, 117, 123)

    def __init__(self, weights, device="cuda", thresh=0.02, nms=NMS):
        from src.video.vendor.retinaface.data.config import cfg_re50
        from src.video.vendor.retinaface.models.retinaface import RetinaFace
        cfg = {**cfg_re50, "pretrain": False}  # do not fetch ImageNet weights; the detector checkpoint is loaded below
        net = RetinaFace(cfg=cfg, phase="test")
        sd = torch.load(str(weights), map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=False)
        self.net, self.cfg, self.torch = net.eval().to(device), cfg, torch
        self.device, self.thresh, self.nms = device, thresh, nms
        self._priors = {}

    def _prior(self, h, w):
        if (h, w) not in self._priors:
            from src.video.vendor.retinaface.layers.functions.prior_box import PriorBox
            self._priors[(h, w)] = PriorBox(self.cfg, image_size=(h, w)).forward().to(self.device)
        return self._priors[(h, w)]

    def detect(self, bgr, size=640, thresh=None):
        from src.video.vendor.retinaface.utils.box_utils import decode, decode_landm
        from src.video.vendor.retinaface.utils.nms.py_cpu_nms import py_cpu_nms
        thresh = self.thresh if thresh is None else thresh
        torch = self.torch
        h0, w0 = bgr.shape[:2]
        f = size / max(h0, w0)
        im = cv2.resize(bgr, (round(w0 * f), round(h0 * f)), interpolation=cv2.INTER_LINEAR) if f != 1 else bgr
        h, w = im.shape[:2]
        x = torch.from_numpy((im.astype(np.float32) - np.array(self.MEAN, np.float32)).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            loc, conf, landms = self.net(x)
        prior = self._prior(h, w)
        var = self.cfg["variance"]
        boxes = decode(loc.squeeze(0), prior, var) * torch.tensor([w, h, w, h], device=self.device)
        kps = decode_landm(landms.squeeze(0), prior, var) * torch.tensor([w, h] * 5, device=self.device)
        scores = conf.squeeze(0)[:, 1]
        keep = scores > thresh
        boxes, kps, scores = boxes[keep].cpu().numpy(), kps[keep].cpu().numpy(), scores[keep].cpu().numpy()
        order = scores.argsort()[::-1][:5000]
        boxes, kps, scores = boxes[order], kps[order], scores[order]
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        k = py_cpu_nms(dets, self.nms)[:750]
        dets, kps = dets[k], kps[k].reshape(-1, 5, 2)
        dets[:, :4] /= f
        return dets, (kps / f).astype(np.float32)


def _tiles(h, w):
    """2x2 grid; each tile is (1/2 + overlap/2) of the frame per axis so neighbours overlap by TILE_OVERLAP of a tile."""
    th, tw = int(round(h / TILE_GRID * (1 + TILE_OVERLAP))), int(round(w / TILE_GRID * (1 + TILE_OVERLAP)))
    ys, xs = np.linspace(0, h - th, TILE_GRID).round().astype(int), np.linspace(0, w - tw, TILE_GRID).round().astype(int)
    return [(int(y), int(x), th, tw) for y in ys for x in xs]


def detect_mode(det, bgr, mode, thresh=None):
    kind, size = MODES[mode]
    if kind == "full":
        return det.detect(bgr, size, thresh)
    all_d, all_k = [], []
    for y, x, th, tw in _tiles(*bgr.shape[:2]):
        d, k = det.detect(bgr[y:y + th, x:x + tw], size, thresh)
        if len(d):
            d, k = d.copy(), k.copy()
            d[:, [0, 2]] += x
            d[:, [1, 3]] += y
            k[..., 0] += x
            k[..., 1] += y
            all_d.append(d)
            all_k.append(k)
    if not all_d:
        return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)
    d, k = np.concatenate(all_d), np.concatenate(all_k)
    order = np.argsort(-d[:, 4], kind="stable")
    d, k = d[order], k[order]
    keep = _nms(d, NMS)
    return d[keep], k[keep]


def crop_detect(det, bgr, lowres_side=112, thresh=0.3):
    """Landmarks of the most central/largest face in a low-resolution crop, or None. Same procedure as
    src/align.py:detect_landmarks (upsample to lowres_side, pad 50% per side), for any Detector."""
    f = lowres_side / max(bgr.shape[:2])
    up = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    p = lowres_side // 2
    img = cv2.copyMakeBorder(up, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    side = img.shape[1] // 32 * 32 + 32
    dets, kps = det.detect(img, side, thresh)
    if len(dets) == 0:
        return None
    kps = (kps - p) / f
    dets = dets.copy()
    dets[:, :4] = (dets[:, :4] - p) / f
    h, w = bgr.shape[:2]
    cx, cy = (dets[:, 0] + dets[:, 2]) / 2 - w / 2, (dets[:, 1] + dets[:, 3]) / 2 - h / 2
    area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    return kps[int(np.argmax(area - 2.0 * (cx ** 2 + cy ** 2)))]
