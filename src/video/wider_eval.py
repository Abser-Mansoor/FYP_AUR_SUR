"""WIDER FACE validation evaluation: the official easy/medium/hard AP protocol (NumPy port of the reference evaluation) and
recall / false positives per face-height bin.

Ground truth: Dataset/WIDER_FACE/eval_tools/ground_truth/*.mat (the official evaluation files).
Predictions: dict {(event, file_name): float array [N,5] = x1,y1,x2,y2,score}, original image coordinates.

AP follows the reference implementation: scores normalised over the whole set, 1000 thresholds, IoU 0.5, "ignore" faces (those
outside the easy/medium/hard subset) neither count as hits nor as false positives. The reference loops over thresholds per image;
here the same quantities are read off the descending score order with searchsorted (identical results, much faster).
Bin recall (ours, not part of the official protocol): a ground-truth face (invalid == 0) counts as found if any detection with
score >= thr has IoU >= 0.5 with it; false positives are detections with score >= thr whose IoU with every ground-truth box
(invalid ones included) is below 0.5.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .vendor._bbox import bbox_overlaps

BINS = [(0, 10), (10, 16), (16, 24), (24, 32), (32, 64), (64, 10 ** 6)]  # face height in px
BIN_NAMES = ["<10", "10-16", "16-24", "24-32", "32-64", ">64"]
THRESH_NUM, IOU = 1000, 0.5


def _s(x):
    while isinstance(x, np.ndarray):  # MATLAB cell/char wrappers nest the string in 1-element arrays
        x = x.ravel()[0]
    return str(x)


def load_gt(root):
    """Returns list of dict(event, file, boxes[N,4] x,y,w,h, keep{easy,medium,hard} = 0-based index arrays) and the txt attributes."""
    root = Path(root)
    gt = loadmat(root / "eval_tools/ground_truth/wider_face_val.mat")
    subs = {k: loadmat(root / f"eval_tools/ground_truth/wider_{k}_val.mat")["gt_list"] for k in ("easy", "medium", "hard")}
    out = []
    for e, ev in enumerate(gt["event_list"]):
        files, boxes = gt["file_list"][e][0], gt["face_bbx_list"][e][0]
        for i in range(len(files)):
            b = np.asarray(boxes[i][0], dtype=np.float64).reshape(-1, 4)
            keep = {k: np.asarray(v[e][0][i][0]).ravel().astype(int) - 1 for k, v in subs.items()}
            out.append(dict(event=_s(ev), file=_s(files[i]), boxes=b, keep=keep))
    return out


def load_attrs(root):
    """Per-image 'invalid' flags from wider_face_val_bbx_gt.txt: {file_stem: array of invalid per face}."""
    lines = (Path(root) / "wider_face_split/wider_face_val_bbx_gt.txt").read_text().splitlines()
    out, i = {}, 0
    while i < len(lines):
        name, n = lines[i].strip(), int(lines[i + 1])
        rows = lines[i + 2: i + 2 + max(n, 1)]
        inv = [int(r.split()[7]) for r in rows] if n > 0 else []
        out[Path(name).stem] = np.array(inv, dtype=int)
        i += 2 + max(n, 1)
    return out


def _voc_ap(rec, prec):
    mrec, mpre = np.concatenate(([0.0], rec, [1.0])), np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def average_precision(preds, gts):
    """Official AP for easy / medium / hard. preds: {(event, file): [N,5]}."""
    allp = [p[:, 4] for p in preds.values() if len(p)]
    lo, hi = min(s.min() for s in allp), max(s.max() for s in allp)
    thr = 1 - (np.arange(THRESH_NUM) + 1) / THRESH_NUM
    res = {}
    for setting in ("easy", "medium", "hard"):
        curve, count = np.zeros((THRESH_NUM, 2)), 0
        for g in gts:
            keep = g["keep"][setting]
            count += len(keep)
            p = preds.get((g["event"], g["file"]))
            if len(g["boxes"]) == 0 or p is None or len(p) == 0:
                continue
            p = p[np.argsort(-p[:, 4], kind="stable")]
            score = (p[:, 4] - lo) / (hi - lo)
            gt = g["boxes"].copy()
            gt[:, 2:] += gt[:, :2]
            ignore = np.zeros(len(gt))
            if len(keep):
                ignore[keep] = 1
            ov = bbox_overlaps(p[:, :4], gt)
            recall_list, proposal = np.zeros(len(gt)), np.ones(len(p))
            pred_recall = np.zeros(len(p))
            for h in range(len(p)):
                j = ov[h].argmax()
                if ov[h, j] >= IOU:
                    if ignore[j] == 0:
                        recall_list[j], proposal[h] = -1, -1
                    elif recall_list[j] == 0:
                        recall_list[j] = 1
                pred_recall[h] = (recall_list == 1).sum()
            k = np.searchsorted(-score, -thr, side="right")  # number of predictions with score >= thr
            ok = k > 0
            cum_prop = np.cumsum(proposal == 1)
            curve[ok, 0] += cum_prop[k[ok] - 1]
            curve[ok, 1] += pred_recall[k[ok] - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            prec, rec = np.nan_to_num(curve[:, 1] / curve[:, 0]), curve[:, 1] / count
        res[setting] = _voc_ap(rec, prec)
    return res


def bin_recall(preds, gts, attrs, thr):
    """Recall per face-height bin and false positives per image at score threshold thr."""
    found = np.zeros(len(BINS))
    total = np.zeros(len(BINS))
    fp, n_img = 0, 0
    for g in gts:
        b = g["boxes"]
        inv = attrs.get(Path(g["file"]).stem, np.zeros(len(b), int))
        inv = inv if len(inv) == len(b) else np.zeros(len(b), int)
        p = preds.get((g["event"], g["file"]), np.empty((0, 5)))
        p = p[p[:, 4] >= thr]
        n_img += 1
        gt = b.copy()
        gt[:, 2:] += gt[:, :2]
        ov = bbox_overlaps(p[:, :4], gt) if len(p) and len(gt) else np.zeros((len(p), len(gt)))
        fp += int((ov.max(1) < IOU).sum()) if len(p) and len(gt) else len(p)
        hit = (ov >= IOU).any(0) if ov.size else np.zeros(len(gt), bool)
        for k, (a, z) in enumerate(BINS):
            m = (inv == 0) & (b[:, 3] >= a) & (b[:, 3] < z)
            total[k] += m.sum()
            found[k] += (hit & m).sum()
    return found, total, fp / max(n_img, 1)
