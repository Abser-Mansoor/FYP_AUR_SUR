"""Diagnostic only (not for the report). CPU-only SCRFD det_10g on WIDER val, two variants, floor 0.001:
  study    = study preprocessing (black pad, int-truncated resize), NMS 0.4
  official = insightface test_widerface.py mode 0 (resize keep_ratio to 640, normalise, then zero pad = gray, NMS 0.45)
Usage: lowthr_cpu.py <variant> [limit]"""
import sys, json, time
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch  # noqa: F401  (import before onnxruntime)
from src.align import SCRFD, _nms
from src.video import wider_eval as W
variant = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
FLOOR = 0.001
det = SCRFD(str(ROOT / "models/detector/det_10g.onnx"), providers=("CPUExecutionProvider",))

def run(bgr):
    H = Wd = 640
    r = bgr.shape[0] / bgr.shape[1]
    if variant == "study":
        nh, nw = (H, int(H / r)) if r > 1 else (int(Wd * r), Wd)
        scale = nh / bgr.shape[0]
        canvas = np.zeros((H, Wd, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128.0, (Wd, H), (127.5,) * 3, swapRB=True)
        nms = 0.4
    else:
        s = min(H / bgr.shape[0], Wd / bgr.shape[1])
        nh, nw = int(bgr.shape[0] * s + 0.5), int(bgr.shape[1] * s + 0.5)
        scale = s
        im = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32)
        im = (im - 127.5) / 128.0
        blob = np.zeros((1, 3, H, Wd), np.float32)  # zero pad after normalisation
        blob[0, :, :nh, :nw] = im.transpose(2, 0, 1)
        nms = 0.45
    outs = det.sess.run(det.output_names, {det.input_name: blob})
    bl, sl = [], []
    for i, st in enumerate(det.strides):
        sc, bb = outs[i], outs[i + 3] * st
        if sc.ndim == 3: sc, bb = sc[0], bb[0]
        c = det._anchor_centers(H // st, Wd // st, st)
        k = np.where(sc.ravel() >= FLOOR)[0]
        box = np.stack([c[:, 0] - bb[:, 0], c[:, 1] - bb[:, 1], c[:, 0] + bb[:, 2], c[:, 1] + bb[:, 3]], -1)
        bl.append(box[k]); sl.append(sc.ravel()[k])
    s_ = np.concatenate(sl)
    if s_.size == 0: return np.empty((0, 5), np.float32)
    o = np.argsort(-s_, kind="stable")
    d = np.hstack([np.concatenate(bl) / scale, s_[:, None]])[o].astype(np.float32)
    return d[_nms(d, nms)]

gts = W.load_gt(ROOT / "Dataset/WIDER_FACE")
if limit: gts = gts[:limit]
preds, t0 = {}, time.time()
for n, g in enumerate(gts):
    p = ROOT / "Dataset/WIDER_FACE/WIDER_val/images" / g["event"] / f"{g['file']}.jpg"
    preds[(g["event"], g["file"])] = run(cv2.imread(str(p)))
    if n % 200 == 0: print(variant, n, f"{time.time()-t0:.0f}s", flush=True)
keys=list(preds)
np.savez_compressed(ROOT / f"outputs/video/detstudy/_diag/preds_{variant}{'_lim'+str(limit) if limit else ''}.npz", names=np.array([k[0]+'/'+k[1] for k in keys]), counts=np.array([len(preds[k]) for k in keys]), dets=np.concatenate([preds[k] for k in keys]))
res = {}
for floor in (0.001, 0.02):
    pf = {k: v[v[:, 4] >= floor] for k, v in preds.items()}
    res[str(floor)] = W.average_precision(pf, gts)
    print(variant, "floor", floor, {k: round(v, 4) for k, v in res[str(floor)].items()}, flush=True)
json.dump(dict(variant=variant, n=len(gts), res=res), open(ROOT / f"outputs/video/detstudy/_diag/lowthr_{variant}{'_lim'+str(limit) if limit else ''}.json", "w"), indent=1)
