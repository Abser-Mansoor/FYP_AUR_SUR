"""Check src/video/detect.py SCRFDDetector(preprocess="official") against the diagnostic run (lowthr_cpu.py official). CPU only."""
import sys, json
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.video import detect as D, wider_eval as W
det = D.SCRFDDetector(ROOT / "models/detector/det_10g.onnx", providers=("CPUExecutionProvider",), thresh=0.02, nms=0.45, preprocess="official")
leg = D.SCRFDDetector(ROOT / "models/detector/det_10g.onnx", providers=("CPUExecutionProvider",), thresh=0.02, nms=0.4)
z = np.load(ROOT / "outputs/video/detstudy/_diag/preds_official.npz")
cs = np.cumsum(z["counts"])[:-1]
ref = dict(zip(z["names"].tolist(), np.split(z["dets"], cs)))
gts = W.load_gt(ROOT / "Dataset/WIDER_FACE")[:500]
preds, same_count, maxd, kps_ok = {}, 0, 0.0, True
for n, g in enumerate(gts):
    p = ROOT / "Dataset/WIDER_FACE/WIDER_val/images" / g["event"] / f"{g['file']}.jpg"
    bgr = cv2.imread(str(p))
    d, k = D.detect_mode(det, bgr, "full640")
    preds[(g["event"], g["file"])] = d
    kps_ok &= bool(np.isfinite(k).all() and k.shape == (len(d), 5, 2))
    r = ref[f"{g['event']}/{g['file']}"]; r = r[r[:, 4] >= 0.02]
    if len(r) == len(d):
        same_count += 1
        if len(d): maxd = max(maxd, float(np.abs(r[:, :4] - d[:, :4]).max()))
    if n < 3:  # the legacy path must still equal src/align.py
        a, _ = leg.detect(bgr, 640); b, _ = leg.det.detect(bgr, (640, 640), 0.02, 0.4)
        assert np.array_equal(a, b)
res = W.average_precision(preds, gts)
out = dict(n=len(gts), same_detection_count=same_count, max_box_diff_px_when_same_count=maxd, kps_shape_ok=kps_ok, ap=res)
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / "outputs/video/detstudy/_diag/verify_official_option.json", "w"), indent=1)
