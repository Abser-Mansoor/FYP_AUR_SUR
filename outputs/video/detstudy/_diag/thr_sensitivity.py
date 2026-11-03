"""Diagnostic only (not for the report): re-score stored scrfd__full640 detections with a higher score floor.
Stored detections already start at 0.02, the official SCRFD test floor. CPU only, no inference."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.video import wider_eval as W
import phase5_merge as M
study = ROOT / "outputs/video/detstudy"
gts = W.load_gt(ROOT / "Dataset/WIDER_FACE")
names, dets, errors, sec = M.load(study, "wider", "scrfd__full640")
preds = {(g["event"], g["file"]): d for g, d in zip(gts, dets)}
allsc = np.concatenate([d[:, 4] for d in dets])
print("n dets", len(allsc), "min score", float(allsc.min()), "max", float(allsc.max()))
out = {}
for floor in (0.02, 0.05, 0.1, 0.3, 0.5):
    p = {k: v[v[:, 4] >= floor] for k, v in preds.items()}
    r = W.average_precision(p, gts)
    out[floor] = r
    print("floor", floor, {k: round(v, 4) for k, v in r.items()}, flush=True)
json.dump({str(k): v for k, v in out.items()}, open(study / "_diag/thr_sensitivity.json", "w"), indent=1)
