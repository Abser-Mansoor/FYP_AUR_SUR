"""Acceptance check 3 (Phase 6): the vendored TrackEval metrics give the known values on a hand-built case.

    envs/video/Scripts/python.exe scripts/video_trackeval_check.py

Case (10 frames, 2 ground-truth objects A and B, perfect boxes): the tracker follows A with id 1 but misses it at frame 3; it follows B with id 2 for frames 0-4
and with id 3 for frames 5-9 (one identity switch). Values worked out by hand from the metric definitions:
  ground-truth detections 20, tracker detections 19, TP 19, FN 1, FP 0, IDSW 1  ->  MOTA = 1 - (1 + 0 + 1) / 20 = 0.9
  IDTP 14 (A-1: 9, B-2 or B-3: 5), IDFN 6, IDFP 5                               ->  IDF1 = 28 / 39 = 0.717949
  DetA = 19 / 20 = 0.95 ; AssA = (9 * 0.9 + 5 * 0.5 + 5 * 0.5) / 19             ->  HOTA = sqrt(DetA * AssA) = 0.80933 (every alpha, boxes are exact)
Also: identical tracks give HOTA = MOTA = IDF1 = 1. Writes outputs/video/trackeval_check.json; exit code 1 on failure.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.video import mot_eval as M  # noqa: E402


def box(x, y):
    return [x, y, x + 20.0, y + 30.0]


def build():
    gt, pred = {}, {}
    for f in range(10):
        a, b = box(10 + 5 * f, 20), box(200 - 5 * f, 80)
        gt[f] = (np.array([0, 1]), np.array([a, b]))
        ids, boxes = [], []
        if f != 3:
            ids.append(1)
            boxes.append(a)
        ids.append(2 if f < 5 else 3)
        boxes.append(b)
        pred[f] = (np.array(ids), np.array(boxes))
    return gt, pred, list(range(10))


def main():
    gt, pred, frames = build()
    r = M.clip_metrics(gt, pred, frames)
    perfect = M.clip_metrics(gt, gt, frames)
    exp = dict(MOTA=0.9, IDF1=28 / 39, HOTA=float(np.sqrt(0.95 * (13.1 / 19))), IDSW=1, CLR_TP=19, CLR_FN=1, CLR_FP=0, IDTP=14, IDFN=6, IDFP=5)
    got = {k: r[k] for k in exp}
    ok = all(abs(got[k] - v) < 1e-6 for k, v in exp.items()) and all(abs(perfect[k] - 1) < 1e-9 for k in ("MOTA", "IDF1", "HOTA"))
    out = dict(expected=exp, got=got, perfect={k: perfect[k] for k in ("MOTA", "IDF1", "HOTA")}, ok=bool(ok),
               note="hand-built case; the bundled MOT17 sample of TrackEval was not used (it is a separate data download)")
    (ROOT / "outputs/video").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/video/trackeval_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
