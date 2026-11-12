"""Overlays for the false-positive check: ground-truth boxes (green) and detections with score >= 0.5 that match no ground-truth box at IoU 0.3 (red, score printed).
    envs/video/Scripts/python.exe scripts/video_fp_overlay.py --clip P2E_S3_C1.1 --n 6
Local only (ChokePoint frames are not committed): outputs/video/overlays_fp/<clip>/."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def iou(a, b):
    iw = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - i + 1e-9)


ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()
t = pd.read_csv(ROOT / f"outputs/video/pipeline/chokepoint/{a.clip}/tracks.csv")
t = t[t.score >= 0.5]
g = np.loadtxt(ROOT / f"outputs/video/gt/mot/{a.clip}.txt", delimiter=",", ndmin=2)  # frame,id,x,y,w,h,...
gt = {int(f): np.c_[r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]] for f in np.unique(g[:, 0]) for r in [g[g[:, 0] == f]]}
cand = []
for f, r in t.groupby("frame"):
    d = r[["x1", "y1", "x2", "y2"]].to_numpy()
    gb = gt.get(int(f), np.zeros((0, 4)))
    un = [i for i in range(len(d)) if len(gb) == 0 or iou(d[i], gb).max() < 0.3]
    if un:
        cand.append((int(f), len(un)))
rng = np.random.default_rng(42)
pick = sorted(int(x) for x in rng.choice([c[0] for c in cand], min(a.n, len(cand)), replace=False))
out = ROOT / "outputs/video/overlays_fp" / a.clip
out.mkdir(parents=True, exist_ok=True)
print("frames with unmatched detections:", len(cand), "of", t.frame.nunique(), "frames with detections")
for f in pick:
    im = cv2.imread(str(ROOT / "Dataset/ChokePoint" / a.clip / f"{f:08d}.jpg"))
    for b in gt.get(f, []):
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 2)
    r = t[t.frame == f]
    gb = gt.get(f, np.zeros((0, 4)))
    for x in r.itertuples():
        d = np.array([x.x1, x.y1, x.x2, x.y2])
        if len(gb) == 0 or iou(d, gb).max() < 0.3:
            cv2.rectangle(im, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 0, 255), 2)
            cv2.putText(im, f"{x.score:.2f}", (int(d[0]), max(int(d[1]) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(out / f"{f:08d}.jpg"), im)
    print(out / f"{f:08d}.jpg")
