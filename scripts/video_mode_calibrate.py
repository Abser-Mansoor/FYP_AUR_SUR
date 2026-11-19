"""Recommend the SCRFD input mode for one camera from a calibration clip (Phase 7 decision 5: a mode fixed per camera, chosen from the face sizes seen by that camera).

    envs/video/Scripts/python.exe scripts/video_mode_calibrate.py --source Dataset/ChokePoint/P2L_S4_C3.2 [--n 100] [--out result.json]

`--source` is a folder of frames or a video file. About `--n` frames are sampled evenly; SCRFD runs in `full640` and `full1280` (official preprocessing), the detections with score
>= 0.5 are merged (NMS 0.45) so that large faces (found by 640) and small faces (found by 1280) are both seen, and the face-height percentiles are printed in the pixels of the
original frame. Rule (REPORT_phase7.md section 12, decision 5): 5th percentile of the face height >= 32 px -> `full640` (same recall as `full1280` for faces of at least 32 px,
about 3.6 times faster on a T4); otherwise `tiled` (the compromise that keeps small faces and does not lose large ones). The percentiles come from the detector itself, so faces it
cannot see at all are not counted: treat the answer as a recommendation to check against a few frames, not a guarantee.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402,F401  (before onnxruntime creates a CUDA session)
from src.video import detect as D  # noqa: E402
from src.video.io import iter_frames, list_frames  # noqa: E402
from src.video.pipeline import build_detector  # noqa: E402
from src.video.track import iou_matrix  # noqa: E402

MIN_H_FOR_640 = 32


def merge(dets):
    """Greedy NMS (IoU 0.45) over the detections of both modes."""
    if len(dets) == 0:
        return dets
    dets = dets[np.argsort(-dets[:, 4], kind="stable")]
    keep = []
    for i in range(len(dets)):
        if all(iou_matrix(dets[i:i + 1, :4], dets[j:j + 1, :4])[0, 0] < 0.45 for j in keep):
            keep.append(i)
    return dets[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--config", default="configs/video.yaml")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    src = Path(a.source)
    total = len(list_frames(src)) if src.is_dir() else int(cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(total // a.n, 1)
    det = build_detector(cfg)
    heights, n_frames = [], 0
    for idx, _ts, bgr in iter_frames(src, cfg["pipeline"]["fps"], None, errors=[]):
        if idx % stride or bgr is None:
            continue
        n_frames += 1
        both = [D.detect_mode(det, bgr, m)[0] for m in ("full640", "full1280")]
        d = merge(np.concatenate([x[x[:, 4] >= 0.5] for x in both]))
        heights += (d[:, 3] - d[:, 1]).tolist()
    h = np.array(heights)
    if len(h) == 0:
        sys.exit("no face found in the sampled frames; cannot recommend a mode")
    p5, p50, p95 = np.percentile(h, [5, 50, 95])
    mode = "full640" if p5 >= MIN_H_FOR_640 else "tiled"
    res = dict(source=str(src), frames_sampled=n_frames, faces=len(h), face_height_px_p5=float(p5), face_height_px_p50=float(p50), face_height_px_p95=float(p95),
               rule=f"p5 >= {MIN_H_FOR_640} px -> full640, else tiled", recommended_mode=mode)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
