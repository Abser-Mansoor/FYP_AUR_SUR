"""Phase 7 YouTube Faces detect + embed (REPORT_phase7.md section 0.5). Resumable per video, atomic writes.

    envs/video/Scripts/python.exe scripts/video_ytf_prepare.py                    # first: writes outputs/video/ytf/frames.csv
    envs/video/Scripts/python.exe scripts/video_ytf_embed.py [--videos 20] [--out DIR]

For each video: the selected frames (frames.csv: up to 50 evenly spaced labelled frames) are read; SCRFD (`ytf.detector_mode`, official preprocessing) runs on each; the subject is the
detection with score >= `ytf.score_floor` that has the highest IoU with the labelled box, accepted if IoU >= `ytf.iou` (frames with no such detection are skipped and counted);
the accepted faces are aligned (5-point, `norm_crop`) and embedded (AdaFace, MagFace raw norm as quality, luminance, Laplacian variance), exactly as the Phase 6 pipeline.
Output: <out>/<video_idx>.npz (frame_no, kps, box, score, iou, emb_adaface, mag_norm, lum, lap_var, counts) and <out>/summary.csv (one row per video).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402,F401  (before onnxruntime creates a CUDA session)
from src.video import detect as D  # noqa: E402
from src.video.embed import FaceEmbedder  # noqa: E402
from src.video.pipeline import build_detector  # noqa: E402

DATA = ROOT / "Dataset/YouTubeFaces/frame_images_DB"


def iou(g, boxes):
    iw = np.clip(np.minimum(g[2], boxes[:, 2]) - np.maximum(g[0], boxes[:, 0]), 0, None)
    ih = np.clip(np.minimum(g[3], boxes[:, 3]) - np.maximum(g[1], boxes[:, 1]), 0, None)
    i = iw * ih
    return i / ((g[2] - g[0]) * (g[3] - g[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - i + 1e-9)


def process(video, g, det, emb, cfg):
    yc = cfg["ytf"]
    t0 = time.perf_counter()
    fno, boxes, scores, ious, crops, kept_kps = [], [], [], [], [], []  # kept_kps: the 5 keypoints of the accepted face (saved so the CAFace arm can align the same faces)
    n_unreadable = n_nodet = n_nomatch = 0
    for r in g.itertuples():
        bgr = cv2.imread(str(DATA / r.path))
        if bgr is None:
            n_unreadable += 1
            continue
        d, k = D.detect_mode(det, bgr, yc["detector_mode"])
        keep = d[:, 4] >= yc["score_floor"]
        d, k = d[keep], k[keep]
        if len(d) == 0:
            n_nodet += 1
            continue
        io = iou(np.array([r.x1, r.y1, r.x2, r.y2]), d[:, :4])
        j = int(io.argmax())
        if io[j] < yc["iou"]:
            n_nomatch += 1
            continue
        fno.append(r.frame_no), boxes.append(d[j, :4]), scores.append(d[j, 4]), ious.append(io[j])
        crops.append(emb.crops(bgr, k[j:j + 1])[0])
        kept_kps.append(k[j])
    if crops:
        e = emb.embed_crops(crops)
        arrays = dict(emb_adaface=e["emb"]["adaface"], mag_norm=e["mag_norm"], lum=e["lum"], lap_var=e["lap_var"])
    else:
        arrays = dict(emb_adaface=np.zeros((0, 512), np.float32), mag_norm=np.zeros(0, np.float32), lum=np.zeros(0, np.float32), lap_var=np.zeros(0, np.float32))
    return dict(frame_no=np.array(fno, int), kps=np.array(kept_kps, np.float32).reshape(-1, 5, 2), box=np.array(boxes, np.float32).reshape(-1, 4), score=np.array(scores, np.float32), iou=np.array(ious, np.float32), **arrays,
                n_selected=len(g), n_unreadable=n_unreadable, n_no_detection=n_nodet, n_no_match=n_nomatch, seconds=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/video.yaml")
    ap.add_argument("--frames", default=str(ROOT / "outputs/video/ytf/frames.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs/video/ytf/embed"))
    ap.add_argument("--videos", type=int, default=None, help="only the first N videos (smoke test; use another --out)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    fr = pd.read_csv(a.frames)
    vids = sorted(fr.video_idx.unique())[:a.videos]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg2 = json.loads(json.dumps(cfg))
    det = build_detector(cfg2)
    emb = FaceEmbedder(cfg2, ("adaface",))
    t_all = time.perf_counter()
    for n, v in enumerate(vids, 1):
        f = out / f"{v}.npz"
        if f.exists():
            continue
        g = fr[fr.video_idx == v]
        res = process(g.video.iloc[0], g, det, emb, cfg2)
        tmp = f.with_name(f.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, **res)
        os.replace(tmp, f)
        if n % 25 == 0 or n == len(vids):
            print(f"[{n}/{len(vids)}] video {v} {g.video.iloc[0]}: kept {len(res['frame_no'])}/{res['n_selected']}  {res['seconds']:.1f}s  elapsed {time.perf_counter() - t_all:.0f}s", flush=True)
    rows = []
    for v in vids:
        f = out / f"{v}.npz"
        if f.exists():
            z = np.load(f)
            rows.append(dict(video_idx=v, n_selected=int(z["n_selected"]), n_kept=len(z["frame_no"]), n_no_detection=int(z["n_no_detection"]), n_no_match=int(z["n_no_match"]),
                             n_unreadable=int(z["n_unreadable"]), seconds=float(z["seconds"])))
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False, float_format="%.2f")
    print("summary written;", len(rows), "videos")


if __name__ == "__main__":
    main()
