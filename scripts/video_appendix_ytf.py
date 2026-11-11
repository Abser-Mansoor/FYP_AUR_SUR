"""Appendix T1: the Phase 5 detector on YouTube Faces (the Kaggle package "YouTube Faces with Facial Keypoints" held in Dataset/YTFaces).

    envs/video/Scripts/python.exe scripts/video_appendix_ytf.py sample                      # lab PC: draw the seeded sample, write the frames and gt_boxes.csv
    envs/video/Scripts/python.exe scripts/video_appendix_ytf.py detect --root <dir> --out <dir>   # lab PC or Colab T4: detector, metrics

What the data is (checked before writing this): 2,194 videos of 828 people, one `.npz` per video holding `colorImages` [H, W, 3, frames] (cropped video frames,
about 120 to 300 px on a side, NOT full frames), `boundingBox` [4, 2, frames] (the four corners as (x, y) pairs; verified on overlays: the box is tight, eyebrows
to chin, like the landmark extent) and 68 landmarks. The original YTF full-frame database (frame_images_DB) is not held. So this test says whether the detector finds
a clearly visible face in cropped video frames with compression and blur; it says nothing about small faces in a wide scene.

sample: seeded (42): 500 people, one video each, 10 evenly spaced frames per video = up to 5,000 frames, saved as PNG with gt_boxes.csv (name, person, video, frame, x1..y2,
face_h). detect: SCRFD det_10g with the official preprocessing (the Phase 5 choice) in the modes full1280 (chosen) and full640 (reference). Per frame: the best-IoU
detection among those with score >= 0.5. Metrics: recall at IoU 0.5 and 0.3, centre hit (the ground-truth centre lies inside a detection with score >= 0.5), the IoU of
the best detection, extra detections per frame (score >= 0.5, IoU < 0.3 with the ground-truth box; the dataset labels only the subject, so these are not all false positives),
median ms per frame; all by ground-truth face-height bin; CI: cluster bootstrap over people (1,000 draws, seed 42).
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED, N_PEOPLE, N_FRAMES = 42, 500, 10
MODES = ["full1280", "full640"]
BINS = [(0, 32), (32, 64), (64, 96), (96, 128), (128, 10 ** 6)]
BIN_NAMES = ["<32", "32-64", "64-96", "96-128", ">128"]


def box_of(bb):
    """boundingBox [4,2] corners as (x, y) -> x1, y1, x2, y2."""
    return float(bb[:, 0].min()), float(bb[:, 1].min()), float(bb[:, 0].max()), float(bb[:, 1].max())


def sample(a):
    import cv2
    src = Path(a.data)
    files = {Path(p).stem: p for p in glob.glob(str(src / "*" / "*" / "*.npz"))}
    csv = pd.read_csv(src / "youtube_faces_with_keypoints_full.csv")
    assert len(files) == len(csv) == 2194, (len(files), len(csv))
    rng = np.random.default_rng(SEED)
    people = sorted(csv["personName"].unique())
    pick = sorted(rng.choice(people, N_PEOPLE, replace=False))
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for person in pick:
        vids = sorted(csv.loc[csv["personName"] == person, "videoID"])
        vid = vids[rng.integers(0, len(vids))]
        z = np.load(files[vid])
        n = z["colorImages"].shape[3]
        for k in sorted(set(np.linspace(0, n - 1, N_FRAMES).round().astype(int).tolist())):
            im = np.ascontiguousarray(z["colorImages"][:, :, ::-1, k])  # RGB -> BGR
            name = f"{vid}__{k:04d}"
            cv2.imwrite(str(out / "frames" / f"{name}.png"), im)
            x1, y1, x2, y2 = box_of(z["boundingBox"][:, :, k])
            rows.append(dict(name=name, person=person, video=vid, frame=int(k), x1=x1, y1=y1, x2=x2, y2=y2, face_h=y2 - y1, face_w=x2 - x1, img_h=im.shape[0], img_w=im.shape[1]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "gt_boxes.csv", index=False)
    print(f"{len(df)} frames, {df.person.nunique()} people, face height median {df.face_h.median():.0f} px, min {df.face_h.min():.0f}, max {df.face_h.max():.0f}")


def iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros(0)
    iw = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    ih = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = iw * ih
    return inter / ((box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter + 1e-9)


def device_info(out):
    import onnxruntime
    import torch
    d = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             onnxruntime=onnxruntime.__version__, ort_providers=onnxruntime.get_available_providers(), python=sys.version.split()[0])
    (Path(out) / "device.json").write_text(json.dumps(d, indent=1))
    return d


def detect(a):
    import cv2
    import torch  # noqa: F401  (before onnxruntime creates a CUDA session)
    from src.video import detect as D
    cfg = yaml.safe_load(open(ROOT / a.config))
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(device_info(out), flush=True)
    sc = cfg["scrfd"]
    prov = ("CPUExecutionProvider",) if a.cpu else ("CUDAExecutionProvider", "CPUExecutionProvider")
    det = D.SCRFDDetector(root / cfg["detectors"]["scrfd"], providers=prov, thresh=cfg["detect_thresh"], nms=sc["nms"], preprocess=sc["preprocess"])
    sdir = Path(a.sample) if a.sample else root / "ytf_sample"
    gt = pd.read_csv(sdir / "gt_boxes.csv")
    if a.limit:
        gt = gt.iloc[:a.limit]
    rows = []
    for mode in MODES:
        for _ in range(5):  # warm-up
            D.detect_mode(det, cv2.imread(str(sdir / "frames" / f"{gt.name.iloc[0]}.png")), mode)
        for r in gt.itertuples():
            bgr = cv2.imread(str(sdir / "frames" / f"{r.name}.png"))
            t = time.perf_counter()
            d, _k = D.detect_mode(det, bgr, mode)
            ms = (time.perf_counter() - t) * 1000
            d = d[d[:, 4] >= 0.5]
            g = np.array([r.x1, r.y1, r.x2, r.y2])
            io = iou(g, d[:, :4])
            cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
            hit = bool(((d[:, 0] <= cx) & (cx <= d[:, 2]) & (d[:, 1] <= cy) & (cy <= d[:, 3])).any()) if len(d) else False
            rows.append(dict(mode=mode, name=r.name, person=r.person, face_h=r.face_h, best_iou=float(io.max()) if len(io) else 0.0, n_det=len(d),
                             n_extra=int((io < 0.3).sum()) if len(io) else 0, centre_hit=hit, ms=ms))
        print(mode, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_frame.csv", index=False, float_format="%.5f")
    summarize(df, out)


def boot_recall(df, col, thr, n=1000, seed=42):
    """cluster bootstrap over people of the recall (best_iou >= thr, or a boolean column)."""
    ok = (df[col] >= thr) if thr is not None else df[col]
    g = pd.DataFrame(dict(p=df.person.to_numpy(), ok=ok.to_numpy().astype(float))).groupby("p").ok.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    d = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def summarize(df, out):
    rows = []
    for mode, g in df.groupby("mode", sort=False):
        for name, (lo, hi) in [("all", (0, 10 ** 9))] + list(zip(BIN_NAMES, BINS)):
            s = g[(g.face_h >= lo) & (g.face_h < hi)]
            if len(s) == 0:
                continue
            row = dict(mode=mode, face_h_bin=name, n=len(s), n_people=s.person.nunique(), extra_per_frame=s.n_extra.mean(), median_ms=s.ms.median(), iou_median=s.best_iou.median(),
                       iou_p10=s.best_iou.quantile(0.1))
            for k, (col, thr) in {"recall_iou50": ("best_iou", 0.5), "recall_iou30": ("best_iou", 0.3), "centre_hit": ("centre_hit", None)}.items():
                v, a, b = boot_recall(s, col, thr)
                row.update({k: v, f"{k}_ci_lo": a, f"{k}_ci_hi": b})
            rows.append(row)
    pd.DataFrame(rows).to_csv(Path(out) / "summary.csv", index=False, float_format="%.5f")
    print(pd.DataFrame(rows)[["mode", "face_h_bin", "n", "recall_iou50", "recall_iou30", "centre_hit", "extra_per_frame", "median_ms"]].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--data", default=str(ROOT / "Dataset/YTFaces"))
    s.add_argument("--out", default=str(ROOT / "outputs/video/appendix_data/ytf_sample"))
    d = sub.add_parser("detect")
    d.add_argument("--config", default="configs/video.yaml")
    d.add_argument("--root", default=str(ROOT), help="folder that holds models/detector/det_10g.onnx and ytf_sample/")
    d.add_argument("--out", default=str(ROOT / "outputs/video/appendix/ytf"))
    d.add_argument("--limit", type=int, default=None)
    d.add_argument("--sample", default=None, help="folder with frames/ and gt_boxes.csv (default: <root>/ytf_sample)")
    d.add_argument("--cpu", action="store_true", help="CPU only (smoke test while the GPU is busy)")
    a = ap.parse_args()
    {"sample": sample, "detect": detect}[a.cmd](a)


if __name__ == "__main__":
    main()
