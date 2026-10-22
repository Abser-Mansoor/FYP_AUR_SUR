"""Phase 5 detector size study: SCRFD and RetinaFace, three modes each, on WIDER FACE val, a ChokePoint frame sample and the
TinyFace / SurvFace eval crops. Chunked and resumable; every chunk is written atomically, so an interrupted run (Colab
disconnect, lab-PC reboot) continues where it stopped. Scoring happens later in scripts/phase5_merge.py.

    envs/video/Scripts/python.exe scripts/video_detector_study.py --job all                # everything
    envs/video/Scripts/python.exe scripts/video_detector_study.py --job wider --detectors scrfd --limit 20   # smoke test

Jobs: wider | chokepoint | crops | runtime | xcheck | all.  Outputs under paths.out (configs/video.yaml):
  chunks/<job>/<detector>__<mode>/<k>.npz   names, counts, dets[M,5]   (crops: names, kps[n,5,2], NaN = no face)
  runtime.json, device.json, xcheck.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sr import common as C  # noqa: E402
from src.video import detect as D  # noqa: E402
from src.video import wider_eval as W  # noqa: E402


def device_info(out):
    import onnxruntime
    import torch
    d = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             onnxruntime=onnxruntime.__version__, ort_providers=onnxruntime.get_available_providers(), python=sys.version.split()[0])
    (out / "device.json").write_text(json.dumps(d, indent=1))
    return d


def build_detector(name, cfg, root):
    if name == "scrfd":
        return D.SCRFDDetector(root / cfg["detectors"]["scrfd"], thresh=cfg["detect_thresh"], nms=cfg["nms"])
    return D.RetinaFaceDetector(root / cfg["detectors"]["retinaface"], thresh=cfg["detect_thresh"], nms=cfg["nms"])


def save_chunk(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def run_chunks(job, det, tag, items, fn, chunk, out, kind="dets"):
    """items: list of (name, path). fn(det, bgr) -> dets[N,5] (kind='dets') or kps[5,2]/None (kind='kps')."""
    d = out / "chunks" / job / tag
    n_chunks = (len(items) + chunk - 1) // chunk
    for k in range(n_chunks):
        f = d / f"{k:05d}.npz"
        if f.exists():
            continue
        t0, part = time.perf_counter(), items[k * chunk:(k + 1) * chunk]
        names, rows, counts, errors = [], [], [], []
        for name, p in part:
            names.append(name)
            try:
                r = fn(det, C.read_bgr(p))
            except Exception as e:  # logged as a result, never dropped
                errors.append(f"{name}: {type(e).__name__}: {e}")
                r = None
            if kind == "dets":
                r = np.empty((0, 5), np.float32) if r is None else r
                rows.append(r)
                counts.append(len(r))
            else:
                rows.append(np.full((5, 2), np.nan, np.float32) if r is None else np.asarray(r, np.float32))
        arrays = dict(names=np.array(names), seconds=np.float64(time.perf_counter() - t0), errors=np.array(errors))
        if kind == "dets":
            arrays.update(counts=np.array(counts), dets=np.concatenate(rows) if rows else np.empty((0, 5), np.float32))
        else:
            arrays.update(kps=np.stack(rows))
        save_chunk(f, **arrays)
        print(f"[{job}/{tag}] chunk {k + 1}/{n_chunks} ({len(part)} items, {arrays['seconds']:.1f}s, {len(errors)} errors)", flush=True)


def wider_items(cfg, root):
    gts = W.load_gt(root / cfg["paths"]["wider"])
    return [(f"{g['event']}/{g['file']}", root / cfg["paths"]["wider"] / "WIDER_val/images" / g["event"] / f"{g['file']}.jpg") for g in gts]


def sample_items(cfg, root):
    idx = pd.read_csv(root / cfg["paths"]["chokepoint_sample"] / "index.csv")
    return [(r["name"], root / cfg["paths"]["chokepoint_sample"] / r["file"]) for _, r in idx.iterrows()]


def crop_items(cfg, root):
    d = pd.read_csv(root / cfg["paths"]["eval_images"])
    return [(r["file"], root / cfg["paths"]["eval_crops"] / r["file"]) for _, r in d.iterrows()]


def job_runtime(cfg, root, out, dets):
    """Median ms per frame at 720p (resized ChokePoint sample frames), batch 1, after warm-up."""
    rt = cfg["runtime"]
    frames = [C.read_bgr(p) for _, p in sample_items(cfg, root)[:rt["n_frames"]]]
    import cv2
    frames = [cv2.resize(f, tuple(rt["size"]), interpolation=cv2.INTER_LINEAR) for f in frames]
    res = {}
    for name in dets:
        det = build_detector(name, cfg, root)
        for mode in cfg["modes"]:
            for f in frames[:rt["warmup"]]:
                D.detect_mode(det, f, mode)
            ms = []
            for f in frames:
                t = time.perf_counter()
                D.detect_mode(det, f, mode)
                ms.append((time.perf_counter() - t) * 1000)
            res[f"{name}__{mode}"] = dict(median_ms=float(np.median(ms)), mean_ms=float(np.mean(ms)), p95_ms=float(np.percentile(ms, 95)), n=len(ms))
            print(f"[runtime] {name} {mode}: median {res[f'{name}__{mode}']['median_ms']:.1f} ms", flush=True)
    (out / "runtime.json").write_text(json.dumps(res, indent=1))


def job_xcheck(cfg, root, out, xdir):
    """Cross-device check. xdir has crops/*.png (100 TinyFace probes) and lab_kps.npy (written by the lab PC; if missing, this run
    writes it). SCRFD padded-crop landmarks must agree within 0.5 px."""
    xdir = Path(xdir)
    files = sorted((xdir / "crops").glob("*.png"))
    det = build_detector("scrfd", cfg, root)
    kps = np.stack([np.full((5, 2), np.nan) if (k := D.crop_detect(det, C.read_bgr(f), cfg["crops"]["lowres_side"], cfg["crops"]["thresh"])) is None
                    else k for f in files]).astype(np.float32)
    ref = xdir / "lab_kps.npy"
    if not ref.exists():
        np.save(ref, kps)
        print("wrote", ref)
        return
    lab = np.load(ref)
    both = ~np.isnan(kps).any((1, 2)) & ~np.isnan(lab).any((1, 2))
    diff = float(np.abs(kps[both] - lab[both]).max()) if both.any() else float("nan")
    res = dict(n=len(files), n_detected_here=int((~np.isnan(kps).any((1, 2))).sum()), n_detected_lab=int((~np.isnan(lab).any((1, 2))).sum()),
               n_detection_mismatch=int((np.isnan(kps).any((1, 2)) != np.isnan(lab).any((1, 2))).sum()), max_abs_diff_px=diff,
               pass_=bool(diff <= 0.5 and (np.isnan(kps).any((1, 2)) == np.isnan(lab).any((1, 2))).all()))
    (out / "xcheck.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if not res["pass_"]:
        sys.exit("cross-check FAILED: stop and report to the lab PC owner")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/video.yaml")
    ap.add_argument("--root", default=str(ROOT), help="data root (Colab: the unpacked bundle)")
    ap.add_argument("--job", default="all", choices=["wider", "chokepoint", "crops", "runtime", "xcheck", "all"])
    ap.add_argument("--detectors", nargs="+", default=["scrfd", "retinaface"])
    ap.add_argument("--modes", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N items per job (smoke test; chunks go to a _smoke folder)")
    ap.add_argument("--xcheck-dir", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    cfg = yaml.safe_load(open(ROOT / a.config))
    out = root / cfg["paths"]["out"] / ("_smoke" if a.limit else "")
    out.mkdir(parents=True, exist_ok=True)
    print(device_info(out), flush=True)
    modes = a.modes or cfg["modes"]
    cfg["modes"] = modes
    chunk = cfg["chunk"]
    jobs = ["xcheck", "wider", "chokepoint", "crops", "runtime"] if a.job == "all" else [a.job]
    for job in jobs:
        if job == "xcheck":
            if a.xcheck_dir or a.job == "xcheck":
                job_xcheck(cfg, root, out, a.xcheck_dir or root / cfg["paths"]["out"] / "xcheck")
            continue
        if job == "runtime":
            job_runtime(cfg, root, out, a.detectors)
            continue
        items = {"wider": wider_items, "chokepoint": sample_items, "crops": crop_items}[job](cfg, root)
        items = items[:a.limit] if a.limit else items
        for name in a.detectors:
            det = build_detector(name, cfg, root)
            if job == "crops":
                run_chunks(job, det, f"{name}__crop", items, lambda d, im: D.crop_detect(d, im, cfg["crops"]["lowres_side"], cfg["crops"]["thresh"]),
                           chunk, out, kind="kps")
            else:
                for mode in modes:
                    run_chunks(job, det, f"{name}__{mode}", items, lambda d, im, m=mode: D.detect_mode(d, im, m)[0], chunk, out)
    print("done")


if __name__ == "__main__":
    main()
