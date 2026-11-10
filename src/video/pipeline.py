"""Per-clip video pipeline: detect -> track -> embed. Resumable per clip, atomic writes (temp file, then rename).

Output folder `<pipeline.out>/<dataset>/<clip>/`:
  detections.npz  frame_idx[M], dets[M,5] (x1,y1,x2,y2,score), kps[M,5,2], frames[F] (every frame index of the clip), shape (H,W), seconds_detect
  tracks.csv      frame, det_id, x1, y1, x2, y2, score, tid_bytetrack, tid_botsort   (one row per detection above the tracker floor; -1 = no track)
  embed.npz       det_id[K] (rows of tracks.csv), emb_<recognizer>[K,512], mag_norm[K], lum[K], lap_var[K], seconds_embed
  done_<stage>.json  stage timings and counts
A stage is skipped when its output exists. `seconds_*` are wall-clock and are the only non-deterministic entries (they are kept out of the byte-identity check).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.video import detect as D
from src.video.io import iter_frames, list_frames
from src.video.track import TrackerAdapter

ROOT = Path(__file__).resolve().parents[2]


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


def clip_out(cfg, dataset, clip):
    return ROOT / cfg["pipeline"]["out"] / dataset / clip


def build_detector(cfg):
    sc = cfg["scrfd"]
    return D.SCRFDDetector(ROOT / cfg["detectors"]["scrfd"], thresh=cfg["detect_thresh"], nms=sc["nms"], preprocess=sc["preprocess"])


def detect_clip(det, cfg, source, out, limit=None):
    """Detect on every frame of `source` (folder or video). Keeps detections from `pipeline.track_input_floor` up."""
    f = Path(out) / "detections.npz"
    if f.exists():
        return json.loads((Path(out) / "done_detect.json").read_text())
    floor, mode = cfg["pipeline"]["track_input_floor"], cfg["pipeline"]["detector_mode"]
    frames, fi, dets, kps, shape = [], [], [], [], None
    t0 = time.perf_counter()
    for idx, _ts, bgr in iter_frames(source, cfg["pipeline"]["fps"], limit):
        shape = bgr.shape[:2]
        d, k = D.detect_mode(det, bgr, mode)
        keep = d[:, 4] >= floor
        d, k = d[keep], k[keep]
        frames.append(idx)
        fi.extend([idx] * len(d))
        dets.append(d)
        kps.append(k)
    sec = time.perf_counter() - t0
    dets = np.concatenate(dets) if dets else np.zeros((0, 5), np.float32)
    kps = np.concatenate(kps) if kps else np.zeros((0, 5, 2), np.float32)
    _atomic(f, lambda fh: np.savez(fh, frame_idx=np.array(fi, int), dets=dets, kps=kps, frames=np.array(frames, int), shape=np.array(shape or (0, 0)), seconds_detect=sec))
    info = dict(n_frames=len(frames), n_dets=int(len(dets)), seconds_detect=sec, ms_per_frame=1000 * sec / max(len(frames), 1))
    _atomic(Path(out) / "done_detect.json", lambda fh: fh.write(json.dumps(info).encode()))
    return info


def track_clip(cfg, out):
    """Run every configured tracker over the clip's detections (same detections for all). Writes tracks.csv."""
    out = Path(out)
    f = out / "tracks.csv"
    if f.exists():
        return json.loads((out / "done_track.json").read_text())
    z = np.load(out / "detections.npz")
    frames, fi, dets, shape = z["frames"], z["frame_idx"], z["dets"], tuple(int(x) for x in z["shape"])
    tcfg = cfg["tracking"]
    trackers = {n: TrackerAdapter(n, tcfg["frame_rate"], match_iou=tcfg["match_iou"]) for n in tcfg["trackers"]}
    starts = np.searchsorted(fi, frames, side="left")
    ends = np.searchsorted(fi, frames, side="right")
    cols = {f"tid_{n}": np.full(len(dets), -1, int) for n in trackers}
    t0 = time.perf_counter()
    for k, frame in enumerate(frames):
        s, e = starts[k], ends[k]
        for n, tr in trackers.items():
            cols[f"tid_{n}"][s:e] = tr.update(dets[s:e], shape)
    sec = time.perf_counter() - t0
    df = pd.DataFrame(dict(frame=fi, det_id=np.arange(len(dets)), x1=dets[:, 0], y1=dets[:, 1], x2=dets[:, 2], y2=dets[:, 3], score=dets[:, 4], **cols))
    _atomic(f, lambda fh: fh.write(df.to_csv(index=False, float_format="%.5f").encode()))
    info = dict(n_frames=len(frames), n_dets=int(len(dets)), seconds_track=sec, ms_per_frame=1000 * sec / max(len(frames), 1),
                **{f"n_tracks_{n}": int(len(set(cols[f'tid_{n}'][cols[f'tid_{n}'] >= 0]))) for n in trackers})
    _atomic(out / "done_track.json", lambda fh: fh.write(json.dumps(info).encode()))
    return info


def embed_clip(embedder, cfg, source, out, limit=None):
    """Align and embed every detection with score >= pipeline.embed_floor (needs the frames again). Writes embed.npz."""
    out = Path(out)
    f = out / "embed.npz"
    if f.exists():
        return json.loads((out / "done_embed.json").read_text())
    z = np.load(out / "detections.npz")
    fi, dets, kps = z["frame_idx"], z["dets"], z["kps"]
    sel = np.flatnonzero(dets[:, 4] >= cfg["pipeline"]["embed_floor"])
    by_frame = {}
    for r in sel:
        by_frame.setdefault(int(fi[r]), []).append(r)
    ids, embs, mag, lum, lap = [], {n: [] for n in embedder.recs}, [], [], []
    t0 = time.perf_counter()
    for idx, _ts, bgr in iter_frames(source, cfg["pipeline"]["fps"], limit):
        rows = by_frame.get(idx)
        if not rows:
            continue
        r = embedder.embed(bgr, kps[rows])
        ids.extend(rows)
        for n in embs:
            embs[n].append(r["emb"][n])
        mag.append(r["mag_norm"])
        lum.append(r["lum"])
        lap.append(r["lap_var"])
    sec = time.perf_counter() - t0
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape, np.float32)  # noqa: E731
    arrays = dict(det_id=np.array(ids, int), mag_norm=cat(mag, (0,)), lum=cat(lum, (0,)), lap_var=cat(lap, (0,)), seconds_embed=sec,
                  **{f"emb_{n}": cat(v, (0, 512)) for n, v in embs.items()})
    _atomic(f, lambda fh: np.savez(fh, **arrays))
    info = dict(n_embedded=len(ids), seconds_embed=sec, ms_per_face=1000 * sec / max(len(ids), 1))
    _atomic(out / "done_embed.json", lambda fh: fh.write(json.dumps(info).encode()))
    return info


def chokepoint_clips(root=None):
    """Clips with eye ground truth (72) and the crowded clips without it (P2E_S5, P2L_S5)."""
    root = Path(root or ROOT / "Dataset/ChokePoint")
    gt = sorted(pd.read_csv(ROOT / "outputs/video/gt/chokepoint_eyes.csv")["clip"].unique())
    crowded = sorted(p.name for p in root.iterdir() if p.is_dir() and (p.name.startswith("P2E_S5") or p.name.startswith("P2L_S5")))
    return gt, crowded


def n_frames(source):
    return len(list_frames(source))
