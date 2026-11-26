"""Phase 9 demo (REPORT_phase9.md section 0.6 F): a video file -> annotated MP4 with face boxes, track ids, identity / "unknown" / "low quality", and the run's frame rate.

    envs/video/Scripts/python.exe scripts/video_demo.py --video Dataset/MEVA/<clip>.avi --mode full1280 [--limit N]
    envs/video/Scripts/python.exe scripts/video_demo.py --calibrate               # the detector mode per camera from its calibration clip (GPU, before --all and the FPS run)
    envs/video/Scripts/python.exe scripts/video_demo.py --all                     # every demo clip in Dataset/MEVA/clips.txt, mode from its calibration clip

Pipeline (the frozen configuration, outputs/video/mevid/phase9/frozen_config.json; configs/video.yaml `final`):
  detect (SCRFD, official preprocessing, mode per camera) -> embed (WebFace4M [+ MS1MV2 if F4] and the MagFace norm, faces with score >= 0.5)
  -> ByteTrack with the face term (lambda 0.5, native height >= 32 px) -> per track: usable faces (score >= 0.5, height >= 16 px), template by the rule,
  quality bar (mean usable-face height >= 23 px, else "low quality"), top-1 against the 170 MEVID gallery templates (FAISS Flat), threshold = demo_threshold (FPIR 0.1).
Outputs: outputs/video/phase9/demo/<clip>/ (annotated.mp4, tracks.csv, cached stages; local only, gitignored) and outputs/video/phase9/demo_summary/<clip>.json (tracked: counts only).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch  # noqa: F401  (before onnxruntime)
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.video import pipeline as P  # noqa: E402
from src.video.identify import LOW, UNKNOWN, decide, track_template  # noqa: E402
from src.video.index import GalleryIndex  # noqa: E402
from src.video.io import iter_frames  # noqa: E402
from src.video.track import TrackerAdapter  # noqa: E402

DEMO = ROOT / "outputs/video/phase9/demo"
SUMMARY = ROOT / "outputs/video/phase9/demo_summary"
MEVA = ROOT / "Dataset/MEVA"
FROZEN = ROOT / "outputs/video/mevid/phase9/frozen_config.json"
COLORS = {"id": (60, 200, 60), UNKNOWN: (0, 165, 255), LOW: (150, 150, 150)}


def gallery(cfg_f):
    """170 MEVID gallery templates of the frozen configuration -> (ids, templates per backbone)."""
    import mevid_errors as E
    gw = np.load(E.M / "gallery_wf.npz", allow_pickle=True)
    ga = np.load(E.M / "gallery.npz", allow_pickle=True)
    ok = ga["face_h"] >= E.GAL_BAR
    g = dict(ids=np.array(sorted(set(ga["person"][ok].tolist()))), ok=ok, person=ga["person"], ms1=ga["emb_adaface"], **{k: gw[k] for k in gw.files if k.startswith("feat")})
    flip = "_flip" if cfg_f["flip"] else ""
    wkey = f"resmatch{flip}" if cfg_f["gallery"] == "resmatch" else f"feat{flip}_native"
    T = {"wf": E.templates(g, wkey)}
    if cfg_f["fusion"]:
        T["ms1"] = E.templates(g, "ms1")
    return g["ids"], T


def run_tracker(cfg, out, emb, lam, min_h):
    z = np.load(out / "detections.npz")
    frames, fi, dets, shape = z["frames"], z["frame_idx"], z["dets"], tuple(int(x) for x in z["shape"])
    e = np.full((len(dets), 512), np.nan, np.float32)
    e[emb["det_id"]] = emb["emb_adaface_wf4m"]
    tr = TrackerAdapter("bytetrack", cfg["tracking"]["frame_rate"], match_iou=cfg["tracking"]["match_iou"], lam=lam, min_h=min_h)
    starts, ends = np.searchsorted(fi, frames, side="left"), np.searchsorted(fi, frames, side="right")
    tid = np.full(len(dets), -1, int)
    t0 = time.perf_counter()
    for k in range(len(frames)):
        s, t = starts[k], ends[k]
        tid[s:t] = tr.update(dets[s:t], shape, embs=e[s:t])
    return fi, dets, tid, time.perf_counter() - t0


def identify(fi, dets, tid, emb, ids, T, fz, fc):
    """Per track -> label, mean usable height, top-1 cosine, top-1 id, counts."""
    row_of = {int(d): i for i, d in enumerate(emb["det_id"])}
    wf, ms1 = emb["emb_adaface_wf4m"], emb.get("emb_adaface")
    index = {k: GalleryIndex(v, "flat") for k, v in T.items()}
    tracks, t0 = [], time.perf_counter()
    for t in np.unique(tid[tid >= 0]):
        rows = np.flatnonzero(tid == t)
        er = [r for r in rows if int(r) in row_of]  # faces with score >= 0.5 (embedded)
        h = dets[er, 3] - dets[er, 1] if er else np.zeros(0)
        use = [r for r, hh in zip(er, h) if hh >= fc["min_face_h"]]
        rec = dict(track=int(t), n_dets=len(rows), n_faces=len(er), n_usable=len(use), first_frame=int(fi[rows[0]]), last_frame=int(fi[rows[-1]]))
        if not use:
            tracks.append(dict(rec, label=LOW, mean_h=np.nan, sim=np.nan, top_id=-1))
            continue
        k = [row_of[int(r)] for r in use]
        q, idx, hu = emb["mag_norm"][k], fi[use], dets[use, 3] - dets[use, 1]
        s = index["wf"].search(track_template(wf[k], q, idx, fz["config"]["rule"])[None].astype(np.float32), len(ids))
        sim = np.zeros(len(ids))
        sim[s[1][0]] = s[0][0]
        if "ms1" in index:
            s2 = index["ms1"].search(track_template(ms1[k], q, idx, fz["config"]["rule"])[None].astype(np.float32), len(ids))
            sim2 = np.zeros(len(ids))
            sim2[s2[1][0]] = s2[0][0]
            sim = 0.5 * sim + 0.5 * sim2
        top = int(np.argmax(sim))
        label = decide(float(hu.mean()), float(sim[top]), int(ids[top]), fz["demo_threshold"], fc["bar_px"])
        tracks.append(dict(rec, label=str(label), mean_h=float(hu.mean()), sim=float(sim[top]), top_id=int(ids[top])))
    return pd.DataFrame(tracks), time.perf_counter() - t0


def render(source, out, fi, dets, tid, tracks, fps_info, limit=None, width=1280):
    lab = {r.track: r for r in tracks.itertuples()}
    starts = {}
    for i, f in enumerate(fi):
        starts.setdefault(int(f), []).append(i)
    vw = None
    for idx, _ts, bgr in iter_frames(source, 30, limit, errors=[]):
        if bgr is None:
            continue
        s = width / bgr.shape[1]
        im = cv2.resize(bgr, (width, round(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
        if vw is None:
            vw = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (im.shape[1], im.shape[0]))
        for i in starts.get(idx, []):
            t = int(tid[i])
            if t < 0 or dets[i, 4] < 0.5:
                continue
            r = lab[t]
            name = f"ID {r.top_id}" if r.label not in (LOW, UNKNOWN) else r.label
            col = COLORS["id"] if r.label not in (LOW, UNKNOWN) else COLORS[r.label]
            x1, y1, x2, y2 = (dets[i, :4] * s).astype(int)
            cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
            cv2.putText(im, f"T{t} {name}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.putText(im, fps_info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(im)
    if vw is not None:
        vw.release()


def run(video, mode, limit=None, role_cam=None):
    cfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    fc, fz = cfg["final"], json.loads(FROZEN.read_text())
    cfg["pipeline"]["detector_mode"] = mode
    cfg["recognizer_flip"] = {**cfg.get("recognizer_flip", {}), "adaface_wf4m": bool(fz["config"]["flip"])}
    clip = Path(video).stem
    out = DEMO / (clip + ("_smoke" if limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    from src.video.embed import FaceEmbedder
    det = P.build_detector(cfg)
    a = P.detect_clip(det, cfg, video, out, limit)
    recs = ("adaface_wf4m",) + (("adaface",) if fz["config"]["fusion"] else ())
    b = P.embed_clip(FaceEmbedder(cfg, recs), cfg, video, out, limit)
    emb = dict(np.load(out / "embed.npz"))
    fi, dets, tid, s_track = run_tracker(cfg, out, emb, fc["face_lambda"], fc["face_min_h"])
    ids, T = gallery(fz["config"])
    tracks, s_id = identify(fi, dets, tid, emb, ids, T, fz, fc)
    n_frames = a["n_frames"]
    total = a["seconds_detect"] + b["seconds_embed"] + s_track + s_id
    fps_info = f"{clip} | mode {mode} | pipeline {n_frames / total:.1f} fps (lab PC, offline)"
    render(video, out, fi, dets, tid, tracks, fps_info, limit)
    pd.DataFrame(dict(frame=fi, x1=dets[:, 0], y1=dets[:, 1], x2=dets[:, 2], y2=dets[:, 3], score=dets[:, 4], track=tid)).to_csv(out / "tracks.csv", index=False, float_format="%.3f")
    tracks.to_csv(out / "track_labels.csv", index=False, float_format="%.5f")
    face = tracks[tracks.n_faces >= 1]
    prim = tracks[tracks.n_usable >= 5]
    ident = ~face.label.isin([LOW, UNKNOWN])
    summ = dict(clip=clip, camera=role_cam, mode=mode, n_frames=int(n_frames), n_face_tracks=int(len(face)), n_tracks_5_usable=int(len(prim)),
                coverage_primary=float((prim.mean_h >= fc["bar_px"]).mean()) if len(prim) else None, coverage_broad=float((face.mean_h >= fc["bar_px"]).mean()) if len(face) else None,
                coverage_floor=0.60, identified=int(ident.sum()), unknown=int((face.label == UNKNOWN).sum()), low_quality=int((face.label == LOW).sum()),
                seconds=dict(detect=a["seconds_detect"], embed=b["seconds_embed"], track=s_track, identify=s_id), fps_offline=n_frames / total, threshold=fz["demo_threshold"], limit=limit)
    summ["below_floor"] = (summ["coverage_primary"] is not None and summ["coverage_primary"] < 0.60)
    if not limit:
        SUMMARY.mkdir(parents=True, exist_ok=True)
        (SUMMARY / f"{clip}.json").write_text(json.dumps(summ, indent=1, sort_keys=True))
    print(json.dumps(summ, indent=1, sort_keys=True))
    return summ


def calibrated_mode(calib_clip):
    f = ROOT / "outputs/video/phase9/calibration" / f"{Path(calib_clip).stem}.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([sys.executable, str(ROOT / "scripts/video_mode_calibrate.py"), "--source", str(MEVA / calib_clip), "--out", str(f)], cwd=ROOT)
        if r.returncode != 0:  # no face in the sampled calibration frames: the compromise mode (REPORT_phase9.md section 0.6 F)
            f.write_text(json.dumps(dict(source=str(MEVA / calib_clip), recommended_mode="tiled", note="no face found in the calibration frames; tiled by the section 0.6 rule")))
    return json.loads(f.read_text())["recommended_mode"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="only the per-camera mode calibration -> outputs/video/phase9/calibration_modes.json")
    a = ap.parse_args()
    if a.calibrate:
        rows = [line.split() for line in (MEVA / "clips.txt").read_text().splitlines() if line.strip()]
        modes = {cam: calibrated_mode(f) for role, cam, f in rows if role == "calib"}
        (ROOT / "outputs/video/phase9/calibration_modes.json").write_text(json.dumps(modes, indent=1, sort_keys=True))
        print(modes)
    elif a.all:
        rows = [line.split() for line in (MEVA / "clips.txt").read_text().splitlines() if line.strip()]
        calib = {cam: f for role, cam, f in rows if role == "calib"}
        for role, cam, f in rows:
            if role == "demo":
                run(MEVA / f, a.mode or calibrated_mode(calib[cam]), a.limit, cam)
    else:
        run(a.video, a.mode or "full1280", a.limit)
