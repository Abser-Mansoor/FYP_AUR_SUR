"""Phase 9 Part D: frame rate at 720p by stage (REPORT_phase9.md section 0.6 D). Single stream, batch 1 (one frame at a time), GPU timings synchronised.

    envs/video/Scripts/python.exe scripts/phase9_fps.py                        # lab PC: the 8 MEVA demo clips in Dataset/MEVA (clips.txt)
    python scripts/phase9_fps.py --clips-dir /content/meva --device-tag t4     # Colab T4 (runbook: research/Sprint_Notes/Phase9_T4_fps_runbook.md)

Per clip: the first 300 frames, resized to 1280 x 720 (INTER_AREA). Stages: decode+resize (per frame); detect in full640, full1280 and tiled (per frame); align + WebFace4M,
align + MS1MV2 (flip, for F4) and the MagFace norm (per face, the faces of one frame as one batch); ByteTrack with the face term (per frame); aggregate (R4*) and search (FAISS Flat,
170 templates) per track. Output: outputs/video/phase9/fps_<tag>.csv (median ms per unit over clips) and, for the lab PC, fps_condition.json (the F4 condition of section 0.4).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.align import norm_crop  # noqa: E402
from src.video import aggregate as A  # noqa: E402
from src.video import detect as D  # noqa: E402
from src.video.index import GalleryIndex  # noqa: E402
from src.video.pipeline import build_detector  # noqa: E402
from src.video.track import TrackerAdapter  # noqa: E402
from src.recognizer import Recognizer  # noqa: E402

N_FRAMES, W, H = 300, 1280, 720
MODES = ["full640", "full1280", "tiled"]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    t = time.perf_counter()
    r = fn()
    sync()
    return r, 1000 * (time.perf_counter() - t)


def clip_frames(path):
    cap, out, ms = cv2.VideoCapture(str(path)), [], []
    while len(out) < N_FRAMES:
        t = time.perf_counter()
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
        ms.append(1000 * (time.perf_counter() - t))
    cap.release()
    return out, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", default=str(ROOT / "Dataset/MEVA"))
    ap.add_argument("--device-tag", default="labpc")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    pre = cfg["preprocess"]
    rec_wf = Recognizer("adaface", ROOT / cfg["recognizers"]["adaface_wf4m"], pre["input_size"], pre["interp"], False, 128, "cuda")
    rec_ms = Recognizer("adaface", ROOT / cfg["recognizers"]["adaface"], pre["input_size"], pre["interp"], True, 128, "cuda")
    rec_mag = Recognizer("magface", ROOT / cfg["recognizers"]["magface"], pre["input_size"], pre["interp"], False, 128, "cuda")
    det = build_detector(cfg)
    rng = np.random.default_rng(42)
    gal = GalleryIndex(rng.normal(size=(170, 512)).astype(np.float32) / np.sqrt(512), "flat")  # timing only: the search cost does not depend on the values
    clips = [line.split()[2] for line in (ROOT / "outputs/video/phase9/meva_clips.txt").read_text().splitlines() if line.startswith("demo")]
    # warm-up (CUDA kernels, cuDNN autotune)
    dummy = np.zeros((H, W, 3), np.uint8)
    for m in MODES:
        D.detect_mode(det, dummy, m)
    crop = [np.zeros((112, 112, 3), np.uint8)] * 4
    for r in (rec_wf, rec_ms, rec_mag):
        r.get_embedding(crop)
    rows = []
    for c in clips:
        p = Path(a.clips_dir) / c
        if not p.exists():
            print("missing", p)
            continue
        frames, dec = clip_frames(p)
        t = {"decode_resize": dec, **{f"detect_{m}": [] for m in MODES}, "embed_wf4m_face": [], "embed_ms1_face": [], "magface_face": [], "track": [], "aggregate_track": [], "search_track": []}
        tr = TrackerAdapter("bytetrack", 30, lam=cfg["final"]["face_lambda"], min_h=cfg["final"]["face_min_h"])
        faces_per_frame, store = [], {}
        for f in frames:
            for m in MODES:
                (d, k), ms = timed(lambda: D.detect_mode(det, f, m))
                t[f"detect_{m}"].append(ms)
                if m == "full1280":
                    dets, kps = d, k
            sel = np.flatnonzero(dets[:, 4] >= 0.5)
            faces_per_frame.append(len(sel))
            e = np.full((len(dets), 512), np.nan, np.float32)
            if len(sel):
                crops = [norm_crop(f, kps[i]) for i in sel]
                ew, ms = timed(lambda: rec_wf.get_embedding(crops))
                t["embed_wf4m_face"].append(ms / len(sel))
                _, ms = timed(lambda: rec_ms.get_embedding(crops))
                t["embed_ms1_face"].append(ms / len(sel))
                (_, q), ms = timed(lambda: rec_mag.get_embedding(crops, return_raw=True))
                t["magface_face"].append(ms / len(sel))
                e[sel] = ew
            ids, ms = timed(lambda: tr.update(dets[dets[:, 4] >= 0.1], (H, W), embs=e[dets[:, 4] >= 0.1]))
            t["track"].append(ms)
            keep = dets[:, 4] >= 0.1
            for j, (tid, ok) in enumerate(zip(ids, np.isfinite(e[keep]).all(1))):
                if tid >= 0 and ok:
                    store.setdefault(int(tid), []).append((e[keep][j], 1.0))
        for tid, v in store.items():
            emb = np.stack([x[0] for x in v])
            qq = np.linspace(20, 30, len(v))
            tmp, ms = timed(lambda: A.r4s_np_star(emb, qq))
            t["aggregate_track"].append(ms)
            _, ms = timed(lambda: gal.search(tmp[None].astype(np.float32), 1))
            t["search_track"].append(ms)
        rows.append(dict(clip=c, frames=len(frames), faces_per_frame=float(np.mean(faces_per_frame)), tracks=len(store), **{f"ms_{k}": (float(np.median(v)) if v else np.nan) for k, v in t.items()}))
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    out = ROOT / "outputs/video/phase9"
    out.mkdir(parents=True, exist_ok=True)
    d.assign(device=dev).to_csv(out / f"fps_{a.device_tag}_per_clip.csv", index=False, float_format="%.3f")
    # end-to-end per frame (median over clips): decode + detect + faces/frame x per-face embedding + track; aggregate and search are per track (amortised over the clip)
    summ = []
    for m in MODES:
        for both in (False, True):
            per = d.ms_decode_resize + d[f"ms_detect_{m}"] + d.faces_per_frame * (d.ms_embed_wf4m_face.fillna(0) + d.ms_magface_face.fillna(0) + (d.ms_embed_ms1_face.fillna(0) if both else 0)) + d.ms_track \
                + d.tracks * (d.ms_aggregate_track.fillna(0) + d.ms_search_track.fillna(0)) / d.frames
            summ.append(dict(device=dev, mode=m, backbones="WebFace4M + MS1MV2 (F4)" if both else "WebFace4M", median_ms_per_frame=float(per.median()), fps=float(1000 / per.median())))
    s = pd.DataFrame(summ)
    s.to_csv(out / f"fps_{a.device_tag}.csv", index=False, float_format="%.3f")
    print(d.round(2).to_string(index=False))
    print(s.round(2).to_string(index=False))
    if a.device_tag == "labpc":
        modes = json.loads((out / "calibration_modes.json").read_text()) if (out / "calibration_modes.json").exists() else None
        both = s[s.backbones.str.contains("F4")]
        used = sorted(set(modes.values())) if modes else MODES  # the modes the calibration assigns to the demo cameras
        worst = float(both[both["mode"].isin(used)].fps.min())
        (out / "fps_condition.json").write_text(json.dumps(dict(at_least_10_fps_with_both_backbones=bool(worst >= 10), fps_both_backbones_by_mode=dict(zip(both["mode"], both.fps)),
                                                                rule="F4 adopted only if the lab-PC end-to-end rate at 720p with both backbones is >= 10 fps in the slowest detector mode the calibration assigns to the demo cameras (section 0.4)",
                                                                modes_used=used, calibration_modes=modes), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
