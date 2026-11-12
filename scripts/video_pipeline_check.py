"""Phase 6 acceptance checks 4 (determinism) and 5 (coverage).

    envs/video/Scripts/python.exe scripts/video_pipeline_check.py [--clip P2L_S4_C3.2]

Coverage: for every clip in outputs/video/pipeline/chokepoint (72 ground-truth + 6 crowded), `detections.npz["frames"]` must equal the listed frames of the clip
(every listed frame has an entry; an unreadable image is allowed only if it is logged in `unreadable`), tracks.csv must exist and embed.npz must exist.
Determinism: rerun one clip end to end into a separate folder with the same config; compare detections.npz, tracks.csv and embed.npz with the original run
(`seconds_*` entries are wall-clock and excluded). Differences are reported as found (max abs difference for floats), not rounded away.
Writes outputs/video/phase6_checks.json (keys coverage, determinism; existing keys are kept).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402,F401
from src.video import pipeline as P  # noqa: E402
from src.video.io import list_frames  # noqa: E402

OUT = ROOT / "outputs/video/phase6_checks.json"


def coverage(cfg):
    gt, crowded = P.chokepoint_clips()
    res, bad = {}, []
    for clip in gt + crowded:
        d = P.clip_out(cfg, "chokepoint", clip)
        listed = [i for i, _ in list_frames(ROOT / "Dataset/ChokePoint" / clip)]
        if not (d / "detections.npz").exists():
            bad.append((clip, "no detections.npz"))
            continue
        z = np.load(d / "detections.npz")
        frames, unread = z["frames"].tolist(), set(z["unreadable"].tolist() if "unreadable" in z.files else [])
        ok = frames == listed and unread <= set(listed) and (d / "tracks.csv").exists() and (d / "embed.npz").exists()
        res[clip] = dict(listed=len(listed), covered=len(frames), unreadable=sorted(unread))
        if not ok:
            bad.append((clip, "frame list differs or tracks/embed missing"))
    return dict(n_clips=len(res), n_gt=len(gt), n_crowded=len(crowded), total_frames=sum(r["listed"] for r in res.values()),
                unreadable={c: r["unreadable"] for c, r in res.items() if r["unreadable"]}, failures=bad, ok=not bad)


def cmp_npz(a, b):
    za, zb = np.load(a), np.load(b)
    out = {}
    for k in sorted(set(za.files) | set(zb.files)):
        if k.startswith("seconds_"):
            continue
        if k not in za.files or k not in zb.files:
            out[k] = "missing in one"
        elif za[k].shape != zb[k].shape:
            out[k] = f"shape {za[k].shape} vs {zb[k].shape}"
        elif np.array_equal(za[k], zb[k]):
            out[k] = "identical"
        else:
            out[k] = f"differs, max abs diff {float(np.abs(za[k].astype(float) - zb[k].astype(float)).max()):.3g}"
    return out


def determinism(cfg, clip):
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["pipeline"]["out"] = "outputs/video/determinism_rerun"
    src = ROOT / "Dataset/ChokePoint" / clip
    out2 = P.clip_out(cfg2, "chokepoint", clip)
    P.detect_clip(P.build_detector(cfg2), cfg2, src, out2)
    P.track_clip(cfg2, out2)
    from src.video.embed import FaceEmbedder
    P.embed_clip(FaceEmbedder(cfg2, tuple(cfg2["pipeline"]["recognizers"])), cfg2, src, out2)
    out1 = P.clip_out(cfg, "chokepoint", clip)
    r = dict(clip=clip, detections=cmp_npz(out1 / "detections.npz", out2 / "detections.npz"), embed=cmp_npz(out1 / "embed.npz", out2 / "embed.npz"),
             tracks_csv_identical=(out1 / "tracks.csv").read_bytes() == (out2 / "tracks.csv").read_bytes())
    r["ok_byte_identical"] = r["tracks_csv_identical"] and all(v == "identical" for d in (r["detections"], r["embed"]) for v in d.values())
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/video.yaml")
    ap.add_argument("--clip", default="P2L_S4_C3.2")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["coverage"] = coverage(cfg)
    print("coverage", json.dumps(res["coverage"], indent=1), flush=True)
    res["determinism"] = determinism(cfg, a.clip)
    print("determinism", json.dumps(res["determinism"], indent=1))
    OUT.write_text(json.dumps(res, indent=1))
