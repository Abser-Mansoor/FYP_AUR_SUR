"""Phase 6 tracker validation on ChokePoint: ground-truth face tracks, the identity check, and ByteTrack against BoT-SORT.

    envs/video/Scripts/python.exe scripts/video_tracker_eval.py --gt-check      # acceptance check 1: ground-truth boxes as detections -> IDF1 >= 0.99
    envs/video/Scripts/python.exe scripts/video_tracker_eval.py --eval          # predicted tracks (needs scripts/run_video.py --stage detect|track)

Ground-truth tracks: per clip and person id, the eye labels become boxes with the fitted rule (outputs/video/gt/eye_box_rule.json). The person id is the identity;
a person who leaves and re-enters keeps the id, so the gap counts against the tracker (fragmentation / identity switch). Written in MOTChallenge format to
outputs/video/gt/mot/<clip>.txt. Frames: every frame of the clip (frames without a labelled person are empty). Metrics: vendored TrackEval (HOTA, CLEAR, Identity)
through src/video/mot_eval.py; pooled over clips with TrackEval's combine_sequences; clip-level bootstrap CI (1,000 draws, seed 42).
Outputs: outputs/video/tracker_eval/{gt_check_clips.csv, gt_check.json, per_clip.csv, pooled.csv}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.video import mot_eval as M  # noqa: E402
from src.video.io import list_frames  # noqa: E402
from src.video.track import TrackerAdapter  # noqa: E402

OUT = ROOT / "outputs/video/tracker_eval"
GT = ROOT / "outputs/video/gt"
KEYS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "CLR_FN", "CLR_FP"]


def load_gt():
    rule = json.loads((GT / "eye_box_rule.json").read_text())
    e = pd.read_csv(GT / "chokepoint_eyes.csv", dtype={"person_id": str})
    cx = (e.lx + e.rx) / 2 + rule["dx"] * e.d
    cy = (e.ly + e.ry) / 2 + rule["dy"] * e.d
    w, h = rule["w"] * e.d, rule["h"] * e.d
    e = e.assign(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2, pid=e.person_id.astype(int))
    out = {}
    for clip, g in e.groupby("clip"):
        per = {}
        for f, r in g.groupby("frame"):
            per[int(f)] = (r.pid.to_numpy(), r[["x1", "y1", "x2", "y2"]].to_numpy())
        out[clip] = per
    return out


def write_mot(gt):
    d = GT / "mot"
    d.mkdir(parents=True, exist_ok=True)
    for clip, per in gt.items():
        rows = [f"{f},{int(i)},{b[0]:.2f},{b[1]:.2f},{b[2] - b[0]:.2f},{b[3] - b[1]:.2f},1,1,1" for f in sorted(per) for i, b in zip(*per[f])]
        (d / f"{clip}.txt").write_text("\n".join(rows) + "\n")


def clip_frames(clip):
    return [i for i, _ in list_frames(ROOT / "Dataset/ChokePoint" / clip)]


def pred_from_tracks(csv, tracker):
    df = pd.read_csv(csv)
    df = df[df[f"tid_{tracker}"] >= 0]
    out = {}
    for f, r in df.groupby("frame"):
        out[int(f)] = (r[f"tid_{tracker}"].to_numpy(), r[["x1", "y1", "x2", "y2"]].to_numpy())
    return out


def bootstrap_pooled(res, n=1000, seed=42):
    clips = sorted(res)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(clips), len(clips))
        draws.append(M.pool({f"{k}_{clips[j]}": res[clips[j]] for k, j in enumerate(pick)}))
    return {k: [float(np.quantile([d[k] for d in draws], 0.025)), float(np.quantile([d[k] for d in draws], 0.975))] for k in KEYS}


def table(res_by_tracker, label):
    rows = []
    for tr, res in res_by_tracker.items():
        pooled = M.pool(res)
        ci = bootstrap_pooled(res)
        rows.append(dict(tracker=tr, subset=label, n_clips=len(res), **{k: pooled[k] for k in KEYS}, **{f"{k}_ci_lo": ci[k][0] for k in KEYS}, **{f"{k}_ci_hi": ci[k][1] for k in KEYS}))
    return rows


def gt_check(cfg, gt, clips):
    """Acceptance check 1: ground-truth boxes as detections (score 0.9) through both trackers."""
    res = {t: {} for t in cfg["tracking"]["trackers"]}
    for clip in clips:
        frames = clip_frames(clip)
        for t in res:
            tr = TrackerAdapter(t, cfg["tracking"]["frame_rate"], match_iou=cfg["tracking"]["match_iou"])
            pred = {}
            for f in frames:
                ids, boxes = gt[clip].get(f, (np.zeros(0, int), np.zeros((0, 4))))
                dets = np.hstack([boxes, np.full((len(boxes), 1), 0.9)]) if len(boxes) else np.zeros((0, 5))
                tid = tr.update(dets, (600, 800))
                keep = tid >= 0
                if keep.any():
                    pred[f] = (tid[keep], boxes[keep])
            res[t][clip] = M.clip_metrics(gt[clip], pred, frames)
        print("gt-check", clip, {t: round(res[t][clip]["IDF1"], 4) for t in res}, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/video.yaml")
    ap.add_argument("--gt-check", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--clips", nargs="+", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    gt = load_gt()
    write_mot(gt)
    clips = a.clips or sorted(gt)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.gt_check:
        res = gt_check(cfg, gt, clips)
        rows = [dict(tracker=t, clip=c, **{k: r[k] for k in KEYS}) for t in res for c, r in res[t].items()]
        pd.DataFrame(rows).to_csv(OUT / "gt_check_clips.csv", index=False, float_format="%.6f")
        pooled = {t: M.pool(res[t]) for t in res}
        ok = {t: bool(pooled[t]["IDF1"] >= 0.99) for t in res}
        out = dict(pooled=pooled, min_clip_idf1={t: float(min(r["IDF1"] for r in res[t].values())) for t in res}, ok_pooled_idf1_ge_099=ok, n_clips=len(clips))
        (OUT / "gt_check.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        sys.exit(0 if all(ok.values()) else 1)
    if a.eval:
        pdir = ROOT / cfg["pipeline"]["out"] / "chokepoint"
        res = {t: {} for t in cfg["tracking"]["trackers"]}
        for clip in clips:
            f = pdir / clip / "tracks.csv"
            if not f.exists():
                print("missing", clip)
                continue
            frames = [int(x) for x in np.load(pdir / clip / "detections.npz")["frames"]]
            for t in res:
                res[t][clip] = M.clip_metrics(gt[clip], pred_from_tracks(f, t), frames)
        rows = [dict(tracker=t, clip=c, **{k: r[k] for k in KEYS}) for t in res for c, r in res[t].items()]
        pd.DataFrame(rows).to_csv(OUT / "per_clip.csv", index=False, float_format="%.6f")
        pd.DataFrame(table(res, "all GT clips")).to_csv(OUT / "pooled.csv", index=False, float_format="%.6f")
        print(pd.read_csv(OUT / "pooled.csv")[["tracker", "n_clips"] + KEYS].round(4).to_string())


if __name__ == "__main__":
    main()
