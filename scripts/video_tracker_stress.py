"""Phase 7 tracker stress test (protocol: REPORT_phase7.md section 0, Addendum 2, written before this script was run).

    envs/video/Scripts/python.exe scripts/video_tracker_stress.py

ByteTrack and BoT-SORT (Phase 6 settings) on the Phase 6 detections of the 72 ground-truth clips under: S0 original; S1/S2/S3 every 3rd/5th/10th frame (tracker frame rate 30/k);
S4/S5 each detection dropped with probability 0.1/0.2. TrackEval metrics against the Phase 6 ground truth on the kept frames. Decision statistic: the per-clip IDF1 difference
ByteTrack minus BoT-SORT averaged over S1-S5, mean over clips, cluster bootstrap over clips (1,000 draws, seed 42); rule: CI excludes zero -> the better tracker, else ByteTrack.
Outputs: outputs/video/agg7/tracker_stress/{per_clip.csv, summary.csv, decision.json}. Resumable per clip (per_clip rows are appended to a cache and reused).
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.video import mot_eval as M  # noqa: E402
from src.video.track import TrackerAdapter  # noqa: E402
from video_tracker_eval import load_gt  # noqa: E402

OUT = ROOT / "outputs/video/agg7/tracker_stress"
PIPE = ROOT / "outputs/video/pipeline/chokepoint"
TRACKERS = ["bytetrack", "botsort"]
CONDS = {"S0": dict(k=1, p=0.0), "S1": dict(k=3, p=0.0), "S2": dict(k=5, p=0.0), "S3": dict(k=10, p=0.0), "S4": dict(k=1, p=0.1), "S5": dict(k=1, p=0.2)}
STRESS = ["S1", "S2", "S3", "S4", "S5"]
KEYS = ["IDF1", "HOTA", "AssA", "DetA", "IDSW"]
SEED, N_BOOT = 42, 1000


def run_clip(clip, ci, gt, cfg, trackers=None, params=None, lam=0.0, embs=None, assign_out=None):
    z = np.load(PIPE / clip / "detections.npz")
    frames, fi, dets, shape = z["frames"].astype(int), z["frame_idx"].astype(int), z["dets"], tuple(int(x) for x in z["shape"])
    starts, ends = np.searchsorted(fi, frames, side="left"), np.searchsorted(fi, frames, side="right")
    rows = []
    for cname, c in CONDS.items():
        k, p = c["k"], c["p"]
        keep = np.ones(len(dets), bool)
        if p > 0:
            keep = np.random.default_rng([SEED, ci, int(round(100 * p))]).random(len(dets)) >= p
        sel = np.arange(0, len(frames), k)
        for tname in (trackers or TRACKERS):
            tr = TrackerAdapter(tname, cfg["tracking"]["frame_rate"] / k, params=params if tname == "botsort" else None, match_iou=cfg["tracking"]["match_iou"], lam=lam)
            pred, ids_seen = {}, set()
            assign = np.full(len(dets), -1, np.int32) if assign_out is not None else None  # track id of every detection row (-1: dropped, frame not kept, or no track)
            for j in sel:
                s, e = starts[j], ends[j]
                m = keep[s:e]
                d = dets[s:e][m]
                tid = tr.update(d, shape, embs=embs[s:e][m] if embs is not None else None)
                if assign is not None:
                    assign[np.arange(s, e)[m]] = tid
                ok = tid >= 0
                if ok.any():
                    pred[int(frames[j])] = (tid[ok], d[ok, :4])
                    ids_seen.update(tid[ok].tolist())
            if assign_out is not None:
                assign_out[f"{cname}|{tname}"] = assign
            res = M.clip_metrics(gt[clip], pred, [int(frames[j]) for j in sel])
            rows.append(dict(clip=clip, cond=cname, tracker=tname, n_frames=len(sel), n_tracks=len(ids_seen), **{kk: res[kk] for kk in KEYS}))
    return rows


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    gt = load_gt()
    clips = sorted(gt)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "per_clip.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    have = set(done.clip.unique()) if len(done) else set()
    rows = [done] if len(done) else []
    t0 = time.perf_counter()
    for ci, clip in enumerate(clips):
        if clip in have:
            continue
        rows.append(pd.DataFrame(run_clip(clip, ci, gt, cfg)))
        pd.concat(rows, ignore_index=True).to_csv(cache, index=False, float_format="%.6f")
        print(f"[{ci + 1}/{len(clips)}] {clip} {time.perf_counter() - t0:.0f}s", flush=True)
    d = pd.read_csv(cache)
    # per condition summary (descriptive)
    piv = {k: d.pivot_table(index=["clip", "cond"], columns="tracker", values=k) for k in KEYS}
    srows = []
    for cname in CONDS:
        r = dict(cond=cname, k=CONDS[cname]["k"], dropout=CONDS[cname]["p"])
        for k in KEYS:
            x = piv[k].xs(cname, level="cond")
            r.update({f"{k}_bytetrack": x["bytetrack"].mean(), f"{k}_botsort": x["botsort"].mean(), f"{k}_diff": (x["bytetrack"] - x["botsort"]).mean()})
        t = d[d.cond == cname].pivot_table(index="clip", columns="tracker", values="n_tracks")
        r.update(tracks_bytetrack=int(t["bytetrack"].sum()), tracks_botsort=int(t["botsort"].sum()), idsw_total_bytetrack=float(d[(d.cond == cname) & (d.tracker == "bytetrack")].IDSW.sum()),
                 idsw_total_botsort=float(d[(d.cond == cname) & (d.tracker == "botsort")].IDSW.sum()))
        srows.append(r)
    pd.DataFrame(srows).to_csv(OUT / "summary.csv", index=False, float_format="%.5f")
    # primary statistic: per-clip IDF1 difference averaged over S1-S5; bootstrap over clips
    idf1 = piv["IDF1"]
    diff = (idf1["bytetrack"] - idf1["botsort"]).unstack("cond")[STRESS].mean(1)  # per clip
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), (N_BOOT, len(diff)))
    boot = diff.to_numpy()[idx].mean(1)
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    mean = float(diff.mean())
    excludes = bool(lo > 0 or hi < 0)
    winner = ("bytetrack" if mean > 0 else "botsort") if excludes else "bytetrack"
    text = (ROOT / "REPORT_phase7.md").read_text(encoding="utf-8")
    m = re.search(r"- \*\*Addendum 2 .*?(?=\n- \*\*Confirmations)", text, re.S)
    out = dict(primary_statistic="mean over clips of the per-clip IDF1 difference (ByteTrack minus BoT-SORT) averaged over S1-S5", mean_difference=mean, ci95=[lo, hi], n_clips=len(diff),
               ci_excludes_zero=excludes, provisional_default=winner, rule="CI excludes zero -> tracker with higher IDF1, else ByteTrack",
               clips_bytetrack_better=int((diff > 0).sum()), clips_equal=int((diff == 0).sum()), clips_botsort_better=int((diff < 0).sum()),
               protocol_paragraph_sha256=hashlib.sha256(m.group(0).encode("utf-8")).hexdigest() if m else None)
    (OUT / "decision.json").write_text(json.dumps(out, indent=1))
    print(pd.DataFrame(srows)[["cond", "IDF1_bytetrack", "IDF1_botsort", "IDF1_diff", "HOTA_diff", "tracks_bytetrack", "tracks_botsort", "idsw_total_bytetrack", "idsw_total_botsort"]].round(4).to_string(index=False))
    print(json.dumps(out, indent=1))


def exploratory(prox):
    """EXPLORATORY (not part of the pre-registered decision): BoT-SORT with `proximity_thresh` relaxed to `prox` (default 0.5: IoU of at least 0.5 is required to associate; ByteTrack's
    `match_thresh` 0.8 accepts IoU of at least 0.2) under the same six conditions, compared with the main run's ByteTrack and default BoT-SORT rows."""
    cfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    gt = load_gt()
    out = OUT / f"exploratory_botsort_proximity{prox}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ci, clip in enumerate(sorted(gt)):
        rows += run_clip(clip, ci, gt, cfg, trackers=["botsort"], params=dict(proximity_thresh=prox))
        print(f"[{ci + 1}/72] {clip}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "per_clip.csv", index=False, float_format="%.6f")
    main_d = pd.read_csv(OUT / "per_clip.csv")
    srows = []
    for cname in CONDS:
        x = d[d.cond == cname].set_index("clip")
        b = main_d[(main_d.cond == cname) & (main_d.tracker == "bytetrack")].set_index("clip")
        o = main_d[(main_d.cond == cname) & (main_d.tracker == "botsort")].set_index("clip")
        srows.append(dict(cond=cname, IDF1_bytetrack=b.IDF1.mean(), IDF1_botsort_default=o.IDF1.mean(), IDF1_botsort_relaxed=x.IDF1.mean(), IDF1_bytetrack_minus_relaxed=(b.IDF1 - x.IDF1).mean(),
                          tracks_bytetrack=int(b.n_tracks.sum()), tracks_botsort_default=int(o.n_tracks.sum()), tracks_botsort_relaxed=int(x.n_tracks.sum()),
                          idsw_bytetrack=float(b.IDSW.sum()), idsw_botsort_default=float(o.IDSW.sum()), idsw_botsort_relaxed=float(x.IDSW.sum())))
    pd.DataFrame(srows).to_csv(out / "summary.csv", index=False, float_format="%.5f")
    print(pd.DataFrame(srows).round(4).to_string(index=False))


def face_embs(clip, n_dets):
    """AdaFace embeddings aligned to the clip's detection rows (NaN where a detection has none, i.e. score < 0.5)."""
    z = np.load(PIPE / clip / "embed.npz")
    e = np.full((n_dets, 512), np.nan, np.float32)
    e[z["det_id"]] = z["emb_adaface"]
    return e


def face_term(lam):
    """Phase 8 Part D: the six conditions with the face term of weight `lam` (0 = the Phase 7 behaviour; acceptance check 5 compares this run with the Phase 7 per_clip.csv).
    Outputs outputs/video/mevid/tracker_face/lam<lam>/{per_clip.csv, summary.csv}."""
    cfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    gt = load_gt()
    out = ROOT / "outputs/video/mevid/tracker_face" / f"lam{lam:g}"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "per_clip.csv"
    rows = [pd.read_csv(cache)] if cache.exists() else []
    have = set(rows[0].clip.unique()) if rows else set()
    for ci, clip in enumerate(sorted(gt)):
        if clip in have:
            continue
        n = len(np.load(PIPE / clip / "detections.npz")["dets"])
        asg = {}
        rows.append(pd.DataFrame(run_clip(clip, ci, gt, cfg, lam=lam, embs=face_embs(clip, n) if lam > 0 else None, assign_out=asg)))
        (out / "tracks").mkdir(exist_ok=True)
        np.savez(out / "tracks" / f"{clip}.npz", **asg)
        pd.concat(rows, ignore_index=True).to_csv(cache, index=False, float_format="%.6f")
        print(f"[{ci + 1}/72] {clip}", flush=True)
    d = pd.read_csv(cache)
    g = d.groupby(["cond", "tracker"])[KEYS + ["n_tracks"]].agg({**{k: "mean" for k in KEYS}, "n_tracks": "sum"}).reset_index()
    g.to_csv(out / "summary.csv", index=False, float_format="%.5f")
    print(g.round(4).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--exploratory-botsort-proximity":
        exploratory(float(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "--lam":
        face_term(float(sys.argv[2]))
    else:
        main()
