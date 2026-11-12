"""Phase 7 benchmark protocols B, D, E on ChokePoint (pre-registered in REPORT_phase6.md section 0b, committed before any number).

    envs/video/Scripts/python.exe scripts/video_phase7_bench.py --part D0          # section 0 rows read at FPIR 0.3 / 0.1 / 0.01 (read-out, no new GPU work)
    envs/video/Scripts/python.exe scripts/video_phase7_bench.py --part B           # thin gallery k = 1 and k = 5, 20 draws, 3 recognizers (cached embeddings)
    envs/video/Scripts/python.exe scripts/video_phase7_bench.py --part E           # degraded probes s in {56,32,24,16,12}, AdaFace, every 5th matched detection

Shared with section 0: matches, groups, SC/MC, left-out identities (rng 42), TPIR20 arithmetic (src/eval_protocols), cluster bootstrap over identities (1,000 draws, seed 42).
Outputs in outputs/video/bench7/: D0.csv, B.csv, E.csv (+ E embedding cache in emb/).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402,F401
import video_single_frame as S  # noqa: E402
from src import eval_protocols as ev  # noqa: E402

OUT = ROOT / "outputs/video/bench7"
FPIRS = [0.3, 0.1, 0.01]
SIZES = [56, 32, 24, 16, 12]
KS, R = [1, 5], 20


def tpir_at(s1, gt):
    """TPIR20 at each FPIR in FPIRS, same threshold grid and pick rule as survface_openset_from_scores."""
    s1s, gts = np.sort(s1), np.sort(gt)
    step = (s1s[-1] - s1s[0]) / 1000
    th = np.arange(s1s[0], s1s[-1] + step / 2, step)
    fpir = ev._count_below(s1s, th) / len(s1s)
    tpir = ev._count_below(gts, th) / len(gts)
    o = np.argsort(fpir, kind="stable")
    fpir, tpir = fpir[o], tpir[o]
    return [ev._pick(fpir, tpir, f) for f in FPIRS]


def direction(gal, probe, left_out):
    """-> closed (pid, ok) frame, open frame (pid, kind, v), n_unmated. Same logic as video_single_frame.score_direction."""
    closed, opn, pt = S.score_direction(gal, probe, left_out)
    return closed, opn, pt


def metrics(closed, opn):
    s1, gt = opn[opn.kind == "s1"].v.to_numpy(), opn[opn.kind == "gt"].v.to_numpy()
    return [float(closed.ok.mean())] + tpir_at(s1, gt)


def boot(parts, rng):
    """parts: list of (closed, opn) that share the identity set (several draws of one condition). One bootstrap draw: identities resampled with replacement
    (with all their probes), metrics averaged over the parts. Returns a vector [rank1, tpir@FPIRS...] or None."""
    ids = np.array(sorted(set().union(*[set(c.pid) | set(o.pid) for c, o in parts])))
    sel = ids[rng.integers(0, len(ids), len(ids))]
    vals = []
    for closed, opn in parts:
        cl = closed.groupby("pid").ok.agg(["sum", "count"])
        s1 = opn[opn.kind == "s1"].groupby("pid").v.apply(np.asarray)
        gt = opn[opn.kind == "gt"].groupby("pid").v.apply(np.asarray)
        a = [s1[i] for i in sel if i in s1.index]
        b = [gt[i] for i in sel if i in gt.index]
        n = sum(cl.loc[i, "count"] for i in sel if i in cl.index)
        if not a or not b or n == 0:
            return None
        vals.append([sum(cl.loc[i, "sum"] for i in sel if i in cl.index) / n] + tpir_at(np.concatenate(a), np.concatenate(b)))
    return np.mean(vals, axis=0)


def row(base, parts_pt, parts_cb, n_boot, extra=None):
    """One output row: point value averaged over the parts, CI from the cluster bootstrap (shared identity resamples)."""
    rng = np.random.default_rng(S.SEED)
    d = np.array([x for x in (boot(parts_cb, rng) for _ in range(n_boot)) if x is not None])
    pts = np.mean([metrics(c, o) for c, o in parts_cb], axis=0)
    names = ["rank1"] + [f"tpir20_fpir{f:g}" for f in FPIRS]
    out = dict(base, **(extra or {}), n_boot_ok=len(d))
    for i, n in enumerate(names):
        out[n] = float(pts[i])
        out[n + "_ci_lo"], out[n + "_ci_hi"] = float(np.quantile(d[:, i], 0.025)), float(np.quantile(d[:, i], 0.975))
    return out


def setup():
    cfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    gt = S.load_gt()
    all_clips = sorted(gt)
    used = sorted({c for g in S.G for cond in ("SC", "MC") for c in S.clips_of(g, cond, all_clips)})
    m, _ = S.get_matches(used, gt, cfg["pipeline"]["embed_floor"])
    ids_all = sorted(m.pid.unique().tolist())
    left_out = set(np.random.default_rng(S.SEED).choice(ids_all, int(round(S.LEAVE_OUT * len(ids_all))), replace=False).tolist())
    return cfg, all_clips, m, left_out


def avg_dirs(rows_by_dir):
    """Average the two directions: point values and CI endpoints of the mean of the per-direction bootstrap draws are not stored, so the CI is
    the per-direction CI averaged element-wise (stated in the report); the point value is the plain mean."""
    keys = [k for k in rows_by_dir[0] if isinstance(rows_by_dir[0][k], float)]
    return {k: float(np.mean([r[k] for r in rows_by_dir])) for k in keys}


def part_D0(a):
    cfg, allc, m, lo = setup()
    rows, box = [], {}
    for rec in a.recognizers:
        for cond in ("SC", "MC"):
            per = []
            for g, p in (("G1", "G2"), ("G2", "G1")):
                gal = S.build(rec, S.clips_of(g, cond, allc), m, cfg, box)
                pro = S.build(rec, S.clips_of(p, cond, allc), m, cfg, box)
                c, o, _ = direction(gal, pro, lo)
                per.append(row(dict(recognizer=rec, condition=cond, direction=f"{g}->{p}"), None, [(c, o)], a.n_boot))
            rows += per + [dict(recognizer=rec, condition=cond, direction="mean", **avg_dirs(per))]
            print(rec, cond, {k: round(v, 4) for k, v in rows[-1].items() if k in ("rank1", "tpir20_fpir0.3", "tpir20_fpir0.1", "tpir20_fpir0.01")}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "D0.csv", index=False, float_format="%.5f")


def thin(gal, k, r):
    """k random rows per identity (all rows if fewer), draw r."""
    rng = np.random.default_rng([S.SEED, r])
    gp, ge = gal
    keep = []
    for i in np.unique(gp):
        ix = np.flatnonzero(gp == i)
        keep.extend(ix if len(ix) <= k else rng.choice(ix, k, replace=False))
    keep = np.array(sorted(keep))
    return gp[keep], ge[keep]


def part_B(a):
    cfg, allc, m, lo = setup()
    rows, box = [], {}
    for rec in a.recognizers:
        for cond in ("SC", "MC"):
            for k in KS:
                per = []
                for g, p in (("G1", "G2"), ("G2", "G1")):
                    gal = S.build(rec, S.clips_of(g, cond, allc), m, cfg, box)
                    pro = S.build(rec, S.clips_of(p, cond, allc), m, cfg, box)
                    parts = [direction(thin(gal, k, r), pro, lo)[:2] for r in range(R)]
                    ms = np.array([metrics(c, o) for c, o in parts])
                    per.append(row(dict(recognizer=rec, condition=cond, k=k, direction=f"{g}->{p}"), None, parts, a.n_boot,
                                   dict(rank1_draw_min=float(ms[:, 0].min()), rank1_draw_max=float(ms[:, 0].max()), n_draws=R)))
                rows += per + [dict(recognizer=rec, condition=cond, k=k, direction="mean", **avg_dirs(per))]
                print("B", rec, cond, "k", k, {kk: round(v, 4) for kk, v in rows[-1].items() if kk in ("rank1", "tpir20_fpir0.3", "tpir20_fpir0.01", "rank1_draw_min")}, flush=True)
                pd.DataFrame(rows).to_csv(OUT / "B.csv", index=False, float_format="%.5f")


def degrade(crop, s):
    return cv2.resize(cv2.resize(crop, (s, s), interpolation=cv2.INTER_AREA), (112, 112), interpolation=cv2.INTER_LINEAR)


def e_embed(clip, sub, cfg, embedder):
    """Degraded-probe AdaFace embeddings and MagFace norms for the subsampled matched detections of one clip. Cached."""
    f = OUT / "emb" / f"{clip}.npz"
    ids = sub.det_id.to_numpy()
    if f.exists():
        z = np.load(f)
        if np.array_equal(z["det_id"], ids):
            return {s: (z[f"emb_{s}"], z[f"mag_{s}"]) for s in SIZES}
    from src.video.io import iter_frames
    d = np.load(ROOT / f"outputs/video/pipeline/chokepoint/{clip}/detections.npz")
    kps, fi = d["kps"], d["frame_idx"]
    by_frame = {}
    for x in ids:
        by_frame.setdefault(int(fi[x]), []).append(int(x))
    crops = {}
    for idx, _ts, bgr in iter_frames(ROOT / "Dataset/ChokePoint" / clip, cfg["pipeline"]["fps"], None, errors=[]):
        rows = by_frame.get(idx)
        if not rows or bgr is None:
            continue
        for r, c in zip(rows, embedder.crops(bgr, kps[rows])):
            crops[r] = c
    ordered = [crops[int(x)] for x in ids]
    out, save = {}, {"det_id": ids}
    for s in SIZES:
        r = embedder.embed_crops([degrade(c, s) for c in ordered])
        out[s] = (r["emb"]["adaface"], r["mag_norm"])
        save[f"emb_{s}"], save[f"mag_{s}"] = out[s]
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **save)
    return out


def part_E(a):
    cfg, allc, m, lo = setup()
    from src.video.embed import FaceEmbedder
    embedder = FaceEmbedder(cfg, ("adaface",))
    sub = {c: m[m["clip"] == c].sort_values(["frame", "det_id"]).iloc[::5] for c in allc}
    cache = {}
    for k, c in enumerate(allc, 1):
        if len(sub[c]):
            cache[c] = e_embed(c, sub[c], cfg, embedder)
        print(f"E embed [{k}/{len(allc)}] {c} {len(sub[c])} probes", flush=True)
    rows, box = [], {}
    for cond in ("SC", "MC"):
        for s in SIZES + [112]:
            per, mags = [], []
            for g, p in (("G1", "G2"), ("G2", "G1")):
                gal = S.build("adaface", S.clips_of(g, cond, allc), m, cfg, box)
                pids, embs, mag = [], [], []
                for c in S.clips_of(p, cond, allc):
                    if not len(sub[c]):
                        continue
                    pids.append(sub[c].pid.to_numpy())
                    if s == 112:
                        embs.append(S.adaface_emb(c, sub[c].det_id.to_numpy()))
                    else:
                        embs.append(cache[c][s][0])
                        mag.append(cache[c][s][1])
                pro = (np.concatenate(pids), ev.l2n(np.concatenate(embs)))
                c_, o_, _ = direction(gal, pro, lo)
                per.append(row(dict(recognizer="adaface", condition=cond, probe_side=s, direction=f"{g}->{p}"), None, [(c_, o_)], a.n_boot, dict(n_probes=len(pro[0]))))
                if mag:
                    mags.append(float(np.concatenate(mag).mean()))
            rows += per + [dict(recognizer="adaface", condition=cond, probe_side=s, direction="mean", mean_mag_norm=float(np.mean(mags)) if mags else np.nan, **avg_dirs(per))]
            print("E", cond, "s", s, {kk: round(v, 4) for kk, v in rows[-1].items() if kk in ("rank1", "tpir20_fpir0.3", "tpir20_fpir0.01", "mean_mag_norm")}, flush=True)
            pd.DataFrame(rows).to_csv(OUT / "E.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["D0", "B", "E"])
    ap.add_argument("--recognizers", nargs="+", default=["adaface", "magface", "curricularface"])
    ap.add_argument("--n-boot", type=int, default=S.N_BOOT)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"D0": part_D0, "B": part_B, "E": part_E}[a.part](a)
