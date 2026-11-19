"""Phase 7 per-track aggregation on ChokePoint (protocol: REPORT_phase7.md section 0, committed before any run).

    envs/video/Scripts/python.exe scripts/video_aggregation.py --part chokepoint     # GT tracks: native, 16 px, 12 px; SC and MC; rules R1-R6; paired deltas with CIs; acceptance check 3
    envs/video/Scripts/python.exe scripts/video_aggregation.py --part predicted      # ByteTrack / BoT-SORT tracks (purity, excluded tracks, same rules)
    envs/video/Scripts/python.exe scripts/video_aggregation.py --part curves         # accuracy vs number of frames and vs track quality

Inputs (all cached, CPU only): outputs/video/baseline/matches.csv (Phase 6 matches), the pipeline embed.npz files (native AdaFace + MagFace norm), the Phase 6 E cache
outputs/video/bench7/emb/<clip>.npz (embeddings of the every-5th-matched-detection subsample degraded to s px). Gallery: native, as Phase 6 section 0.
Per track (GT: all matched detections of one (clip, person id)): single-frame accuracy = mean over its frames of the frame-level correctness; aggregated accuracy = correctness
of the track's template; paired delta per track; cluster bootstrap over identities (1,000 draws, seed 42, resample identities with all their tracks); the two directions
(G1->G2, G2->G1) are averaged draw by draw. Closed-set tracks are those whose identity is enrolled in the gallery.
Outputs: outputs/video/agg7/{chokepoint.csv, predicted.csv, curves_frames.csv, curves_quality.csv, checks.json}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402,F401
import video_phase7_bench as B7  # noqa: E402
import video_single_frame as S  # noqa: E402
from src import eval_protocols as ev  # noqa: E402
from src.video import aggregate as A  # noqa: E402

OUT = ROOT / "outputs/video/agg7"
PIPE = ROOT / "outputs/video/pipeline/chokepoint"
SIZES = [112, 16, 12]  # 112 = native
RULE_NAMES = list(A.RULES) + ["R5", "R6"]
LAM_SENS = [10, 100]  # R4 sensitivity (lambda 50 is R4)
N_BOOT, SEED = 1000, 42
BONF = 9
OPEN_RULES = ["R1"]  # open-set TPIR with CIs only for the primary rule (and the single-frame reference)


# ------------------------------------------------------------------ data
def mag_of(clip, det_ids):
    z = np.load(PIPE / clip / "embed.npz")
    pos = {int(d): i for i, d in enumerate(z["det_id"])}
    return z["mag_norm"][[pos[int(d)] for d in det_ids]]


def frames_of(clips, m, size, sub):
    """Frame-level table of the matched detections of `clips` at `size` (112 = native; else the E subsample): pid, clip, frame, det_id, emb (L2-normalised), q."""
    P, C, F, D, E, Q = [], [], [], [], [], []
    for c in clips:
        s = sub[c] if size != 112 else m[m["clip"] == c].sort_values(["frame", "det_id"])
        if len(s) == 0:
            continue
        ids = s.det_id.to_numpy()
        if size == 112:
            emb, q = S.adaface_emb(c, ids), mag_of(c, ids)
        else:
            z = np.load(ROOT / f"outputs/video/bench7/emb/{c}.npz")
            assert np.array_equal(z["det_id"], ids)
            emb, q = z[f"emb_{size}"], z[f"mag_{size}"]
        P.append(s.pid.to_numpy()); C += [c] * len(s); F.append(s.frame.to_numpy()); D.append(ids); E.append(emb); Q.append(q)
    return dict(pid=np.concatenate(P), clip=np.array(C), frame=np.concatenate(F), det=np.concatenate(D), emb=ev.l2n(np.concatenate(E).astype(np.float32)), q=np.concatenate(Q).astype(np.float64))


def gallery(gp, ge):
    ids = np.array(sorted(set(np.unique(gp))))
    T = ev.l2n(np.stack([ge[gp == i].mean(0) for i in ids]))
    return ids, T


def gt_tracks(fr):
    """list of index arrays, one per (clip, pid)."""
    return [np.asarray(v) for v in pd.DataFrame({"c": fr["clip"], "p": fr["pid"]}).groupby(["c", "p"]).indices.values()]


def make_template(rule, emb, q, idx, T=None):
    """-> template [512] for embedding rules; for R5 the score vector [T]; for R6 the winning template index."""
    if rule == "R5":
        return A.score_mean(emb, T, idx)
    if rule == "R6":
        return A.vote(emb, T, idx)
    if rule.startswith("R4_lam"):
        return A.r4_np(emb, q, idx, lam=float(rule[6:]))
    return A.RULES[rule](emb, q, idx)


def winner(rule, out, T, ids):
    if rule == "R5":
        return ids[int(np.argmax(out))]
    if rule == "R6":
        return ids[out]
    return ids[int(np.argmax(T @ out))]


def track_records(fr, tracks, ids, T, rules):
    """One row per track: pid, n, q_mean, single (mean frame correctness), ok_<rule>. Only tracks whose identity is enrolled (closed set)."""
    sims = fr["emb"] @ T.T
    frame_ok = ids[sims.argmax(1)] == fr["pid"]
    rows = []
    for ix in tracks:
        pid = fr["pid"][ix[0]]
        if pid not in set(ids.tolist()):
            continue
        emb, q, idx = fr["emb"][ix], fr["q"][ix], fr["frame"][ix]
        r = dict(clip=fr["clip"][ix[0]], pid=int(pid), n=len(ix), q_mean=float(q.mean()), single=float(frame_ok[ix].mean()))
        for rule in rules:
            r[f"ok_{rule}"] = float(winner(rule, make_template(rule, emb, q, idx, T), T, ids) == pid)
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ statistics
def draws_mean(values, pids, rng, n=N_BOOT):
    """Cluster bootstrap over identities of the track-weighted mean of `values`."""
    u = np.unique(pids)
    s = np.array([values[pids == i].sum() for i in u])
    c = np.array([(pids == i).sum() for i in u], float)
    idx = rng.integers(0, len(u), (n, len(u)))
    return s[idx].sum(1) / c[idx].sum(1)


def ci(d, lo=0.025):
    return float(np.quantile(d, lo)), float(np.quantile(d, 1 - lo))


def evaluate(fr, tracks, gal, left_out, rules, open_rules, cond, size, direction_name, dir_k):
    ids, T = gallery(*gal)
    rec = track_records(fr, tracks, ids, T, rules)
    pids = rec.pid.to_numpy()
    out = {}
    for rule in rules:
        delta = rec[f"ok_{rule}"].to_numpy() - rec.single.to_numpy()
        rng = np.random.default_rng([SEED, dir_k])
        d_delta = draws_mean(delta, pids, rng)
        rng = np.random.default_rng([SEED, dir_k])
        d_agg = draws_mean(rec[f"ok_{rule}"].to_numpy(), pids, rng)
        out[rule] = dict(delta=float(delta.mean()), agg=float(rec[f"ok_{rule}"].mean()), draws_delta=d_delta, draws_agg=d_agg)
        if "ok_R1" in rec:  # paired difference against the primary rule R1 (pre-registered: Bonferroni over the nine comparisons with R1)
            dv = rec[f"ok_{rule}"].to_numpy() - rec["ok_R1"].to_numpy()
            out[rule].update(vs1=float(dv.mean()), draws_vs1=draws_mean(dv, pids, np.random.default_rng([SEED, dir_k])))
    rng = np.random.default_rng([SEED, dir_k])
    out["single"] = dict(agg=float(rec.single.mean()), draws_agg=draws_mean(rec.single.to_numpy(), pids, rng))
    return rec, out


def open_set(fr, tracks, gal, left_out, rule, n_boot):
    """TPIR20 at FPIR 0.3 / 0.1 / 0.01 and Rank-1 (frame-level for R0, track templates for a rule), through the Phase 6 functions."""
    if rule == "R0":
        probe = (fr["pid"], fr["emb"])
    else:
        P, E = [], []
        for ix in tracks:
            P.append(fr["pid"][ix[0]]); E.append(make_template(rule, fr["emb"][ix], fr["q"][ix], fr["frame"][ix]))
        probe = (np.array(P), ev.l2n(np.stack(E).astype(np.float32)))
    closed, opn, _ = B7.direction(gal, probe, left_out)
    return closed, opn


def run_chokepoint(a):
    cfg, allc, m, lo = B7.setup()
    sub = {c: m[m["clip"] == c].sort_values(["frame", "det_id"]).iloc[::5] for c in allc}
    rules = RULE_NAMES + [f"R4_lam{x}" for x in LAM_SENS]
    rows, check3 = [], []
    box = {}
    for cond in ("SC", "MC"):
        for size in SIZES:
            per_dir = {}
            for k, (g, p) in enumerate((("G1", "G2"), ("G2", "G1"))):
                gal = S.build("adaface", S.clips_of(g, cond, allc), m, cfg, box)
                fr = frames_of(S.clips_of(p, cond, allc), m, size, sub)
                tracks = gt_tracks(fr)
                rec, res = evaluate(fr, tracks, gal, lo, rules, OPEN_RULES, cond, size, f"{g}->{p}", k)
                # open set: single-frame reference (R0, frames) and R1 on tracks
                o = {}
                for rule in ["R0"] + OPEN_RULES:
                    c_, o_ = open_set(fr, tracks, gal, lo, rule, a.n_boot)
                    rng = np.random.default_rng(S.SEED)
                    d = np.array([x for x in (B7.boot([(c_, o_)], rng) for _ in range(a.n_boot)) if x is not None])
                    o[rule] = dict(point=B7.metrics(c_, o_), draws=d)
                per_dir[k] = (rec, res, o, f"{g}->{p}")
                if size == 112 or True:  # check 3 inputs: frame-level R0 point values
                    check3.append(dict(cond=cond, size=size, direction=f"{g}->{p}", point=o["R0"]["point"]))
            rows += summarize(cond, size, per_dir, rules)
            print(cond, size, "done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "chokepoint.csv", index=False, float_format="%.5f")
    check_3(check3)
    print(pd.DataFrame(rows).query("rule in ['R1'] and kind == 'closed'")[["cond", "size", "n_tracks", "single_rank1", "agg_rank1", "delta", "delta_ci_lo", "delta_ci_hi"]].round(4).to_string())


def closed_rows(cond, size, per_dir, rules, extra=None):
    """Closed-set rows: per rule the mean of the two directions, CIs from the averaged draws (plain 95% and Bonferroni over 9 comparisons). per_dir[k] = (rec, res, ...)."""
    rows = []
    n_tracks = sum(len(per_dir[k][0]) for k in per_dir)
    for rule in rules:
        dd = [per_dir[k][1][rule] for k in per_dir]
        n = min(len(x["draws_delta"]) for x in dd)
        d_delta = np.mean([x["draws_delta"][:n] for x in dd], axis=0)
        d_agg = np.mean([x["draws_agg"][:n] for x in dd], axis=0)
        single = np.mean([per_dir[k][1]["single"]["agg"] for k in per_dir])
        sd = np.mean([per_dir[k][1]["single"]["draws_agg"][:n] for k in per_dir], axis=0)
        lo, hi = ci(d_delta)
        blo, bhi = ci(d_delta, 0.05 / BONF / 2)
        a_lo, a_hi = ci(d_agg)
        row = dict(kind="closed", cond=cond, size=size, rule=rule, n_tracks=n_tracks, single_rank1=single, single_ci_lo=ci(sd)[0], single_ci_hi=ci(sd)[1], agg_rank1=float(np.mean([x["agg"] for x in dd])),
                   agg_ci_lo=a_lo, agg_ci_hi=a_hi, delta=float(np.mean([x["delta"] for x in dd])), delta_ci_lo=lo, delta_ci_hi=hi, delta_bonf_lo=blo, delta_bonf_hi=bhi,
                   delta_dir1=dd[0]["delta"], delta_dir2=dd[1]["delta"], **(extra or {}))
        if "draws_vs1" in dd[0]:
            d1 = np.mean([x["draws_vs1"][:n] for x in dd], axis=0)
            l1, h1 = ci(d1)
            lb1, hb1 = ci(d1, 0.05 / BONF / 2)
            row.update(vs_R1=float(np.mean([x["vs1"] for x in dd])), vs_R1_ci_lo=l1, vs_R1_ci_hi=h1, vs_R1_bonf_lo=lb1, vs_R1_bonf_hi=hb1)
        rows.append(row)
    return rows


def summarize(cond, size, per_dir, rules):
    rows = closed_rows(cond, size, per_dir, rules)
    n_tracks = sum(len(per_dir[k][0]) for k in per_dir)
    names = ["rank1", "tpir_fpir0.3", "tpir_fpir0.1", "tpir_fpir0.01"]
    for rule in ["R0"] + OPEN_RULES:
        pts = np.mean([per_dir[k][2][rule]["point"] for k in per_dir], axis=0)
        n = min(len(per_dir[k][2][rule]["draws"]) for k in per_dir)
        dr = np.mean([per_dir[k][2][rule]["draws"][:n] for k in per_dir], axis=0)
        r = dict(kind="open", cond=cond, size=size, rule=rule, n_tracks=n_tracks if rule != "R0" else np.nan)
        for i, nm in enumerate(names):
            r[nm] = float(pts[i]); r[nm + "_ci_lo"], r[nm + "_ci_hi"] = ci(dr[:, i])
        rows.append(r)
    return rows


# ------------------------------------------------------------------ predicted tracks
def predicted_tracks(clips, m, size, sub, tracker, native_stats=None):
    """Tracks of one tracker in `clips`. Unit = (clip, track id) over all detections with score >= 0.5. Label = majority person id of its matched detections; a track with fewer
    than 50% matched detections (or none) is counted and not scored. size != 112: only the E-subsample detections exist, so a track keeps those of its detections that are in the
    subsample (the inclusion decision is taken on the native composition). Returns the frame table, the track index lists and the per-track stats."""
    P, C, F, Em, Q, tracks, stats = [], [], [], [], [], [], []
    n = 0
    for c in clips:
        t = pd.read_csv(PIPE / c / "tracks.csv")
        t = t[(t.score >= 0.5) & (t[f"tid_{tracker}"] >= 0)]
        mp = dict(zip(m.loc[m["clip"] == c, "det_id"], m.loc[m["clip"] == c, "pid"]))
        z = np.load(PIPE / c / "embed.npz")
        pos = {int(d): i for i, d in enumerate(z["det_id"])}
        if size != 112:
            zz = np.load(ROOT / f"outputs/video/bench7/emb/{c}.npz")
            dpos = {int(d): i for i, d in enumerate(zz["det_id"])}
        for tid, g in t.groupby(f"tid_{tracker}"):
            dets = g.det_id.to_numpy()
            matched = [mp[int(d)] for d in dets if int(d) in mp]
            frac = len(matched) / len(dets)
            maj, cnt = (pd.Series(matched).value_counts().index[0], int(pd.Series(matched).value_counts().iloc[0])) if matched else (-1, 0)
            used = dets if size == 112 else np.array([d for d in dets if int(d) in dpos])
            stats.append(dict(clip=c, tid=int(tid), n_dets=len(dets), n_matched=len(matched), matched_frac=frac, label=int(maj), purity=cnt / len(matched) if matched else np.nan,
                              scored=bool(frac >= 0.5 and matched and len(used) > 0), n_used=len(used)))
            if not stats[-1]["scored"]:
                continue
            fr_no = g.set_index("det_id").loc[used, "frame"].to_numpy()
            if size == 112:
                emb, q = z["emb_adaface"][[pos[int(d)] for d in used]], z["mag_norm"][[pos[int(d)] for d in used]]
            else:
                emb, q = zz[f"emb_{size}"][[dpos[int(d)] for d in used]], zz[f"mag_{size}"][[dpos[int(d)] for d in used]]
            tracks.append(np.arange(n, n + len(used)))
            n += len(used)
            P.append(np.full(len(used), maj)); C += [c] * len(used); F.append(fr_no); Em.append(emb); Q.append(q)
    fr = dict(pid=np.concatenate(P), clip=np.array(C), frame=np.concatenate(F), emb=ev.l2n(np.concatenate(Em).astype(np.float32)), q=np.concatenate(Q).astype(np.float64))
    return fr, tracks, pd.DataFrame(stats)


def run_predicted(a):
    cfg, allc, m, lo = B7.setup()
    sub = {c: m[m["clip"] == c].sort_values(["frame", "det_id"]).iloc[::5] for c in allc}
    rows, tstats = [], []
    box = {}
    for tracker in ("bytetrack", "botsort"):
        for cond in ("SC", "MC"):
            for size in SIZES:
                per_dir = {}
                for k, (g, p) in enumerate((("G1", "G2"), ("G2", "G1"))):
                    gal = S.build("adaface", S.clips_of(g, cond, allc), m, cfg, box)
                    fr, tracks, st = predicted_tracks(S.clips_of(p, cond, allc), m, size, sub, tracker)
                    rec, res = evaluate(fr, tracks, gal, lo, RULE_NAMES, [], cond, size, f"{g}->{p}", k)
                    per_dir[k] = (rec, res)
                    tstats.append(dict(tracker=tracker, cond=cond, size=size, direction=f"{g}->{p}", n_tracks_total=len(st), n_scored=int(st.scored.sum()), n_excluded=int((~st.scored).sum()),
                                       mean_purity_scored=float(st[st.scored].purity.mean()), median_purity_scored=float(st[st.scored].purity.median()),
                                       share_dets_in_excluded=float(st[~st.scored].n_dets.sum() / st.n_dets.sum())))
                rows += closed_rows(cond, size, per_dir, RULE_NAMES, dict(tracker=tracker))
                print(tracker, cond, size, "done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "predicted.csv", index=False, float_format="%.5f")
    pd.DataFrame(tstats).to_csv(OUT / "predicted_track_stats.csv", index=False, float_format="%.5f")
    print(pd.DataFrame(rows).query("rule == 'R1' and cond == 'SC'")[["tracker", "size", "n_tracks", "single_rank1", "agg_rank1", "delta", "delta_ci_lo", "delta_ci_hi"]].round(4).to_string(index=False))


# ------------------------------------------------------------------ curves
NS = [1, 2, 4, 8, 16, "all"]


def run_curves(a):
    cfg, allc, m, lo = B7.setup()
    sub = {c: m[m["clip"] == c].sort_values(["frame", "det_id"]).iloc[::5] for c in allc}
    rows_f, rows_q = [], []
    box = {}
    for cond in ("SC", "MC"):
        for size in SIZES:
            recs, accs = [], {n: [] for n in NS}
            for k, (g, p) in enumerate((("G1", "G2"), ("G2", "G1"))):
                gal = S.build("adaface", S.clips_of(g, cond, allc), m, cfg, box)
                ids, T = gallery(*gal)
                fr = frames_of(S.clips_of(p, cond, allc), m, size, sub)
                tracks = [ix for ix in gt_tracks(fr) if fr["pid"][ix[0]] in set(ids.tolist())]
                recs.append(track_records(fr, gt_tracks(fr), ids, T, ["R1"]))
                for n in NS:
                    draws = []
                    for r in range(1 if n == "all" else 20):
                        rng = np.random.default_rng([SEED, k, r, 0 if n == "all" else int(n)])
                        ok = []
                        for ix in tracks:
                            sel = ix if n == "all" or len(ix) <= n else rng.choice(ix, int(n), replace=False)
                            t = A.r1_mean(fr["emb"][sel], fr["q"][sel], fr["frame"][sel])
                            ok.append(float(ids[int(np.argmax(T @ t))] == fr["pid"][ix[0]]))
                        draws.append(np.mean(ok))
                    accs[n].append(draws)
            for n in NS:
                d = np.mean([np.array(x) for x in accs[n]], axis=0)  # mean of the two directions, per draw
                rows_f.append(dict(cond=cond, size=size, n_frames=str(n), rule="R1", acc_mean_over_draws=float(d.mean()), acc_min=float(d.min()), acc_max=float(d.max()), n_draws=len(d)))
            rec = pd.concat(recs, ignore_index=True)
            rec["quartile"] = pd.qcut(rec.q_mean, 4, labels=["Q1 (lowest norm)", "Q2", "Q3", "Q4 (highest norm)"])
            for qn, gq in rec.groupby("quartile", observed=True):
                rows_q.append(dict(cond=cond, size=size, quartile=str(qn), n_tracks=len(gq), q_mean=float(gq.q_mean.mean()), single_frame_acc=float(gq.single.mean()), R1_acc=float(gq.ok_R1.mean())))
            print("curves", cond, size, "done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_f).to_csv(OUT / "curves_frames.csv", index=False, float_format="%.5f")
    pd.DataFrame(rows_q).to_csv(OUT / "curves_quality.csv", index=False, float_format="%.5f")
    print(pd.DataFrame(rows_f).query("cond == 'SC'")[["size", "n_frames", "acc_mean_over_draws", "acc_min", "acc_max"]].round(4).to_string(index=False))


def check_3(c3):
    """Acceptance check 3: the single-frame reference reproduces Phase 6 (D0.csv native, E.csv degraded) to the stored precision."""
    d0 = pd.read_csv(ROOT / "outputs/video/bench7/D0.csv")
    e = pd.read_csv(ROOT / "outputs/video/bench7/E.csv")
    res, ok = [], True
    for r in c3:
        if r["size"] == 112:
            ref = d0[(d0.recognizer == "adaface") & (d0.condition == r["cond"]) & (d0.direction == r["direction"])].iloc[0]
        else:
            ref = e[(e.condition == r["cond"]) & (e.probe_side == r["size"]) & (e.direction == r["direction"])].iloc[0]
        want = [ref["rank1"], ref["tpir20_fpir0.3"], ref["tpir20_fpir0.1"], ref["tpir20_fpir0.01"]]
        same = bool(np.allclose(np.round(r["point"], 5), np.round(want, 5), atol=1e-9))
        ok &= same
        res.append(dict(r, reference=[float(x) for x in want], point=[float(x) for x in r["point"]], equal_to_5_decimals=same))
    allres = json.loads((OUT / "checks.json").read_text()) if (OUT / "checks.json").exists() else {}
    allres["check3_single_frame_reference"] = dict(all_equal=bool(ok), rows=res)
    (OUT / "checks.json").write_text(json.dumps(allres, indent=1))
    print("check 3 (single-frame reference equals Phase 6):", "PASS" if ok else "FAIL")


# ------------------------------------------------------------------ YouTube Faces (REPORT_phase7.md section 0.5)
YTF = ROOT / "outputs/video/ytf"
YTF_RULES = ["R1", "R2", "R3_k1", "R3_k3", "R3_k5", "R3_k10", "R4", "R4*", "R5", "R4_lam10", "R4_lam100"]


def ytf_videos():
    """{video_idx: (emb[K,512] L2-normalised, mag_norm[K], frame_no[K])} for the videos with at least one accepted frame."""
    out = {}
    for f in (YTF / "embed").glob("*.npz"):
        z = np.load(f)
        if len(z["frame_no"]):
            out[int(f.stem)] = (z["emb_adaface"].astype(np.float64), z["mag_norm"].astype(np.float64), z["frame_no"])
    return out


def cv_threshold_acc(s, same, fold):
    """Verification accuracy with the threshold chosen on the 9 other folds (maximising their accuracy); returns per-pair correctness and the 10 fold accuracies."""
    ok = np.zeros(len(s), bool)
    accs = []
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        t = np.sort(s[tr])
        cand = np.r_[t[0] - 1e-9, (t[1:] + t[:-1]) / 2, t[-1] + 1e-9]
        st, ys = np.sort(s[tr]), same[tr][np.argsort(s[tr], kind="stable")]
        neg_below = np.r_[0, np.cumsum(~ys)]  # negatives with score < candidate
        pos_above = ys.sum() - np.r_[0, np.cumsum(ys)]  # positives with score >= candidate
        best = cand[int(np.argmax(neg_below + pos_above))]
        ok[te] = (s[te] >= best) == same[te]
        accs.append(ok[te].mean())
    return ok, np.array(accs)


def auc(s, same):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = same.sum(), (~same).sum()
    return float((r[same].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tar_at_far(s, same, far):
    thr = np.quantile(s[~same], 1 - far)
    return float((s[same] >= thr).mean())


def ytf_report(scores, single, same, fold, person, methods, prefix):
    """Verification metrics of `scores[method]` (arrays over the pairs) and of the single-frame draws `single` (list of arrays); paired differences against the single-frame reference
    with a bootstrap over persons (the person of the first video). Writes OUT/<prefix>_verification.csv and <prefix>_paired.csv; returns (rows, paired rows)."""
    rows, ok_by = [], {}
    for rule in methods:
        ok, accs = cv_threshold_acc(scores[rule], same, fold)
        ok_by[rule] = ok
        rows.append(dict(method=rule, accuracy=float(accs.mean()), accuracy_se=float(accs.std(ddof=1) / np.sqrt(len(accs))), auc=auc(scores[rule], same), tar_far0_01=tar_at_far(scores[rule], same, 0.01),
                         tar_far0_1=tar_at_far(scores[rule], same, 0.1), n_pairs=len(same)))
    sm, ok_single = [], []
    for sc in single:
        ok, accs = cv_threshold_acc(sc, same, fold)
        ok_single.append(ok)
        sm.append(dict(accuracy=float(accs.mean()), se=float(accs.std(ddof=1) / np.sqrt(len(accs))), auc=auc(sc, same), t1=tar_at_far(sc, same, 0.01), t2=tar_at_far(sc, same, 0.1)))
    sm = pd.DataFrame(sm)
    rows.insert(0, dict(method="single frame (mean of 20 draws)", accuracy=float(sm.accuracy.mean()), accuracy_se=float(sm.se.mean()), auc=float(sm.auc.mean()), tar_far0_01=float(sm.t1.mean()),
                        tar_far0_1=float(sm.t2.mean()), n_pairs=len(same), accuracy_min_draw=float(sm.accuracy.min()), accuracy_max_draw=float(sm.accuracy.max())))
    pd.DataFrame(rows).to_csv(OUT / f"{prefix}_verification.csv", index=False, float_format="%.5f")
    single_ok = np.mean(ok_single, axis=0)
    prow = []
    for rule in methods:
        delta = ok_by[rule].astype(float) - single_ok
        d = draws_mean(delta, person, np.random.default_rng([SEED, 99]))
        lo, hi = ci(d)
        blo, bhi = ci(d, 0.05 / BONF / 2)
        row = dict(method=rule, delta_accuracy=float(delta.mean()), ci_lo=lo, ci_hi=hi, bonferroni_lo=blo, bonferroni_hi=bhi, above_zero=bool(lo > 0))
        ref = next((m for m in methods if m.startswith("R1")), None)
        if ref:  # paired difference against the primary rule R1 (same persons bootstrap)
            dv = ok_by[rule].astype(float) - ok_by[ref].astype(float)
            dd = draws_mean(dv, person, np.random.default_rng([SEED, 99]))
            vl, vh = ci(dd)
            vbl, vbh = ci(dd, 0.05 / BONF / 2)
            row.update(minus_R1=float(dv.mean()), minus_R1_ci_lo=vl, minus_R1_ci_hi=vh, minus_R1_bonf_lo=vbl, minus_R1_bonf_hi=vbh)
        prow.append(row)
    pd.DataFrame(prow).to_csv(OUT / f"{prefix}_paired.csv", index=False, float_format="%.5f")
    return rows, prow


def run_ytf(a):
    import yaml
    pairs = pd.read_csv(YTF / "pairs.csv")
    frames = pd.read_csv(YTF / "frames.csv", usecols=["video_idx", "label"]).drop_duplicates("video_idx").set_index("video_idx").label
    vids = ytf_videos()
    have = pairs.v1.isin(vids) & pairs.v2.isin(vids)
    p = pairs[have].reset_index(drop=True)
    same, fold = p.same.to_numpy().astype(bool), p.fold.to_numpy()
    print(f"pairs scored {len(p)} of {len(pairs)}; failed {int((~have).sum())}", flush=True)

    def templ(rule, v):
        e, q, fn = vids[v]
        if rule == "R5":
            return e.mean(0)  # pair score = mean pairwise cosine = dot of the two (unnormalised) mean embeddings
        return make_template(rule, e, q, fn)

    scores = {}
    for rule in YTF_RULES:
        cache = {v: templ(rule, v) for v in set(p.v1) | set(p.v2)}
        a1, a2 = np.stack([cache[v] for v in p.v1]), np.stack([cache[v] for v in p.v2])
        scores[rule] = (a1 * a2).sum(1) if rule == "R5" else (a1 * a2).sum(1) / (np.linalg.norm(a1, axis=1) * np.linalg.norm(a2, axis=1))
    # single-frame reference: one random frame per video, 20 seeded draws
    single = []
    for r in range(20):
        rng = np.random.default_rng([SEED, r])
        pick = {v: vids[v][0][int(rng.integers(0, len(vids[v][0])))] for v in vids}
        single.append(np.array([pick[x] @ pick[y] for x, y in zip(p.v1, p.v2)]))
    rows, prow = ytf_report(scores, single, same, fold, frames.reindex(p.v1).to_numpy(), YTF_RULES, "ytf")
    summ = pd.read_csv(YTF / "embed/summary.csv") if (YTF / "embed/summary.csv").exists() else None
    cnt = json.loads((YTF / "counts.json").read_text())
    c4 = dict(official_counts=dict(videos=3425, people=1595, pairs=5000, folds=10), found=cnt, pairs_scored=len(p), pairs_failed=int((~have).sum()),
              videos_with_embeddings=len(vids), frames_selected=int(summ.n_selected.sum()) if summ is not None else None, frames_kept=int(summ.n_kept.sum()) if summ is not None else None,
              frames_no_detection=int(summ.n_no_detection.sum()) if summ is not None else None, frames_no_match=int(summ.n_no_match.sum()) if summ is not None else None,
              frames_unreadable=int(summ.n_unreadable.sum()) if summ is not None else None)
    allres = json.loads((OUT / "checks.json").read_text()) if (OUT / "checks.json").exists() else {}
    allres["check4_ytf_counts"] = c4
    (OUT / "checks.json").write_text(json.dumps(allres, indent=1))
    print(pd.DataFrame(rows)[["method", "accuracy", "accuracy_se", "auc", "tar_far0_01", "tar_far0_1"]].round(4).to_string(index=False))
    print(pd.DataFrame(prow).round(4).to_string(index=False))


# ------------------------------------------------------------------ CAFace arm (features from scripts/video_caface.py, envs/caface)
CAF = ROOT / "outputs/video/caface"


def caface_frames(clips, size):
    """Frame table from the CAFace cache (matched detections; size 112 = native, else the E subsample), plus the fused feature of every GT track {(clip, pid): fused}."""
    P, C, F, Em, Q, fused = [], [], [], [], [], {}
    for c in clips:
        z = np.load(CAF / "emb" / f"{c}.npz")
        tag = str(size)
        sel = np.arange(len(z["pid"])) if size == 112 else z["sub_idx"]
        P.append(z["pid"][sel]); C += [c] * len(sel); F.append(z["frame"][sel]); Em.append(z[f"feat{tag}"]); Q.append(z[f"norm{tag}"])
        for pid, fv in zip(z[f"track_pids{tag}"], z[f"fused{tag}"]):
            fused[(c, int(pid))] = fv
    return dict(pid=np.concatenate(P), clip=np.array(C), frame=np.concatenate(F), emb=ev.l2n(np.concatenate(Em).astype(np.float32)), q=np.concatenate(Q).astype(np.float64)), fused


def run_caface_chokepoint(a):
    cfg, allc, m, lo = B7.setup()
    rows = []
    for cond in ("SC", "MC"):
        for size in SIZES:
            per_dir = {}
            for k, (g, p) in enumerate((("G1", "G2"), ("G2", "G1"))):
                gfr, _ = caface_frames(S.clips_of(g, cond, allc), 112)
                ids, T = gallery(gfr["pid"], gfr["emb"])
                fr, fused = caface_frames(S.clips_of(p, cond, allc), size)
                tracks = gt_tracks(fr)
                rec = track_records(fr, tracks, ids, T, ["R1"])
                rec["ok_CAFACE"] = [float(ids[int(np.argmax(T @ fused[(c, int(pid))]))] == pid) for c, pid in zip(rec["clip"], rec.pid)]
                pids = rec.pid.to_numpy()
                res = {}
                for rule in ("R1", "CAFACE"):
                    delta = rec[f"ok_{rule}"].to_numpy() - rec.single.to_numpy()
                    res[rule] = dict(delta=float(delta.mean()), agg=float(rec[f"ok_{rule}"].mean()), draws_delta=draws_mean(delta, pids, np.random.default_rng([SEED, k])),
                                     draws_agg=draws_mean(rec[f"ok_{rule}"].to_numpy(), pids, np.random.default_rng([SEED, k])))
                res["single"] = dict(agg=float(rec.single.mean()), draws_agg=draws_mean(rec.single.to_numpy(), pids, np.random.default_rng([SEED, k])))
                for rule in ("R1", "CAFACE"):
                    dv = rec[f"ok_{rule}"].to_numpy() - rec["ok_R1"].to_numpy()
                    res[rule].update(vs1=float(dv.mean()), draws_vs1=draws_mean(dv, pids, np.random.default_rng([SEED, k])))
                per_dir[k] = (rec, res)
            rows += closed_rows(cond, size, per_dir, ["R1", "CAFACE"], dict(arm="CAFace backbone (AdaFace R100 WebFace4M)"))
            print("caface", cond, size, "done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "caface_chokepoint.csv", index=False, float_format="%.5f")
    print(pd.DataFrame(rows)[["cond", "size", "rule", "n_tracks", "single_rank1", "agg_rank1", "delta", "delta_ci_lo", "delta_ci_hi"]].round(4).to_string(index=False))


def run_caface_ytf(a):
    pairs = pd.read_csv(YTF / "pairs.csv")
    frames = pd.read_csv(YTF / "frames.csv", usecols=["video_idx", "label"]).drop_duplicates("video_idx").set_index("video_idx").label
    vids = {int(f.stem): np.load(f) for f in (CAF / "ytf").glob("*.npz")}
    vids = {v: (z["feat"].astype(np.float64), z["fused"].astype(np.float64)) for v, z in vids.items()}
    have = pairs.v1.isin(vids) & pairs.v2.isin(vids)
    p = pairs[have].reset_index(drop=True)
    same, fold = p.same.to_numpy().astype(bool), p.fold.to_numpy()
    print(f"pairs scored {len(p)} of {len(pairs)}", flush=True)

    def cosv(x, y):
        return (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

    mean_t = {v: A.l2n(vids[v][0].mean(0)) for v in vids}
    scores = {"R1 (plain mean of CAFace-backbone features)": cosv(np.stack([mean_t[v] for v in p.v1]), np.stack([mean_t[v] for v in p.v2])),
              "CAFace aggregator": cosv(np.stack([vids[v][1] for v in p.v1]), np.stack([vids[v][1] for v in p.v2]))}
    single = []
    for r in range(20):
        rng = np.random.default_rng([SEED, r])
        pick = {v: vids[v][0][int(rng.integers(0, len(vids[v][0])))] for v in vids}
        single.append(np.array([pick[x] @ pick[y] / (np.linalg.norm(pick[x]) * np.linalg.norm(pick[y])) for x, y in zip(p.v1, p.v2)]))
    rows, prow = ytf_report(scores, single, same, fold, frames.reindex(p.v1).to_numpy(), list(scores), "ytf_caface")
    print(pd.DataFrame(rows)[["method", "accuracy", "accuracy_se", "auc", "tar_far0_01", "tar_far0_1"]].round(4).to_string(index=False))
    print(pd.DataFrame(prow).round(4).to_string(index=False))


# ------------------------------------------------------------------ backbone comparison (exploratory addendum, REPORT_phase7.md section 0.10)
def run_backbone(a):
    """Same crops, same tracks, same gallery protocol; AdaFace R100 trained on MS1MV2 (A) against AdaFace R100 trained on WebFace4M (B, the CAFace backbone).
    Paired per track (ChokePoint, GT tracks) and per pair (YouTube Faces); cluster bootstrap over identities / persons."""
    cfg, allc, m, lo = B7.setup()
    sub = {c: m[m["clip"] == c].sort_values(["frame", "det_id"]).iloc[::5] for c in allc}
    rows, box = [], {}
    for cond in ("SC", "MC"):
        for size in SIZES:
            dd = {"single": [], "R1": [], "gain": []}
            for k, (g, p_) in enumerate((("G1", "G2"), ("G2", "G1"))):
                gal = S.build("adaface", S.clips_of(g, cond, allc), m, cfg, box)
                idsA, TA = gallery(*gal)
                frA = frames_of(S.clips_of(p_, cond, allc), m, size, sub)
                gfr, _ = caface_frames(S.clips_of(g, cond, allc), 112)
                idsB, TB = gallery(gfr["pid"], gfr["emb"])
                frB, _ = caface_frames(S.clips_of(p_, cond, allc), size)
                recA = track_records(frA, gt_tracks(frA), idsA, TA, ["R1"]).set_index(["clip", "pid"])
                recB = track_records(frB, gt_tracks(frB), idsB, TB, ["R1"]).set_index(["clip", "pid"])
                common = recA.index.intersection(recB.index)
                assert len(common) == len(recA) == len(recB), (len(common), len(recA), len(recB))
                recA, recB = recA.loc[common], recB.loc[common]
                pid = recA.index.get_level_values("pid").to_numpy()
                dA, dB = recA.ok_R1 - recA.single, recB.ok_R1 - recB.single
                for name, x in (("single", recB.single - recA.single), ("R1", recB.ok_R1 - recA.ok_R1), ("gain", dB - dA)):
                    dd[name].append((float(x.mean()), draws_mean(x.to_numpy(), pid, np.random.default_rng([SEED, k]))))
                dd.setdefault("meanA", []).append((float(recA.single.mean()), float(recA.ok_R1.mean())))
                dd.setdefault("meanB", []).append((float(recB.single.mean()), float(recB.ok_R1.mean())))
            row = dict(cond=cond, size=size, n_tracks=len(common) * 2, single_A=np.mean([x[0] for x in dd["meanA"]]), R1_A=np.mean([x[1] for x in dd["meanA"]]),
                       single_B=np.mean([x[0] for x in dd["meanB"]]), R1_B=np.mean([x[1] for x in dd["meanB"]]))
            for name in ("single", "R1", "gain"):
                n = min(len(x[1]) for x in dd[name])
                dr = np.mean([x[1][:n] for x in dd[name]], axis=0)
                lo_, hi_ = ci(dr)
                row.update({f"B_minus_A_{name}": float(np.mean([x[0] for x in dd[name]])), f"B_minus_A_{name}_ci_lo": lo_, f"B_minus_A_{name}_ci_hi": hi_})
            rows.append(row)
            print("backbone", cond, size, "done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "backbone_chokepoint.csv", index=False, float_format="%.5f")
    # YouTube Faces
    pairs = pd.read_csv(YTF / "pairs.csv")
    lab = pd.read_csv(YTF / "frames.csv", usecols=["video_idx", "label"]).drop_duplicates("video_idx").set_index("video_idx").label
    vA = ytf_videos()
    vB = {int(f.stem): np.load(f)["feat"].astype(np.float64) for f in (CAF / "ytf").glob("*.npz")}
    have = pairs.v1.isin(vA) & pairs.v2.isin(vA) & pairs.v1.isin(vB) & pairs.v2.isin(vB)
    pp = pairs[have].reset_index(drop=True)
    same, fold = pp.same.to_numpy().astype(bool), pp.fold.to_numpy()
    person = lab.reindex(pp.v1).to_numpy()

    def ok_arrays(feats):
        mt = {v: A.l2n(feats[v].mean(0)) for v in set(pp.v1) | set(pp.v2)}
        r1 = np.array([mt[x] @ mt[y] for x, y in zip(pp.v1, pp.v2)])
        oks = []
        for r in range(20):
            rng = np.random.default_rng([SEED, r])
            pick = {v: A.l2n(feats[v][int(rng.integers(0, len(feats[v])))]) for v in sorted(feats)}
            oks.append(cv_threshold_acc(np.array([pick[x] @ pick[y] for x, y in zip(pp.v1, pp.v2)]), same, fold)[0])
        return cv_threshold_acc(r1, same, fold)[0].astype(float), np.mean(oks, axis=0)

    okA1, okAs = ok_arrays({v: vA[v][0] for v in vA})
    okB1, okBs = ok_arrays(vB)
    yrow = dict(pairs=len(pp), single_A=float(okAs.mean()), R1_A=float(okA1.mean()), single_B=float(okBs.mean()), R1_B=float(okB1.mean()))
    for name, x in (("single", okBs - okAs), ("R1", okB1 - okA1), ("gain", (okB1 - okBs) - (okA1 - okAs))):
        d = draws_mean(x, person, np.random.default_rng([SEED, 99]))
        lo_, hi_ = ci(d)
        yrow.update({f"B_minus_A_{name}": float(x.mean()), f"B_minus_A_{name}_ci_lo": lo_, f"B_minus_A_{name}_ci_hi": hi_})
    pd.DataFrame([yrow]).to_csv(OUT / "backbone_ytf.csv", index=False, float_format="%.5f")
    print(pd.DataFrame(rows)[["cond", "size", "single_A", "single_B", "R1_A", "R1_B", "B_minus_A_single", "B_minus_A_single_ci_lo", "B_minus_A_single_ci_hi", "B_minus_A_gain", "B_minus_A_gain_ci_lo", "B_minus_A_gain_ci_hi"]].round(4).to_string(index=False))
    print(pd.DataFrame([yrow]).round(4).to_string(index=False))


def run_thin(a):
    """Secondary condition: B k = 1 (one random enrolment frame per identity, 20 draws) with R0 (single frame) and R1 (aggregated) probes, GT tracks."""
    cfg, allc, m, lo = B7.setup()
    sub = {c: m[m["clip"] == c].sort_values(["frame", "det_id"]).iloc[::5] for c in allc}
    rows, box = [], {}
    for cond in ("SC", "MC"):
        for size in SIZES:
            per_dir = {}
            for k, (g, p) in enumerate((("G1", "G2"), ("G2", "G1"))):
                gal = S.build("adaface", S.clips_of(g, cond, allc), m, cfg, box)
                fr = frames_of(S.clips_of(p, cond, allc), m, size, sub)
                tracks = gt_tracks(fr)
                recs = []
                for r in range(B7.R):
                    ids, T = gallery(*B7.thin(gal, 1, r))
                    recs.append(track_records(fr, tracks, ids, T, ["R1"]))
                rec = recs[0].copy()
                for col in ("single", "ok_R1"):
                    rec[col] = np.mean([x[col].to_numpy() for x in recs], axis=0)  # per-track mean over the 20 gallery draws
                pids = rec.pid.to_numpy()
                res = {}
                delta = rec.ok_R1.to_numpy() - rec.single.to_numpy()
                res["R1"] = dict(delta=float(delta.mean()), agg=float(rec.ok_R1.mean()), draws_delta=draws_mean(delta, pids, np.random.default_rng([SEED, k])),
                                 draws_agg=draws_mean(rec.ok_R1.to_numpy(), pids, np.random.default_rng([SEED, k])))
                res["single"] = dict(agg=float(rec.single.mean()), draws_agg=draws_mean(rec.single.to_numpy(), pids, np.random.default_rng([SEED, k])))
                per_dir[k] = (rec, res)
            rows += closed_rows(cond, size, per_dir, ["R1"], dict(gallery="k=1 (20 draws)"))
            print("thin", cond, size, "done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "chokepoint_thin_gallery.csv", index=False, float_format="%.5f")
    print(pd.DataFrame(rows)[["cond", "size", "n_tracks", "single_rank1", "agg_rank1", "delta", "delta_ci_lo", "delta_ci_hi"]].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["chokepoint", "predicted", "curves", "thin", "ytf", "caface", "caface-ytf", "backbone"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    {"chokepoint": run_chokepoint, "predicted": run_predicted, "curves": run_curves, "thin": run_thin, "ytf": run_ytf, "caface": run_caface_chokepoint, "caface-ytf": run_caface_ytf, "backbone": run_backbone}[a.part](a)
