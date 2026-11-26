"""Phase 9 Part B: error investigation and reduction on MEVID (protocol: REPORT_phase9.md sections 0.2-0.5, committed before any number). CPU, from cache.

    envs/video/Scripts/python.exe scripts/mevid_errors.py --stage dev            # B1 (default rule), B2, F1-F5 on the development split (train persons), B5 on development, audit sheet
    envs/video/Scripts/python.exe scripts/mevid_errors.py --stage combine        # the adopted fixes together, on development -> outputs/video/mevid/phase9/frozen_config.json (for section 0b)
    envs/video/Scripts/python.exe scripts/mevid_errors.py --stage confirm --confirm   # ONE look at the confirmation split (test persons): H1, H2, coverage/error at 23 px, curves, B5
    envs/video/Scripts/python.exe scripts/mevid_errors.py --selftest

Features: WebFace4M re-embedded in envs/video (scripts/phase9_embed.py: emb/wf_<tracklet>.npz feat / feat_flip, gallery_wf.npz native / h24 / h32), MS1MV2 and the MagFace norm from
the Phase 8 cache (emb/emb_<tracklet>.npz, gallery.npz), keypoints and detector scores from emb/det_<tracklet>.npz. Tracklets: the 3,188 Phase 8 tracklets (eligible, person enrolled at the
48 px gallery bar); gallery: all 170 persons enrolled at 48 px. Bootstrap: clusters = persons, 1,000 draws, seed 42; paired (same person resamples for both arms).
Outputs: outputs/video/mevid/phase9/*.csv / *.json (tracked; no image, no embedding).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.video import aggregate as A  # noqa: E402
from mevid_quality import GRID, curve, reject_table  # noqa: E402

M = ROOT / "outputs/video/mevid"
P9 = M / "phase9"
FPS_COND = ROOT / "outputs/video/phase9/fps_condition.json"  # written by scripts/phase9_fps.py (Part D)
SEED, N_BOOT, N_FOLDS, FPIR = 42, 1000, 5, 0.1
GAL_BAR, BAR_PX, MIN_FACE_H = 48, 23.0, 16.0
YAW_MAX, SCORE_MIN, KEEP_MIN = 0.35, 0.6, 3          # F1
N_COHORT_TOP = 10                                     # F5
FIXES = ["F1_frame_filter", "F2_resmatched_gallery", "F3_flip", "F4_fusion"]
HBINS = [(16, 20), (20, 24), (24, 32), (32, 48), (48, 10 ** 6)]
PX_GRID = np.arange(16, 49)


def l2n(x):
    x = np.asarray(x, np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


# ----------------------------------------------------------------------------- data
def yaw_proxy(kps):
    """|x_nose - (x_le + x_re)/2| / |x_re - x_le| per face (SCRFD order: left eye, right eye, nose, left mouth, right mouth)."""
    le, re, no = kps[:, 0, 0], kps[:, 1, 0], kps[:, 2, 0]
    return np.abs(no - (le + re) / 2) / np.maximum(np.abs(re - le), 1e-6)


def load():
    tr = pd.read_csv(M / "tracklets.csv")
    tr = tr[tr.detected.fillna(False).astype(bool) & tr.eligible.fillna(False).astype(bool)].reset_index(drop=True)
    gw = np.load(M / "gallery_wf.npz", allow_pickle=True)
    ga = np.load(M / "gallery.npz", allow_pickle=True)
    assert np.array_equal(gw["photo"], ga["photo"])
    gal = pd.DataFrame(dict(person=ga["person"], face_h=ga["face_h"]))
    ok = gal.face_h.to_numpy() >= GAL_BAR
    ids = np.array(sorted(gal.person[ok].unique()))
    tr = tr[tr.person.isin(set(ids.tolist()))].reset_index(drop=True)
    frames = {}
    for r in tr.itertuples():
        z, e, w = (np.load(M / "emb" / f"{p}_{r.tracklet}.npz") for p in ("det", "emb", "wf"))
        assert np.array_equal(e["sel"], w["sel"])
        sel = e["sel"]
        frames[r.tracklet] = dict(wf=l2n(w["feat"]), wf_flip=l2n(w["feat_flip"]), ms1=l2n(e["emb_adaface"]), q=np.asarray(e["mag_norm"], np.float64), idx=e["chip_idx"], h=e["face_h"].astype(np.float64),
                                  score=z["box"][sel, 4].astype(np.float64), yaw=yaw_proxy(z["kps"][sel]).astype(np.float64), lap=e["lap_var"].astype(np.float64))
    best_gal_h = gal[ok].groupby("person").face_h.max()
    tr["best_gallery_h"] = tr.person.map(best_gal_h)
    tr["role"] = np.where(tr.split == "train", "dev", "conf")
    g = dict(ids=ids, ok=ok, person=gal.person.to_numpy(), **{k: gw[k] for k in gw.files if k.startswith("feat")}, ms1=ga["emb_adaface"])
    return tr, frames, g


def templates(g, key):
    """Gallery templates [P,512] of the 170 persons: normalised mean of the usable (>= 48 px) faces. key: a gallery_wf feature name, 'ms1', or 'resmatch[_flip]' (F2: native + h24 + h32)."""
    if key.startswith("resmatch"):
        f = "_flip" if key.endswith("flip") else ""
        feats = [l2n(g[f"feat{f}_{t}"]) for t in ("native", "h24", "h32")]
    else:
        feats = [l2n(g[key])]
    T = []
    for p in g["ids"]:
        m = g["ok"] & (g["person"] == p)
        T.append(l2n(np.concatenate([x[m] for x in feats]).mean(0)))
    return np.stack(T)


# ----------------------------------------------------------------------------- configurations
BASE = dict(backbone="wf", flip=False, rule="R4*", filt=False, gallery="native", fusion=False)


def frame_sel(f, cfg):
    if not cfg["filt"]:
        return np.arange(len(f["q"]))
    keep = np.flatnonzero((f["yaw"] <= YAW_MAX) & (f["score"] >= SCORE_MIN))
    return keep if len(keep) >= KEEP_MIN else np.arange(len(f["q"]))


def sims(tr, frames, g, cfg, cache):
    """-> S [n_tracklets, P] cosine of each tracklet template (config cfg) to each gallery template; single [n] expected single-frame Rank-1 (backbone side only)."""
    def side(bk):
        fkey = ("wf_flip" if cfg["flip"] else "wf") if bk == "wf" else "ms1"
        gkey = ("resmatch_flip" if cfg["flip"] else "resmatch") if (bk == "wf" and cfg["gallery"] == "resmatch") else (("feat_flip_native" if cfg["flip"] else "feat_native") if bk == "wf" else "ms1")
        if gkey not in cache:
            cache[gkey] = templates(g, gkey)
        T = cache[gkey]
        S, single = [], []
        for r in tr.itertuples():
            f = frames[r.tracklet]
            k = frame_sel(f, cfg)
            e = f[fkey][k]
            if cfg["rule"] == "single":
                t = A.r1_mean(e, f["q"][k], f["idx"][k])
            else:
                t = A.RULES[cfg["rule"]](e, f["q"][k], f["idx"][k])
            S.append(T @ t)
            single.append(float((g["ids"][(f[fkey] @ T.T).argmax(1)] == r.person).mean()))
        return np.array(S), np.array(single)
    S, single = side(cfg["backbone"])
    if cfg["fusion"]:
        S2, _ = side("ms1")
        S = 0.5 * S + 0.5 * S2
    return S, single


def correct(tr, S, g):
    return (g["ids"][S.argmax(1)] == tr.person.to_numpy()).astype(float)


# ----------------------------------------------------------------------------- statistics
def person_groups(pids):
    u = np.unique(pids)
    return u, [np.flatnonzero(pids == p) for p in u]


def boot_mean_diff(a, b, pids, n=N_BOOT):
    """Paired cluster bootstrap over persons of mean(a) - mean(b) (tracklet-weighted); -> (diff, draws)."""
    u, grp = person_groups(pids)
    sa = np.array([a[i].sum() for i in grp])
    sb = np.array([b[i].sum() for i in grp])
    c = np.array([len(i) for i in grp], float)
    idx = np.random.default_rng(SEED).integers(0, len(u), (n, len(u)))
    return float(a.mean() - b.mean()), (sa[idx].sum(1) - sb[idx].sum(1)) / c[idx].sum(1)


def boot_mean(a, pids, n=N_BOOT):
    u, grp = person_groups(pids)
    s = np.array([a[i].sum() for i in grp])
    c = np.array([len(i) for i in grp], float)
    idx = np.random.default_rng(SEED).integers(0, len(u), (n, len(u)))
    return s[idx].sum(1) / c[idx].sum(1)


def q(d, lo):
    return float(np.quantile(d, lo)), float(np.quantile(d, 1 - lo))


# ----------------------------------------------------------------------------- open set (folds within one split's persons)
def folds_of(persons):
    return [set(x.tolist()) for x in np.array_split(np.random.default_rng(SEED).permutation(np.array(sorted(persons))), N_FOLDS)]


def open_arrays(tr, S, g, folds, cohort_mu=None):
    """Per fold: mated (top-1 correct flag, top-1 score, person) and non-enrolled (top-1 score, person) with the held persons removed from the gallery. cohort_mu [n]: F5 subtraction."""
    ids, pid = g["ids"], tr.person.to_numpy()
    adj = S - (cohort_mu[:, None] if cohort_mu is not None else 0.0)
    out = []
    for held in folds:
        keep = np.array([p not in held for p in ids])
        Sk = adj[:, keep]
        top = ids[keep][Sk.argmax(1)]
        s1 = Sk.max(1)
        is_non = np.array([p in held for p in pid])
        out.append(dict(m_ok=(top == pid)[~is_non], m_s=s1[~is_non], m_p=pid[~is_non], n_s=s1[is_non], n_p=pid[is_non]))
    return out


def tpir(fold, f=FPIR, mi=None, ni=None):
    ms, mok = (fold["m_s"], fold["m_ok"]) if mi is None else (fold["m_s"][mi], fold["m_ok"][mi])
    ns = fold["n_s"] if ni is None else fold["n_s"][ni]
    t = np.quantile(ns, 1 - f)
    return float(np.mean(mok & (ms >= t)))


def open_paired(fa, fb, n=N_BOOT):
    """Fold-averaged TPIR1 at FPIR 0.1 of arms a and b, and the paired bootstrap of a - b (the same person resamples for both arms, per fold)."""
    pa, pb = np.mean([tpir(x) for x in fa]), np.mean([tpir(x) for x in fb])
    rng = np.random.default_rng(SEED)
    prep = []
    for x, y in zip(fa, fb):
        assert np.array_equal(x["m_p"], y["m_p"]) and np.array_equal(x["n_p"], y["n_p"])
        prep.append((x, y, person_groups(x["m_p"]), person_groups(x["n_p"])))
    da, db = [], []
    for _ in range(n):
        va, vb = [], []
        for x, y, (um, gm), (un, gn) in prep:
            mi = np.concatenate([gm[i] for i in rng.integers(0, len(um), len(um))])
            ni = np.concatenate([gn[i] for i in rng.integers(0, len(un), len(un))])
            va.append(tpir(x, FPIR, mi, ni))
            vb.append(tpir(y, FPIR, mi, ni))
        da.append(np.mean(va))
        db.append(np.mean(vb))
    da, db = np.array(da), np.array(db)
    return float(pa), float(pb), da, db


def cohort_mu(tr, frames, cfg, cohort):
    """F5: per tracklet, mean of its 10 highest cosines to the cohort (same backbone/flip as the tracklet template)."""
    fkey = "wf_flip" if cfg["flip"] else "wf"
    mu = []
    for r in tr.itertuples():
        f = frames[r.tracklet]
        k = frame_sel(f, cfg)
        t = A.RULES[cfg["rule"]](f[fkey][k], f["q"][k], f["idx"][k])
        s = cohort @ t
        mu.append(np.sort(s)[-N_COHORT_TOP:].mean())
    return np.array(mu)


def load_cohort(flip):
    z = np.load(ROOT / "outputs/video/phase9/tinyface_wf.npz")
    return l2n(z["feat_flip" if flip else "feat"][z["role"] == "cohort"])


# ----------------------------------------------------------------------------- stages
def stage_dev(tr, frames, g):
    P9.mkdir(parents=True, exist_ok=True)
    dv = tr[tr.role == "dev"].reset_index(drop=True)
    pid = dv.person.to_numpy()
    cache, rows, ok = {}, [], {}
    # B1: rules with WebFace4M (and MS1MV2 for reference)
    for bk in ("wf", "ms1"):
        for rule in ("R1", "R2", "R4*"):
            S, single = sims(dv, frames, g, dict(BASE, backbone=bk, rule=rule), cache)
            ok[(bk, rule)] = correct(dv, S, g)
            d, dd = boot_mean_diff(ok[(bk, rule)], single, pid)
            rows.append(dict(part="B1", backbone=bk, rule=rule, n_tracklets=len(dv), n_persons=len(np.unique(pid)), single_rank1=single.mean(), rank1=ok[(bk, rule)].mean(),
                             minus_single=d, ci_lo=q(dd, 0.025)[0], ci_hi=q(dd, 0.025)[1]))
    d, dd = boot_mean_diff(ok[("wf", "R4*")], ok[("wf", "R2")], pid)
    fallback = bool(q(dd, 0.025)[1] < 0)
    default_rule = "R2" if fallback else "R4*"
    rows.append(dict(part="B1 D3 fallback test", backbone="wf", rule="R4* minus R2", rank1=np.nan, minus_single=d, ci_lo=q(dd, 0.025)[0], ci_hi=q(dd, 0.025)[1], note=f"fallback applies: {fallback}; default rule = {default_rule}"))
    base = dict(BASE, rule=default_rule)
    S0, _ = sims(dv, frames, g, base, cache)
    ok0 = correct(dv, S0, g)
    # B4: F1-F4 (closed set, Bonferroni over 4)
    variants = {"F1_frame_filter": dict(base, filt=True), "F2_resmatched_gallery": dict(base, gallery="resmatch"), "F3_flip": dict(base, flip=True), "F4_fusion": dict(base, fusion=True)}
    adopted = {}
    for name, cfg in variants.items():
        S, _ = sims(dv, frames, g, cfg, cache)
        o = correct(dv, S, g)
        d, dd = boot_mean_diff(o, ok0, pid)
        blo, bhi = q(dd, 0.025 / len(FIXES))
        adopted[name] = bool(blo > 0)
        rows.append(dict(part="B4", backbone="wf", rule=default_rule, fix=name, rank1=o.mean(), default_rank1=ok0.mean(), minus_default=d, ci_lo=q(dd, 0.025)[0], ci_hi=q(dd, 0.025)[1], bonf_lo=blo, bonf_hi=bhi,
                         adopted=adopted[name] if name != "F4_fusion" else f"{adopted[name]} (subject to the 10 fps condition, Part D)"))
    # F5: open set on the development persons
    folds = folds_of(np.unique(pid))
    coh = load_cohort(False)
    fa = open_arrays(dv, S0, g, folds, cohort_mu(dv, frames, base, coh))
    fb = open_arrays(dv, S0, g, folds)
    pa, pb, da, db = open_paired(fa, fb)
    lo, hi = q(da - db, 0.025)
    adopted["F5_cohort_norm"] = bool(lo > 0)
    rows.append(dict(part="B4", backbone="wf", rule=default_rule, fix="F5_cohort_norm", metric="TPIR1@FPIR0.1 (5 folds of development persons)", rank1=pa, default_rank1=pb, minus_default=pa - pb, ci_lo=lo, ci_hi=hi, adopted=adopted["F5_cohort_norm"]))
    pd.DataFrame(rows).to_csv(P9 / "dev_b1_b4.csv", index=False, float_format="%.5f")
    # B2: decomposition of the default's errors
    err = 1 - ok0
    feat = dv.assign(err=err, yaw=[frames[t]["yaw"].mean() for t in dv.tracklet], lap=[frames[t]["lap"].mean() for t in dv.tracklet], score=[frames[t]["score"].mean() for t in dv.tracklet],
                     n_faces=[len(frames[t]["q"]) for t in dv.tracklet], q_mean=[frames[t]["q"].mean() for t in dv.tracklet])
    b2 = []
    for lo_, hi_ in HBINS:
        s = feat[(feat.mean_face_h >= lo_) & (feat.mean_face_h < hi_)]
        b2.append(dict(factor="mean face height (px)", level=f"{lo_}-{hi_ if hi_ < 10 ** 6 else 'up'}", n=len(s), error=s.err.mean() if len(s) else np.nan))
    for col, name in (("yaw", "yaw proxy"), ("lap", "Laplacian variance"), ("score", "detector score"), ("n_faces", "usable faces"), ("best_gallery_h", "best gallery face height (px)")):
        qs = pd.qcut(feat[col], 4, duplicates="drop")
        for lev, s in feat.groupby(qs, observed=True):
            b2.append(dict(factor=name, level=str(lev), n=len(s), error=s.err.mean()))
    for cam, s in feat.groupby("camera"):
        b2.append(dict(factor="camera", level=str(cam), n=len(s), error=s.err.mean()))
    rank = (S0 > S0[np.arange(len(dv)), np.searchsorted(g["ids"], pid)][:, None]).sum(1) + 1
    wrong = err == 1
    per_p = feat.groupby("person").err.sum().sort_values(ascending=False)
    top10 = per_p.head(max(1, int(np.ceil(0.1 * len(per_p))))).sum() / max(per_p.sum(), 1)
    pairs = pd.Series(list(zip(pid[wrong], g["ids"][S0.argmax(1)][wrong]))).value_counts().head(10)
    summary = dict(n_tracklets=len(dv), n_errors=int(wrong.sum()), error=float(err.mean()), share_errors_true_rank_2_to_5=float(np.mean((rank[wrong] >= 2) & (rank[wrong] <= 5))),
                   share_errors_true_rank_above_20=float(np.mean(rank[wrong] > 20)), share_errors_from_worst_10pct_persons=float(top10),
                   top_confusions=[dict(true=int(a), predicted=int(b), n=int(c)) for (a, b), c in pairs.items()], default_rule=default_rule, fallback=fallback, adopted=adopted)
    pd.DataFrame(b2).to_csv(P9 / "dev_b2_decomposition.csv", index=False, float_format="%.5f")
    (P9 / "dev_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    # B5 on development (tracklets kept by the 23 px bar)
    b5(feat, "development").to_csv(P9 / "dev_b5_residual.csv", index=False, float_format="%.5f")
    audit_sheet(dv, err, S0, g)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(json.dumps(summary, indent=1, sort_keys=True))


def b5(feat, label):
    k = feat[feat.mean_face_h >= BAR_PX].reset_index(drop=True)
    df = pd.DataFrame(dict(person=k.person, key=np.arange(len(k)), err=k.err, mean_magface_norm=k.q_mean, neg_yaw_proxy=-k.yaw, mean_laplacian_variance=k.lap, mean_face_height=k.mean_face_h))
    t, _ = reject_table(df, ["mean_magface_norm", "neg_yaw_proxy", "mean_laplacian_variance", "mean_face_height"], f"MEVID {label}, kept by the {BAR_PX:g} px bar")
    return t


def audit_sheet(dv, err, S, g):
    """B3: 120 wrong tracklets with mean face height >= 23 px + 40 correct controls (same height range), seeded, shuffled. The sheet holds ids only; montages are made locally by --stage montage."""
    rng = np.random.default_rng(SEED)
    hi = dv.mean_face_h.to_numpy() >= BAR_PX
    w = np.flatnonzero(hi & (err == 1))
    c = np.flatnonzero(hi & (err == 0))
    pick = np.concatenate([rng.choice(w, min(120, len(w)), replace=False), rng.choice(c, min(40, len(c)), replace=False)])
    pick = pick[rng.permutation(len(pick))]
    pred = g["ids"][S.argmax(1)]
    sheet = pd.DataFrame(dict(audit_id=[f"A{i:03d}" for i in range(len(pick))], tracklet=dv.tracklet.to_numpy()[pick], photo_a_person=0, photo_b_person=0))
    # gallery photos shown as A/B in a random order, so the labeler cannot tell true from predicted
    swap = rng.random(len(pick)) < 0.5
    t_, p_ = dv.person.to_numpy()[pick], pred[pick]
    sheet["photo_a_person"] = np.where(swap, p_, t_)
    sheet["photo_b_person"] = np.where(swap, t_, p_)
    key = sheet.assign(true_person=t_, predicted_person=p_, wrong=err[pick].astype(int))
    (M / "audit").mkdir(parents=True, exist_ok=True)
    sheet.to_csv(M / "audit" / "sheet.csv", index=False)
    key.to_csv(M / "audit" / "key.csv", index=False)  # not shown to the labelers
    tags = ["not_face_or_back", "profile", "blur", "occlusion", "low_light_or_ir", "tiny", "wrong_person_in_tracklet", "gallery_mismatch", "look_alike", "nothing_visible"]
    pd.DataFrame(dict(audit_id=sheet.audit_id, **{t: "" for t in tags}, comment="")).to_csv(M / "audit" / "labels_TEMPLATE.csv", index=False)


def run_config(tr, frames, g, cfg):
    S, _ = sims(tr, frames, g, cfg, {})
    return S, correct(tr, S, g)


def stage_combine(tr, frames, g):
    s = json.loads((P9 / "dev_summary.json").read_text())
    dv = tr[tr.role == "dev"].reset_index(drop=True)
    base = dict(BASE, rule=s["default_rule"])
    keys = {"F1_frame_filter": ("filt", True), "F2_resmatched_gallery": ("gallery", "resmatch"), "F3_flip": ("flip", True), "F4_fusion": ("fusion", True)}
    fps_ok = json.loads(FPS_COND.read_text())["at_least_10_fps_with_both_backbones"] if FPS_COND.exists() else None
    adopted = [k for k in FIXES if s["adopted"][k] and (k != "F4_fusion" or fps_ok)]
    t = pd.read_csv(P9 / "dev_b1_b4.csv")
    t = t[(t.part == "B4") & t.fix.isin(adopted)]
    cfg = dict(base, **{keys[k][0]: keys[k][1] for k in adopted})
    rows = [dict(config="default", rank1=run_config(dv, frames, g, base)[1].mean())]
    if len(adopted) > 1:
        comb = run_config(dv, frames, g, cfg)[1].mean()
        best = t.sort_values("rank1", ascending=False).iloc[0]
        rows.append(dict(config="combination " + "+".join(adopted), rank1=comb))
        if comb < best.rank1:
            cfg = dict(base, **{keys[best.fix][0]: keys[best.fix][1]})
            adopted = [best.fix]
            rows.append(dict(config=f"combination below best single fix ({best.fix}, {best.rank1:.5f}): only that fix kept", rank1=best.rank1))
    elif adopted:
        rows.append(dict(config="single adopted fix " + adopted[0], rank1=float(t.iloc[0].rank1)))
    # demo threshold (section 0.6 F): mean over the 5 development folds of the FPIR 0.1 threshold of the frozen configuration (cohort-normalised scores if F5 is adopted)
    S, _ = run_config(dv, frames, g, cfg)
    cn = bool(s["adopted"]["F5_cohort_norm"])
    mu = cohort_mu(dv, frames, cfg, load_cohort(cfg["flip"])) if cn else None
    thr = float(np.mean([np.quantile(x["n_s"], 1 - FPIR) for x in open_arrays(dv, S, g, folds_of(np.unique(dv.person)), mu)]))
    frozen = dict(config=cfg, adopted_fixes=adopted, cohort_norm=cn, f4_fps_condition=fps_ok, bar_px=BAR_PX, fpir=FPIR, gallery_bar_px=GAL_BAR, demo_threshold=thr)
    pd.DataFrame(rows).to_csv(P9 / "dev_combination.csv", index=False, float_format="%.5f")
    (P9 / "frozen_config.json").write_text(json.dumps(frozen, indent=1, sort_keys=True))
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(frozen, indent=1, sort_keys=True))


def committed(path):
    r = subprocess.run(["git", "log", "--format=%H", "--", str(path)], cwd=ROOT, capture_output=True, text=True)
    return bool(r.stdout.strip())


def stage_confirm(tr, frames, g):
    fz = P9 / "frozen_config.json"
    if not committed(fz) or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(fz)], cwd=ROOT).returncode != 0:
        sys.exit("refusing: frozen_config.json (section 0b) must be committed, unchanged, before the confirmation split is read")
    if (P9 / "conf_h1_h2.csv").exists():
        sys.exit("refusing: the confirmation split has already been read once (conf_h1_h2.csv exists)")
    frozen = json.loads(fz.read_text())
    cf = tr[tr.role == "conf"].reset_index(drop=True)
    pid = cf.person.to_numpy()
    S, ok = run_config(cf, frames, g, frozen["config"])
    base = pd.read_csv(M / "results/closed_records_adaface_bar48.csv").set_index("tracklet").loc[cf.tracklet]
    ok_b = base.ok_R1.to_numpy(float)
    Sb, ok_b2 = run_config(cf, frames, g, dict(BASE, backbone="ms1", rule="R1"))
    assert np.array_equal(ok_b, ok_b2), "the Phase 8 baseline records are not reproduced"
    d1, dd1 = boot_mean_diff(ok, ok_b, pid)
    folds = folds_of(np.unique(pid))
    mu = cohort_mu(cf, frames, frozen["config"], load_cohort(frozen["config"]["flip"])) if frozen["cohort_norm"] else None
    fa, fb = open_arrays(cf, S, g, folds, mu), open_arrays(cf, Sb, g, folds)
    pa, pb, da, db = open_paired(fa, fb)
    h = [dict(hypothesis="H1 closed-set Rank-1, final minus Phase 8 baseline (MS1MV2 R1)", n_tracklets=len(cf), n_persons=len(np.unique(pid)), final=ok.mean(), baseline=ok_b.mean(), difference=d1,
              ci_lo=q(dd1, 0.025)[0], ci_hi=q(dd1, 0.025)[1], supported=bool(q(dd1, 0.025)[0] > 0)),
         dict(hypothesis="H2 open-set TPIR1 at FPIR 0.1, final minus baseline (5 folds of confirmation persons, paired bootstrap)", n_tracklets=len(cf), n_persons=len(np.unique(pid)), final=pa, baseline=pb, difference=pa - pb,
              ci_lo=q(da - db, 0.025)[0], ci_hi=q(da - db, 0.025)[1], supported=bool(q(da - db, 0.025)[0] > 0))]
    pd.DataFrame(h).to_csv(P9 / "conf_h1_h2.csv", index=False, float_format="%.5f")
    for role, part in (("conf", cf), ("dev", tr[tr.role == "dev"].reset_index(drop=True))):
        o = ok if role == "conf" else run_config(part, frames, g, frozen["config"])[1]
        coverage_curves(part, o, role)
    feat = cf.assign(err=1 - ok, yaw=[frames[t]["yaw"].mean() for t in cf.tracklet], lap=[frames[t]["lap"].mean() for t in cf.tracklet], q_mean=[frames[t]["q"].mean() for t in cf.tracklet])
    b5(feat, "confirmation").to_csv(P9 / "conf_b5_residual.csv", index=False, float_format="%.5f")
    print(pd.DataFrame(h).round(4).to_string(index=False))


def coverage_curves(part, ok, role):
    """Error of the kept tracklets against the px threshold (16-48) and against the rejected fraction (0-0.5, by mean face height); the 23 px point with bootstrap CIs."""
    h, err, pid = part.mean_face_h.to_numpy(), 1 - ok, part.person.to_numpy()
    rows = []
    for t in PX_GRID:
        k = h >= t
        rows.append(dict(split=role, kind="px threshold", x=float(t), coverage=k.mean(), error_kept=err[k].mean() if k.any() else np.nan))
    for x, e in zip(GRID, curve(err, h, np.arange(len(h)))):
        rows.append(dict(split=role, kind="rejected fraction", x=float(x), coverage=1 - x, error_kept=e))
    pd.DataFrame(rows).to_csv(P9 / f"{role}_coverage_curves.csv", index=False, float_format="%.5f")
    k = (h >= BAR_PX).astype(float)
    cov = boot_mean(k, pid)
    u, grp = person_groups(pid)
    idx = np.random.default_rng(SEED).integers(0, len(u), (N_BOOT, len(u)))
    se = np.array([err[i][k[i] == 1].sum() for i in grp])
    sk = np.array([k[i].sum() for i in grp])
    ek = se[idx].sum(1) / np.maximum(sk[idx].sum(1), 1)
    at = dict(split=role, bar_px=BAR_PX, n_tracklets=len(h), coverage=k.mean(), coverage_ci=q(cov, 0.025), error_all=err.mean(), error_kept=err[k == 1].mean(), error_kept_ci=q(ek, 0.025))
    (P9 / f"{role}_at_bar.json").write_text(json.dumps(at, indent=1, sort_keys=True))


# ----------------------------------------------------------------------------- self-test (synthetic, no MEVID)
def selftest():
    rng = np.random.default_rng(0)
    P, n = 30, 300
    ids = np.arange(P)
    G = l2n(rng.normal(size=(P, 512)))
    pid = rng.integers(0, P, n)
    S = l2n(G[pid] + rng.normal(scale=1.2, size=(n, 512)) / np.sqrt(512) * 20) @ G.T
    tr = pd.DataFrame(dict(person=pid))
    g = dict(ids=ids)
    ok = correct(tr, S, g)
    d, dd = boot_mean_diff(ok, ok, pid)
    folds = folds_of(np.unique(pid))
    fa = open_arrays(tr, S, g, folds)
    pa, pb, da, db = open_paired(fa, fa, n=50)
    held_once = sorted(sum((sorted(f) for f in folds), [])) == sorted(np.unique(pid).tolist())
    checks = dict(identical_arms_zero_difference=bool(d == 0 and np.all(dd == 0)), open_identical_arms_zero=bool(pa == pb and np.allclose(da, db)), every_person_held_once=held_once,
                  yaw_proxy_frontal_zero=bool(np.isclose(yaw_proxy(np.array([[[0, 0], [2, 0], [1, 1], [0, 2], [2, 2]]], float))[0], 0)))
    print(json.dumps(checks, indent=1))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "combine", "confirm"])
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.stage == "confirm" and not a.confirm:
        sys.exit("the confirmation split is read only with --stage confirm --confirm (once, after the section 0b commit)")
    tr, frames, g = load()
    assert not set(tr[tr.role == "dev"].person) & set(tr[tr.role == "conf"].person), "development and confirmation persons overlap"
    {"dev": stage_dev, "combine": stage_combine, "confirm": stage_confirm}[a.stage](tr, frames, g)
