"""Extra study: why does the SR penalty differ between SurvFace and TinyFace? Cached embeddings only (no restoration, no inference).

    envs/diag/Scripts/python.exe scripts/extra_sr_asymmetry.py --config configs/sr.yaml [--workers 3] [--out DIR] [--recs adaface] [--n-variants 6]

The tests, the decision criteria and their operationalisation are fixed in REPORT_extra_sr_asymmetry.md (pre-registration and addendum, committed
before this script existed). CPU only, seed 42, 1,000 bootstrap draws; all variants share the same resamples (index sets depend only on the sample
size). Stage 1 = E0 (common scale) and H4 (genuine / impostor split); stage 2 = H2, H3, H5, H6, run only if H1 (protocol artefact) is not supported.
H7 is deferred (needs re-embedding) and is not run. Outputs (deterministic, no timestamps): outputs/sr/results/asymmetry/*.csv, asym_tables.md.
"""
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "outputs/sr/emb/eval"
RECS = ["adaface", "magface", "curricularface"]  # non-leaky; ArcFace is left out (appendix only, not run)
VARIANTS = [("bicubic", "pre512"), ("esrgan", "native"), ("invsr", "pre64_st1"), ("invsr", "pre128_st1"), ("tadsr", "pre128_t250"), ("osdface", "pre128")]
MODES = ("both", "probes")
QUERY_SPLITS = ("probe", "mated_probe", "unmated_probe")
M_BONF = 36  # 6 variants x 3 recognizers x 2 modes
SPLITS = [("tinyface", "probe"), ("tinyface", "gallery_match"), ("tinyface", "distractor"), ("survface", "gallery"),
          ("survface", "mated_probe"), ("survface", "unmated_probe"), ("survface", "verification")]
PIX_BINS = [12, 16, 24, 32]  # min_side bins: <=12, 13-16, 17-24, 25-32, >32
_CTX = {}


# ------------------------------------------------------------------ context
def ctx(out_dir=None):
    if _CTX:
        return _CTX
    import torch
    torch.set_num_threads(max(1, (torch.get_num_threads() // 3) or 1))
    cfg = yaml.safe_load(open(ROOT / "configs/sr.yaml"))
    d = pd.read_csv(ROOT / "outputs/sr/subset/eval_images.csv")
    pairs = pd.read_csv(ROOT / "outputs/sr/subset/eval_pairs.csv")
    assert len(d) == 22898, len(d)
    ix = {k: np.flatnonzero(((d["dataset"] == k[0]) & (d["split"] == k[1])).to_numpy()) for k in SPLITS}
    q_rows = np.concatenate([np.flatnonzero((d["split"] == s).to_numpy()) for s in QUERY_SPLITS])
    ids = d["id"].to_numpy()
    names = d["name"].to_numpy()
    c = dict(cfg=cfg, B=cfg["bootstrap"]["eval_n"], S=cfg["bootstrap"]["seed"], d=d, ix=ix, q_rows=q_rows, ids=ids, names=names, idx_cache={})
    # SurvFace verification pairs: rows of the images in d
    vpos = {n: r for n, r in zip(names[ix[("survface", "verification")]], ix[("survface", "verification")])}
    c["s_pairs"] = {k: (np.array([vpos[x] for x in pairs.loc[pairs["kind"] == k, "a"]]), np.array([vpos[x] for x in pairs.loc[pairs["kind"] == k, "b"]]))
                    for k in ("pos", "neg")}
    c["t_pairs"] = tinyface_pairs(c, Path(out_dir) if out_dir else None)
    # per-crop measures of the mated probes (Phase 2 bin files)
    meas = {}
    for ds, sp in (("tinyface", "probe"), ("survface", "mated_probe")):
        m = pd.read_csv(ROOT / f"outputs/stratify/bins/{ds}_{sp}.csv")[["name", "min_side", "luminance", "lap_var", "size_q"]].drop_duplicates("name")
        rows = d.iloc[ix[(ds, sp)]][["name", "sharp_bin_sz"]].merge(m, on="name", how="left")
        meas[ds] = rows
    c["meas"] = meas
    # SurvFace comparisons (mated probe vs every same-id gallery image) and cameras
    cam = lambda s: int(re.search(r"_cam(\d+)_", s).group(1))  # noqa: E731
    sg, sm = ix[("survface", "gallery")], ix[("survface", "mated_probe")]
    gid = ids[sg]
    by_id = {}
    for j, i_ in enumerate(gid):
        by_id.setdefault(i_, []).append(j)
    qpos, gpos = [], []
    for i, i_ in enumerate(ids[sm]):
        for j in by_id[i_]:
            qpos.append(i)
            gpos.append(j)
    c["cmp_q"], c["cmp_g"] = np.array(qpos), np.array(gpos)
    gi, gj = [], []  # supplementary: pairs of gallery images of the same identity (mated query-gallery pairs are all cross-camera)
    for lst in by_id.values():
        for a_ in range(len(lst)):
            for b_ in range(a_ + 1, len(lst)):
                gi.append(lst[a_])
                gj.append(lst[b_])
    c["gg_i"], c["gg_j"] = np.array(gi), np.array(gj)
    c["cam_g"] = np.array([cam(n) for n in names[sg]])
    c["cam_q"] = np.array([cam(n) for n in names[sm]])
    _CTX.update(c)
    return _CTX


def tinyface_pairs(c, out_dir):
    """1,000 genuine and 1,000 impostor probe-gallery pairs (seed 42), written to tinyface_pairs.csv before any scoring."""
    ix, ids, names = c["ix"], c["ids"], c["names"]
    tp, tg = ix[("tinyface", "probe")], ix[("tinyface", "gallery_match")]
    rng = np.random.default_rng(c["S"])
    gal_by_id = {}
    for r in tg:
        gal_by_id.setdefault(ids[r], []).append(r)
    ok = np.array([i for i, r in enumerate(tp) if ids[r] in gal_by_id])
    gen_p = tp[np.sort(rng.choice(ok, 1000, replace=False))]
    gen_g = np.array([gal_by_id[ids[r]][rng.integers(0, len(gal_by_id[ids[r]]))] for r in gen_p])
    imp_p = tp[np.sort(rng.choice(len(tp), 1000, replace=False))]
    imp_g = rng.choice(tg, 1000)
    for k in range(1000):
        while ids[imp_g[k]] == ids[imp_p[k]]:
            imp_g[k] = rng.choice(tg)
    df = pd.DataFrame(dict(label=["genuine"] * 1000 + ["impostor"] * 1000, probe_row=np.concatenate([gen_p, imp_p]), gallery_row=np.concatenate([gen_g, imp_g])))
    df["probe_name"], df["gallery_name"] = names[df.probe_row], names[df.gallery_row]
    df["probe_id"], df["gallery_id"] = ids[df.probe_row], ids[df.gallery_row]
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "tinyface_pairs.csv", index=False)
    return dict(pos=(gen_p, gen_g), neg=(imp_p, imp_g))


def IDX(n, tag=0):
    """Shared bootstrap index sets: depend only on (sample size, tag), so every variant and recognizer uses the same resamples."""
    c = _CTX
    if (n, tag) not in c["idx_cache"]:
        c["idx_cache"][(n, tag)] = np.random.default_rng([c["S"], n, tag]).integers(0, n, size=(c["B"], n))
    return c["idx_cache"][(n, tag)]


def ci(draws):
    a = 0.05
    lo, hi = np.quantile(draws, [a / 2, 1 - a / 2])
    blo, bhi = np.quantile(draws, [a / 2 / M_BONF, 1 - a / 2 / M_BONF])
    return float(lo), float(hi), float(blo), float(bhi)


def ci_cols(prefix, draws):
    lo, hi, blo, bhi = ci(draws)
    return {f"{prefix}_ci_lo": lo, f"{prefix}_ci_hi": hi, f"{prefix}_bonf_lo": blo, f"{prefix}_bonf_hi": bhi}


# ------------------------------------------------------------------ embeddings and modes
def load_rec(rec, variants):
    dire = np.load(CACHE / "direct/resize/seed0" / f"eval__{rec}.npy")
    rest = {v: np.load(CACHE / v[0] / v[1] / "seed0" / f"eval__{rec}.npy") for v in variants}
    assert dire.shape[0] == 22898 and all(r.shape == dire.shape for r in rest.values())
    return dire, rest


def emb_mode(dire, rest, mode):
    if mode == "direct":
        return dire
    if mode == "both":
        return rest
    h = dire.copy()
    h[_CTX["q_rows"]] = rest[_CTX["q_rows"]]
    return h


def l2n(x):
    from src import eval_protocols as ev
    return ev.l2n(x)


def pair_scores(dire, rest, mode, a, b):
    """cosine and Euclidean distance (normalised embeddings) of pairs (a, b). Probes-only: first image restored, second direct."""
    A = l2n((dire if mode == "direct" else rest)[a])
    Bm = l2n(rest[b] if mode == "both" else dire[b])
    cos = (A * Bm).sum(1).astype(np.float64)
    return cos, np.sqrt(np.clip(2 - 2 * cos, 0, None))


def identification(x, probe_rows, gal_rows, dist_rows):
    from src import eval_protocols as ev
    c = _CTX
    ids = c["ids"]
    empty = x[:0]
    r = ev.tinyface_identification(x[probe_rows], ids[probe_rows], x[gal_rows], ids[gal_rows], x[dist_rows] if dist_rows is not None else empty, device="cpu")
    return r["_per_sample"]["first_rank"]


def auc(pos, neg):
    from src import eval_protocols as ev
    return ev.survface_verification_from_scores(pos, neg, accuracy=False)["AUC"]


def auc_draws(pos, neg, tag):
    ip, inn = IDX(len(pos), tag), IDX(len(neg), tag + 1)
    return np.array([auc(pos[ip[b]], neg[inn[b]]) for b in range(_CTX["B"])])


def dprime(g, i):
    return (g.mean(-1) - i.mean(-1)) / np.sqrt((g.var(-1) + i.var(-1)) / 2)


def dprime_draws(g, i, tag):
    return dprime(g[IDX(len(g), tag)], i[IDX(len(i), tag + 1)])


# ------------------------------------------------------------------ stage 1: E0 and H4
def stage1(rec, variants):
    c = _CTX
    ix = c["ix"]
    dire, rest = load_rec(rec, variants)
    tp, tg, td = ix[("tinyface", "probe")], ix[("tinyface", "gallery_match")], ix[("tinyface", "distractor")]
    sg, sm = ix[("survface", "gallery")], ix[("survface", "mated_probe")]

    def stats(x, mode, d_, r_):
        """per-sample arrays of one embedding configuration. x: identification embeddings; d_, r_ direct/restored arrays for pair scoring."""
        out = dict(t_r1=(identification(x, tp, tg, td) < 1).astype(float), s_r1=(identification(x, sm, sg, None) < 1).astype(float))
        for ds, pp in (("t", c["t_pairs"]), ("s", c["s_pairs"])):
            for k in ("pos", "neg"):
                out[f"{ds}_{k}_cos"], out[f"{ds}_{k}_dist"] = pair_scores(d_, r_, mode, *pp[k])
        return out

    base = stats(dire, "direct", dire, dire)
    # cached direct draws
    dd = dict(t_r1=base["t_r1"][IDX(len(base["t_r1"]), 10)].mean(1), s_r1=base["s_r1"][IDX(len(base["s_r1"]), 20)].mean(1),
              t_auc=auc_draws(base["t_pos_dist"], base["t_neg_dist"], 30), s_auc=auc_draws(base["s_pos_dist"], base["s_neg_dist"], 40),
              t_dp=dprime_draws(base["t_pos_cos"], base["t_neg_cos"], 50), s_dp=dprime_draws(base["s_pos_cos"], base["s_neg_cos"], 60))
    dpt = {ds: (auc(base[f"{ds}_pos_dist"], base[f"{ds}_neg_dist"]), float(dprime(base[f"{ds}_pos_cos"], base[f"{ds}_neg_cos"]))) for ds in ("t", "s")}
    e0, gap, h4, check = [], [], [], []
    for v in variants:
        for mode in MODES:
            x = emb_mode(dire, rest[v], mode)
            st = stats(x, mode, dire, rest[v])
            rd = dict(t_r1=st["t_r1"][IDX(len(st["t_r1"]), 10)].mean(1), s_r1=st["s_r1"][IDX(len(st["s_r1"]), 20)].mean(1),
                      t_auc=auc_draws(st["t_pos_dist"], st["t_neg_dist"], 30), s_auc=auc_draws(st["s_pos_dist"], st["s_neg_dist"], 40),
                      t_dp=dprime_draws(st["t_pos_cos"], st["t_neg_cos"], 50), s_dp=dprime_draws(st["s_pos_cos"], st["s_neg_cos"], 60))
            point = dict(t_r1=(base["t_r1"].mean(), st["t_r1"].mean()), s_r1=(base["s_r1"].mean(), st["s_r1"].mean()),
                         t_auc=(dpt["t"][0], auc(st["t_pos_dist"], st["t_neg_dist"])), s_auc=(dpt["s"][0], auc(st["s_pos_dist"], st["s_neg_dist"])),
                         t_dp=(dpt["t"][1], float(dprime(st["t_pos_cos"], st["t_neg_cos"]))), s_dp=(dpt["s"][1], float(dprime(st["s_pos_cos"], st["s_neg_cos"]))))
            delta = {}
            for key in point:
                ds = "tinyface" if key[0] == "t" else "survface"
                metric = {"r1": "closed_set_rank1", "auc": "verification_auc", "dp": "verification_dprime"}[key[2:]]
                dv, rv = point[key]
                dl = rd[key] - dd[key]
                delta[key] = dl
                row = dict(arm=v[0], setting=v[1], recognizer=rec, mode=mode, dataset=ds, metric=metric, direct=dv, restored=rv, delta=rv - dv,
                           rel_delta=(rv - dv) / dv, **ci_cols("delta", dl), **ci_cols("rel", dl / dd[key]))
                e0.append(row)
            for m_ in ("r1", "auc", "dp"):
                g = delta[f"t_{m_}"] - delta[f"s_{m_}"]
                pt = (point[f"t_{m_}"][1] - point[f"t_{m_}"][0]) - (point[f"s_{m_}"][1] - point[f"s_{m_}"][0])
                lo, hi, blo, bhi = ci(g)
                gap.append(dict(arm=v[0], setting=v[1], recognizer=rec, mode=mode, metric={"r1": "closed_set_rank1", "auc": "verification_auc", "dp": "verification_dprime"}[m_],
                                delta_tinyface=point[f"t_{m_}"][1] - point[f"t_{m_}"][0], delta_survface=point[f"s_{m_}"][1] - point[f"s_{m_}"][0], gap=pt,
                                gap_ci_lo=lo, gap_ci_hi=hi, gap_bonf_lo=blo, gap_bonf_hi=bhi, ci_contains_zero=bool(lo <= 0 <= hi), bonf_contains_zero=bool(blo <= 0 <= bhi)))
            # H4: genuine / impostor decomposition on the pair sets
            for ds in ("t", "s"):
                gd, idd = base[f"{ds}_pos_cos"], base[f"{ds}_neg_cos"]
                g_r, i_r = st[f"{ds}_pos_cos"], st[f"{ds}_neg_cos"]
                tg_, ti_ = (60, 61) if ds == "s" else (50, 51)
                ip, inn = IDX(len(gd), tg_), IDX(len(idd), ti_)
                dg = g_r[ip].mean(1) - gd[ip].mean(1)
                di = i_r[inn].mean(1) - idd[inn].mean(1)
                dgp, dip = g_r.mean() - gd.mean(), i_r.mean() - idd.mean()
                h4.append(dict(arm=v[0], setting=v[1], recognizer=rec, mode=mode, dataset="tinyface" if ds == "t" else "survface",
                               delta_genuine=dgp, delta_impostor=dip, delta_dprime=point[f"{ds}_dp"][1] - point[f"{ds}_dp"][0],
                               delta_auc=point[f"{ds}_auc"][1] - point[f"{ds}_auc"][0], impostor_minus_genuine=dip - dgp,
                               **ci_cols("delta_genuine", dg), **ci_cols("delta_impostor", di), **ci_cols("impostor_minus_genuine", di - dg),
                               **ci_cols("delta_dprime", delta[f"{ds}_dp"]), **ci_cols("delta_auc", delta[f"{ds}_auc"])))
            if mode == "both":  # cross-check against the Phase 3 results file (read only)
                check.append(dict(arm=v[0], setting=v[1], recognizer=rec, tinyface_rank1_direct_here=point["t_r1"][0], tinyface_rank1_both_here=point["t_r1"][1],
                                  survface_auc_direct_here=point["s_auc"][0], survface_auc_both_here=point["s_auc"][1]))
    return dict(e0=e0, gap=gap, h4=h4, check=check)


# ------------------------------------------------------------------ stage 2: H2, H3, H5, H6
def gen_sim(x, probe_rows, gal_rows):
    """Mean cosine between each probe and all gallery images of the same identity."""
    ids = _CTX["ids"]
    P, G = l2n(x[probe_rows]), l2n(x[gal_rows])
    S = P @ G.T
    m = ids[probe_rows][:, None] == ids[gal_rows][None, :]
    return (S * m).sum(1) / m.sum(1)


def boot_mean_ci(x, tag):
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    dr = x[IDX(n, tag)].mean(1)
    lo, hi, _, _ = ci(dr)
    return float(x.mean()), lo, hi


def gap_draws(dT, dS, kT, kS, K):
    """Unmatched and matched dataset gap (TinyFace minus SurvFace) in the change of genuine similarity, point + bootstrap draws.
    k = stratum code (0..K-1) or -1 (stratum not present in both datasets)."""
    def one(xT, sT, xS, sS):
        unm = xT.mean() - xS.mean()
        cT, cS = np.bincount(sT + 1, minlength=K + 1)[1:], np.bincount(sS + 1, minlength=K + 1)[1:]
        mT = np.bincount(sT + 1, weights=xT, minlength=K + 1)[1:] / np.maximum(cT, 1)
        mS = np.bincount(sS + 1, weights=xS, minlength=K + 1)[1:] / np.maximum(cS, 1)
        ok = (cT > 0) & (cS > 0)
        w = (cT + cS) * ok
        mat = float(((mT - mS) * w).sum() / w.sum()) if w.sum() > 0 else np.nan
        return unm, mat

    p = one(dT, kT, dS, kS)
    iT, iS = IDX(len(dT), 70), IDX(len(dS), 71)
    dr = np.array([one(dT[iT[b]], kT[iT[b]], dS[iS[b]], kS[iS[b]]) for b in range(_CTX["B"])])
    return p, dr


def strata_codes(keyT, keyS):
    common = sorted(set(keyT) & set(keyS))
    pos = {k: i for i, k in enumerate(common)}
    return np.array([pos.get(k, -1) for k in keyT]), np.array([pos.get(k, -1) for k in keyS]), common


def cam_rows(store, same, c, rec, label):
    """Same-camera vs cross-camera change in cosine for the comparisons in `store` ({(variant, mode): delta array}); bootstrap over comparisons."""
    cells = sorted(store, key=lambda t: (t[0], t[1]))
    sa, cr = np.flatnonzero(same), np.flatnonzero(~same)
    rng = np.random.default_rng([c["S"], len(same), 5])
    d_same = {k: store[k][sa] for k in cells}
    d_cross = {k: store[k][cr] for k in cells}
    draws = {k: np.full((c["B"], 2), np.nan) for k in cells}
    for b in range(c["B"]):
        i1 = rng.integers(0, len(sa), len(sa)) if len(sa) else None
        i2 = rng.integers(0, len(cr), len(cr)) if len(cr) else None
        for k in cells:
            draws[k][b] = (d_same[k][i1].mean() if len(sa) else np.nan, d_cross[k][i2].mean() if len(cr) else np.nan)
    rows = []
    for k in cells:
        v, mode = k
        dr = draws[k]
        ps = float(d_same[k].mean()) if len(sa) else np.nan
        pc = float(d_cross[k].mean()) if len(cr) else np.nan
        if len(sa) and len(cr):
            lo, hi, blo, bhi = ci(dr[:, 1] - dr[:, 0])
        else:
            lo = hi = blo = bhi = np.nan
        rows.append(dict(arm=v[0], setting=v[1], recognizer=rec, mode=mode, comparisons=label, n_same_camera=len(sa), n_cross_camera=len(cr), delta_same_camera=ps, delta_cross_camera=pc,
                         cross_minus_same=pc - ps, ci_lo=lo, ci_hi=hi, bonf_lo=blo, bonf_hi=bhi, ci_below_zero=bool(hi < 0) if np.isfinite(hi) else False))
    return rows


def stage2(rec, variants):
    from src import eval_protocols as ev
    c = _CTX
    ix, ids = c["ix"], c["ids"]
    dire, rest = load_rec(rec, variants)
    tp, tg, td = ix[("tinyface", "probe")], ix[("tinyface", "gallery_match")], ix[("tinyface", "distractor")]
    sg, sm, su = ix[("survface", "gallery")], ix[("survface", "mated_probe")], ix[("survface", "unmated_probe")]
    meas = c["meas"]
    h2b, h2m, h3, h3i, h5c, h5k, h6 = [], [], [], [], [], [], []
    # ---- H2 inputs
    gd = dict(tinyface=gen_sim(dire, tp, tg), survface=gen_sim(dire, sm, sg))
    mT, mS = meas["tinyface"], meas["survface"]
    ok = dict(tinyface=mT[["min_side", "luminance", "lap_var", "size_q"]].notna().all(1).to_numpy(), survface=mS[["min_side", "luminance", "lap_var", "size_q"]].notna().all(1).to_numpy())
    pool = {f: np.concatenate([mT[f].to_numpy()[ok["tinyface"]], mS[f].to_numpy()[ok["survface"]]]) for f in ("min_side", "luminance", "lap_var")}
    edges = {"min_side": np.array(PIX_BINS, float), "luminance": np.quantile(pool["luminance"], [.2, .4, .6, .8]), "lap_var": np.quantile(pool["lap_var"], [.2, .4, .6, .8])}
    pixbin = lambda m: pd.Series(np.digitize(m["min_side"].to_numpy(), np.array(PIX_BINS) + 0.5)).astype(str)  # noqa: E731
    key_pre = {ds: (m["size_q"].fillna(-9).astype(int).astype(str) + "|" + m["sharp_bin_sz"].astype(str)).to_numpy() for ds, m in (("tinyface", mT), ("survface", mS))}
    key_pix = {ds: (pixbin(m) + "|" + m["sharp_bin_sz"].astype(str)).to_numpy() for ds, m in (("tinyface", mT), ("survface", mS))}
    # ---- H5 inputs (comparisons)
    qpos, gpos, camq, camg = c["cmp_q"], c["cmp_g"], c["cam_q"], c["cam_g"]
    same = camq[qpos] == camg[gpos]
    G0 = l2n(dire[sg])
    Qd = l2n(dire[sm])
    cos_d = (Qd @ G0.T)[qpos, gpos]

    def knn_correct(G):
        S = G @ G.T
        np.fill_diagonal(S, -np.inf)
        nn = np.argpartition(-S, 5, axis=1)[:, :5]
        ncam = max(camg.max(), 1) + 1
        votes = np.zeros((len(G), ncam), int)
        for k in range(5):
            np.add.at(votes, (np.arange(len(G)), camg[nn[:, k]]), 1)
        return (votes.argmax(1) == camg).astype(float)

    kd = knn_correct(G0)
    # ---- H3 / H6 direct baselines
    fr_d = identification(dire, tp, tg, td)
    r1_d = (fr_d < 1).astype(float)
    r1_dd = r1_d[IDX(len(r1_d), 10)].mean(1)
    Pn, Gn, Dn = l2n(dire[tp]), l2n(dire[tg]), l2n(dire[td])
    samegal = ids[tp][:, None] == ids[tg][None, :]
    Sg_d = Pn @ Gn.T
    imp_gal_d = np.where(samegal, -2, Sg_d).max(1)
    imp_dis_d = (Pn @ Dn.T).max(1)
    sf_d = ev.survface_openset(dire[sg], ids[sg], dire[sm], ids[sm], dire[su], device="cpu")["_per_sample"]
    iM, iU = IDX(len(sm), 80), IDX(len(su), 81)

    def tpir_th(s1, gt, th=None):
        s1s, gts = np.sort(s1), np.sort(gt)
        step = (s1s[-1] - s1s[0]) / 1000
        grid = np.arange(s1s[0], s1s[-1] + step / 2, step)
        fp = ev._count_below(s1s, grid) / len(s1s)
        tp_ = ev._count_below(gts, grid) / len(gts)
        o = np.argsort(fp, kind="stable")
        k = np.argsort(np.abs(fp[o] - 0.3), kind="stable")[0]
        t = grid[o][k] if th is None else th
        return float(ev._count_below(gts, np.array([t]))[0] / len(gts)) if th is not None else float(tp_[o][k]), float(t)

    def h6_parts(s1d, gtd, s1r, gtr):
        a, th_d = tpir_th(s1d, gtd)
        full, _ = tpir_th(s1r, gtr)
        mixed, _ = tpir_th(s1r, gtr, th_d)
        return full - a, mixed - a, full - mixed  # total, mated-score part, threshold-shift part

    assert np.allclose(h6_parts(sf_d["s1"], sf_d["gt"], sf_d["s1"], sf_d["gt"]), 0)  # direct against itself
    assert abs(tpir_th(sf_d["s1"], sf_d["gt"])[0] - ev.survface_openset_from_scores(sf_d["s1"], sf_d["gt"])["TPIR20@FPIR=0.3"]) < 1e-12  # same arithmetic as the official code
    cmp_store, gg_store = {}, {}
    cosgg_d = (G0 @ G0.T)[c["gg_i"], c["gg_j"]]
    for v in variants:
        R = rest[v]
        # ---- H2 (mode both)
        gr = dict(tinyface=gen_sim(R, tp, tg), survface=gen_sim(R, sm, sg))
        dl = {ds: gr[ds] - gd[ds] for ds in gd}
        for ds, m in (("tinyface", mT), ("survface", mS)):
            for f in ("min_side", "luminance", "lap_var"):
                b = np.digitize(m[f].to_numpy(), edges[f] + (0.5 if f == "min_side" else 0))
                for q in range(5):
                    sel = ok[ds] & (b == q)
                    mean, lo, hi = boot_mean_ci(dl[ds][sel], 90 + q)
                    h2b.append(dict(arm=v[0], setting=v[1], recognizer=rec, dataset=ds, feature=f, bin=q + 1, n=int(sel.sum()), mean_delta_genuine=mean, ci_lo=lo, ci_hi=hi))
        for label, keys in (("preregistered_size_q_x_sharp", key_pre), ("sensitivity_pixel_bins_x_sharp", key_pix)):
            kT, kS, common = strata_codes(keys["tinyface"][ok["tinyface"]], keys["survface"][ok["survface"]])
            p, dr = gap_draws(dl["tinyface"][ok["tinyface"]], dl["survface"][ok["survface"]], kT, kS, len(common))
            lo_u, hi_u, _, _ = ci(dr[:, 0])
            lo_m, hi_m, _, _ = ci(dr[np.isfinite(dr[:, 1]), 1])
            h2m.append(dict(arm=v[0], setting=v[1], recognizer=rec, mode="both", strata=label, n_common_strata=len(common), unmatched_gap=p[0], unmatched_ci_lo=lo_u, unmatched_ci_hi=hi_u,
                            matched_gap=p[1], matched_ci_lo=lo_m, matched_ci_hi=hi_m,
                            shrink_fraction=(1 - abs(p[1]) / abs(p[0])) if p[0] != 0 else np.nan, criterion_met=bool(abs(p[1]) <= 0.5 * abs(p[0]))))
        # ---- H3 (mode both)
        hyb = dire.copy()
        hyb[tp], hyb[tg] = R[tp], R[tg]
        r1_r = (identification(R, tp, tg, td) < 1).astype(float)
        r1_h = (identification(hyb, tp, tg, td) < 1).astype(float)
        ii = IDX(len(r1_d), 10)
        d_std, d_h3 = r1_r[ii].mean(1) - r1_dd, r1_h[ii].mean(1) - r1_dd
        lo, hi, _, _ = ci(d_std)
        chg = (r1_h.mean() - r1_r.mean())
        h3.append(dict(arm=v[0], setting=v[1], recognizer=rec, mode="both", delta_standard=r1_r.mean() - r1_d.mean(), delta_standard_ci_lo=lo, delta_standard_ci_hi=hi,
                       delta_distractors_direct=r1_h.mean() - r1_d.mean(), change_in_delta=chg, ci_half_width=(hi - lo) / 2, criterion_met=bool(abs(chg) > (hi - lo) / 2),
                       **ci_cols("change_in_delta", d_h3 - d_std)))
        Pr, Gr, Dr = l2n(R[tp]), l2n(R[tg]), l2n(R[td])
        ig = np.where(samegal, -2, Pr @ Gr.T).max(1) - imp_gal_d
        idis = (Pr @ Dr.T).max(1) - imp_dis_d
        for nm, arr in (("gallery_match_impostors", ig), ("distractor_impostors", idis)):
            m_, lo_, hi_ = boot_mean_ci(arr, 12 if nm.startswith("g") else 13)
            h3i.append(dict(arm=v[0], setting=v[1], recognizer=rec, mode="both", impostor_set=nm, delta_mean=m_, ci_lo=lo_, ci_hi=hi_))
        # ---- H5 kNN camera accuracy (mode both) and comparisons (both, probes)
        Gr_ = l2n(R[sg])
        kr = knn_correct(Gr_)
        dk = kr[IDX(len(kr), 14)].mean(1) - kd[IDX(len(kd), 14)].mean(1)
        lo, hi, blo, bhi = ci(dk)
        h5k.append(dict(arm=v[0], setting=v[1], recognizer=rec, mode="both", knn_acc_direct=kd.mean(), knn_acc_restored=kr.mean(), delta=kr.mean() - kd.mean(),
                        ci_lo=lo, ci_hi=hi, bonf_lo=blo, bonf_hi=bhi, ci_above_zero=bool(lo > 0)))
        for mode in MODES:
            Qr = l2n(R[sm])
            Gm = Gr_ if mode == "both" else G0
            cmp_store[(v, mode)] = (Qr @ Gm.T)[qpos, gpos] - cos_d
        gg_store[(v, "both")] = (Gr_ @ Gr_.T)[c["gg_i"], c["gg_j"]] - cosgg_d
        # ---- H6 (both, probes)
        for mode in MODES:
            x = emb_mode(dire, R, mode)
            sf = ev.survface_openset(x[sg], ids[sg], x[sm], ids[sm], x[su], device="cpu")["_per_sample"]
            pt = h6_parts(sf_d["s1"], sf_d["gt"], sf["s1"], sf["gt"])
            dr = np.array([h6_parts(sf_d["s1"][iU[b]], sf_d["gt"][iM[b]], sf["s1"][iU[b]], sf["gt"][iM[b]]) for b in range(c["B"])])
            row = dict(arm=v[0], setting=v[1], recognizer=rec, mode=mode, delta_tpir_total=pt[0], mated_score_part=pt[1], threshold_shift_part=pt[2],
                       criterion_met=bool(abs(pt[2]) > abs(pt[1])))
            for j, nm in enumerate(("delta_tpir_total", "mated_score_part", "threshold_shift_part")):
                row.update(ci_cols(nm, dr[:, j]))
            h6.append(row)
    # ---- H5 comparisons: one bootstrap loop over draws for all cells (resample comparisons)
    h5c = cam_rows(cmp_store, same, c, rec, "mated_query_vs_gallery")
    h5g = cam_rows(gg_store, camg[c["gg_i"]] == camg[c["gg_j"]], c, rec, "gallery_vs_gallery_supplementary")
    return dict(h2b=h2b, h2m=h2m, h3=h3, h3i=h3i, h5c=h5c, h5g=h5g, h5k=h5k, h6=h6)


# ------------------------------------------------------------------ driver
def worker(args):
    rec, variants, stage, _ = args
    ctx(None)  # the main process wrote tinyface_pairs.csv; the pair list is deterministic, so every worker regenerates the same one
    return rec, (stage1 if stage == 1 else stage2)(rec, variants)


def run_stage(stage, recs, variants, workers, out_dir):
    tasks = [(r, variants, stage, out_dir) for r in recs]
    if workers <= 1:
        res = [worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(worker, tasks))
    merged = {}
    for _, r in sorted(res, key=lambda x: recs.index(x[0])):
        for k, v in r.items():
            merged.setdefault(k, []).extend(v)
    return {k: pd.DataFrame(v) for k, v in merged.items()}


def verdict(n_met, n):
    f = n_met / n
    return "supported" if f > 0.5 else ("not supported" if f < 1 / 3 else "inconclusive")


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def fmt(v, lo=None, hi=None, d=3):
    return f"{v:+.{d}f}" + (f" [{lo:+.{d}f}, {hi:+.{d}f}]" if lo is not None else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sr.yaml")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "outputs/sr/results/asymmetry"))
    ap.add_argument("--recs", nargs="+", default=RECS)
    ap.add_argument("--n-variants", type=int, default=len(VARIANTS))
    ap.add_argument("--exploratory-stage2", action="store_true", help="run stage 2 (H2, H3, H5, H6) even though H1 is supported; an exploratory extension requested by Abser, "
                    "not part of the pre-registered stop rule. Use with --out outputs/sr/results/asymmetry_exploratory")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = VARIANTS[:a.n_variants]
    full = len(a.recs) == 3 and len(variants) == 6
    ctx(out)  # writes tinyface_pairs.csv before any scoring
    print("pairs written:", out / "tinyface_pairs.csv", flush=True)
    s1 = run_stage(1, a.recs, variants, a.workers, out)
    for k in ("e0", "gap", "h4"):
        df = s1[k].sort_values([c for c in ("arm", "setting", "recognizer", "mode", "dataset", "metric") if c in s1[k].columns]).reset_index(drop=True)
        df.to_csv(out / {"e0": "asym_common_scale.csv", "gap": "asym_dataset_gap.csv", "h4": "h4_genuine_impostor.csv"}[k], index=False, float_format="%.8g")
    s1["check"].sort_values(["arm", "setting", "recognizer"]).to_csv(out / "phase3_crosscheck_inputs.csv", index=False, float_format="%.8g")
    verdicts = []
    # ---- H1
    gap = s1["gap"]
    h1 = {}
    for m_ in ("closed_set_rank1", "verification_auc"):
        g = gap[gap.metric == m_]
        h1[m_] = (int(g.ci_contains_zero.sum()), len(g), int(g.bonf_contains_zero.sum()))
    h1_supported = all(n / t > 0.5 for n, t, _ in h1.values())
    verdicts.append(dict(hypothesis="H1 protocol artefact", criterion="CI of (TinyFace change - SurvFace change) contains zero in more than half of the cells, for both matched metrics",
                         detail=json.dumps(h1), verdict="supported" if h1_supported else "not supported" if all(n / t < 1 / 3 for n, t, _ in h1.values()) else "inconclusive",
                         bonferroni_detail="; ".join(f"{k}: {b}/{t} cells contain zero under Bonferroni" for k, (_, t, b) in h1.items())))
    # ---- H4
    h4 = s1["h4"]
    sp = []
    for (ds, rec, mode), g in h4.groupby(["dataset", "recognizer", "mode"]):
        sp.append(dict(dataset=ds, recognizer=rec, mode=mode, n_variants=len(g), spearman_penalty_vs_impostor_minus_genuine=spearman(-g.delta_auc.to_numpy(), g.impostor_minus_genuine.to_numpy())))
    sp = pd.DataFrame(sp)
    sp["criterion_met"] = sp.spearman_penalty_vs_impostor_minus_genuine > 0
    sp.to_csv(out / "h4_spearman.csv", index=False, float_format="%.8g")
    verdicts.append(dict(hypothesis="H4 homogenisation (mechanism)", criterion="Spearman across variants between penalty and (delta impostor - delta genuine) above zero, per dataset x recognizer x mode",
                         detail=f"{int(sp.criterion_met.sum())}/{len(sp)} groups", verdict=verdict(int(sp.criterion_met.sum()), len(sp)), bonferroni_detail="not applicable (no CI)"))
    s2 = None
    if not h1_supported or a.exploratory_stage2:
        s2 = run_stage(2, a.recs, variants, a.workers, out)
        names = dict(h2b="h2_binned_delta_genuine.csv", h2m="h2_matched_strata.csv", h3="h3_distractors.csv", h3i="h3_impostor_split.csv", h5c="h5_camera_pairs.csv", h5g="h5_supplementary_gallery_pairs.csv",
                     h5k="h5_knn_camera.csv", h6="h6_threshold_split.csv")
        for k, f in names.items():
            df = s2[k]
            df = df.sort_values([c for c in ("arm", "setting", "recognizer", "mode", "dataset", "feature", "bin", "strata") if c in df.columns]).reset_index(drop=True)
            df.to_csv(out / f, index=False, float_format="%.8g")
        m = s2["h2m"][s2["h2m"].strata == "preregistered_size_q_x_sharp"]
        verdicts.append(dict(hypothesis="H2 input information", criterion="matched-strata gap in change of genuine similarity at most half the unmatched gap (pre-registered strata, mode both)",
                             detail=f"{int(m.criterion_met.sum())}/{len(m)} cells", verdict=verdict(int(m.criterion_met.sum()), len(m)),
                             bonferroni_detail="sensitivity (pixel-size strata): " + f"{int(s2['h2m'][s2['h2m'].strata != 'preregistered_size_q_x_sharp'].criterion_met.sum())}/{len(m)} cells"))
        h3 = s2["h3"]
        verdicts.append(dict(hypothesis="H3 distractor composition", criterion="distractors at direct feed change the TinyFace Rank-1 change by more than the CI half-width of the standard change (mode both)",
                             detail=f"{int(h3.criterion_met.sum())}/{len(h3)} cells", verdict=verdict(int(h3.criterion_met.sum()), len(h3)), bonferroni_detail="not applicable (no CI on the verdict quantity)"))
        k5, c5 = s2["h5k"], s2["h5c"][s2["h5c"]["mode"] == "both"].reset_index(drop=True)
        k5 = k5.sort_values(["arm", "setting", "recognizer"]).reset_index(drop=True)
        c5 = c5.sort_values(["arm", "setting", "recognizer"]).reset_index(drop=True)
        usable = c5.n_same_camera.to_numpy() > 0
        met = k5.ci_above_zero.to_numpy() | (c5.ci_below_zero.to_numpy() & usable)
        g5 = s2["h5g"][s2["h5g"]["mode"] == "both"]
        verdicts.append(dict(hypothesis="H5 camera domain", criterion="kNN camera accuracy rises (CI above zero) or cross-camera change minus same-camera change has a CI below zero (mode both)",
                             detail=(f"{int(met.sum())}/{len(met)} cells; kNN part {int(k5.ci_above_zero.sum())}; cross-camera part not testable: {int((~usable).sum())}/{len(usable)} cells have no same-camera "
                                     f"mated comparison (every mated query-gallery pair is cross-camera)"), verdict=verdict(int(met.sum()), len(met)),
                             bonferroni_detail=f"supplementary gallery-vs-gallery comparisons (not used for the verdict): cross-camera change below same-camera in {int(g5.ci_below_zero.sum())}/{len(g5)} cells"))
        h6 = s2["h6"]
        verdicts.append(dict(hypothesis="H6 open-set threshold", criterion="absolute threshold-shift part larger than absolute mated-score part of the TPIR@FPIR=0.3 change",
                             detail=f"{int(h6.criterion_met.sum())}/{len(h6)} cells (both {int(h6[h6['mode'] == 'both'].criterion_met.sum())}/{int((h6['mode'] == 'both').sum())}, probes {int(h6[h6['mode'] == 'probes'].criterion_met.sum())}/{int((h6['mode'] == 'probes').sum())})",
                             verdict=verdict(int(h6.criterion_met.sum()), len(h6)), bonferroni_detail="see h6_threshold_split.csv"))
    verdicts.append(dict(hypothesis="H7 quality inflation", criterion="deferred (needs re-embedding)", detail="not run", verdict="deferred", bonferroni_detail=""))
    vdf = pd.DataFrame(verdicts)
    vdf["status"] = ["exploratory (run although H1 was supported)" if (h1_supported and v["hypothesis"][:2] in ("H2", "H3", "H5", "H6")) else "pre-registered" for v in verdicts]
    vdf.to_csv(out / "verdicts.csv", index=False)
    # ---- tables for the report
    md = ["### Common scale (E0): change (restored minus direct) with 95% CI [Bonferroni over 36 cells in brackets in the CSV]", ""]
    e0 = s1["e0"]
    for m_, ttl in (("closed_set_rank1", "Closed-set Rank-1"), ("verification_auc", "Verification AUC"), ("verification_dprime", "Verification d'")):
        md += [f"**{ttl}**", "", "| Variant | Recognizer | Mode | TinyFace direct -> restored | TinyFace change | SurvFace direct -> restored | SurvFace change | Gap (TinyFace - SurvFace) | CI contains 0 |", "|---|---|---|---|---|---|---|---|---|"]
        g = gap[gap.metric == m_].reset_index(drop=True)
        for _, r in g.iterrows():
            t = e0[(e0.arm == r.arm) & (e0.setting == r.setting) & (e0.recognizer == r.recognizer) & (e0["mode"] == r["mode"]) & (e0.metric == m_) & (e0.dataset == "tinyface")].iloc[0]
            s = e0[(e0.arm == r.arm) & (e0.setting == r.setting) & (e0.recognizer == r.recognizer) & (e0["mode"] == r["mode"]) & (e0.metric == m_) & (e0.dataset == "survface")].iloc[0]
            md.append(f"| {r.arm}: {r.setting} | {r.recognizer} | {r['mode']} | {t.direct:.3f} -> {t.restored:.3f} | {fmt(t.delta, t.delta_ci_lo, t.delta_ci_hi)} | {s.direct:.3f} -> {s.restored:.3f} | "
                      f"{fmt(s.delta, s.delta_ci_lo, s.delta_ci_hi)} | {fmt(r.gap, r.gap_ci_lo, r.gap_ci_hi)} | {'yes' if r.ci_contains_zero else 'no'} |")
        md.append("")
    (out / "asym_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(pd.DataFrame(verdicts)[["hypothesis", "detail", "verdict"]].to_string(), flush=True)
    print("full run" if full else "PARTIAL RUN (test only)", "->", out)


if __name__ == "__main__":
    main()
