"""Validation of the bins against Phase 1 direct-feed per-sample scores (no re-embedding).

Slices are defined per crop; query-level protocols restrict queries to the slice, the gallery is
unchanged. SurvFace verification pairs use the WORSE image: 'any' rule for low-end slices (either
image in the slice), 'all' rule for normal/high-end slices, 'minq' (lower quantile of the two) for
quantile slices."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from .. import eval_protocols as ev, metrics
from .binning import low_mask, top_mask

NQ = 5
HEADLINE = {"tinyface": ("Rank-1", "mAP"), "survface_openset": ("TPIR20@FPIR=0.3", "TPIR20@FPIR=0.1"),
            "survface_verification": ("AUC", "TAR@FAR=0.1")}
EXTRA = {"tinyface": ("Rank-5", "Rank-10"), "survface_openset": ("TPIR20@FPIR=0.2", "AUC"),
         "survface_verification": ("TAR@FAR=0.3", "TAR@FAR=0.01")}
PROTOCOLS = ("tinyface", "survface_openset", "survface_verification")


# ------------------------------------------------------------------ slices
def slice_specs(have_mag):
    """(name, column, values, rule)."""
    sp = [("full", None, None, "all"),
          ("dark", "lum_bin", ["dark"], "any"), ("normal_lum", "lum_bin", ["normal"], "all"),
          ("low_sharpness", "sharp_bin", ["low_sharpness"], "any"), ("sharp", "sharp_bin", ["sharp"], "all"),
          ("not_low_sharpness", "sharp_bin", ["mid", "sharp"], "all"),
          ("fft_low", "fft_q", [0], "any"),
          ("dark10", "lum_bin10", ["dark10"], "any"),
          ("low_sharpness_sz", "sharp_bin_sz", ["low_sharpness"], "any"), ("sharp_sz", "sharp_bin_sz", ["sharp"], "all"),
          ("not_low_sharpness_sz", "sharp_bin_sz", ["mid", "sharp"], "all")]
    if have_mag:
        sp.append(("mag_low", "mag_q", [0], "any"))
    for col, nm in (("lum_q", "lum"), ("sharp_q", "sharp"), ("fft_q", "fft"), ("size_q", "size")):
        sp += [(f"{nm}_Q{k + 1}", col, [k], "minq") for k in range(NQ)]
    if have_mag:
        sp += [(f"mag_Q{k + 1}", "mag_q", [k], "minq") for k in range(NQ)]
    return sp


def _flag(df, col, values):
    return np.ones(len(df), bool) if col is None else df[col].isin(values).to_numpy()


def _pair_flag(da, db, col, values, rule):
    if col is None:
        return np.ones(len(da), bool)
    if rule == "minq":
        return np.minimum(da[col].to_numpy(), db[col].to_numpy()) == values[0]
    fa, fb = da[col].isin(values).to_numpy(), db[col].isin(values).to_numpy()
    return (fa | fb) if rule == "any" else (fa & fb)


class Context:
    """Per-split bin tables + Phase 1 per-sample arrays, aligned to dataset order."""

    def __init__(self, bins, pairs, ps_dir):
        self.b, self.pairs, self.ps_dir = bins, pairs, ps_dir
        vdf = bins[("survface", "verification")]
        idx = {n: i for i, n in enumerate(vdf["name"])}
        self.pair_idx = {k: (np.array([idx[a] for a, _ in pairs[k]]), np.array([idx[b] for _, b in pairs[k]])) for k in ("pos", "neg")}
        self._ps = {}

    def per_sample(self, rec, protocol):
        if (rec, protocol) not in self._ps:
            self._ps[(rec, protocol)] = dict(np.load(self.ps_dir / f"{rec}_{protocol}_direct.npz"))
        return self._ps[(rec, protocol)]

    def have_mag(self, protocol):
        key = {"tinyface": ("tinyface", "probe"), "survface_openset": ("survface", "mated_probe"),
               "survface_verification": ("survface", "verification")}[protocol]
        return bool(self.b[key]["mag_q"].max() >= 0)

    def masks(self, protocol, spec):
        _, col, values, rule = spec
        if protocol == "tinyface":
            return {"p": _flag(self.b[("tinyface", "probe")], col, values)}
        if protocol == "survface_openset":
            return {"m": _flag(self.b[("survface", "mated_probe")], col, values), "u": _flag(self.b[("survface", "unmated_probe")], col, values)}
        v = self.b[("survface", "verification")]
        out = {}
        for k in ("pos", "neg"):
            ia, ib = self.pair_idx[k]
            out[k] = _pair_flag(v.iloc[ia], v.iloc[ib], col, values, rule)
        return out


# ----------------------------------------------------------------- metrics
def _metrics(protocol, ps, m):
    if protocol == "tinyface":
        return ev.tinyface_from_scores(ps["first_rank"][m["p"]], ps["ap"][m["p"]])
    if protocol == "survface_openset":
        return ev.survface_openset_from_scores(ps["s1"][m["u"]], ps["gt"][m["m"]])
    return ev.survface_verification_from_scores(ps["pos_dist"][m["pos"]], ps["neg_dist"][m["neg"]], accuracy=False)


def _counts(m):
    return {k: int(v.sum()) for k, v in m.items()}


def _ok(m):
    return all(v >= 20 for v in _counts(m).values())


def _ci(protocol, ps, m, n, seed):
    if protocol == "tinyface":
        fr, ap = ps["first_rank"][m["p"]], ps["ap"][m["p"]]
        return metrics._paired_groups_ci(lambda i: ev.tinyface_from_scores(fr[i], ap[i]), len(fr), n, seed)
    if protocol == "survface_openset":
        return metrics.bootstrap(lambda s1, gt: ev.survface_openset_from_scores(s1, gt),
                                 {"s1": ps["s1"][m["u"]], "gt": ps["gt"][m["m"]]}, n, seed)
    return metrics.bootstrap(lambda pos, neg: ev.survface_verification_from_scores(pos, neg, accuracy=False),
                             {"pos": ps["pos_dist"][m["pos"]], "neg": ps["neg_dist"][m["neg"]]}, n, seed)


def _row_metrics(protocol, ps, m, cfg):
    n, seed = cfg["validation"]["bootstrap_n"], cfg["validation"]["bootstrap_seed"]
    pt, ci = _metrics(protocol, ps, m), _ci(protocol, ps, m, n, seed)
    row = {}
    for k in HEADLINE[protocol] + EXTRA[protocol]:
        row[k] = pt[k]
        row[f"{k}_lo"], row[f"{k}_hi"] = ci[k]
    return row


def metrics_by_slice(ctx, recs, cfg, log):
    rows = []
    for rec in recs:
        for protocol in PROTOCOLS:
            ps = ctx.per_sample(rec, protocol)
            for spec in slice_specs(ctx.have_mag(protocol)):
                m = ctx.masks(protocol, spec)
                row = dict(recognizer=rec, protocol=protocol, slice=spec[0], **{f"n_{k}": v for k, v in _counts(m).items()})
                if not _ok(m):
                    row["status"] = "TOO_SMALL (<20 per group)"
                else:
                    row["status"] = "OK"
                    row.update(_row_metrics(protocol, ps, m, cfg))
                rows.append(row)
            log.info(f"  slices done: {rec}/{protocol}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------- random baseline
def random_baseline(ctx, recs, cfg, log, slices=("dark", "dark10", "low_sharpness", "low_sharpness_sz", "fft_low", "mag_low")):
    draws, seed = cfg["validation"]["random_baseline_draws"], cfg["validation"]["bootstrap_seed"]
    rows = []
    for rec in recs:
        for protocol in PROTOCOLS:
            ps = ctx.per_sample(rec, protocol)
            head = HEADLINE[protocol]
            for sname in slices:
                spec = next((s for s in slice_specs(True) if s[0] == sname), None)
                if sname == "mag_low" and not ctx.have_mag(protocol):
                    continue
                m = ctx.masks(protocol, spec)
                if not _ok(m):
                    continue
                rng = np.random.default_rng(seed)
                vals = {k: [] for k in head}
                for _ in range(draws):
                    rm = {}
                    for g, f in m.items():
                        r = np.zeros(len(f), bool)
                        r[rng.choice(len(f), int(f.sum()), replace=False)] = True
                        rm[g] = r
                    pt = _metrics(protocol, ps, rm)
                    for k in head:
                        vals[k].append(pt[k])
                pt = _metrics(protocol, ps, m)
                row = dict(recognizer=rec, protocol=protocol, slice=sname, draws=draws, **{f"n_{k}": v for k, v in _counts(m).items()})
                for k in head:
                    a = np.array(vals[k])
                    lo, hi = np.quantile(a, 0.025), np.quantile(a, 0.975)
                    row.update({k: pt[k], f"{k}_rand_mean": a.mean(), f"{k}_rand_lo": lo, f"{k}_rand_hi": hi,
                                f"{k}_below_random": bool(pt[k] < lo), f"{k}_above_random": bool(pt[k] > hi)})
                rows.append(row)
            log.info(f"  random baseline done: {rec}/{protocol}")
    return pd.DataFrame(rows)


# ------------------------------------------------- agreement / confounders
def agreement(bins, q):
    rows = []
    for (ds, split), d in bins.items():
        cols = [c for c in ("lap_var", "fft_hf", "magface_norm") if d[c].notna().sum() > 1]
        ok = d[cols].notna().all(axis=1)
        x = d[ok]
        low = {c: low_mask(x[c].to_numpy(), q) for c in cols}
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                inter, uni = int((low[a] & low[b]).sum()), int((low[a] | low[b]).sum())
                rows.append(dict(dataset=ds, split=split, a=a, b=b, n=int(ok.sum()), spearman=float(spearmanr(x[a], x[b])[0]),
                                 jaccard_low=inter / uni if uni else np.nan, kappa_low=float(cohen_kappa_score(low[a], low[b])),
                                 frac_a_low_also_b_low=inter / max(int(low[a].sum()), 1)))
    return pd.DataFrame(rows)


def confounders(bins):
    rows = []
    for (ds, split), d in bins.items():
        for a, b in (("lap_var", "min_side"), ("fft_hf", "min_side"), ("magface_norm", "min_side"), ("luminance", "min_side"),
                     ("lap_var", "luminance"), ("fft_hf", "luminance"), ("magface_norm", "luminance")):
            x = d[[a, b]].dropna()
            if len(x) > 2 and x[a].nunique() > 1 and x[b].nunique() > 1:
                rows.append(dict(dataset=ds, split=split, a=a, b=b, n=len(x), spearman=float(spearmanr(x[a], x[b])[0])))
    return pd.DataFrame(rows)


def within_size(ctx, recs, cfg, log):
    """Low vs top sharpness quantile within native-size tertiles (TinyFace probes, SurvFace identification queries)."""
    qlow, qtop = cfg["bins"]["low_sharp_quantile"], cfg["bins"]["sharp_quantile"]
    nt = cfg["validation"]["size_tertiles"]
    rows = []
    for protocol, splits in (("tinyface", {"p": ("tinyface", "probe")}),
                             ("survface_openset", {"m": ("survface", "mated_probe"), "u": ("survface", "unmated_probe")})):
        tert = {}
        for g, key in splits.items():
            d = ctx.b[key]
            t = pd.qcut(d["min_side"], nt, labels=False, duplicates="drop").fillna(-1).astype(int).to_numpy()
            lo, tp = np.zeros(len(d), bool), np.zeros(len(d), bool)
            lv = d["lap_var"].to_numpy()
            for k in np.unique(t[t >= 0]):
                sel = np.flatnonzero(t == k)
                lo[sel], tp[sel] = low_mask(lv[sel], qlow), top_mask(lv[sel], qtop)
            tert[g] = (t, {"low": lo, "top": tp}, d["min_side"].to_numpy())
        first = next(iter(splits))
        for k in np.unique(tert[first][0]):
            if k < 0:
                continue
            for end in ("low", "top"):
                m = {g: (t == k) & e[end] for g, (t, e, _) in tert.items()}
                if not _ok(m):
                    continue
                szs = tert[first][2][tert[first][0] == k]
                for rec in recs:
                    ps = ctx.per_sample(rec, protocol)
                    row = dict(recognizer=rec, protocol=protocol, size_tertile=int(k) + 1, min_side_range=f"{szs.min():.0f}-{szs.max():.0f}",
                               sharp_end=end, **{f"n_{g}": int(v.sum()) for g, v in m.items()})
                    pt, ci = _metrics(protocol, ps, m), _ci(protocol, ps, m, cfg["validation"]["bootstrap_n"], cfg["validation"]["bootstrap_seed"])
                    for h in HEADLINE[protocol]:
                        row[h] = pt[h]
                        row[f"{h}_lo"], row[f"{h}_hi"] = ci[h]
                    rows.append(row)
        log.info(f"  within-size done: {protocol}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- dark flags
def dark_flags(mslice, thr, vcfg):
    """Pre-registered checks that the dark bin is usable for low-light work."""
    out = {}
    for key, t in thr.items():
        d = t["lum_dark"]
        out[key] = dict(n_dark=d["n"], dark_luminance_range=d["luminance_range"], flag_small=d["n"] < vcfg["min_dark_n"],
                        flag_not_dark=(d["luminance_range"][1] or 0) > vcfg["max_dark_luminance"])
    ci = {}
    for protocol, head in (("tinyface", "Rank-1"), ("survface_openset", "TPIR20@FPIR=0.3"), ("survface_verification", "AUC")):
        s = mslice[(mslice["protocol"] == protocol) & (mslice["slice"] == "dark") & (mslice["status"] == "OK")]
        hw = (s[f"{head}_hi"] - s[f"{head}_lo"]) / 2
        ci[protocol] = dict(metric=head, max_halfwidth=float(hw.max()) if len(hw) else None,
                            flag_wide_ci=bool(len(hw) and hw.max() > vcfg["max_ci_halfwidth"]))
    out["_ci"] = ci
    return out
