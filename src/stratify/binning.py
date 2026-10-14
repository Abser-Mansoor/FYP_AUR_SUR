"""Quantile bin assignment. Quantiles of the OBSERVED distribution, never fixed thresholds.

Rank-based: a stable argsort (ties broken by dataset order) puts exactly floor(q*n) crops into the
extreme bin, so bin sizes are exact and reruns are identical. NaN measures never enter a bin."""
import numpy as np
import pandas as pd


def ranks(x):
    """Integer rank (0 = smallest, stable) among finite values, -1 for NaN; also returns n_finite."""
    x = np.asarray(x, np.float64)
    out = np.full(len(x), -1, np.int64)
    ok = np.flatnonzero(np.isfinite(x))
    out[ok[np.argsort(x[ok], kind="stable")]] = np.arange(len(ok))
    return out, len(ok)


def _rank01(x):
    """Rank in [0,1) among finite values (NaN -> NaN); used for ordering only."""
    r, n = ranks(x)
    return np.where(r < 0, np.nan, r / max(n, 1))


def low_mask(x, q):
    r, n = ranks(x)
    return (r >= 0) & (r < int(q * n))


def top_mask(x, q):
    r, n = ranks(x)
    return (r >= 0) & (r >= n - int(q * n))


def _quantile_index(x, nq):
    r, n = ranks(x)
    return np.where(r < 0, -1, (r * nq) // max(n, 1)).astype(int)  # 0 = lowest, -1 = no value


def assign_bins(df, bcfg):
    """Adds lum_bin, sharp_bin, lum_q, sharp_q, fft_q, mag_q, size_q columns to a copy of df."""
    d = df.copy()
    lum_ok, lap_ok = np.isfinite(d["luminance"]), np.isfinite(d["lap_var"])
    d["lum_bin"] = np.where(~lum_ok, "", np.where(low_mask(d["luminance"], bcfg["dark_quantile"]), "dark", "normal"))
    d["sharp_bin"] = np.where(~lap_ok, "", np.where(low_mask(d["lap_var"], bcfg["low_sharp_quantile"]), "low_sharpness",
                                                   np.where(top_mask(d["lap_var"], bcfg["sharp_quantile"]), "sharp", "mid")))
    nq = bcfg["n_quantiles"]
    d["lum_q"] = _quantile_index(d["luminance"], nq)
    d["sharp_q"] = _quantile_index(d["lap_var"], nq)
    d["fft_q"] = _quantile_index(d["fft_hf"], nq)
    d["mag_q"] = _quantile_index(d["magface_norm"], nq)
    # size has heavy ties (most crops share one size): value-based quantiles, duplicate edges merged
    try:
        q = pd.qcut(d["min_side"], nq, labels=False, duplicates="drop")
    except ValueError:
        q = pd.Series(0, index=d.index)
    d["size_q"] = q.fillna(-1).astype(int)
    # strict dark bin and size-controlled sharpness (rank of lap_var inside native-size groups)
    d["lum_bin10"] = np.where(low_mask(d["luminance"], bcfg["dark_quantile_strict"]), "dark10", "")
    try:
        g = pd.qcut(d["min_side"], bcfg["size_groups"], labels=False, duplicates="drop").fillna(-1).astype(int).to_numpy()
    except ValueError:
        g = np.zeros(len(d), int)
    d["size_group"] = g
    lv, sb = d["lap_var"].to_numpy(), np.full(len(d), "", dtype=object)
    for k in np.unique(g[g >= 0]):
        sel = np.flatnonzero(g == k)
        sub = sb[sel]
        sub[:] = "mid"
        sub[low_mask(lv[sel], bcfg["low_sharp_quantile"])] = "low_sharpness"
        sub[top_mask(lv[sel], bcfg["sharp_quantile"])] = "sharp"
        sb[sel] = sub
    d["sharp_bin_sz"] = sb
    return d


def thresholds(d):
    """Value bounds of each bin, for the report."""
    def rng(mask, col):
        v = d.loc[mask, col]
        return [float(v.min()), float(v.max())] if len(v) else [None, None]
    out = {"n": int(len(d)), "n_unreadable": int((d["error"] != "").sum())}
    for b in ("dark", "normal"):
        m = d["lum_bin"] == b
        out[f"lum_{b}"] = {"n": int(m.sum()), "luminance_range": rng(m, "luminance")}
    for b in ("low_sharpness", "mid", "sharp"):
        m = d["sharp_bin"] == b
        out[f"sharp_{b}"] = {"n": int(m.sum()), "lap_var_range": rng(m, "lap_var")}
    return out
