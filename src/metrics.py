"""Bootstrap CIs, embedding drift, impostor-score shift, genuine/impostor
separation and paired per-sample deltas. All CIs are percentile bootstrap.

Protocol metrics are recomputed from per-sample scores (eval_protocols.*_from_scores);
resampling is stratified: each named group (e.g. mated / unmated queries,
positive / negative pairs) is resampled independently with replacement.
"""
import numpy as np

from . import eval_protocols as ev


def bootstrap(fn, groups, n=1000, seed=42, alpha=0.05):
    """fn(**resampled_groups) -> dict of scalars. groups: {name: array (first axis = samples)}.
    Returns {metric: [lo, hi]}."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        draws.append(fn(**{k: v[rng.integers(0, len(v), len(v))] for k, v in groups.items()}))
    keys = draws[0].keys()
    return {k: [float(np.quantile([d[k] for d in draws], alpha / 2)), float(np.quantile([d[k] for d in draws], 1 - alpha / 2))] for k in keys}


def protocol_ci(protocol, ps, n=1000, seed=42):
    """Bootstrap CIs for the official metrics of one protocol from its per-sample scores."""
    if protocol == "tinyface":
        # first_rank and ap belong to the same probe: resample probes jointly
        return _paired_groups_ci(lambda idx: ev.tinyface_from_scores(ps["first_rank"][idx], ps["ap"][idx]), len(ps["ap"]), n, seed)
    if protocol == "survface_openset":
        return bootstrap(lambda s1, gt: ev.survface_openset_from_scores(s1, gt), {"s1": ps["s1"], "gt": ps["gt"]}, n, seed)
    if protocol == "survface_verification":
        return bootstrap(lambda pos, neg: ev.survface_verification_from_scores(pos, neg),
                         {"pos": ps["pos_dist"], "neg": ps["neg_dist"]}, n, seed)
    raise ValueError(protocol)


def _paired_groups_ci(fn_idx, n_samples, n=1000, seed=42, alpha=0.05):
    """Bootstrap where several per-sample arrays must be resampled with the SAME indices."""
    rng = np.random.default_rng(seed)
    draws = [fn_idx(rng.integers(0, n_samples, n_samples)) for _ in range(n)]
    return {k: [float(np.quantile([d[k] for d in draws], alpha / 2)), float(np.quantile([d[k] for d in draws], 1 - alpha / 2))] for k in draws[0]}


def mean_ci(x, n=1000, seed=42):
    x = np.asarray(x, np.float64)
    ci = _paired_groups_ci(lambda i: {"m": x[i].mean()}, len(x), n, seed)["m"]
    return {"mean": float(x.mean()), "ci95": ci}


def separation(genuine, impostor, n=1000, seed=42):
    """Genuine/impostor cosine statistics and decidability d' = |mu_g - mu_i| / sqrt((var_g + var_i)/2)."""
    g, i = np.asarray(genuine, np.float64), np.asarray(impostor, np.float64)

    def stats(g, i):
        return {"genuine_mean": g.mean(), "impostor_mean": i.mean(), "margin": g.mean() - i.mean(),
                "d_prime": (g.mean() - i.mean()) / np.sqrt((g.var() + i.var()) / 2)}

    out = {k: float(v) for k, v in stats(g, i).items()}
    out.update(genuine_std=float(g.std()), impostor_std=float(i.std()))
    ci = bootstrap(lambda g, i: stats(g, i), {"g": g, "i": i}, n, seed)
    out.update({f"{k}_ci95": v for k, v in ci.items()})
    return out


def embedding_drift(e_ref, e_var, n=1000, seed=42):
    """Per-crop cosine between direct-feed (ref) and variant embeddings of the same crops."""
    cos = (ev.l2n(e_ref) * ev.l2n(e_var)).sum(1)
    out = mean_ci(cos, n, seed)
    out.update(median=float(np.median(cos)), p05=float(np.quantile(cos, 0.05)))
    return out, cos


def paired_delta(x_ref, x_var, n=1000, seed=42):
    """Mean of per-sample (variant - ref) on the same samples, with bootstrap CI."""
    d = np.asarray(x_var, np.float64) - np.asarray(x_ref, np.float64)
    out = mean_ci(d, n, seed)
    out.update(n=len(d), frac_improved=float((d > 0).mean()), frac_worse=float((d < 0).mean()))
    return out


def impostor_shift(imp_ref, imp_var, n=1000, seed=42):
    """Shift of impostor scores (same queries) variant vs ref. Positive = impostors look more similar."""
    return paired_delta(imp_ref, imp_var, n, seed)
