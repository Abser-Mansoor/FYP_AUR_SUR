"""Phase 8 Part C: frame-quality reject test (protocol: REPORT_phase8.md section 0.5, committed before any number). CPU only, from cache.

    envs/video/Scripts/python.exe scripts/mevid_quality.py --dataset mevid          # primary: MEVID eligible tracklets of enrolled persons, closed-set R1, gallery bar 48 px
    envs/video/Scripts/python.exe scripts/mevid_quality.py --dataset chokepoint     # secondary: ChokePoint GT tracks at 12 and 16 px (SC, both directions pooled)
    envs/video/Scripts/python.exe scripts/mevid_quality.py --selftest

Per tracklet: a quality signal (higher = more reliable) and the closed-set R1 outcome. Curve: the error rate of the kept tracklets when the lowest-quality fraction q (0 to 0.5 in steps of
0.01; ties broken by tracklet order) is rejected; AUC = the area under that curve over [0, 0.5] (trapezoid). Random rejection leaves the expected error unchanged, so its area is the overall
error rate times 0.5. A signal helps if its area is below the random area and the paired cluster-bootstrap CI over persons (1,000 draws, seed 42), Bonferroni-adjusted over the signals of the
dataset, excludes zero. Luminance is scored as the plan lists it (mean luminance, higher = better); the variant -|mean luminance - 128| is an exploratory extra row outside the Bonferroni count and the verdict. On ChokePoint face height, Laplacian variance, luminance and the AdaFace raw norm are not available for the degraded
faces (they keep the native detection or were not cached), so only the signals that exist there are evaluated. Outputs: outputs/video/mevid/results/quality_*.csv.
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
OUT = ROOT / "outputs/video/mevid"
SEED, N_BOOT = 42, 1000
GRID = np.round(np.arange(0, 0.5 + 1e-9, 0.01), 2)


def self_consistency(emb):
    """Mean cosine of each face to the leave-one-out mean of the tracklet's other faces (embeddings L2-normalised)."""
    s = emb.sum(0)
    loo = s - emb
    loo = loo / (np.linalg.norm(loo, axis=1, keepdims=True) + 1e-12)
    return float((emb * loo).sum(1).mean())


def curve(err, quality, order_key):
    """Error of the kept tracklets for each rejection fraction in GRID; ties in quality are broken by `order_key`."""
    o = np.lexsort((order_key, quality))  # ascending quality, then tracklet order
    e = err[o]
    n = len(e)
    csum_from_top = np.cumsum(e[::-1])  # error counts of the best k tracklets
    out = np.empty(len(GRID))
    for i, q in enumerate(GRID):
        k = n - int(np.floor(q * n))
        out[i] = csum_from_top[k - 1] / k
    return out


def area(c):
    return float(np.trapezoid(c, GRID)) if hasattr(np, "trapezoid") else float(np.trapz(c, GRID))


def reject_table(df, signals, label, exploratory=()):
    """df: person, key (tracklet order), err (1 = wrong), one column per signal (higher = better). -> (table, curves)."""
    err = df.err.to_numpy(float)
    key = df.key.to_numpy()
    pid = df.person.to_numpy()
    persons = np.unique(pid)
    where = {p: np.flatnonzero(pid == p) for p in persons}
    rng = np.random.default_rng(SEED)
    exploratory = list(exploratory)
    signals = list(signals) + exploratory  # exploratory signals are evaluated and reported but excluded from the Bonferroni count and from the verdict
    draws = {s: [] for s in signals}
    for _ in range(N_BOOT):
        idx = np.concatenate([where[p] for p in persons[rng.integers(0, len(persons), len(persons))]])
        e, k = err[idx], key[idx]
        rand_area = 0.5 * e.mean()
        for s in signals:
            draws[s].append(area(curve(e, df[s].to_numpy(float)[idx], k)) - rand_area)
    rows, curves = [], []
    m = len(signals) - len(exploratory)
    for s in signals:
        c = curve(err, df[s].to_numpy(float), key)
        d = np.array(draws[s])
        rand = 0.5 * err.mean()
        rows.append(dict(dataset=label, signal=s, n_tracklets=len(df), n_persons=len(persons), overall_error=float(err.mean()), auc=area(c), auc_random=float(rand), auc_minus_random=area(c) - rand,
                         ci_lo=float(np.quantile(d, 0.025)), ci_hi=float(np.quantile(d, 0.975)), bonf_lo=float(np.quantile(d, 0.025 / m)), bonf_hi=float(np.quantile(d, 1 - 0.025 / m)),
                         exploratory=bool(s in exploratory), helps=(None if s in exploratory else bool(np.quantile(d, 1 - 0.025 / m) < 0))))
        curves += [dict(dataset=label, signal=s, reject_fraction=float(q), error=float(v)) for q, v in zip(GRID, c)]
    return pd.DataFrame(rows), pd.DataFrame(curves)


# ----------------------------------------------------------------------------- MEVID
def mevid():
    from mevid_openset import closed_records, gallery_templates, load_real
    d = load_real("adaface")
    ids, T = gallery_templates(d, 48)
    rec = closed_records(d, ids, T, ["R1"]).set_index("tracklet")
    rows = []
    for t, r in rec.iterrows():
        f = d.frames[t]
        z = np.load(OUT / "emb" / f"emb_{t}.npz")
        rows.append(dict(tracklet=t, person=r.person, err=1.0 - r.ok_R1, n_usable_faces=float(len(f["emb"])), self_consistency=self_consistency(f["emb"]), mean_face_height=float(np.mean(z["face_h"])),
                         mean_laplacian_variance=float(np.mean(z["lap_var"])), mean_luminance=float(np.mean(z["lum"])), luminance_closeness_to_midgray=-abs(float(np.mean(z["lum"])) - 128.0), mean_adaface_raw_norm=float(np.mean(z["adaface_norm"])),
                         mean_magface_norm=float(np.mean(z["mag_norm"]))))
    df = pd.DataFrame(rows)
    df["key"] = np.arange(len(df))
    sig = [c for c in df.columns if c not in ("tracklet", "person", "err", "key", "luminance_closeness_to_midgray")]
    return reject_table(df, sig, "MEVID (gallery bar 48 px)", exploratory=["luminance_closeness_to_midgray"])


# ----------------------------------------------------------------------------- ChokePoint
def chokepoint():
    import video_aggregation as VA
    import video_phase7_bench as B7
    import video_single_frame as S
    cfg, allc, m, lo = B7.setup()
    sub = {c: m[m["clip"] == c].sort_values(["frame", "det_id"]).iloc[::5] for c in allc}
    rows, box = [], {}
    for size in (16, 12):
        for k, (g, p) in enumerate((("G1", "G2"), ("G2", "G1"))):
            gal = S.build("adaface", S.clips_of(g, "SC", allc), m, cfg, box)
            ids, T = VA.gallery(*gal)
            fr = VA.frames_of(S.clips_of(p, "SC", allc), m, size, sub)
            tracks = VA.gt_tracks(fr)
            rec = VA.track_records(fr, tracks, ids, T, ["R1"])
            enrolled = set(ids.tolist())
            ix_by = {(fr["clip"][ix[0]], int(fr["pid"][ix[0]])): ix for ix in tracks}
            for r in rec.itertuples():
                ix = ix_by[(r.clip, r.pid)]
                rows.append(dict(size=size, direction=k, person=r.pid, err=1.0 - r.ok_R1, n_usable_faces=float(len(ix)), self_consistency=self_consistency(fr["emb"][ix]), mean_magface_norm=float(fr["q"][ix].mean())))
    df = pd.DataFrame(rows)
    out_t, out_c = [], []
    for size, g in df.groupby("size"):
        g = g.reset_index(drop=True)
        g["key"] = np.arange(len(g))
        t, c = reject_table(g, ["n_usable_faces", "self_consistency", "mean_magface_norm"], f"ChokePoint {size} px (SC, GT tracks, both directions pooled)")
        out_t.append(t)
        out_c.append(c)
    return pd.concat(out_t, ignore_index=True), pd.concat(out_c, ignore_index=True)


# ----------------------------------------------------------------------------- self-test
def selftest():
    rng = np.random.default_rng(0)
    n = 400
    good = rng.normal(size=n)  # a genuinely informative signal
    err = (rng.random(n) < 1 / (1 + np.exp(2.0 * good + 1.0))).astype(float)
    df = pd.DataFrame(dict(person=np.repeat(np.arange(n // 4), 4), key=np.arange(n), err=err, informative=good, noise=rng.normal(size=n)))
    t, c = reject_table(df, ["informative", "noise"], "synthetic")
    r = t.set_index("signal")
    checks = dict(informative_signal_helps=bool(r.loc["informative", "helps"]), noise_signal_does_not_help=bool(not r.loc["noise", "helps"]), curve_starts_at_overall_error=bool(np.isclose(c[(c.signal == "noise")].error.iloc[0], err.mean())),
                  random_area_is_half_overall_error=bool(np.isclose(r.loc["noise", "auc_random"], 0.5 * err.mean())))
    print(t.round(4).to_string(index=False))
    print(json.dumps(checks, indent=1))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mevid", "chokepoint"], default="mevid")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    t, c = {"mevid": mevid, "chokepoint": chokepoint}[a.dataset]()
    res = OUT / "results"
    res.mkdir(parents=True, exist_ok=True)
    t.to_csv(res / f"quality_{a.dataset}.csv", index=False, float_format="%.5f")
    c.to_csv(res / f"quality_{a.dataset}_curves.csv", index=False, float_format="%.5f")
    print(t.round(4).to_string(index=False))
