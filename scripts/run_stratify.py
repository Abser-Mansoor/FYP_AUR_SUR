"""Phase 2 (Task 4): stratify every crop by luminance and low-sharpness; validate the bins.

    envs/diag/Scripts/python.exe scripts/run_stratify.py --config configs/stratify.yaml [--stages measure,bin,sheets,analysis] [--force]

Outputs (paths.out): measures/<ds>_<split>.csv (raw per-crop measures), bins/<ds>_<split>.csv (bin assignments used by
all later tasks), thresholds.json, contact_sheets/*.png, analysis/*.csv|json|png, run.log.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import datasets  # noqa: E402
from src.recognizer import Recognizer  # noqa: E402
from src.stratify import analysis as an  # noqa: E402
from src.stratify import binning, sheets  # noqa: E402
from src.stratify.measure_split import measure_split  # noqa: E402
from src.stratify.measures import crop_measures  # noqa: E402
from src.utils import seed_all  # noqa: E402

log = logging.getLogger("stratify")


def split_data(cfg):
    out = {}
    tf = datasets.tinyface(ROOT / cfg["paths"]["tinyface"])
    for s, (p, ids) in tf.items():
        out[("tinyface", s)] = (p, ids)
    sf = datasets.survface_identification(ROOT / cfg["paths"]["survface"])
    for s, (p, ids) in sf.items():
        out[("survface", s)] = (p, ids)
    vpaths, pairs = datasets.survface_verification(ROOT / cfg["paths"]["survface"])
    out[("survface", "verification")] = (vpaths, None)
    return out, pairs


def selftest():
    import cv2
    rng = np.random.default_rng(0)
    im = (rng.random((40, 36, 3)) * 255).astype(np.uint8)
    base = crop_measures(im)
    blur = crop_measures(cv2.GaussianBlur(im, (7, 7), 2.0))
    dark = crop_measures((im * 0.3).astype(np.uint8))
    assert blur["lap_var"] < base["lap_var"] and blur["fft_hf"] < base["fft_hf"], (base, blur)
    assert dark["luminance"] < base["luminance"], (base, dark)
    d = pd.DataFrame({"luminance": rng.random(1003), "lap_var": rng.random(1003), "fft_hf": rng.random(1003),
                      "magface_norm": np.nan, "min_side": rng.integers(10, 50, 1003), "error": ""})
    b = binning.assign_bins(d, dict(dark_quantile=0.2, low_sharp_quantile=0.2, sharp_quantile=0.2, n_quantiles=5, dark_quantile_strict=0.1, size_groups=3))
    assert (b["lum_bin"] == "dark").sum() == int(0.2 * 1003) and (b["sharp_bin"] == "low_sharpness").sum() == int(0.2 * 1003)
    assert b.loc[b["lum_bin"] == "dark", "luminance"].max() <= b.loc[b["lum_bin"] == "normal", "luminance"].min()
    assert (b["mag_q"] == -1).all()
    log.info("selftest passed: blur lowers lap_var/fft_hf, darkening lowers luminance, quantile bin sizes exact")


# ------------------------------------------------------------------ stages
def stage_measure(cfg, data, out, force):
    mdir = out / "measures"
    mdir.mkdir(exist_ok=True)
    mag_splits = {tuple(x) for x in cfg["magface_norm_splits"]}
    mag = None
    for ds, split in map(tuple, cfg["splits"]):
        f = mdir / f"{ds}_{split}.csv"
        if f.exists() and not force:
            log.info(f"measure {ds}/{split}: cached")
            continue
        log.info(f"measure {ds}/{split}: {len(data[(ds, split)][0])} crops")
        if (ds, split) in mag_splits and mag is None:
            mag = Recognizer("magface", ROOT / cfg["recognizers"]["magface"], (112, 112), "bilinear", False, cfg["batch_size"], cfg["device"])
        paths, ids = data[(ds, split)]
        df = measure_split(paths, ids, cfg, mag if (ds, split) in mag_splits else None)
        df.to_csv(f, index=False)
    del mag
    torch.cuda.empty_cache()


def load_measures(cfg, out):
    return {(ds, s): pd.read_csv(out / "measures" / f"{ds}_{s}.csv", keep_default_na=False, na_values=[""], dtype={"error": str}).fillna({"error": ""})
            for ds, s in map(tuple, cfg["splits"])}


def stage_bin(cfg, out):
    ms = load_measures(cfg, out)
    bdir = out / "bins"
    bdir.mkdir(exist_ok=True)
    bc = cfg["bins"]
    bins = {}
    if bc["threshold_scope"] == "split":
        for k, d in ms.items():
            bins[k] = binning.assign_bins(d, bc)
    else:  # dataset-pooled quantiles
        for ds in {k[0] for k in ms}:
            keys = [k for k in ms if k[0] == ds]
            cat = pd.concat([ms[k].assign(_split=k[1]) for k in keys], ignore_index=True)
            cat = binning.assign_bins(cat, bc)
            for k in keys:
                bins[k] = cat[cat["_split"] == k[1]].drop(columns="_split").reset_index(drop=True)
    thr = {}
    for (ds, s), d in bins.items():
        d.to_csv(bdir / f"{ds}_{s}.csv", index=False)
        thr[f"{ds}/{s}"] = binning.thresholds(d)
        t = thr[f"{ds}/{s}"]
        log.info(f"bins {ds}/{s}: n={t['n']} dark={t['lum_dark']['n']} (lum<={t['lum_dark']['luminance_range'][1]}) "
                 f"low_sharpness={t['sharp_low_sharpness']['n']} (lap_var<={t['sharp_low_sharpness']['lap_var_range'][1]}) "
                 f"unreadable={t['n_unreadable']}")
    (out / "thresholds.json").write_text(json.dumps({"threshold_scope": bc["threshold_scope"], "bins": bc, "splits": thr}, indent=2))
    return bins


def load_bins(cfg, out):
    return {(ds, s): pd.read_csv(out / "bins" / f"{ds}_{s}.csv", keep_default_na=False, na_values=[""], dtype={"error": str, "lum_bin": str, "sharp_bin": str})
            .fillna({"error": "", "lum_bin": "", "sharp_bin": ""}) for ds, s in map(tuple, cfg["splits"])}


def stage_sheets(cfg, out, bins):
    sdir = out / "contact_sheets"
    sdir.mkdir(exist_ok=True)
    n, seed = cfg["validation"]["contact_sheet_n"], cfg["seed"]
    q = cfg["bins"]["low_sharp_quantile"]
    specs = [("lap_var", "low", "lap_var_LOW", "{:.0f}"), ("lap_var", "high", "lap_var_HIGH", "{:.0f}"),
             ("fft_hf", "low", "fft_hf_LOW", "{:.3f}"), ("fft_hf", "high", "fft_hf_HIGH", "{:.3f}"),
             ("luminance", "low", "luminance_DARK", "{:.0f}"), ("luminance", "high", "luminance_BRIGHT", "{:.0f}"),
             ("magface_norm", "low", "magface_norm_LOW", "{:.1f}"), ("magface_norm", "high", "magface_norm_HIGH", "{:.1f}")]
    for (ds, s), d in bins.items():
        for col, end, tag, fmt in specs:
            if d[col].notna().sum() < 2 * n:
                continue
            img = sheets.contact_sheet(d, col, end, q, n, seed, fmt)
            sheets.save_png(sdir / f"{ds}_{s}__{tag}.png", img)
        log.info(f"contact sheets: {ds}/{s}")


def stage_analysis(cfg, out, bins, pairs):
    adir = out / "analysis"
    adir.mkdir(exist_ok=True)
    thr = json.loads((out / "thresholds.json").read_text())["splits"]
    ps_dir = ROOT / cfg["paths"]["phase1"]
    recs = list(cfg["recognizers"])
    ctx = an.Context(bins, pairs, ps_dir)
    # order guards: per-sample arrays must align with the bin tables
    ps = ctx.per_sample(recs[0], "tinyface"), ctx.per_sample(recs[0], "survface_openset"), ctx.per_sample(recs[0], "survface_verification")
    assert len(ps[0]["ap"]) == len(bins[("tinyface", "probe")])
    assert len(ps[1]["gt"]) == len(bins[("survface", "mated_probe")]) and len(ps[1]["s1"]) == len(bins[("survface", "unmated_probe")])
    assert len(ps[2]["pos_dist"]) == len(pairs["pos"]) and len(ps[2]["neg_dist"]) == len(pairs["neg"])

    ms = an.metrics_by_slice(ctx, recs, cfg, log)
    ms.to_csv(adir / "metrics_by_slice.csv", index=False)
    an.random_baseline(ctx, recs, cfg, log).to_csv(adir / "random_baseline.csv", index=False)
    an.agreement(bins, cfg["bins"]["low_sharp_quantile"]).to_csv(adir / "agreement.csv", index=False)
    an.confounders(bins).to_csv(adir / "confounders.csv", index=False)
    an.within_size(ctx, recs, cfg, log).to_csv(adir / "within_size.csv", index=False)
    flags = an.dark_flags(ms, thr, cfg["validation"])
    (adir / "dark_flags.json").write_text(json.dumps(flags, indent=2))
    plot_trends(ms, adir)
    log.info("analysis complete")


def plot_trends(ms, adir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    heads = {"tinyface": "Rank-1", "survface_openset": "TPIR20@FPIR=0.3", "survface_verification": "AUC"}
    for nm, title in (("sharp", "Laplacian-variance quintile (Q1 = lowest sharpness)"), ("fft", "FFT high-frequency quintile"),
                      ("lum", "Luminance quintile (Q1 = darkest)"), ("size", "Native min-side quintile (Q1 = smallest)"),
                      ("mag", "MagFace feature-norm quintile")):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (proto, h) in zip(axs, heads.items()):
            for rec, g in ms[(ms["protocol"] == proto) & ms["slice"].str.startswith(f"{nm}_Q") & (ms["status"] == "OK")].groupby("recognizer"):
                g = g.sort_values("slice")
                x = g["slice"].str[-1].astype(int)
                ax.errorbar(x, g[h], yerr=[g[h] - g[f"{h}_lo"], g[f"{h}_hi"] - g[h]], marker="o", capsize=2, label=rec)
            ax.set_title(f"{proto}: {h}")
            ax.set_xlabel("quintile")
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(adir / f"trend_{nm}.png", dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stratify.yaml")
    ap.add_argument("--stages", default="selftest,measure,bin,sheets,analysis")
    ap.add_argument("--force", action="store_true", help="recompute the measures")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    out = ROOT / cfg["paths"]["out"]
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler(sys.stdout)])
    seed_all(cfg["seed"], True, True)
    stages = args.stages.split(",")
    log.info(f"Phase 2 stratification | config {args.config} | stages {stages}")
    if "selftest" in stages:
        selftest()
    data, pairs = split_data(cfg)
    if "measure" in stages:
        stage_measure(cfg, data, out, args.force)
    if "bin" in stages:
        stage_bin(cfg, out)
    bins = load_bins(cfg, out) if {"sheets", "analysis"} & set(stages) else None
    if "sheets" in stages:
        stage_sheets(cfg, out, bins)
    if "analysis" in stages:
        stage_analysis(cfg, out, bins, pairs)
    log.info("done")


if __name__ == "__main__":
    main()
