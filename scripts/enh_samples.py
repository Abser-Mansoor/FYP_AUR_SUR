"""Phase 4 labelled visual samples (used by sr_sample_figs.py / sr_brief_strips.py with --config configs/enhance.yaml).

Samples are random (seed 42), identical for every arm, and never chosen by outcome. Dark examples are drawn from the Phase 2 `dark10`
bin, blur examples from the lowest `sharp_bin_sz` bin (`low_sharpness`), per dataset, among probes. Each row is labelled with the dataset,
the method and the problem seen (dark, or blurred), as in the supervisor brief appendix; the first column is the untouched input and
every crop is enlarged with nearest-neighbour so that the pixels are visible.
Outputs: <results>/../samples/{problem}_{dataset}.png and samples.csv; strips (one row per sample) in docs/figures/enh_*.png.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED, N, SHOW = 42, 3, 224
PROBLEMS = {"dark": ("lum_bin10", "dark10", "dark (darkest 10%)", ["gamma", "clahe", "zerodce"]),
            "blur": ("sharp_bin_sz", "low_sharpness", "blurred (lowest-sharpness bin)", ["unsharp", "nafnet"])}


def load(p):
    return Image.open(p).convert("RGB").resize((SHOW, SHOW), Image.NEAREST)


def selected(res, arm):
    f = res / f"selection_{arm}.json"
    return json.loads(f.read_text())[0]["setting"] if f.exists() and json.loads(f.read_text()) else None


def samples(res, arms=None):
    """-> list of (problem, dataset, k, row, [(label, image path)]) for every sampled crop."""
    df = pd.read_csv(ROOT / "outputs/sr/subset/eval_images.csv")
    out = []
    for prob, (col, val, desc, parr) in PROBLEMS.items():
        use = [a for a in parr if (arms is None or a in arms) and selected(res, a)]
        for ds, split in (("tinyface", "probe"), ("survface", "mated_probe")):
            d = df[(df.dataset == ds) & (df.split == split) & (df[col] == val)].sample(N, random_state=SEED)
            for k, (_, r) in enumerate(d.iterrows(), 1):
                cols = [("Input", ROOT / "outputs/sr/subset/eval_input" / r.file)]
                for a in use:
                    st = selected(res, a)
                    cols.append((f"{a} ({st})", ROOT / "outputs/sr/restored/eval" / f"{a}__{st}" / "seed0" / r.file))
                out.append((prob, ds, k, r, cols))
    return out


def sheets(res, arms=None):
    outd = res.parent / "samples"
    outd.mkdir(parents=True, exist_ok=True)
    smp = samples(res, arms)
    log = []
    for prob in PROBLEMS:
        for ds in ("tinyface", "survface"):
            rows = [s for s in smp if s[0] == prob and s[1] == ds]
            if not rows:
                continue
            ncol = len(rows[0][4])
            fig, ax = plt.subplots(len(rows), ncol, figsize=(2.6 * ncol, 2.8 * len(rows)))
            ax = np.atleast_2d(ax)
            for r, (_, _, k, row, cols) in enumerate(rows):
                for c, (lab, p) in enumerate(cols):
                    ax[r, c].imshow(load(p)); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
                    if r == 0:
                        ax[r, c].set_title(lab, fontsize=10)
                ax[r, 0].set_ylabel(f"{ds}\n{row['name']}\n{row.w}x{row.h} px\nproblem: {PROBLEMS[prob][2]}", fontsize=7)
                log.append((prob, ds, k, row["name"], row.h, row.w))
            fig.suptitle(f"{ds}: random {PROBLEMS[prob][2]} probes (seed {SEED}); same crops for every method", fontsize=11)
            fig.tight_layout()
            fig.savefig(outd / f"{prob}_{ds}.png", dpi=120)
            plt.close(fig)
    pd.DataFrame(log, columns=["problem", "dataset", "sample", "name", "h", "w"]).to_csv(outd / "samples.csv", index=False)
    print("wrote", outd)


def strips(res, arms=None):
    fig_dir = ROOT / "docs/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for prob, ds, k, row, cols in samples(res, arms):
        fig, ax = plt.subplots(1, len(cols), figsize=(2.0 * len(cols), 2.7))
        for a, (lab, p) in zip(np.atleast_1d(ax), cols):
            a.imshow(load(p)); a.set_xticks([]); a.set_yticks([]); a.set_title(lab, fontsize=9)
        np.atleast_1d(ax)[0].set_xlabel(f"{ds}, {PROBLEMS[prob][2]}", fontsize=7)
        fig.tight_layout(pad=0.4)
        fig.savefig(fig_dir / f"enh_{prob}_{ds}_{k}.png", dpi=150)
        plt.close(fig)
    print("wrote strips in", fig_dir)
