"""One strip per comparison (input | bicubic | model output) for the supervisor brief, into docs/figures/.
Same random crops as scripts/sr_sample_figs.py (seed 42). Run: envs/diag/Scripts/python.exe scripts/sr_brief_strips.py
Phase 4: envs/score/Scripts/python.exe scripts/sr_brief_strips.py --config configs/enhance.yaml [--arms ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_sample_figs as S  # noqa: E402

FIG = S.ROOT / "docs/figures"
FIG.mkdir(parents=True, exist_ok=True)


def strip(ims, titles, path):
    fig, ax = plt.subplots(1, len(ims), figsize=(2.0 * len(ims), 2.4))
    for a, im, t in zip(ax, ims, titles):
        a.imshow(im); a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=11)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    a = ap.parse_args()
    if a.config:  # Phase 4
        import yaml
        import enh_samples
        cfg = yaml.safe_load(open(S.ROOT / a.config))
        return enh_samples.strips(S.ROOT / cfg["paths"]["results"], a.arms)
    df = pd.read_csv(S.SR / "subset/eval_images.csv")
    t3 = ["Input (enlarged)", "Bicubic", "Model output"]
    for ds, split in (("tinyface", "probe"), ("survface", "mated_probe")):
        d = df[(df.dataset == ds) & (df.split == split)].sample(S.N, random_state=S.SEED)
        for _, arm in S.MODELS:
            for k, (_, r) in enumerate(d.iterrows(), 1):
                f = r.file
                strip([S.load(S.SR / "subset/eval_input" / f, nearest=True), S.load(S.SR / "restored/eval/bicubic__pre512/seed0" / f),
                       S.restored(arm, f)], t3, FIG / f"{ds}_{arm.split('__')[0]}_{k}.png")
    n = len(list((S.SR / "lfw/hr112").glob("*.png")))
    for k, i in enumerate(np.random.RandomState(S.SEED).choice(n, S.N, replace=False), 1):
        nm = f"{i:05d}.png"
        strip([S.load(S.SR / "lfw/hr112" / nm), S.load(S.SR / "subset/lfw10_input" / nm, nearest=True),
               S.load(S.SR / "restored/lfw10/bicubic__pre512/seed0" / nm), S.load(S.SR / "restored/lfw10/osdface__pre128/seed0" / nm)],
              ["Original (112 px)", "Degraded 10x10", "Bicubic", "OSDFace"], FIG / f"lfw_osdface_{k}.png")
    print(sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
