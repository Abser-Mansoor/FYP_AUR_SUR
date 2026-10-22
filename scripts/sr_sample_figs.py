"""Visual comparison sheets for the supervisor: input vs bicubic vs restored output.

Samples are random (seed 42), identical across models within a dataset. Nothing is selected by outcome.
Outputs: outputs/sr/samples/{tinyface,survface,lfw_osdface}.png and samples.csv (which crops were shown).
Run: envs/diag/Scripts/python.exe scripts/sr_sample_figs.py
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SR = ROOT / "outputs/sr"
OUT = SR / "samples"
OUT.mkdir(parents=True, exist_ok=True)
SEED, N, SHOW = 42, 2, 256
MODELS = [("InvSR (pre-64, 1 step)", "invsr__pre64_st1"), ("TADSR (pre-128, t250)", "tadsr__pre128_t250"),
          ("OSDFace (pre-128)", "osdface__pre128")]


def load(p, nearest=False):
    im = Image.open(p).convert("RGB")
    return im.resize((SHOW, SHOW), Image.NEAREST if nearest else Image.BICUBIC)


_thumbs = {}


def osdface_eval(file):
    if not _thumbs:
        for z in sorted((SR / "thumbs").glob("eval__osdface__pre128__seed0__*/*.zip")):
            zf = zipfile.ZipFile(z)
            for n in zf.namelist():
                _thumbs[Path(n).stem] = (zf, n)
    zf, n = _thumbs[Path(file).stem]
    return Image.open(io.BytesIO(zf.read(n))).convert("RGB").resize((SHOW, SHOW), Image.BICUBIC)


def restored(arm, file):
    if arm.startswith("osdface"):
        return osdface_eval(file)
    return load(SR / f"restored/eval/{arm}/seed0/{file}")


def sheet(rows, titles, path, head):
    fig, ax = plt.subplots(len(rows), len(titles), figsize=(3.1 * len(titles), 3.2 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, (label, ims) in enumerate(rows):
        for c, im in enumerate(ims):
            ax[r, c].imshow(im); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(titles[c], fontsize=11)
        ax[r, 0].set_ylabel(label, fontsize=9)
    fig.suptitle(head, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    df = pd.read_csv(SR / "subset/eval_images.csv")
    log = []
    for ds, split in (("tinyface", "probe"), ("survface", "mated_probe")):
        d = df[(df.dataset == ds) & (df.split == split)].sample(N, random_state=SEED)
        rows = []
        for label, arm in MODELS:
            for k, (_, r) in enumerate(d.iterrows(), 1):
                f = r.file
                rows.append((f"{label}\nsample {k}: {r['name']}\n({r.w}x{r.h} px)",
                             [load(SR / "subset/eval_input" / f, nearest=True), load(SR / "restored/eval/bicubic__pre512/seed0" / f),
                              restored(arm, f)]))
                log.append((ds, label, k, r["name"], r.h, r.w))
        sheet(rows, ["Input (native crop, enlarged)", "Bicubic (reference)", "Model output"], OUT / f"{ds}.png",
              f"{ds}: random {split} crops (seed {SEED}), same crops for each model")

    # LFW: original aligned 112 px face, 10 px degraded, OSDFace output (restored/lfw10) and bicubic
    uniq_n = len(list((SR / "lfw/hr112").glob("*.png")))
    idx = np.random.RandomState(SEED).choice(uniq_n, N, replace=False)
    rows = []
    for k, i in enumerate(idx, 1):
        n = f"{i:05d}.png"
        rows.append((f"sample {k} (LFW image #{i})",
                     [load(SR / "lfw/hr112" / n), load(SR / "subset/lfw10_input" / n, nearest=True),
                      load(SR / "restored/lfw10/bicubic__pre512/seed0" / n), load(SR / "restored/lfw10/osdface__pre128/seed0" / n)]))
        log.append(("lfw", "OSDFace (pre-128)", k, n, 10, 10))
    sheet(rows, ["Original (aligned 112 px)", "Degraded (10x10, enlarged)", "Bicubic from 10 px", "OSDFace from 10 px"],
          OUT / "lfw_osdface.png", f"LFW: synthetic 10 px degradation, random images (seed {SEED})")
    pd.DataFrame(log, columns=["dataset", "model", "sample", "name", "h", "w"]).to_csv(OUT / "samples.csv", index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
