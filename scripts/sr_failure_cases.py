"""Failure-case sheets for the selected variants on the eval subset (runs in envs/diag after `sr_pipeline.py eval`).

    envs/diag/Scripts/python.exe scripts/sr_failure_cases.py [--n 12]
    envs/score/Scripts/python.exe scripts/sr_failure_cases.py --config configs/enhance.yaml [--arms gamma ...]   (Phase 4: outputs/enhance/failure_cases)

Per arm, three sheets in outputs/sr/failure_cases/:
  <arm>__identity_changed.png : the queries whose restored embedding moved furthest from the direct-feed embedding
                                (mean cosine over the non-leaky recognizers): input | restored  -> "identity changed"
  <arm>__lost_match.png       : TinyFace probes that were rank-1 correct with direct feed and are wrong after restoration
                                for at least 2 of the 3 non-leaky recognizers: input | restored | true gallery image
  <arm>__random.png           : random queries, unselected, so artifacts / invented detail can be judged without selection bias
Selection is by embedding drift or rank change only; artifacts and hallucinations are then judged by eye and described
in the report. A manifest CSV with the exact files is written beside each sheet.
"""
import argparse
import json
import zipfile
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sr import common as C  # noqa: E402
from src.sr import eval as E  # noqa: E402

T = 160


def tile(src, label):
    im = src if isinstance(src, np.ndarray) else C.read_bgr(src)
    t = cv2.resize(im, (T, T), interpolation=cv2.INTER_CUBIC if im.shape[0] < T else cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (T, 13), (0, 0, 0), -1)
    cv2.putText(t, label, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def sheet(rows, cols_fn, out):
    lines = [np.hstack([tile(p, lab) for p, lab in cols_fn(r)]) for r in rows]
    if lines:
        C.write_png(out, np.vstack(lines))


class Images:
    """Restored image of a crop: the full-size PNG if it is on this PC, else the 256-px JPEG thumbnail written by a
    remote (Colab) job (outputs/sr/thumbs/<job_id>/<chunk>.zip)."""

    def __init__(self, arm, setting):
        self.rd = ROOT / "outputs/sr/restored/eval" / f"{arm}__{setting}" / "seed0"
        self.thumbs = {}
        for z in sorted((ROOT / "outputs/sr/thumbs").glob(f"eval__{arm}__{setting}__seed0*/*.zip")):
            zf = zipfile.ZipFile(z)
            for n in zf.namelist():
                self.thumbs[Path(n).stem] = (zf, n)
        self.source = "thumbnail 256 px" if self.thumbs and not any(self.rd.glob("*.png")) else "full size"

    def __call__(self, file):
        p = self.rd / file
        if p.exists():
            return p
        zf, n = self.thumbs[Path(file).stem]
        return cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR)


def selected_variants(res, cache, cfg=None, arms=None):
    """(arm, setting) of every arm's rule-selected setting with scored eval embeddings, plus the InvSR sensitivity setting."""
    out = []
    for arm in (["bicubic", "gamma", "clahe", "zerodce", "unsharp", "nafnet"] if cfg is not None else ("bicubic", "esrgan", "invsr", "tadsr", "osdface")):
        f = res / f"selection_{arm}.json"
        if f.exists() and json.loads(f.read_text()) and (not arms or arm in arms):
            out.append((arm, json.loads(f.read_text())[0]["setting"]))
    if cfg is None:
        out.append(("invsr", "pre128_st1"))
    keep = []
    for arm, setting in out:
        if (cache / arm / setting / "seed0" / "eval__adaface__tinyface_ps.npz").exists():
            keep.append((arm, setting))
        else:
            print(f"{arm}__{setting}: no scored eval results yet, skipped")
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--config", default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / (a.config or "configs/sr.yaml")))
    d, _ = E.load_subset(ROOT, "eval")
    native = ROOT / "outputs/sr/subset/eval_input"
    cache = ROOT / "outputs/sr/emb/eval"
    res = ROOT / cfg["paths"].get("results", "outputs/sr/results")
    out = res.parent / "failure_cases"  # Phase 3: outputs/sr/failure_cases
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    q_idx = np.flatnonzero(d["split"].isin(["probe", "mated_probe"]).to_numpy())
    dire = {r: np.load(cache / "direct/resize/seed0" / f"eval__{r}.npy") for r in E.NON_LEAKY}
    for arm, setting in selected_variants(res, cache, cfg if a.config else None, a.arms):
        rng = np.random.default_rng(cfg["seed"])  # same random crops for every arm, so the random sheets are comparable
        img = Images(arm, setting)
        tag = f"{arm}__{setting}"
        em = {r: np.load(cache / arm / setting / "seed0" / f"eval__{r}.npy") for r in E.NON_LEAKY}
        cos = np.mean([(dire[r] / np.linalg.norm(dire[r], axis=1, keepdims=True) * em[r] / np.linalg.norm(em[r], axis=1, keepdims=True)).sum(1)
                       for r in E.NON_LEAKY], axis=0)
        worst = q_idx[np.argsort(cos[q_idx])[:a.n]]
        sub = d.iloc[worst].assign(drift_cos=cos[worst])
        sub[["file", "dataset", "split", "drift_cos"]].to_csv(out / f"{tag}__identity_changed.csv", index=False)
        sheet([r for _, r in sub.iterrows()], lambda r: [(native / r["file"], f"input {r['w']}x{r['h']}"),
                                                           (img(r["file"]), f"{arm} cos={r['drift_cos']:.2f}")], out / f"{tag}__identity_changed.png")
        # lost matches (TinyFace probes only): direct rank-1 correct -> restored wrong for >=2 of 3 recognizers
        pr = np.flatnonzero(((d["dataset"] == "tinyface") & (d["split"] == "probe")).to_numpy())
        lost = np.zeros(len(pr), int)
        for r in E.NON_LEAKY:
            dps = np.load(cache / "direct/resize/seed0" / f"eval__{r}__tinyface_ps.npz")
            rps = np.load(cache / arm / setting / "seed0" / f"eval__{r}__tinyface_ps.npz")
            lost += ((dps["first_rank"] == 0) & (rps["first_rank"] > 0)).astype(int)
        hit = pr[lost >= 2]
        gal = d[(d["dataset"] == "tinyface") & (d["split"] == "gallery_match")]
        gal_by_id = gal.groupby("id")["file"].first().to_dict()
        pick = hit[rng.permutation(len(hit))[:a.n]]
        sub = d.iloc[pick]
        sub[["file", "id"]].assign(n_lost_total=len(hit), n_probes=len(pr)).to_csv(out / f"{tag}__lost_match.csv", index=False)
        sheet([r for _, r in sub.iterrows()], lambda r: [(native / r["file"], "input"), (img(r["file"]), f"{arm}"),
                                                           (native / gal_by_id[r["id"]], "true gallery image")], out / f"{tag}__lost_match.png")
        rnd = d.iloc[rng.choice(q_idx, a.n, replace=False)]
        sheet([r for _, r in rnd.iterrows()], lambda r: [(native / r["file"], f"input {r['w']}x{r['h']}"), (img(r["file"]), arm)],
              out / f"{tag}__random.png")
        print(f"{tag}: lost matches {len(hit)} of {len(pr)} TinyFace probes (>=2 of 3 non-leaky recognizers); images: {img.source}")


if __name__ == "__main__":
    main()
