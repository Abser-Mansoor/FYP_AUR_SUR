"""Phase 4 LFW synthetic check: does a low-light / deblur arm recover identity?

    envs\\score\\Scripts\\python.exe scripts\\enh_lfw_check.py calibrate | restore | score | all

Same design as scripts/sr_lfw_check.py (6,000 official LFW pairs, 7,701 aligned 112 px crops in outputs/sr/lfw/hr112); the helpers
(lfw_pairs, unique_images, sims, auc, tar_at_far, the paired bootstrap) are imported from it. The degraded side is the sharp crop:
  dark    : I * s, then Poisson-Gaussian noise (shot noise Poisson(255 * x) / 255, read noise sigma 0.02), clipped, seeded per image;
  gauss   : Gaussian blur, sigma in pixels;
  motion  : 11 px linear motion blur at a seeded angle in [0, 180) (fixed length; its direct AUC is recorded, not tuned).
Calibration (direct feed only, written to lfw_enh_calibration.csv BEFORE any arm is restored; `restore` refuses without it): for dark
(s) and gauss (sigma) the strength whose mean non-leaky cross-degraded AUC (degraded vs sharp partner) is closest to 0.80, within
0.70-0.90 (closest overall, flagged, if none falls in range).
Arms (selected settings from selection_<arm>.json): low-light arms (gamma, clahe, zerodce) on dark, deblur arms (unsharp, nafnet) on
gauss and motion, all restored at 112 px; the embedding rule is the arm's (bilinear). noop is the degraded crop itself, so the paired
contrast against noop equals the contrast against cross-degraded.
Rule (pre-registered): an arm "recovers identity" if the paired 95% CI of AUC(cross-enhanced) - AUC(cross-degraded) is above zero for
at least 2 of the 3 non-leaky recognizers. Outputs: <results>/lfw_enh_calibration.csv, lfw_enh_check.csv, lfw_enh_contrasts.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import sr_lfw_check as L  # noqa: E402
from src.sr import common as C  # noqa: E402

CFG = ROOT / "configs/enhance.yaml"
RECS, NON_LEAKY, SEED = L.RECS, L.NON_LEAKY, 42
SIZE = 112
GRID = {"dark": [0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03], "gauss": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]}
FAMILY_ARMS = {"dark": ["gamma", "clahe", "zerodce"], "gauss": ["unsharp", "nafnet"], "motion": ["unsharp", "nafnet"]}
MOTION_LEN = 11
POISSON_PEAK, READ_SIGMA = 255.0, 0.02


def load_cfg():
    cfg = yaml.safe_load(open(CFG))
    return cfg, ROOT / cfg["paths"].get("results", "outputs/sr/results")


def load_hr(n):
    return [C.read_bgr(L.OUT / "hr112" / f"{i:05d}.png") for i in range(n)]


def degrade(kind, img, strength, i):
    """Deterministic per image (rng seeded by SEED + i), independent of processing order."""
    rng = np.random.default_rng(SEED * 100003 + i)
    if kind == "dark":
        x = img.astype(np.float64) / 255.0 * strength
        y = rng.poisson(POISSON_PEAK * x) / POISSON_PEAK + rng.normal(0.0, READ_SIGMA, x.shape)
        return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)
    if kind == "gauss":
        return cv2.GaussianBlur(img, (0, 0), float(strength))
    if kind == "motion":
        ang = rng.uniform(0, 180)
        k = np.zeros((MOTION_LEN, MOTION_LEN), np.float32)
        c = (MOTION_LEN - 1) / 2.0
        dx, dy = np.cos(np.radians(ang)) * c, np.sin(np.radians(ang)) * c
        cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
        return cv2.filter2D(img, -1, k / k.sum(), borderType=cv2.BORDER_REFLECT)
    raise ValueError(kind)


def make_recs(cfg, names):
    from src.recognizer import Recognizer
    return {r: Recognizer(r, ROOT / cfg["recognizers"][r], (SIZE, SIZE), "bilinear", True, cfg.get("batch_size", 128), "cuda") for r in names}


def embed_imgs(recs, imgs, interp="bilinear"):
    out = {r: [] for r in recs}
    for s in range(0, len(imgs), 512):
        for r, rec in recs.items():
            rec.pre["interp"] = interp
            out[r].append(rec.get_embedding(imgs[s:s + 512]))
    return {r: np.concatenate(v) for r, v in out.items()}


def cache_embed(recs, name, imgs_fn, interp, emb_dir):
    f = {r: emb_dir / f"{name}__{r}.npy" for r in recs}
    if all(p.exists() for p in f.values()):
        return {r: np.load(p) for r, p in f.items()}
    e = embed_imgs(recs, imgs_fn(), interp)
    emb_dir.mkdir(parents=True, exist_ok=True)
    for r, p in f.items():
        np.save(p, e[r])
    return e


def calibrate():
    cfg, res = load_cfg()
    a, b, same, uniq, idx = L.unique_images()
    ia = np.array([idx[p] for p in a]); ib = np.array([idx[p] for p in b])
    n = len(uniq)
    hr = load_hr(n)
    recs = make_recs(cfg, NON_LEAKY)
    emb_dir = L.OUT / "emb_enh"
    hre = cache_embed(recs, "hr112", lambda: hr, "bilinear", emb_dir)
    rows = []
    for kind, grid in GRID.items():
        for st in grid:
            e = cache_embed(recs, f"cal_{kind}_{st}", lambda: [degrade(kind, im, st, i) for i, im in enumerate(hr)], "bilinear", emb_dir)
            aucs = {r: L.auc(*(lambda s: (s[same], s[~same]))(L.sims(e[r][ia], hre[r][ib]))) for r in NON_LEAKY}
            rows.append(dict(degradation=kind, strength=st, mean_nonleaky_cross_auc=float(np.mean(list(aucs.values()))), **{f"auc_{r}": v for r, v in aucs.items()}))
            print(kind, st, rows[-1]["mean_nonleaky_cross_auc"], flush=True)
    # motion blur: fixed length, direct AUC recorded only
    e = cache_embed(recs, "cal_motion_11", lambda: [degrade("motion", im, MOTION_LEN, i) for i, im in enumerate(hr)], "bilinear", emb_dir)
    aucs = {r: L.auc(*(lambda s: (s[same], s[~same]))(L.sims(e[r][ia], hre[r][ib]))) for r in NON_LEAKY}
    rows.append(dict(degradation="motion", strength=MOTION_LEN, mean_nonleaky_cross_auc=float(np.mean(list(aucs.values()))), **{f"auc_{r}": v for r, v in aucs.items()}))
    ceil = float(np.mean([L.auc(*(lambda s: (s[same], s[~same]))(L.sims(hre[r][ia], hre[r][ib]))) for r in NON_LEAKY]))
    df = pd.DataFrame(rows)
    df["chosen"] = False
    for kind in GRID:
        t = df[df.degradation == kind]
        inr = t[(t.mean_nonleaky_cross_auc >= 0.70) & (t.mean_nonleaky_cross_auc <= 0.90)]
        pool = inr if len(inr) else t
        df.loc[(pool.assign(d=(pool.mean_nonleaky_cross_auc - 0.80).abs())).sort_values("d").index[0], "chosen"] = True
        df.loc[t.index, "in_range"] = ((t.mean_nonleaky_cross_auc >= 0.70) & (t.mean_nonleaky_cross_auc <= 0.90)).to_numpy()
    df.loc[df.degradation == "motion", ["chosen", "in_range"]] = [True, bool(0.70 <= df[df.degradation == "motion"].mean_nonleaky_cross_auc.iloc[0] <= 0.90)]
    df["ceiling_mean_nonleaky_auc"] = ceil
    res.mkdir(parents=True, exist_ok=True)
    df.to_csv(res / "lfw_enh_calibration.csv", index=False)
    print(df[df.chosen][["degradation", "strength", "mean_nonleaky_cross_auc", "in_range"]].to_string(index=False), "\nceiling AUC (non-leaky mean):", ceil)


def chosen():
    cfg, res = load_cfg()
    f = res / "lfw_enh_calibration.csv"
    if not f.exists():
        sys.exit("lfw_enh_calibration.csv is missing: run `calibrate` first (it must exist before any arm is restored)")
    df = pd.read_csv(f)
    return {k: float(df[(df.degradation == k) & df.chosen].strength.iloc[0]) for k in ("dark", "gauss", "motion")}


def part_of(kind):
    return f"lfwenh_{kind}"


def prepare_inputs(kind, st, n):
    d = ROOT / f"outputs/sr/subset/{part_of(kind)}_input"
    d.mkdir(parents=True, exist_ok=True)
    hr = None
    for i in range(n):
        if (d / f"{i:05d}.png").exists():
            continue
        if hr is None:
            hr = load_hr(n)
        C.write_png(d / f"{i:05d}.png", degrade(kind, hr[i], st, i))
    return d


def selected_variants(cfg, res, arms):
    import sr_pipeline as P
    out = []
    for arm in arms:
        sel = json.loads((res / f"selection_{arm}.json").read_text())[0]["setting"]
        out.append(P.find_variant(cfg, arm, sel))
    return out


def restore():
    import logging
    import sr_pipeline as P
    cfg, res = load_cfg()
    st = chosen()  # refuses to run without the calibration file
    n = len(L.unique_images()[3])
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [lfw-enh] %(message)s",
                        handlers=[logging.FileHandler(res.parent / "run.log"), logging.StreamHandler(sys.stdout)])
    for kind, arms in FAMILY_ARMS.items():
        prepare_inputs(kind, st[kind], n)
        for v in selected_variants(cfg, res, arms):
            P.restore(cfg, part_of(kind), v, 0)


def score():
    import sr_pipeline as P
    from src import eval_protocols as ev
    cfg, res = load_cfg()
    st = chosen()
    a, b, same, uniq, idx = L.unique_images()
    ia = np.array([idx[p] for p in a]); ib = np.array([idx[p] for p in b])
    n = len(uniq)
    recs = make_recs(cfg, RECS)
    emb_dir = L.OUT / "emb_enh"
    hre = cache_embed(recs, "hr112", lambda: load_hr(n), "bilinear", emb_dir)
    rows, contrasts = [], []
    for kind, arms in FAMILY_ARMS.items():
        part = part_of(kind)
        deg_dir = ROOT / f"outputs/sr/subset/{part}_input"
        deg = cache_embed(recs, f"deg_{kind}_{st[kind]}", lambda: [C.read_bgr(deg_dir / f"{i:05d}.png") for i in range(n)], "bilinear", emb_dir)
        items = [("noop", "default", None)]
        for v in selected_variants(cfg, res, arms):
            arm, setting = P.vid(v)
            od = P.out_dir(part, v, 0)
            if len(list(od.glob("*.png"))) < n:
                print(f"{kind} {arm}__{setting}: restoration incomplete; skipped", flush=True)
                continue
            interp = P.variant_interp(cfg, v)
            items.append((arm, setting, cache_embed(recs, f"{kind}_{arm}__{setting}", lambda od=od: [C.read_bgr(od / f"{i:05d}.png") for i in range(n)], interp, emb_dir)))
        for arm, setting, enh in items:
            for r in RECS:
                conds = {"ceiling (HR, HR)": (hre[r][ia], hre[r][ib]), "cross degraded (deg, HR)": (deg[r][ia], hre[r][ib]),
                         "symmetric degraded (deg, deg)": (deg[r][ia], deg[r][ib])}
                if enh is not None:
                    conds["cross enhanced (enh, HR)"] = (enh[r][ia], hre[r][ib])
                    conds["symmetric enhanced (enh, enh)"] = (enh[r][ia], enh[r][ib])
                sim = {c: L.sims(x, y) for c, (x, y) in conds.items()}
                fns = {}
                for c in sim:
                    fns[f"auc|{c}"] = lambda d, c=c: L.auc(*d[c])
                    fns[f"tar|{c}"] = lambda d, c=c: L.tar_at_far(*d[c])
                if enh is not None:
                    fns["d_auc|cross"] = lambda d: L.auc(*d["cross enhanced (enh, HR)"]) - L.auc(*d["cross degraded (deg, HR)"])
                    fns["d_auc|symmetric"] = lambda d: L.auc(*d["symmetric enhanced (enh, enh)"]) - L.auc(*d["symmetric degraded (deg, deg)"])
                ci = L.boot(sim, same, fns, np.random.default_rng(SEED))
                pairs_ = {c: (s[same], s[~same]) for c, s in sim.items()}
                for c, s in sim.items():
                    if arm == "noop" and "enhanced" in c:
                        continue
                    rows.append(dict(degradation=kind, strength=st[kind], arm=arm, setting=setting, recognizer=r, condition=c,
                                     AUC=L.auc(*pairs_[c]), AUC_ci95=json.dumps(ci[f"auc|{c}"]),
                                     TAR_at_FAR_0_1=L.tar_at_far(*pairs_[c]), TAR_ci95=json.dumps(ci[f"tar|{c}"]),
                                     accuracy_10fold=ev.lfw_verification(conds[c][0], conds[c][1], same)["accuracy"]))
                if enh is not None:
                    for key in ("cross", "symmetric"):
                        contrasts.append(dict(degradation=kind, strength=st[kind], arm=arm, setting=setting, recognizer=r, contrast=key,
                                              delta_AUC=fns[f"d_auc|{key}"](pairs_), delta_AUC_ci95=json.dumps(ci[f"d_auc|{key}"])))
                print(f"scored {kind} {arm}__{setting} {r}", flush=True)
    pd.DataFrame(rows).to_csv(res / "lfw_enh_check.csv", index=False)
    cdf = pd.DataFrame(contrasts)
    cdf.to_csv(res / "lfw_enh_contrasts.csv", index=False)
    print("\nRULE: cross-enhanced AUC minus cross-degraded AUC, paired 95% CI above zero for >=2 of 3 non-leaky recognizers")
    for (kind, arm, setting), t in cdf[(cdf.contrast == "cross") & cdf.recognizer.isin(NON_LEAKY)].groupby(["degradation", "arm", "setting"]):
        k = sum(json.loads(c)[0] > 0 for c in t.delta_AUC_ci95)
        print(f"  {kind} {arm}__{setting}: {k} of {len(t)} -> {'RECOVERS IDENTITY (rule met)' if k >= 2 else 'not supported'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "restore", "score", "all"])
    a = ap.parse_args()
    if a.cmd in ("calibrate", "all"):
        calibrate()
    if a.cmd in ("restore", "all"):
        restore()
    if a.cmd in ("score", "all"):
        score()


if __name__ == "__main__":
    main()
