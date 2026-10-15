"""LFW high-resolution-gallery check (low-res side: 10 px, see LR): does a restoration arm recover identity from a tiny crop?

    envs/diag/Scripts/python.exe scripts/sr_lfw_check.py prepare      # aligned 112 px high-res crops + 24 px low-res crops
    envs/diag/Scripts/python.exe scripts/sr_lfw_check.py restore      # restore the 24 px crops with every arm (resumable)
    envs/diag/Scripts/python.exe scripts/sr_lfw_check.py score        # embeddings, verification metrics, paired CIs, decision rule
    envs/diag/Scripts/python.exe scripts/sr_lfw_check.py all

Data: all 6,000 official LFW pairs (3,000 positive), 7,701 unique images. High-res side: Phase 0 landmarks, 112 px
aligned crop (resize fallback on detection failure). Low-res side: that crop downsampled to LR px (INTER_AREA); LR = 10 was chosen
by a direct-feed-only calibration (24 px is at the ceiling: AUC 0.998-0.999).
Conditions per recognizer (cosine similarity): ceiling (HR, HR); cross-direct (24 px direct, HR); cross-restored
(24 px restored, HR); symmetric-direct (24 px, 24 px); symmetric-restored (restored, restored).
Decision rule (fixed before running): an arm "recovers identity" if the paired 95% CI of
AUC(cross-restored) - AUC(cross-direct) is above zero for at least 2 of the 3 non-leaky recognizers.
Restored crops use the arms' Phase 3 settings. Outputs: outputs/sr/results/lfw_check.csv, lfw_check_contrasts.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.align import norm_crop  # noqa: E402
from src.sr import common as C  # noqa: E402

LFW = ROOT / "data/lfw/lfw_home"
OUT = ROOT / "outputs/sr/lfw"
RES = ROOT / "outputs/sr/results"
RECS = ["arcface", "adaface", "magface", "curricularface"]
NON_LEAKY = ["adaface", "magface", "curricularface"]
LR = 10  # calibrated on direct feed only (outputs/sr/results/lfw_calibration_direct.csv): size closest to cross-direct AUC 0.80 inside 0.70-0.90
PART = f"lfw{LR}"
B, SEED = 1000, 42


def lfw_pairs():
    lines = (LFW / "pairs.txt").read_text().split("\n")[1:]
    img = lambda n, i: LFW / "lfw" / n / f"{n}_{int(i):04d}.jpg"  # noqa: E731
    a, b, same = [], [], []
    for ln in lines:
        t = ln.split()
        if len(t) == 3:
            a.append(img(t[0], t[1])); b.append(img(t[0], t[2])); same.append(True)
        elif len(t) == 4:
            a.append(img(t[0], t[1])); b.append(img(t[2], t[3])); same.append(False)
    assert len(a) == 6000
    return a, b, np.array(same)


def unique_images():
    a, b, same = lfw_pairs()
    uniq = sorted(set(a) | set(b))
    return a, b, same, uniq, {p: i for i, p in enumerate(uniq)}


def variants(cfg):
    import sr_pipeline as P
    out = []
    for arm in ("bicubic", "esrgan", "invsr", "tadsr", "osdface"):
        f = RES / f"selection_{arm}.json"
        out.append(P.find_variant(cfg, arm, json.loads(f.read_text())[0]["setting"]))
    return out


def prepare():
    a, b, same, uniq, idx = unique_images()
    z = np.load(ROOT / "outputs/cache/landmarks_lfw.npz")
    assert len(z["ok"]) == len(uniq)
    (OUT / "hr112").mkdir(parents=True, exist_ok=True)
    lr = ROOT / f"outputs/sr/subset/{PART}_input"  # runners and sr_pipeline.restore read <part>_input under outputs/sr/subset
    lr.mkdir(parents=True, exist_ok=True)
    n_fb = 0
    for i, p in enumerate(uniq):
        if (lr / f"{i:05d}.png").exists() and (OUT / "hr112" / f"{i:05d}.png").exists():
            continue
        im = C.read_bgr(p)
        if z["ok"][i]:
            hr = norm_crop(im, z["kps"][i], 112)
        else:
            hr = cv2.resize(im, (112, 112), interpolation=cv2.INTER_LINEAR)
            n_fb += 1
        C.write_png(OUT / "hr112" / f"{i:05d}.png", hr)
        C.write_png(lr / f"{i:05d}.png", cv2.resize(hr, (LR, LR), interpolation=cv2.INTER_AREA))
    print(f"prepared {len(uniq)} images ({int((~z['ok']).sum())} landmark failures -> resize fallback); low-res {LR}x{LR} in {lr}")


def restore():
    import sr_pipeline as P
    cfg = yaml.safe_load(open(ROOT / "configs/sr.yaml"))
    for v in variants(cfg):
        P.restore(cfg, PART, v, 0)


def embed_dir(recs, folder, interp, n):
    out = {r: [] for r in recs}
    for s in range(0, n, 256):
        imgs = [C.read_bgr(Path(folder) / f"{i:05d}.png") for i in range(s, min(s + 256, n))]
        for r, rec in recs.items():
            rec.pre["interp"] = interp
            out[r].append(rec.get_embedding(imgs))
    return {r: np.concatenate(v) for r, v in out.items()}


def sims(ea, eb):
    ea = ea / np.linalg.norm(ea, axis=1, keepdims=True)
    eb = eb / np.linalg.norm(eb, axis=1, keepdims=True)
    return (ea * eb).sum(1)


def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tar_at_far(pos, neg, far=0.1):
    return float((pos > np.quantile(neg, 1 - far)).mean())


def boot(sim_by_cond, same, fn_by_name, rng):
    """Paired bootstrap: the same resampled pairs (positive and negative resampled separately) for every condition."""
    pi, ni = np.flatnonzero(same), np.flatnonzero(~same)
    draws = {k: [] for k in fn_by_name}
    for _ in range(B):
        p = rng.choice(pi, len(pi)); n = rng.choice(ni, len(ni))
        s = {c: (v[p], v[n]) for c, v in sim_by_cond.items()}
        for k, fn in fn_by_name.items():
            draws[k].append(fn(s))
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in draws.items()}


def score():
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    import sr_pipeline as P
    from src import eval_protocols as ev
    from src.recognizer import Recognizer
    cfg = yaml.safe_load(open(ROOT / "configs/sr.yaml"))
    a, b, same, uniq, idx = unique_images()
    ia = np.array([idx[p] for p in a]); ib = np.array([idx[p] for p in b])
    n = len(uniq)
    recs = {r: Recognizer(r, ROOT / cfg["recognizers"][r], (112, 112), "bilinear", True, 256, "cuda") for r in RECS}
    emb_dir = OUT / f"emb{LR}"
    emb_dir.mkdir(parents=True, exist_ok=True)

    def get(name, folder, interp):
        f = {r: emb_dir / f"{name}__{r}.npy" for r in RECS}
        if all(p.exists() for p in f.values()):
            return {r: np.load(p) for r, p in f.items()}
        e = embed_dir(recs, folder, interp, n)
        for r, p in f.items():
            np.save(p, e[r])
        return e
    hr = get("hr112", OUT / "hr112", "bilinear")
    lr = get(f"lr{LR}_direct", ROOT / f"outputs/sr/subset/{PART}_input", "bilinear")
    rows, contrasts = [], []
    arms = [("direct", "resize", None)]
    for v in variants(cfg):
        arm, setting = P.vid(v)
        od = P.out_dir(PART, v, 0)
        if len(list(od.glob("*.png"))) < n:
            print(f"{arm}__{setting}: restoration incomplete ({len(list(od.glob('*.png')))}/{n}); skipped", flush=True)
            continue
        arms.append((arm, setting, get(f"{arm}__{setting}", od, "area")))
    for arm, setting, rest in arms:
        for r in RECS:
            conds = {"ceiling (HR, HR)": (hr[r][ia], hr[r][ib]), "cross direct (LRpx, HR)": (lr[r][ia], hr[r][ib]),
                     "symmetric direct (LRpx, LRpx)": (lr[r][ia], lr[r][ib])}
            if rest is not None:
                conds["cross restored (restored, HR)"] = (rest[r][ia], hr[r][ib])
                conds["symmetric restored (restored, restored)"] = (rest[r][ia], rest[r][ib])
            sim = {c: sims(x, y) for c, (x, y) in conds.items()}
            fns = {}
            for c, s in sim.items():
                fns[f"auc|{c}"] = lambda d, c=c: auc(*d[c])
                fns[f"tar|{c}"] = lambda d, c=c: tar_at_far(*d[c])
            if rest is not None:
                fns["d_auc|cross"] = lambda d: auc(*d["cross restored (restored, HR)"]) - auc(*d["cross direct (LRpx, HR)"])
                fns["d_auc|symmetric"] = lambda d: auc(*d["symmetric restored (restored, restored)"]) - auc(*d["symmetric direct (LRpx, LRpx)"])
                fns["d_tar|cross"] = lambda d: tar_at_far(*d["cross restored (restored, HR)"]) - tar_at_far(*d["cross direct (LRpx, HR)"])
            ci = boot(sim, same, fns, np.random.default_rng(SEED))
            for c, s in sim.items():
                if arm == "direct" and c.startswith(("cross restored", "symmetric restored")):
                    continue
                rows.append(dict(arm=arm, setting=setting, recognizer=r, leaky=(arm, r) == ("osdface", "arcface"), condition=c,
                                 AUC=auc(s[same], s[~same]), AUC_ci95=json.dumps(ci[f"auc|{c}"]),
                                 TAR_at_FAR_0_1=tar_at_far(s[same], s[~same]), TAR_ci95=json.dumps(ci[f"tar|{c}"]),
                                 accuracy_10fold=ev.lfw_verification(conds[c][0], conds[c][1], same)["accuracy"]))
            if rest is not None:
                for name, key in (("cross: restored minus direct (LR vs HR gallery)", "cross"), ("symmetric: restored minus direct (both LR)", "symmetric")):
                    d = fns[f"d_auc|{key}"](dict(zip(sim.keys(), [(s[same], s[~same]) for s in sim.values()])))
                    contrasts.append(dict(arm=arm, setting=setting, recognizer=r, leaky=(arm, r) == ("osdface", "arcface"), contrast=name,
                                          delta_AUC=d, delta_AUC_ci95=json.dumps(ci[f"d_auc|{key}"]),
                                          delta_TAR_at_FAR_0_1=fns[f"d_tar|{key}"](dict(zip(sim.keys(), [(s[same], s[~same]) for s in sim.values()]))) if key == "cross" else None,
                                          delta_TAR_ci95=json.dumps(ci["d_tar|cross"]) if key == "cross" else None))
            print(f"scored {arm}__{setting} {r}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "lfw_check.csv", index=False)
    cdf = pd.DataFrame(contrasts)
    cdf.to_csv(RES / "lfw_check_contrasts.csv", index=False)
    if cdf.empty:
        print("no restored arm was complete: nothing to decide yet")
        return
    # decision rule, applied mechanically
    print("\nDECISION RULE: cross-restored AUC minus cross-direct AUC, paired 95% CI above zero for >=2 of 3 non-leaky recognizers")
    for arm in cdf["arm"].unique():
        t = cdf[(cdf.arm == arm) & cdf.contrast.str.startswith("cross") & cdf.recognizer.isin(NON_LEAKY)]
        n_pos = sum(json.loads(c)[0] > 0 for c in t.delta_AUC_ci95)
        print(f"  {arm}: {n_pos} of {len(t)} non-leaky recognizers positive -> {'RECOVERS IDENTITY (rule met)' if n_pos >= 2 else 'not supported'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "restore", "score", "all"])
    a = ap.parse_args()
    if a.cmd in ("prepare", "all"):
        prepare()
    if a.cmd in ("restore", "all"):
        import logging
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [lfw] %(message)s",
                            handlers=[logging.FileHandler(ROOT / "outputs/sr/run.log"), logging.StreamHandler(sys.stdout)])
        restore()
    if a.cmd in ("score", "all"):
        score()


if __name__ == "__main__":
    main()
