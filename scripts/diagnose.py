"""Phase 1 post-mortem diagnostics. Run with envs/diag (onnxruntime for alignment).

  python scripts/diagnose.py lfw         # HR control: aligned LFW, channel x norm per model
  python scripts/diagnose.py grid        # TinyFace match-only, resize-only preprocessing grid
  python scripts/diagnose.py align       # TinyFace match-only, aligned vs resize
  python scripts/diagnose.py full --channel-norm official|best [--align]   # full official protocols
"""
import argparse
import csv
import itertools
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import datasets, eval_protocols as ev  # noqa: E402
from src.models import OFFICIAL, load_recognizer  # noqa: E402
from src.preprocess import extract, load_bgr  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "configs/diagnostics.yaml"))
P = {k: ROOT / v for k, v in CFG["paths"].items()}
P["out"].mkdir(parents=True, exist_ok=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def log_row(csv_name, row):
    f = P["out"] / csv_name
    row = {"time": datetime.now().isoformat(timespec="seconds"), **row}
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row))


def landmarks(name, paths, lowres):
    """Detect 5-point landmarks once per image set; cached. None = detection failed."""
    f = P["cache"] / f"landmarks_{name}.npz"
    if f.exists():
        d = np.load(f)
        kps, ok = d["kps"], d["ok"]
    else:
        from src.align import SCRFD, detect_landmarks
        det = SCRFD(str(P["detector"]))
        kps, ok = np.zeros((len(paths), 5, 2), np.float32), np.zeros(len(paths), bool)
        for i, p in enumerate(paths):
            k = detect_landmarks(det, load_bgr(p), CFG["align"]["lowres_side"] if lowres else None, CFG["align"]["det_thresh"])
            if k is not None:
                kps[i], ok[i] = k, True
        np.savez(f, kps=kps, ok=ok)
    return [k if o else None for k, o in zip(kps, ok)], float(ok.mean())


# ---------------------------------------------------------------- LFW control
def lfw_pairs():
    root = P["lfw"]
    lines = (root / "pairs.txt").read_text().split("\n")[1:]
    img = lambda n, i: root / "lfw" / n / f"{n}_{int(i):04d}.jpg"  # noqa: E731
    a, b, same = [], [], []
    for ln in lines:
        t = ln.split()
        if len(t) == 3:
            a.append(img(t[0], t[1])); b.append(img(t[0], t[2])); same.append(True)
        elif len(t) == 4:
            a.append(img(t[0], t[1])); b.append(img(t[2], t[3])); same.append(False)
    assert len(a) == 6000
    return a, b, np.array(same)


def cmd_lfw(args):
    a, b, same = lfw_pairs()
    uniq = sorted(set(a) | set(b))
    idx = {p: i for i, p in enumerate(uniq)}
    kps, rate = landmarks("lfw", uniq, lowres=False)
    print(f"LFW landmark detection rate: {rate:.4f}")
    imgs = [load_bgr(p) for p in uniq]
    for m, w in CFG["recognizers"].items():
        embed = load_recognizer(m, ROOT / w)
        for ch, nm in itertools.product(CFG["grid"]["channel"], CFG["grid"]["norm"]):
            cfg = dict(channel=ch, norm=nm, interp="bicubic", flip=False)
            E = extract(embed, imgs, cfg, landmarks=kps, batch=CFG["batch_size"])
            r = ev.lfw_verification(E[[idx[p] for p in a]], E[[idx[p] for p in b]], same)
            log_row("lfw_control.csv", dict(recognizer=m, align=True, det_rate=round(rate, 4), **cfg,
                                            official=(OFFICIAL[m] == dict(channel=ch, norm=nm)), **r))
        del embed; torch.cuda.empty_cache()


# ---------------------------------------------------------- TinyFace subset
def tinyface_subset():
    tf = datasets.tinyface(P["tinyface"])
    (pp, pid), (gp, gid) = tf["probe"], tf["gallery_match"]
    return pp, pid, gp, gid


def match_only(P_, pid, G_, gid):
    """Rank-1 / Rank-20 / mAP vs the 4443-image match gallery (no distractors)."""
    return ev.tinyface_identification(P_, pid, G_, gid, np.zeros((0, G_.shape[1]), np.float32))


def cmd_grid(args):
    pp, pid, gp, gid = tinyface_subset()
    imgs = [load_bgr(p) for p in pp + gp]
    n = len(pp)
    for m, w in CFG["recognizers"].items():
        embed = load_recognizer(m, ROOT / w)
        g = CFG["grid"]
        for ch, nm, it, fl in itertools.product(g["channel"], g["norm"], g["interp"], g["flip"]):
            cfg = dict(channel=ch, norm=nm, interp=it, flip=fl)
            t = time.time()
            E = extract(embed, imgs, cfg, batch=CFG["batch_size"])
            r = match_only(E[:n], pid, E[n:], gid)
            log_row("tinyface_grid.csv", dict(recognizer=m, align=False, **cfg, official=(OFFICIAL[m] == dict(channel=ch, norm=nm)),
                                              **{k: r[k] for k in ("Rank-1", "Rank-20", "mAP")}, sec=round(time.time() - t, 1)))
        del embed; torch.cuda.empty_cache()


def cmd_align(args):
    pp, pid, gp, gid = tinyface_subset()
    paths = pp + gp
    kps, rate = landmarks("tinyface_subset", paths, lowres=True)
    print(f"TinyFace subset landmark detection rate: {rate:.4f}")
    imgs = [load_bgr(p) for p in paths]
    n = len(pp)
    for m, w in CFG["recognizers"].items():
        embed = load_recognizer(m, ROOT / w)
        for fl in (False, True):
            cfg = dict(**OFFICIAL[m], interp="bicubic", flip=fl)
            for use in (False, True):
                E = extract(embed, imgs, cfg, landmarks=kps if use else None, batch=CFG["batch_size"])
                r = match_only(E[:n], pid, E[n:], gid)
                log_row("tinyface_align.csv", dict(recognizer=m, align=use, det_rate=round(rate, 4), **cfg,
                                                   **{k: r[k] for k in ("Rank-1", "Rank-20", "mAP")}))
        del embed; torch.cuda.empty_cache()


def main():
    seed_all(CFG["seed"])
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lfw", "grid", "align"])
    args = ap.parse_args()
    {"lfw": cmd_lfw, "grid": cmd_grid, "align": cmd_align}[args.cmd](args)


if __name__ == "__main__":
    main()
