"""Regression test for src/eval_protocols.py: re-score the cached embeddings of
the wiped Phase 1 run and compare against outputs/results/baseline_results.csv."""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import datasets, eval_protocols as ev  # noqa: E402

CACHE = ROOT / "outputs/cache"
SF = ROOT / "Dataset/QMUL-SurvFace-v1/QMUL-SurvFace"
TF = ROOT / "Dataset/tinyface/tinyface"
NAMES = {"arcface": "ArcFace", "adaface": "AdaFace", "magface": "MagFace", "curricularface": "CurricularFace"}


def load(model, ds, split):
    d = Path(glob.glob(str(CACHE / f"{model}_{ds}_{split}_*"))[0])
    return np.load(d / "embeddings.npy"), json.load(open(d / "names.json"))


def by_name(emb, names, wanted):
    idx = {n: i for i, n in enumerate(names)}
    return emb[[idx[Path(w).name] for w in wanted]]


def main():
    logged = {(r["recognizer"], r["protocol"]): r for r in csv.DictReader(open(ROOT / "outputs/results/baseline_results.csv"))}
    tf, sfi = datasets.tinyface(TF), datasets.survface_identification(SF)
    _, pairs = datasets.survface_verification(SF)
    checks = [("Verification", "AUC"), ("Verification", "TAR@FAR=0.01"), ("Verification", "mean_accuracy"),
              ("OpenSetIdentification", "TPIR20@FPIR=0.3"), ("OpenSetIdentification", "AUC"),
              ("ClosedSetIdentification", "Rank-1"), ("ClosedSetIdentification", "Rank-20"), ("ClosedSetIdentification", "mAP")]
    worst = 0.0
    for m, M in NAMES.items():
        E, n = load(m, "survface", "verification")
        get = lambda col: by_name(E, n, [p[col] for p in pairs[k]])  # noqa: E731
        res = {}
        k = "pos"; pa, pb = get(0), get(1)
        k = "neg"; na, nb = get(0), get(1)
        res["Verification"] = ev.survface_verification(pa, pb, na, nb)
        G, gn = load(m, "survface", "gallery"); Mt, mn = load(m, "survface", "mated_probe"); U, _ = load(m, "survface", "unmated_probe")
        res["OpenSetIdentification"] = ev.survface_openset(by_name(G, gn, sfi["gallery"][0]), sfi["gallery"][1],
                                                           by_name(Mt, mn, sfi["mated_probe"][0]), sfi["mated_probe"][1], U)
        P, pn = load(m, "tinyface", "probe"); Gm, gmn = load(m, "tinyface", "gallery_match"); D, _ = load(m, "tinyface", "distractor")
        res["ClosedSetIdentification"] = ev.tinyface_identification(by_name(P, pn, tf["probe"][0]), tf["probe"][1],
                                                                    by_name(Gm, gmn, tf["gallery_match"][0]), tf["gallery_match"][1], D)
        for proto, key in checks:
            old, new = float(logged[(M, proto)][key]), res[proto][key]
            worst = max(worst, abs(old - new))
            print(f"{M:15s} {proto:24s} {key:16s} logged={old:.6f} rescored={new:.6f} diff={new - old:+.6f}")
    print(f"MAX ABS DIFF = {worst:.6f}")


if __name__ == "__main__":
    main()
