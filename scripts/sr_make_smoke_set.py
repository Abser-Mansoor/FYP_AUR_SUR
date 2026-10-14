"""Seeded smoke-test set for the SR arms: native-size crops written as PNG plus a manifest CSV.

    envs/diag/Scripts/python.exe scripts/sr_make_smoke_set.py [--n-per-group 25]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [("tinyface", "probe"), ("survface", "mated_probe"), ("survface", "verification")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-group", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/sr/smoke")
    a = ap.parse_args()
    out = ROOT / a.out
    (out / "input").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for ds, split in GROUPS:
        d = pd.read_csv(ROOT / "outputs/stratify/bins" / f"{ds}_{split}.csv", keep_default_na=False)
        # a third from the low-sharpness bin, a third from the darkest 10%, the rest random: covers the hard cases
        pick = set()
        for mask, k in ((d["sharp_bin_sz"] == "low_sharpness", a.n_per_group // 3), (d["lum_bin10"] == "dark10", a.n_per_group // 3)):
            pick |= set(rng.choice(np.flatnonzero(mask), k, replace=False).tolist())
        while len(pick) < a.n_per_group:
            pick.add(int(rng.integers(len(d))))
        for i in sorted(pick):
            r = d.iloc[i]
            name = f"{ds}__{split}__{Path(r['name']).stem}.png"
            im = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_COLOR)
            ok, buf = cv2.imencode(".png", im)
            buf.tofile(str(out / "input" / name))
            rows.append(dict(file=name, dataset=ds, split=split, src_name=r["name"], h=im.shape[0], w=im.shape[1],
                             sharp_bin_sz=r["sharp_bin_sz"], lum_bin=r["lum_bin"], lum_bin10=r["lum_bin10"]))
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print(f"{len(rows)} crops -> {out}")
    print(pd.DataFrame(rows).groupby(["dataset", "split"]).agg(n=("file", "size"), h_med=("h", "median"), w_med=("w", "median")))


if __name__ == "__main__":
    main()
