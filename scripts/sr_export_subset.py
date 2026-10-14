"""Write the subset crops (native size, lossless PNG of the decoded pixels) for the SR runners.

    envs/diag/Scripts/python.exe scripts/sr_export_subset.py
Output: outputs/sr/subset/<part>_input/<dataset>__<split>__<name-stem>.png
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sr import common as C  # noqa: E402


def key(r):
    return f"{r['dataset']}__{r['split']}__{Path(r['name']).stem}.png"


def main():
    for part in ("tuning", "eval"):
        d = pd.read_csv(ROOT / "outputs/sr/subset" / f"{part}_images.csv")
        d["file"] = [key(r) for _, r in d.iterrows()]
        assert d["file"].is_unique, "file names must be unique"
        d.to_csv(ROOT / "outputs/sr/subset" / f"{part}_images.csv", index=False)
        out = ROOT / "outputs/sr/subset" / f"{part}_input"

        def one(r):
            f = out / r["file"]
            if not f.exists():
                C.write_png(f, C.read_bgr(r["path"]))
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(one, [r for _, r in d.iterrows()]))
        print(part, len(d), "crops ->", out)


if __name__ == "__main__":
    main()
