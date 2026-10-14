"""Side-by-side montage: native input | one or more output folders, one row per crop.

    envs/diag/Scripts/python.exe scripts/sr_montage.py --input-dir D --out out.png --n 8 --cols name=dir [name=dir ...]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sr import common as C  # noqa: E402

T = 192


def tile(img, label):
    t = cv2.resize(img, (T, T), interpolation=cv2.INTER_CUBIC if img.shape[0] < T else cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (T, 14), (0, 0, 0), -1)
    cv2.putText(t, label, (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cols", nargs="+", required=True)
    a = ap.parse_args()
    cols = [c.split("=", 1) for c in a.cols]
    files = [f for f in C.list_images(a.input_dir) if (Path(cols[0][1]) / f.name).exists()]
    files = files[::max(len(files) // a.n, 1)][:a.n]
    rows = []
    for f in files:
        im = C.read_bgr(f)
        r = [tile(im, f"input {im.shape[1]}x{im.shape[0]}")]
        for name, d in cols:
            p = Path(d) / f.name
            r.append(tile(C.read_bgr(p), name) if p.exists() else np.zeros((T, T, 3), np.uint8))
        rows.append(np.hstack(r))
    C.write_png(a.out, np.vstack(rows))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
