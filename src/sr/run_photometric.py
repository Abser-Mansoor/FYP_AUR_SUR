"""Photometric (OpenCV, CPU) arms: noop, gamma, clahe, unsharp. Same-size output, no model, no randomness.

    envs/score/Scripts/python.exe src/sr/run_photometric.py --input-dir D --out-dir O --params '{"arm": "gamma", "g": 0.7}'
Resumable (existing outputs are skipped), atomic PNG writes, every failure is logged in timing.json.
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.sr import common as C  # noqa: E402


def gamma_fn(g):
    def f(bgr):
        if g == "adaptive":  # log(0.45)/log(mean luminance/255), clamped to [0.3, 1]; mean luminance as the grey-scale mean
            m = max(float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()), 1.0) / 255.0
            gg = min(max(math.log(0.45) / math.log(m), 0.3), 1.0) if m < 1.0 else 1.0
        else:
            gg = float(g)
        lut = np.clip(255.0 * (np.arange(256) / 255.0) ** gg + 0.5, 0, 255).astype(np.uint8)
        return cv2.LUT(bgr, lut)
    return f


def clahe_fn(c, t):
    cl = cv2.createCLAHE(clipLimit=float(c), tileGridSize=(int(t), int(t)))

    def f(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cl.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return f


def unsharp_fn(s, a):
    def f(bgr):
        blur = cv2.GaussianBlur(bgr, (0, 0), float(s))
        return cv2.addWeighted(bgr, 1.0 + float(a), blur, -float(a), 0)  # saturating uint8
    return f


def make_fn(p):
    arm = p["arm"]
    if arm == "gamma":
        return gamma_fn(p["g"])
    if arm == "clahe":
        return clahe_fn(p["c"], p["t"])
    if arm == "unsharp":
        return unsharp_fn(p["s"], p["a"])
    raise ValueError(f"unknown arm {arm}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--params", required=True, help="variant json, e.g. '{\"arm\": \"clahe\", \"c\": 2, \"t\": 4}'")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    p = json.loads(a.params)
    inp = Path(a.input_dir) if Path(a.input_dir).is_absolute() else ROOT / a.input_dir
    out = Path(a.out_dir) if Path(a.out_dir).is_absolute() else ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    files = C.list_images(inp, a.limit)
    tm = C.Timing(out, "cpu")
    fn = None if p["arm"] == "noop" else make_fn(p)

    def one(f):
        if p["arm"] == "noop":  # byte copy of the input (atomic)
            tmp = out / (f.name + ".tmp")
            shutil.copyfile(f, tmp)
            tmp.replace(out / f.name)
        else:
            C.write_png(out / f.name, fn(C.read_bgr(f)))

    for f in files:
        if C.out_done(out, f.name):
            continue
        tm.run(f.name, lambda f=f: one(f))
    s = tm.summary()
    s.update(model=p["arm"], params=p, device="cpu")
    C.save_summary(out / "timing.json", s)
    print(p["arm"], "done", s["n_ok"], "ok", s["n_failed"], "failed")


if __name__ == "__main__":
    main()
