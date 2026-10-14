"""Bicubic upsampling baseline (no model): stretch each native crop to size^2 with cv2 bicubic.

    envs/diag/Scripts/python.exe src/sr/run_bicubic.py --input-dir D --out-root O --sizes 128 512
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.sr import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--out-template", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    files = C.list_images(ROOT / a.input_dir if not Path(a.input_dir).is_absolute() else a.input_dir, a.limit)
    root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    for S in a.sizes:
        t = time.perf_counter()
        o = (ROOT / a.out_template.format(pre=S)) if a.out_template else root / f"size{S}"
        for f in files:
            C.write_png(o / f.name, C.pre_upsample(C.read_bgr(f), S))
        C.save_summary(o / "timing.json", dict(model="bicubic", size=S, n_ok=len(files), sec_per_crop_mean=(time.perf_counter() - t) / len(files), peak_vram_mb=0))
        print("bicubic", S, "done")


if __name__ == "__main__":
    main()
