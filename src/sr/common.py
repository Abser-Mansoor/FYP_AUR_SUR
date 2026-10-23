"""Helpers shared by the per-model runners. Dependencies: numpy, opencv only (runs in every model venv)."""
import json
import time
from pathlib import Path

import cv2
import numpy as np


def read_bgr(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(f"unreadable image: {path}")
    return im


def write_png(path, bgr):
    """Atomic write (temp file + rename) so an interrupted run never leaves a truncated PNG that counts as done."""
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    tmp = path.with_name(path.name + ".tmp")
    buf.tofile(str(tmp))
    os.replace(tmp, path)


def pre_upsample(bgr, size):
    """Stretch-resize a native crop to size x size with bicubic (same square stretch the recognizers apply)."""
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_CUBIC)


def list_images(folder, limit=None, file_list=None):
    """Images of a folder, sorted. file_list: text file with one file name per line; returns exactly those, in that order."""
    if file_list:
        names = [x.strip() for x in Path(file_list).read_text(encoding="utf-8").splitlines() if x.strip()]
        return [Path(folder) / n for n in names]
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return files[:limit] if limit else files


class Timing:
    """Per-crop wall time (CUDA-synchronised) and peak allocated VRAM for one setting.
    Resumable: timings of earlier (interrupted) invocations are read back from <out>/timing.json.
    device="cpu": pure wall-time path (OpenCV runners): torch is never imported, VRAM is reported as 0."""

    def __init__(self, out_dir=None, device="cuda"):
        self.sec, self.errors, self.prev_vram, self.out, self.cuda = [], [], 0.0, out_dir, device == "cuda"
        p = Path(out_dir) / "timing.json" if out_dir else None
        if p is not None and p.exists():
            old = json.loads(p.read_text())
            self.sec = list(old.get("sec_list", []))
            self.prev_vram = old.get("peak_vram_mb") or 0.0

    def run(self, name, fn):
        if self.cuda:
            import torch
            torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            out = fn()
        except Exception as e:  # logged as a result, never dropped
            self.errors.append({"file": name, "error": f"{type(e).__name__}: {e}"})
            return None
        if self.cuda:
            torch.cuda.synchronize()
        self.sec.append(time.perf_counter() - t)
        return out

    def summary(self):
        vram = 0.0
        if self.cuda:
            import torch
            vram = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        s = np.array(self.sec)
        n_files = len(list(Path(self.out).glob('*.png'))) if self.out else len(s)  # resumed runs: count outputs on disk
        return dict(n_ok=max(len(s), n_files), n_timed=len(s), n_failed=len(self.errors), sec_per_crop_mean=float(s.mean()) if len(s) else None,
                    sec_per_crop_median=float(np.median(s)) if len(s) else None,
                    peak_vram_mb=max(self.prev_vram, vram),
                    errors=self.errors[:20], sec_list=[round(x, 4) for x in self.sec])


def save_summary(path, d):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=2))


def out_done(out_dir, name):
    return (Path(out_dir) / name).exists()
