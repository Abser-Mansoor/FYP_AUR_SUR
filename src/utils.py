"""Seeding, timing, VRAM measurement and results logging."""
import csv
import json
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def seed_all(seed, deterministic=True, tf32=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # TF32 convs are PyTorch's default; deterministic run-to-run on the same GPU.
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32


class Timer:
    def __init__(self):
        self.sec = 0.0

    @contextmanager
    def __call__(self):
        t = time.perf_counter()
        try:
            yield self
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.sec += time.perf_counter() - t


@contextmanager
def vram_peak(out):
    """Write peak torch.cuda.max_memory_allocated (MB) for the block into out['vram_mb']."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        out["vram_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if torch.cuda.is_available() else None


def env_info():
    return {
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(), "time": datetime.now().isoformat(timespec="seconds"),
    }


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


class ResultsLog:
    """Append rows to a CSV (union of columns, rewritten on new columns) and a JSON list."""

    def __init__(self, csv_path, json_path):
        self.csv_path, self.json_path = Path(csv_path), Path(json_path)
        self.rows = json.loads(self.json_path.read_text()) if self.json_path.exists() else []

    def add(self, row):
        row = {k: _jsonable(v) for k, v in row.items()}
        self.rows.append(row)
        self.json_path.write_text(json.dumps(self.rows, indent=2))
        cols = []
        for r in self.rows:
            cols += [k for k in r if k not in cols]
        with open(self.csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
