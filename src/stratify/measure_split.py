"""Measure every crop of one split (decode once; CPU measures + optional MagFace feature norm)."""
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from ..preprocess import load_bgr
from .measures import crop_measures

log = logging.getLogger("stratify")
CHUNK = 4096


def measure_split(paths, ids, cfg, mag=None):
    """Returns a DataFrame, one row per path in dataset order. Unreadable images are logged as rows
    (error column set, measures NaN), never dropped. `mag` = Recognizer for MagFace norms or None."""
    m = cfg["measures"]
    rows = []

    def one(p):
        try:
            im = load_bgr(p)
            return im, crop_measures(im, m["lap_size"], m["fft_cutoff"]), None
        except Exception as e:  # logged as a result
            return None, dict(h=np.nan, w=np.nan, min_side=np.nan, luminance=np.nan, lap_var=np.nan, fft_hf=np.nan), f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(cfg["decode_threads"]) as ex:
        for s in range(0, len(paths), CHUNK):
            res = list(ex.map(one, paths[s:s + CHUNK]))
            norms = np.full(len(res), np.nan)
            if mag is not None:
                ok = [i for i, r in enumerate(res) if r[0] is not None]
                with torch.no_grad():
                    for b in range(0, len(ok), mag.batch_size):
                        sel = ok[b:b + mag.batch_size]
                        x = mag._batch_tensor([res[i][0] for i in sel])
                        e = mag._embed(x).float()
                        if m["magface_norm_flip"]:
                            e = e + mag._embed(x.flip(3)).float()
                        norms[sel] = e.norm(dim=1).cpu().numpy()
            for i, (im, d, err) in enumerate(res):
                rows.append({**d, "magface_norm": norms[i], "error": err or ""})
            log.info(f"    measured {min(s + CHUNK, len(paths))}/{len(paths)}")
    df = pd.DataFrame(rows)
    df.insert(0, "name", [p.name for p in paths])
    df.insert(1, "path", [str(p) for p in paths])
    df.insert(2, "id", ids if ids is not None else -100)
    n_bad = int((df["error"] != "").sum())
    if n_bad:
        log.warning(f"    {n_bad} unreadable images logged with NaN measures")
    return df
