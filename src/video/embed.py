"""Align + embed + quality measures for detected faces (the Phase 6 embedding stage).

One rule for every face: 5-point similarity alignment to 112 px (`src/align.py:norm_crop`), then the Phase 1 recognizer path
(`src/recognizer.py:Recognizer`, bilinear, flip-sum). The MagFace raw norm is measured as in Phase 2 (unflipped); per-face luminance and Laplacian
variance as in `src/stratify/measures.py` (on the aligned crop).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.align import norm_crop  # noqa: E402
from src.recognizer import Recognizer  # noqa: E402
from src.stratify.measures import laplacian_variance, luminance  # noqa: E402


class FaceEmbedder:
    def __init__(self, cfg, recognizers=("adaface",), device="cuda"):
        """cfg: the `recognizers` (name -> weight path), `preprocess` and `batch_size` entries of configs/video.yaml."""
        pre, bs = cfg["preprocess"], cfg.get("batch_size", 128)
        make = lambda n: Recognizer(n, ROOT / cfg["recognizers"][n], pre["input_size"], pre["interp"], pre["flip"], bs, device)  # noqa: E731
        self.recs = {n: make(n) for n in recognizers}
        self.mag = self.recs.get("magface") or make("magface")

    @staticmethod
    def crops(bgr, kps):
        return [norm_crop(bgr, k) for k in kps]

    def embed_crops(self, crops):
        """crops: list of aligned 112x112 BGR faces. Returns dict: emb {name: [N,512]}, mag_norm [N], lum [N], lap_var [N]."""
        out = {"emb": {}}
        for n, rec in self.recs.items():
            if n == "magface":
                out["emb"][n], out["mag_norm"] = rec.get_embedding(crops, return_raw=True)
            else:
                out["emb"][n] = rec.get_embedding(crops)
        if "mag_norm" not in out:
            out["mag_norm"] = self.mag.get_embedding(crops, return_raw=True)[1]
        out["lum"] = np.array([luminance(c) for c in crops], np.float32)
        out["lap_var"] = np.array([laplacian_variance(c) for c in crops], np.float32)
        return out

    def embed(self, bgr, kps):
        if len(kps) == 0:
            return dict(emb={n: np.zeros((0, 512), np.float32) for n in self.recs}, mag_norm=np.zeros(0, np.float32),
                        lum=np.zeros(0, np.float32), lap_var=np.zeros(0, np.float32))
        return self.embed_crops(self.crops(bgr, kps))
