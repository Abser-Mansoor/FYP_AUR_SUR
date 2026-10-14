"""Common recognizer interface: Recognizer.get_embedding(images) -> L2-normalized [N, 512].

Preprocessing (tiny-crop protocol) is one rule for every model and every
restoration variant: resize the crop to `input_size` with `interp`, then
apply the model's official channel order and normalization (models.OFFICIAL),
optionally sum with the horizontally flipped embedding (`flip`).
"""
import hashlib
import json

import cv2
import numpy as np
import torch

from .models import OFFICIAL, load_recognizer
from .preprocess import INTERP


class Recognizer:
    def __init__(self, name, weights, input_size=(112, 112), interp="bilinear", flip=True, batch_size=256, device="cuda"):
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.pre = dict(input_size=list(input_size), interp=interp, flip=bool(flip), **OFFICIAL[name])
        self._embed = load_recognizer(name, weights, device)

    @property
    def pre_hash(self):
        return hashlib.sha1(json.dumps(self.pre, sort_keys=True).encode()).hexdigest()[:10]

    def _batch_tensor(self, imgs):
        w, h = self.pre["input_size"]
        x = np.stack([cv2.resize(im, (w, h), interpolation=INTERP[self.pre["interp"]]) for im in imgs])  # N,H,W,3 BGR uint8
        if self.pre["channel"] == "RGB":
            x = x[..., ::-1]
        x = torch.from_numpy(np.ascontiguousarray(x)).to(self.device).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        if self.pre["norm"] == "pm1":
            x = x.sub_(0.5).div_(0.5)
        return x

    @torch.no_grad()
    def get_embedding(self, images):
        """images: list of HxWx3 BGR uint8 arrays (any size). Returns float32 numpy [N, 512], unit L2 norm."""
        out = []
        for s in range(0, len(images), self.batch_size):
            x = self._batch_tensor(images[s:s + self.batch_size])
            e = self._embed(x).float()
            if self.pre["flip"]:
                e = e + self._embed(torch.flip(x, dims=[3])).float()
            out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)
