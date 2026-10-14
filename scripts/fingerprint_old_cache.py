"""Identify the preprocessing the wiped Phase 1 code used: re-embed the first
probe images under candidate pipelines and compare with the old cached
embeddings (cosine ~1.0 = same pipeline)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import datasets  # noqa: E402
from src.models import load_recognizer  # noqa: E402
from src.preprocess import extract, load_bgr  # noqa: E402
import yaml  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "configs/diagnostics.yaml"))
N = 64


def pil_bicubic(paths):
    # torchvision/PIL-style resize, returned as BGR arrays already 112x112
    return [np.asarray(Image.open(p).convert("RGB").resize((112, 112), Image.BICUBIC))[:, :, ::-1].copy() for p in paths]


def main():
    tf = datasets.tinyface(ROOT / CFG["paths"]["tinyface"])
    paths = tf["probe"][0][:N]
    cv_imgs = [load_bgr(p) for p in paths]
    pil_imgs = pil_bicubic(paths)
    for m, w in CFG["recognizers"].items():
        d = Path(glob.glob(str(ROOT / f"outputs/cache/{m}_tinyface_probe_*"))[0])
        old = np.load(d / "embeddings.npy")
        names = json.load(open(d / "names.json"))
        idx = {n: i for i, n in enumerate(names)}
        old = old[[idx[p.name] for p in paths]]
        embed = load_recognizer(m, ROOT / w)
        res = []
        for resize, imgs in (("cv2_bicubic", cv_imgs), ("pil_bicubic", pil_imgs)):
            for ch in ("RGB", "BGR"):
                for nm in ("pm1", "01"):
                    for fl in (False, True):
                        e = extract(embed, imgs, dict(channel=ch, norm=nm, interp="bicubic", flip=fl))
                        e /= np.linalg.norm(e, axis=1, keepdims=True)
                        res.append((float((e * old).sum(1).mean()), resize, ch, nm, fl))
        res.sort(reverse=True)
        print(m, "best matches (mean cosine to old cache):")
        for r in res[:3]:
            print("   %.5f  %s %s %s flip=%s" % r)
        del embed; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
