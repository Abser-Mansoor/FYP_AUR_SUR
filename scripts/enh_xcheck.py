"""Phase 4 cross-device check: direct-feed embeddings on this GPU vs the lab-PC direct embeddings.

    envs\\score\\Scripts\\python.exe scripts\\enh_xcheck.py

Embeds the direct feed (native crop, bilinear to 112, flip-sum; the Phase 1/3 rule) of the tuning subset for the 4
recognizers, compares each crop with outputs/enhance/xcheck/lab_direct/<part>__<recognizer>.npy and writes
outputs/enhance/results/xcheck.json. Per-crop cosine must exceed 0.9999 for every recognizer, else exit code 1 (STOP).
The laptop embeddings are kept under outputs/enhance/xcheck/laptop_direct/ and do not touch the Phase 3 cache layout.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recognizer import Recognizer  # noqa: E402
from src.sr import eval as E  # noqa: E402
from src.utils import seed_all  # noqa: E402

THRESH = 0.9999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sr.yaml")
    ap.add_argument("--parts", nargs="+", default=["tuning"])
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    seed_all(cfg["seed"], True, True)
    lab = ROOT / "outputs/enhance/xcheck/lab_direct"
    mine = ROOT / "outputs/enhance/xcheck/laptop_direct"
    mine.mkdir(parents=True, exist_ok=True)
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "threshold": THRESH, "batch": a.batch, "parts": {}}
    ok = True
    for name, w in cfg["recognizers"].items():
        rec = Recognizer(name, ROOT / w, (112, 112), "bilinear", True, a.batch, "cuda")
        for part in a.parts:
            d, _ = E.load_subset(ROOT, part)
            native = ROOT / "outputs/sr/subset" / f"{part}_input"
            f = mine / f"{part}__{name}.npy"
            if f.exists():
                emb = np.load(f)
            else:
                emb = E.embed_dir({name: rec}, d, lambda r: native / r["file"])[name]
                np.save(f, emb)
            ref = np.load(lab / f"{part}__{name}.npy")
            assert emb.shape == ref.shape == (len(d), 512), (emb.shape, ref.shape, len(d))
            cos = (emb * ref).sum(1) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(ref, axis=1))
            passed = bool(cos.min() > THRESH)
            ok &= passed
            res["parts"].setdefault(part, {})[name] = dict(n=int(len(cos)), cos_min=float(cos.min()), cos_mean=float(cos.mean()),
                                                           n_below=int((cos <= THRESH).sum()), bit_identical=bool(np.array_equal(emb, ref)),
                                                           passed=passed)
            print(part, name, res["parts"][part][name], flush=True)
        del rec
        torch.cuda.empty_cache()
    res["passed"] = bool(ok)
    out = ROOT / "outputs/enhance/results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "xcheck.json").write_text(json.dumps(res, indent=2))
    print("XCHECK", "PASSED" if ok else "FAILED: STOP and report to Abser")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
