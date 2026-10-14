"""Phase 1 diagnostic: 5-point aligned SurvFace (open-set 1:N + verification 1:1). NOT the baseline.

    envs/diag/Scripts/python.exe scripts/run_aligned_survface.py --config configs/baseline.yaml

Reuses run_baseline's data/embedding code. Writes aligned_survface_results.{csv,json},
per_sample/<model>_survface_*_aligned.npz and aligned_survface.log under paths.results;
baseline_results.* is not touched.
"""
import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_baseline as rb  # noqa: E402
from run_baseline import ROOT, ev, metrics, Recognizer, ResultsLog, Timer, seed_all  # noqa: E402

log = logging.getLogger("aligned_survface")
SPLITS = ("gallery", "mated_probe", "unmated_probe", "verification")


def aligned_survface_images(cfg, emb):
    from src.align import SCRFD, detect_landmarks, norm_crop
    det = SCRFD(str(ROOT / cfg["paths"]["detector"]))
    dg = cfg["diagnostics"]
    out, rates = {}, {}
    for split in SPLITS:
        f = ROOT / cfg["paths"]["cache"] / f"landmarks_survface_{split}.npz"
        imgs = emb.imgs("survface", split)
        if f.exists():
            z = np.load(f)
            kps, ok = z["kps"], z["ok"]
        else:
            t = Timer()
            with t(), ThreadPoolExecutor(6) as ex:
                res = list(ex.map(lambda im: detect_landmarks(det, im, dg["align_lowres_side"], dg["align_det_thresh"]), imgs, chunksize=256))
            ok = np.array([k is not None for k in res])
            kps = np.stack([k if k is not None else np.zeros((5, 2), np.float32) for k in res]).astype(np.float32)
            np.savez(f, kps=kps, ok=ok)
            log.info(f"  landmarks survface/{split}: rate {ok.mean():.4f} in {t.sec:.0f}s ({det.sess.get_providers()[0]})")
        out[split] = [norm_crop(im, k) if o else im for im, k, o in zip(imgs, kps, ok)]
        rates[split] = float(ok.mean())
        emb.images.pop(("survface", split), None)  # keep RAM bounded; aligned crops are kept in `out`
    return out, rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    cfg["datasets"] = ["survface"]
    res_dir = ROOT / cfg["paths"]["results"]
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        handlers=[logging.FileHandler(res_dir / "aligned_survface.log"), logging.StreamHandler(sys.stdout)])
    seed_all(cfg["seed"], cfg["deterministic"], cfg.get("tf32", True))
    B, S = cfg["metrics"]["bootstrap_n"], cfg["metrics"]["bootstrap_seed"]
    for f in ("aligned_survface_results.csv", "aligned_survface_results.json"):
        (res_dir / f).unlink(missing_ok=True)
    results = ResultsLog(res_dir / "aligned_survface_results.csv", res_dir / "aligned_survface_results.json")

    data = rb.build_data(cfg)
    emb = rb.Embeddings(cfg, data, use_cache=True)
    sf, pairs = data["survface"], data["_survface_pairs"]
    aligned, rates = aligned_survface_images(cfg, emb)
    log.info(f"landmark detection rates: {json.dumps(rates)}")
    idx = {p.name: i for i, p in enumerate(sf["verification"][0])}

    for name, wpath in cfg["recognizers"].items():
        wh = rb.sha256_prefix(ROOT / wpath)
        base = dict(recognizer=name, weights=wpath, weights_sha256=wh, variant="aligned_DIAGNOSTIC", dataset="SurvFace", det_rate=rates)
        rec = Recognizer(name, ROOT / wpath, cfg["preprocess"]["input_size"], cfg["preprocess"]["interp"],
                         cfg["preprocess"]["flip"], cfg["batch_size"], cfg["device"])
        try:
            G, ig = emb.get(rec, wh, "survface", "gallery", "aligned", aligned["gallery"])
            M, im = emb.get(rec, wh, "survface", "mated_probe", "aligned", aligned["mated_probe"])
            U, iu = emb.get(rec, wh, "survface", "unmated_probe", "aligned", aligned["unmated_probe"])
            V, iv = emb.get(rec, wh, "survface", "verification", "aligned", aligned["verification"])
            sel = lambda k, c: V[[idx[p[c]] for p in pairs[k]]]  # noqa: E731
            r_open = ev.survface_openset(G, sf["gallery"][1], M, sf["mated_probe"][1], U)
            r_ver = ev.survface_verification(sel("pos", 0), sel("pos", 1), sel("neg", 0), sel("neg", 1))
            for proto, r, key, infos in (("OpenSetIdentification", r_open, "survface_openset", [ig, im, iu]),
                                         ("Verification", r_ver, "survface_verification", [iv])):
                ps = r.pop("_per_sample")
                np.savez(res_dir / "per_sample" / f"{name}_{key}_aligned.npz", **ps)
                results.add({**base, "protocol": proto, "status": "OK", **r,
                             **{f"{k}_ci95": v for k, v in metrics.protocol_ci(key, ps, B, S).items()},
                             **{f"sep_{k}": v for k, v in metrics.separation(ps["genuine"], ps["impostor"], B, S).items()},
                             **rb.summarize_infos(infos)})
            log.info(f"{name}/survface ALIGNED (diagnostic): TPIR20@FPIR=0.3 {r_open['TPIR20@FPIR=0.3']:.4f} | "
                     f"ver AUC {r_ver['AUC']:.4f} TAR@FAR=0.1 {r_ver['TAR@FAR=0.1']:.4f}")
        except Exception as e:
            log.exception(f"{name}: aligned survface failed")
            results.add({**base, "status": f"FAILED: {type(e).__name__}: {e}"})
        del rec
        torch.cuda.empty_cache()
    log.info("Aligned SurvFace diagnostic complete.")


if __name__ == "__main__":
    main()
