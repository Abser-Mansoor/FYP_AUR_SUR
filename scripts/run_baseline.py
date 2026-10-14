"""Phase 1 (Task 0): direct-feed baseline on TinyFace and QMUL-SurvFace for all
recognizers, with bootstrap CIs, genuine/impostor separation and sanity tests.

    envs/diag/Scripts/python.exe scripts/run_baseline.py --config configs/baseline.yaml [--no-cache]

Outputs (paths.results): baseline_results.{csv,json}, sanity.{csv,json},
per_sample/<model>_<protocol>.npz, run.log. Embeddings are cached under
paths.cache keyed by recognizer, weights hash, preprocessing hash and split.
"""
import argparse
import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import datasets, eval_protocols as ev, metrics  # noqa: E402
from src.preprocess import load_bgr  # noqa: E402
from src.recognizer import Recognizer  # noqa: E402
from src.utils import ResultsLog, Timer, env_info, seed_all, vram_peak  # noqa: E402

log = logging.getLogger("baseline")


def sha256_prefix(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:n]


def load_images(paths, threads):
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(load_bgr, paths, chunksize=512))


# ------------------------------------------------------------------ data
def _smoke(d, n=600):
    """Truncate identification splits for a quick end-to-end check (results are NOT valid)."""
    for ds, splits in d.items():
        if ds.startswith("_"):
            continue
        for s, (p, ids) in list(splits.items()):
            if s != "verification":
                splits[s] = (p[:n], None if ids is None else ids[:n])
    if "survface" in d:  # mated queries must have their ID in the gallery
        g = set(d["survface"]["gallery"][1].tolist())
        p, ids = d["survface"]["mated_probe"]
        keep = [i for i, x in enumerate(ids) if x in g] or [0]
        d["survface"]["mated_probe"] = ([p[i] for i in keep], ids[keep])
    if "tinyface" in d:
        g = set(d["tinyface"]["gallery_match"][1].tolist())
        p, ids = d["tinyface"]["probe"]
        keep = [i for i, x in enumerate(ids) if x in g]
        d["tinyface"]["probe"] = ([p[i] for i in keep], ids[keep])
    return d


def build_data(cfg):
    """Official splits -> {dataset: {split: (paths, ids_or_None)}}, plus SurvFace verification pairs.
    Galleries are fixed here once and shared by every recognizer and variant."""
    d = {}
    if "tinyface" in cfg["datasets"]:
        d["tinyface"] = datasets.tinyface(ROOT / cfg["paths"]["tinyface"])
    if "survface" in cfg["datasets"]:
        sf = datasets.survface_identification(ROOT / cfg["paths"]["survface"])
        vpaths, pairs = datasets.survface_verification(ROOT / cfg["paths"]["survface"])
        sf["verification"] = (vpaths, None)
        d["survface"] = sf
        d["_survface_pairs"] = pairs
    return d


class Embeddings:
    """Embed each (dataset, split) once per recognizer, with an on-disk cache."""

    def __init__(self, cfg, data, use_cache):
        self.cfg, self.data, self.use_cache = cfg, data, use_cache
        self.cache_dir = ROOT / cfg["paths"]["cache"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.images = {}

    def imgs(self, ds, split):
        key = (ds, split)
        if key not in self.images:
            paths = self.data[ds][split][0]
            t = Timer()
            with t():
                self.images[key] = load_images(paths, self.cfg["decode_threads"])
            log.info(f"  decoded {ds}/{split}: {len(paths)} images in {t.sec:.1f}s")
        return self.images[key]

    def get(self, rec, wh, ds, split, variant="direct", images=None):
        paths = self.data[ds][split][0]
        stem = f"{rec.name}_{wh}_{rec.pre_hash}_{variant}_{ds}_{split}"
        f, meta = self.cache_dir / f"{stem}.npy", self.cache_dir / f"{stem}.json"
        if self.use_cache and f.exists():
            E = np.load(f)
            assert json.loads(meta.read_text())["names"] == [p.name for p in paths], f"cache order mismatch: {f}"
            return E, {"cached": True}
        info = {}
        t = Timer()
        imgs = images if images is not None else self.imgs(ds, split)
        with vram_peak(info), t():
            E = rec.get_embedding(imgs)
        np.save(f, E)
        meta.write_text(json.dumps({"recognizer": rec.name, "weights_sha256": wh, "preprocess": rec.pre, "variant": variant,
                                    "dataset": ds, "split": split, "n": len(paths), "names": [p.name for p in paths]}))
        info.update(cached=False, sec=round(t.sec, 1), img_per_sec=round(len(paths) / max(t.sec, 1e-9), 1))
        log.info(f"  embedded {ds}/{split} [{variant}]: {len(paths)} in {t.sec:.1f}s, peak VRAM {info['vram_mb']} MB")
        return E, info


# --------------------------------------------------------------- protocols
def eval_tinyface(emb, rec, wh, data, variant="direct", images=None):
    tf = data["tinyface"]
    get = lambda s: emb.get(rec, wh, "tinyface", s, variant, None if images is None else images[s])  # noqa: E731
    (P, ip), (G, ig), (D, idd) = get("probe"), get("gallery_match"), get("distractor")
    r = ev.tinyface_identification(P, tf["probe"][1], G, tf["gallery_match"][1], D)
    return r, {"probe": P, "gallery_match": G, "distractor": D}, [ip, ig, idd]


def eval_survface(emb, rec, wh, data):
    sf, pairs = data["survface"], data["_survface_pairs"]
    (G, ig), (M, im), (U, iu) = (emb.get(rec, wh, "survface", s) for s in ("gallery", "mated_probe", "unmated_probe"))
    r_open = ev.survface_openset(G, sf["gallery"][1], M, sf["mated_probe"][1], U)
    V, iv = emb.get(rec, wh, "survface", "verification")
    idx = {p.name: i for i, p in enumerate(sf["verification"][0])}
    sel = lambda k, c: V[[idx[p[c]] for p in pairs[k]]]  # noqa: E731
    r_ver = ev.survface_verification(sel("pos", 0), sel("pos", 1), sel("neg", 0), sel("neg", 1))
    return r_open, r_ver, [ig, im, iu, iv]


def summarize_infos(infos):
    fresh = [i for i in infos if not i.get("cached")]
    return {"embed_sec": round(sum(i["sec"] for i in fresh), 1) if fresh else None,
            "vram_mb": max((i["vram_mb"] for i in fresh), default=None), "from_cache": not fresh}


# ------------------------------------------------------------------ sanity
def sanity_tests(rec, emb, wh, data, tf_result, ver_result, cfg, res_dir):
    out = {"recognizer": rec.name}
    imgs = emb.imgs("tinyface", "probe")[:64] if "tinyface" in data else emb.imgs("survface", "verification")[:64]
    # 1. identical images -> cosine 1.0 (same crop twice in one batch, and across two calls)
    e = rec.get_embedding(imgs + imgs)
    e2 = rec.get_embedding(imgs)
    c_in = (e[:64] * e[64:]).sum(1)
    c_x = (e[:64] * e2).sum(1)
    out.update(identical_cos_min=float(min(c_in.min(), c_x.min())), identical_pass=bool(min(c_in.min(), c_x.min()) > 0.9999))
    # 2. determinism: re-embed a subset and compare with the stored embeddings
    # same batch composition as the stored pass (cuDNN kernel choice depends on batch shape)
    n = min(cfg["batch_size"], len(data["tinyface"]["probe"][0])) if "tinyface" in data else 0
    if "tinyface" in data:
        stored = emb.get(rec, wh, "tinyface", "probe")[0][:n]
        fresh = rec.get_embedding(emb.imgs("tinyface", "probe")[:n])
        out["determinism_max_abs_diff"] = float(np.abs(stored - fresh).max())
        out["determinism_pass"] = bool(out["determinism_max_abs_diff"] < 1e-4)
    rng = np.random.default_rng(cfg["sanity"]["shuffled_label_seed"])
    # 3. shuffled labels -> chance. TinyFace: permute probe IDs.
    if tf_result is not None:
        tf = data["tinyface"]
        E = tf_result[1]
        pid = rng.permutation(tf["probe"][1])
        r = ev.tinyface_identification(E["probe"], pid, E["gallery_match"], tf["gallery_match"][1], E["distractor"])
        gal_ids = tf["gallery_match"][1]
        n_gal = len(gal_ids) + len(E["distractor"])
        counts = {i: c for i, c in zip(*np.unique(gal_ids, return_counts=True))}
        chance_r1 = float(np.mean([counts.get(i, 0) / n_gal for i in pid]))
        out.update(shuffled_tinyface_rank1=r["Rank-1"], shuffled_tinyface_chance_rank1=chance_r1,
                   shuffled_tinyface_pass=bool(r["Rank-1"] < 0.01))
    # SurvFace verification: shuffle the pos/neg labels over all pairs -> AUC ~ 0.5
    if ver_result is not None:
        ps = ver_result["_per_sample"]
        allp = np.concatenate([ps["pos_dist"], ps["neg_dist"]])
        perm = rng.permutation(len(allp))
        k = len(ps["pos_dist"])
        r = ev.survface_verification_from_scores(allp[perm[:k]], allp[perm[k:]])
        out.update(shuffled_survface_ver_auc=r["AUC"], shuffled_survface_ver_acc=r["mean_accuracy"],
                   shuffled_survface_pass=bool(abs(r["AUC"] - 0.5) < 0.03))
    # 4. comparison utilities on identical inputs: drift == 1, paired delta == 0
    if tf_result is not None:
        E = tf_result[1]["probe"]
        d, _ = metrics.embedding_drift(E, E, n=100)
        pdlt = metrics.paired_delta(tf_result[0]["_per_sample"]["ap"], tf_result[0]["_per_sample"]["ap"], n=100)
        out.update(self_drift=d["mean"], self_paired_delta=pdlt["mean"],
                   self_compare_pass=bool(abs(d["mean"] - 1) < 1e-5 and pdlt["mean"] == 0))
    out["all_pass"] = all(v for k, v in out.items() if k.endswith("_pass"))
    return out


# ----------------------------------------------------------- aligned arm
def aligned_images(cfg, emb, data):
    """5-point aligned 112x112 crops for every TinyFace split (fallback: original crop). Diagnostic only."""
    from src.align import SCRFD, detect_landmarks, norm_crop
    det = SCRFD(str(ROOT / cfg["paths"]["detector"]))
    out, rates = {}, {}
    for split in ("probe", "gallery_match", "distractor"):
        f = ROOT / cfg["paths"]["cache"] / f"landmarks_tinyface_{split}.npz"
        imgs = emb.imgs("tinyface", split)
        if f.exists():
            z = np.load(f); kps, ok = z["kps"], z["ok"]
        else:
            t = Timer()
            kps, ok = np.zeros((len(imgs), 5, 2), np.float32), np.zeros(len(imgs), bool)
            with t():
                for i, im in enumerate(imgs):
                    k = detect_landmarks(det, im, cfg["diagnostics"]["align_lowres_side"], cfg["diagnostics"]["align_det_thresh"])
                    if k is not None:
                        kps[i], ok[i] = k, True
            np.savez(f, kps=kps, ok=ok)
            log.info(f"  landmarks tinyface/{split}: rate {ok.mean():.4f} in {t.sec:.0f}s")
        out[split] = [norm_crop(im, k) if o else im for im, k, o in zip(imgs, kps, ok)]
        rates[split] = float(ok.mean())
    return out, rates


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--no-cache", action="store_true", help="recompute all embeddings")
    ap.add_argument("--smoke", action="store_true", help="tiny subsets, separate output dirs; numbers are not valid")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    if args.smoke:
        cfg["paths"]["cache"] += "_smoke"
        cfg["paths"]["results"] += "_smoke"
        cfg["metrics"]["bootstrap_n"] = 50
    res_dir = ROOT / cfg["paths"]["results"]
    (res_dir / "per_sample").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        handlers=[logging.FileHandler(res_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    seed_all(cfg["seed"], cfg["deterministic"], cfg.get("tf32", True))
    env = env_info()
    log.info("=" * 70)
    log.info(f"Phase 1 baseline | config {args.config} | env {json.dumps(env)}")
    log.info(f"preprocess {json.dumps(cfg['preprocess'])} | seed {cfg['seed']} | bootstrap n={cfg['metrics']['bootstrap_n']}")
    for f in ("baseline_results.csv", "baseline_results.json", "sanity.csv", "sanity.json"):
        (res_dir / f).unlink(missing_ok=True)  # fresh tables every run; embeddings may come from cache
    results = ResultsLog(res_dir / "baseline_results.csv", res_dir / "baseline_results.json")
    sanity = ResultsLog(res_dir / "sanity.csv", res_dir / "sanity.json")

    data = build_data(cfg)
    if args.smoke:
        data = _smoke(data)
    emb = Embeddings(cfg, data, use_cache=not args.no_cache)
    B, S = cfg["metrics"]["bootstrap_n"], cfg["metrics"]["bootstrap_seed"]
    aligned, align_rates = None, None

    for name, wpath in cfg["recognizers"].items():
        wh = sha256_prefix(ROOT / wpath)
        base = dict(recognizer=name, weights=wpath, weights_sha256=wh, variant="direct", **{f"pre_{k}": v for k, v in cfg["preprocess"].items()})
        log.info("-" * 70)
        try:
            rec = Recognizer(name, ROOT / wpath, cfg["preprocess"]["input_size"], cfg["preprocess"]["interp"],
                             cfg["preprocess"]["flip"], cfg["batch_size"], cfg["device"])
            log.info(f"{name}: weights sha256 {wh}, preprocessing {json.dumps(rec.pre)}")
        except Exception as e:  # log failures as results
            log.exception(f"{name}: load failed")
            results.add({**base, "status": f"LOAD_FAILED: {type(e).__name__}: {e}"})
            continue
        tf_res = ver_res = None
        for ds in cfg["datasets"]:
            try:
                if ds == "tinyface":
                    r, E, infos = eval_tinyface(emb, rec, wh, data)
                    ps = r.pop("_per_sample")
                    tf_res = ({**r, "_per_sample": ps}, E)  # sanity tests need the per-sample arrays
                    np.savez(res_dir / "per_sample" / f"{name}_tinyface_direct.npz", **ps)
                    results.add({**base, "dataset": "TinyFace", "protocol": "ClosedSetIdentification", "status": "OK", **r,
                                 **{f"{k}_ci95": v for k, v in metrics.protocol_ci("tinyface", ps, B, S).items()},
                                 **{f"sep_{k}": v for k, v in metrics.separation(ps["genuine"], ps["impostor"], B, S).items()},
                                 **summarize_infos(infos)})
                else:
                    r_open, r_ver, infos = eval_survface(emb, rec, wh, data)
                    ver_res = dict(r_ver)
                    for proto, r, key in (("OpenSetIdentification", r_open, "survface_openset"), ("Verification", r_ver, "survface_verification")):
                        ps = r.pop("_per_sample")
                        np.savez(res_dir / "per_sample" / f"{name}_{key}_direct.npz", **ps)
                        results.add({**base, "dataset": "SurvFace", "protocol": proto, "status": "OK", **r,
                                     **{f"{k}_ci95": v for k, v in metrics.protocol_ci(key, ps, B, S).items()},
                                     **{f"sep_{k}": v for k, v in metrics.separation(ps["genuine"], ps["impostor"], B, S).items()},
                                     **summarize_infos(infos)})
                log.info(f"{name}/{ds}: " + json.dumps({k: round(v, 4) for k, v in results.rows[-1].items()
                                                        if isinstance(v, float) and not k.startswith(("sep_", "embed", "vram"))}))
            except Exception as e:
                log.exception(f"{name}/{ds}: failed")
                results.add({**base, "dataset": ds, "status": f"FAILED: {type(e).__name__}: {e}"})

        try:
            row = sanity_tests(rec, emb, wh, data, tf_res, ver_res, cfg, res_dir)
            sanity.add(row)
            log.info(f"{name} sanity: {json.dumps(row)}")
        except Exception as e:
            log.exception(f"{name}: sanity failed")
            sanity.add({"recognizer": name, "status": f"FAILED: {type(e).__name__}: {e}"})

        if cfg["diagnostics"].get("aligned_tinyface") and "tinyface" in cfg["datasets"]:
            try:
                if aligned is None:
                    aligned, align_rates = aligned_images(cfg, emb, data)
                r, E, infos = eval_tinyface(emb, rec, wh, data, variant="aligned", images=aligned)
                ps = r.pop("_per_sample")
                np.savez(res_dir / "per_sample" / f"{name}_tinyface_aligned.npz", **ps)
                direct = np.load(res_dir / "per_sample" / f"{name}_tinyface_direct.npz")
                drift, _ = metrics.embedding_drift(tf_res[1]["probe"], E["probe"], B, S)
                results.add({**base, "variant": "aligned_DIAGNOSTIC", "dataset": "TinyFace", "protocol": "ClosedSetIdentification",
                             "status": "OK", **r, "det_rate": align_rates,
                             **{f"{k}_ci95": v for k, v in metrics.protocol_ci("tinyface", ps, B, S).items()},
                             **{f"sep_{k}": v for k, v in metrics.separation(ps["genuine"], ps["impostor"], B, S).items()},
                             **{f"drift_{k}": v for k, v in drift.items()},
                             **{f"paired_rank1_{k}": v for k, v in metrics.paired_delta(direct["first_rank"] < 1, ps["first_rank"] < 1, B, S).items()},
                             **{f"paired_ap_{k}": v for k, v in metrics.paired_delta(direct["ap"], ps["ap"], B, S).items()},
                             **{f"impostor_shift_{k}": v for k, v in metrics.impostor_shift(direct["impostor"], ps["impostor"], B, S).items()},
                             **summarize_infos(infos)})
                log.info(f"{name}/tinyface ALIGNED (diagnostic): Rank-1 {r['Rank-1']:.4f} mAP {r['mAP']:.4f}")
            except Exception as e:
                log.exception(f"{name}: aligned diagnostic failed")
                results.add({**base, "variant": "aligned_DIAGNOSTIC", "dataset": "TinyFace", "status": f"FAILED: {type(e).__name__}: {e}"})

        del rec
        torch.cuda.empty_cache()
    log.info("Phase 1 baseline complete.")


if __name__ == "__main__":
    main()
