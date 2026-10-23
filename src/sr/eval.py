"""Evaluation of SR / restoration variants on the fixed subsets (runs in envs/diag).

For every variant (a folder of restored PNGs named like the subset's `file` column) and every recognizer:
  * embeddings (restored 512-px outputs are resized to 112 with INTER_AREA; direct feed keeps the Phase 1
    bilinear rule; both use the flip-sum), cached per variant,
  * TinyFace identification, SurvFace open-set identification and SurvFace verification on the subset,
  * bootstrap CIs, genuine/impostor separation, and paired changes vs direct feed on the SAME crops:
    embedding drift, impostor-score shift, paired Rank-1 / AP deltas.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import eval_protocols as ev, metrics
from . import common as C

NAMES = ("arcface", "adaface", "magface", "curricularface")
NON_LEAKY = ("adaface", "magface", "curricularface")  # selection / headline recognizers (ArcFace leaks into OSDFace)
LEAKY = {"osdface": ("arcface",)}
CHUNK = 512


def load_subset(root, part):
    d = pd.read_csv(root / "outputs/sr/subset" / f"{part}_images.csv")
    pairs = pd.read_csv(root / "outputs/sr/subset" / f"{part}_pairs.csv")
    return d, pairs


def embed_dir(recs, d, dirs, threads=16):
    """dirs: callable(row) -> Path of the image for that row. Returns {rec_name: [N,512]} in d order."""
    out = {n: [] for n in recs}
    paths = [dirs(r) for _, r in d.iterrows()]
    with ThreadPoolExecutor(threads) as ex:
        for s in range(0, len(paths), CHUNK):
            imgs = list(ex.map(C.read_bgr, paths[s:s + CHUNK]))
            for n, rec in recs.items():
                out[n].append(rec.get_embedding(imgs))
    return {n: np.concatenate(v) for n, v in out.items()}


def get_embeddings(recs, d, root, tag, part, restored_dir, restore_splits, cache_dir, direct, interp="area"):
    """Cached embeddings of a variant (tag) for every recognizer. direct=True: native crops, bilinear.
    Restored rows use `interp` (area for the 512-px SR outputs; bilinear for same-size enhancement), direct rows stay bilinear."""
    f = {n: cache_dir / tag / f"{part}__{n}.npy" for n in recs}
    if all(p.exists() for p in f.values()):
        return {n: np.load(p) for n, p in f.items()}
    native = root / "outputs/sr/subset" / f"{part}_input"
    def pick(r):
        if direct or r["split"] not in restore_splits:
            return native / r["file"]
        return restored_dir / r["file"]
    # rows not restored use the direct-feed rule even inside a restored variant
    emb = {n: np.zeros((len(d), 512), np.float32) for n in recs}
    for is_direct_rule, mode in ((True, "bilinear"), (False, interp)):
        sel = [(i, r) for i, (_, r) in enumerate(d.iterrows()) if ((direct or r["split"] not in restore_splits) == is_direct_rule)]
        if not sel:
            continue
        for rec in recs.values():
            rec.pre["interp"] = mode
        sub = d.iloc[[i for i, _ in sel]]
        e = embed_dir(recs, sub, pick)
        for n in recs:
            emb[n][[i for i, _ in sel]] = e[n]
    for n, p in f.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, emb[n])
    return emb


def protocols(emb, d, pairs, B, S):
    """Returns {protocol: (result dict without _per_sample, per_sample dict, extra)} for one recognizer's embeddings."""
    ix = {k: np.flatnonzero(((d["dataset"] == k[0]) & (d["split"] == k[1])).to_numpy()) for k in
          [("tinyface", "probe"), ("tinyface", "gallery_match"), ("tinyface", "distractor"), ("survface", "gallery"),
           ("survface", "mated_probe"), ("survface", "unmated_probe"), ("survface", "verification")]}
    g = lambda k: emb[ix[k]]  # noqa: E731
    ids = lambda k: d["id"].to_numpy()[ix[k]]  # noqa: E731
    res = {}
    r = ev.tinyface_identification(g(("tinyface", "probe")), ids(("tinyface", "probe")), g(("tinyface", "gallery_match")),
                                   ids(("tinyface", "gallery_match")), g(("tinyface", "distractor")))
    res["tinyface"] = (r, {"queries": g(("tinyface", "probe"))})
    r = ev.survface_openset(g(("survface", "gallery")), ids(("survface", "gallery")), g(("survface", "mated_probe")),
                            ids(("survface", "mated_probe")), g(("survface", "unmated_probe")))
    res["survface_openset"] = (r, {"queries": g(("survface", "mated_probe"))})
    names = d["name"].to_numpy()[ix[("survface", "verification")]]
    pos = {n: i for i, n in enumerate(names)}
    V = g(("survface", "verification"))
    sel = lambda k, c: V[[pos[x] for x in pairs.loc[pairs["kind"] == k, c]]]  # noqa: E731
    r = ev.survface_verification(sel("pos", "a"), sel("pos", "b"), sel("neg", "a"), sel("neg", "b"))
    res["survface_verification"] = (r, {"queries": V})
    return res


def evaluate_variant(recs, d, pairs, root, arm, setting, seed, part, restored_dir, restore_splits, cache_dir, B, S, direct_cache, interp="area"):
    """One result row per (recognizer x protocol). direct_cache: dict filled with the direct-feed per-sample arrays."""
    tag = f"{arm}/{setting}/seed{seed}"
    emb = get_embeddings(recs, d, root, tag, part, restored_dir, restore_splits, cache_dir, direct=(arm == "direct"), interp=interp)
    rows = []
    for rn in recs:
        res = protocols(emb[rn], d, pairs, B, S)
        for proto, (r, extra) in res.items():
            ps = r.pop("_per_sample")
            key = proto
            row = dict(arm=arm, setting=setting, seed=seed, part=part, recognizer=rn, leaky=rn in LEAKY.get(arm, ()), protocol=proto, **r)
            row.update({f"{k}_ci95": v for k, v in metrics.protocol_ci(key, ps, B, S).items()})
            row.update({f"sep_{k}": v for k, v in metrics.separation(ps["genuine"], ps["impostor"], B, S).items()})
            if arm == "direct":
                direct_cache[(rn, proto)] = (ps, extra["queries"])
                if proto == "tinyface":
                    np.savez(cache_dir / tag / f"{part}__{rn}__tinyface_ps.npz", **ps)
            elif (rn, proto) in direct_cache:
                dps, dq = direct_cache[(rn, proto)]
                drift, _ = metrics.embedding_drift(dq, extra["queries"], B, S)
                row.update({f"drift_{k}": v for k, v in drift.items()})
                row.update({f"impostor_shift_{k}": v for k, v in metrics.impostor_shift(dps["impostor"], ps["impostor"], B, S).items()
                            if k in ("mean", "ci95", "n")})
                if proto == "tinyface":
                    row.update({f"paired_rank1_{k}": v for k, v in metrics.paired_delta(dps["first_rank"] < 1, ps["first_rank"] < 1, B, S).items()})
                    row.update({f"paired_ap_{k}": v for k, v in metrics.paired_delta(dps["ap"], ps["ap"], B, S).items()})
                    np.savez(cache_dir / tag / f"{part}__{rn}__tinyface_ps.npz", **ps)
            rows.append(row)
    return rows


def timing_info(restored_dir):
    p = Path(restored_dir) / "timing.json"
    if not p.exists():
        return {}
    t = json.loads(p.read_text())
    return {k: t.get(k) for k in ("sec_per_crop_median", "sec_per_crop_mean", "peak_vram_mb", "n_ok", "n_failed")}
