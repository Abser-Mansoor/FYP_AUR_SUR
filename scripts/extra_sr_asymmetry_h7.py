"""Exploratory H7 (quality inflation) for the SR asymmetry study, four variants that have full-size restored PNGs on this PC.

    envs/diag/Scripts/python.exe scripts/extra_sr_asymmetry_h7.py [--out outputs/sr/results/asymmetry_exploratory]

This is NEW MagFace inference on the restored PNGs (lab-PC GPU), requested by Abser after the pre-registered study stopped at H1; it is described in the
second addendum of REPORT_extra_sr_asymmetry.md, committed before this script ran. Variants: bicubic pre512, Real-ESRGAN native, InvSR pre64_st1,
TADSR pre128_t250 (OSDFace has only thumbnails and InvSR pre128_st1 has no stored PNGs, so they are not covered). Only the mated probes are embedded.
The raw (un-normalised) MagFace norm is measured exactly as in Phase 2 (`magface_norm_flip: false`: unflipped, bilinear for direct crops; restored 512-px outputs
are resized with INTER_AREA as in Phase 3). The same pass also builds the flip-summed normalised embedding, which must equal the cached Phase 3 embedding.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import extra_sr_asymmetry as A  # noqa: E402
from src.recognizer import Recognizer  # noqa: E402
from src.sr import common as C  # noqa: E402

H7_VARIANTS = [("bicubic", "pre512"), ("esrgan", "native"), ("invsr", "pre64_st1"), ("tadsr", "pre128_t250")]


@torch.no_grad()
def embed(rec, paths, interp, batch=128):
    """Returns (raw unflipped MagFace norm, flip-summed L2-normalised embedding) for every path."""
    rec.pre["interp"] = interp
    norms, embs = [], []
    for s in range(0, len(paths), batch):
        x = rec._batch_tensor([C.read_bgr(p) for p in paths[s:s + batch]])
        eu = rec._embed(x).float()
        ef = rec._embed(torch.flip(x, dims=[3])).float()
        norms.append(eu.norm(dim=1).cpu().numpy())
        embs.append(torch.nn.functional.normalize(eu + ef, dim=1).cpu().numpy())
    return np.concatenate(norms), np.concatenate(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/sr/results/asymmetry_exploratory"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = A.ctx(None)
    d, ix, cfg = c["d"], c["ix"], c["cfg"]
    probes = dict(tinyface=ix[("tinyface", "probe")], survface=ix[("survface", "mated_probe")])
    rows_all = np.concatenate([probes["tinyface"], probes["survface"]])
    rec = Recognizer("magface", ROOT / cfg["recognizers"]["magface"], (112, 112), "bilinear", True, 128, "cuda")
    native = ROOT / "outputs/sr/subset/eval_input"
    files = d["file"].to_numpy()[rows_all]
    cache_dir = A.CACHE
    cached = {None: np.load(cache_dir / "direct/resize/seed0/eval__magface.npy")}
    norm = {}
    checks = {}
    nd, ed = embed(rec, [native / f for f in files], "bilinear")
    norm["direct"] = nd
    checks["direct"] = float((ed * cached[None][rows_all]).sum(1).min())
    for arm, st in H7_VARIANTS:
        rdir = ROOT / "outputs/sr/restored/eval" / f"{arm}__{st}" / "seed0"
        n_, e_ = embed(rec, [rdir / f for f in files], "area")
        norm[f"{arm}/{st}"] = n_
        cached[(arm, st)] = np.load(cache_dir / arm / st / "seed0/eval__magface.npy")
        checks[f"{arm}/{st}"] = float((e_ * cached[(arm, st)][rows_all]).sum(1).min())
        print("embedded", arm, st, flush=True)
    # preprocessing checks: new embeddings against the cache, direct norm against the Phase 2 measure
    p2 = pd.read_csv(ROOT / "outputs/stratify/bins/tinyface_probe.csv")[["name", "magface_norm"]].drop_duplicates("name").set_index("name")["magface_norm"]
    names = d["name"].to_numpy()[probes["tinyface"]]
    ref = p2.reindex(names).to_numpy()
    mine = nd[:len(probes["tinyface"])]
    okm = np.isfinite(ref)
    checks["min_cosine_to_cached_embedding"] = {k: v for k, v in checks.items()}
    checks = dict(min_cosine_to_cached_embedding=checks["min_cosine_to_cached_embedding"],
                  tinyface_probe_direct_norm_vs_phase2=dict(n_compared=int(okm.sum()), max_abs_rel_diff=float(np.nanmax(np.abs(mine[okm] - ref[okm]) / ref[okm]))))
    (out / "h7_checks.json").write_text(json.dumps(checks, indent=1))
    df = pd.DataFrame(dict(dataset=["tinyface"] * len(probes["tinyface"]) + ["survface"] * len(probes["survface"]), name=d["name"].to_numpy()[rows_all]))
    for k, v in norm.items():
        df[f"norm_{k}"] = v
    df.to_csv(out / "h7_norms.csv", index=False, float_format="%.6g")
    # ---- analysis
    e0 = pd.read_csv(ROOT / "outputs/sr/results/asymmetry/asym_common_scale.csv")
    e0 = e0[(e0.recognizer == "magface") & (e0.metric == "verification_auc") & (e0["mode"] == "both")]
    tp, tg, sg, sm = ix[("tinyface", "probe")], ix[("tinyface", "gallery_match")], ix[("survface", "gallery")], ix[("survface", "mated_probe")]
    dire = cached[None]
    gen_d = dict(tinyface=A.gen_sim(dire, tp, tg), survface=A.gen_sim(dire, sm, sg))
    ntiny = len(tp)
    res = []
    for arm, st in H7_VARIANTS:
        R = cached[(arm, st)]
        gen_r = dict(tinyface=A.gen_sim(R, tp, tg), survface=A.gen_sim(R, sm, sg))
        pen = {r.dataset: r.delta for _, r in e0[(e0.arm == arm) & (e0.setting == st)].iterrows()}
        worse = min(pen, key=pen.get)
        for ds, sl in (("tinyface", slice(0, ntiny)), ("survface", slice(ntiny, None))):
            dn = norm[f"{arm}/{st}"][sl] - norm["direct"][sl]
            dg = gen_r[ds] - gen_d[ds]
            mn, lo_n, hi_n = A.boot_mean_ci(dn, 201)
            mg, lo_g, hi_g = A.boot_mean_ci(dg, 202)
            rel = dn / norm["direct"][sl]
            mr, lo_r, hi_r = A.boot_mean_ci(rel, 203)
            sp = A.spearman(dn, dg)
            res.append(dict(arm=arm, setting=st, dataset=ds, n=len(dn), larger_penalty_dataset=(ds == worse), delta_auc_magface_both=pen[ds], norm_direct_mean=float(norm["direct"][sl].mean()),
                            norm_restored_mean=float(norm[f"{arm}/{st}"][sl].mean()), delta_norm=mn, delta_norm_ci_lo=lo_n, delta_norm_ci_hi=hi_n, rel_delta_norm=mr, rel_ci_lo=lo_r, rel_ci_hi=hi_r,
                            delta_genuine_similarity=mg, delta_genuine_ci_lo=lo_g, delta_genuine_ci_hi=hi_g, spearman_dnorm_dgenuine=sp,
                            criterion_met=bool(ds == worse and lo_n > 0 and not (lo_g > 0))))
    r = pd.DataFrame(res)
    r.to_csv(out / "h7_summary.csv", index=False, float_format="%.8g")
    met = int(r.criterion_met.sum())
    v = pd.DataFrame([dict(hypothesis="H7 quality inflation (exploratory, 4 of 6 variants, mode both, MagFace)",
                           criterion="on the larger-penalty dataset the mean MagFace norm rises (CI above zero) and the mean genuine similarity does not rise (CI not entirely above zero)",
                           detail=f"{met}/{len(H7_VARIANTS)} variants", verdict=A.verdict(met, len(H7_VARIANTS)), status="exploratory (new inference, not pre-registered as run)")])
    v.to_csv(out / "h7_verdict.csv", index=False)
    print(r[["arm", "setting", "dataset", "larger_penalty_dataset", "delta_norm", "delta_norm_ci_lo", "delta_norm_ci_hi", "delta_genuine_similarity", "criterion_met"]].round(4).to_string())
    print(v[["detail", "verdict"]].to_string())
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
