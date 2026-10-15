"""Follow-up analyses for Phase 3 (no restoration, no new GPU inference): cached embeddings only.

    envs/diag/Scripts/python.exe scripts/sr_extra_analysis.py [--workers 4]

A. Probes-only variant ("restore_gallery: false"): probes / queries use the restored embeddings, the gallery,
   distractors and templates keep the direct-feed embeddings. TinyFace identification and SurvFace open-set are
   defined directly. SurvFace verification has no probe/gallery roles, so the first image of each pair (a) is
   restored and the second (b) is direct ("one side restored"; the a/b roles are arbitrary).
B. Per-bin breakdown of the existing both-sides-restored results with the Phase 2 bins stored in the subset CSV:
   `low_sharpness_sz` (size-controlled low sharpness) and `dark10` (darkest 10%), each with its complement.
   Verification pairs use the worse-image rule of Phase 2 (a pair is "low"/"dark" if either image is).

Outputs: outputs/sr/results/probes_only.csv, per_bin.csv, extra_tables.md. Arms and settings come from the
selection files (plus InvSR pre-128). Bootstrap: 1,000 draws, seed 42, as everywhere else.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

RECS = ["arcface", "adaface", "magface", "curricularface"]
LEAKY = {("osdface", "arcface")}
RES = ROOT / "outputs/sr/results"
SLICES = [("full", None, None), ("low_sharpness", "sharp_bin_sz", "low_sharpness"), ("not_low_sharpness", "sharp_bin_sz", "NOT low_sharpness"),
          ("dark10", "lum_bin10", "dark10"), ("not_dark10", "lum_bin10", "NOT dark10")]
QUERY_SPLITS = ("probe", "mated_probe", "unmated_probe")


def variants():
    out = []
    for arm in ("bicubic", "esrgan", "invsr", "tadsr", "osdface"):
        f = RES / f"selection_{arm}.json"
        if f.exists() and json.loads(f.read_text()):
            out.append((arm, json.loads(f.read_text())[0]["setting"]))
    out.append(("invsr", "pre128_st1"))
    return out


def _flag(d, ix, col, val):
    s = d[col].to_numpy()[ix].astype(str)
    return (s != val[4:]) if val.startswith("NOT ") else (s == val)


def worker(arm, setting, B, S):
    """All rows for one (arm, setting): probes-only analysis and per-bin analysis."""
    from src import eval_protocols as ev, metrics
    from src.sr import eval as E
    d, pairs = E.load_subset(ROOT, "eval")
    ix = {k: np.flatnonzero(((d["dataset"] == k[0]) & (d["split"] == k[1])).to_numpy()) for k in
          [("tinyface", "probe"), ("survface", "mated_probe"), ("survface", "unmated_probe"), ("survface", "verification")]}
    q_rows = np.concatenate([np.flatnonzero((d["split"] == s).to_numpy()) for s in QUERY_SPLITS])
    vnames = d["name"].to_numpy()[ix[("survface", "verification")]]
    vpos = {n: i for i, n in enumerate(vnames)}
    ai = {k: np.array([vpos[x] for x in pairs.loc[pairs["kind"] == k, "a"]]) for k in ("pos", "neg")}
    bi = {k: np.array([vpos[x] for x in pairs.loc[pairs["kind"] == k, "b"]]) for k in ("pos", "neg")}
    cache = ROOT / "outputs/sr/emb/eval"
    probes_rows, bin_rows = [], []
    for rec in RECS:
        dire = np.load(cache / "direct/resize/seed0" / f"eval__{rec}.npy")
        rest = np.load(cache / arm / setting / "seed0" / f"eval__{rec}.npy")
        hyb = dire.copy()
        hyb[q_rows] = rest[q_rows]
        res = {n: E.protocols(x, d, pairs, B, S) for n, x in (("direct", dire), ("both", rest), ("probes", hyb))}
        ps = {n: {p: r.pop("_per_sample") for p, (r, _) in v.items()} for n, v in res.items()}
        r = {n: {p: rr for p, (rr, _) in v.items()} for n, v in res.items()}
        # one-side-restored verification
        Vd, Vr = dire[ix[("survface", "verification")]], rest[ix[("survface", "verification")]]
        ver = ev.survface_verification(Vr[ai["pos"]], Vd[bi["pos"]], Vr[ai["neg"]], Vd[bi["neg"]])
        ps["probes"]["survface_verification"] = ver.pop("_per_sample")
        r["probes"]["survface_verification"] = ver
        base = dict(arm=arm, setting=setting, recognizer=rec, leaky=(arm, rec) in LEAKY)
        # ---- A. probes-only
        for proto, cols in (("tinyface", ("Rank-1", "mAP")), ("survface_openset", ("TPIR20@FPIR=0.3",)), ("survface_verification", ("AUC", "TAR@FAR=0.1"))):
            for col in cols:
                row = dict(base, protocol=proto, metric=col)
                for n in ("direct", "both", "probes"):
                    row[n] = r[n][proto][col]
                    row[f"{n}_ci95"] = json.dumps(metrics.protocol_ci(proto, ps[n][proto], B, S)[col])
                if proto == "tinyface":
                    if col == "Rank-1":
                        x = metrics.paired_delta(ps["direct"][proto]["first_rank"] < 1, ps["probes"][proto]["first_rank"] < 1, B, S)
                    else:
                        x = metrics.paired_delta(ps["direct"][proto]["ap"], ps["probes"][proto]["ap"], B, S)
                    row["probes_minus_direct_paired"], row["probes_minus_direct_ci95"] = x["mean"], json.dumps(x["ci95"])
                probes_rows.append(row)
        # ---- B. per-bin (both sides restored vs direct)
        for sname, col, val in SLICES:
            m_p = np.ones(len(ix[("tinyface", "probe")]), bool) if col is None else _flag(d, ix[("tinyface", "probe")], col, val)
            m_m = np.ones(len(ix[("survface", "mated_probe")]), bool) if col is None else _flag(d, ix[("survface", "mated_probe")], col, val)
            m_u = np.ones(len(ix[("survface", "unmated_probe")]), bool) if col is None else _flag(d, ix[("survface", "unmated_probe")], col, val)
            if col is None:
                m_pos, m_neg = np.ones(len(ai["pos"]), bool), np.ones(len(ai["neg"]), bool)
            else:
                flagged = _flag(d, ix[("survface", "verification")], col, val.replace("NOT ", ""))  # image carries the bin label itself
                any_f = {k: flagged[ai[k]] | flagged[bi[k]] for k in ("pos", "neg")}
                m_pos, m_neg = (~any_f["pos"], ~any_f["neg"]) if val.startswith("NOT ") else (any_f["pos"], any_f["neg"])
            n = dict(n_probes=int(m_p.sum()), n_mated=int(m_m.sum()), n_unmated=int(m_u.sum()), n_pos_pairs=int(m_pos.sum()), n_neg_pairs=int(m_neg.sum()))
            row = dict(base, slice=sname, **n)
            if m_p.sum() >= 20:
                for nm in ("direct", "both"):
                    p = ps[nm]["tinyface"]
                    fr, ap = p["first_rank"][m_p], p["ap"][m_p]
                    row[f"tinyface_rank1_{nm}"] = float(np.mean(fr < 1))
                    row[f"tinyface_rank1_{nm}_ci95"] = json.dumps(metrics._paired_groups_ci(lambda i, fr=fr: {"r": float(np.mean(fr[i] < 1))}, len(fr), B, S)["r"])
                x = metrics.paired_delta(ps["direct"]["tinyface"]["first_rank"][m_p] < 1, ps["both"]["tinyface"]["first_rank"][m_p] < 1, B, S)
                row["tinyface_rank1_delta"], row["tinyface_rank1_delta_ci95"] = x["mean"], json.dumps(x["ci95"])
            if m_m.sum() >= 20 and m_u.sum() >= 20:
                for nm in ("direct", "both"):
                    p = ps[nm]["survface_openset"]
                    s1, gt = p["s1"][m_u], p["gt"][m_m]
                    row[f"tpir_{nm}"] = ev.survface_openset_from_scores(s1, gt)["TPIR20@FPIR=0.3"]
                    row[f"tpir_{nm}_ci95"] = json.dumps(metrics.bootstrap(lambda s1, gt: ev.survface_openset_from_scores(s1, gt), {"s1": s1, "gt": gt}, B, S)["TPIR20@FPIR=0.3"])
            if m_pos.sum() >= 20 and m_neg.sum() >= 20:
                for nm in ("direct", "both"):
                    p = ps[nm]["survface_verification"]
                    pos, neg = p["pos_dist"][m_pos], p["neg_dist"][m_neg]
                    row[f"auc_{nm}"] = ev.survface_verification_from_scores(pos, neg, accuracy=False)["AUC"]
                    row[f"auc_{nm}_ci95"] = json.dumps(metrics.bootstrap(lambda pos, neg: ev.survface_verification_from_scores(pos, neg, accuracy=False),
                                                                           {"pos": pos, "neg": neg}, B, S)["AUC"])
            bin_rows.append(row)
    return probes_rows, bin_rows


def fmt_ci(v, d=3):
    try:
        lo, hi = json.loads(v) if isinstance(v, str) else v
        return f"[{lo:.{d}f}, {hi:.{d}f}]"
    except (TypeError, ValueError):
        return ""


def tables(po, pb):
    out = []
    mark = lambda arm, rec: " †" if (arm, rec) in LEAKY else ""  # noqa: E731
    out += ["### Probes-only restoration vs both sides restored (eval subset)", "",
            "Probes / queries restored, gallery, distractors and templates left as direct feed. Delta = probes-only minus direct feed, paired on the same probes "
            "(TinyFace); SurvFace cells show the unpaired CI. Verification: first image of each pair restored, second direct (roles arbitrary).", ""]
    for proto, metric, title, dg in (("tinyface", "Rank-1", "TinyFace Rank-1", 3), ("survface_openset", "TPIR20@FPIR=0.3", "SurvFace open-set TPIR20@FPIR=0.3", 4),
                                     ("survface_verification", "AUC", "SurvFace verification AUC (one side restored)", 3)):
        out += [f"**{title}**", "", "| Arm | Recognizer | Direct feed | Both sides restored | Probes-only restored | Probes-only minus direct (paired) |", "|---|---|---|---|---|---|"]
        t = po[(po.protocol == proto) & (po.metric == metric)]
        for _, r in t.iterrows():
            dl = f"{r['probes_minus_direct_paired']:+.3f} {fmt_ci(r['probes_minus_direct_ci95'])}" if proto == "tinyface" and pd.notna(r.get("probes_minus_direct_paired")) else "-"
            out.append(f"| {r['arm']}: {r['setting']} | {r['recognizer']}{mark(r['arm'], r['recognizer'])} | {r['direct']:.{dg}f} {fmt_ci(r['direct_ci95'], dg)} | "
                       f"{r['both']:.{dg}f} {fmt_ci(r['both_ci95'], dg)} | {r['probes']:.{dg}f} {fmt_ci(r['probes_ci95'], dg)} | {dl} |")
        out.append("")
    out += ["### Per-bin breakdown of the both-sides-restored results (Phase 2 bins)", "",
            "`low_sharpness` = size-controlled lowest 20% of Laplacian variance; `dark10` = darkest 10% (Phase 2 definitions, per split). "
            "n = number of probes (TinyFace) / mated queries (SurvFace) in the bin; cells with fewer than 20 items are omitted. "
            "TinyFace: Rank-1 change vs direct feed, paired [95% CI]. SurvFace: direct -> restored (unpaired CIs are in `per_bin.csv`).", ""]
    for sl in ("low_sharpness", "dark10"):
        for comp in (sl, "not_" + sl):
            t = pb[pb.slice == comp]
            out += [f"**Slice `{comp}`**", "", "| Arm | Recognizer | n probes | TinyFace Rank-1: direct -> restored (paired delta) | n mated | SurvFace TPIR20@FPIR=0.3: direct -> restored | n pos pairs | SurvFace AUC: direct -> restored |",
                    "|---|---|---|---|---|---|---|---|"]
            for _, r in t.iterrows():
                a = (f"{r['tinyface_rank1_direct']:.3f} -> {r['tinyface_rank1_both']:.3f} ({r['tinyface_rank1_delta']:+.3f} {fmt_ci(r['tinyface_rank1_delta_ci95'])})"
                     if pd.notna(r.get("tinyface_rank1_delta")) else "-")
                b = f"{r['tpir_direct']:.4f} -> {r['tpir_both']:.4f}" if pd.notna(r.get("tpir_both")) else "-"
                c = f"{r['auc_direct']:.3f} -> {r['auc_both']:.3f}" if pd.notna(r.get("auc_both")) else "-"
                out.append(f"| {r['arm']}: {r['setting']} | {r['recognizer']}{mark(r['arm'], r['recognizer'])} | {r['n_probes']} | {a} | {r['n_mated']} | {b} | {r['n_pos_pairs']} | {c} |")
            out.append("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/sr.yaml"))
    B, S = cfg["bootstrap"]["eval_n"], cfg["bootstrap"]["seed"]
    vs = variants()
    print("variants:", vs, flush=True)
    po, pb = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(worker, arm, st, B, S): (arm, st) for arm, st in vs}
        for f in futs:
            p, b = f.result()
            po += p
            pb += b
            print("done", futs[f], flush=True)
    po, pb = pd.DataFrame(po), pd.DataFrame(pb)
    po.to_csv(RES / "probes_only.csv", index=False)
    pb.to_csv(RES / "per_bin.csv", index=False)
    md = "\n".join(tables(po, pb))
    (RES / "extra_tables.md").write_text(md, encoding="utf-8")
    rep = ROOT / "REPORT_phase3.md"
    if rep.exists():
        txt = rep.read_text(encoding="utf-8")
        b0, e0 = "<!-- EXTRA:BEGIN -->", "<!-- EXTRA:END -->"
        if b0 in txt and e0 in txt:
            rep.write_text(txt[:txt.index(b0) + len(b0)] + "\n" + md + "\n" + txt[txt.index(e0):], encoding="utf-8")
            print("tables injected into", rep)
    print("wrote", RES / "probes_only.csv", RES / "per_bin.csv", RES / "extra_tables.md")


if __name__ == "__main__":
    main()
