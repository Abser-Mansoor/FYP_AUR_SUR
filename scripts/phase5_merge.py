"""Merge and score the detector-study chunks (lab PC, envs/video). No new inference.

    envs/video/Scripts/python.exe scripts/phase5_merge.py [--study DIR]      # DIR: where the chunks are (default: paths.out)

Checks that every chunk set covers the expected items exactly, once and in order; then writes under <study>/results/:
wider_ap.csv, wider_bins.csv, chokepoint_bins.csv, crops_rate.csv, runtime.csv, selection.json, detector_tables.md.
The selection rule comes from configs/video.yaml and was fixed before the T4 run.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.video import wider_eval as W  # noqa: E402


def load(study, job, tag):
    d = study / "chunks" / job / tag
    parts = [np.load(f, allow_pickle=False) for f in sorted(d.glob("*.npz"))]
    if not parts:
        return None
    names = np.concatenate([p["names"] for p in parts]).tolist()
    errors = [e for p in parts for e in p["errors"].tolist()]
    seconds = float(sum(p["seconds"] for p in parts))
    if "kps" in parts[0].files:
        return names, np.concatenate([p["kps"] for p in parts]), errors, seconds
    counts = np.concatenate([p["counts"] for p in parts])
    dets = np.concatenate([p["dets"] for p in parts])
    return names, np.split(dets, np.cumsum(counts)[:-1]), errors, seconds


def check(names, expected, what):
    if names != expected:
        i = next((k for k, (a, b) in enumerate(zip(names, expected)) if a != b), min(len(names), len(expected)))
        raise ValueError(f"{what}: chunk names do not match the expected items (got {len(names)}, expected {len(expected)}; first mismatch at {i})")


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/video.yaml")
    ap.add_argument("--study", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    study = Path(a.study) if a.study else ROOT / cfg["paths"]["out"]
    res = study / "results"
    res.mkdir(parents=True, exist_ok=True)
    thrs = cfg["score_thresholds"]
    tags = [f"{d}__{m}" for d in ("scrfd", "retinaface") for m in cfg["modes"]]
    bins_names = W.BIN_NAMES
    tables, ap_rows, bin_rows, ch_rows, crop_rows, problems = [], [], [], [], [], []

    # ---- WIDER FACE
    gts = W.load_gt(ROOT / cfg["paths"]["wider"])
    attrs = W.load_attrs(ROOT / cfg["paths"]["wider"])
    expected = [f"{g['event']}/{g['file']}" for g in gts]
    for tag in tags:
        r = load(study, "wider", tag)
        if r is None:
            continue
        names, dets, errors, sec = r
        check(names, expected, f"wider/{tag}")
        problems += [f"wider/{tag}: {e}" for e in errors]
        preds = {(g["event"], g["file"]): d for g, d in zip(gts, dets)}
        ap_ = W.average_precision(preds, gts)
        ap_rows.append(dict(config=tag, **{k: v for k, v in ap_.items()}, n_images=len(names), n_errors=len(errors)))
        for thr in thrs:
            found, total, fp = W.bin_recall(preds, gts, attrs, thr)
            for k, bn in enumerate(bins_names):
                bin_rows.append(dict(config=tag, thr=thr, bin=bn, found=int(found[k]), total=int(total[k]), recall=found[k] / total[k] if total[k] else np.nan, fp_per_image=fp))
    pd.DataFrame(ap_rows).to_csv(res / "wider_ap.csv", index=False)
    wb = pd.DataFrame(bin_rows)
    wb.to_csv(res / "wider_bins.csv", index=False)

    # ---- ChokePoint sample
    sdir = ROOT / cfg["paths"]["chokepoint_sample"]
    if (sdir / "gt_boxes.csv").exists():
        gt = pd.read_csv(sdir / "gt_boxes.csv")
        idx = pd.read_csv(sdir / "index.csv")
        exp = idx["name"].tolist()
        cg = [dict(event="", file=n, boxes=np.array([[r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1] for r in gt[gt["name"] == n].itertuples()], float).reshape(-1, 4)) for n in exp]
        for tag in tags:
            r = load(study, "chokepoint", tag)
            if r is None:
                continue
            names, dets, errors, _ = r
            check(names, exp, f"chokepoint/{tag}")
            problems += [f"chokepoint/{tag}: {e}" for e in errors]
            preds = {("", n): d for n, d in zip(names, dets)}
            for thr in thrs:
                found, total, fp = W.bin_recall(preds, cg, {}, thr)
                for k, bn in enumerate(bins_names):
                    ch_rows.append(dict(config=tag, thr=thr, bin=bn, found=int(found[k]), total=int(total[k]), recall=found[k] / total[k] if total[k] else np.nan, fp_per_frame=fp))
    pd.DataFrame(ch_rows).to_csv(res / "chokepoint_bins.csv", index=False)

    # ---- TinyFace / SurvFace crops: detection rate by native crop size
    img = pd.read_csv(ROOT / cfg["paths"]["eval_images"])
    side = np.maximum(img["h"], img["w"]).to_numpy()
    edges, labels = [0, 16, 24, 32, 48, 10 ** 6], ["<16", "16-24", "24-32", "32-48", ">48"]
    for det in ("scrfd", "retinaface"):
        r = load(study, "crops", f"{det}__crop")
        if r is None:
            continue
        names, kps, errors, _ = r
        check(names, img["file"].tolist(), f"crops/{det}")
        problems += [f"crops/{det}: {e}" for e in errors]
        ok = ~np.isnan(kps).any((1, 2))
        for ds in ("tinyface", "survface"):
            m = (img["dataset"] == ds).to_numpy()
            crop_rows.append(dict(detector=det, dataset=ds, size="all", n=int(m.sum()), detected=int(ok[m].sum()), rate=float(ok[m].mean())))
            for k, lb in enumerate(labels):
                mm = m & (side >= edges[k]) & (side < edges[k + 1])
                if mm.sum():
                    crop_rows.append(dict(detector=det, dataset=ds, size=lb, n=int(mm.sum()), detected=int(ok[mm].sum()), rate=float(ok[mm].mean())))
    cr = pd.DataFrame(crop_rows)
    cr.to_csv(res / "crops_rate.csv", index=False)

    # ---- runtime
    rt_f = study / "runtime.json"
    rt = json.loads(rt_f.read_text()) if rt_f.exists() else {}
    rtd = pd.DataFrame([dict(config=k, **v) for k, v in rt.items()])
    rtd.to_csv(res / "runtime.csv", index=False)

    # ---- selection (rule fixed in configs/video.yaml before the T4 run)
    sel = cfg["selection"]
    cand = []
    for tag in tags:
        w = wb[(wb.config == tag) & (wb.thr == sel["score"]) & wb.bin.isin(sel["bins"])]
        if w.empty or tag not in rt:
            continue
        cand.append(dict(config=tag, recall=w.found.sum() / w.total.sum(), median_ms=rt[tag]["median_ms"]))
    sc = pd.DataFrame(cand)
    chosen = None
    if not sc.empty:
        ok = sc[sc.median_ms <= sel["max_ms"]]
        if ok.empty:
            chosen = dict(note=f"no configuration meets the {sel['max_ms']} ms limit", candidates=sc.to_dict("records"))
        else:
            top = ok[ok.recall >= ok.recall.max() - sel["tie_recall"]].sort_values("median_ms")
            chosen = dict(config=top.iloc[0]["config"], recall=float(top.iloc[0]["recall"]), median_ms=float(top.iloc[0]["median_ms"]),
                          candidates=sc.to_dict("records"), rule=sel)
    (res / "selection.json").write_text(json.dumps(chosen, indent=1))
    (res / "problems.txt").write_text("\n".join(problems))

    # ---- markdown tables
    md = ["### WIDER FACE val: official AP", md_table(pd.DataFrame(ap_rows)) if ap_rows else "(not run)"]
    for thr in thrs:
        if not wb.empty:
            p = wb[wb.thr == thr].pivot(index="config", columns="bin", values="recall")[bins_names]
            fp = wb[wb.thr == thr].groupby("config").fp_per_image.first().rename("FP/image")
            md += [f"### WIDER FACE val: recall by face height, score >= {thr}", md_table(p.join(fp).reset_index())]
    for thr in thrs:
        if ch_rows:
            c = pd.DataFrame(ch_rows)
            p = c[c.thr == thr].pivot(index="config", columns="bin", values="recall")[bins_names]
            fp = c[c.thr == thr].groupby("config").fp_per_frame.first().rename("FP/frame")
            md += [f"### ChokePoint sample: recall by face height, score >= {thr}", md_table(p.join(fp).reset_index())]
    if not cr.empty:
        md += ["### TinyFace / SurvFace crops: detection rate by native crop size (longest side, px)",
               md_table(cr.pivot_table(index=["detector", "dataset"], columns="size", values="rate", sort=False).reset_index())]
    if not rtd.empty:
        md += ["### Runtime (median ms per 720p frame, batch 1)", md_table(rtd[["config", "median_ms", "p95_ms", "n"]])]
    md += ["### Selected detector (Phase 6)", "```", json.dumps(chosen, indent=1), "```"]
    (res / "detector_tables.md").write_text("\n\n".join(md))
    print("\n\n".join(md[:8]))
    if problems:
        print(f"\n{len(problems)} per-item errors logged in {res / 'problems.txt'}")


if __name__ == "__main__":
    main()
