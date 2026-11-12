"""Inject the Phase 6 result tables into REPORT_phase6.md between <!-- TABLES:BEGIN --> and <!-- TABLES:END -->. Every number comes from a CSV written by a script.

    envs/video/Scripts/python.exe scripts/phase6_report_tables.py
A table whose CSV does not exist yet is skipped with a note, so the script can be rerun when B and E finish.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "outputs/video"
BEGIN, END = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def ci(r, k, nd=4):
    return f"{r[k]:.{nd}f} ({r[k + '_ci_lo']:.{nd}f}-{r[k + '_ci_hi']:.{nd}f})"


def read(p):
    return pd.read_csv(p) if Path(p).exists() else None


def tracker_table():
    out = []
    for f, label in (("pooled.csv", "all detections tracked (floor 0.1)"), ("pooled_minscore0.5.csv", "scored on detections >= 0.5 only")):
        d = read(V / "tracker_eval" / f)
        if d is None:
            continue
        for r in d.to_dict("records"):
            out.append({"tracker": r["tracker"], "scoring": label, "HOTA": ci(r, "HOTA", 3), "AssA": ci(r, "AssA", 3), "IDF1": ci(r, "IDF1", 3), "DetA": ci(r, "DetA", 3), "MOTA": ci(r, "MOTA", 3),
                        "IDSW (total)": f"{r['IDSW']:.0f}", "FN": f"{r['CLR_FN']:.0f}", "FP": f"{r['CLR_FP']:.0f}"})
    return md(pd.DataFrame(out))


def group_table():
    d = read(V / "tracker_eval/per_clip.csv")
    d = d[d.tracker == "bytetrack"].assign(group=lambda x: x["clip"].str[:3])
    g = d.groupby("group").agg(clips=("clip", "count"), HOTA=("HOTA", "mean"), IDF1=("IDF1", "mean"), DetA=("DetA", "mean"), FN=("CLR_FN", "sum"), FP=("CLR_FP", "sum"), IDSW=("IDSW", "sum")).reset_index()
    return md(g.round(3))


def baseline_table():
    d = read(V / "baseline/summary.csv")
    d["Rank-1 (95% CI)"] = [ci(r, "rank1") for r in d.to_dict("records")]
    d["TPIR20@FPIR0.3 (95% CI)"] = [f"{r['tpir20_fpir03']:.4f} ({r['tpir_ci_lo']:.4f}-{r['tpir_ci_hi']:.4f})" for r in d.to_dict("records")]
    return md(d[["recognizer", "condition", "Rank-1 (95% CI)", "TPIR20@FPIR0.3 (95% CI)"]])


def d0_table():
    d = read(V / "bench7/D0.csv")
    if d is None:
        return None
    d = d[d.direction == "mean"]
    return md(pd.DataFrame([{"recognizer": r["recognizer"], "condition": r["condition"], "TPIR20@0.3": f"{r['tpir20_fpir0.3']:.4f}", "@0.1": f"{r['tpir20_fpir0.1']:.4f}", "@0.01": f"{r['tpir20_fpir0.01']:.4f}"} for r in d.to_dict("records")]))


def b_table():
    d = read(V / "bench7/B.csv")
    if d is None:
        return None
    d = d[d.direction == "mean"]
    return md(pd.DataFrame([{"recognizer": r["recognizer"], "cond": r["condition"], "k": int(r["k"]), "Rank-1": f"{r['rank1']:.4f}", "TPIR20@0.3": f"{r['tpir20_fpir0.3']:.4f}", "@0.1": f"{r['tpir20_fpir0.1']:.4f}", "@0.01": f"{r['tpir20_fpir0.01']:.4f}",
                            "Rank-1 over draws (min-max, mean of directions)": f"{r['rank1_draw_min']:.4f}-{r['rank1_draw_max']:.4f}"} for r in d.to_dict("records")]))


def e_table():
    d = read(V / "bench7/E.csv")
    if d is None:
        return None
    d = d[d.direction == "mean"]
    return md(pd.DataFrame([{"cond": r["condition"], "probe side (px)": int(r["probe_side"]), "Rank-1": f"{r['rank1']:.4f}", "TPIR20@0.3": f"{r['tpir20_fpir0.3']:.4f}", "@0.1": f"{r['tpir20_fpir0.1']:.4f}", "@0.01": f"{r['tpir20_fpir0.01']:.4f}",
                            "mean MagFace norm": "" if pd.isna(r.get("mean_mag_norm")) else f"{r['mean_mag_norm']:.2f}"} for r in d.to_dict("records")]))


def timings_table():
    d = read(V / "pipeline_timings.csv")
    return md(d[["stage", "unit", "n_clips", "n_units", "median_ms_per_unit", "pooled_ms_per_unit", "p10_clip", "p90_clip"]].round(2))


def ytf_table():
    d = read(V / "appendix/ytf/summary.csv")
    return md(pd.DataFrame([{"mode": r["mode"], "face height (px)": r["face_h_bin"], "n": int(r["n"]), "recall IoU 0.5 (95% CI)": ci(r, "recall_iou50", 3), "recall IoU 0.3": f"{r['recall_iou30']:.3f}",
                             "centre hit": f"{r['centre_hit']:.3f}", "extra/frame": f"{r['extra_per_frame']:.3f}", "median ms": f"{r['median_ms']:.1f}"} for r in d.to_dict("records")]))


def checks():
    import json
    c = json.loads((V / "phase6_checks.json").read_text())
    cov, det = c["coverage"], c["determinism"]
    return (f"- Coverage: {cov['n_clips']} clips ({cov['n_gt']} ground-truth, {cov['n_crowded']} crowded), {cov['total_frames']} frames, failures: {len(cov['failures'])}; unreadable frames logged: {cov['unreadable']}.\n"
            f"- Determinism ({det['clip']}): byte-identical = {det['ok_byte_identical']} (detections, keypoints, tracks.csv, AdaFace embeddings, raw norm, luminance, Laplacian variance).")


def build():
    parts = [("Tracker validation, pooled over the 72 ground-truth clips (cluster bootstrap over clips, 1,000 draws, seed 42)", tracker_table()),
             ("ByteTrack by group (mean over clips; FP and FN are sums)", group_table()),
             ("Acceptance checks 4 and 5", checks()),
             ("Single-frame baseline (pre-registered section 0; mean of the two directions; CI is the cluster bootstrap over identities)", baseline_table()),
             ("D: the baseline read at stricter FPIR (mean of the two directions)", d0_table()),
             ("B: thin gallery (k frames per identity, 20 draws)", b_table()),
             ("E: degraded probes (AdaFace; enrolment native; every 5th matched detection)", e_table()),
             ("Stage timings (lab PC, RTX 4000 Ada, 800 x 600; detect and embed include JPEG decoding; track is both trackers together)", timings_table()),
             ("Appendix T1: SCRFD on the YouTube Faces Kaggle crops (Colab T4)", ytf_table())]
    out = []
    for title, body in parts:
        out.append(f"#### {title}\n\n" + (body if body else "_not available yet (the run has not finished)_") + "\n")
    return "\n".join(out)


if __name__ == "__main__":
    p = ROOT / "REPORT_phase6.md"
    s = p.read_text(encoding="utf-8")
    block = f"{BEGIN}\n\n{build()}\n{END}"
    if BEGIN in s:
        s = s[:s.index(BEGIN)] + block + s[s.index(END) + len(END):]
    else:
        s = s.rstrip() + "\n\n## Tables\n\n" + block + "\n"
    p.write_text(s, encoding="utf-8")
    print("tables injected")
