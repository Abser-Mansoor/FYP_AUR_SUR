"""Build every Phase 3 report table from the result files (no number is typed by hand).

    envs/diag/Scripts/python.exe scripts/sr_report_tables.py
Writes outputs/sr/results/report_tables.md. Missing results appear as "pending", never as blanks or guesses.
Leaky cells (OSDFace x ArcFace) are marked with a dagger.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs/sr/results"
RECS = ["arcface", "adaface", "magface", "curricularface"]
NAMES = {"arcface": "ArcFace", "adaface": "AdaFace", "magface": "MagFace", "curricularface": "CurricularFace"}
LEAKY = {("osdface", "arcface")}
ARMS = [("direct", "resize", "Direct feed (resize)"), ("bicubic", None, "Bicubic"), ("esrgan", None, "Real-ESRGAN x4plus"),
        ("invsr", None, "InvSR (rule-selected)"), ("invsr", "pre128_st1", "InvSR pre-128 (sensitivity)"),
        ("tadsr", None, "TADSR"), ("osdface", None, "OSDFace")]


def selected(arm):
    f = RES / f"selection_{arm}.json"
    return json.loads(f.read_text())[0]["setting"] if f.exists() and json.loads(f.read_text()) else None


def load_eval():
    frames = [pd.read_csv(f) for f in sorted(RES.glob("eval_*.csv"))]
    d = pd.concat(frames, ignore_index=True)
    d = d[d["status"] == "OK"].drop_duplicates(subset=["arm", "setting", "seed", "recognizer", "protocol"])
    return d


def ci(v):
    try:
        lo, hi = json.loads(v) if isinstance(v, str) else v
        return f" [{lo:.3f}, {hi:.3f}]"
    except (TypeError, ValueError):
        return ""


def cell(d, arm, setting, rec, proto, col, digits=3, with_ci=True):
    r = d[(d.arm == arm) & (d.setting == setting) & (d.recognizer == rec) & (d.protocol == proto)]
    if r.empty or pd.isna(r.iloc[0].get(col)):
        return "pending"
    v = r.iloc[0][col]
    s = f"{v:.{digits}f}" + (ci(r.iloc[0].get(f"{col}_ci95")) if with_ci else "")
    return s + (" †" if (arm, rec) in LEAKY else "")


def rows_for(d):
    out = []
    for arm, setting, label in ARMS:
        st = setting or selected(arm)
        if st is None:
            out.append((arm, None, f"{label} (not selected yet)"))
        else:
            out.append((arm, st, f"{label}: {st}" if arm != "direct" else label))
    return out


def metric_table(d, proto, col, title, digits=3):
    lines = [f"### {title}", "", "| Arm | " + " | ".join(NAMES[r] for r in RECS) + " |",
             "|---|" + "---|" * len(RECS)]
    for arm, st, label in rows_for(d):
        cells = [cell(d, arm, st, r, proto, col, digits) if st else "pending" for r in RECS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines + [""]


def paired_table(d):
    lines = ["### Paired change vs direct feed on the same crops (eval subset; mean [95% CI])", "",
             "Drift = cosine between the restored and the direct-feed embedding of the same crop (1 = unchanged). "
             "Rank-1 delta = paired change in TinyFace rank-1 hit rate. Impostor shift > 0 = impostors look more similar.", "",
             "| Arm | Recognizer | Drift (TinyFace probes) | TinyFace Rank-1 delta | Impostor shift (TinyFace) | d' TinyFace | d' SurvFace 1:1 |",
             "|---|---|---|---|---|---|---|"]
    for arm, st, label in rows_for(d):
        if arm == "direct" or st is None:
            continue
        for r in RECS:
            t = d[(d.arm == arm) & (d.setting == st) & (d.recognizer == r) & (d.protocol == "tinyface")]
            v = d[(d.arm == arm) & (d.setting == st) & (d.recognizer == r) & (d.protocol == "survface_verification")]
            if t.empty:
                lines.append(f"| {label} | {r} | pending | pending | pending | pending | pending |")
                continue
            t = t.iloc[0]
            mark = " †" if (arm, r) in LEAKY else ""
            dv = f"{v.iloc[0]['sep_d_prime']:.2f}" if not v.empty else "pending"
            lines.append(f"| {label} | {r}{mark} | {t['drift_mean']:.3f}{ci(t['drift_ci95'])} | {t['paired_rank1_mean']:+.3f}{ci(t['paired_rank1_ci95'])} | "
                         f"{t['impostor_shift_mean']:+.3f}{ci(t['impostor_shift_ci95'])} | {t['sep_d_prime']:.2f} | {dv} |")
    return lines + [""]


def seed_table():
    lines = ["### Seed study (tuning subset, seeds 0-2; mean +- SD across seeds)", "",
             "| Arm | Setting | Recognizer | TinyFace Rank-1 | SurvFace verification AUC | SurvFace TPIR20@FPIR=0.3 |", "|---|---|---|---|---|---|"]
    for arm in ("invsr", "tadsr", "osdface"):
        f = RES / f"seed_study_{arm}.csv"
        if not f.exists():
            lines.append(f"| {arm} | - | - | pending | pending | pending |")
            continue
        s = pd.read_csv(f)
        s = s[(s.status == "OK") & (s.arm == arm)]
        for r in RECS:
            vals = []
            for proto, col in (("tinyface", "Rank-1"), ("survface_verification", "AUC"), ("survface_openset", "TPIR20@FPIR=0.3")):
                x = s[(s.recognizer == r) & (s.protocol == proto)][col]
                vals.append(f"{x.mean():.3f} +- {x.std(ddof=1):.3f} (n={len(x)})" if len(x) else "pending")
            mark = " †" if (arm, r) in LEAKY else ""
            lines.append(f"| {arm} | {s.setting.iloc[0] if len(s) else '-'} | {r}{mark} | " + " | ".join(vals) + " |")
    return lines + [""]


def cost_table():
    lines = ["### Cost and failure counts", "",
             "Speed: isolated re-timing on the lab PC (RTX 4000 Ada), 200 seeded eval crops, seed 0, first 5 crops excluded as warm-up. "
             "Lost matches: TinyFace probes rank-1 correct with direct feed but wrong after restoration for at least 2 of the 3 "
             "non-leaky recognizers (of 1,155).", "",
             "| Arm | Setting | s/crop median | s/crop mean | Peak VRAM (MB) | Lost matches | Remote device |", "|---|---|---|---|---|---|---|"]
    t = pd.read_csv(RES / "timing_isolated.csv") if (RES / "timing_isolated.csv").exists() else pd.DataFrame()
    fc = ROOT / "outputs/sr/failure_cases"
    for arm, st, label in rows_for(pd.DataFrame()):
        if arm == "direct" or st is None:
            continue
        row = t[(t.arm == arm) & (t.setting == st)] if len(t) else t
        med = f"{row.iloc[0]['sec_per_crop_median']:.3f}" if len(row) and pd.notna(row.iloc[0].get("sec_per_crop_median")) else "n/a"
        mean = f"{row.iloc[0]['sec_per_crop_mean']:.3f}" if len(row) and pd.notna(row.iloc[0].get("sec_per_crop_mean")) else "pending"
        vram = f"{row.iloc[0]['peak_vram_mb']:.0f}" if len(row) and pd.notna(row.iloc[0].get("peak_vram_mb")) else "pending"
        lm = fc / f"{arm}__{st}__lost_match.csv"
        lost = str(int(pd.read_csv(lm)["n_lost_total"].iloc[0])) if lm.exists() and len(pd.read_csv(lm)) else ("0" if lm.exists() else "pending")
        tj = ROOT / "outputs/sr/restored/eval" / f"{arm}__{st}" / "seed0" / "timing.json"
        dev = "-"
        if tj.exists() and json.loads(tj.read_text()).get("remote"):
            dv = json.loads(tj.read_text()).get("device") or []
            dev = dv if isinstance(dv, str) else ", ".join(dv)
        lines.append(f"| {label} | {st} | {med} | {mean} | {vram} | {lost} | {dev} |")
    return lines + [""]


def tuning_tables():
    lines = ["### Tuning sweeps (tuning subset; score = mean over AdaFace, MagFace, CurricularFace of [TinyFace Rank-1, SurvFace verification AUC])", ""]
    for arm in ("bicubic", "esrgan", "invsr", "tadsr", "osdface"):
        f = RES / f"tuning_selection_{arm}.csv"
        if not f.exists():
            lines += [f"{arm}: pending", ""]
            continue
        g = pd.read_csv(f)
        lines += [f"**{arm}** (selected: {selected(arm)})", "", "| Setting | Score | Rank-1 | AUC |", "|---|---|---|---|"]
        lines += [f"| {r.setting} | {r.score:.4f} | {r.rank1:.4f} | {r.auc:.4f} |" for r in g.itertuples()]
        lines.append("")
    return lines


def main():
    d = load_eval()
    n = {p: len(pd.read_csv(ROOT / f"outputs/sr/subset/{p}_images.csv")) for p in ("tuning", "eval")}
    out = ["# Phase 3 report tables (generated by scripts/sr_report_tables.py)", "",
           f"Eval subset: {n['eval']} crops; tuning subset: {n['tuning']} crops. Values: point estimate [95% bootstrap CI, 1,000 draws]. "
           "† = leaky recognizer (OSDFace's identity loss uses ArcFace); conclusions rest on the other three.", ""]
    out += metric_table(d, "tinyface", "Rank-1", "TinyFace Rank-1 (closed-set identification)")
    out += metric_table(d, "tinyface", "mAP", "TinyFace mAP")
    out += metric_table(d, "survface_verification", "TAR@FAR=0.1", "SurvFace verification TAR@FAR=0.1")
    out += metric_table(d, "survface_verification", "TAR@FAR=0.01", "SurvFace verification TAR@FAR=0.01")
    out += metric_table(d, "survface_verification", "AUC", "SurvFace verification AUC")
    out += metric_table(d, "survface_openset", "TPIR20@FPIR=0.3", "SurvFace open-set TPIR20@FPIR=0.3", 4)
    out += paired_table(d)
    out += seed_table()
    out += cost_table()
    out += tuning_tables()
    (RES / "report_tables.md").write_text("\n".join(out), encoding="utf-8")
    rep = ROOT / "REPORT_phase3.md"  # inject the generated tables between markers, so no table is ever typed by hand
    if rep.exists():
        txt = rep.read_text(encoding="utf-8")
        b, e = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"
        if b in txt and e in txt:
            body = "\n".join(out[2:])  # drop the generated file's own title
            rep.write_text(txt[:txt.index(b) + len(b)] + "\n" + body + "\n" + txt[txt.index(e):], encoding="utf-8")
            print("tables injected into", rep)
    print("wrote", RES / "report_tables.md", "| pending cells:", "\n".join(out).count("pending"))


if __name__ == "__main__":
    main()
