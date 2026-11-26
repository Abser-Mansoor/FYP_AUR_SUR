"""Inject the Phase 9 result tables into REPORT_phase9.md between <!-- TABLES:BEGIN --> and <!-- TABLES:END -->. Every number comes from a CSV or JSON written by a script.

    envs/video/Scripts/python.exe scripts/phase9_report_tables.py
A table whose file does not exist yet is replaced by a note, so the script can be rerun as results arrive.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "outputs/video/mevid/phase9"
P = ROOT / "outputs/video/phase9"
BEGIN, END = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"


def md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join("" if pd.isna(x) else str(x) for x in r) + " |" for r in df.itertuples(index=False)])


def csv(p, nd=4):
    p = Path(p)
    return None if not p.exists() else md(pd.read_csv(p).round(nd))


def js(p):
    p = Path(p)
    return None if not p.exists() else "```\n" + json.dumps(json.loads(p.read_text()), indent=1, sort_keys=True) + "\n```"


def curves(role):
    p = B / f"{role}_coverage_curves.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    px = d[d.kind == "px threshold"][["x", "coverage", "error_kept"]].rename(columns={"x": "threshold_px"})
    fr = d[(d.kind == "rejected fraction") & (d.x.round(2).isin([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]))][["x", "coverage", "error_kept"]].rename(columns={"x": "rejected_fraction"})
    return "By pixel threshold:\n\n" + md(px.round(4)) + "\n\nBy rejected fraction (lowest mean face height first):\n\n" + md(fr.round(4))


def demo():
    files = sorted((P / "demo_summary").glob("*.json")) if (P / "demo_summary").exists() else []
    if not files:
        return None
    rows = []
    for f in files:
        s = json.loads(f.read_text())
        rows.append(dict(clip=s["clip"], camera=s.get("camera"), mode=s["mode"], frames=s["n_frames"], face_tracks=s["n_face_tracks"], tracks_5_usable=s["n_tracks_5_usable"],
                         coverage_primary=s["coverage_primary"], coverage_broad=s["coverage_broad"], below_60pct_floor=s["below_floor"], identified=s["identified"], unknown=s["unknown"],
                         low_quality=s["low_quality"], fps_offline=s["fps_offline"]))
    return md(pd.DataFrame(rows).round(3))


def checks():
    p = ROOT / "outputs/video/phase9_checks.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    return "\n".join(f"- `{k}`: {json.dumps(v, sort_keys=True)[:420]}" for k, v in sorted(c.items()))


def build():
    parts = [
        ("Pre-registration numbers (from Phase 8 outputs)", js(P / "prereg_numbers.json")),
        ("B1 and B4: rules and fixes on the development split", csv(B / "dev_b1_b4.csv")),
        ("B2: error decomposition on the development split", csv(B / "dev_b2_decomposition.csv")),
        ("B2: summary (ranks, concentration, confusions, adoption)", js(B / "dev_summary.json")),
        ("B3: audit agreement (Cohen's kappa)", csv(ROOT / "outputs/video/mevid/audit/agreement.csv")),
        ("B3: audit taxonomy (both labelers agree)", csv(ROOT / "outputs/video/mevid/audit/taxonomy.csv")),
        ("B5: residual quality test, development", csv(B / "dev_b5_residual.csv")),
        ("Combination on development and the frozen configuration (section 0b)", (csv(B / "dev_combination.csv") or "") + ("\n\n" + js(B / "frozen_config.json") if (B / "frozen_config.json").exists() else "") or None),
        ("H1 and H2 on the confirmation split", csv(B / "conf_h1_h2.csv")),
        ("Coverage and error at 23 px, confirmation", js(B / "conf_at_bar.json")),
        ("Coverage and error at 23 px, development", js(B / "dev_at_bar.json")),
        ("Error against coverage, confirmation", curves("conf")),
        ("Error against coverage, development", curves("dev")),
        ("B5: residual quality test, confirmation", csv(B / "conf_b5_residual.csv")),
        ("Part A/C: ChokePoint tracker with the WebFace4M face term (IDF1, HOTA, ID switches)", csv(P / "tracker_face/lam0.5_webface4m/summary.csv")),
        ("Part C: ChokePoint open-set track identity, final against Phase 8 configuration", csv(P / "chokepoint_identity_summary.csv")),
        ("Part C: paired per-clip difference", js(P / "chokepoint_identity_diff.json")),
        ("Part D: frame rate at 720p, lab PC (end to end)", csv(P / "fps_labpc.csv", 2)),
        ("Part D: per-stage medians, lab PC", csv(P / "fps_labpc_per_clip.csv", 2)),
        ("Part D: frame rate at 720p, T4 (end to end)", csv(P / "fps_t4.csv", 2)),
        ("Part D: per-stage medians, T4", csv(P / "fps_t4_per_clip.csv", 2)),
        ("Part D: the F4 speed condition", js(P / "fps_condition.json")),
        ("Part E: retrieval, FAISS Flat against HNSW", csv(P / "retrieval.csv", 5)),
        ("Part F: detector mode per MEVA camera (calibration)", js(P / "calibration_modes.json")),
        ("Part F: MEVA demo per clip", demo()),
        ("Acceptance checks", checks()),
    ]
    return "\n".join(f"#### {t}\n\n" + (b if b else "_not available yet_") + "\n" for t, b in parts)


if __name__ == "__main__":
    p = ROOT / "REPORT_phase9.md"
    s = p.read_text(encoding="utf-8")
    block = f"{BEGIN}\n\n{build()}\n{END}"
    if BEGIN in s:
        s = s[:s.index(BEGIN)] + block + s[s.index(END) + len(END):]
    else:
        s = s.rstrip() + "\n\n## Result tables\n\n" + block + "\n"
    p.write_text(s, encoding="utf-8")
    print("tables injected")
