"""Generate the LFW-check tables into REPORT_phase3.md (between <!-- LFW:BEGIN --> and <!-- LFW:END -->).

    envs/diag/Scripts/python.exe scripts/sr_lfw_tables.py
Reads outputs/sr/results/lfw_check.csv, lfw_check_contrasts.csv, lfw_check_vs_bicubic.csv.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs/sr/results"
RECS = ["arcface", "adaface", "magface", "curricularface"]
NAMES = {"arcface": "ArcFace", "adaface": "AdaFace", "magface": "MagFace", "curricularface": "CurricularFace"}
ARMS = ["bicubic", "esrgan", "invsr", "tadsr", "osdface"]
LABEL = {"bicubic": "Bicubic", "esrgan": "Real-ESRGAN native", "invsr": "InvSR pre-64", "tadsr": "TADSR pre-128 / t250", "osdface": "OSDFace pre-128"}


def ci(v):
    lo, hi = json.loads(v)
    return f"[{lo:+.3f}, {hi:+.3f}]"


def hdr(first):
    return [f"| {first} | " + " | ".join(NAMES[r] + (" †" if False else "") for r in RECS) + " |", "|---|" + "---|" * len(RECS)]


def main():
    d = pd.read_csv(RES / "lfw_check.csv")
    c = pd.read_csv(RES / "lfw_check_contrasts.csv")
    b = pd.read_csv(RES / "lfw_check_vs_bicubic.csv")
    out = ["Low-resolution side: 10 px (chosen on direct feed only; at 24 px direct feed is at the ceiling, AUC 0.998-0.999). 6,000 LFW pairs. "
           "Cells: AUC; contrasts are paired differences with 95% CIs (positive and negative pairs resampled separately, 1,000 draws). † = leaky.", ""]
    # absolute AUC
    out += ["**Absolute verification AUC by condition**", "", "| Condition | Arm | " + " | ".join(NAMES[r] for r in RECS) + " |", "|---|---|" + "---|" * len(RECS)]
    for cond, arms in (("ceiling (HR, HR)", ["direct"]), ("cross direct (LRpx, HR)", ["direct"]), ("symmetric direct (LRpx, LRpx)", ["direct"]),
                       ("cross restored (restored, HR)", ARMS), ("symmetric restored (restored, restored)", ARMS)):
        for arm in arms:
            t = d[(d.arm == arm) & (d.condition == cond)].set_index("recognizer")
            cells = [f"{t.loc[r, 'AUC']:.3f}" + (" †" if (arm, r) == ("osdface", "arcface") else "") if r in t.index else "-" for r in RECS]
            out.append(f"| {cond} | {LABEL.get(arm, 'direct feed')} | " + " | ".join(cells) + " |")
    out.append("")
    # rule contrast
    out += ["**Pre-registered contrast: cross-restored minus cross-direct AUC** (rule: CI above zero for at least 2 of the 3 non-leaky recognizers)", "",
            "| Arm | " + " | ".join(NAMES[r] for r in RECS) + " | Non-leaky positive | Rule |", "|---|" + "---|" * len(RECS) + "---|---|"]
    for arm in ARMS:
        t = c[(c.arm == arm) & c.contrast.str.startswith("cross")].set_index("recognizer")
        cells = [f"{t.loc[r, 'delta_AUC']:+.3f} {ci(t.loc[r, 'delta_AUC_ci95'])}" + (" †" if (arm, r) == ("osdface", "arcface") else "") for r in RECS]
        npos = sum(json.loads(t.loc[r, "delta_AUC_ci95"])[0] > 0 for r in RECS if r != "arcface" or arm != "osdface" and False)
        npos = sum(json.loads(t.loc[r, "delta_AUC_ci95"])[0] > 0 for r in ("adaface", "magface", "curricularface"))
        out.append(f"| {LABEL[arm]} | " + " | ".join(cells) + f" | {npos} of 3 | {'met' if npos >= 2 else 'not supported'} |")
    out.append("")
    out += ["**Same contrast with both images low-resolution (symmetric restored minus symmetric direct)**", "",
            "| Arm | " + " | ".join(NAMES[r] for r in RECS) + " |", "|---|" + "---|" * len(RECS)]
    for arm in ARMS:
        t = c[(c.arm == arm) & c.contrast.str.startswith("symmetric")].set_index("recognizer")
        out.append(f"| {LABEL[arm]} | " + " | ".join(f"{t.loc[r, 'delta_AUC']:+.3f} {ci(t.loc[r, 'delta_AUC_ci95'])}" for r in RECS) + " |")
    out.append("")
    for kind in ("cross", "symmetric"):
        out += [f"**Post-hoc: arm minus bicubic, {kind} condition** (added after seeing that bicubic met the rule; bicubic is the fair non-generative comparator)", "",
                "| Arm | " + " | ".join(NAMES[r] for r in RECS) + " |", "|---|" + "---|" * len(RECS)]
        for arm in ("tadsr", "osdface", "invsr", "esrgan"):
            t = b[(b.arm == arm) & (b.condition == kind)].set_index("recognizer")
            out.append(f"| {LABEL[arm]} | " + " | ".join(f"{t.loc[r, 'delta_AUC_vs_bicubic']:+.3f} {ci(t.loc[r, 'ci95'])}" + (" †" if (arm, r) == ("osdface", "arcface") else "") for r in RECS) + " |")
        out.append("")
    md = "\n".join(out)
    (RES / "lfw_tables.md").write_text(md, encoding="utf-8")
    rep = ROOT / "REPORT_phase3.md"
    txt = rep.read_text(encoding="utf-8")
    b0, e0 = "<!-- LFW:BEGIN -->", "<!-- LFW:END -->"
    if b0 in txt and e0 in txt:
        rep.write_text(txt[:txt.index(b0) + len(b0)] + "\n" + md + "\n" + txt[txt.index(e0):], encoding="utf-8")
        print("tables injected into", rep)
    else:
        print("markers not found; wrote", RES / "lfw_tables.md")


if __name__ == "__main__":
    main()
