"""Phase 4 tables for REPORT_phase4.md (imported by scripts/sr_report_tables.py --config configs/enhance.yaml; no number typed by hand).

stop_rule_table: the pre-registered stop rule on the probes-only TinyFace Rank-1 paired gain (results/probes_only.csv).
lfw_tables: the synthetic LFW check (lfw_enh_calibration.csv, lfw_enh_check.csv, lfw_enh_contrasts.csv).
cost_table: s/crop and peak VRAM of the eval restorations, plus the CPU vs GPU smoke timings (outputs/enhance/smoke/*/timing.json).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECS = ["arcface", "adaface", "magface", "curricularface"]
NON_LEAKY = ["adaface", "magface", "curricularface"]
NAMES = {"arcface": "ArcFace", "adaface": "AdaFace", "magface": "MagFace", "curricularface": "CurricularFace"}


def ci(v, d=3):
    lo, hi = json.loads(v) if isinstance(v, str) else v
    return f"[{lo:+.{d}f}, {hi:+.{d}f}]"


def stop_rule_table(res):
    f = res / "probes_only.csv"
    if not f.exists():
        return ["Stop-rule table: pending (probes_only.csv not found)", ""]
    po = pd.read_csv(f)
    t = po[(po.protocol == "tinyface") & (po.metric == "Rank-1") & po.recognizer.isin(NON_LEAKY)]
    n_tests = int(t.groupby(["arm", "setting"]).ngroups)
    out = [f"**Stop rule: probes-only TinyFace Rank-1 paired gain (eval subset, non-leaky recognizers), {n_tests} tests**", "",
           "Gain = probes restored / enhanced, gallery and distractors direct, minus direct feed, paired on the same probes, 95% bootstrap CI. "
           "'Bonferroni' is the same gain with the CI at 0.05 / number of tests (sensitivity check). Rule: closes restoration unless some arm has "
           "the CI above zero for at least 2 of the 3 recognizers.", "",
           "| Arm | " + " | ".join(NAMES[r] for r in NON_LEAKY) + " | Mean gain | CI > 0 (of 3) | Bonferroni CI > 0 (of 3) |", "|---|" + "---|" * len(NON_LEAKY) + "---|---|---|"]
    best = None
    for (arm, st), g in t.groupby(["arm", "setting"], sort=False):
        g = g.set_index("recognizer")
        cells = [f"{g.loc[r, 'probes_minus_direct_paired']:+.4f} {ci(g.loc[r, 'probes_minus_direct_ci95'], 4)}" for r in NON_LEAKY]
        mean = float(np.mean([g.loc[r, "probes_minus_direct_paired"] for r in NON_LEAKY]))
        k = sum(json.loads(g.loc[r, "probes_minus_direct_ci95"])[0] > 0 for r in NON_LEAKY)
        kb = (sum(json.loads(g.loc[r, "probes_minus_direct_ci95_bonferroni"])[0] > 0 for r in NON_LEAKY)
              if "probes_minus_direct_ci95_bonferroni" in g and g["probes_minus_direct_ci95_bonferroni"].notna().all() else None)
        out.append(f"| {arm}: {st} | " + " | ".join(cells) + f" | {mean:+.4f} | {k} | {kb if kb is not None else 'n/a'} |")
        if k >= 2 and (best is None or mean > best[1]):
            best = (f"{arm}: {st}", mean)
    out += ["", f"**Verdict (mechanical): {'winner ' + best[0] + f' (mean gain {best[1]:+.4f}); it becomes the Phase 8 gate arm' if best else 'no arm meets the rule: restoration is closed'}.**", ""]
    return out


def lfw_tables(res):
    need = [res / n for n in ("lfw_enh_calibration.csv", "lfw_enh_check.csv", "lfw_enh_contrasts.csv")]
    if not all(p.exists() for p in need):
        return ["LFW tables: pending", ""]
    cal, d, c = (pd.read_csv(p) for p in need)
    out = ["Degradations of the sharp 112 px LFW crops (6,000 official pairs); strengths chosen on direct feed only, before any arm was run "
           "(`lfw_enh_calibration.csv`): the strength whose mean non-leaky cross-degraded AUC is closest to 0.80 inside 0.70-0.90. Motion blur is a fixed "
           "11 px line at a seeded angle (not tuned).", "",
           "| Degradation | Strength | Mean non-leaky cross AUC (direct) | In 0.70-0.90 |", "|---|---|---|---|"]
    for r in cal[cal.chosen].itertuples():
        out.append(f"| {r.degradation} | {r.strength:g} | {r.mean_nonleaky_cross_auc:.3f} | {'yes' if r.in_range else 'NO'} |")
    out += ["", f"Ceiling (sharp, sharp), mean non-leaky AUC: {cal.ceiling_mean_nonleaky_auc.iloc[0]:.4f}.", ""]
    out += ["**Absolute verification AUC by condition**", "", "| Degradation | Condition | Arm | " + " | ".join(NAMES[r] for r in RECS) + " |", "|---|---|---|" + "---|" * len(RECS)]
    for (kind, cond, arm, st), g in d.groupby(["degradation", "condition", "arm", "setting"], sort=False):
        if arm == "noop" and not cond.startswith(("ceiling", "cross degraded", "symmetric degraded")):
            continue
        if cond.startswith("ceiling") and arm != "noop":
            continue
        g = g.set_index("recognizer")
        out.append(f"| {kind} | {cond} | {arm}{'' if arm == 'noop' else ': ' + st} | " + " | ".join(f"{g.loc[r, 'AUC']:.3f}" for r in RECS) + " |")
    out.append("")
    out += ["**Pre-registered contrast: cross-enhanced minus cross-degraded AUC** (rule: CI above zero for at least 2 of the 3 non-leaky recognizers). "
            "noop is the degraded crop itself, so this is also the contrast against no-op.", "",
            "| Degradation | Arm | " + " | ".join(NAMES[r] for r in RECS) + " | Non-leaky positive | Rule |", "|---|---|" + "---|" * len(RECS) + "---|---|"]
    for (kind, arm, st), g in c[c.contrast == "cross"].groupby(["degradation", "arm", "setting"], sort=False):
        g = g.set_index("recognizer")
        k = sum(json.loads(g.loc[r, "delta_AUC_ci95"])[0] > 0 for r in NON_LEAKY)
        out.append(f"| {kind} | {arm}: {st} | " + " | ".join(f"{g.loc[r, 'delta_AUC']:+.3f} {ci(g.loc[r, 'delta_AUC_ci95'])}" for r in RECS) + f" | {k} of 3 | {'met' if k >= 2 else 'not supported'} |")
    out += ["", "**Same contrast with both images degraded (symmetric)**", "", "| Degradation | Arm | " + " | ".join(NAMES[r] for r in RECS) + " |", "|---|---|" + "---|" * len(RECS)]
    for (kind, arm, st), g in c[c.contrast == "symmetric"].groupby(["degradation", "arm", "setting"], sort=False):
        g = g.set_index("recognizer")
        out.append(f"| {kind} | {arm}: {st} | " + " | ".join(f"{g.loc[r, 'delta_AUC']:+.3f} {ci(g.loc[r, 'delta_AUC_ci95'])}" for r in RECS) + " |")
    return out + [""]


def cost_table(res, arm_rows):
    """arm_rows: [(arm, setting, label)] of the restored arms. Eval timings are on the device that ran them (RTX 3000 or CPU)."""
    out = ["### Cost (s/crop; eval restoration on the device shown, smoke = 75 crops on CPU vs GPU)", "",
           "| Arm | Setting | Eval device | Eval s/crop median | Eval s/crop mean | Peak VRAM (MB) | Failed | Smoke CPU s/crop | Smoke GPU s/crop |", "|---|---|---|---|---|---|---|---|---|"]
    smoke = ROOT / "outputs/enhance/smoke"

    def sm(name):
        p = smoke / name / "timing.json"
        return f"{json.loads(p.read_text())['sec_per_crop_median']:.4f}" if p.exists() else "n/a"
    names = {"zerodce": ("zerodce_cpu", "zerodce_gpu"), "nafnet": {"pre0": ("nafnet_native_cpu", "nafnet_native_cuda"), "pre128": ("nafnet_pre128_cpu", "nafnet_pre128_cuda")}}
    for arm, st, label in arm_rows:
        tj = ROOT / "outputs/sr/restored/eval" / f"{arm}__{st}" / "seed0" / "timing.json"
        if not tj.exists():
            out.append(f"| {arm} | {st} | pending | pending | pending | pending | pending | | |")
            continue
        t = json.loads(tj.read_text())
        pair = names.get(arm)
        if isinstance(pair, dict):
            pair = pair.get(st)
        c_cpu, c_gpu = (sm(pair[0]), sm(pair[1])) if pair else ("n/a", "n/a")
        out.append(f"| {arm} | {st} | {t.get('device', 'cpu')} | {t.get('sec_per_crop_median') or 0:.4f} | {t.get('sec_per_crop_mean') or 0:.4f} | "
                   f"{t.get('peak_vram_mb') or 0:.0f} | {t.get('n_failed', 0)} | {c_cpu} | {c_gpu} |")
    return out + [""]
