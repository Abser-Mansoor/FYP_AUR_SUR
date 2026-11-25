"""Inject the Phase 8 result tables into REPORT_phase8.md between <!-- TABLES:BEGIN --> and <!-- TABLES:END -->. Every number comes from a CSV or JSON written by a script.

    envs/video/Scripts/python.exe scripts/phase8_report_tables.py
A table whose file does not exist yet is replaced by a note, so the script can be rerun as results arrive.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs/video/mevid"
R = M / "results"
BEGIN, END = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"


class Row(dict):
    __getattr__ = dict.__getitem__


def rows_of(df):
    return [Row(r) for r in df.to_dict("records")]


def md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False)])


def rd(p):
    return pd.read_csv(p) if Path(p).exists() else None


def f4(x):
    return "" if pd.isna(x) else f"{x:.4f}"


def pm(r, k, nd=4):
    return f"{r[k]:+.{nd}f} ({r[k + '_ci_lo']:+.{nd}f} to {r[k + '_ci_hi']:+.{nd}f})"


def js(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def primary():
    c, o = rd(R / "closed_adaface.csv"), rd(R / "open_adaface.csv")
    if c is None or o is None:
        return None
    h1 = c[(c.rule == "R1") & c.setting.str.contains("bar 48")].iloc[0]
    h2 = o[(o.metric == "TPIR1") & (o.fpir == 0.1) & (o.probe == "R1 minus single frame") & o.setting.str.contains("bar 48")].iloc[0]
    s2 = o[(o.metric == "TPIR1") & (o.fpir == 0.1) & (o.probe == "single frame") & o.setting.str.contains("bar 48")].iloc[0]
    r2 = o[(o.metric == "TPIR1") & (o.fpir == 0.1) & (o.probe == "R1 tracklet") & o.setting.str.contains("bar 48")].iloc[0]
    return md(pd.DataFrame([
        dict(hypothesis="H1: closed-set Rank-1, R1 minus single frame (paired per tracklet)", tracklets=int(h1.n_tracklets), persons=int(h1.n_persons), single=f4(h1.single_rank1), R1=f4(h1.agg_rank1),
             difference_95CI=f"{h1.delta:+.4f} ({h1.delta_ci_lo:+.4f} to {h1.delta_ci_hi:+.4f})", above_zero="yes" if h1.delta_ci_lo > 0 else "no"),
        dict(hypothesis="H2: open-set TPIR1 at FPIR 0.1, R1 minus single frame (fold-averaged)", tracklets="", persons="", single=f4(s2.value), R1=f4(r2.value),
             difference_95CI=f"{h2.value:+.4f} ({h2.ci_lo:+.4f} to {h2.ci_hi:+.4f})", above_zero="yes" if h2.ci_lo > 0 else "no")]))


def closed(model):
    c = rd(R / f"closed_{model}.csv")
    if c is None:
        return None
    return md(pd.DataFrame([dict(setting=r.setting, rule=r.rule, tracklets=int(r.n_tracklets), persons=int(r.n_persons), single=f4(r.single_rank1), aggregated=f4(r.agg_rank1),
                                 difference_vs_single=f"{r.delta:+.4f} ({r.delta_ci_lo:+.4f} to {r.delta_ci_hi:+.4f})",
                                 minus_R1=(f"{r['minus_R1']:+.4f} ({r['minus_R1_ci_lo']:+.4f} to {r['minus_R1_ci_hi']:+.4f})" if "minus_R1" in r and not pd.isna(r["minus_R1"]) else ""),
                                 minus_R1_Bonferroni_CI=(f"{r['minus_R1_bonf_lo']:+.4f} to {r['minus_R1_bonf_hi']:+.4f}" if "minus_R1_bonf_lo" in r and not pd.isna(r["minus_R1_bonf_lo"]) else "")) for r in rows_of(c)]))


def open_set(model):
    o = rd(R / f"open_{model}.csv")
    if o is None:
        return None
    return md(pd.DataFrame([dict(setting=r.setting, metric=r.metric, fpir=r.fpir, probe=r.probe, value=f"{r.value:.4f}", CI=f"{r.ci_lo:.4f} to {r.ci_hi:.4f}") for r in rows_of(o)]))


def simple(name):
    d = rd(R / name)
    return None if d is None else md(d.round(4))


def eligibility():
    t = rd(M / "tracklets.csv")
    if t is None:
        return None
    t = t[t.detected.fillna(False).astype(bool)]
    g = t.groupby(["split", "camera"]).agg(tracklets=("tracklet", "size"), eligible=("eligible", "sum")).reset_index()
    g["share_eligible"] = (g.eligible / g.tracklets).round(3)
    tot = t.groupby("split").agg(tracklets=("tracklet", "size"), eligible=("eligible", "sum"), persons_with_eligible=("person", lambda s: s[t.loc[s.index, "eligible"]].nunique())).reset_index()
    return md(tot) + "\n\nBy camera:\n\n" + md(g)


def quality():
    out = []
    for n, title in (("quality_mevid.csv", "MEVID (primary)"), ("quality_chokepoint.csv", "ChokePoint 12 and 16 px (secondary)")):
        d = rd(R / n)
        if d is not None:
            out.append(f"{title}\n\n" + md(pd.DataFrame([dict(dataset=r.dataset, signal=r.signal, tracklets=int(r.n_tracklets), overall_error=f4(r.overall_error), AUC=f4(r.auc), AUC_random=f4(r.auc_random),
                                                              difference=f"{r.auc_minus_random:+.4f}", CI95=f"{r.ci_lo:+.4f} to {r.ci_hi:+.4f}", Bonferroni_CI=f"{r.bonf_lo:+.4f} to {r.bonf_hi:+.4f}", helps=("exploratory (not in the verdict)" if r.get("exploratory") else ("yes" if r.helps else "no"))) for r in rows_of(d)])))
    return "\n\n".join(out) if out else None


def tracker():
    dec, s = js(R / "tracker_decision.json"), rd(R / "tracker_identity_summary.csv")
    if dec is None or s is None:
        return None
    t = md(pd.DataFrame([dict(lambda_=r.lam, condition=r.cond, ByteTrack=f4(r.bytetrack), BoT_SORT=f4(r.botsort), difference=f"{r.bytetrack_minus_botsort:+.4f}") for r in rows_of(s)]))
    for k in ("lam0", "lam0.5"):
        if k in dec:
            x = dec[k]
            t += (f"\n\nlambda {k[3:]}: mean per-clip difference ByteTrack minus BoT-SORT over S0 to S5 = {x['mean_diff_bytetrack_minus_botsort']:+.4f} (95% CI {x['ci95'][0]:+.4f} to {x['ci95'][1]:+.4f}, "
                  f"{x['n_clips']} clips); CI excludes zero: {x['ci_excludes_zero']}.")
    t += f"\n\n**Decision by the rule fixed in section 0: {dec['decision']}.**"
    return t


def tracker_idf1():
    a, b = rd(M / "tracker_face/lam0/summary.csv"), rd(M / "tracker_face/lam0.5/summary.csv")
    if a is None or b is None:
        return None
    a, b = a.assign(lam=0.0), b.assign(lam=0.5)
    d = pd.concat([a, b])
    return md(d[["lam", "cond", "tracker", "IDF1", "HOTA", "IDSW", "n_tracks"]].round(4))


def checks():
    c = js(M / "checks.json")
    if c is None:
        return None
    return "\n".join(f"- `{k}`: {json.dumps(v, sort_keys=True)[:360]}" for k, v in sorted(c.items()))


def build():
    ic = js(M / "index_counts.json")
    parts = [("Counts (acceptance check 1)", None if ic is None else "```\n" + json.dumps(ic, indent=1, sort_keys=True) + "\n```"),
             ("Tracklet eligibility (>= 5 usable faces of up to 50 chips)", eligibility()),
             ("Enrolment (persons with a usable gallery photo)", simple("enrolment_adaface.csv")),
             ("Primary hypotheses H1 and H2 (AdaFace MS1MV2, gallery bar 48 px)", primary()),
             ("Closed set: rules against the single frame and against R1 (AdaFace MS1MV2)", closed("adaface")),
             ("Open set: TPIR1 and TPIR20, fold-averaged (AdaFace MS1MV2)", open_set("adaface")),
             ("Open set per fold", simple("folds_adaface.csv")),
             ("People without a usable gallery photo as non-enrolled probes", simple("no_photo_adaface.csv")),
             ("By camera (closed set, R1 against single frame)", simple("camera_adaface.csv")),
             ("By mean face height (closed set, R1 against single frame)", simple("height_adaface.csv")),
             ("WebFace4M backbone: closed set", closed("webface4m")),
             ("WebFace4M backbone: open set", open_set("webface4m")),
             ("Backbone comparison, exploratory (not pre-registered): paired per tracklet, B = WebFace4M minus A = MS1MV2", simple("backbone_pair.csv")),
             ("Part C: frame-quality reject test (AUC of error against rejected fraction, difference to random rejection)", quality()),
             ("Part D: tracker stress conditions with and without the face term (IDF1, HOTA, ID switches)", tracker_idf1()),
             ("Part D: downstream open-set track-identity accuracy and the decision", tracker()),
             ("Acceptance checks", checks())]
    return "\n".join(f"#### {t}\n\n" + (b if b else "_not available yet_") + "\n" for t, b in parts)


if __name__ == "__main__":
    p = ROOT / "REPORT_phase8.md"
    s = p.read_text(encoding="utf-8")
    block = f"{BEGIN}\n\n{build()}\n{END}"
    if BEGIN in s:
        s = s[:s.index(BEGIN)] + block + s[s.index(END) + len(END):]
    else:
        s = s.rstrip() + "\n\n## Result tables\n\n" + block + "\n"
    p.write_text(s, encoding="utf-8")
    print("tables injected")
