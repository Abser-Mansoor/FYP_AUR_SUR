"""Inject the Phase 7 result tables into REPORT_phase7.md between <!-- TABLES:BEGIN --> and <!-- TABLES:END -->. Every number comes from a CSV or JSON written by a script.

    envs/video/Scripts/python.exe scripts/phase7_report_tables.py
A table whose file does not exist yet is replaced by a note, so the script can be rerun as results arrive.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "outputs/video"
BEGIN, END = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"
SIZE = {112: "native", 16: "16 px", 12: "12 px"}


class R(dict):
    __getattr__ = dict.__getitem__


def rows_of(df):
    return [R(r) for r in df.to_dict("records")]


def md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False)])


def rd(p):
    return pd.read_csv(p) if Path(p).exists() else None


def f(x, nd=4):
    return "" if pd.isna(x) else f"{x:.{nd}f}"


def ci(r, k, nd=4, lo="_ci_lo", hi="_ci_hi"):
    return f"{r[k]:+.{nd}f} ({r[k + lo]:+.{nd}f} to {r[k + hi]:+.{nd}f})"


def primary():
    d = rd(V / "agg7/chokepoint.csv")
    if d is None:
        return None
    r = d[(d.kind == "closed") & (d.cond == "SC") & (d["size"] == 12) & (d.rule == "R1")].iloc[0]
    return md(pd.DataFrame([dict(condition="AdaFace, GT tracks, SC, 12 px probes", n_tracks=int(r.n_tracks), single_frame_rank1=f(r.single_rank1), R1_aggregated_rank1=f(r.agg_rank1),
                                 difference_95CI=ci(r, "delta"), above_zero="yes" if r.delta_ci_lo > 0 else "no")]))


def rules(cond, size):
    d = rd(V / "agg7/chokepoint.csv")
    if d is None:
        return None
    x = d[(d.kind == "closed") & (d.cond == cond) & (d["size"] == size)]
    rows = [dict(rule=r.rule, aggregated_rank1=f(r.agg_rank1), difference_vs_single=ci(r, "delta"), bonferroni_CI=f"{r.delta_bonf_lo:+.4f} to {r.delta_bonf_hi:+.4f}", dir1=f(r.delta_dir1), dir2=f(r.delta_dir2),
                 minus_R1=ci(r, "vs_R1", lo="_ci_lo", hi="_ci_hi") if "vs_R1" in r and not pd.isna(r["vs_R1"]) else "", minus_R1_bonferroni_CI=f"{r['vs_R1_bonf_lo']:+.4f} to {r['vs_R1_bonf_hi']:+.4f}" if "vs_R1_bonf_lo" in r else "")
            for r in rows_of(x)]
    return f"single-frame Rank-1 (track-weighted): {f(x.single_rank1.iloc[0])}; tracks: {int(x.n_tracks.iloc[0])}\n\n" + md(pd.DataFrame(rows))


def main_grid():
    d = rd(V / "agg7/chokepoint.csv")
    if d is None:
        return None
    x = d[(d.kind == "closed") & (d.rule == "R1")]
    return md(pd.DataFrame([dict(cond=r.cond, probes=SIZE[int(r.size)], n_tracks=int(r.n_tracks), single=f(r.single_rank1), R1=f(r.agg_rank1), difference=ci(r, "delta")) for r in rows_of(x)]))


def open_set():
    d = rd(V / "agg7/chokepoint.csv")
    if d is None:
        return None
    x = d[d.kind == "open"]
    return md(pd.DataFrame([dict(cond=r.cond, probes=SIZE[int(r.size)], rule="R0 single frame (all frames)" if r.rule == "R0" else "R1 (track template)", rank1=f(r.rank1),
                                 tpir_fpir0_3=f(r["tpir_fpir0.3"]), tpir_fpir0_1=f(r["tpir_fpir0.1"]), tpir_fpir0_01=f"{r['tpir_fpir0.01']:.4f} ({r['tpir_fpir0.01_ci_lo']:.4f}-{r['tpir_fpir0.01_ci_hi']:.4f})")
                            for r in rows_of(x)]))


def thin():
    d = rd(V / "agg7/chokepoint_thin_gallery.csv")
    if d is None:
        return None
    return md(pd.DataFrame([dict(cond=r.cond, probes=SIZE[int(r.size)], n_tracks=int(r.n_tracks), single=f(r.single_rank1), R1=f(r.agg_rank1), difference=ci(r, "delta")) for r in rows_of(d)]))


def predicted():
    d = rd(V / "agg7/predicted.csv")
    if d is None:
        return None
    x = d[d.rule == "R1"]
    return md(pd.DataFrame([dict(tracker=r.tracker, cond=r.cond, probes=SIZE[int(r.size)], n_tracks=int(r.n_tracks), single=f(r.single_rank1), R1=f(r.agg_rank1), difference=ci(r, "delta")) for r in rows_of(x)]))


def predicted_stats():
    d = rd(V / "agg7/predicted_track_stats.csv")
    if d is None:
        return None
    x = d[d["size"] == 112]
    return md(pd.DataFrame([dict(tracker=r.tracker, cond=r.cond, direction=r.direction, tracks=int(r.n_tracks_total), scored=int(r.n_scored), excluded=int(r.n_excluded), mean_purity_scored=f(r.mean_purity_scored, 3),
                                 share_of_detections_in_excluded=f(r.share_dets_in_excluded, 3)) for r in rows_of(x)]))


def curves():
    d = rd(V / "agg7/curves_frames.csv")
    if d is None:
        return None
    x = d[d.cond == "SC"].pivot(index="n_frames", columns="size", values="acc_mean_over_draws").reindex(["1", "2", "4", "8", "16", "all"])
    x.columns = [SIZE[int(c)] for c in x.columns]
    x = x.round(4).reset_index().rename(columns={"n_frames": "frames per track"})
    q = rd(V / "agg7/curves_quality.csv")
    q = q[(q.cond == "SC") & (q["size"] == 12)]
    qt = md(pd.DataFrame([dict(quartile=r.quartile, tracks=int(r.n_tracks), mean_MagFace_norm=f(r.q_mean, 2), single_frame=f(r.single_frame_acc), R1=f(r.R1_acc)) for r in rows_of(q)]))
    return md(x[["frames per track", "native", "16 px", "12 px"]]) + "\n\nBy the track's mean MagFace norm (SC, 12 px, quartiles over tracks):\n\n" + qt


def checks():
    p = V / "agg7/checks.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    out = []
    if "unit" in c:
        out.append(f"- Checks 1 and 2 (synthetic tracks, all rules): all pass = {c['unit']['all_pass']}.")
    if "check3_single_frame_reference" in c:
        out.append(f"- Check 3 (single-frame reference equals Phase 6 section 0 and E to 5 decimals, {len(c['check3_single_frame_reference']['rows'])} rows): {c['check3_single_frame_reference']['all_equal']}.")
    for k in sorted(x for x in c if x.startswith("check4") or x.startswith("check5")):
        out.append(f"- {k}: {json.dumps(c[k])[:300]}")
    return "\n".join(out)


def ytf_tables():
    out = []
    for name, title in (("ytf_verification.csv", "YouTube Faces verification (mean over folds, SE)"), ("ytf_paired.csv", "YouTube Faces: paired difference against the single-frame reference")):
        d = rd(V / "agg7" / name)
        if d is not None:
            out.append(f"{title}\n\n" + md(d.round(4)))
    return "\n\n".join(out) if out else None


def caface_cp():
    d = rd(V / "agg7/caface_chokepoint.csv")
    if d is None:
        return None
    return md(pd.DataFrame([dict(cond=r.cond, probes=SIZE[int(r.size)], n_tracks=int(r.n_tracks), method="CAFace aggregator" if r.rule == "CAFACE" else "R1 plain mean (CAFace backbone)", single_frame=f(r.single_rank1),
                                 aggregated=f(r.agg_rank1), difference_vs_single=ci(r, "delta"), minus_R1=ci(r, "vs_R1") if "vs_R1" in r and not pd.isna(r["vs_R1"]) else "") for r in rows_of(d)]))


def caface_ytf():
    out = []
    for name, title in (("ytf_caface_verification.csv", "CAFace backbone on YouTube Faces (mean over folds)"), ("ytf_caface_paired.csv", "CAFace backbone on YouTube Faces: paired differences")):
        d = rd(V / "agg7" / name)
        if d is not None:
            out.append(f"{title}" + chr(10) * 2 + md(d.round(4)))
    return (chr(10) * 2).join(out) if out else None


def backbone():
    d = rd(V / "agg7/backbone_chokepoint.csv")
    y = rd(V / "agg7/backbone_ytf.csv")
    if d is None or y is None:
        return None

    def c(r, k, nd=4):
        return f"{r[k]:+.{nd}f} ({r[k + '_ci_lo']:+.{nd}f} to {r[k + '_ci_hi']:+.{nd}f})"
    rows = [dict(data="ChokePoint GT tracks", cond=r.cond, probes=SIZE[int(r["size"])], units=int(r.n_tracks), single_A=f(r.single_A), single_B=f(r.single_B), R1_A=f(r.R1_A), R1_B=f(r.R1_B),
                 B_minus_A_single=c(r, "B_minus_A_single"), B_minus_A_R1=c(r, "B_minus_A_R1"), B_minus_A_aggregation_gain=c(r, "B_minus_A_gain")) for r in rows_of(d)]
    r = rows_of(y)[0]
    rows.append(dict(data="YouTube Faces (pairs)", cond="", probes="native", units=int(r.pairs), single_A=f(r.single_A), single_B=f(r.single_B), R1_A=f(r.R1_A), R1_B=f(r.R1_B),
                     B_minus_A_single=c(r, "B_minus_A_single"), B_minus_A_R1=c(r, "B_minus_A_R1"), B_minus_A_aggregation_gain=c(r, "B_minus_A_gain")))
    return md(pd.DataFrame(rows))


def stress():
    d = rd(V / "agg7/tracker_stress/summary.csv")
    if d is None:
        return None
    lab = {"S0": "S0 original (reference)", "S1": "S1 every 3rd frame", "S2": "S2 every 5th frame", "S3": "S3 every 10th frame", "S4": "S4 10% detections dropped", "S5": "S5 20% detections dropped"}
    t = md(pd.DataFrame([dict(condition=lab[r.cond], IDF1_ByteTrack=f(r.IDF1_bytetrack), IDF1_BoT_SORT=f(r.IDF1_botsort), IDF1_difference=f"{r.IDF1_diff:+.4f}", HOTA_difference=f"{r.HOTA_diff:+.4f}",
                                tracks_BT_vs_BoT=f"{int(r.tracks_bytetrack)} / {int(r.tracks_botsort)}", ID_switches_BT_vs_BoT=f"{int(r.idsw_total_bytetrack)} / {int(r.idsw_total_botsort)}") for r in rows_of(d)]))
    p = V / "agg7/tracker_stress/decision.json"
    if p.exists():
        j = json.loads(p.read_text())
        t += (f"\n\nPrimary statistic (mean over clips of the per-clip IDF1 difference, ByteTrack minus BoT-SORT, averaged over S1 to S5): {j['mean_difference']:+.4f} "
              f"(95% CI {j['ci95'][0]:+.4f} to {j['ci95'][1]:+.4f}); ByteTrack better in {j['clips_bytetrack_better']} of {j['n_clips']} clips, BoT-SORT in {j['clips_botsort_better']}; "
              f"CI excludes zero: {j['ci_excludes_zero']}; provisional default: {j['provisional_default']}.")
    e = rd(V / "agg7/tracker_stress/exploratory_botsort_proximity0.8/summary.csv")
    if e is not None:
        t += ("\n\nExploratory (not part of the decision): BoT-SORT with its association gate relaxed from IoU 0.5 to 0.2 (`proximity_thresh` 0.8):\n\n" +
              md(pd.DataFrame([dict(condition=lab[r.cond], IDF1_ByteTrack=f(r.IDF1_bytetrack), IDF1_BoT_SORT_default=f(r.IDF1_botsort_default), IDF1_BoT_SORT_relaxed=f(r.IDF1_botsort_relaxed),
                                    tracks_BT_default_relaxed=f"{int(r.tracks_bytetrack)} / {int(r.tracks_botsort_default)} / {int(r.tracks_botsort_relaxed)}",
                                    ID_switches_BT_default_relaxed=f"{int(r.idsw_bytetrack)} / {int(r.idsw_botsort_default)} / {int(r.idsw_botsort_relaxed)}") for r in rows_of(e)])))
    return t


def build():
    parts = [("Primary hypothesis (ChokePoint, pre-registered)", primary()),
             ("Rules at 12 px (SC): R1 against the single frame, plain and Bonferroni CIs", rules("SC", 12)),
             ("Rules at 16 px (SC)", rules("SC", 16)),
             ("Rules at 12 px (MC)", rules("MC", 12)),
             ("R1 against the single frame, all conditions (GT tracks, AdaFace, mean of the two directions; difference = aggregated minus single frame, track-weighted)", main_grid()),
             ("Open set (TPIR20 at FPIR 0.3, 0.1, 0.01; CI for 0.01)", open_set()),
             ("B k = 1 gallery (one enrolment frame, 20 draws) with aggregated probes", thin()),
             ("Predicted tracks (ByteTrack, BoT-SORT), R1 against the single frame", predicted()),
             ("Predicted tracks: purity and excluded tracks (native, per direction)", predicted_stats()),
             ("Tracker stress test (Addendum 2): mean over the 72 clips", stress()),
             ("Curves: accuracy against the number of frames used (R1, SC, 20 seeded draws)", curves()),
             ("Acceptance checks", checks()),
             ("YouTube Faces", ytf_tables()),
             ("CAFace arm on ChokePoint GT tracks (CAFace backbone: AdaFace R100 WebFace4M; never pooled with the MS1MV2 rows)", caface_cp()),
             ("CAFace arm on YouTube Faces", caface_ytf()),
             ("Backbone comparison (exploratory, Addendum 1): A = AdaFace R100 MS1MV2, B = AdaFace R100 WebFace4M; paired differences B minus A", backbone())]
    return "\n".join(f"#### {t}\n\n" + (b if b else "_not available yet_") + "\n" for t, b in parts)


if __name__ == "__main__":
    p = ROOT / "REPORT_phase7.md"
    s = p.read_text(encoding="utf-8")
    block = f"{BEGIN}\n\n{build()}\n{END}"
    if BEGIN in s:
        s = s[:s.index(BEGIN)] + block + s[s.index(END) + len(END):]
    else:
        s = s.rstrip() + "\n\n## Result tables\n\n" + block + "\n"
    p.write_text(s, encoding="utf-8")
    print("tables injected")
