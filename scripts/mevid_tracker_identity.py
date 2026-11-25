"""Phase 8 Part D: downstream open-set track-identity accuracy and the tracker decision (protocol: REPORT_phase8.md section 0.6, committed before any number). CPU, from cache.

    envs/video/Scripts/python.exe scripts/video_tracker_stress.py --lam 0        # writes outputs/video/mevid/tracker_face/lam0/{per_clip.csv, tracks/}  (acceptance check 5 baseline)
    envs/video/Scripts/python.exe scripts/video_tracker_stress.py --lam 0.5      # the face term (the Phase 7 S0-S5 conditions with the added cost)
    envs/video/Scripts/python.exe scripts/mevid_tracker_identity.py --check5     # lambda 0 reproduces the Phase 7 per_clip.csv byte for byte
    envs/video/Scripts/python.exe scripts/mevid_tracker_identity.py              # track-identity accuracy, summary tables, decision.json

Definition (section 0.6): the 29 ChokePoint identities are cut into 5 folds (numpy default_rng(42)); in fold k the held-out group is not enrolled. Gallery templates: the Phase 6 protocol,
G1 enrols and G2 probes and the reverse (SC), template = the normalised mean of the person's matched embeddings in the enrolling group. Probes: predicted tracks (all their detections with an
embedding, i.e. score >= 0.5; label = majority person id of the matched ones; tracks with < 50% matched detections are not scored, as Phase 7), template = R1. Output of a probe: the top-1 enrolled
identity if its similarity >= t_0.1 (the 0.9 quantile of the top-1 similarities of the held-out group's tracks of the same condition, tracker, lambda and direction), else "unknown". Correct:
an enrolled person's track identified correctly, or a held-out person's track output as "unknown". Per clip: correct / scored tracks over folds; the decision statistic is the per-clip difference
ByteTrack minus BoT-SORT averaged over S0-S5 (with the face term), cluster bootstrap over clips (1,000 draws, seed 42); the tracker with the higher mean wins if the CI excludes zero, else ByteTrack.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "outputs/video/mevid"
PIPE = ROOT / "outputs/video/pipeline/chokepoint"
CONDS = ["S0", "S1", "S2", "S3", "S4", "S5"]
TRACKERS = ["bytetrack", "botsort"]
LAMS = [0.0, 0.5]
SEED, N_BOOT, N_FOLDS, FPIR = 42, 1000, 5, 0.1


def l2n(x):
    x = np.asarray(x, np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def clip_tracks(clip, assign, match_pid):
    """Predicted tracks of one clip: list of (label, template, n_dets) for the scored tracks. assign = track id per detection row (-1 none); match_pid = {det_id: person}."""
    z = np.load(PIPE / clip / "embed.npz")
    row_of = {int(d): i for i, d in enumerate(z["det_id"])}
    emb = z["emb_adaface"]
    out = []
    for tid in np.unique(assign[assign >= 0]):
        rows = [r for r in np.flatnonzero(assign == tid) if int(r) in row_of]  # detections with an embedding (score >= 0.5)
        if not rows:
            continue
        pids = [match_pid[int(r)] for r in rows if int(r) in match_pid]
        if not pids or len(pids) / len(rows) < 0.5:
            continue
        label = pd.Series(pids).value_counts().index[0]
        out.append((int(label), l2n(emb[[row_of[int(r)] for r in rows]].mean(0)), len(rows)))
    return out


def identity(lam_dirs=LAMS):
    import video_phase7_bench as B7
    import video_single_frame as S
    cfg, allc, m, lo = B7.setup()
    persons = sorted(m.pid.unique().tolist())
    folds = [set(g.tolist()) for g in np.array_split(np.random.default_rng(SEED).permutation(np.array(persons)), N_FOLDS)]
    box = {}
    gal = {}
    for g in ("G1", "G2"):
        pid, e = S.build("adaface", S.clips_of(g, "SC", allc), m, cfg, box)
        ids = np.array(sorted(set(pid.tolist())))
        gal[g] = (ids, np.stack([l2n(e[pid == p].mean(0)) for p in ids]))
    mp = {c: dict(zip(g.det_id, g.pid)) for c, g in m.groupby("clip")}
    rows = []
    for lam in lam_dirs:
        for g, pg in (("G1", "G2"), ("G2", "G1")):
            ids, T = gal[g]
            for clip in S.clips_of(pg, "SC", allc):
                z = np.load(OUT / "tracker_face" / f"lam{lam:g}" / "tracks" / f"{clip}.npz")
                for cond in CONDS:
                    for tname in TRACKERS:
                        tr = clip_tracks(clip, z[f"{cond}|{tname}"], mp.get(clip, {}))
                        correct = total = 0
                        for held in folds:
                            keep = np.array([p not in held for p in ids])
                            ids_k, T_k = ids[keep], T[keep]
                            non = [t for t in tr if t[0] in held]
                            mated = [t for t in tr if t[0] in set(ids_k.tolist())]
                            if not non:
                                continue
                            sim_non = np.array([(T_k @ t[1]).max() for t in non])
                            thr = np.quantile(sim_non, 1 - FPIR)
                            correct += sum(1 for s_ in sim_non if s_ < thr)
                            for t in mated:
                                s = T_k @ t[1]
                                correct += int(ids_k[int(np.argmax(s))] == t[0] and s.max() >= thr)
                            total += len(non) + len(mated)
                        rows.append(dict(lam=lam, direction=f"{g}->{pg}", clip=clip, cond=cond, tracker=tname, scored_tracks=len(tr), evaluations=total, correct=correct,
                                         accuracy=(correct / total) if total else np.nan))
        print("lambda", lam, "done", flush=True)
    return pd.DataFrame(rows)


def decide(d):
    res = OUT / "results"
    d.to_csv(res / "tracker_identity_per_clip.csv", index=False, float_format="%.5f")
    s = d.groupby(["lam", "cond", "tracker"]).accuracy.mean().unstack("tracker").reset_index()
    s["bytetrack_minus_botsort"] = s.bytetrack - s.botsort
    cnt = d.groupby(["lam", "cond", "tracker"]).scored_tracks.sum().unstack("tracker").reset_index().rename(columns={"bytetrack": "scored_tracks_bytetrack", "botsort": "scored_tracks_botsort"})
    s = s.merge(cnt, on=["lam", "cond"])  # the accuracy is over scored tracks only, so the counts are shown next to it
    s.to_csv(res / "tracker_identity_summary.csv", index=False, float_format="%.5f")
    out = {}
    for lam in sorted(d.lam.unique()):
        x = d[d.lam == lam].pivot_table(index=["clip", "cond"], columns="tracker", values="accuracy")
        diff = (x.bytetrack - x.botsort).unstack("cond")[CONDS].mean(axis=1, skipna=True).dropna()
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(diff), (N_BOOT, len(diff)))
        b = diff.to_numpy()[idx].mean(1)
        lo_, hi_ = float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
        mean = float(diff.mean())
        excl = bool(lo_ > 0 or hi_ < 0)
        out[f"lam{lam:g}"] = dict(mean_diff_bytetrack_minus_botsort=mean, ci95=[lo_, hi_], n_clips=int(len(diff)), ci_excludes_zero=excl, winner=("bytetrack" if mean > 0 else "botsort") if excl else "bytetrack (default)",
                                  mean_accuracy=dict(bytetrack=float(x.bytetrack.mean()), botsort=float(x.botsort.mean())))
    out["decision_rule"] = "the tracker with the higher mean track-identity accuracy over S0-S5 with the face term (lam 0.5) wins if the CI of the per-clip difference excludes zero; otherwise ByteTrack stays the default"
    out["decision"] = out["lam0.5"]["winner"]
    (res / "tracker_decision.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(s.round(4).to_string(index=False))
    print(json.dumps(out, indent=1, sort_keys=True))


def check5():
    a = ROOT / "outputs/video/agg7/tracker_stress/per_clip.csv"
    b = OUT / "tracker_face/lam0/per_clip.csv"
    ha, hb = hashlib.sha256(a.read_bytes()).hexdigest(), hashlib.sha256(b.read_bytes()).hexdigest()
    res = dict(phase7_per_clip_sha256=ha, lambda0_per_clip_sha256=hb, byte_identical=bool(ha == hb), rows_phase7=int(len(pd.read_csv(a))), rows_lambda0=int(len(pd.read_csv(b))))
    p = OUT / "checks.json"
    allres = json.loads(p.read_text()) if p.exists() else {}
    allres["check5_lambda0_reproduces_phase7_tracker_results"] = res
    p.write_text(json.dumps(allres, indent=1, sort_keys=True))
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check5", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="rebuild the summary and the decision from results/tracker_identity_per_clip.csv without recomputing")
    a = ap.parse_args()
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    check5() if a.check5 else decide(pd.read_csv(OUT / "results" / "tracker_identity_per_clip.csv") if a.summarize else identity())
