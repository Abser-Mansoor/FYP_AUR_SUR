"""Phase 9 Part C: ChokePoint end-to-end with the final configuration (REPORT_phase9.md section 0.6 C). CPU, from cache. Not held out (Phases 6-8 used ChokePoint).

    envs/video/Scripts/python.exe scripts/video_tracker_stress.py --lam 0.5 --backbone webface4m   # tracks with the face term on WebFace4M cosines (Part A)
    envs/video/Scripts/python.exe scripts/phase9_chokepoint.py

Two arms on the same folds and protocol as Phase 8 Part D (scripts/mevid_tracker_identity.py: 29 SC identities, 5 rotated folds, G1<->G2, FPIR 0.1):
  phase8 = MS1MV2 embeddings, R1 track template, tracks with the MS1MV2 face term (outputs/video/mevid/tracker_face/lam0.5)
  final  = WebFace4M embeddings (embed_webface4m.npz), the frozen rule (R4*, q = MagFace norm), tracks with the WebFace4M face term (outputs/video/phase9/tracker_face/lam0.5_webface4m)
Gallery templates: the mean of the person's matched embeddings in the enrolling group (Phase 6 protocol), in each arm's backbone.
Also: the share of scored tracks under the 23 px bar (mean height of usable faces: score >= 0.5, height >= 16 px). Outputs: outputs/video/phase9/chokepoint_*.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.video import aggregate as A  # noqa: E402
from mevid_tracker_identity import CONDS, FPIR, N_FOLDS, PIPE, SEED, l2n  # noqa: E402

P9 = ROOT / "outputs/video/phase9"
FROZEN = ROOT / "outputs/video/mevid/phase9/frozen_config.json"
ARMS = {"phase8": dict(file="embed.npz", key="emb_adaface", rule="R1", tracks=ROOT / "outputs/video/mevid/tracker_face/lam0.5"),
        "final": dict(file="embed_webface4m.npz", key="emb_adaface_wf4m", rule=None, tracks=P9 / "tracker_face/lam0.5_webface4m")}
BAR, MIN_H = 23.0, 16.0


def clip_tracks(clip, assign, match_pid, arm):
    z = np.load(PIPE / clip / arm["file"])
    zq = np.load(PIPE / clip / "embed.npz")
    assert np.array_equal(z["det_id"], zq["det_id"])
    det = np.load(PIPE / clip / "detections.npz")["dets"]
    row_of = {int(d): i for i, d in enumerate(z["det_id"])}
    emb, q = z[arm["key"]], zq["mag_norm"]
    out = []
    for tid in np.unique(assign[assign >= 0]):
        rows = [r for r in np.flatnonzero(assign == tid) if int(r) in row_of]
        if not rows:
            continue
        pids = [match_pid[int(r)] for r in rows if int(r) in match_pid]
        if not pids or len(pids) / len(rows) < 0.5:
            continue
        label = pd.Series(pids).value_counts().index[0]
        k = [row_of[int(r)] for r in rows]
        t = A.RULES[arm["rule"]](l2n(emb[k]), q[k], np.array(rows))
        h = det[rows, 3] - det[rows, 1]
        hu = h[h >= MIN_H]
        out.append((int(label), t, len(rows), float(hu.mean()) if len(hu) else np.nan))
    return out


def run():
    import video_phase7_bench as B7
    import video_single_frame as S
    cfg, allc, m, lo = B7.setup()
    rule = json.loads(FROZEN.read_text())["config"]["rule"] if FROZEN.exists() else "R4*"
    ARMS["final"]["rule"] = rule
    persons = sorted(m.pid.unique().tolist())
    folds = [set(g.tolist()) for g in np.array_split(np.random.default_rng(SEED).permutation(np.array(persons)), N_FOLDS)]
    mp = {c: dict(zip(g.det_id, g.pid)) for c, g in m.groupby("clip")}
    rows = []
    for name, arm in ARMS.items():
        gal = {}
        for g in ("G1", "G2"):
            pid, e = [], []
            for c in S.clips_of(g, "SC", allc):
                s_ = m[m["clip"] == c].sort_values(["frame", "det_id"])
                if len(s_) == 0:
                    continue
                z = np.load(PIPE / c / arm["file"])
                pos = {int(d): i for i, d in enumerate(z["det_id"])}
                e.append(z[arm["key"]][[pos[int(d)] for d in s_.det_id]])
                pid.append(s_.pid.to_numpy())
            pid, e = np.concatenate(pid), l2n(np.concatenate(e))
            ids = np.array(sorted(set(pid.tolist())))
            gal[g] = (ids, np.stack([l2n(e[pid == p].mean(0)) for p in ids]))
        for g, pg in (("G1", "G2"), ("G2", "G1")):
            ids, T = gal[g]
            for clip in S.clips_of(pg, "SC", allc):
                z = np.load(arm["tracks"] / "tracks" / f"{clip}.npz")
                for cond in CONDS:
                    tr = clip_tracks(clip, z[f"{cond}|bytetrack"], mp.get(clip, {}), arm)
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
                    hs = np.array([t[3] for t in tr])
                    rows.append(dict(arm=name, direction=f"{g}->{pg}", clip=clip, cond=cond, scored_tracks=len(tr), evaluations=total, correct=correct, accuracy=(correct / total) if total else np.nan,
                                     tracks_below_bar=int(np.sum(~(hs >= BAR))) if len(hs) else 0))
        print(name, "done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(P9 / "chokepoint_identity_per_clip.csv", index=False, float_format="%.5f")
    s = d.groupby(["cond", "arm"]).agg(accuracy=("accuracy", "mean"), scored_tracks=("scored_tracks", "sum"), tracks_below_bar=("tracks_below_bar", "sum")).unstack("arm")
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.reset_index()
    # paired per clip (final minus phase8, mean over S0-S5), cluster bootstrap over clips
    x = d.pivot_table(index=["clip", "cond"], columns="arm", values="accuracy")
    diff = (x["final"] - x["phase8"]).unstack("cond")[CONDS].mean(axis=1, skipna=True).dropna()
    b = diff.to_numpy()[np.random.default_rng(SEED).integers(0, len(diff), (1000, len(diff)))].mean(1)
    s.to_csv(P9 / "chokepoint_identity_summary.csv", index=False, float_format="%.5f")
    res = dict(mean_diff_final_minus_phase8=float(diff.mean()), ci95=[float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))], n_clips=int(len(diff)), rule_final=rule)
    (P9 / "chokepoint_identity_diff.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    print(s.round(4).to_string(index=False))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    run()
