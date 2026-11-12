"""Phase 6 single-frame identification baseline on ChokePoint. Protocol: REPORT_phase6.md section 0 (pre-registered, committed before any number).

    envs/video/Scripts/python.exe scripts/video_single_frame.py --recognizers adaface magface curricularface

Steps (all resumable; caches under outputs/video/baseline/):
  1. match   detections (score >= pipeline.embed_floor) to ground-truth boxes per frame, IoU >= 0.5, one-to-one, highest IoU first -> matches.csv, match_stats.json
  2. embed   AdaFace embeddings come from the pipeline run (embed.npz); other recognizers are computed for the matched detections only -> emb/<rec>/<clip>.npz
  3. score   G1 enrols / G2 probes and the reverse; SC (listed camera) and MC (all three cameras of the listed sequence); closed-set Rank-1 and open-set TPIR20@FPIR=0.3
             (survface_openset: top-20 list, the Phase 1/3 arithmetic); 20% of identities left out of the gallery (rng 42, drawn once); cluster bootstrap over identities
Outputs: results.csv (per direction), summary.csv (mean of the two directions, CI over bootstrap draws), match_stats.json, dropped.csv.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402,F401  (before onnxruntime)
from src import eval_protocols as ev  # noqa: E402
from video_tracker_eval import load_gt  # noqa: E402

OUT = ROOT / "outputs/video/baseline"
PIPE = "chokepoint"
G = {"G1": "P1E_S1_C1 P1E_S2_C2 P2E_S2_C2 P2E_S1_C3 P1L_S1_C1 P1L_S2_C2 P2L_S2_C2 P2L_S1_C1".split(),
     "G2": "P1E_S3_C3 P1E_S4_C1 P2E_S4_C2 P2E_S3_C1 P1L_S3_C3 P1L_S4_C1 P2L_S4_C2 P2L_S3_C3".split()}
N_BOOT, SEED, LEAVE_OUT = 1000, 42, 0.2


def base(clip):  # P2E_S2_C2.1 -> P2E_S2_C2
    return re.sub(r"\.\d+$", "", clip)


def session(clip):  # P2E_S2_C2.1 -> P2E_S2
    return re.sub(r"_C\d$", "", base(clip))


def clips_of(group, cond, all_clips):
    """SC: the clips of the listed sequences. MC: every camera of the listed sessions."""
    seqs = set(G[group])
    if cond == "SC":
        return [c for c in all_clips if base(c) in seqs]
    sess = {session(s) for s in seqs}
    return [c for c in all_clips if session(c) in sess]


def iou_mat(a, b):
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    i = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return i / (aa[:, None] + ab[None, :] - i + 1e-9)


def match_clip(clip, gt, floor, thr=0.5):
    t = pd.read_csv(ROOT / f"outputs/video/pipeline/{PIPE}/{clip}/tracks.csv")
    t = t[t.score >= floor]
    rows, n_fp = [], 0
    for f, r in t.groupby("frame"):
        d = r[["x1", "y1", "x2", "y2"]].to_numpy()
        ids, boxes = gt[clip].get(int(f), (np.zeros(0, int), np.zeros((0, 4))))
        used_d, used_g = set(), set()
        if len(boxes):
            m = iou_mat(d, boxes)
            for k in np.argsort(-m, axis=None, kind="stable"):
                i, j = divmod(int(k), m.shape[1])
                if m[i, j] < thr:
                    break
                if i in used_d or j in used_g:
                    continue
                used_d.add(i), used_g.add(j)
                rows.append((clip, int(f), int(ids[j]), int(r.det_id.iloc[i]), float(m[i, j])))
        n_fp += len(d) - len(used_d)
    n_gt = sum(len(v[0]) for v in gt[clip].values())
    return rows, dict(n_det=int(len(t)), n_fp=int(n_fp), n_gt=int(n_gt))


def get_matches(clips, gt, floor):
    f = OUT / "matches.csv"
    if f.exists():
        m = pd.read_csv(f)
        if set(m["clip"].unique()) | set(json.loads((OUT / "match_stats.json").read_text())) >= set(clips):
            return m, json.loads((OUT / "match_stats.json").read_text())
    rows, stats = [], {}
    for k, c in enumerate(clips, 1):
        r, s = match_clip(c, gt, floor)
        rows += r
        stats[c] = dict(s, n_matched=len(r))
        print(f"match [{k}/{len(clips)}] {c} {stats[c]}", flush=True)
    m = pd.DataFrame(rows, columns=["clip", "frame", "pid", "det_id", "iou"])
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(f, index=False)
    (OUT / "match_stats.json").write_text(json.dumps(stats, indent=1))
    return m, stats


def adaface_emb(clip, det_ids):
    z = np.load(ROOT / f"outputs/video/pipeline/{PIPE}/{clip}/embed.npz")
    pos = {int(d): i for i, d in enumerate(z["det_id"])}
    return z["emb_adaface"][[pos[int(d)] for d in det_ids]]


def other_emb(rec, clip, det_ids, cfg, embedder_box):
    """Embeddings of the matched detections for a recognizer that the pipeline run did not compute. Cached per clip."""
    f = OUT / "emb" / rec / f"{clip}.npz"
    if f.exists():
        z = np.load(f)
        if np.array_equal(z["det_id"], det_ids):
            return z["emb"]
    from src.video.embed import FaceEmbedder
    from src.video.io import iter_frames
    if rec not in embedder_box:
        embedder_box.clear()
        embedder_box[rec] = FaceEmbedder(cfg, (rec,))
    emb = embedder_box[rec]
    kps = np.load(ROOT / f"outputs/video/pipeline/{PIPE}/{clip}/detections.npz")["kps"]
    fi = np.load(ROOT / f"outputs/video/pipeline/{PIPE}/{clip}/detections.npz")["frame_idx"]
    by_frame = {}
    for d in det_ids:
        by_frame.setdefault(int(fi[d]), []).append(int(d))
    got = {}
    for idx, _ts, bgr in iter_frames(ROOT / "Dataset/ChokePoint" / clip, cfg["pipeline"]["fps"], None, errors=[]):
        rows = by_frame.get(idx)
        if not rows or bgr is None:
            continue
        e = emb.embed(bgr, kps[rows])["emb"][rec]
        for r, v in zip(rows, e):
            got[r] = v
    out = np.stack([got[int(d)] for d in det_ids]).astype(np.float32)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, det_id=det_ids, emb=out)
    return out


def build(rec, clips, m, cfg, box):
    """-> DataFrame-free arrays: pid[N], emb[N,512] for the matched detections of `clips`."""
    pids, embs = [], []
    for c in clips:
        s = m[m["clip"] == c].sort_values(["frame", "det_id"])
        if len(s) == 0:
            continue
        ids = s.det_id.to_numpy()
        e = adaface_emb(c, ids) if rec == "adaface" else other_emb(rec, c, ids, cfg, box)
        pids.append(s.pid.to_numpy())
        embs.append(e)
    return np.concatenate(pids), ev.l2n(np.concatenate(embs).astype(np.float32))


def score_direction(gal, probe, left_out):
    """gal/probe: (pid, emb). Returns per-identity arrays and the point metrics."""
    gp, ge = gal
    qp, qe = probe
    enrolled = sorted(set(np.unique(gp)) - left_out)  # identities in the gallery (open set)
    all_enrolled = sorted(set(np.unique(gp)))  # identities in the gallery (closed set)
    # closed set: all enrolled identities, probes of enrolled identities only
    T = np.stack([ge[gp == i].mean(0) for i in all_enrolled])
    T = ev.l2n(T)
    keep = np.isin(qp, all_enrolled)
    sims = qe[keep] @ T.T
    pred = np.array(all_enrolled)[sims.argmax(1)]
    closed = pd.DataFrame(dict(pid=qp[keep], ok=(pred == qp[keep]).astype(float)))
    # open set: left-out identities removed from the gallery; their probes, and probes of never-enrolled identities, are unmated
    g_keep = np.isin(gp, enrolled)
    mated_mask = np.isin(qp, enrolled)
    r = ev.survface_openset(ge[g_keep], gp[g_keep], qe[mated_mask], qp[mated_mask], qe[~mated_mask], device="cuda")
    ps = r["_per_sample"]
    pid_unm, pid_mat = qp[~mated_mask], qp[mated_mask]
    opn = pd.concat([pd.DataFrame(dict(pid=pid_unm, kind="s1", v=ps["s1"])), pd.DataFrame(dict(pid=pid_mat, kind="gt", v=ps["gt"]))], ignore_index=True)
    point = dict(rank1=float(closed.ok.mean()), tpir20_fpir03=float(r["TPIR20@FPIR=0.3"]), n_enrolled_closed=len(all_enrolled), n_enrolled_open=len(enrolled),
                 n_probes=int(len(qp)), n_probes_closed=int(keep.sum()), n_mated=int(mated_mask.sum()), n_unmated=int((~mated_mask).sum()),
                 dropped_probes_closed=int((~keep).sum()), dropped_ids_closed=sorted(int(i) for i in set(np.unique(qp)) - set(all_enrolled)))
    return closed, opn, point


def boot_direction(closed, opn, rng):
    """One cluster-bootstrap draw per call: identities resampled with replacement, with all their probes."""
    ids = np.array(sorted(set(closed.pid) | set(opn.pid)))
    pick = rng.integers(0, len(ids), len(ids))
    cl = closed.groupby("pid").ok.agg(["sum", "count"])
    s1 = opn[opn.kind == "s1"].groupby("pid").v.apply(np.asarray)
    gt = opn[opn.kind == "gt"].groupby("pid").v.apply(np.asarray)
    sel = ids[pick]
    c_s = sum(cl.loc[i, "sum"] for i in sel if i in cl.index)
    c_n = sum(cl.loc[i, "count"] for i in sel if i in cl.index)
    a = [s1[i] for i in sel if i in s1.index]
    b = [gt[i] for i in sel if i in gt.index]
    if not a or not b or c_n == 0:
        return None
    o = ev.survface_openset_from_scores(np.concatenate(a), np.concatenate(b))
    return c_s / c_n, o["TPIR20@FPIR=0.3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/video.yaml")
    ap.add_argument("--recognizers", nargs="+", default=["adaface"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    gt = load_gt()
    all_clips = sorted(gt)
    used = sorted({c for g in G for cond in ("SC", "MC") for c in clips_of(g, cond, all_clips)})
    print(len(used), "clips used")
    m, stats = get_matches(used, gt, cfg["pipeline"]["embed_floor"])
    ids_all = sorted(m.pid.unique().tolist())
    rng0 = np.random.default_rng(SEED)
    left_out = set(rng0.choice(ids_all, int(round(LEAVE_OUT * len(ids_all))), replace=False).tolist())
    print("identities", len(ids_all), "left out of the gallery:", sorted(left_out))
    box, res_rows, sum_rows, drop_rows = {}, [], [], []
    for rec in a.recognizers:
        for cond in ("SC", "MC"):
            draws, points = [], {}
            for gal_g, pro_g in (("G1", "G2"), ("G2", "G1")):
                gal = build(rec, clips_of(gal_g, cond, all_clips), m, cfg, box)
                pro = build(rec, clips_of(pro_g, cond, all_clips), m, cfg, box)
                closed, opn, pt = score_direction(gal, pro, left_out)
                rng = np.random.default_rng(SEED)
                d = [x for x in (boot_direction(closed, opn, rng) for _ in range(a.n_boot)) if x is not None]
                d = np.array(d)
                pt.update(recognizer=rec, condition=cond, direction=f"{gal_g}->{pro_g}", n_boot_ok=len(d),
                          rank1_ci_lo=float(np.quantile(d[:, 0], 0.025)), rank1_ci_hi=float(np.quantile(d[:, 0], 0.975)),
                          tpir_ci_lo=float(np.quantile(d[:, 1], 0.025)), tpir_ci_hi=float(np.quantile(d[:, 1], 0.975)))
                drop_rows.append(dict(recognizer=rec, condition=cond, direction=pt["direction"], dropped_ids=pt["dropped_ids_closed"], dropped_probes=pt["dropped_probes_closed"]))
                res_rows.append({k: v for k, v in pt.items() if k != "dropped_ids_closed"})
                points[pt["direction"]] = pt
                draws.append(d)
                print(rec, cond, pt["direction"], {k: round(v, 4) if isinstance(v, float) else v for k, v in pt.items() if k in ("rank1", "tpir20_fpir03", "n_probes", "n_enrolled_closed")}, flush=True)
            n = min(len(draws[0]), len(draws[1]))
            avg = (draws[0][:n] + draws[1][:n]) / 2
            sum_rows.append(dict(recognizer=rec, condition=cond, rank1=np.mean([p["rank1"] for p in points.values()]), rank1_ci_lo=float(np.quantile(avg[:, 0], 0.025)),
                                 rank1_ci_hi=float(np.quantile(avg[:, 0], 0.975)), tpir20_fpir03=np.mean([p["tpir20_fpir03"] for p in points.values()]),
                                 tpir_ci_lo=float(np.quantile(avg[:, 1], 0.025)), tpir_ci_hi=float(np.quantile(avg[:, 1], 0.975))))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res_rows).to_csv(OUT / "results.csv", index=False, float_format="%.5f")
    pd.DataFrame(sum_rows).to_csv(OUT / "summary.csv", index=False, float_format="%.5f")
    pd.DataFrame(drop_rows).to_csv(OUT / "dropped.csv", index=False)
    tot = {k: sum(v[k] for v in stats.values()) for k in ("n_det", "n_fp", "n_gt", "n_matched")}
    json.dump(dict(totals=tot, left_out_identities=sorted(left_out), n_identities=len(ids_all)), open(OUT / "baseline_meta.json", "w"), indent=1)
    print(pd.read_csv(OUT / "summary.csv").round(4).to_string())


if __name__ == "__main__":
    main()
