"""Study C: audit of unmatched detections. Protocol (read it before labelling): research/Sprint_Notes/Phase6_studyC_audit_protocol.md

Lab PC (once):
    envs/video/Scripts/python.exe scripts/video_studyC_audit.py sample      # seeded sample of 350 -> outputs/video/studyC/sheet.csv (has scores: not for labellers)
    envs/video/Scripts/python.exe scripts/video_studyC_audit.py pack        # pre-rendered images + blind sheet -> outputs/video/studyC/pack/ (what the labellers get)
Labellers (any PC with Python, opencv-python, numpy, pandas; needs only the pack folder and this script):
    python video_studyC_audit.py label --who abser --pack <pack folder> [--stage shared|own]
    keys: a o r n u = label, b = back one crop, q = save and quit. Labels are saved to <pack>/labels_<who>.csv after every key; send that small file back.
Lab PC again (labels copied to outputs/video/studyC/):
    envs/video/Scripts/python.exe scripts/video_studyC_audit.py agreement   # Fleiss kappa on the 50 shared crops
    envs/video/Scripts/python.exe scripts/video_studyC_audit.py analyze     # final labels and composition with CIs

Labels: a live face / annotated person, o live face / other person, r reflection or image of a face, n not a face, u unclear.
The images show no score, track or recogniser output, and the pack sheet has only sample_id and assignment.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "outputs/video/studyC"
PIPE = ROOT / "outputs/video/pipeline/chokepoint"
DATA = ROOT / "Dataset/ChokePoint"
WHO = ["abser", "raghib", "umar"]
LABELS = {"a": "live face, annotated person", "o": "live face, other person", "r": "reflection or image of a face", "n": "not a face", "u": "unclear"}
N_SAMPLE, N_SHARED, SEED, REF_WINDOW = 350, 50, 42, 30


def sample(a):
    from video_tracker_eval import load_gt
    gt = load_gt()
    m = pd.read_csv(ROOT / "outputs/video/baseline/matches.csv")
    parts = []
    for clip in sorted(gt):
        t = pd.read_csv(PIPE / clip / "tracks.csv")
        t = t[t.score >= 0.5]
        u = t[~t.det_id.isin(set(m.loc[m["clip"] == clip, "det_id"]))].copy()
        u.insert(0, "clip", clip)
        u["unit"] = np.where(u.tid_bytetrack >= 0, clip + "#" + u.tid_bytetrack.astype(str), clip + "@" + u.det_id.astype(str))
        parts.append(u)
    U = pd.concat(parts, ignore_index=True)
    size = U.groupby("unit").size()
    rng = np.random.default_rng(SEED)
    units = sorted(size.index)
    pick = rng.choice(units, N_SAMPLE, replace=False)
    by = {k: g for k, g in U.groupby("unit")}
    rows = []
    for un in pick:
        g = by[un]
        r = g.iloc[int(rng.integers(0, len(g)))]
        rows.append(dict(clip=r["clip"], frame=int(r.frame), det_id=int(r.det_id), x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, score=r.score, unit=un, weight=int(size[un])))
    S = pd.DataFrame(rows)
    order = rng.permutation(len(S))
    S = S.iloc[order].reset_index(drop=True)
    S.insert(0, "sample_id", np.arange(len(S)))
    assigned = ["all"] * N_SHARED
    own = (len(S) - N_SHARED) // len(WHO)
    for w in WHO:
        assigned += [w] * own
    S["assigned"] = assigned
    OUT.mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT / "sheet.csv", index=False, float_format="%.3f")
    meta = dict(n_unmatched_total=int(len(U)), n_units=int(len(size)), n_sampled=len(S), n_shared=N_SHARED, per_labeller=own, seed=SEED)
    (OUT / "sheet_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


def crop(im, box, pad, side):
    h, w = im.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * (1 + pad)
    a, b, c, d = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0)), int(min(cx + s / 2, w)), int(min(cy + s / 2, h))
    if c <= a or d <= b:
        return np.zeros((side, side, 3), np.uint8)
    return cv2.resize(im[b:d, a:c], (side, side), interpolation=cv2.INTER_CUBIC)


def refs(gt, clip, frame, n=2):
    """Up to n annotated people from the annotated frame nearest to `frame` (within REF_WINDOW)."""
    per = gt[clip]
    for d in range(REF_WINDOW + 1):
        for f in (frame - d, frame + d):
            if f in per:
                ids, boxes = per[f]
                return f, [(int(i), b) for i, b in list(zip(ids, boxes))[:n]]
    return None, []


def composite(r, gt):
    """Image for one sample: face crop (left), full frame with the box (middle), reference crops of annotated people (right). No text about the detection."""
    im = cv2.imread(str(DATA / r.clip / f"{int(r.frame):08d}.jpg"))
    box = (r.x1, r.y1, r.x2, r.y2)
    full = im.copy()
    cv2.rectangle(full, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 0, 255), 2)
    left = crop(im, box, 0.4, 320)
    full = cv2.resize(full, (640, 480))
    f, rl = refs(gt, r.clip, int(r.frame))
    right = np.full((480, 220, 3), 40, np.uint8)
    for i, (pid, b) in enumerate(rl):
        fr = cv2.imread(str(DATA / r.clip / f"{f:08d}.jpg"))
        right[i * 230:i * 230 + 200, 10:210] = crop(fr, b, 0.4, 200)
        cv2.putText(right, f"annotated person {chr(65 + i)}", (10, i * 230 + 218), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if not rl:
        cv2.putText(right, "no annotated person", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(right, "within 30 frames", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    col = np.full((480, 320, 3), 40, np.uint8)
    col[:320] = left
    return np.hstack([col, full, right])


def pack(a):
    """Pre-render every sampled crop to a JPEG and write the blind sheet. Nothing but these files goes to the labellers."""
    import hashlib
    from video_tracker_eval import load_gt
    gt = load_gt()
    S = pd.read_csv(OUT / "sheet.csv")
    d = OUT / "pack"
    (d / "img").mkdir(parents=True, exist_ok=True)
    for r in S.itertuples():
        cv2.imwrite(str(d / "img" / f"{r.sample_id:03d}.jpg"), composite(r, gt), [cv2.IMWRITE_JPEG_QUALITY, 92])
    S[["sample_id", "assigned"]].to_csv(d / "sheet_blind.csv", index=False)
    (d / "SAMPLE_SHA256.txt").write_text(hashlib.sha256((OUT / "sheet.csv").read_bytes()).hexdigest() + "  sheet.csv (the full sample list, kept on the lab PC)\n")
    print(len(S), "images in", d, "- size MB:", round(sum(p.stat().st_size for p in (d / "img").iterdir()) / 1e6, 1))


def read_labels(who):
    f = OUT / f"labels_{who}.csv"
    return pd.read_csv(f, dtype={"label": str}) if f.exists() else pd.DataFrame(columns=["sample_id", "label"])


def label(a):
    d = Path(a.pack) if a.pack else OUT / "pack"
    S = pd.read_csv(d / "sheet_blind.csv")
    pick = ["all"] if a.stage == "shared" else [a.who]
    mine = S[S.assigned.isin(pick)].sample_id.tolist()
    f = d / f"labels_{a.who}.csv"
    rows = list(pd.read_csv(f, dtype={"label": str}).itertuples(index=False, name=None)) if f.exists() else []
    save = lambda: pd.DataFrame(rows, columns=["sample_id", "label"]).to_csv(f, index=False)  # noqa: E731
    got = lambda: {x[0] for x in rows}  # noqa: E731
    print(f"{a.who}, stage {a.stage}: {len([m for m in mine if m in got()])} of {len(mine)} done")
    while True:
        todo = [m for m in mine if m not in got()]
        if not todo:
            break
        k = len([m for m in mine if m in got()]) + 1
        im = cv2.imread(str(d / "img" / f"{todo[0]:03d}.jpg"))
        for i, line in enumerate([f"{a.who}  {a.stage} {k}/{len(mine)}", "a live face, annotated person", "o live face, other person", "r reflection / image of a face", "n not a face", "u unclear",
                                  "b back   q save and quit"]):
            cv2.putText(im, line, (6, 340 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow("studyC", im)
        key = chr(cv2.waitKey(0) & 0xFF)
        if key == "q":
            break
        if key == "b" and rows and rows[-1][0] in mine:
            rows.pop()
        elif key in LABELS:
            rows.append((todo[0], key))
        save()
    save()
    cv2.destroyAllWindows()
    print("saved", len(rows), "labels to", f)


def fleiss(M):
    """M: [N items, k categories] counts, constant raters per item."""
    N, k = M.shape
    n = M.sum(1)[0]
    p = M.sum(0) / (N * n)
    Pi = ((M ** 2).sum(1) - n) / (n * (n - 1))
    Pbar, Pe = Pi.mean(), (p ** 2).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else 1.0


def agreement(a=None):
    S = pd.read_csv(OUT / "sheet.csv")
    sh = S[S.assigned == "all"].sample_id.tolist()
    L = {w: read_labels(w).set_index("sample_id").label for w in WHO}
    missing = {w: [s for s in sh if s not in L[w].index] for w in WHO}
    if any(missing.values()):
        sys.exit("shared crops not yet labelled: " + json.dumps({w: len(v) for w, v in missing.items()}))
    cats = list(LABELS)
    M = np.array([[sum(L[w][s] == c for w in WHO) for c in cats] for s in sh])
    pair = {f"{x}-{y}": float(np.mean([L[x][s] == L[y][s] for s in sh])) for i, x in enumerate(WHO) for y in WHO[i + 1:]}
    res = dict(n_shared=len(sh), all_three_agree=float(np.mean(M.max(1) == 3)), pairwise_agreement=pair, fleiss_kappa=fleiss(M), kappa_below_threshold_0_60=bool(fleiss(M) < 0.60),
               label_counts={c: int(M[:, i].sum()) for i, c in enumerate(cats)})
    (OUT / "agreement.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def analyze(a=None):
    S = pd.read_csv(OUT / "sheet.csv")
    meta = json.loads((OUT / "sheet_meta.json").read_text())
    L = {w: read_labels(w).set_index("sample_id").label for w in WHO}
    final = []
    for r in S.itertuples():
        if r.assigned == "all":
            v = [L[w].get(r.sample_id) for w in WHO]
            if any(x is None for x in v):
                sys.exit("not all labels are in yet")
            top = max(set(v), key=v.count)
            lab = top if v.count(top) >= 2 else "u"
        else:
            lab = L[r.assigned].get(r.sample_id)
            if lab is None:
                sys.exit("not all labels are in yet")
        final.append(lab)
    S["final"] = final
    S["band"] = pd.cut(S.score, [0.5, 0.7, 0.85, 1.01], right=False, labels=["0.5-0.7", "0.7-0.85", ">=0.85"]).astype(str)
    S["group"] = S["clip"].str[:3]
    S.to_csv(OUT / "final_labels.csv", index=False, float_format="%.3f")
    total = meta["n_unmatched_total"]

    def share(df, classes):
        w = df.weight.to_numpy(float)
        return float(w[df.final.isin(classes).to_numpy()].sum() / w.sum()) if len(df) else np.nan

    def boot(df, classes, n=1000):
        rng = np.random.default_rng(SEED)
        clips = df["clip"].unique()
        parts = {c: g for c, g in df.groupby("clip")}
        d = []
        for _ in range(n):
            pick = rng.integers(0, len(clips), len(clips))
            d.append(share(pd.concat([parts[clips[j]] for j in pick]), classes))
        return float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))

    defs = {"not a live face (r+n)": ["r", "n"], "reflection or image (r)": ["r"], "not a face (n)": ["n"], "live face, annotated person (a)": ["a"], "live face, other person (o)": ["o"],
            "any live face (a+o)": ["a", "o"], "unclear (u)": ["u"], "sensitivity: u counted as not a live face": ["r", "n", "u"], "sensitivity: u counted as live face": ["a", "o", "u"]}
    rows = []
    for sname, sub in [("all", S)] + [(f"group {g}", S[S.group == g]) for g in sorted(S.group.unique())] + [(f"band {b}", S[S.band == b]) for b in ["0.5-0.7", "0.7-0.85", ">=0.85"]]:
        for name, cl in defs.items():
            p = share(sub, cl)
            lo, hi = boot(sub, cl) if len(sub) else (np.nan, np.nan)
            rows.append(dict(stratum=sname, outcome=name, n_sampled=len(sub), share=p, ci_lo=lo, ci_hi=hi, est_count_in_stratum_total=p * total if sname == "all" else np.nan))
    pd.DataFrame(rows).to_csv(OUT / "composition.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows).query("stratum == 'all'")[["outcome", "n_sampled", "share", "ci_lo", "ci_hi", "est_count_in_stratum_total"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    p = sub.add_parser("label")
    p.add_argument("--who", required=True, choices=WHO)
    p.add_argument("--pack", default=None, help="the pack folder (default: outputs/video/studyC/pack)")
    p.add_argument("--stage", default="shared", choices=["shared", "own"], help="shared = the 50 crops all three label first; own = your 100")
    sub.add_parser("pack")
    sub.add_parser("agreement")
    sub.add_parser("analyze")
    a = ap.parse_args()
    {"sample": sample, "pack": pack, "label": label, "agreement": agreement, "analyze": analyze}[a.cmd](a)
