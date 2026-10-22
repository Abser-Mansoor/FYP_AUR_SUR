"""ChokePoint: fit the eye-to-face-box rule and build the 1,000-frame detector-study sample.

    envs/video/Scripts/python.exe scripts/video_chokepoint_sample.py

Needs the extracted frames (Dataset/ChokePoint/<clip>/<8-digit frame>.jpg) and outputs/video/gt/chokepoint_eyes.csv
(scripts/video_chokepoint_gt.py). Steps, all seeded (42):
 1. Eye-to-box rule, fitted from data, not guessed: SCRFD (score >= 0.9) on 300 random labelled frames; a detection is paired with the
    ground-truth person whose eye midpoint lies inside it (one-to-one); the box centre offset and the box width/height are
    expressed in units of the inter-eye distance d; the rule is the median of each. Written to outputs/video/gt/eye_box_rule.json.
 2. Twenty overlay frames in outputs/video/gt/overlays/ for a visual check.
 3. Sample: 1,000 labelled frames, stratified over clips (equal quota per clip, remainder drawn at random), copied to
    outputs/video/chokepoint_sample/ with index.csv (name, file, clip, frame) and gt_boxes.csv (name, person_id, x1,y1,x2,y2).
Frames with no labelled person are never sampled.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sr import common as C  # noqa: E402
from src.video import detect as D  # noqa: E402

DATA = ROOT / "Dataset/ChokePoint"
GT = ROOT / "outputs/video/gt"
SAMPLE = ROOT / "outputs/video/chokepoint_sample"
SEED, N_FIT, N_SAMPLE, N_OVERLAY = 42, 300, 1000, 20


def frame_path(clip, frame):
    return DATA / clip / f"{int(frame):08d}.jpg"


def box_from_eyes(row, rule):
    d = row["d"]
    cx, cy = (row["lx"] + row["rx"]) / 2 + rule["dx"] * d, (row["ly"] + row["ry"]) / 2 + rule["dy"] * d
    w, h = rule["w"] * d, rule["h"] * d
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def fit_rule(eyes, det):
    rng = np.random.default_rng(SEED)
    frames = eyes.groupby(["clip", "frame"]).size().reset_index()[["clip", "frame"]]
    frames = frames[[frame_path(c, f).exists() for c, f in zip(frames["clip"], frames.frame)]]
    pick = frames.iloc[rng.permutation(len(frames))[:N_FIT]]
    dx, dy, w, h, n_pair = [], [], [], [], 0
    for c, f in zip(pick["clip"], pick.frame):
        sub = eyes[(eyes["clip"] == c) & (eyes.frame == f)]
        dets, _ = det.detect(C.read_bgr(frame_path(c, f)), 640, 0.9)
        used = set()
        for x1, y1, x2, y2, _s in dets:
            inside = [(i, r) for i, r in sub.iterrows() if i not in used and x1 <= (r.lx + r.rx) / 2 <= x2 and y1 <= (r.ly + r.ry) / 2 <= y2]
            if len(inside) != 1:
                continue
            i, r = inside[0]
            used.add(i)
            d = r.d
            dx.append(((x1 + x2) / 2 - (r.lx + r.rx) / 2) / d)
            dy.append(((y1 + y2) / 2 - (r.ly + r.ry) / 2) / d)
            w.append((x2 - x1) / d)
            h.append((y2 - y1) / d)
            n_pair += 1
    rule = dict(dx=float(np.median(dx)), dy=float(np.median(dy)), w=float(np.median(w)), h=float(np.median(h)), n_pairs=n_pair,
                iqr={k: float(np.subtract(*np.percentile(v, [75, 25]))) for k, v in dict(dx=dx, dy=dy, w=w, h=h).items()},
                note="units of the inter-eye distance d; box = eye midpoint + (dx, dy)*d, size (w, h)*d; fitted on SCRFD boxes (score >= 0.9)")
    return rule


def main():
    eyes = pd.read_csv(GT / "chokepoint_eyes.csv")
    det = D.SCRFDDetector(ROOT / "models/detector/det_10g.onnx")
    avail = {c for c in eyes["clip"].unique() if (DATA / c).is_dir()}
    print(f"{len(avail)} of {eyes["clip"].nunique()} labelled clips have extracted frames", flush=True)
    eyes = eyes[eyes["clip"].isin(avail)].reset_index(drop=True)
    rule = fit_rule(eyes, det)
    (GT / "eye_box_rule.json").write_text(json.dumps(rule, indent=1))
    print(json.dumps(rule, indent=1), flush=True)

    # overlays
    rng = np.random.default_rng(SEED + 1)
    (GT / "overlays").mkdir(exist_ok=True)
    fr = eyes.groupby(["clip", "frame"]).size().reset_index()[["clip", "frame"]]
    for k, i in enumerate(rng.permutation(len(fr))[:N_OVERLAY]):
        c, f = fr.iloc[i]
        im = C.read_bgr(frame_path(c, f))
        for _, r in eyes[(eyes["clip"] == c) & (eyes.frame == f)].iterrows():
            x1, y1, x2, y2 = map(int, box_from_eyes(r, rule))
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(im, (int(r.lx), int(r.ly)), 2, (0, 0, 255), -1)
            cv2.circle(im, (int(r.rx), int(r.ry)), 2, (0, 0, 255), -1)
        cv2.imwrite(str(GT / "overlays" / f"{k:02d}_{c}_{f:08d}.jpg"), im)

    # stratified sample: equal quota per clip, remainder at random
    rng = np.random.default_rng(SEED)
    clips = sorted(avail)
    quota = N_SAMPLE // len(clips)
    chosen = []
    pools = {c: fr[fr["clip"] == c].reset_index(drop=True) for c in clips}
    for c in clips:
        p = pools[c]
        take = p.iloc[rng.permutation(len(p))[:quota]]
        chosen += list(zip(take["clip"], take.frame))
    rest = [(c, f) for c in clips for f in pools[c].frame if (c, f) not in set(chosen)]
    for i in rng.permutation(len(rest))[:N_SAMPLE - len(chosen)]:
        chosen.append(rest[i])
    chosen = sorted(chosen)
    if SAMPLE.exists():
        shutil.rmtree(SAMPLE)
    (SAMPLE / "frames").mkdir(parents=True)
    idx, gtb = [], []
    for c, f in chosen:
        name = f"{c}__{int(f):08d}"
        shutil.copyfile(frame_path(c, f), SAMPLE / "frames" / f"{name}.jpg")
        idx.append(dict(name=name, file=f"frames/{name}.jpg", clip=c, frame=int(f)))
        for _, r in eyes[(eyes["clip"] == c) & (eyes.frame == f)].iterrows():
            gtb.append(dict(name=name, person_id=r.person_id, **dict(zip(["x1", "y1", "x2", "y2"], box_from_eyes(r, rule)))))
    pd.DataFrame(idx).to_csv(SAMPLE / "index.csv", index=False)
    pd.DataFrame(gtb).to_csv(SAMPLE / "gt_boxes.csv", index=False)
    print(f"sample: {len(idx)} frames, {len(gtb)} ground-truth boxes, {len(clips)} clips -> {SAMPLE}")


if __name__ == "__main__":
    main()
