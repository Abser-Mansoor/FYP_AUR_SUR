"""Phase 7 YouTube Faces preparation (REPORT_phase7.md section 0.5): parse the official splits and the labelled boxes, choose the frames, check the counts.

    envs/video/Scripts/python.exe scripts/video_ytf_prepare.py            # writes outputs/video/ytf/{frames.csv, pairs.csv, counts.json}
    envs/video/Scripts/python.exe scripts/video_ytf_prepare.py --overlay 10   # 10 seeded overlays of the box convention -> outputs/video/ytf/overlays/ (local only)

Data (Dataset/YouTubeFaces, from the official archives): meta_data/meta_and_splits.mat (video_names 'Person/ID', video_labels, Splits [500, 3, 10]: 1-based video index 1,
video index 2, same-person flag) and frame_images_DB/<Person>/<ID>/<ID>.<frame>.jpg with frame_images_DB/<Person>.labeled_faces.txt (lines
`Person\\ID\\ID.<frame>.jpg,0,cx,cy,w,h,0.0,1`: the box is centre x, centre y, width, height).
Frame choice (pre-registered): up to 50 evenly spaced frames among the video's labelled frames (np.linspace over the frame-number-sorted list, rounded, unique).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Dataset/YouTubeFaces"
OUT = ROOT / "outputs/video/ytf"
MAX_FRAMES = 50


def load_meta():
    m = sio.loadmat(DATA / "meta_data/meta_and_splits.mat")
    names = [str(x[0][0]) if len(x[0]) else "" for x in m["video_names"]]
    return names, m["video_labels"][0].astype(int), m["Splits"].astype(int)


def read_boxes(person):
    """{video id: [(frame number, relative path, cx, cy, w, h)]} from the person's labeled_faces.txt."""
    f = DATA / "frame_images_DB" / f"{person}.labeled_faces.txt"
    out = {}
    if not f.exists():
        return out
    for ln in f.read_text().splitlines():
        p = ln.strip().split(",")
        if len(p) < 6:
            continue
        rel = p[0].replace("\\", "/")
        parts = rel.split("/")
        mm = re.match(r"(.+)\.(\d+)\.jpg$", parts[-1])
        out.setdefault(parts[1], []).append((int(mm.group(2)), rel, float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    return out


def prepare():
    names, labels, splits = load_meta()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_person = [], {}
    n_missing_label = 0
    for vi, name in enumerate(names, 1):
        person, vid = name.split("/")
        if person not in per_person:
            per_person[person] = read_boxes(person)
        fr = sorted(per_person[person].get(vid, []))
        if not fr:
            n_missing_label += 1
            continue
        pick = sorted(set(np.linspace(0, len(fr) - 1, min(len(fr), MAX_FRAMES)).round().astype(int).tolist()))
        for k in pick:
            n, rel, cx, cy, w, h = fr[k]
            rows.append(dict(video_idx=vi, video=name, person=person, label=int(labels[vi - 1]), frame_no=n, path=rel, n_labelled=len(fr), cx=cx, cy=cy, w=w, h=h,
                             x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "frames.csv", index=False, float_format="%.2f")
    pairs = [dict(fold=f + 1, v1=int(splits[i, 0, f]), v2=int(splits[i, 1, f]), same=int(splits[i, 2, f])) for f in range(splits.shape[2]) for i in range(splits.shape[0])]
    pd.DataFrame(pairs).to_csv(OUT / "pairs.csv", index=False)
    p = pd.DataFrame(pairs)
    exists = df.path.map(lambda r: (DATA / "frame_images_DB" / r).exists())
    counts = dict(n_videos_in_meta=len(names), n_people_in_meta=int(len(set(labels.tolist()))), n_videos_with_labels=int(df.video_idx.nunique()), n_videos_without_labels=int(n_missing_label),
                  n_people_with_labels=int(df.person.nunique()), n_pairs=int(len(p)), n_folds=int(p.fold.nunique()), same_per_fold=p.groupby("fold").same.sum().tolist(),
                  pairs_per_fold=p.groupby("fold").size().tolist(), frames_selected=int(len(df)), videos_below_50_labelled_frames=int((df.groupby("video_idx").n_labelled.first() < MAX_FRAMES).sum()),
                  selected_frames_missing_on_disk=int((~exists).sum()),
                  pairs_with_a_video_without_frames=int((~p.v1.isin(df.video_idx) | ~p.v2.isin(df.video_idx)).sum()))
    (OUT / "counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts, indent=1))


def overlay(n):
    import cv2
    df = pd.read_csv(OUT / "frames.csv")
    rng = np.random.default_rng(42)
    d = OUT / "overlays"
    d.mkdir(exist_ok=True)
    tiles = []
    for k in rng.choice(len(df), n, replace=False):
        r = df.iloc[int(k)]
        im = cv2.imread(str(DATA / "frame_images_DB" / r.path))
        cv2.rectangle(im, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 255, 0), 2)
        tiles.append(cv2.resize(im, (320, 240)))
    cols = 5
    cv2.imwrite(str(d / "box_convention.jpg"), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
    print("wrote", d / "box_convention.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", type=int, default=0)
    a = ap.parse_args()
    overlay(a.overlay) if a.overlay else prepare()
