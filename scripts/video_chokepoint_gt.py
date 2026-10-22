"""Parse the ChokePoint eye ground truth into one CSV and a counts summary.

    envs/diag/Scripts/python.exe scripts/video_chokepoint_gt.py

Reads Dataset/ChokePoint/archives/groundtruth.tar.xz (the archive is read-only and untouched). One row per (clip, frame, person):
clip (e.g. P1E_S1_C1 or P2E_S1_C1.1), frame, person_id, lx, ly, rx, ry. Counts are reported as found and compared with the
dataset page's figures (64,204 face crops, "48 sequences") without adjusting either.
"""
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Dataset/ChokePoint/archives/groundtruth.tar.xz"
OUT = ROOT / "outputs/video/gt"
PAGE = {"face_crops": 64204, "sequences": 48}  # arma.sourceforge.net/chokepoint, per Sprint_2_video_notes.md (to be re-checked)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, clips, n_frames, errors = [], [], 0, []
    with tarfile.open(ARCHIVE, "r:xz") as t:
        for m in sorted((m for m in t.getmembers() if m.name.endswith(".xml")), key=lambda m: m.name):
            clip = Path(m.name).stem
            clips.append(clip)
            try:
                root = ET.fromstring(t.extractfile(m).read())
            except ET.ParseError as e:
                errors.append({"clip": clip, "error": str(e)})
                continue
            for fr in root.iter("frame"):
                n_frames += 1
                for p in fr.iter("person"):
                    le, re_ = p.find("leftEye"), p.find("rightEye")
                    if le is None or re_ is None:
                        errors.append({"clip": clip, "frame": fr.get("number"), "person": p.get("id"), "error": "missing eye"})
                        continue
                    rows.append((clip, int(fr.get("number")), p.get("id"), float(le.get("x")), float(le.get("y")),
                                 float(re_.get("x")), float(re_.get("y"))))
    df = pd.DataFrame(rows, columns=["clip", "frame", "person_id", "lx", "ly", "rx", "ry"])
    df["d"] = ((df.rx - df.lx) ** 2 + (df.ry - df.ly) ** 2) ** 0.5  # inter-eye distance in px
    df.to_csv(OUT / "chokepoint_eyes.csv", index=False)
    # sequence = clip without the camera suffix (P1E_S1_C1 -> P1E_S1; P2E_S1_C1.1 -> P2E_S1 + recording .1)
    seq = lambda c: re.sub(r"_C\d+(\.\d+)?$", "", c)  # noqa: E731
    summary = dict(
        n_clips=len(clips), n_sequences_without_camera=len({seq(c) for c in clips}), n_frames=n_frames, n_person_entries=len(df),
        n_persons=df.person_id.nunique(), persons=sorted(df.person_id.unique().tolist()), n_parse_errors=len(errors), errors=errors[:20],
        inter_eye_px=df.d.describe().round(2).to_dict(), page_figures=PAGE,
        match=dict(face_crops=len(df) == PAGE["face_crops"], sequences=len(clips) == PAGE["sequences"]),
        per_prefix_entries=df.assign(p=df["clip"].str[:3]).groupby("p").size().to_dict())
    (OUT / "chokepoint_gt_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "persons"}, indent=1))


if __name__ == "__main__":
    main()
