"""Build the Phase 5 Colab bundle and publish it to the shared Drive folder.

    envs/video/Scripts/python.exe scripts/phase5_bundle.py [--publish]

Writes (outputs/video/phase5_bundle/, copied to G:/My Drive/FYP_SR_SHARED/phase5/ with --publish):
  code.zip              src/ (without models), scripts/video_detector_study.py, configs/video.yaml, the eval subset CSV
  detectors.zip         models/detector/det_10g.onnx, models/detector/retinaface/Resnet50_Final.pth
  wider_val.zip         WIDER_val images, wider_face_split, eval_tools/ground_truth (the official evaluation files)
  chokepoint_sample.zip index.csv, gt_boxes.csv, 1,000 frames   (run scripts/video_chokepoint_sample.py first)
  eval_crops.zip        the 22,898 TinyFace / SurvFace eval crops (Phase 3 subset)
  xcheck.zip            xcheck/crops (100 seeded TinyFace probes) + xcheck/lab_kps.npy written by THIS machine
  weights_sha256.json   sha256 of the two detector weights
  MANIFEST.json         size + sha256 of every zip (the notebook refuses anything that has not finished syncing)
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/video/phase5_bundle"
SHARED = Path("G:/My Drive/FYP_SR_SHARED")
WEIGHTS = ["models/detector/det_10g.onnx", "models/detector/retinaface/Resnet50_Final.pth"]


def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def zip_files(zpath, items):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:  # stored: JPEG / PNG / weights do not compress
        for p, arc in items:
            z.write(p, arc)


def tree(base, rel, suffixes=None, skip=("__pycache__",)):
    return [(p, p.relative_to(ROOT).as_posix()) for p in sorted((ROOT / base).rglob("*"))
            if p.is_file() and not any(s in p.parts for s in skip) and p.suffix != ".pyc" and (suffixes is None or p.suffix in suffixes)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    sample = ROOT / "outputs/video/chokepoint_sample"
    if not (sample / "index.csv").exists():
        sys.exit("run scripts/video_chokepoint_sample.py first (the ChokePoint sample does not exist yet)")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    zip_files(OUT / "code.zip", tree("src", None) + [(ROOT / "scripts/video_detector_study.py", "scripts/video_detector_study.py"),
                                                    (ROOT / "configs/video.yaml", "configs/video.yaml"),
                                                    (ROOT / "outputs/sr/subset/eval_images.csv", "outputs/sr/subset/eval_images.csv")])
    zip_files(OUT / "detectors.zip", [(ROOT / w, w) for w in WEIGHTS])
    (OUT / "weights_sha256.json").write_text(json.dumps({w: sha256(ROOT / w) for w in WEIGHTS}, indent=1))
    wider = ROOT / "Dataset/WIDER_FACE"
    items = [(p, p.relative_to(ROOT).as_posix()) for p in sorted((wider / "WIDER_val/images").rglob("*.jpg"))]
    items += [(p, p.relative_to(ROOT).as_posix()) for sub in ("wider_face_split", "eval_tools/ground_truth") for p in sorted((wider / sub).rglob("*")) if p.is_file()]
    zip_files(OUT / "wider_val.zip", items)
    zip_files(OUT / "chokepoint_sample.zip", [(p, p.relative_to(ROOT).as_posix()) for p in sorted(sample.rglob("*")) if p.is_file()])
    eval_dir = ROOT / "outputs/sr/subset/eval_input"
    zip_files(OUT / "eval_crops.zip", [(eval_dir / f, f"outputs/sr/subset/eval_input/{f}") for f in pd.read_csv(ROOT / "outputs/sr/subset/eval_images.csv")["file"]])

    # cross-device check data: 100 seeded TinyFace probes + this machine's SCRFD keypoints
    xd = OUT / "_xcheck"
    (xd / "crops").mkdir(parents=True)
    img = pd.read_csv(ROOT / "outputs/sr/subset/eval_images.csv")
    for f in img[(img.dataset == "tinyface") & (img.split == "probe")].sample(100, random_state=42)["file"]:
        shutil.copyfile(eval_dir / f, xd / "crops" / f)
    r = subprocess.run([str(ROOT / "envs/video/Scripts/python.exe"), str(ROOT / "scripts/video_detector_study.py"), "--job", "xcheck", "--xcheck-dir", str(xd)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 or not (xd / "lab_kps.npy").exists():
        sys.exit("lab keypoints failed:\n" + (r.stdout or "")[-1500:] + (r.stderr or "")[-1500:])
    zip_files(OUT / "xcheck.zip", [(p, f"xcheck/{p.relative_to(xd).as_posix()}") for p in sorted(xd.rglob("*")) if p.is_file()])
    shutil.rmtree(xd)

    man = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(OUT.glob("*.zip"))}
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    if a.publish:
        dest = SHARED / "phase5"
        dest.mkdir(parents=True, exist_ok=True)
        for p in [*sorted(OUT.glob("*.zip")), OUT / "weights_sha256.json", OUT / "MANIFEST.json"]:  # MANIFEST last
            shutil.copyfile(p, dest / p.name)
            print("published", p.name, flush=True)
    print("done")


if __name__ == "__main__":
    main()
