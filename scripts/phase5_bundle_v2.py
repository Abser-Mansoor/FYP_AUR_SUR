"""Phase 5 re-run (v2): build the code bundle with the official SCRFD preprocessing and publish it next to, not over, the first bundle.

    envs/video/Scripts/python.exe scripts/phase5_bundle_v2.py [--publish]

The large zips (detectors, WIDER val, ChokePoint sample, eval crops, xcheck) are unchanged and stay in G:/My Drive/FYP_SR_SHARED/phase5/.
Only the code changed, so this writes a new folder G:/My Drive/FYP_SR_SHARED/phase5_v2/ with code_v2.zip, MANIFEST.json and the v2 notebook.
Nothing in phase5/ is touched. The v2 notebook reads the old zips (except code.zip) and code_v2.zip, and writes to phase5_v2/results/.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_bundle as B  # noqa: E402

OUT = ROOT / "outputs/video/phase5_bundle_v2"
DEST = B.SHARED / "phase5_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    B.zip_files(OUT / "code_v2.zip", B.tree("src", None) + [(ROOT / "scripts/video_detector_study.py", "scripts/video_detector_study.py"),
                                                           (ROOT / "configs/video.yaml", "configs/video.yaml"),
                                                           (ROOT / "outputs/sr/subset/eval_images.csv", "outputs/sr/subset/eval_images.csv")])
    man = {p.name: {"bytes": p.stat().st_size, "sha256": B.sha256(p)} for p in sorted(OUT.glob("*.zip"))}
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    if a.publish:
        if DEST.exists() and any(DEST.glob("code_v2.zip")):
            sys.exit(f"{DEST} already holds code_v2.zip: choose a new folder name instead of overwriting")
        DEST.mkdir(parents=True, exist_ok=True)
        for p in (OUT / "code_v2.zip", OUT / "MANIFEST.json"):  # MANIFEST last
            shutil.copyfile(p, DEST / p.name)
            print("published", p.name, flush=True)
    print("done")


if __name__ == "__main__":
    main()
