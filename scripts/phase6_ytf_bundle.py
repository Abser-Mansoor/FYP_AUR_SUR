"""Bundle for the Appendix T1 Colab run (the Phase 5 detector on the YouTube Faces sample).

    envs/video/Scripts/python.exe scripts/phase6_ytf_bundle.py [--publish]

Writes outputs/video/phase6_ytf_bundle/: code_ytf.zip (src/, configs/video.yaml, scripts/video_appendix_ytf.py), ytf_sample.zip (the 5,000 sampled frames and gt_boxes.csv;
run `scripts/video_appendix_ytf.py sample` first) and MANIFEST.json (size and SHA-256 of each zip). The detector weights are NOT bundled again: the notebook reads
phase5/detectors.zip and checks it against phase5/MANIFEST.json and phase5/weights_sha256.json, which are already on the shared Drive.
--publish copies the zips, the manifest and the notebook to H:/My Drive/FYP_SR_SHARED/phase6_ytf/, a NEW folder (it refuses if the folder already holds a MANIFEST.json,
and it never overwrites or deletes anything).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_bundle as B  # noqa: E402

OUT = ROOT / "outputs/video/phase6_ytf_bundle"
SAMPLE = ROOT / "outputs/video/appendix_data/ytf_sample"
DEST = Path("H:/My Drive/FYP_SR_SHARED/phase6_ytf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    if not (SAMPLE / "gt_boxes.csv").exists():
        sys.exit("run scripts/video_appendix_ytf.py sample first")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    B.zip_files(OUT / "code_ytf.zip", B.tree("src", None) + [(ROOT / "scripts/video_appendix_ytf.py", "scripts/video_appendix_ytf.py"), (ROOT / "configs/video.yaml", "configs/video.yaml")])
    items = [(p, "ytf_sample/" + p.relative_to(SAMPLE).as_posix()) for p in sorted(SAMPLE.rglob("*")) if p.is_file()]
    B.zip_files(OUT / "ytf_sample.zip", items)
    man = {p.name: {"bytes": p.stat().st_size, "sha256": B.sha256(p)} for p in sorted(OUT.glob("*.zip"))}
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    print("total MB: %.0f" % (sum(m["bytes"] for m in man.values()) / 1e6))
    if a.publish:
        if (DEST / "MANIFEST.json").exists():
            sys.exit(f"{DEST} already holds a MANIFEST.json: not overwriting")
        DEST.mkdir(parents=True, exist_ok=True)
        nb = ROOT / "colab/phase6_ytf_colab.ipynb"
        for p in [*sorted(OUT.glob("*.zip")), nb, OUT / "MANIFEST.json"]:  # MANIFEST last
            shutil.copyfile(p, DEST / p.name)
            print("published", p.name, flush=True)
    print("done")


if __name__ == "__main__":
    main()
