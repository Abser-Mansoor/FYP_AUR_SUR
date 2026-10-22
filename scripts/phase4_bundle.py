"""Build the Phase 4 data bundle for Raghib's laptop and publish it to the shared Drive folder.

    envs/diag/Scripts/python.exe scripts/phase4_bundle.py                  # build into outputs/enhance/phase4_bundle/
    envs/diag/Scripts/python.exe scripts/phase4_bundle.py --publish        # ... and copy to G:/My Drive/FYP_SR_SHARED/phase4

Bundle (see research/Sprint_Notes/Phase4_restoration_runbook.md, step 0):
  subset_inputs.zip   tuning + eval native crops (the Phase 3 file, copied unchanged)
  recognizers.zip     the 4 recognizer weights (the Phase 3 file, copied unchanged; members checked against MODEL_INVENTORY prefixes)
  phase4_data.zip     raw Phase 2 measures (luminance, lap_var), subset CSVs, LFW hr112 crops + pairs.txt, the 75 smoke crops,
                      and the lab-PC direct-feed embeddings (cross-device check only, under outputs/enhance/xcheck/lab_direct/)
  MANIFEST.json       size + sha256 of every file; scripts/phase4_unpack.py refuses a file that does not match
Nothing in the repo is modified. Zips are uncompressed (PNG and weights do not compress).
"""
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sr_colab_bundle import RECOGNIZER_FILES, WEIGHT_SHA  # noqa: E402

OUT = ROOT / "outputs/enhance/phase4_bundle"
SHARED = Path("G:/My Drive/FYP_SR_SHARED")
PHASE3_BUNDLE = SHARED / "bundle"
RECS = ["arcface", "adaface", "magface", "curricularface"]


def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def zip_files(zpath, items):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for p, arc in items:
            z.write(p, arc)


def data_items():
    items = []
    for p in sorted((ROOT / "outputs/stratify/measures").glob("*.csv")):
        items.append((p, f"outputs/stratify/measures/{p.name}"))
    for part in ("tuning", "eval"):
        for kind in ("images", "pairs"):
            p = ROOT / f"outputs/sr/subset/{part}_{kind}.csv"
            items.append((p, p.relative_to(ROOT).as_posix()))
    for p in sorted((ROOT / "outputs/sr/lfw/hr112").glob("*.png")):
        items.append((p, f"outputs/sr/lfw/hr112/{p.name}"))
    items.append((ROOT / "data/lfw/lfw_home/pairs.txt", "data/lfw/lfw_home/pairs.txt"))
    for p in sorted((ROOT / "outputs/sr/smoke/input").glob("*.png")):
        items.append((p, f"outputs/sr/smoke/input/{p.name}"))
    items.append((ROOT / "outputs/sr/smoke/manifest.csv", "outputs/sr/smoke/manifest.csv"))
    for part in ("tuning", "eval"):
        for r in RECS:
            p = ROOT / f"outputs/sr/emb/{part}/direct/resize/seed0/{part}__{r}.npy"
            items.append((p, f"outputs/enhance/xcheck/lab_direct/{p.name}"))
    missing = [str(p) for p, _ in items if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} bundle inputs missing, e.g. {missing[:3]}")
    return items


def check_recognizers(z):
    """Members of recognizers.zip must match the SHA-256 prefixes recorded in MODEL_INVENTORY.md."""
    with zipfile.ZipFile(z) as zf:
        assert sorted(zf.namelist()) == sorted(f"models/weights/{f}" for f in RECOGNIZER_FILES), "unexpected members in recognizers.zip"
        for n in zf.namelist():
            h = hashlib.sha256()
            with zf.open(n) as f:
                for b in iter(lambda: f.read(1 << 22), b""):
                    h.update(b)
            assert h.hexdigest().startswith(WEIGHT_SHA[n]), f"{n}: sha256 does not match MODEL_INVENTORY prefix {WEIGHT_SHA[n]}"
            print(f"  {n}: sha256 ok ({h.hexdigest()[:12]})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for f in ("subset_inputs.zip", "recognizers.zip"):
        print(f"copy {f} from the Phase 3 bundle", flush=True)
        shutil.copyfile(PHASE3_BUNDLE / f, OUT / f)
    print("check recognizer weights", flush=True)
    check_recognizers(OUT / "recognizers.zip")
    items = data_items()
    print(f"phase4_data.zip: {len(items)} files", flush=True)
    zip_files(OUT / "phase4_data.zip", items)
    man = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(OUT.glob("*.zip"))}
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    if a.publish:
        dest = SHARED / "phase4"
        dest.mkdir(parents=True, exist_ok=True)
        for p in [*sorted(OUT.glob("*.zip")), OUT / "MANIFEST.json"]:  # MANIFEST last: Raghib's unpack waits for it
            shutil.copyfile(p, dest / p.name)
            print("published", p.name, flush=True)
    print("done")


if __name__ == "__main__":
    main()
