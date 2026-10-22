"""Verify and unpack the Phase 4 bundle into the repo (run on Raghib's laptop; Python standard library only).

    python scripts\\phase4_unpack.py --src "G:\\My Drive\\FYP_SR_SHARED\\phase4"

Every zip is checked against MANIFEST.json (size and sha256) first; any mismatch (for example a file Drive has not finished
syncing) stops the script before anything is written. Existing files are never overwritten.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    src = Path(ap.parse_args().src)
    man_file = src / "MANIFEST.json"
    if not man_file.exists():
        sys.exit(f"{man_file} not found: the bundle has not finished syncing")
    man = json.loads(man_file.read_text())
    for name, m in man.items():
        p = src / name
        if not p.exists() or p.stat().st_size != m["bytes"]:
            sys.exit(f"{name}: missing or wrong size (expected {m['bytes']} bytes); wait for Drive to finish syncing")
        print(f"checking {name} ...", flush=True)
        if sha256(p) != m["sha256"]:
            sys.exit(f"{name}: sha256 mismatch; do not use this bundle")
    for name in man:
        n_new = n_skip = 0
        with zipfile.ZipFile(src / name) as z:
            for info in z.infolist():
                dest = ROOT / info.filename
                if dest.exists():
                    n_skip += 1
                    continue
                z.extract(info, ROOT)
                n_new += 1
        print(f"{name}: {n_new} files written, {n_skip} already present", flush=True)
    print("done")


if __name__ == "__main__":
    main()
