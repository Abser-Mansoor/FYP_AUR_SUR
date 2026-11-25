"""MEVID probe, Step 1 (second half): SHA-256 and byte count of every archive, then extraction into Dataset/MEVID/.

    envs/video/Scripts/python.exe scripts/mevid_extract.py

Writes outputs/video/mevid_probe/archives_sha256.txt (`sha256  bytes  name`; an archive already recorded with the same size is not hashed again). The zips are extracted with zipfile (it drops absolute and `..` paths) and the tgz with tarfile's
`data` filter (refuses links and paths that leave the target). Archives are untrusted downloads: they are extracted into new folders under Dataset/MEVID/ and never executed.
Resumable in the sense that an archive whose extraction folder already exists with a `.done` marker is skipped.
"""
import argparse
import hashlib
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "Dataset/MEVID/archives"
DEST = ROOT / "Dataset/MEVID"
OUT = ROOT / "outputs/video/mevid_probe"
ANN = DEST / "mevid-v1-annotation-data"
MAX_CHIPS = 50  # as scripts/mevid_embed.py: at most 50 evenly spaced chips per tracklet
FILES = ["mevid-v1-annotation-data.zip", "mevid-v1-actor-checkin-photos.zip", "mevid-v1-bbox-test.tgz", "mevid-v1-bbox-train.tgz"]  # train added for Phase 8


def sha256(p, blk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blk), b""):
            h.update(b)
    return h.hexdigest()


def needed_train_chips():
    """Names of the train chips that Phase 8 reads: up to 50 evenly spaced chips per train tracklet (the same rule as scripts/mevid_embed.py:pick_chips)."""
    idx = pd.read_csv(ROOT / "outputs/video/mevid/tracklets_index.csv")
    names = (ANN / "train_name.txt").read_text().split()
    need = set()
    for r in idx[idx.split == "train"].itertuples():
        ids = np.unique(np.round(np.linspace(r.start, r.end, min(MAX_CHIPS, r.end - r.start + 1))).astype(int))
        need.update(names[int(i)] for i in ids)
    return need


def extract_train_needed(p):
    """Stream the train archive once and write only the needed chips (about 317,000 of 3.6 million). A deviation from a full extraction, recorded in REPORT_phase8.md; the data read are identical."""
    need = needed_train_chips()
    done = 0
    with tarfile.open(p, "r|gz") as t:
        for mem in t:
            if mem.isfile() and Path(mem.name).name in need:
                t.extract(mem, DEST, filter="data")
                done += 1
                if done % 50000 == 0:
                    print(f"  {done} of {len(need)} needed train chips written", flush=True)
    assert done == len(need), f"found {done} of {len(need)} needed train chips in the archive"
    print(f"train: {done} needed chips extracted", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = OUT / "archives_sha256.txt"
    known = {l.split()[2]: l.strip() for l in rec.read_text().splitlines() if l.strip()} if rec.exists() else {}
    lines = []
    for n in FILES:
        p = ARCH / n
        if n in known and int(known[n].split()[1]) == p.stat().st_size:
            lines.append(known[n])  # already hashed (same size): keep the recorded line
        else:
            lines.append(f"{sha256(p)}  {p.stat().st_size}  {n}")
        print(lines[-1], flush=True)
    rec.write_text(chr(10).join(lines) + chr(10))
    for n in FILES:
        marker = DEST / f".{n}.done"
        if marker.exists():
            print(n, "already extracted", flush=True)
            continue
        p = ARCH / n
        if n.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                z.extractall(DEST)
        elif n == "mevid-v1-bbox-train.tgz" and ARGS.train_needed_only:
            extract_train_needed(p)
        else:
            with tarfile.open(p, "r:gz") as t:
                t.extractall(DEST, filter="data")
        marker.write_text("done\n")
        print(n, "extracted", flush=True)
    print("EXTRACT_FINISHED", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-needed-only", action="store_true", help="extract only the train chips that Phase 8 reads (up to 50 per tracklet)")
    ARGS = ap.parse_args()
    sys.exit(main())
