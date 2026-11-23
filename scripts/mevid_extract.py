"""MEVID probe, Step 1 (second half): SHA-256 and byte count of every archive, then extraction into Dataset/MEVID/.

    envs/video/Scripts/python.exe scripts/mevid_extract.py

Writes outputs/video/mevid_probe/archives_sha256.txt (`sha256  bytes  name`). The zips are extracted with zipfile (it drops absolute and `..` paths) and the tgz with tarfile's
`data` filter (refuses links and paths that leave the target). Archives are untrusted downloads: they are extracted into new folders under Dataset/MEVID/ and never executed.
Resumable in the sense that an archive whose extraction folder already exists with a `.done` marker is skipped.
"""
import hashlib
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "Dataset/MEVID/archives"
DEST = ROOT / "Dataset/MEVID"
OUT = ROOT / "outputs/video/mevid_probe"
FILES = ["mevid-v1-annotation-data.zip", "mevid-v1-actor-checkin-photos.zip", "mevid-v1-bbox-test.tgz"]


def sha256(p, blk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for n in FILES:
        p = ARCH / n
        lines.append(f"{sha256(p)}  {p.stat().st_size}  {n}")
        print(lines[-1], flush=True)
    (OUT / "archives_sha256.txt").write_text("\n".join(lines) + "\n")
    for n in FILES:
        marker = DEST / f".{n}.done"
        if marker.exists():
            print(n, "already extracted", flush=True)
            continue
        p = ARCH / n
        if n.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                z.extractall(DEST)
        else:
            with tarfile.open(p, "r:gz") as t:
                t.extractall(DEST, filter="data")
        marker.write_text("done\n")
        print(n, "extracted", flush=True)
    print("EXTRACT_FINISHED", flush=True)


if __name__ == "__main__":
    sys.exit(main())
