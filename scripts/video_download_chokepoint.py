"""Download ChokePoint (Zenodo 815657, CC BY-NC 4.0) into Dataset/ChokePoint/archives/, verify md5, extract.

    envs/diag/Scripts/python.exe scripts/video_download_chokepoint.py [--no-extract]

Resumable: a partial <file>.part is continued with an HTTP Range request; a finished file with a matching md5 is skipped;
extraction is skipped for an archive whose marker <name>.extracted exists. Datasets are read-only after this step.
"""
import argparse
import hashlib
import json
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "Dataset/ChokePoint"
ARCH = DEST / "archives"
API = "https://zenodo.org/api/records/815657"


def md5(path, block=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, out, size):
    part = out.with_name(out.name + ".part")
    for attempt in range(1, 9):
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored the Range header: appending would corrupt the file
                    part.unlink()
                    print(f"  {out.name}: no range support (HTTP {r.status}); restarting from zero", flush=True)
                    continue
                with open(part, "ab" if have else "wb") as f:  # closed (flushed) even when the connection drops mid-read
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
        except Exception as e:  # network drop: retry from where it stopped
            print(f"  {out.name}: attempt {attempt} failed ({type(e).__name__}: {e}); retrying", flush=True)
            time.sleep(10 * attempt)
    if not part.exists() or part.stat().st_size != size:
        raise IOError(f"{out.name}: incomplete download")
    part.replace(out)


def extract(p):
    """Extract an archive into DEST. Some are nested (P2E_S5.tar.xz holds P2E_S5_C1.tar.xz ...): those are extracted too, then the
    inner tarball is removed from DEST (the original stays in archives/)."""
    with tarfile.open(p, "r:xz") as t:
        members = t.getnames()
        t.extractall(DEST, filter="data")
    for n in members:
        if n.endswith(".tar.xz"):
            inner = DEST / n
            with tarfile.open(inner, "r:xz") as t:
                t.extractall(DEST, filter="data")
            inner.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--extract-only", action="store_true", help="skip the download step (archives must already be complete)")
    ap.add_argument("--workers", type=int, default=1, help="parallel file downloads (Zenodo limits each connection)")
    a = ap.parse_args()
    ARCH.mkdir(parents=True, exist_ok=True)
    if a.extract_only:
        for p in sorted(ARCH.glob("*.tar.xz")):
            mark = ARCH / (p.name + ".extracted")
            if not mark.exists():
                print(f"extract {p.name}", flush=True)
                extract(p)
                mark.write_text("ok")
        print("done", flush=True)
        return
    rec = json.load(urllib.request.urlopen(API, timeout=60))
    (DEST / "zenodo_record.json").write_text(json.dumps(rec["metadata"], indent=1), encoding="utf-8")
    files = sorted(rec["files"], key=lambda f: f["size"])  # small files first: groundtruth and README are usable early
    def get(i_f):
        i, f = i_f
        out, want = ARCH / f["key"], f["checksum"].split(":")[1]
        if out.exists() and out.stat().st_size == f["size"] and md5(out) == want:
            print(f"[{i}/{len(files)}] {f['key']}: already complete", flush=True)
            return
        print(f"[{i}/{len(files)}] {f['key']} ({f['size'] / 1e6:.0f} MB) ...", flush=True)
        fetch(f["links"]["self"], out, f["size"])
        got = md5(out)
        if got != want:
            out.replace(out.with_name(out.name + ".BAD"))
            raise IOError(f"{f['key']}: md5 mismatch (got {got}, expected {want}); moved to .BAD")
        print(f"  {f['key']}: md5 ok", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(get, enumerate(files, 1)))  # re-raises the first failure
    if not a.no_extract:
        for p in sorted(ARCH.glob("*.tar.xz")):
            mark = ARCH / (p.name + ".extracted")
            if mark.exists():
                continue
            print(f"extract {p.name}", flush=True)
            extract(p)
            mark.write_text("ok")
    print("done", flush=True)


if __name__ == "__main__":
    main()
