"""Resumable parallel download of the MEVID probe archives (MEVID_feasibility_probe.md, Step 1).

    envs/video/Scripts/python.exe scripts/mevid_download.py            # downloads the three archives into Dataset/MEVID/archives/

Why not a single `curl -L -C -`: the S3 host gave about 0.2 to 0.3 MB/s per connection on the lab network while several connections together gave several times that
(measured with range requests), so each file is cut into 128 MB byte ranges fetched by up to 32 `curl` processes at once, all files in one pool. Every range is a separate file under
`archives/.parts/` that resumes from the bytes it already has; when all ranges are complete they are joined in order and checked: size equal to Content-Length, then
`zipfile.testzip()` for a zip or a full gzip read (CRC) for a tgz. The data are the same files from the same URLs as the plan; only the transport differs.
Nothing is committed; the archives stay under Dataset/ (git-ignored).
"""
import concurrent.futures as cf
import gzip
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "Dataset/MEVID/archives"
BASE = "https://mevadata-public-01.s3.amazonaws.com/mevid-annotations/"
FILES = ["mevid-v1-annotation-data.zip", "mevid-v1-actor-checkin-photos.zip", "mevid-v1-bbox-test.tgz"]
CHUNK, WORKERS = 128 * 1024 * 1024, 32
CURL = "curl.exe" if sys.platform == "win32" else "curl"


def head_size(url):
    out = subprocess.run([CURL, "-sIL", "-m", "60", url], capture_output=True, text=True).stdout
    sizes = [int(l.split(":")[1]) for l in out.splitlines() if l.lower().startswith("content-length:")]
    assert sizes, f"no Content-Length for {url}"
    return sizes[-1]


def fetch_part(url, part, start, end):
    """Download bytes start..end into `part`, resuming from what is already there; returns when complete."""
    want = end - start + 1
    for attempt in range(1000):  # a connection slower than 100 KB/s for 20 s is dropped and retried; the range resumes from the bytes it already has
        have = part.stat().st_size if part.exists() else 0
        if have >= want:
            return
        p = subprocess.Popen([CURL, "-sS", "-L", "-m", "600", "--speed-limit", "100000", "--speed-time", "20", "-r", f"{start + have}-{end}", url], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        with open(part, "ab") as f:
            while True:
                b = p.stdout.read(1 << 20)
                if not b:
                    break
                f.write(b)
        p.wait()
    assert part.exists() and part.stat().st_size == want, f"{part.name}: incomplete after retries"


def plan(name):
    """-> (url, out, size, parts_dir, ranges) for one archive; ranges = [(index, start, end)]."""
    url, out = BASE + name, DEST / name
    size = head_size(url)
    parts_dir = DEST / ".parts" / name
    ranges = [(i, s, min(s + CHUNK, size) - 1) for i, s in enumerate(range(0, size, CHUNK))]
    return url, out, size, parts_dir, ranges


def finish(name, out, size, parts_dir, ranges):
    """Join the complete ranges in order, check the size and the archive's own integrity."""
    if not (out.exists() and out.stat().st_size == size):
        tmp = out.with_name(out.name + ".joining")
        with open(tmp, "wb") as w:
            for i, _, _ in ranges:
                w.write((parts_dir / f"{i:04d}").read_bytes())
        assert tmp.stat().st_size == size, f"{name}: joined size {tmp.stat().st_size} != {size}"
        tmp.replace(out)
        for i, _, _ in ranges:
            (parts_dir / f"{i:04d}").unlink()
        parts_dir.rmdir()
    if name.endswith(".zip"):
        bad = zipfile.ZipFile(out).testzip()
        assert bad is None, f"{name}: corrupt member {bad}"
    else:
        with gzip.open(out, "rb") as g:
            while g.read(1 << 24):
                pass
    print(f"{name}: OK, {size} bytes, integrity check passed", flush=True)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    plans = {n: plan(n) for n in FILES}
    jobs = []
    for n, (url, out, size, parts_dir, ranges) in plans.items():
        if out.exists() and out.stat().st_size == size:
            continue
        parts_dir.mkdir(parents=True, exist_ok=True)
        jobs += [(n, url, parts_dir / f"{i:04d}", s_, e_) for i, s_, e_ in ranges]
    total = sum(p[2] for p in plans.values())
    done = 0
    # every range of every file goes into one pool, so a slow range never holds up the others
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch_part, url, part, s_, e_): (n, e_ - s_ + 1) for n, url, part, s_, e_ in jobs}
        for fu in cf.as_completed(futs):
            fu.result()
            done += futs[fu][1]
            print(f"downloaded {done / 1e9:.2f} GB of {sum(plans[n][2] for n in {j[0] for j in jobs}) / 1e9:.2f} GB (ranges still to fetch: {len(futs) - 0})", flush=True)
    for n, (url, out, size, parts_dir, ranges) in plans.items():
        finish(n, out, size, parts_dir, ranges)
    print("DOWNLOAD_FINISHED", flush=True)
