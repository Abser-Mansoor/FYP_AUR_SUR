"""Sync Phase 4 results back to the shared Drive folder (run on Raghib's laptop at the end of a working session and at the end).

    envs\\score\\Scripts\\python.exe scripts\\phase4_upload.py [--dest "G:\\My Drive\\FYP_SR_SHARED\\phase4"] [--thumbs-n 1000]

Writes <dest>\\results_<date>.zip and results_<date>.manifest.json (SHA-256 and size of the zip and of every member). Contents:
  outputs/enhance/results/   (all CSV / JSON results and generated tables)
  REPORT_phase4.md
  outputs/enhance/failure_cases/, outputs/enhance/samples/, outputs/enhance/smoke/montage.png, docs/figures/enh_*.png
  logs: outputs/enhance/run.log, *_stdout.log, *_stderr.log
  thumbs/<arm>__<setting>/*.jpg : 256 px JPEG thumbnails of the selected variants' eval outputs (Phase 3 style); a seeded (42) random
                                  sample of --thumbs-n crops per variant (0 = every crop). Full-size restored PNGs are not uploaded.
Existing destination files are never overwritten (the date suffix gets a counter).
"""
import argparse
import datetime
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTS = [r"G:\My Drive\FYP_SR_SHARED\phase4", r"G:\.shortcut-targets-by-id\1jj9zaEy9AHTi9VIn4AAbz37lf4domATL\FYP_SR_SHARED\phase4"]


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=None)
    ap.add_argument("--thumbs-n", type=int, default=1000)
    a = ap.parse_args()
    dest = Path(a.dest) if a.dest else next((Path(p) for p in DEFAULT_DESTS if Path(p).exists()), None)
    if dest is None or not dest.exists():
        sys.exit("destination folder not found: pass --dest")
    date = datetime.date.today().isoformat()
    n = 0
    while (dest / f"results_{date}{'' if n == 0 else '_' + str(n)}.zip").exists():
        n += 1
    stem = f"results_{date}{'' if n == 0 else '_' + str(n)}"
    zpath, mpath = dest / f"{stem}.zip", dest / f"{stem}.manifest.json"
    files = []
    for pat in ("outputs/enhance/results/*", "outputs/enhance/failure_cases/*", "outputs/enhance/samples/*", "docs/figures/enh_*.png",
                "outputs/enhance/*.log", "outputs/enhance/smoke/montage.png"):
        files += [p for p in sorted(ROOT.glob(pat)) if p.is_file()]
    files.append(ROOT / "REPORT_phase4.md")
    members = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.exists():
                rel = p.relative_to(ROOT).as_posix()
                z.write(p, rel)
                members.append(dict(name=rel, bytes=p.stat().st_size, sha256=sha256_file(p)))
        import yaml
        cfg = yaml.safe_load(open(ROOT / "configs/enhance.yaml"))
        res = ROOT / cfg["paths"]["results"]
        eval_files = sorted(p.name for p in (ROOT / "outputs/sr/subset/eval_input").glob("*.png"))
        pick = eval_files if a.thumbs_n <= 0 else sorted(np.random.default_rng(42).choice(eval_files, min(a.thumbs_n, len(eval_files)), replace=False))
        sel = []
        for arm in cfg["arms"]:
            f = res / f"selection_{arm}.json"
            if arm != "noop" and f.exists() and json.loads(f.read_text()):
                sel.append((arm, json.loads(f.read_text())[0]["setting"]))
        for arm, st in sel:
            d = ROOT / "outputs/sr/restored/eval" / f"{arm}__{st}" / "seed0"
            for name in pick:
                p = d / name
                if not p.exists():
                    continue
                im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
                s = 256 / max(im.shape[:2])
                im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
                rel = f"thumbs/{arm}__{st}/{Path(name).stem}.jpg"
                z.writestr(rel, buf.tobytes())
                members.append(dict(name=rel, bytes=len(buf), sha256=sha256(buf.tobytes())))
    man = dict(zip_file=zpath.name, zip_bytes=zpath.stat().st_size, zip_sha256=sha256_file(zpath), created=datetime.datetime.now().isoformat(timespec="seconds"),
               thumbs_per_variant=len(pick), members=members)
    mpath.write_text(json.dumps(man, indent=1))
    print(f"wrote {zpath} ({man['zip_bytes'] / 2**20:.1f} MB, {len(members)} members) and {mpath.name}")


if __name__ == "__main__":
    main()
