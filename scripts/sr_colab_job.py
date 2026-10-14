"""Colab / remote worker for ONE Phase 3 job, processed in chunks so a disconnect loses at most one chunk.

    python scripts/sr_colab_job.py --job <shared>/jobs/<job_id>/job.json --work /content/work

Run it with the MODEL venv's python (it calls the repo runner with the same interpreter and embeds with the
4 recognizers in the same process). Per chunk of crops:
  1. restore the crops locally with the unchanged runner (src/sr/run_<arm>.py, --file-list),
  2. embed every restored crop with all 4 recognizers (INTER_AREA to 112, flip-sum: same rule as src/sr/eval.py),
  3. write to the shared job folder: names.json, emb_<recognizer>.npy, timing.json, thumbs.zip (256-px JPEGs),
  4. write the chunk marker `embedded` LAST (atomic), then delete the local PNGs.
A chunk without its marker is redone. The job writes DONE only when every chunk is complete and complete in size.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

os.environ["MPLBACKEND"] = "Agg"  # Colab sets an inline backend that does not exist in the venv; children inherit this
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recognizer import Recognizer  # noqa: E402
from src.sr import common as C  # noqa: E402

RECS = ("arcface", "adaface", "magface", "curricularface")


def atomic_save_npy(path, arr):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def runner_cmd(job, names_file, out_dir):
    """Build the runner command line for this job's arm (one output folder for the chunk)."""
    arm = job["arm"]
    base = [sys.executable, "-W", "ignore", str(ROOT / f"src/sr/run_{arm}.py"), "--input-dir", f"outputs/sr/subset/{job['part']}_input",
            "--file-list", str(names_file)]
    if arm == "osdface":
        a = job["args"]
        return base + ["--out-root", str(out_dir.parent / "_tmp"), "--pre-sizes", str(a["pre"]), "--seed", str(job["seed"]),
                       "--out-template", str(out_dir), "--sd-path", a["sd_path"]]
    if arm == "invsr":
        a = job["args"]
        return base + ["--out-dir", str(out_dir), "--pre-size", str(a["pre"]), "--steps", str(a["steps"]), "--seed", str(job["seed"])]
    raise SystemExit(f"arm {arm} is not supported on remote workers (only osdface and invsr)")


def embed_dir(recs, out_dir, names, batch=256):
    """Embeddings of the restored crops of a chunk, in `names` order, for all recognizers."""
    res = {r: [] for r in recs}
    for s in range(0, len(names), batch):
        imgs = [C.read_bgr(out_dir / n) for n in names[s:s + batch]]
        for r, rec in recs.items():
            rec.pre["interp"] = "area"
            res[r].append(rec.get_embedding(imgs))
    return {r: np.concatenate(v).astype(np.float32) for r, v in res.items()}


def write_thumbs(zip_path, out_dir, names, size=256):
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            im = C.read_bgr(out_dir / n)
            t = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) if im.shape[0] != size else im
            ok, buf = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 88])
            z.writestr(Path(n).stem + ".jpg", buf.tobytes())
    os.replace(tmp, zip_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="path to <shared>/jobs/<job_id>/job.json")
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--limit-chunks", type=int, default=None, help="test only")
    ap.add_argument("--keep", action="store_true", help="test only: keep the local restored PNGs")
    ap.add_argument("--first", type=int, default=None, help="test only: use only the first N crops of the job")
    a = ap.parse_args()
    jp = Path(a.job)
    job = json.loads(jp.read_text())
    jdir = jp.parent
    cfg = yaml.safe_load(open(ROOT / "configs/sr.yaml"))
    d = pd.read_csv(ROOT / "outputs/sr/subset" / f"{job['part']}_images.csv")
    files = d["file"].tolist()
    if job.get("range"):  # shard of the subset: crops [start, end) in subset order
        files = files[job["range"][0]:job["range"][1]]
    if a.first:
        files = files[:a.first]
    chunks = [files[i:i + a.chunk] for i in range(0, len(files), a.chunk)]
    work = Path(a.work).resolve() / "restored" / job["job_id"]
    work.mkdir(parents=True, exist_ok=True)
    recs = {r: Recognizer(r, ROOT / cfg["recognizers"][r], (112, 112), "area", True, 128, "cuda") for r in RECS}
    t_start = time.time()
    failed = []
    for k, names in enumerate(chunks):
        if a.limit_chunks is not None and k >= a.limit_chunks:
            break
        cdir = jdir / "chunks" / f"{k:04d}"
        if (cdir / "embedded").exists():
            print(f"chunk {k}/{len(chunks)}: already done", flush=True)
            continue
        cdir.mkdir(parents=True, exist_ok=True)
        out = work / f"{k:04d}"
        nf = work / f"names_{k:04d}.txt"
        nf.write_text("\n".join(names), encoding="utf-8")
        t0 = time.time()
        for attempt in range(1, 4):
            r = subprocess.run(runner_cmd(job, nf, out), cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            have = sum((out / n).exists() for n in names)
            if r.returncode == 0 and have == len(names):
                break
            print(f"chunk {k}: attempt {attempt} rc={r.returncode} have {have}/{len(names)}\n{(r.stderr or '')[-600:]}", flush=True)
            time.sleep(20)
        have = sum((out / n).exists() for n in names)
        if have != len(names):
            failed.append(k)
            atomic_write_text(cdir / "FAILED.txt", f"restored {have}/{len(names)} crops after 3 attempts")
            continue
        emb = embed_dir(recs, out, names)
        for rn, arr in emb.items():
            atomic_save_npy(cdir / f"emb_{rn}.npy", arr)
        write_thumbs(cdir / "thumbs.zip", out, names)
        tj = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
        tj.pop("sec_list", None)
        atomic_write_text(cdir / "timing.json", json.dumps(tj, indent=1))
        atomic_write_text(cdir / "names.json", json.dumps(names))
        atomic_write_text(cdir / "embedded", f"{len(names)} crops, {time.time() - t0:.0f}s")
        if not a.keep:
            shutil.rmtree(out, ignore_errors=True)
        print(f"chunk {k + 1}/{len(chunks)} done: {len(names)} crops in {time.time() - t0:.0f}s "
              f"(median {tj.get('sec_per_crop_median')} s/crop, peak VRAM {tj.get('peak_vram_mb')} MB)", flush=True)
    done = sum((jdir / 'chunks' / f'{k:04d}' / 'embedded').exists() for k in range(len(chunks)))
    if done == len(chunks) and not failed and a.limit_chunks is None and not a.first:
        import torch
        dev = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda, python=sys.version.split()[0],
                   finished=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(time.time() - t_start))
        atomic_write_text(jdir / "device.json", json.dumps(dev, indent=1))
        atomic_write_text(jdir / "DONE", f"{len(files)} crops in {len(chunks)} chunks")
        print("JOB DONE", dev, flush=True)
    else:
        print(f"job NOT complete: {done}/{len(chunks)} chunks done; failed chunks {failed}", flush=True)
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
