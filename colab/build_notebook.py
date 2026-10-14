"""Generate colab/sr_phase3_colab.ipynb (one notebook for all three Colab accounts).

    envs/diag/Scripts/python.exe colab/build_notebook.py [--publish]
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = Path("G:/My Drive/FYP_SR_SHARED")

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Phase 3 remote worker (Colab T4)

**Before the first run (once per account):** open the shared folder `FYP_SR_SHARED` in Google Drive and choose
*Add shortcut to My Drive* (not needed on the owner account). Then **Runtime > Change runtime type > T4 GPU**,
set `MACHINE` below, and **Run all**.

After any disconnect: run all cells again. Finished chunks are skipped, so at most one chunk (1,000 crops) is redone.
Keep this tab open and active.

JOB_TABLE
""")

code('''
MACHINE = "T4-1"          # <-- T4-1 | T4-2 | T4-3

SHARED = "/content/drive/MyDrive/FYP_SR_SHARED"
JOBS = JOBS_MAPPING
import os
os.environ["MPLBACKEND"] = "Agg"   # Colab's inline backend breaks matplotlib inside the venv
JOBLIST = JOBS[MACHINE]            # run in this order; finished jobs are skipped
JOB = JOBLIST[0]
ARM = JOB.split("__")[1]
WORK, REPO, VENV = "/content/work", "/content/work/repo", "/content/venv"
PY = VENV + "/bin/python"
print(MACHINE, "->", JOBLIST, "| arm:", ARM)
''')

code('''
# 1. Drive, GPU, and bundle integrity (a file that has not finished syncing from the PC fails the size/hash check)
from google.colab import drive
drive.mount("/content/drive")
import subprocess, json, hashlib, os, shutil, sys, time, zipfile, pathlib

def sh(cmd, check=True, tail=15, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    out = (r.stdout or "") + (r.stderr or "")
    print("\\n".join(out.strip().splitlines()[-tail:]))
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd[:120]}")
    return r

gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
print("GPU:", gpu)
assert "T4" in gpu, "This job is defined for a T4 (one GPU type per arm). Change the runtime to T4."

B = pathlib.Path(SHARED) / "bundle"
man = json.loads((B / "MANIFEST.json").read_text())
def sha256(p, blk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blk), b""):
            h.update(b)
    return h.hexdigest()
for rel, meta in man.items():
    if rel.startswith("jobs/"):
        continue
    p = B / rel
    assert p.exists() and p.stat().st_size == meta["bytes"], f"{rel}: missing or incomplete; wait for Drive sync on the PC, then rerun"
    if meta["bytes"] > 1_000_000 and sha256(p) != meta["sha256"]:
        raise RuntimeError(f"{rel}: checksum mismatch")
print("bundle verified:", len(man), "files")
''')

code('''
# 2. unpack code, subset inputs, recognizer weights, cross-check data
os.makedirs(REPO, exist_ok=True)
for z in ["code.zip", "subset_inputs.zip", "recognizers.zip", "xcheck.zip"]:
    shutil.copy(B / z, f"{WORK}/{z}")        # always take the current bundle (code may have been updated)
    zipfile.ZipFile(f"{WORK}/{z}").extractall(REPO)
print(sorted(os.listdir(REPO)))
''')

code('''
# 3. Python 3.10 venv with the same torch / package pins as the PC venv for this arm
sh("pip -q install uv gdown", tail=2)
if not os.path.exists(PY):
    sh(f"uv venv {VENV} --python 3.10", tail=3)
sh(f"uv pip install --python {PY} torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu124", tail=4)
sh(f"uv pip install --python {PY} --no-deps -r {SHARED}/bundle/locks/{ARM}.txt", tail=4)
sh(f"{PY} -c \\"import torch,cv2,numpy,pandas,yaml;print(torch.__version__,torch.cuda.is_available())\\"", tail=2)
''')

code('''
# 4. model repo at the pinned commit
PIN = {"osdface": ("https://github.com/jkwang28/OSDFace.git", "OSDFace", "065222742848c7dda06e5a6e99f47e4ceace18d6"),
       "invsr":   ("https://github.com/zsyOAOA/InvSR.git", "InvSR", "609788cd076f1990e38d3912092ca67286e1e0bc")}
url, name, commit = PIN[ARM]
dest = f"{REPO}/models/{name}"
if not os.path.exists(dest):
    sh(f"git clone {url} {dest}", tail=2)
sh(f"git -C {dest} checkout {commit}", tail=2)
print(sh(f"git -C {dest} rev-parse HEAD", tail=1).stdout)
''')

code('''
# 5. weights from the official sources (same ones as the PC), then SHA-256 prefix check against the PC's files
W = f"{REPO}/models/weights"
def hf(repo, local, patterns):
    code_ = ("from huggingface_hub import snapshot_download;"
             f"snapshot_download('{repo}', local_dir='{local}', allow_patterns={patterns!r}, max_workers=2)")
    for attempt in range(3):
        r = sh(f"{PY} -c \\"{code_}\\"", check=False, tail=3)
        if r.returncode == 0:
            return
        time.sleep(15)
    raise RuntimeError("download failed: " + repo)

SD_FILES = ["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json", "text_encoder/model.safetensors",
            "unet/config.json", "unet/diffusion_pytorch_model.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.safetensors", "feature_extractor/*"]
if ARM == "osdface":
    off = f"{W}/osdface/official"
    if not os.path.exists(off + "/associate_2.ckpt"):
        r = sh(f"gdown --folder https://drive.google.com/drive/folders/1Nci6KufB8t2Uj-6tobrw3S7kkfQUTLHV -O {off}", check=False, tail=3)
        if not os.path.exists(off + "/associate_2.ckpt"):          # quota fallback: the PC places a copy in the shared bundle
            fb = B / "osdface_official.zip"
            assert fb.exists(), "gdown hit its quota and no fallback zip is in the bundle: ask the PC owner to publish osdface_official.zip"
            zipfile.ZipFile(fb).extractall(off)
    hf("sd2-community/stable-diffusion-2-1-base", f"{W}/osdface/sd21base", SD_FILES)   # hash-verified identical to the ModelScope copy
else:
    hf("OAOA/InvSR", f"{W}/invsr/InvSR", ["noise_predictor_sd_turbo_v5.pth"])
    hf("stabilityai/sd-turbo", f"{W}/invsr/sd-turbo", SD_FILES)

exp = json.loads((B / "weights_sha256.json").read_text())
bad = []
for rel, prefix in exp.items():
    p = f"{REPO}/{rel}"
    if os.path.exists(p) and not sha256(p).startswith(prefix):
        bad.append(rel)
missing = [rel for rel in exp if (ARM in rel or "osdface" not in rel and "invsr" not in rel) and not os.path.exists(f"{REPO}/{rel}")]
print("checksum mismatches:", bad, "| missing:", missing)
assert not bad and not missing, "weights do not match the PC's files; stop and report"
''')

code('''
# 6. cross-device check: reproduce the PC's smoke outputs and direct-feed embeddings; also measures T4 seconds per crop
os.makedirs(f"{SHARED}/jobs/{JOB}", exist_ok=True)
r = sh(f"{PY} scripts/sr_colab_xcheck.py --arm {ARM} --xcheck {REPO}/xcheck --out {SHARED}/jobs/{JOB}/xcheck.json --work {WORK}",
       check=False, cwd=REPO, tail=40)
assert r.returncode == 0, "cross-device check FAILED (see xcheck.json in the job folder): do not run the job; report to the PC owner"
for other in JOBLIST[1:]:            # same machine, same check: record it in every job folder
    os.makedirs(f"{SHARED}/jobs/{other}", exist_ok=True)
    shutil.copy(f"{SHARED}/jobs/{JOB}/xcheck.json", f"{SHARED}/jobs/{other}/xcheck.json")
''')

code('''
# 7. run this machine's jobs in order, in 1,000-crop chunks (resumable: rerun this cell after a disconnect)
for job in JOBLIST:
    if os.path.exists(f"{SHARED}/jobs/{job}/DONE"):
        print(job, ": already DONE"); continue
    print("=== starting", job)
    p = subprocess.Popen([PY, "-u", "scripts/sr_colab_job.py", "--job", f"{SHARED}/jobs/{job}/job.json", "--work", WORK, "--chunk", "1000"],
                         cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "chunk" in line or "JOB" in line or "job" in line:
            print(line.rstrip())
    p.wait()
    print(job, "exit code", p.returncode, "| DONE marker:", os.path.exists(f"{SHARED}/jobs/{job}/DONE"))
    if p.returncode != 0:
        raise RuntimeError(f"{job} did not finish: rerun this cell (finished chunks are kept)")
''')


def fill_jobs():
    """Insert the machine -> job-list mapping and the job table from outputs/sr/colab_bundle/jobs/*/job.json."""
    jobs = [json.loads(p.read_text()) for p in sorted((ROOT / "outputs/sr/colab_bundle/jobs").glob("*/job.json"))]
    jobs.sort(key=lambda j: (j["machine"], j["part"] != "eval", j["job_id"]))   # eval shard first, then seeds
    mapping = {}
    for j in jobs:
        mapping.setdefault(j["machine"], []).append(j["job_id"])
    rows = ["| MACHINE | Jobs, in order |", "|---|---|"]
    for m, ids in sorted(mapping.items()):
        rows.append(f"| {m} | " + "; then ".join(ids) + " |")
    for c in CELLS:
        src = "".join(c["source"]).replace("JOBS_MAPPING", json.dumps(mapping, indent=1)).replace("JOB_TABLE", "\n".join(rows))
        c["source"] = src.splitlines(keepends=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    fill_jobs()
    nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"name": "sr_phase3_colab.ipynb"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "colab" / "sr_phase3_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)
    if a.publish:
        shutil.copy2(out, SHARED / "sr_phase3_colab.ipynb")
        print("published to", SHARED)


if __name__ == "__main__":
    main()
