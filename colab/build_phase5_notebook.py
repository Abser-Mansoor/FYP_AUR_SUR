"""Generate colab/phase5_detector_colab.ipynb (the Phase 5 detector study on one Colab T4).

    envs/diag/Scripts/python.exe colab/build_phase5_notebook.py [--publish]

--publish also copies the notebook to G:/My Drive/FYP_SR_SHARED/phase5/. The bundle itself comes from scripts/phase5_bundle.py.
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
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Phase 5 detector study (Colab T4)

Compares SCRFD and RetinaFace (full frame at 640, full frame at 1280, 2x2 tiles) on WIDER FACE val, a ChokePoint frame sample
and the TinyFace/SurvFace eval crops. **Inference only. Every number reported comes from a T4**, so run it on a T4 and nothing else.

**Before the first run (once per account):** open the shared folder `FYP_SR_SHARED` in Google Drive and choose
*Add shortcut to My Drive* (not needed on the owner account). Then **Runtime > Change runtime type > T4 GPU** and **Run all**.

The notebook stops by itself if the bundle has not finished syncing, if the GPU is not a T4, or if the cross-device check fails
(SCRFD keypoints on 100 TinyFace probes must agree with the lab PC within 0.5 px). In that case do not continue: tell the lab PC owner.

After any disconnect run all cells again. Finished chunks (200 images each) are skipped. Keep this tab open and active.
Results go to `FYP_SR_SHARED/phase5/results/` on Drive; the lab PC merges them.
""")

code('''
# 1. Drive, GPU, bundle integrity
import os
os.environ["MPLBACKEND"] = "Agg"
from google.colab import drive
drive.mount("/content/drive")
import subprocess, json, hashlib, shutil, sys, time, zipfile, pathlib

def sh(cmd, check=True, tail=15, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    out = (r.stdout or "") + (r.stderr or "")
    print("\\n".join(out.strip().splitlines()[-tail:]))
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd[:120]}")
    return r

gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
print("GPU:", gpu)
assert "T4" in gpu, "Every reported number must come from a T4. Change the runtime to T4."

SHARED = "/content/drive/MyDrive/FYP_SR_SHARED"
B = pathlib.Path(SHARED) / "phase5"
man = json.loads((B / "MANIFEST.json").read_text())
def sha256(p, blk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blk), b""):
            h.update(b)
    return h.hexdigest()
for rel, meta in man.items():
    p = B / rel
    assert p.exists() and p.stat().st_size == meta["bytes"], f"{rel}: missing or incomplete; wait for Drive to finish syncing, then rerun"
    if sha256(p) != meta["sha256"]:
        raise RuntimeError(f"{rel}: checksum mismatch")
print("bundle verified:", len(man), "files")
''')

code('''
# 2. unpack into /content/work (code first: it overwrites nothing else)
WORK = "/content/work"
os.makedirs(WORK, exist_ok=True)
for z in man:
    if z.endswith(".zip"):
        zipfile.ZipFile(B / z).extractall(WORK)
print(sorted(os.listdir(WORK)))
''')

code('''
# 3. packages. Colab already has torch; add onnxruntime-gpu and the small dependencies, then confirm ONNX Runtime really uses the GPU.
sh("pip -q install onnxruntime-gpu scipy pandas pyyaml opencv-python-headless", tail=3)
import importlib
code_ = ("import onnxruntime as ort, sys; "
         "s = ort.InferenceSession(sys.argv[1], providers=['CUDAExecutionProvider', 'CPUExecutionProvider']); "
         "print(ort.__version__, s.get_providers())")
r = sh(f"python -c \\"{code_}\\" {WORK}/models/detector/det_10g.onnx", check=False, tail=3)
assert "CUDAExecutionProvider" in r.stdout.split("[", 1)[-1].split(",")[0], (
    "ONNX Runtime is not using the GPU (CPU timing would be meaningless). Stop and report the printed line above.")
''')

code('''
# 4. SHA-256 of the detector weights against the lab PC's files
exp = json.loads((B / "weights_sha256.json").read_text())
for rel, h in exp.items():
    assert sha256(f"{WORK}/{rel}") == h, f"{rel}: weights differ from the lab PC's"
print("detector weights verified")
''')

code('''
# 5. cross-device check (SCRFD keypoints on 100 TinyFace probes vs the lab PC); stops the notebook on failure
r = sh("python scripts/video_detector_study.py --job xcheck --xcheck-dir xcheck --root /content/work", check=False, cwd=WORK, tail=25)
assert r.returncode == 0, "cross-device check FAILED: do not continue; report to the lab PC owner"
''')

code('''
# 6. the study: resumable chunks, written straight to Drive (a rerun skips finished chunks)
OUT = f"{SHARED}/phase5/results"
os.makedirs(OUT, exist_ok=True)
# send the chunks to Drive by pointing the output folder there: the config's paths.out is relative to --root, so mirror it
os.makedirs(f"{WORK}/outputs/video", exist_ok=True)
if not os.path.islink(f"{WORK}/outputs/video/detstudy"):
    for f in ("xcheck.json", "device.json"):       # keep the cross-check evidence next to the results
        if os.path.exists(f"{WORK}/outputs/video/detstudy/{f}"):
            shutil.copy(f"{WORK}/outputs/video/detstudy/{f}", f"{OUT}/{f}")
    shutil.rmtree(f"{WORK}/outputs/video/detstudy", ignore_errors=True)
    os.symlink(OUT, f"{WORK}/outputs/video/detstudy")
p = subprocess.Popen(["python", "-u", "scripts/video_detector_study.py", "--job", "all", "--root", WORK], cwd=WORK,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if "chunk" in line or "runtime" in line or "done" in line or "Error" in line or "error" in line:
        print(line.rstrip())
p.wait()
print("exit code", p.returncode)
assert p.returncode == 0, "the study did not finish: rerun this cell (finished chunks are kept)"
open(f"{OUT}/DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    p = ROOT / "colab/phase5_detector_colab.ipynb"
    p.write_text(json.dumps(nb, indent=1))
    print("wrote", p)
    if a.publish:
        d = SHARED / "phase5"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, d / p.name)
        print("published", d / p.name)


if __name__ == "__main__":
    main()
