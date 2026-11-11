"""Generate colab/phase6_ytf_colab.ipynb (Appendix T1: the Phase 5 detector on the YouTube Faces sample, one Colab T4).

    envs/diag/Scripts/python.exe colab/build_phase6_ytf_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Appendix T1: the Phase 5 detector on YouTube Faces (Colab T4)

Runs SCRFD `det_10g` (the Phase 5 detector, official preprocessing) on a seeded sample of 5,000 YouTube Faces frames (500 people, 10 frames each) in two modes,
`full1280` (the Phase 5 choice) and `full640` (reference), and writes recall against the dataset's face boxes. **Inference only.** The numbers come from a T4.

**Before the first run:** the account that opens this notebook must see the folder `FYP_SR_SHARED` in My Drive (as the owner, or through *Add shortcut to My Drive*).
Then **Runtime > Change runtime type > T4 GPU** and **Run all**.

The notebook stops by itself if the GPU is not a T4, if a bundle file has not finished syncing or has a wrong checksum, if the detector weights differ from the lab PC's,
or if ONNX Runtime is not using the GPU. In that case do not continue: copy the message and send it to the team chat for the lab PC.
A rerun starts again from the beginning (the whole job takes minutes). Results go to `FYP_SR_SHARED/phase6_ytf/results/`. Nothing on the Drive is deleted or overwritten.
""")

code('''
# 1. Drive, GPU, bundle integrity
import os
os.environ["MPLBACKEND"] = "Agg"
from google.colab import drive
drive.mount("/content/drive")
import subprocess, json, hashlib, sys, time, zipfile, pathlib

def sh(cmd, check=True, tail=15, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    out = (r.stdout or "") + (r.stderr or "")
    print("\\n".join(out.strip().splitlines()[-tail:]))
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd[:120]}")
    return r

gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
print("GPU:", gpu)
assert "T4" in gpu, "These numbers must come from a T4. Change the runtime to T4."

SHARED = "/content/drive/MyDrive/FYP_SR_SHARED"
def sha256(p, blk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blk), b""):
            h.update(b)
    return h.hexdigest()
def verify(folder, only=None):
    man = json.loads((pathlib.Path(folder) / "MANIFEST.json").read_text())
    for rel, meta in man.items():
        if only and rel not in only:
            continue
        p = pathlib.Path(folder) / rel
        assert p.exists() and p.stat().st_size == meta["bytes"], f"{p}: missing or incomplete; wait for Drive to finish syncing, then rerun"
        if sha256(p) != meta["sha256"]:
            raise RuntimeError(f"{p}: checksum mismatch")
    return man
B6, B5 = pathlib.Path(SHARED) / "phase6_ytf", pathlib.Path(SHARED) / "phase5"
man6 = verify(B6)
verify(B5, only={"detectors.zip"})
print("bundles verified:", list(man6), "+ phase5/detectors.zip")
''')

code('''
# 2. unpack into /content/work
WORK = "/content/work"
os.makedirs(WORK, exist_ok=True)
for z in (B5 / "detectors.zip", B6 / "code_ytf.zip", B6 / "ytf_sample.zip"):
    zipfile.ZipFile(z).extractall(WORK)
print(sorted(os.listdir(WORK)), len(os.listdir(f"{WORK}/ytf_sample/frames")), "frames")
''')

code('''
# 3. packages; confirm ONNX Runtime really uses the GPU
sh("pip -q install onnxruntime-gpu scipy pandas pyyaml opencv-python-headless", tail=3)
code_ = ("import onnxruntime as ort, sys; "
         "s = ort.InferenceSession(sys.argv[1], providers=['CUDAExecutionProvider', 'CPUExecutionProvider']); "
         "print(ort.__version__, s.get_providers())")
r = sh(f"python -c \\"{code_}\\" {WORK}/models/detector/det_10g.onnx", check=False, tail=3)
assert "CUDAExecutionProvider" in r.stdout.split("[", 1)[-1].split(",")[0], "ONNX Runtime is not using the GPU: stop and report the printed line above."
''')

code('''
# 4. detector weights against the lab PC
exp = json.loads((B5 / "weights_sha256.json").read_text())
rel = "models/detector/det_10g.onnx"
assert sha256(f"{WORK}/{rel}") == exp[rel], "det_10g.onnx differs from the lab PC's"
print("detector weights verified")
''')

code('''
# 5. the test: results straight to Drive
OUT = f"{SHARED}/phase6_ytf/results"
os.makedirs(OUT, exist_ok=True)
p = subprocess.Popen(["python", "-u", "scripts/video_appendix_ytf.py", "detect", "--root", WORK, "--out", OUT], cwd=WORK,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line.rstrip())
p.wait()
print("exit code", p.returncode)
assert p.returncode == 0, "the run did not finish: report the output above"
open(f"{OUT}/DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
''')

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
p = ROOT / "colab/phase6_ytf_colab.ipynb"
p.write_text(json.dumps(nb, indent=1))
print("wrote", p)
