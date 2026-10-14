"""Build the Phase 3 remote-worker bundle and (optionally) publish it to the shared Drive folder.

    envs/diag/Scripts/python.exe scripts/sr_colab_bundle.py                 # build into outputs/sr/colab_bundle/
    envs/diag/Scripts/python.exe scripts/sr_colab_bundle.py --publish       # ... and copy to G:/My Drive/FYP_SR_SHARED

Bundle: code.zip, subset_inputs.zip, recognizers.zip, xcheck.zip, locks/{osdface,invsr}.txt, weights_sha256.json,
MANIFEST.json (size + sha256 of every file; the notebook refuses a file that has not finished syncing),
and ../jobs/<job_id>/job.json for each job.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/sr/colab_bundle"
SHARED = Path("G:/My Drive/FYP_SR_SHARED")

def phase_b_jobs():
    """Phase B: OSDFace eval at the selected setting in 3 crop ranges + seeds 1 and 2 on the tuning subset.
    The setting is read from the merged selection, so the jobs cannot disagree with it."""
    sel = json.loads((ROOT / "outputs/sr/results/selection_osdface.json").read_text())[0]["setting"]
    pre = int(sel.replace("pre", ""))
    n = len(pd.read_csv(ROOT / "outputs/sr/subset/eval_images.csv"))
    cuts = [0, 6640, 13280, n]  # sized from measured T4 speed so each machine finishes in about 4.2 h
    args = dict(pre=pre, sd_path="models/weights/osdface/sd21base")
    jobs = [dict(job_id=f"eval__osdface__{sel}__seed0__r{cuts[k]:05d}-{cuts[k + 1]:05d}", arm="osdface", part="eval", seed=0,
                 machine=f"T4-{k + 1}", phase="B", range=[cuts[k], cuts[k + 1]], args=args) for k in range(3)]
    jobs += [dict(job_id=f"tuning__osdface__{sel}__seed{sd}", arm="osdface", part="tuning", seed=sd, machine=f"T4-{sd}", phase="B", args=args)
             for sd in (1, 2)]
    return jobs


RECOGNIZER_FILES = ["arcface_iresnet100.pth", "adaface_ir101_ms1mv2.ckpt", "magface_iresnet100_MS1MV2.pth", "CurricularFace_Backbone.pth"]
# expected SHA-256 prefixes (MODEL_INVENTORY.md); the notebook checks each downloaded / copied file against these
WEIGHT_SHA = {
    "models/weights/arcface_iresnet100.pth": "a566a62357f0", "models/weights/adaface_ir101_ms1mv2.ckpt": "4a26839460d8",
    "models/weights/magface_iresnet100_MS1MV2.pth": "cfeba792dada", "models/weights/CurricularFace_Backbone.pth": "031ea0f7ba5f",
    "models/weights/osdface/official/associate_2.ckpt": "d51ff7f0ddb0bd84",
    "models/weights/osdface/official/embedding_change_weights.pth": "3b13ab83fed678ed",
    "models/weights/osdface/official/pytorch_lora_weights.safetensors": "bf7095b4d79f3e99",
    "models/weights/osdface/sd21base/unet/diffusion_pytorch_model.safetensors": "6dfae3e5f7d459b5",
    "models/weights/osdface/sd21base/text_encoder/model.safetensors": "cce6febb0b6d876e",
    "models/weights/osdface/sd21base/vae/diffusion_pytorch_model.safetensors": "a1d993488569e928",
    "models/weights/invsr/InvSR/noise_predictor_sd_turbo_v5.pth": "e1a03b0f447d6f26",
    "models/weights/invsr/sd-turbo/unet/diffusion_pytorch_model.safetensors": "438da6db739c8651",
}
SKIP_LOCK = ("torch", "torchvision", "torchaudio", "xformers", "triton", "pywin32", "win32-setctime", "pywinpty", "colorama", "nvidia-")


def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def zip_files(zpath, items):
    """items: list of (disk_path, arcname). Stored (no compression): PNG / weights do not compress."""
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for p, arc in items:
            z.write(p, arc)


def lock_file(venv, dest):
    out = subprocess.run([str(ROOT / "envs/diag/Scripts/uv.exe"), "pip", "freeze", "--python", str(ROOT / venv / "Scripts/python.exe")],
                         capture_output=True, text=True).stdout.splitlines()
    keep = [l for l in out if l.strip() and not l.startswith(("-e ", "#")) and not l.lower().startswith(SKIP_LOCK) and " @ " not in l]
    dest.write_text("\n".join(keep) + "\n")
    return len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "locks").mkdir(parents=True)
    (OUT / "jobs").mkdir()
    # code.zip
    items = []
    for sub in ("src",):
        for p in (ROOT / sub).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                items.append((p, p.relative_to(ROOT).as_posix()))
    for f in ("scripts/sr_colab_job.py", "scripts/sr_colab_xcheck.py", "configs/sr.yaml"):
        items.append((ROOT / f, f))
    for part in ("tuning", "eval"):
        for kind in ("images", "pairs"):
            items.append((ROOT / f"outputs/sr/subset/{part}_{kind}.csv", f"outputs/sr/subset/{part}_{kind}.csv"))
    zip_files(OUT / "code.zip", items)
    # subset_inputs.zip
    items = [(p, p.relative_to(ROOT).as_posix()) for part in ("tuning", "eval") for p in (ROOT / f"outputs/sr/subset/{part}_input").glob("*.png")]
    zip_files(OUT / "subset_inputs.zip", items)
    # recognizers.zip
    zip_files(OUT / "recognizers.zip", [(ROOT / "models/weights" / f, f"models/weights/{f}") for f in RECOGNIZER_FILES])
    # xcheck.zip: smoke crops, PC outputs, 200 direct crops + PC embeddings
    xc = OUT / "_xcheck"
    items = []
    for p in sorted((ROOT / "outputs/sr/smoke/input").glob("*.png")):
        items.append((p, f"xcheck/smoke_inputs/{p.name}"))
    for arm, d in (("osdface", "outputs/sr/smoke/osdface/pre128_seed0"), ("invsr", "outputs/sr/smoke/invsr/pre128_s1_ref_percrop")):
        for p in sorted((ROOT / d).glob("*.png")):
            items.append((p, f"xcheck/pc_{arm}/{p.name}"))
    ev = pd.read_csv(ROOT / "outputs/sr/subset/eval_images.csv")
    names = ev["file"].tolist()[:200]
    for n in names:
        items.append((ROOT / "outputs/sr/subset/eval_input" / n, f"xcheck/direct_inputs/{n}"))
    xc.mkdir()
    (xc / "direct_names.json").write_text(json.dumps(names))
    items.append((xc / "direct_names.json", "xcheck/direct_names.json"))
    for r in ("arcface", "adaface", "magface", "curricularface"):
        arr = np.load(ROOT / f"outputs/sr/emb/eval/direct/resize/seed0/eval__{r}.npy")[:200]
        np.save(xc / f"direct_emb_{r}.npy", arr)
        items.append((xc / f"direct_emb_{r}.npy", f"xcheck/direct_emb_{r}.npy"))
    zip_files(OUT / "xcheck.zip", items)
    shutil.rmtree(xc)
    # locks, weights hashes, jobs
    n_os = lock_file("envs/osdface", OUT / "locks/osdface.txt")
    n_in = lock_file("envs/invsr", OUT / "locks/invsr.txt")
    (OUT / "weights_sha256.json").write_text(json.dumps(WEIGHT_SHA, indent=1))
    for j in phase_b_jobs():
        d = OUT / "jobs" / j["job_id"]
        d.mkdir()
        (d / "job.json").write_text(json.dumps(j, indent=1))
    # manifest
    man = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            man[p.relative_to(OUT).as_posix()] = dict(bytes=p.stat().st_size, sha256=sha256(p))
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    tot = sum(v["bytes"] for v in man.values()) / 2 ** 30
    print(f"built {len(man)} files, {tot:.2f} GB in {OUT}; locks: osdface {n_os} pkgs, invsr {n_in} pkgs")
    if a.publish:
        # data files first, MANIFEST last: the notebook only trusts files listed in a MANIFEST that exists
        (SHARED / "bundle").mkdir(parents=True, exist_ok=True)
        for rel in man:
            dst = SHARED / ("bundle/" + rel if not rel.startswith("jobs/") else rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.stat().st_size == man[rel]["bytes"] and sha256(dst) == man[rel]["sha256"]:
                continue  # unchanged: do not rewrite (avoids re-uploading large files)
            shutil.copy2(OUT / rel, dst)
        shutil.copy2(OUT / "MANIFEST.json", SHARED / "bundle/MANIFEST.json")
        print("published to", SHARED)


if __name__ == "__main__":
    main()
