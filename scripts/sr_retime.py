"""Isolated speed measurement of every selected SR setting on the lab PC (the report's speed table).

    envs/diag/Scripts/python.exe scripts/sr_retime.py [--n 200]

Runs each arm's runner alone (nothing else should use the GPU) on the same seeded sample of eval crops
(stratified by dataset/split), seed 0, and records seconds per crop (median, mean, p90; the first 5 crops are
excluded as warm-up) and peak allocated VRAM. Output: outputs/sr/results/timing_isolated.csv. Restored images go to
a scratch folder and are deleted afterwards.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sr_pipeline as P  # noqa: E402

WARMUP = 5


def runner_cmd(cfg, v, inp, out):
    a = v["arm"]
    arm = cfg["arms"][a]
    cmd = [str(ROOT / arm["python"]), "-W", "ignore", str(ROOT / arm["runner"]), "--input-dir", str(inp)]
    tmp = ["--out-root", str(out.parent / "_tmp")]
    if a == "bicubic":
        return cmd + tmp + ["--sizes", str(v["pre"]), "--out-template", str(out)]
    if a == "esrgan":
        return cmd + tmp + ["--pre-sizes", str(v["pre"]), "--out-template", str(out)]
    if a == "invsr":
        return cmd + ["--out-dir", str(out), "--pre-size", str(v["pre"]), "--steps", str(v["steps"]), "--seed", "0"]
    if a == "tadsr":
        return cmd + tmp + ["--pre-size", str(v["pre"]), "--timesteps", str(v["t"]), "--seed", "0", "--out-template", str(out)]
    return cmd + tmp + ["--pre-sizes", str(v["pre"]), "--seed", "0", "--out-template", str(out), "--sd-path", cfg["osdface_sd_path"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/sr.yaml"))
    res = ROOT / "outputs/sr/results"
    # variants: each arm's selected setting, plus the InvSR pre-128 sensitivity setting
    variants = []
    for arm in ("bicubic", "esrgan", "invsr", "tadsr", "osdface"):
        f = res / f"selection_{arm}.json"
        if f.exists() and json.loads(f.read_text()):
            variants.append(P.find_variant(cfg, arm, json.loads(f.read_text())[0]["setting"]))
    variants.append(P.find_variant(cfg, "invsr", "pre128_st1"))
    # seeded sample, proportional to dataset/split
    d = pd.read_csv(ROOT / "outputs/sr/subset/eval_images.csv")
    pick = d.groupby(["dataset", "split"], group_keys=False).apply(
        lambda g: g.sample(max(1, round(a.n * len(g) / len(d))), random_state=cfg["seed"]))
    work = ROOT / "outputs/sr/retime"
    if work.exists():
        shutil.rmtree(work)
    inp = work / "input"
    inp.mkdir(parents=True)
    for f in pick["file"]:
        shutil.copy2(ROOT / "outputs/sr/subset/eval_input" / f, inp / f)
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    rows = []
    for v in variants:
        arm, setting = P.vid(v)
        out = work / f"{arm}__{setting}"
        print(f"timing {arm}__{setting} on {len(pick)} crops ...", flush=True)
        r = subprocess.run(runner_cmd(cfg, v, inp, out), cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        t = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
        sec = np.array(t.get("sec_list") or [])[WARMUP:]
        row = dict(arm=arm, setting=setting, n_crops=len(pick), n_timed=len(sec), device=gpu, returncode=r.returncode,
                   peak_vram_mb=t.get("peak_vram_mb"), n_failed=t.get("n_failed"))
        if len(sec):
            row.update(sec_per_crop_median=float(np.median(sec)), sec_per_crop_mean=float(sec.mean()), sec_per_crop_p90=float(np.quantile(sec, 0.9)))
        else:  # bicubic writes only a mean (no per-crop list)
            row.update(sec_per_crop_mean=t.get("sec_per_crop_mean"), note="mean over all crops incl. file I/O; no per-crop list")
        if r.returncode != 0:
            row["error_tail"] = (r.stderr or "")[-500:]
        rows.append(row)
        print({k: row.get(k) for k in ("arm", "setting", "sec_per_crop_median", "sec_per_crop_mean", "peak_vram_mb", "returncode")}, flush=True)
    pd.DataFrame(rows).to_csv(res / "timing_isolated.csv", index=False)
    shutil.rmtree(work, ignore_errors=True)
    print("wrote", res / "timing_isolated.csv")


if __name__ == "__main__":
    main()
