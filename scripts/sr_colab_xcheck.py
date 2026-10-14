"""Cross-device check (run on each Colab before the job): does this machine reproduce the PC's numbers?

    python scripts/sr_colab_xcheck.py --arm osdface --xcheck <work>/repo/xcheck --out <shared>/jobs/<job_id>/xcheck.json

1. Direct-feed embeddings of 200 native crops vs the PC's embeddings (max |delta cos|, same code and weights).
2. Restore the 75 smoke crops with the arm's runner (same setting as the PC smoke test) and compare with the PC's
   restored crops: mean absolute pixel difference (0-255) and the cosine between the embeddings of both versions.
3. Seconds per crop on this GPU.
Exit code 2 when the mean embedding cosine of restored crops is < --halt-below (default 0.99).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

os.environ["MPLBACKEND"] = "Agg"  # Colab sets an inline backend that does not exist in the venv; children inherit this
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recognizer import Recognizer  # noqa: E402
from src.sr import common as C  # noqa: E402

RECS = ("arcface", "adaface", "magface", "curricularface")


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["osdface", "invsr"])
    ap.add_argument("--xcheck", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--halt-below", type=float, default=0.99)
    a = ap.parse_args()
    xc = Path(a.xcheck).resolve()
    cfg = yaml.safe_load(open(ROOT / "configs/sr.yaml"))
    import torch
    res = dict(arm=a.arm, gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda)
    recs = {r: Recognizer(r, ROOT / cfg["recognizers"][r], (112, 112), "bilinear", True, 128, "cuda") for r in RECS}

    # 1. direct-feed embeddings
    names = json.loads((xc / "direct_names.json").read_text())
    imgs = [C.read_bgr(xc / "direct_inputs" / n) for n in names]
    dres = {}
    for r, rec in recs.items():
        rec.pre["interp"] = "bilinear"
        e = rec.get_embedding(imgs)
        pc = np.load(xc / f"direct_emb_{r}.npy")
        dres[r] = float(np.abs((l2(e) * l2(pc)).sum(1) - 1).max())
    res["direct_max_abs_dcos"] = dres

    # 2. restoration of the smoke crops
    smoke = xc / "smoke_inputs"
    work = Path(a.work).resolve() / "xcheck_out"
    work.mkdir(parents=True, exist_ok=True)
    pc_out = xc / f"pc_{a.arm}"
    files = sorted(p.name for p in smoke.glob("*.png"))
    nf = work / "names.txt"
    nf.write_text("\n".join(files), encoding="utf-8")
    if a.arm == "osdface":
        cmd = [sys.executable, "-W", "ignore", str(ROOT / "src/sr/run_osdface.py"), "--input-dir", str(smoke), "--file-list", str(nf),
               "--out-root", str(work / "_tmp"), "--pre-sizes", "128", "--seed", "0", "--out-template", str(work / "out"),
               "--sd-path", "models/weights/osdface/sd21base"]
    else:
        cmd = [sys.executable, "-W", "ignore", str(ROOT / "src/sr/run_invsr.py"), "--input-dir", str(smoke), "--file-list", str(nf),
               "--out-dir", str(work / "out"), "--pre-size", "128", "--steps", "1", "--seed", "0"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    res["restore_returncode"] = r.returncode
    if r.returncode != 0:  # keep the runner's own error so the failure can be diagnosed from xcheck.json alone
        res["restore_error_tail"] = ((r.stdout or "") + "\n--- stderr ---\n" + (r.stderr or ""))[-4000:]
        print(res["restore_error_tail"])
    res["restore_wall_s"] = round(time.time() - t0, 1)
    out = work / "out"
    tj = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
    res["sec_per_crop_median"] = tj.get("sec_per_crop_median")
    res["peak_vram_mb"] = tj.get("peak_vram_mb")
    have = [n for n in files if (out / n).exists()]
    res["n_restored"], res["n_expected"] = len(have), len(files)
    if have:
        mad = [float(np.abs(C.read_bgr(out / n).astype(np.float32) - C.read_bgr(pc_out / n).astype(np.float32)).mean()) for n in have]
        res["mean_abs_pixel_diff_0_255"] = float(np.mean(mad))
        coss = {}
        for rn, rec in recs.items():
            rec.pre["interp"] = "area"
            e1 = rec.get_embedding([C.read_bgr(out / n) for n in have])
            e2 = rec.get_embedding([C.read_bgr(pc_out / n) for n in have])
            c = (l2(e1) * l2(e2)).sum(1)
            coss[rn] = dict(mean=float(c.mean()), min=float(c.min()))
        res["embedding_cos_vs_pc"] = coss
        m = float(np.mean([v["mean"] for v in coss.values()]))
        res["mean_embedding_cos"] = m
        res["pass"] = bool(m >= a.halt_below and len(have) == len(files) and max(dres.values()) < 5e-3)
    else:
        res["pass"] = False
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["pass"] else 2)


if __name__ == "__main__":
    main()
