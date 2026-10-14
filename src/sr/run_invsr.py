"""InvSR runner (executes in envs/invsr). Reads native crops, stretches to pre_size^2 (bicubic), runs InvSR
(x4) with chopping_size == pre_size so each crop is ONE unpadded pass, writes PNG outputs and timing.json.

    envs/invsr/Scripts/python.exe src/sr/run_invsr.py --input-dir D --out-dir O --pre-size 128 --steps 1 --seed 0
"""
import argparse
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "InvSR"))
import torch  # noqa: E402

from src.sr import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pre-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--file-list", default=None, help="text file of input file names (one per line) to process, in order")
    ap.add_argument("--noise-predictor", default="models/weights/invsr/InvSR/noise_predictor_sd_turbo_v5.pth")
    ap.add_argument("--sd-path", default="models/weights/invsr/sd-turbo")
    a = ap.parse_args()

    import os
    os.chdir(ROOT / "models" / "InvSR")  # repo-relative config paths
    from inference_invsr import get_configs
    from sampler_invsr import InvSamplerSR
    args = Namespace(in_path="", out_path="", bs=1, chopping_bs=8, timesteps=None, num_steps=a.steps,
                     cfg_path="./configs/sample-sd-turbo.yaml", sd_path=str(ROOT / a.sd_path),
                     started_ckpt_path=str(ROOT / a.noise_predictor), tiled_vae=True, color_fix="", chopping_size=a.pre_size,
                     started_step=None)
    cfg = get_configs(args)
    sampler = InvSamplerSR(cfg)
    from utils import util_image
    files = C.list_images(ROOT / a.input_dir if not Path(a.input_dir).is_absolute() else a.input_dir, a.limit, a.file_list)
    out = Path(a.out_dir) if Path(a.out_dir).is_absolute() else ROOT / a.out_dir
    torch.cuda.reset_peak_memory_stats()
    tm = C.Timing(out)
    for f in files:
        if C.out_done(out, f.name):
            continue
        def go():
            sampler.setup_seed(a.seed)  # per-crop seed: output independent of processing order
            bgr = C.pre_upsample(C.read_bgr(f), a.pre_size)
            x = util_image.img2tensor(bgr[..., ::-1].astype("float32") / 255.0).cuda()
            res = sampler.sample_func(x).squeeze(0)               # h x w x 3 RGB [0,1]
            return (res[..., ::-1] * 255.0 + 0.5).clip(0, 255).astype("uint8")
        r = tm.run(f.name, go)
        if r is not None:
            C.write_png(out / f.name, r)
    s = tm.summary()
    s.update(model="InvSR", pre_size=a.pre_size, steps=a.steps, seed=a.seed, out_size=a.pre_size * 4, n_inputs=len(files))
    C.save_summary(out / "timing.json", s)
    print({k: v for k, v in s.items() if k != "errors"}, "errors:", s["errors"][:3])


if __name__ == "__main__":
    main()
