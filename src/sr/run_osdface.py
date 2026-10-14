"""OSDFace runner (executes in envs/osdface). Pre-upsample each native crop to pre_size^2 (bicubic); the repo's own
code then bilinear-resizes square inputs to 512 and runs the one-step model (timestep 399, merged LoRA, fp32 as in
the official test.sh). Each crop is seeded independently (VAE posterior sampling is stochastic).

    envs/osdface/Scripts/python.exe src/sr/run_osdface.py --input-dir D --out-root O --pre-sizes 128 256 512 --seed 0
"""
import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "models" / "OSDFace"
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.sr import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-root", required=True, help="writes <out-root>/pre<S>_seed<k>/")
    ap.add_argument("--pre-sizes", type=int, nargs="+", default=[512])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-template", default=None, help="e.g. outputs/sr/restored/tuning/osdface_pre{pre}/seed{seed}")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--file-list", default=None, help="text file of input file names (one per line) to process, in order")
    ap.add_argument("--weights", default="models/weights/osdface/official")
    ap.add_argument("--sd-path", default="models/weights/osdface/sd21base")
    a = ap.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import infer as osd
    import torch.nn.functional as Fun
    import torchvision.transforms.functional as F

    wdir = ROOT / a.weights
    args = Namespace(pretrained_model_name_or_path=str(ROOT / a.sd_path), seed=a.seed, process_size=512, ckpt_path=str(wdir),
                     mixed_precision="fp32", img_encoder_weight=str(wdir / "associate_2.ckpt"), gpu_ids=[0],
                     cat_prompt_embedding=False, use_pos_embedding=False, use_att_pool=False, merge_lora=True,
                     lora_rank=16, lora_alpha=16.0)
    unet = osd.merge_Unet(args)
    model = osd.OSDFace_test(args, 0, unet)
    files = C.list_images(ROOT / a.input_dir if not Path(a.input_dir).is_absolute() else a.input_dir, a.limit, a.file_list)
    root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    for S in a.pre_sizes:
        out = (ROOT / a.out_template.format(pre=S, seed=a.seed)) if a.out_template else root / f"pre{S}_seed{a.seed}"
        torch.cuda.reset_peak_memory_stats()
        tm = C.Timing(out)
        for f in files:
            if C.out_done(out, f.name):
                continue
            def go():
                torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed)
                from PIL import Image
                pre = C.pre_upsample(C.read_bgr(f), S)
                lq = F.to_tensor(Image.fromarray(pre[..., ::-1]).convert("RGB")).unsqueeze(0).cuda() * 2 - 1
                if lq.shape[2] == lq.shape[3]:
                    lq = Fun.interpolate(lq, (512, 512), mode="bilinear", align_corners=True)
                with torch.no_grad():
                    o = model(lq)
                return (o[0].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255.0 + 0.5).clip(0, 255).astype("uint8")
            r = tm.run(f.name, go)
            if r is not None:
                C.write_png(out / f.name, r)
        s = tm.summary()
        s.update(model="OSDFace", pre_size=S, seed=a.seed, out_size=512, n_inputs=len(files))
        C.save_summary(out / "timing.json", s)
        print({k: v for k, v in s.items() if k != "errors"}, "errors:", s["errors"][:3], flush=True)


if __name__ == "__main__":
    main()
