"""TADSR runner (executes in envs/tadsr). Same pre-upsample as the other arms (bicubic to pre_size^2), then the
repo's own pipeline: x4 resize, RAM/DAPE tagging, one-step TADSR at the given timestep, adain colour fix.
Several timesteps can be swept in one model load. Each crop is seeded independently.

    envs/tadsr/Scripts/python.exe src/sr/run_tadsr.py --input-dir D --out-root O --pre-size 128 --timesteps 250 500 750 --seed 0
"""
import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "models" / "TADSR"
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.sr import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-root", required=True, help="writes <out-root>/pre<S>_t<T>_seed<k>/")
    ap.add_argument("--pre-size", type=int, default=128)
    ap.add_argument("--timesteps", type=int, nargs="+", default=[500])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-template", default=None, help="e.g. outputs/sr/restored/tuning/tadsr_pre{pre}_t{t}/seed{seed}")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from tadsr import TADSR_test
    from my_utils.wavelet_color_fix import adain_color_fix
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    from torchvision import transforms

    w = REPO / "preset" / "weights"
    args = Namespace(pretrained_model_name_or_path=str(w), seed=a.seed, process_size=4 * a.pre_size, upscale=4, align_method="adain",
                     tadsr_path=str(w / "tadsr.pkl"), prompt="", ram_path=str(w / "ram_swin_large_14m.pth"),
                     ram_ft_path=str(w / "DAPE.pth"), mixed_precision="fp16", merge_and_unload_lora=False,
                     vae_decoder_tiled_size=224, vae_encoder_tiled_size=1024, latent_tiled_size=96, latent_tiled_overlap=32, timesteps=500)
    model = TADSR_test(args)
    dape = ram(pretrained=args.ram_path, pretrained_condition=args.ram_ft_path, image_size=384, vit="swin_l").eval().to("cuda").half()
    to_t = transforms.ToTensor()
    ram_t = transforms.Compose([transforms.Resize((384, 384)), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    in_dir = Path(a.input_dir) if Path(a.input_dir).is_absolute() else ROOT / a.input_dir
    files = C.list_images(in_dir, a.limit)
    root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    for T in a.timesteps:
        out = (ROOT / a.out_template.format(pre=a.pre_size, t=T, seed=a.seed)) if a.out_template else root / f"pre{a.pre_size}_t{T}_seed{a.seed}"
        torch.cuda.reset_peak_memory_stats()
        tm = C.Timing(out)
        for f in files:
            if C.out_done(out, f.name):
                continue
            def go():
                torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed)
                pre = C.pre_upsample(C.read_bgr(f), a.pre_size)
                img = Image.fromarray(pre[..., ::-1]).convert("RGB")
                img = img.resize((img.size[0] * 4, img.size[1] * 4))
                img = img.resize((img.size[0] - img.size[0] % 8, img.size[1] - img.size[1] % 8), Image.LANCZOS)
                lq = to_t(img).unsqueeze(0).cuda()
                tags = inference(ram_t(lq).half(), dape)
                model.timesteps = T  # upstream tiled-latent branch reads self.timesteps, which the repo never sets
                with torch.no_grad():
                    o = model(lq * 2 - 1, prompt=f"{tags[0]}, ,", timesteps=T)
                    pil = transforms.ToPILImage()(o[0].cpu() * 0.5 + 0.5)
                    pil = adain_color_fix(target=pil, source=img)
                return np.asarray(pil)[..., ::-1].copy(), tags[0]
            r = tm.run(f.name, go)
            if r is not None:
                C.write_png(out / f.name, r[0])
                (out / "tags").mkdir(exist_ok=True)
                (out / "tags" / f"{f.stem}.txt").write_text(r[1], encoding="utf-8")
        s = tm.summary()
        s.update(model="TADSR", pre_size=a.pre_size, timestep=T, seed=a.seed, out_size=a.pre_size * 4, n_inputs=len(files))
        C.save_summary(out / "timing.json", s)
        print({k: v for k, v in s.items() if k != "errors"}, "errors:", s["errors"][:3], flush=True)


if __name__ == "__main__":
    main()
