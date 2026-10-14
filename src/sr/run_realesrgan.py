"""Real-ESRGAN x4plus baseline runner (executes in envs/realesrgan; RRDBNet vendored from BasicSR).
pre_size 0 = feed the native crop (x4 of its own size); otherwise bicubic-stretch to pre_size^2 first. Deterministic.

    envs/realesrgan/Scripts/python.exe src/sr/run_realesrgan.py --input-dir D --out-root O --pre-sizes 0 64 128
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.sr import common as C  # noqa: E402
from src.sr.vendor.rrdbnet import RRDBNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--pre-sizes", type=int, nargs="+", default=[0])
    ap.add_argument("--out-template", default=None, help="e.g. outputs/sr/restored/tuning/esrgan_pre{pre}/seed0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--weights", default="models/weights/realesrgan/RealESRGAN_x4plus.pth")
    a = ap.parse_args()
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    net.load_state_dict(torch.load(ROOT / a.weights, map_location="cpu", weights_only=True)["params_ema"], strict=True)
    net.eval().cuda()
    files = C.list_images(ROOT / a.input_dir if not Path(a.input_dir).is_absolute() else a.input_dir, a.limit)
    root = Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root
    for S in a.pre_sizes:
        out = (ROOT / a.out_template.format(pre=S)) if a.out_template else root / f"pre{S}"
        torch.cuda.reset_peak_memory_stats()
        tm = C.Timing(out)
        for f in files:
            if C.out_done(out, f.name):
                continue
            def go():
                bgr = C.read_bgr(f)
                if S:
                    bgr = C.pre_upsample(bgr, S)
                x = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
                with torch.no_grad():
                    y = net(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                return (y[..., ::-1] * 255.0 + 0.5).astype("uint8")
            r = tm.run(f.name, go)
            if r is not None:
                C.write_png(out / f.name, r)
        s = tm.summary()
        s.update(model="Real-ESRGAN x4plus", pre_size=S, out_size="4x" if not S else S * 4, n_inputs=len(files))
        C.save_summary(out / "timing.json", s)
        print({k: v for k, v in s.items() if k != "errors"}, "errors:", s["errors"][:3], flush=True)


if __name__ == "__main__":
    main()
