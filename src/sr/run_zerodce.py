"""Zero-DCE++ low-light enhancement (Epoch99.pth), same-size output, per crop on the GPU (about 10k parameters).

    envs/zerodce/Scripts/python.exe src/sr/run_zerodce.py --input-dir D --out-dir O --params '{"arm": "zerodce", "s": 1}' --device cuda
params.s = scale_factor (1 = full-resolution curve estimation, no down/up-sampling). Resumable, atomic writes, failures logged.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.sr import common as C  # noqa: E402
from src.sr.vendor.zerodce.model import enhance_net_nopool  # noqa: E402

WEIGHTS = ROOT / "models/weights/zerodce/Epoch99.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    p = json.loads(a.params)
    inp = Path(a.input_dir) if Path(a.input_dir).is_absolute() else ROOT / a.input_dir
    out = Path(a.out_dir) if Path(a.out_dir).is_absolute() else ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    net = enhance_net_nopool(int(p.get("s", 1))).to(dev).eval()
    net.load_state_dict(torch.load(WEIGHTS, map_location=dev, weights_only=True))
    tm = C.Timing(out, "cuda" if dev == "cuda" else "cpu")

    def one(f):
        rgb = C.read_bgr(f)[..., ::-1].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            y, _ = net(x)
        y = (y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        C.write_png(out / f.name, np.ascontiguousarray(y[..., ::-1]))

    for f in C.list_images(inp, a.limit):
        if C.out_done(out, f.name):
            continue
        tm.run(f.name, lambda f=f: one(f))
    s = tm.summary()
    s.update(model="zerodce++", params=p, device=dev, weights=str(WEIGHTS.relative_to(ROOT)))
    C.save_summary(out / "timing.json", s)
    print("zerodce done", s["n_ok"], "ok", s["n_failed"], "failed")


if __name__ == "__main__":
    main()
