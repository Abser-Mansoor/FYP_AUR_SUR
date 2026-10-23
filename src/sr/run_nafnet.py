"""NAFNet-GoPro-width64 deblurring, batched on the GPU.

    envs/nafnet/Scripts/python.exe src/sr/run_nafnet.py --input-dir D --out-dir O --params '{"arm": "nafnet", "pre": 128}' --device cuda
params.pre = 0: native size (reflect-padded to a multiple of 16, cropped back; batch 32).
params.pre = N: bicubic to N x N first, then NAFNet (batch 8). Out-of-memory: the batch is halved and retried (each retry is logged in
timing.json "oom_retries"). A non-OOM batch error falls back to single crops so one bad crop never drops a batch.
Resumable, atomic writes, failures logged.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.sr import common as C  # noqa: E402
from src.sr.vendor.nafnet.NAFNet_arch import NAFNet  # noqa: E402

WEIGHTS = ROOT / "models/weights/nafnet/NAFNet-GoPro-width64.pth"
MULT = 16  # 2 ** 4 encoder stages


def load_net(dev):
    net = NAFNet(img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    net.load_state_dict(sd["params"] if "params" in sd else sd, strict=True)
    return net.to(dev).eval()


def prep(f, pre):
    """-> RGB uint8 [H', W', 3] reflect-padded to a multiple of 16, and the unpadded (h, w)."""
    bgr = C.read_bgr(f)
    if pre:
        bgr = C.pre_upsample(bgr, pre)
    h, w = bgr.shape[:2]
    ph, pw = (-h) % MULT, (-w) % MULT
    rgb = bgr[..., ::-1]
    if ph or pw:
        rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")  # numpy reflects repeatedly if the pad exceeds the size
    return np.ascontiguousarray(rgb), (h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    a = ap.parse_args()
    p = json.loads(a.params)
    pre = int(p.get("pre", 0))
    batch0 = a.batch or (8 if pre else 32)
    inp = Path(a.input_dir) if Path(a.input_dir).is_absolute() else ROOT / a.input_dir
    out = Path(a.out_dir) if Path(a.out_dir).is_absolute() else ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    net = load_net(dev)
    tm = C.Timing(out, "cuda" if dev == "cuda" else "cpu")
    retries = []
    if (out / "timing.json").exists():
        retries = json.loads((out / "timing.json").read_text()).get("oom_retries", [])

    todo = [f for f in C.list_images(inp, a.limit) if not C.out_done(out, f.name)]
    items = []  # (padded shape, file, padded rgb, unpadded (h, w)); crops are small so all are held in memory
    for f in todo:
        try:
            rgb, hw = prep(f, pre)
            items.append((rgb.shape[:2], f, rgb, hw))
        except Exception as e:
            tm.errors.append({"file": f.name, "error": f"{type(e).__name__}: {e}"})

    def run_batch(chunk):
        x = torch.from_numpy(np.stack([c[2] for c in chunk])).permute(0, 3, 1, 2).float().div(255.0).to(dev)
        with torch.no_grad():
            y = net(x).clamp(0, 1)
        y = (y.permute(0, 2, 3, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        for c, im in zip(chunk, y):
            h, w = c[3]
            C.write_png(out / c[1].name, np.ascontiguousarray(im[:h, :w, ::-1]))

    def timed(chunk):
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        run_batch(chunk)
        if dev == "cuda":
            torch.cuda.synchronize()
        tm.sec.extend([(time.perf_counter() - t) / len(chunk)] * len(chunk))

    def process(chunk, bs):
        """Run chunk with batch size bs; halve on OOM, isolate crops on other errors."""
        for s in range(0, len(chunk), bs):
            sub = chunk[s:s + bs]
            try:
                timed(sub)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    for c in sub:
                        tm.errors.append({"file": c[1].name, "error": "OutOfMemoryError at batch 1"})
                    continue
                retries.append({"shape": list(sub[0][0]), "batch": bs, "next_batch": max(1, bs // 2)})
                print(f"OOM at batch {bs} shape {sub[0][0]}: retrying with {max(1, bs // 2)}", flush=True)
                process(sub, max(1, bs // 2))
            except Exception as e:
                if len(sub) == 1:
                    tm.errors.append({"file": sub[0][1].name, "error": f"{type(e).__name__}: {e}"})
                else:
                    process(sub, 1)

    for sh in sorted({it[0] for it in items}):
        process([it for it in items if it[0] == sh], batch0)
    s = tm.summary()
    s.update(model="nafnet-gopro-w64", params=p, device=dev, batch=batch0, oom_retries=retries, weights=str(WEIGHTS.relative_to(ROOT)))
    C.save_summary(out / "timing.json", s)
    print("nafnet done", s["n_ok"], "ok", s["n_failed"], "failed", len(retries), "oom retries")


if __name__ == "__main__":
    main()
