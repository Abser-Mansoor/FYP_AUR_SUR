"""Phase 5 check: the SCRFD wrapper reproduces the Phase 1 landmarks.

    envs/video/Scripts/python.exe scripts/phase5_verify_scrfd.py

Takes 100 seeded TinyFace probes of the eval subset, runs src/video/detect.py:crop_detect (SCRFDDetector) on the exported PNG
crops and compares with (a) src/align.py:detect_landmarks on the same crops and (b) the Phase 1 cache
outputs/cache/phase1/landmarks_tinyface_probe.npz. Expected: wrapper identical to detect_landmarks (0 px); Phase 1 cache within 0.05 px (the cache was written in an earlier session, so bit equality is not required; the Colab check uses 0.5 px). Writes outputs/video/verify_scrfd.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import datasets as DS  # noqa: E402
from src.align import SCRFD, detect_landmarks  # noqa: E402
from src.sr import common as C  # noqa: E402
from src.video import detect as D  # noqa: E402


def main():
    img = pd.read_csv(ROOT / "outputs/sr/subset/eval_images.csv")
    pr = img[(img.dataset == "tinyface") & (img.split == "probe")].sample(100, random_state=42)
    paths = DS.tinyface(ROOT / "Dataset/tinyface/tinyface")["probe"][0]
    index = {p.name: i for i, p in enumerate(paths)}
    z = np.load(ROOT / "outputs/cache/phase1/landmarks_tinyface_probe.npz")
    ref_kps, ref_ok = z["kps"], z["ok"]
    det, raw = D.SCRFDDetector(ROOT / "models/detector/det_10g.onnx"), SCRFD(str(ROOT / "models/detector/det_10g.onnx"))
    d_wrap, d_phase1, n_det_mismatch = [], [], 0
    for _, r in pr.iterrows():
        im = C.read_bgr(ROOT / "outputs/sr/subset/eval_input" / r.file)
        k_new = D.crop_detect(det, im, 112, 0.3)
        k_old = detect_landmarks(raw, im, 112, 0.3)
        i = index[r["name"]]
        k_p1 = ref_kps[i] if ref_ok[i] else None
        if (k_new is None) != (k_old is None):
            n_det_mismatch += 1
        elif k_new is not None:
            d_wrap.append(float(np.abs(k_new - k_old).max()))
        if (k_new is None) != (k_p1 is None):
            n_det_mismatch += 1
        elif k_new is not None:
            d_phase1.append(float(np.abs(k_new - k_p1).max()))
    res = dict(n=100, max_abs_diff_vs_detect_landmarks_px=max(d_wrap), max_abs_diff_vs_phase1_cache_px=max(d_phase1),
               n_compared_wrapper=len(d_wrap), n_compared_phase1=len(d_phase1), n_detection_mismatch=n_det_mismatch)
    res["pass"] = bool(res["max_abs_diff_vs_detect_landmarks_px"] == 0 and res["max_abs_diff_vs_phase1_cache_px"] <= 0.05 and n_det_mismatch == 0)
    (ROOT / "outputs/video").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/video/verify_scrfd.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
