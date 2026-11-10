"""Acceptance check 2 (Phase 6): the video embedding path equals the Phase 1 aligned path, and `return_raw` changes nothing.

    envs/video/Scripts/python.exe scripts/video_embed_check.py

1. 100 seeded faces from the ChokePoint detector-study sample (SCRFD full1280, the largest face with score >= 0.5 per frame). Video path:
   `src/video/embed.py:FaceEmbedder.embed` (recognizers and preprocessing from configs/video.yaml). Phase 1 path: `norm_crop` + `Recognizer` built from
   configs/baseline.yaml (its own weights and preprocess entries, as scripts/run_aligned_survface.py does). Cosine must be above 0.9999 for every face.
2. `get_embedding(crops)` equals `get_embedding(crops, return_raw=True)[0]` (np.array_equal).
3. The raw MagFace norm of 100 seeded TinyFace probes (direct feed) equals the Phase 2 `magface_norm` of the same images (relative difference below 1e-4).
Writes outputs/video/embed_check.json; exit code 1 on failure.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402,F401  (before onnxruntime)
from src.align import norm_crop  # noqa: E402
from src.recognizer import Recognizer  # noqa: E402
from src.sr import common as C  # noqa: E402
from src.video import detect as D  # noqa: E402
from src.video.embed import FaceEmbedder  # noqa: E402


def main():
    vcfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    bcfg = yaml.safe_load(open(ROOT / "configs/baseline.yaml"))
    sc = vcfg["scrfd"]
    det = D.SCRFDDetector(ROOT / vcfg["detectors"]["scrfd"], thresh=vcfg["detect_thresh"], nms=sc["nms"], preprocess=sc["preprocess"])
    idx = pd.read_csv(ROOT / "outputs/video/chokepoint_sample/index.csv")
    pick = idx.iloc[np.random.default_rng(42).permutation(len(idx))]
    faces = []
    for _, r in pick.iterrows():
        bgr = C.read_bgr(ROOT / "outputs/video/chokepoint_sample" / r["file"])
        dets, kps = D.detect_mode(det, bgr, vcfg["pipeline"]["detector_mode"])
        ok = np.flatnonzero(dets[:, 4] >= 0.5)
        if len(ok):
            i = ok[np.argmax((dets[ok, 2] - dets[ok, 0]) * (dets[ok, 3] - dets[ok, 1]))]
            faces.append((bgr, kps[i]))
        if len(faces) == 100:
            break
    assert len(faces) == 100, len(faces)
    emb = FaceEmbedder(vcfg, ("adaface",))
    v = np.concatenate([emb.embed(b, k[None])["emb"]["adaface"] for b, k in faces])
    pre = bcfg["preprocess"]
    rec1 = Recognizer("adaface", ROOT / bcfg["recognizers"]["adaface"], pre["input_size"], pre["interp"], pre["flip"], bcfg["batch_size"], bcfg["device"])
    crops = [norm_crop(b, k) for b, k in faces]
    p1 = rec1.get_embedding(crops)
    cos = (v * p1).sum(1)
    same_raw = bool(np.array_equal(p1, rec1.get_embedding(crops, return_raw=True)[0]))
    # raw MagFace norm against Phase 2
    mag = Recognizer("magface", ROOT / vcfg["recognizers"]["magface"], pre["input_size"], pre["interp"], False, 128, "cuda")
    ev = pd.read_csv(ROOT / "outputs/sr/subset/eval_images.csv")
    pr = ev[(ev.dataset == "tinyface") & (ev.split == "probe")].sample(100, random_state=42)
    ph2 = pd.read_csv(ROOT / "outputs/stratify/bins/tinyface_probe.csv").drop_duplicates("name").set_index("name")["magface_norm"]
    ref = ph2.reindex(pr["name"]).to_numpy()
    _, raw = mag.get_embedding([C.read_bgr(ROOT / "outputs/sr/subset/eval_input" / f) for f in pr["file"]], return_raw=True)
    rel = float(np.nanmax(np.abs(raw - ref) / ref))
    out = dict(n_faces=len(faces), min_cosine_video_vs_phase1=float(cos.min()), return_raw_embedding_equal=same_raw, n_norms_compared=int(np.isfinite(ref).sum()),
               max_rel_diff_raw_norm_vs_phase2=rel,
               ok=bool(cos.min() > 0.9999 and same_raw and rel < 1e-4 and np.isfinite(ref).all()))
    (ROOT / "outputs/video/embed_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
