"""Phase 6 pipeline runner.

    envs/video/Scripts/python.exe scripts/run_video.py --stage detect --dataset chokepoint --clips P1E_S1_C1 --limit 300    # smoke / timing
    envs/video/Scripts/python.exe scripts/run_video.py --stage all --dataset chokepoint                                      # every clip, resumable

Stages: detect (GPU, SCRFD) | track (CPU, ByteTrack + BoT-SORT on the same detections) | embed (GPU, aligned AdaFace + MagFace norm) | all.
`--limit N` runs the first N frames of each clip and writes to <out>/_smoke/ so it never mixes with a full run.
Datasets: chokepoint (the 72 clips with eye ground truth, plus the crowded clips with `--crowded`); `--source PATH` runs any folder of frames or a video file as one clip.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402,F401  (before onnxruntime)
from src.video import pipeline as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/video.yaml")
    ap.add_argument("--stage", default="all", choices=["detect", "track", "embed", "all"])
    ap.add_argument("--dataset", default="chokepoint")
    ap.add_argument("--clips", nargs="+", default=None)
    ap.add_argument("--crowded", action="store_true", help="also run the crowded clips without ground truth")
    ap.add_argument("--source", default=None, help="a folder of frames or a video file, run as one clip named after it")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--recognizers", nargs="+", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    ds = a.dataset + ("/_smoke" if a.limit else "")
    if a.source:
        src = Path(a.source)
        jobs = [(src.stem, src)]
    else:
        gt, crowded = P.chokepoint_clips()
        names = a.clips or (gt + (crowded if a.crowded else []))
        jobs = [(c, ROOT / "Dataset/ChokePoint" / c) for c in names]
    det = emb = None
    t_all = time.perf_counter()
    for k, (clip, src) in enumerate(jobs, 1):
        out = P.clip_out(cfg, ds, clip)
        msg = {}
        if a.stage in ("detect", "all"):
            det = det or P.build_detector(cfg)
            msg["detect"] = P.detect_clip(det, cfg, src, out, a.limit)
        if a.stage in ("track", "all"):
            msg["track"] = P.track_clip(cfg, out)
        if a.stage in ("embed", "all"):
            if emb is None:
                from src.video.embed import FaceEmbedder
                emb = FaceEmbedder(cfg, tuple(a.recognizers or cfg["pipeline"]["recognizers"]))
            msg["embed"] = P.embed_clip(emb, cfg, src, out, a.limit)
        print(f"[{k}/{len(jobs)}] {clip}: {json.dumps({s: {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in m.items()} for s, m in msg.items()})}", flush=True)
    print(f"done in {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
