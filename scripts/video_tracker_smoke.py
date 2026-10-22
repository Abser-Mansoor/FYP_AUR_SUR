"""Smoke test of the vendored trackers: two same-size objects crossing, with seeded detection jitter.

    envs/video/Scripts/python.exe scripts/video_tracker_smoke.py

Acceptance (Phase 5): each ground-truth object keeps ONE track id for at least 95% of its frames and there are no id switches,
for both ByteTrack and BoT-SORT. Writes outputs/video/tracker_smoke.json. Exit code 1 on failure.
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.video.vendor.botsort.bot_sort import BoTSORT  # noqa: E402
from src.video.vendor.bytetrack.byte_tracker import BYTETracker  # noqa: E402

H, W, N = 480, 640, 100


def synthetic(seed=42):
    """Per frame: list of (gt_id, [x1,y1,x2,y2]). A moves right, B moves left, same size, crossing in the middle."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(N):
        ax, bx = 40 + 5.0 * t, 540 - 5.0 * t
        ay, by = 200 + 0.2 * t, 230 - 0.2 * t
        objs = []
        for gid, (x, y) in ((0, (ax, ay)), (1, (bx, by))):
            j = rng.normal(0, 1.0, 4)
            objs.append((gid, np.array([x, y, x + 60, y + 80]) + j))
        frames.append(objs)
    return frames


def run(name, tracker, frames, call):
    last, switches, hits = {0: None, 1: None}, 0, {0: {}, 1: {}}
    for objs in frames:
        dets = np.array([[*b, 0.9] for _, b in objs], dtype=np.float64)
        tracks = call(tracker, dets)
        for gid, box in objs:
            best, iou_best = None, 0.0
            for tr in tracks:
                x1, y1, x2, y2 = tr[0]
                iw, ih = min(x2, box[2]) - max(x1, box[0]), min(y2, box[3]) - max(y1, box[1])
                inter = max(iw, 0) * max(ih, 0)
                iou = inter / ((x2 - x1) * (y2 - y1) + (box[2] - box[0]) * (box[3] - box[1]) - inter + 1e-9)
                if iou > iou_best:
                    best, iou_best = tr[1], iou
            if best is None or iou_best < 0.3:
                continue
            hits[gid][best] = hits[gid].get(best, 0) + 1
            if last[gid] is not None and last[gid] != best:
                switches += 1
            last[gid] = best
    dominant = {g: max(h.values()) / N if h else 0.0 for g, h in hits.items()}
    return dict(tracker=name, id_switches=switches, dominant_id_fraction=dominant, ids_per_object={g: sorted(h) for g, h in hits.items()},
                ok=bool(switches == 0 and all(v >= 0.95 for v in dominant.values())))


def main():
    frames = synthetic()
    bt = BYTETracker(Namespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False), frame_rate=30)

    def call_bt(tr, dets):
        out = tr.update(dets.copy(), (H, W), (H, W))
        return [(t.tlbr, t.track_id) for t in out]
    bs = BoTSORT(Namespace(track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7, track_buffer=30, match_thresh=0.8,
                           proximity_thresh=0.5, appearance_thresh=0.25, with_reid=False, cmc_method="none", name="smoke",
                           ablation=False, mot20=False), frame_rate=30)

    def call_bs(tr, dets):
        out = tr.update(dets.copy(), np.zeros((H, W, 3), np.uint8))  # 5 cols: x1y1x2y2, score (6 cols would multiply the score by col 5)
        return [(t.tlbr, t.track_id) for t in out]
    res = [run("bytetrack", bt, frames, call_bt), run("botsort", bs, frames, call_bs)]
    out = ROOT / "outputs/video"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tracker_smoke.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if all(r["ok"] for r in res) else 1)


if __name__ == "__main__":
    main()
