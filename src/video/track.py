"""One adapter over the vendored ByteTrack and BoT-SORT trackers.

    tr = TrackerAdapter("bytetrack", frame_rate=30)
    ids = tr.update(dets, (H, W))          # dets [N,5] = x1,y1,x2,y2,score  ->  track id per detection (-1 = not in any track)

Input is always 5 columns (a 6th column would multiply the score in BoT-SORT). Re-ID is off and camera-motion compensation is off ("none": ChokePoint
and the test videos use fixed cameras). Track ids restart at 1 for every adapter (the vendored classes keep a global counter, which is reset here).
The trackers return Kalman-smoothed boxes, so each track is mapped back to the detection with the highest IoU (at least `match_iou`).
"""
from argparse import Namespace

import numpy as np

from src.video.vendor.botsort.basetrack import BaseTrack as BSBase
from src.video.vendor.botsort.bot_sort import BoTSORT
from src.video.vendor.bytetrack.basetrack import BaseTrack as BTBase
from src.video.vendor.bytetrack.byte_tracker import BYTETracker

DEFAULTS = {
    "bytetrack": dict(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False),
    "botsort": dict(track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6, track_buffer=30, match_thresh=0.8, proximity_thresh=0.5,
                    appearance_thresh=0.25, with_reid=False, cmc_method="none", name="video", ablation=False, mot20=False),
}


def iou_matrix(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ub = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ua[:, None] + ub[None, :] - inter + 1e-9)


class TrackerAdapter:
    def __init__(self, name, frame_rate=30, params=None, match_iou=0.3, lam=0.0, min_h=32.0):
        """lam > 0 adds the Phase 8 face term to the association cost of both trackers (REPORT_phase8.md section 0.6):
        cost = (1 - lam) * (tracker's own IoU distance) + lam * (1 - cos) for a track-detection pair whose faces are both gated in
        (detector score >= 0.5, an embedding supplied, native box height >= min_h). lam = 0 (default) leaves the vendored trackers' behaviour unchanged."""
        assert name in DEFAULTS, name
        self.name, self.match_iou = name, match_iou
        self.lam, self.min_h = float(lam), float(min_h)
        p = {**DEFAULTS[name], **(params or {})}
        if name == "bytetrack":
            BTBase._count = 0
            self.t = BYTETracker(Namespace(**p), frame_rate=frame_rate)
        else:
            BSBase._count = 0
            self.t = BoTSORT(Namespace(**p), frame_rate=frame_rate)
        if self.lam > 0:
            self.t.face_hook = self._face_hook

    def _face_hook(self, tracks, dets, dists):
        out = np.array(dists, dtype=float, copy=True)
        for i, t in enumerate(tracks):
            if getattr(t, "emb", None) is None:
                continue
            for j, d in enumerate(dets):
                if getattr(d, "emb", None) is None:
                    continue
                out[i, j] = (1.0 - self.lam) * dists[i, j] + self.lam * (1.0 - float(np.dot(t.emb, d.emb)))
        return out

    def update(self, dets, hw, frame=None, embs=None):
        """dets [N,5] float. hw = (H, W). embs [N,512] L2-normalised face embeddings (NaN row = none), used only when lam > 0.
        Returns int array [N]: the track id of each detection, -1 if none."""
        dets = np.asarray(dets, np.float64).reshape(-1, 5)
        if self.lam > 0 and embs is not None and len(dets):
            gate = (dets[:, 4] >= 0.5) & ((dets[:, 3] - dets[:, 1]) >= self.min_h) & np.isfinite(np.asarray(embs)).all(1)
            arr = np.empty(len(dets), dtype=object)
            for i in range(len(dets)):
                arr[i] = np.asarray(embs[i], np.float64) if gate[i] else None
            self.t.det_embs = arr
        else:
            self.t.det_embs = None
        H, W = hw
        if self.name == "bytetrack":
            tracks = self.t.update(dets.copy(), (H, W), (H, W))
        else:
            tracks = self.t.update(dets.copy(), frame if frame is not None else np.zeros((H, W, 3), np.uint8))
        ids = np.full(len(dets), -1, int)
        if len(dets) == 0 or len(tracks) == 0:
            return ids
        iou = iou_matrix(dets[:, :4], np.array([t.tlbr for t in tracks]))
        # greedy one-to-one assignment, highest IoU first, deterministic tie-break by index
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None, kind="stable"), iou.shape))[0]
        used_d, used_t = set(), set()
        for d, t in order:
            if iou[d, t] < self.match_iou:
                break
            if d in used_d or t in used_t:
                continue
            ids[d] = tracks[t].track_id
            used_d.add(d)
            used_t.add(t)
        return ids
