"""HOTA, CLEAR (MOTA, ID switches) and Identity (IDF1) of tracks against ground-truth tracks, with the vendored TrackEval metric classes.

    gt, pred: {frame_idx: (ids[N] int, boxes[N,4] x1,y1,x2,y2)} ;  frames: sorted list of every frame of the clip (frames without entries are empty)
    res = clip_metrics(gt, pred, frames)          # dict of scalars for one clip
    pooled = pool([res_clip1, res_clip2, ...])    # TrackEval's combine_sequences over the clips

The TrackEval data dict is built here (the same fields as `MotChallenge2DBox.get_preprocessed_seq_data`): ids remapped to 0..n-1, IoU similarity, no
distractor or class filtering. IoU is the standard (no +1) box IoU, as in TrackEval.
"""
import numpy as np

from src.video.track import iou_matrix
from src.video.vendor.trackeval.metrics import CLEAR, HOTA, Identity

_CFG = {"PRINT_CONFIG": False}
FIELDS = {"HOTA": ["HOTA", "DetA", "AssA", "LocA"], "CLEAR": ["MOTA", "MOTP", "IDSW", "CLR_TP", "CLR_FN", "CLR_FP", "MT", "PT", "ML", "Frag"],
          "Identity": ["IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP"]}


def _metrics():
    return {"HOTA": HOTA(dict(_CFG)), "CLEAR": CLEAR(dict(_CFG)), "Identity": Identity(dict(_CFG))}


def build_data(gt, pred, frames):
    gt_map, tr_map = {}, {}
    for d, m in ((gt, gt_map), (pred, tr_map)):
        for f in frames:
            for i in d.get(f, (np.zeros(0, int), None))[0]:
                m.setdefault(int(i), len(m))
    data = dict(num_timesteps=len(frames), num_gt_ids=len(gt_map), num_tracker_ids=len(tr_map), gt_ids=[], tracker_ids=[], similarity_scores=[],
                num_gt_dets=0, num_tracker_dets=0)
    for f in frames:
        gi, gb = gt.get(f, (np.zeros(0, int), np.zeros((0, 4))))
        ti, tb = pred.get(f, (np.zeros(0, int), np.zeros((0, 4))))
        data["gt_ids"].append(np.array([gt_map[int(i)] for i in gi], int))
        data["tracker_ids"].append(np.array([tr_map[int(i)] for i in ti], int))
        data["similarity_scores"].append(iou_matrix(np.asarray(gb, float).reshape(-1, 4), np.asarray(tb, float).reshape(-1, 4)))
        data["num_gt_dets"] += len(gi)
        data["num_tracker_dets"] += len(ti)
    return data


def eval_data(data):
    return {k: m.eval_sequence(data) for k, m in _metrics().items()}


def flatten(res):
    out = {}
    for k, fields in FIELDS.items():
        for f in fields:
            v = res[k][f]
            out[f] = float(np.mean(v)) if np.ndim(v) else float(v)
    return out


def clip_metrics(gt, pred, frames):
    res = eval_data(build_data(gt, pred, frames))
    return dict(raw=res, **flatten(res))


def pool(results):
    """results: {clip: clip_metrics output}. Returns the pooled (combine_sequences) scalars."""
    ms = _metrics()
    return flatten({k: m.combine_sequences({c: r["raw"][k] for c, r in results.items()}) for k, m in ms.items()})
