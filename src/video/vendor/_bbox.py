"""NumPy replacement for cython_bbox.bbox_overlaps (same +1 pixel convention as the Cython original), so the vendored
trackers need no compiled extension."""
import numpy as np


def bbox_overlaps(boxes, query_boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    q = np.asarray(query_boxes, dtype=np.float64)
    out = np.zeros((len(boxes), len(q)), dtype=np.float64)
    if out.size == 0:
        return out
    q_area = (q[:, 2] - q[:, 0] + 1) * (q[:, 3] - q[:, 1] + 1)
    b_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    iw = np.minimum(boxes[:, None, 2], q[None, :, 2]) - np.maximum(boxes[:, None, 0], q[None, :, 0]) + 1
    ih = np.minimum(boxes[:, None, 3], q[None, :, 3]) - np.maximum(boxes[:, None, 1], q[None, :, 1]) + 1
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    union = b_area[:, None] + q_area[None, :] - inter
    return np.where((iw > 0) & (ih > 0), inter / union, 0.0)
