import numpy as np
def bbox_overlaps(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    out = np.zeros((len(a), len(b)))
    for k in range(len(b)):
        x1 = np.maximum(a[:, 0], b[k, 0]); y1 = np.maximum(a[:, 1], b[k, 1])
        x2 = np.minimum(a[:, 2], b[k, 2]); y2 = np.minimum(a[:, 3], b[k, 3])
        w = x2 - x1 + 1; h = y2 - y1 + 1
        inter = w * h
        aa = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
        ba = (b[k, 2] - b[k, 0] + 1) * (b[k, 3] - b[k, 1] + 1)
        o = inter / (aa + ba - inter)
        o[(w <= 0) | (h <= 0)] = 0
        out[:, k] = o
    return out
