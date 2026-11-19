"""Per-track aggregation rules of Phase 7 (REPORT_phase7.md section 0.2). Pure NumPy, no GPU.

Every embedding rule has one signature: rule(emb, q, idx=None, **kw) -> template[512], L2-normalised.
  emb  [N, 512]  L2-normalised face embeddings of one track
  q    [N]       quality (MagFace raw norm of the same faces)
  idx  [N]       frame indices. The frames are first put in frame-index order, so every rule gives the identical result however the frames were passed in,
                 and ties (equal quality in top-k) go to the lower frame index.
Score-level rules work on the per-frame cosine scores to the gallery templates: see `score_mean` and `vote`.
"""
import numpy as np


def l2n(x, axis=-1, eps=1e-12):
    x = np.asarray(x, np.float64)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def _prep(emb, q, idx):
    """Frames in frame-index order (stable), as float64."""
    idx = np.arange(len(q)) if idx is None else np.asarray(idx)
    o = np.argsort(idx, kind="stable")
    return np.asarray(emb, np.float64)[o], np.asarray(q, np.float64)[o]


def _combine(e, w):
    return l2n((e * w[:, None]).sum(0))


def _softmax(z):
    c = np.exp(z - z.max())
    return c / c.sum()


def r1_mean(emb, q, idx=None):
    e, _ = _prep(emb, q, idx)
    return _combine(e, np.ones(len(e)))


def r2_qmean(emb, q, idx=None):
    e, q = _prep(emb, q, idx)
    return _combine(e, q)


def r3_topk(emb, q, idx=None, k=5):
    """Mean of the k frames with the highest quality (ties: lower frame index first). k >= N gives the plain mean."""
    e, q = _prep(emb, q, idx)
    keep = np.zeros(len(q))
    keep[np.lexsort((np.arange(len(q)), -q))[:k]] = 1.0
    return _combine(e, keep)


def r4_np(emb, q, idx=None, lam=50.0):
    """Norm pooling (arXiv 2409.09832, Eq. 4-6): c = softmax(lam * q / max q); template = sum c_i e_i."""
    e, q = _prep(emb, q, idx)
    return _combine(e, _softmax(lam * q / q.max()))


def r4s_np_star(emb, q, idx=None, lam=5.0):
    """Norm pooling, min-max variant (Eq. 7): l = (q - min q) / (max q - min q) (all zero if the norms are equal)."""
    e, q = _prep(emb, q, idx)
    span = q.max() - q.min()
    return _combine(e, _softmax(lam * ((q - q.min()) / span if span > 0 else np.zeros(len(q)))))


RULES = {"R1": r1_mean, "R2": r2_qmean, "R3_k1": lambda e, q, idx=None: r3_topk(e, q, idx, 1), "R3_k3": lambda e, q, idx=None: r3_topk(e, q, idx, 3),
         "R3_k5": lambda e, q, idx=None: r3_topk(e, q, idx, 5), "R3_k10": lambda e, q, idx=None: r3_topk(e, q, idx, 10), "R4": r4_np, "R4*": r4s_np_star}


def score_mean(emb, templates, idx=None):
    """R5: mean over frames of the cosine to each gallery template -> [T] scores (identity = argmax). emb [N,512], templates [T,512] (L2-normalised)."""
    e, _ = _prep(emb, np.zeros(len(emb)), idx)
    return (e @ np.asarray(templates, np.float64).T).mean(0)


def vote(emb, templates, idx=None):
    """R6: majority vote of the per-frame Rank-1 identities; ties go to the higher mean score. Returns the winning template index (closed set only)."""
    e, _ = _prep(emb, np.zeros(len(emb)), idx)
    s = e @ np.asarray(templates, np.float64).T
    counts = np.bincount(s.argmax(1), minlength=s.shape[1])
    best = np.flatnonzero(counts == counts.max())
    return int(best[s.mean(0)[best].argmax()])
