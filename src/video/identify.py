"""Phase 9 identity stage (REPORT_phase9.md section 0): track template -> quality bar -> gallery search -> identity, "unknown" or "low quality". Pure NumPy.

    template = aggregate rule (configs/video.yaml final.rule, default R4*) over the track's usable faces, quality q = MagFace raw norm
    usable face = score >= embed floor (0.5) and native box height >= final.min_face_h (16 px)
    low quality  = mean height of the usable faces < final.bar_px (23 px), or no usable face  -> no identity is given (D2)
    otherwise    = top-1 gallery identity if its cosine >= the open-set threshold (FPIR 0.1, D6), else "unknown"
"""
import numpy as np

from src.video import aggregate as A

LOW, UNKNOWN = "low quality", "unknown"


def usable(score, h, floor=0.5, min_h=16.0):
    return (np.asarray(score) >= floor) & (np.asarray(h) >= min_h)


def track_template(emb, q, idx, rule="R4*"):
    """emb [N,512] (L2-normalised), q [N], idx [N] frame indices -> template [512]."""
    return A.RULES[rule](emb, q, idx)


def decide(mean_h, sim_top1, top_id, thr, bar_px=23.0):
    """One track: identity label from its mean usable-face height, top-1 cosine and top-1 id. mean_h = nan (no usable face) is low quality."""
    if not np.isfinite(mean_h) or mean_h < bar_px:
        return LOW
    return top_id if sim_top1 >= thr else UNKNOWN


def identify_track(emb, q, idx, h, gallery_ids, index, thr, rule="R4*", bar_px=23.0):
    """-> dict(label, mean_h, sim, top_id). `index` is a src.video.index.GalleryIndex over the gallery templates; emb/q/idx/h are the usable faces only."""
    if len(emb) == 0:
        return dict(label=LOW, mean_h=float("nan"), sim=float("nan"), top_id=None)
    mean_h = float(np.mean(h))
    t = track_template(emb, q, idx, rule)
    sim, pos = index.search(t[None].astype(np.float32), 1)
    s, top = float(sim[0, 0]), gallery_ids[int(pos[0, 0])]
    return dict(label=decide(mean_h, s, top, thr, bar_px), mean_h=mean_h, sim=s, top_id=top)
