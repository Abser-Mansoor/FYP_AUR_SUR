"""Official protocol file lists. Labels always come from the .mat files
(TinyFace filename prefixes do NOT match the .mat IDs)."""
from pathlib import Path

import numpy as np
import scipy.io as sio


def _mat(path):
    m = sio.loadmat(path)
    return {k: v for k, v in m.items() if not k.startswith("__")}


def _names(cell):
    return [str(x[0]) if x.size else "" for x in np.asarray(cell).ravel()]


def tinyface(root):
    t = Path(root) / "Testing_Set"
    g = _mat(t / "gallery_match_img_ID_pairs.mat")
    p = _mat(t / "probe_img_ID_pairs.mat")
    distractors = sorted(x.name for x in (t / "Gallery_Distractor").iterdir())
    return {
        "probe": ([t / "Probe" / n for n in _names(p["probe_set"])], p["probe_ids"].ravel().astype(np.int64)),
        "gallery_match": ([t / "Gallery_Match" / n for n in _names(g["gallery_set"])], g["gallery_ids"].ravel().astype(np.int64)),
        "distractor": ([t / "Gallery_Distractor" / n for n in distractors], None),
    }


def survface_identification(root):
    t = Path(root) / "Face_Identification_Test_Set"
    g = _mat(t / "gallery_img_ID_pairs.mat")
    m = _mat(t / "mated_probe_img_ID_pairs.mat")
    unmated = sorted(x.name for x in (t / "unmated_probe").iterdir())
    return {
        "gallery": ([t / "gallery" / n for n in _names(g["gallery_set"])], g["gallery_ids"].ravel().astype(np.int64)),
        "mated_probe": ([t / "mated_probe" / n for n in _names(m["mated_probe_set"])], m["mated_probe_ids"].ravel().astype(np.int64)),
        "unmated_probe": ([t / "unmated_probe" / n for n in unmated], None),
    }


def survface_verification(root):
    t = Path(root) / "Face_Verification_Test_Set"
    pos = _mat(t / "positive_pairs_names.mat")
    neg = _mat(t / "negative_pairs_names.mat")
    pos = np.asarray(next(iter(pos.values())), dtype=object)
    neg = np.asarray(next(iter(neg.values())), dtype=object)
    pairs = {k: [(str(a[0]), str(b[0])) for a, b in v] for k, v in (("pos", pos), ("neg", neg))}
    names = sorted({n for v in pairs.values() for p in v for n in p})
    return [t / "verification_images" / n for n in names], pairs
