"""Phase 9 embedding jobs (GPU; no metric is computed here). All in envs/video, through the pipeline path (src/align.py:norm_crop + src/recognizer.py, alias adaface_wf4m),
so every Phase 9 number uses the crops the pipeline makes (REPORT_phase9.md section 0.7, check 1: OpenCV 5.1 in envs/caface warps tiny faces slightly differently).

    envs/video/Scripts/python.exe scripts/phase9_embed.py --part mevid       # outputs/video/mevid/emb/wf_<tracklet>.npz: feat (flip off), feat_flip (flip sum), same face order as emb_<tracklet>.npz
    envs/video/Scripts/python.exe scripts/phase9_embed.py --part gallery     # outputs/video/mevid/gallery_wf.npz: native, 24 px and 32 px versions (F2), flip off and on (F3)
    envs/video/Scripts/python.exe scripts/phase9_embed.py --part tinyface    # outputs/video/phase9/tinyface_wf.npz: 2,000 cohort (F5) + 10,000 padding (Part E) distractors, disjoint, seeded
Resumable per tracklet (mevid); the other parts write one file each.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch  # noqa: F401
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.align import norm_crop  # noqa: E402
from src.recognizer import Recognizer  # noqa: E402

MEVID = ROOT / "outputs/video/mevid"
P9 = ROOT / "outputs/video/phase9"
GAL_SIZES = (24, 32)  # F2
N_COHORT, N_PAD = 2000, 10000


def recognizers():
    cfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    pre, w = cfg["preprocess"], ROOT / cfg["recognizers"]["adaface_wf4m"]
    arch = cfg["recognizer_arch"]["adaface_wf4m"]
    return (Recognizer(arch, w, pre["input_size"], pre["interp"], False, 256, "cuda"),
            Recognizer(arch, w, pre["input_size"], pre["interp"], True, 256, "cuda"))


def both(recs, crops):
    if not crops:
        z = np.zeros((0, 512), np.float32)
        return z, z
    return recs[0].get_embedding(crops), recs[1].get_embedding(crops)


def savez(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def part_mevid(recs, limit=None):
    import mevid_embed as M
    tr = pd.read_csv(MEVID / "tracklets.csv")
    tr = tr[tr.detected.fillna(False).astype(bool) & tr.eligible.fillna(False).astype(bool)].reset_index(drop=True)
    if limit:
        tr = tr.head(limit)
    names = {s: M.load_names(s) for s in ("train", "test")}
    t0 = time.time()
    for n, r in enumerate(tr.itertuples(), 1):
        f = MEVID / "emb" / f"wf_{r.tracklet}.npz"
        if f.exists():
            continue
        z, e = np.load(MEVID / "emb" / f"det_{r.tracklet}.npz"), np.load(MEVID / "emb" / f"emb_{r.tracklet}.npz")
        crops = [norm_crop(cv2.imread(str(M.chip_path(r.split, names[r.split][int(z["chip_idx"][q])]))), z["kps"][q]) for q in e["sel"]]
        a, b = both(recs, crops)
        savez(f, sel=e["sel"], feat=a, feat_flip=b)
        if n % 250 == 0:
            print(f"[{n}/{len(tr)}] {r.tracklet} {time.time() - t0:.0f}s", flush=True)
    print("mevid done", flush=True)


def part_gallery(recs):
    from mevid_embed import PHOTOS
    g = np.load(MEVID / "gallery.npz", allow_pickle=True)
    gf = pd.read_csv(MEVID / "gallery_faces.csv").set_index("file")
    out = {}
    for tag, size in (("native", None), *((f"h{s}", s) for s in GAL_SIZES)):
        crops = []
        for fn in g["photo"]:
            r = gf.loc[fn]
            kp = np.array([[r[f"kp{q}_x"], r[f"kp{q}_y"]] for q in range(5)], np.float32)
            im = cv2.imread(str(PHOTOS / fn))
            if size is not None:  # F2: the photo resized so that the face box is `size` px high (INTER_AREA), keypoints scaled with it
                s = size / float(r["face_h"])
                im = cv2.resize(im, (max(1, round(im.shape[1] * s)), max(1, round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
                kp = kp * np.array([im.shape[1] / (r["photo_w"]), im.shape[0] / (r["photo_h"])], np.float32)
            crops.append(norm_crop(im, kp))
        out[f"feat_{tag}"], out[f"feat_flip_{tag}"] = both(recs, crops)
        print("gallery", tag, flush=True)
    savez(MEVID / "gallery_wf.npz", photo=g["photo"], person=g["person"], face_h=g["face_h"], **out)


def part_tinyface(recs):
    files = sorted((ROOT / "Dataset/tinyface/tinyface/Testing_Set/Gallery_Distractor").glob("*.jpg"))
    pick = np.random.default_rng(42).choice(len(files), N_COHORT + N_PAD, replace=False)
    sel = [files[i] for i in pick]
    crops = [cv2.imread(str(p)) for p in sel]  # resize-only (the tiny-crop protocol of Phases 1-3): src/recognizer.py resizes to 112 x 112, bilinear
    a, b = both(recs, crops)
    role = np.array(["cohort"] * N_COHORT + ["pad"] * N_PAD)
    savez(P9 / "tinyface_wf.npz", fname=np.array([p.name for p in sel]), role=role, feat=a, feat_flip=b)
    print("tinyface done", len(sel), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["mevid", "gallery", "tinyface", "all"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    recs = recognizers()
    for p in (["gallery", "tinyface", "mevid"] if a.part == "all" else [a.part]):
        {"mevid": lambda: part_mevid(recs, a.limit), "gallery": lambda: part_gallery(recs), "tinyface": lambda: part_tinyface(recs)}[p]()
