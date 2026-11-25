"""MEVID detection, gallery and embedding for Phase 8 (protocol: REPORT_phase8.md section 0; plan: research/Sprint_Notes/Phase8_plan.md, Part A).

    envs/video/Scripts/python.exe scripts/mevid_embed.py --stage index      # tracklets.csv (train + test), count check                      (CPU)
    envs/video/Scripts/python.exe scripts/mevid_embed.py --stage check2     # acceptance check 2: the probe's 3,949 chips give the same boxes (GPU, detector only)
    envs/video/Scripts/python.exe scripts/mevid_embed.py --stage detect     # up to 50 chips per tracklet, SCRFD size 640, face rule      (GPU, detector only)
    envs/video/Scripts/python.exe scripts/mevid_embed.py --stage eligibility # tracklets.csv: eligibility and counts per tracklet                      (CPU)
    envs/video/Scripts/python.exe scripts/mevid_embed.py --stage gallery    # front check-in photos, SCRFD size 1280 -> gallery_faces.csv  (GPU, detector only)
    envs/video/Scripts/python.exe scripts/mevid_embed.py --stage embed --i-confirm-section0-is-committed   # AdaFace (+ raw norms) of the usable faces of the ELIGIBLE tracklets and the gallery faces (GPU; ONLY after section 0 is committed)

The stages `index`, `check2`, `detect` and `gallery` use the detector only; no embedding or recognition score exists until `embed` runs, and `embed` must not be run before
REPORT_phase8.md section 0 is committed (the plan's rule). Everything is resumable per tracklet / per file and written atomically. Outputs under outputs/video/mevid/:
tracklets.csv and gallery_faces.csv (tracked), emb/det_<tracklet>.npz, emb/emb_<tracklet>.npz and gallery.npz (local only, git-ignored). No MEVID image is copied anywhere.
Face rule (the probe's, fixed in M7): the highest-scoring SCRFD detection with score >= 0.5 whose box centre is in the upper 50% of the chip; usable if its native height >= 16 px.
Gallery rule (M2 to M4): front photos only (`f`/`F`), SCRFD with the photo resized so that the longest side is 1280, the largest face with score >= 0.5; usable at >= 48 px (flag at 64 px).
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "Dataset/MEVID"
ANN = DATA / "mevid-v1-annotation-data"
PHOTOS = DATA / "mevid-v1-actor-checkin-photos"
OUT = ROOT / "outputs/video/mevid"
PROBE = ROOT / "outputs/video/mevid_probe"
MAX_CHIPS, SCORE, MIN_FACE_H, GAL_SIZE, GAL_BAR, GAL_BAR_SENS = 50, 0.5, 16, 1280, 48, 64
README_COUNTS = dict(tracklets=8092, identities=158)  # plan, acceptance check 1
CHIP_RE = re.compile(r"(\d{4})O(\d{3})C(\d{3})T(\d{3})F(\d{5})\.jpg$")
PHOTO_RE = re.compile(r"(\d{3})-(\d{8})-(\w{3})-\s*(\d{3})-([fbFB])\.jpg$")


def atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load_split(split):
    t = np.loadtxt(ANN / f"track_{split}_info.txt").astype(int)
    df = pd.DataFrame(t, columns=["start", "end", "person", "outfit", "camera"])
    df.insert(0, "tracklet", [f"{split[:2]}{i:05d}" for i in range(len(df))])
    df.insert(1, "split", split)
    df["idx"] = np.arange(len(df))
    df["n_chips"] = df.end - df.start + 1
    return df


def load_names(split):
    return (ANN / f"{split}_name.txt").read_text().split()


def chip_path(split, name):
    return DATA / f"bbox_{split}" / name[:4] / name


def pick_chips(start, end):
    n = end - start + 1
    return np.unique(np.round(np.linspace(start, end, min(MAX_CHIPS, n))).astype(int))


def assign_face(dets, chip_h):
    """The probe's face rule. dets [N,5] (x1,y1,x2,y2,score) in native chip pixels -> (index of the face or -1)."""
    ok = np.flatnonzero((dets[:, 4] >= SCORE) & (((dets[:, 1] + dets[:, 3]) / 2) < 0.5 * chip_h))
    return int(ok[np.argmax(dets[ok, 4])]) if len(ok) else -1


def build_detector():
    import torch  # noqa: F401  (before onnxruntime creates a CUDA session)
    import yaml
    from src.video.pipeline import build_detector as bd
    cfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    h = hashlib.sha256((ROOT / cfg["detectors"]["scrfd"]).read_bytes()).hexdigest()
    assert h in (ROOT / "MODEL_INVENTORY.md").read_text(), "det_10g.onnx does not match MODEL_INVENTORY.md"
    return bd(cfg), cfg, h


# ----------------------------------------------------------------------------- index
def stage_index(a):
    tr = pd.concat([load_split("train"), load_split("test")], ignore_index=True)
    query = set(np.loadtxt(ANN / "query_IDX.txt").astype(int).tolist())
    tr["is_query"] = (tr.split == "test") & tr.idx.isin(query)
    for split, g in tr.groupby("split"):
        names = load_names(split)
        ok = (g.start.iloc[0] == 0) and (g.end.iloc[-1] == len(names) - 1) and (g.start.iloc[1:].to_numpy() == g.end.iloc[:-1].to_numpy() + 1).all()
        print(f"{split}: {len(g)} tracklets, {len(names)} chips, ranges contiguous and cover the name list: {bool(ok)}")
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["tracklet", "split", "idx", "start", "end", "person", "outfit", "camera", "n_chips", "is_query"]
    tr[cols].to_csv(OUT / "tracklets_index.csv", index=False)
    empty = tr[tr.n_chips <= 0]
    res = dict(tracklets_train=int((tr.split == "train").sum()), tracklets_test=int((tr.split == "test").sum()), tracklets_total=int(len(tr)), persons_train=int(tr[tr.split == "train"].person.nunique()),
               persons_test=int(tr[tr.split == "test"].person.nunique()), persons_total=int(tr.person.nunique()), readme_counts=README_COUNTS,
               tracklets_equal_readme=bool(len(tr) == README_COUNTS["tracklets"]), persons_equal_readme=bool(tr.person.nunique() == README_COUNTS["identities"]),
               empty_tracklets=[dict(tracklet=r.tracklet, start=int(r.start), end=int(r.end), person=int(r.person)) for r in empty.itertuples()],
               tracklets_with_at_least_one_chip=int((tr.n_chips > 0).sum()), tracklets_with_chips_equal_readme=bool((tr.n_chips > 0).sum() == README_COUNTS["tracklets"]))
    ph = pd.read_csv(PROBE / "checkin_photos.csv")
    res["checkin_photos_equal_probe_inventory"] = bool(len(ph) == 2898 and ph.person.nunique() == 172)
    res["checkin_photos"], res["checkin_persons"] = int(len(ph)), int(ph.person.nunique())
    (OUT / "index_counts.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res, indent=1, sort_keys=True))


# ----------------------------------------------------------------------------- detection helpers
def detect_chips(det, split, names, chip_ids):
    """-> list of (chip_idx, h, w, face_idx, dets) for the chips (images read in a thread pool)."""
    def read(i):
        return cv2.imread(str(chip_path(split, names[i])))
    out = []
    with ThreadPoolExecutor(8) as ex:
        for i, im in zip(chip_ids, ex.map(read, chip_ids)):
            if im is None:
                out.append((int(i), 0, 0, -1, np.zeros((0, 5), np.float32), None))
                continue
            d, k = det.detect(im, size=640)
            j = assign_face(d, im.shape[0])
            out.append((int(i), im.shape[0], im.shape[1], j, d, k))
    return out


# ----------------------------------------------------------------------------- check 2
def stage_check2(a):
    """Acceptance check 2: the probe's 3,949 sampled chips give the same face boxes with this script's detector and rule."""
    det, cfg, h = build_detector()
    sm = pd.read_csv(PROBE / "sample.csv")
    pd_ = pd.read_csv(PROBE / "detections.csv")
    names = load_names("test")
    rows = []
    t0 = time.time()
    by_tracklet = {t: g for t, g in sm.groupby("tracklet")}
    for t, g in by_tracklet.items():
        res = detect_chips(det, "test", names, g.chip_idx.to_numpy())
        for (i, ch, cw, j, d, k), name in zip(res, g.chip_file):
            row = dict(chip_file=name, face=bool(j >= 0))
            if j >= 0:
                row.update(score=float(d[j, 4]), x1=float(d[j, 0]), y1=float(d[j, 1]), x2=float(d[j, 2]), y2=float(d[j, 3]), face_h=float(d[j, 3] - d[j, 1]), usable=bool(d[j, 3] - d[j, 1] >= MIN_FACE_H))
            else:
                row.update(score=np.nan, x1=np.nan, y1=np.nan, x2=np.nan, y2=np.nan, face_h=np.nan, usable=False)
            rows.append(row)
    new = pd.DataFrame(rows).set_index("chip_file")
    ref = pd_.set_index("chip_file")
    cols = ["score", "x1", "y1", "x2", "y2", "face_h"]
    same_face = bool((new.face == ref.face).all())
    both = new.face & ref.face
    diff = float(np.nanmax(np.abs(new.loc[both, cols].round(3).to_numpy() - ref.loc[both, cols].to_numpy()))) if both.any() else 0.0
    res = dict(chips=int(len(new)), faces_in_probe=int(ref.face.sum()), faces_here=int(new.face.sum()), same_face_flag_for_every_chip=same_face, max_abs_difference_in_box_or_score_after_rounding_to_3_decimals=diff,
               identical=bool(same_face and diff == 0.0), seconds=round(time.time() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "checks.json"
    allres = json.loads(p.read_text()) if p.exists() else {}
    allres["check2_probe_boxes_reproduced"] = res
    p.write_text(json.dumps(allres, indent=1, sort_keys=True))
    print(json.dumps(res, indent=1, sort_keys=True))


# ----------------------------------------------------------------------------- detect (tracklets)
def stage_detect(a):
    det, cfg, h = build_detector()
    tr = pd.read_csv(OUT / "tracklets_index.csv")
    names = {s: load_names(s) for s in ("train", "test")}
    if a.split:
        tr = tr[tr.split == a.split]
    todo = tr if not a.limit else tr.head(a.limit)
    (OUT / "emb").mkdir(parents=True, exist_ok=True)
    t0, skipped = time.time(), 0
    for n, r in enumerate(todo.itertuples(), 1):
        f = OUT / "emb" / f"det_{r.tracklet}.npz"
        if f.exists():
            continue
        if r.n_chips <= 0:
            continue  # an empty annotation row (found at tr01889): no chips, never eligible
        ids = pick_chips(r.start, r.end)
        if not all(chip_path(r.split, names[r.split][int(i)]).exists() for i in (ids[0], ids[len(ids) // 2], ids[-1])):
            skipped += 1  # chips not extracted yet: never record a missing image as "no face"
            continue
        res = detect_chips(det, r.split, names[r.split], ids)
        face = np.array([x[3] >= 0 for x in res])
        box = np.full((len(res), 5), np.nan, np.float32)
        kps = np.full((len(res), 5, 2), np.nan, np.float32)
        for q, (i, ch, cw, j, d, k) in enumerate(res):
            if j >= 0:
                box[q], kps[q] = d[j], k[j]
        fh = box[:, 3] - box[:, 1]
        atomic_savez(f, chip_idx=ids, chip_h=np.array([x[1] for x in res]), chip_w=np.array([x[2] for x in res]), face=face, box=box, kps=kps, face_h=fh, usable=face & (fh >= MIN_FACE_H))
        if n % 200 == 0:
            print(f"[{n}/{len(todo)}] {r.tracklet} elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"detect done; tracklets skipped because their chips are not extracted yet: {skipped}", flush=True)


# ----------------------------------------------------------------------------- eligibility (CPU, from the detect outputs)
def stage_eligibility(a):
    tr = pd.read_csv(OUT / "tracklets_index.csv")
    rows = []
    for r in tr.itertuples():
        f = OUT / "emb" / f"det_{r.tracklet}.npz"
        if not f.exists():
            rows.append(dict(tracklet=r.tracklet, detected=False))
            continue
        z = np.load(f)
        u = z["usable"]
        rows.append(dict(tracklet=r.tracklet, detected=True, n_sampled=int(len(z["face"])), n_face=int(z["face"].sum()), n_usable=int(u.sum()), eligible=bool(u.sum() >= 5),
                         mean_face_h=float(np.nanmean(z["face_h"][u])) if u.any() else np.nan))
    d = tr.merge(pd.DataFrame(rows), on="tracklet")
    d.to_csv(OUT / "tracklets.csv", index=False, float_format="%.3f")
    det = d[d.detected]
    print(f"detected {len(det)} of {len(d)} tracklets; eligible (>= 5 usable faces): {int(det.eligible.sum())}; persons with an eligible tracklet: {det[det.eligible].person.nunique()}")
    print(det.groupby("split").eligible.agg(["sum", "count"]).to_string())


# ----------------------------------------------------------------------------- gallery (detection)
def stage_gallery(a):
    det, cfg, h = build_detector()
    ph = pd.read_csv(PROBE / "checkin_photos.csv")
    ph = ph[ph.view == "f"].reset_index(drop=True)  # front photos only (M2)
    rows = []
    t0 = time.time()
    for r in ph.itertuples():
        im = cv2.imread(str(PHOTOS / r.file))
        d, k = det.detect(im, size=GAL_SIZE)
        d_ok = d[d[:, 4] >= SCORE]
        row = dict(file=r.file, person=r.person, photo_h=r.h, photo_w=r.w, n_faces=len(d_ok))
        if len(d_ok):
            j = int(np.argmax((d_ok[:, 2] - d_ok[:, 0]) * (d_ok[:, 3] - d_ok[:, 1])))
            fh = float(d_ok[j, 3] - d_ok[j, 1])
            row.update(score=float(d_ok[j, 4]), x1=float(d_ok[j, 0]), y1=float(d_ok[j, 1]), x2=float(d_ok[j, 2]), y2=float(d_ok[j, 3]), face_h=fh, usable48=bool(fh >= GAL_BAR), usable64=bool(fh >= GAL_BAR_SENS))
            kp = k[np.flatnonzero(d[:, 4] >= SCORE)[j]]
            row.update({f"kp{q}_{ax}": float(kp[q, m]) for q in range(5) for m, ax in enumerate("xy")})
        else:
            row.update(score=np.nan, x1=np.nan, y1=np.nan, x2=np.nan, y2=np.nan, face_h=np.nan, usable48=False, usable64=False)
        rows.append(row)
    g = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT / "gallery_faces.csv", index=False, float_format="%.3f")
    enrolled48, enrolled64 = int(g[g.usable48].person.nunique()), int(g[g.usable64].person.nunique())
    print(f"front photos {len(g)}, persons {g.person.nunique()}, usable at {GAL_BAR} px: {int(g.usable48.sum())} photos / {enrolled48} persons; at {GAL_BAR_SENS} px: {int(g.usable64.sum())} / {enrolled64}; seconds {time.time() - t0:.0f}")


# ----------------------------------------------------------------------------- embed (gated by section 0)
def stage_embed(a):
    if not a.i_confirm_section0_is_committed:
        sys.exit("refusing to embed: section 0 of REPORT_phase8.md must be committed first; rerun with --i-confirm-section0-is-committed")
    det, cfg, h = build_detector()
    from src.align import norm_crop
    from src.stratify.measures import laplacian_variance, luminance
    from src.video.embed import FaceEmbedder
    fe = FaceEmbedder(cfg, ("adaface",))
    # only eligible tracklets (>= 5 usable faces) are probes (M7): run `--stage eligibility` after `detect` first
    tr = pd.read_csv(OUT / "tracklets.csv")
    tr = tr[tr.detected.fillna(False).astype(bool) & tr.eligible.fillna(False).astype(bool)].reset_index(drop=True)
    names = {s: load_names(s) for s in ("train", "test")}

    def embed_crops(crops):
        e, raw = fe.recs["adaface"].get_embedding(crops, return_raw=True)
        mag = fe.mag.get_embedding(crops, return_raw=True)[1]
        return e, raw, mag, np.array([luminance(c) for c in crops], np.float32), np.array([laplacian_variance(c) for c in crops], np.float32)

    t0 = time.time()
    todo = tr if not a.limit else tr.head(a.limit)
    for n, r in enumerate(todo.itertuples(), 1):
        f, d = OUT / "emb" / f"emb_{r.tracklet}.npz", OUT / "emb" / f"det_{r.tracklet}.npz"
        if f.exists() or not d.exists():
            continue
        z = np.load(d)
        sel = np.flatnonzero(z["usable"])
        crops = []
        for q in sel:
            im = cv2.imread(str(chip_path(r.split, names[r.split][int(z["chip_idx"][q])])))
            crops.append(norm_crop(im, z["kps"][q]))
        if crops:
            e, raw, mag, lum, lap = embed_crops(crops)
        else:
            e, raw, mag, lum, lap = (np.zeros((0, 512), np.float32), *[np.zeros(0, np.float32)] * 4)
        atomic_savez(f, sel=sel, chip_idx=z["chip_idx"][sel], face_h=z["face_h"][sel], emb_adaface=e, adaface_norm=raw, mag_norm=mag, lum=lum, lap_var=lap)
        if n % 200 == 0:
            print(f"[{n}/{len(todo)}] {r.tracklet} elapsed {time.time() - t0:.0f}s", flush=True)
    # gallery faces (usable at >= 48 px, front photos)
    g = pd.read_csv(OUT / "gallery_faces.csv")
    g = g[g.usable48].reset_index(drop=True)
    crops = []
    for r in g.itertuples():
        im = cv2.imread(str(PHOTOS / r.file))
        kp = np.array([[getattr(r, f"kp{q}_x"), getattr(r, f"kp{q}_y")] for q in range(5)], np.float32)
        crops.append(norm_crop(im, kp))
    e, raw, mag, lum, lap = embed_crops(crops)
    atomic_savez(OUT / "gallery.npz", photo=g.file.to_numpy(), person=g.person.to_numpy(), face_h=g.face_h.to_numpy(np.float32), emb_adaface=e, adaface_norm=raw, mag_norm=mag, lum=lum, lap_var=lap)
    print("embed done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["index", "check2", "detect", "eligibility", "gallery", "embed"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N tracklets (smoke test; use with --out)")
    ap.add_argument("--split", choices=["train", "test"], default=None, help="detect: only this split")
    ap.add_argument("--out", default=None, help="write to this folder instead of outputs/video/mevid (smoke tests)")
    ap.add_argument("--i-confirm-section0-is-committed", action="store_true", help="required for --stage embed")
    a = ap.parse_args()
    if a.out:
        OUT = Path(a.out)
    {"index": stage_index, "check2": stage_check2, "detect": stage_detect, "eligibility": stage_eligibility, "gallery": stage_gallery, "embed": stage_embed}[a.stage](a)
