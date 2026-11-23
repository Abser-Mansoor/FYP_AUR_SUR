"""MEVID feasibility probe (protocol: research/Sprint_Notes/MEVID_feasibility_probe.md, pre-registered on `main` before step 1).

    envs/video/Scripts/python.exe scripts/mevid_probe.py --part inventory   # step 2: counts, chip sizes, check-in photos      -> inventory.json
    envs/video/Scripts/python.exe scripts/mevid_probe.py --part sample      # step 3: 200 tracklets x 20 frames, seeded         -> sample.csv   (commit BEFORE detect)
    envs/video/Scripts/python.exe scripts/mevid_probe.py --part detect      # step 4: SCRFD full640 on chips and check-in photos -> detections.csv, checkin_faces.csv, overlays/
    envs/video/Scripts/python.exe scripts/mevid_probe.py --part summary     # step 5: shares, percentiles, G1-G3, verdict         -> summary.json, probe_tables.md

Data availability only: no embedding, no recognition score, no identity score is computed anywhere in this script. Seed 42. MEVID images are never copied out of Dataset/MEVID/
except into the local, git-ignored overlays/ folder. Outputs: outputs/video/mevid_probe/.
Name formats (verified in the inventory): body chip `PPPPOoooCcccTtttFfffff.jpg` = person, outfit, camera, tracklet, frame, stored as bbox_test/<PPPP>/<name>; test_name.txt lists the
chips in order and track_test_info.txt gives per tracklet `start, end, person, outfit, camera` (start/end index into test_name.txt); query_IDX.txt lists the query tracklet indices.
Check-in photo `PPP-YYYYMMDD-NNN-SSS-v.jpg` (PPP the person ID, NNN a 3-character field that is `xxx` for 22 files, v = f/b or F/B front/back; one file name has a space before SSS).
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "Dataset/MEVID"
ANN = DATA / "mevid-v1-annotation-data"
PHOTOS = DATA / "mevid-v1-actor-checkin-photos"
CHIPS = DATA / "bbox_test"
OUT = ROOT / "outputs/video/mevid_probe"
REAL = OUT  # sample.csv and checkin_photos.csv are always read from here; --out only redirects what a part writes (smoke tests)
SEED = 42
README = dict(identities=54, outfits=113, tracklets=1754)  # MEVID README counts for the test split (plan, step 2)
N_TRACKLETS, PER_PERSON, N_FRAMES = 200, 3, 20
MIN_FACE_H, MIN_GALLERY_H, SCORE = 16, 64, 0.5
USABLE_PER_TRACKLET = 5
GO = dict(G1=0.30, G2=20, G3=27)  # thresholds fixed in the plan
CHIP_RE = re.compile(r"(\d{4})O(\d{3})C(\d{3})T(\d{3})F(\d{5})\.jpg$")
PHOTO_RE = re.compile(r"(\d{3})-(\d{8})-(\w{3})-\s*(\d{3})-([fbFB])\.jpg$")  # the third field is `xxx` in 22 names and one name has a space before the sequence number (found in the first inventory run)


def load_tracks():
    t = np.loadtxt(ANN / "track_test_info.txt").astype(int)
    df = pd.DataFrame(t, columns=["start", "end", "person", "outfit", "camera"])
    df.insert(0, "tracklet", np.arange(len(df)))
    df["n_chips"] = df.end - df.start + 1
    q = set(np.loadtxt(ANN / "query_IDX.txt").astype(int).tolist())
    df["is_query"] = df.tracklet.isin(q)
    return df


def load_names():
    return (ANN / "test_name.txt").read_text().split()


def chip_file(name):
    return CHIPS / name[:4] / name


def pct(a, q=(5, 25, 50, 75, 95)):
    return {f"p{p}": float(np.percentile(a, p)) for p in q}


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True))


# ---------------------------------------------------------------- step 2
def inventory():
    import cv2
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    tr, names = load_tracks(), load_names()
    inv = dict(readme_counts=README)
    found = dict(identities=int(tr.person.nunique()), outfits=int(tr[["person", "outfit"]].drop_duplicates().shape[0]), tracklets=int(len(tr)))
    inv["found_counts"] = found
    inv["counts_match_readme"] = bool(found == README)
    inv["cameras"] = int(tr.camera.nunique())
    inv["query_tracklets"], inv["gallery_tracklets"] = int(tr.is_query.sum()), int((~tr.is_query).sum())
    inv["chips_total_in_test_name_txt"] = len(names)
    inv["chips_per_tracklet"] = dict(min=int(tr.n_chips.min()), max=int(tr.n_chips.max()), mean=float(tr.n_chips.mean()), **pct(tr.n_chips))
    # internal consistency of the annotation files
    contiguous = bool((tr.start.iloc[1:].to_numpy() == tr.end.iloc[:-1].to_numpy() + 1).all() and tr.start.iloc[0] == 0 and tr.end.iloc[-1] == len(names) - 1)
    bad = 0
    for r in tr.itertuples():
        for i in (r.start, r.end):
            m = CHIP_RE.match(names[i])
            bad += not (m and (int(m.group(1)), int(m.group(2)), int(m.group(3))) == (r.person, r.outfit, r.camera))
    inv["tracklet_ranges_contiguous_and_cover_test_name_txt"] = contiguous
    inv["tracklet_endpoints_whose_chip_name_disagrees_with_the_info_row"] = int(bad)
    # extraction completeness
    on_disk = sum(1 for d in CHIPS.iterdir() if d.is_dir() for _ in d.iterdir())
    inv["chip_files_on_disk"] = on_disk
    inv["chip_files_on_disk_equal_test_name_txt"] = bool(on_disk == len(names))
    # body-chip sizes on a seeded sample of 2,000 chips
    idx = np.sort(np.random.default_rng(SEED).choice(len(names), 2000, replace=False))
    hw, missing = [], 0
    for i in idx:
        im = cv2.imread(str(chip_file(names[i])))
        if im is None:
            missing += 1
            continue
        hw.append(im.shape[:2])
    hw = np.array(hw)
    inv["chip_size_sample"] = dict(n=int(len(hw)), unreadable_or_missing=int(missing), height=pct(hw[:, 0]), width=pct(hw[:, 1]))
    # check-in photos
    files = sorted(p.name for p in PHOTOS.iterdir() if p.is_file())
    rows, unparsed = [], []
    for f in files:
        m = PHOTO_RE.match(f)
        if not m:
            unparsed.append(f)
            continue
        with Image.open(PHOTOS / f) as im:
            w, h = im.size
        rows.append(dict(file=f, person=int(m.group(1)), date=m.group(2), view=m.group(5).lower(), w=w, h=h))
    ph = pd.DataFrame(rows)
    test_p, train_p = sorted(tr.person.unique().tolist()), sorted(np.loadtxt(ANN / "track_train_info.txt").astype(int)[:, 2].tolist())
    train_p = sorted(set(train_p))
    cnt = ph.groupby("person").size()
    front = ph[ph.view == "f"].groupby("person").size()
    inv["checkin"] = dict(
        files=len(files), unparsed_file_names=unparsed, persons_with_photos=int(ph.person.nunique()), photos_per_person=pct(cnt, (0, 25, 50, 75, 100)),
        view_counts={k: int(v) for k, v in ph.view.value_counts().items()}, image_height=pct(ph.h), image_width=pct(ph.w),
        mapping_rule="person ID = integer value of the 3-digit file-name prefix (found by naming convention)",
        mapping_evidence=dict(test_persons=len(test_p), test_persons_that_occur_as_a_photo_prefix=int(sum(p in set(cnt.index) for p in test_p)),
                              train_persons=len(train_p), train_persons_that_occur_as_a_photo_prefix=int(sum(p in set(cnt.index) for p in train_p)),
                              persons_in_train_plus_test=len(set(test_p) | set(train_p))),
        test_persons_with_at_least_one_photo=int(sum(p in set(cnt.index) for p in test_p)), test_persons_with_at_least_one_front_photo=int(sum(p in set(front.index) for p in test_p)),
        photos_of_test_persons=int(ph.person.isin(test_p).sum()))
    ph.to_csv(OUT / "checkin_photos.csv", index=False)
    dump(inv, OUT / "inventory.json")
    print(json.dumps(inv, indent=1, sort_keys=True)[:3000])


# ---------------------------------------------------------------- step 3
def sample():
    tr, names = load_tracks(), load_names()
    rng = np.random.default_rng(SEED)
    chosen, short = [], 0
    for p, g in tr.groupby("person"):
        k = min(PER_PERSON, len(g))
        short += PER_PERSON - k
        chosen += rng.choice(g.tracklet.to_numpy(), k, replace=False).tolist()
    rest = tr.loc[~tr.tracklet.isin(chosen), "tracklet"].to_numpy()
    need = N_TRACKLETS - len(chosen)
    chosen += rng.choice(rest, need, replace=False).tolist()
    rows = []
    for t in sorted(chosen):
        r = tr.iloc[t]
        n = int(r.n_chips)
        pos = np.unique(np.round(np.linspace(r.start, r.end, min(N_FRAMES, n))).astype(int))
        for k, i in enumerate(pos):
            rows.append(dict(tracklet=int(t), person=int(r.person), outfit=int(r.outfit), camera=int(r.camera), is_query=bool(r.is_query), n_chips=n, position=k, chip_idx=int(i), chip_file=names[i]))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "sample.csv", index=False)
    print(f"{df.tracklet.nunique()} tracklets, {df.person.nunique()} persons, {len(df)} frames; shortfall redistributed: {short}; per person min/max: {df.groupby('person').tracklet.nunique().min()}/{df.groupby('person').tracklet.nunique().max()}")


# ---------------------------------------------------------------- step 4
def detect():
    import cv2
    import torch  # noqa: F401  (before onnxruntime creates a CUDA session)
    import yaml
    from src.video.pipeline import build_detector
    cfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    wpath = ROOT / cfg["detectors"]["scrfd"]
    h = hashlib.sha256(wpath.read_bytes()).hexdigest()
    assert h in (ROOT / "MODEL_INVENTORY.md").read_text(), "det_10g.onnx does not match MODEL_INVENTORY.md"
    det = build_detector(cfg)
    sm = pd.read_csv(REAL / "sample.csv")
    if ARGS.only_existing:
        sm = sm[sm.chip_file.map(lambda f: chip_file(f).exists())]
    if ARGS.limit:
        sm = sm.head(ARGS.limit)
    t0 = time.time()
    rows = []
    for r in sm.itertuples():
        im = cv2.imread(str(chip_file(r.chip_file)))
        if im is None:
            rows.append(dict(chip_file=r.chip_file, tracklet=r.tracklet, person=r.person, camera=r.camera, position=r.position, chip_h=np.nan, chip_w=np.nan, n_det05=0, face=False,
                             score=np.nan, x1=np.nan, y1=np.nan, x2=np.nan, y2=np.nan, face_h=np.nan, usable=False, unreadable=True))
            continue
        ch, cw = im.shape[:2]
        d, _ = det.detect(im, size=640)
        d = d[d[:, 4] >= SCORE]
        ok = d[((d[:, 1] + d[:, 3]) / 2) < 0.5 * ch] if len(d) else d  # centre in the upper 50% of the chip
        row = dict(chip_file=r.chip_file, tracklet=r.tracklet, person=r.person, camera=r.camera, position=r.position, chip_h=ch, chip_w=cw, n_det05=len(d), unreadable=False)
        if len(ok):
            b = ok[int(np.argmax(ok[:, 4]))]
            fh = float(b[3] - b[1])
            row.update(face=True, score=float(b[4]), x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), face_h=fh, usable=bool(fh >= MIN_FACE_H))
        else:
            row.update(face=False, score=np.nan, x1=np.nan, y1=np.nan, x2=np.nan, y2=np.nan, face_h=np.nan, usable=False)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "detections.csv", index=False, float_format="%.3f")
    # check-in photos of the 54 test persons
    ph = pd.read_csv(REAL / "checkin_photos.csv")
    test_p = set(load_tracks().person.unique().tolist())
    prows = []
    for r in (ph[ph.person.isin(test_p)].head(8) if ARGS.limit else ph[ph.person.isin(test_p)]).itertuples():
        im = cv2.imread(str(PHOTOS / r.file))
        d, _ = det.detect(im, size=640)
        d = d[d[:, 4] >= SCORE]
        row = dict(file=r.file, person=r.person, view=r.view, w=r.w, h=r.h, n_faces=len(d))
        if len(d):
            b = d[int(np.argmax((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])))]
            row.update(score=float(b[4]), x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), face_h=float(b[3] - b[1]), usable_gallery=bool(b[3] - b[1] >= MIN_GALLERY_H))
        else:
            row.update(score=np.nan, x1=np.nan, y1=np.nan, x2=np.nan, y2=np.nan, face_h=np.nan, usable_gallery=False)
        prows.append(row)
    pd.DataFrame(prows).to_csv(OUT / "checkin_faces.csv", index=False, float_format="%.3f")
    # overlays: 20 seeded chips and 10 seeded photos, LOCAL ONLY (git-ignored)
    ov = OUT / "overlays"
    ov.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    dt = pd.DataFrame(rows)
    for k, i in enumerate(sorted(rng.choice(len(dt), min(20, len(dt)), replace=False))):
        r = dt.iloc[i]
        im = cv2.imread(str(chip_file(r.chip_file)))
        im = cv2.resize(im, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        if r.face:
            cv2.rectangle(im, (int(r.x1 * 3), int(r.y1 * 3)), (int(r.x2 * 3), int(r.y2 * 3)), (0, 255, 0) if r.usable else (0, 165, 255), 2)
        cv2.putText(im, f"{'face h=%.0f' % r.face_h if r.face else 'no face'}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(ov / f"chip_{k:02d}.jpg"), im)
    pf = pd.DataFrame(prows)
    for k, i in enumerate(sorted(rng.choice(len(pf), min(10, len(pf)), replace=False))):
        r = pf.iloc[i]
        im = cv2.imread(str(PHOTOS / r.file))
        if not np.isnan(r.x1):
            cv2.rectangle(im, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 255, 0), 4)
        f = 800 / max(im.shape[:2])
        cv2.imwrite(str(ov / f"photo_{k:02d}.jpg"), cv2.resize(im, None, fx=f, fy=f))
    dump(dict(det_10g_sha256=h, weights_checked_against="MODEL_INVENTORY.md", chips_processed=len(rows), photos_processed=len(prows), seconds=round(time.time() - t0, 1), detector="SCRFD det_10g, official preprocessing, size 640"), OUT / "detect_meta.json")
    print("detections:", len(rows), "photos:", len(prows), "seconds", round(time.time() - t0, 1))


# ---------------------------------------------------------------- step 5
def boot(fn, df, key, n=1000):
    """Cluster bootstrap over persons: resample persons with replacement (all their rows), recompute `fn`."""
    rng = np.random.default_rng(SEED)
    groups = {k: g for k, g in df.groupby(key)}
    keys = np.array(sorted(groups))
    vals = []
    for _ in range(n):
        pick = keys[rng.integers(0, len(keys), len(keys))]
        vals.append(fn(pd.concat([groups[k] for k in pick], ignore_index=True)))
    v = np.array(vals, float)
    return [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]


def summary():
    d = pd.read_csv(OUT / "detections.csv")
    c = pd.read_csv(OUT / "checkin_faces.csv")
    sm = pd.read_csv(OUT / "sample.csv")
    inv = json.loads((OUT / "inventory.json").read_text())
    d["usable"] = d.usable.astype(bool)
    tk = d.groupby(["tracklet", "person", "camera"]).agg(n=("chip_file", "size"), usable=("usable", "sum")).reset_index()
    tk["ok"] = tk.usable >= USABLE_PER_TRACKLET
    s = dict(inventory_counts_match_readme=inv["counts_match_readme"], n_tracklets_sampled=int(len(tk)), n_chips_sampled=int(len(d)), n_unreadable_chips=int(d.unreadable.sum()))
    share = lambda x, col: float(x[col].mean())  # noqa: E731
    s["share_chips_with_a_face"] = dict(value=share(d, "face"), ci95=boot(lambda x: x.face.mean(), d, "person"))
    s["share_chips_with_a_usable_face"] = dict(value=share(d, "usable"), ci95=boot(lambda x: x.usable.mean(), d, "person"))
    fh = d.loc[d.face, "face_h"]
    s["face_height_px_of_chips_with_a_face"] = dict(n=int(len(fh)), **pct(fh))
    s["G1_share_of_tracklets_with_at_least_5_usable_faces"] = dict(value=float(tk.ok.mean()), ci95=boot(lambda x: x.ok.mean(), tk, "person"), n_tracklets=int(len(tk)), threshold=GO["G1"])
    s["G1_by_camera"] = {str(k): dict(tracklets=int(len(g)), value=float(g.ok.mean())) for k, g in tk.groupby("camera")}
    g2 = int(tk.groupby("person").ok.any().sum())
    s["G2_persons_with_at_least_one_usable_tracklet"] = dict(value=g2, of=int(inv["found_counts"]["identities"]), threshold=GO["G2"],
                                                           note="all 54 persons are enumerated; the only randomness is the tracklet sample, so no person-level CI is given")
    cp = c.groupby("person").usable_gallery.any()
    g3 = int(cp.sum())
    s["G3_test_persons_with_at_least_one_usable_checkin_photo"] = dict(value=g3, of=int(inv["found_counts"]["identities"]), threshold=GO["G3"], photos_processed=int(len(c)),
                                                                      share_of_photos_usable=float(c.usable_gallery.mean()), photo_face_height_px=pct(c.face_h.dropna()),
                                                                      photos_with_no_face=int((c.n_faces == 0).sum()), persons_without_any_photo=int(inv["found_counts"]["identities"] - c.person.nunique()))
    g1ok, g2ok, g3ok = s["G1_share_of_tracklets_with_at_least_5_usable_faces"]["value"] >= GO["G1"], g2 >= GO["G2"], g3 >= GO["G3"]
    s["criteria_pass"] = dict(G1=bool(g1ok), G2=bool(g2ok), G3=bool(g3ok))
    s["verdict"] = ("GO (high-resolution gallery route)" if g1ok and g2ok and g3ok else "GO (tracklet-only route)" if g1ok and g2ok else "NO-GO")
    s["rule"] = "G1 >= 30% of tracklets with >= 5 usable faces of 20; G2 >= 20 of 54 persons with a usable tracklet; G3 >= 27 of 54 test persons with a usable check-in photo (fixed in the plan)"
    dump(s, OUT / "summary.json")
    L = ["# MEVID probe: tables (generated by scripts/mevid_probe.py --part summary)", "", f"Verdict: **{s['verdict']}**", ""]
    L += ["| Quantity | Value | 95% CI (cluster bootstrap over persons) |", "|---|---|---|"]
    for k, lab in (("share_chips_with_a_face", "Sampled chips with a face"), ("share_chips_with_a_usable_face", "Sampled chips with a usable face (score >= 0.5, height >= 16 px)"),
                   ("G1_share_of_tracklets_with_at_least_5_usable_faces", "G1: tracklets with >= 5 usable faces of 20")):
        L.append(f"| {lab} | {s[k]['value']:.3f} | {s[k]['ci95'][0]:.3f} to {s[k]['ci95'][1]:.3f} |")
    L += ["", "Face height (native chip pixels), chips with a face: " + ", ".join(f"{k} {v:.1f}" for k, v in s["face_height_px_of_chips_with_a_face"].items() if k != "n"), "",
          "| Criterion | Value | Threshold | Pass |", "|---|---|---|---|",
          f"| G1 share of usable tracklets | {s['G1_share_of_tracklets_with_at_least_5_usable_faces']['value']:.3f} | {GO['G1']} | {g1ok} |",
          f"| G2 persons with a usable tracklet | {g2} of 54 | {GO['G2']} | {g2ok} |", f"| G3 test persons with a usable check-in photo | {g3} of 54 | {GO['G3']} | {g3ok} |", "",
          "| Camera | Tracklets sampled | Share with >= 5 usable faces |", "|---|---|---|"]
    L += [f"| C{k} | {v['tracklets']} | {v['value']:.3f} |" for k, v in s["G1_by_camera"].items()]
    (OUT / "probe_tables.md").write_text("\n".join(L) + "\n")
    print(json.dumps(s, indent=1, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["inventory", "sample", "detect", "summary"])
    ap.add_argument("--out", default=None, help="write to this folder instead of outputs/video/mevid_probe (smoke tests)")
    ap.add_argument("--limit", type=int, default=None, help="detect: only the first N sampled chips and 8 photos (smoke test)")
    ap.add_argument("--only-existing", action="store_true", help="detect: only chips that are already extracted (smoke test)")
    ARGS = ap.parse_args()
    a = ARGS
    if a.out:
        OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    {"inventory": inventory, "sample": sample, "detect": detect, "summary": summary}[a.part]()
