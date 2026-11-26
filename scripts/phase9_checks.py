"""Phase 9 acceptance checks (REPORT_phase9.md section 0.7). Results go to outputs/video/phase9_checks.json (one key per check, merged).

    envs/caface/Scripts/python.exe scripts/phase9_checks.py --check 1a   # the 100 seeded MEVID crops made with envs/caface's OpenCV (as the Phase 8 cf_*.npz features)
    envs/video/Scripts/python.exe scripts/phase9_checks.py --check 1     # GPU: WebFace4M via src/recognizer.py (alias adaface_wf4m) on those crops vs the Phase 8 features
    envs/video/Scripts/python.exe scripts/phase9_checks.py --check 2     # lambda 0 tracker table == Phase 7 (CPU) and the default embedder == the stored Phase 6 embeddings (GPU)
    envs/video/Scripts/python.exe scripts/phase9_checks.py --check 4     # unit test of the quality bar (src/video/identify.py)
    envs/video/Scripts/python.exe scripts/phase9_checks.py --check 5     # FAISS Flat top-1 == exact NumPy search (synthetic and, if present, the MEVID templates)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "outputs/video/phase9_checks.json"
MEVID = ROOT / "outputs/video/mevid"


def save(key, res):
    allres = json.loads(OUT.read_text()) if OUT.exists() else {}
    allres[key] = res
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(allres, indent=1, sort_keys=True))
    print(json.dumps({key: res}, indent=1, sort_keys=True))
    return res


CROPS = ROOT / "outputs/video/phase9/check1_crops_{env}.npz"  # MEVID face crops: local only (gitignored)


def crops100():
    """100 seeded MEVID usable faces (one per tracklet), aligned with this environment's OpenCV; with the Phase 8 cf_ feature of each."""
    import cv2
    import pandas as pd
    import mevid_embed as M
    from src.align import norm_crop
    tr = pd.read_csv(MEVID / "tracklets.csv")
    tr = tr[tr.detected.fillna(False).astype(bool) & tr.eligible.fillna(False).astype(bool)].reset_index(drop=True)
    rng = np.random.default_rng(42)
    pick = tr.iloc[np.sort(rng.choice(len(tr), 100, replace=False))]
    names = {s: M.load_names(s) for s in ("train", "test")}
    crops, ref = [], []
    for r in pick.itertuples():
        z, e, c = (np.load(MEVID / "emb" / f"{p}_{r.tracklet}.npz") for p in ("det", "emb", "cf"))
        j = int(rng.integers(0, len(e["sel"])))
        q = int(e["sel"][j])
        crops.append(norm_crop(cv2.imread(str(M.chip_path(r.split, names[r.split][int(z["chip_idx"][q])]))), z["kps"][q]))
        ref.append(c["feat"][j] / np.linalg.norm(c["feat"][j]))
    return np.stack(crops), np.stack(ref), cv2.__version__


def check1a():
    """Run in envs/caface: the crops made with that environment's OpenCV (the one that made the Phase 8 cf_ features)."""
    crops, ref, ver = crops100()
    CROPS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(CROPS).format(env="caface"), crops=crops, ref=ref, opencv=ver)
    print("saved", crops.shape, "opencv", ver)
    return dict(passed=True)


def check1():
    """Run in envs/video after check1a: the pipeline recognizer on (i) the envs/caface crops (the check: same network) and (ii) crops made here (reported: the OpenCV effect)."""
    import torch  # noqa: F401
    import yaml
    from src.video.embed import FaceEmbedder
    cfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    fe = FaceEmbedder(cfg, ("adaface_wf4m",), quality=False)
    cf = np.load(str(CROPS).format(env="caface"))
    own, ref, ver = crops100()
    cos_cf = (fe.embed_crops(list(cf["crops"]))["emb"]["adaface_wf4m"] * cf["ref"]).sum(1)
    cos_own = (fe.embed_crops(list(own))["emb"]["adaface_wf4m"] * ref).sum(1)
    d = np.abs(own.astype(int) - cf["crops"].astype(int))
    return save("check1_webface4m_recognizer_vs_phase8_features", dict(
        n_faces=len(cos_cf), passed=bool(cos_cf.min() > 0.9999),
        same_crops_cosine_min=float(cos_cf.min()), same_crops_cosine_median=float(np.median(cos_cf)),
        envs_video_crops_cosine_min=float(cos_own.min()), envs_video_crops_cosine_median=float(np.median(cos_own)),
        crop_pixel_diff_mean=float(d.mean()), crop_pixel_diff_max=int(d.max()), opencv_envs_caface=str(cf["opencv"]), opencv_envs_video=ver,
        note="network check on identical crops (section 0.7, amended before the section 0 commit); the OpenCV difference is why Phase 9 re-embeds WebFace4M in envs/video"))


def check2():
    """(a) The tracker code at lambda 0 reproduces the Phase 7 per-clip tracker table byte for byte (CPU). (b) FaceEmbedder with its defaults (MS1MV2, flip on) gives exactly the
    stored Phase 6 embeddings on 100 seeded ChokePoint faces (GPU), i.e. the Phase 9 alias/quality switches change nothing by default."""
    import hashlib
    import cv2
    import pandas as pd
    import torch  # noqa: F401
    import yaml
    import video_tracker_stress as V
    from src.align import norm_crop
    from src.video.embed import FaceEmbedder
    from src.video.io import list_frames
    cfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    gt = V.load_gt()
    rows = [pd.DataFrame(V.run_clip(clip, ci, gt, cfg, lam=0.0, embs=None, assign_out={})) for ci, clip in enumerate(sorted(gt))]
    f = ROOT / "outputs/video/phase9/check2_lam0_per_clip.csv"
    pd.concat(rows, ignore_index=True).to_csv(f, index=False, float_format="%.6f")
    ref = ROOT / "outputs/video/agg7/tracker_stress/per_clip.csv"
    ha, hb = hashlib.sha256(ref.read_bytes()).hexdigest(), hashlib.sha256(f.read_bytes()).hexdigest()
    pipe = ROOT / "outputs/video/pipeline/chokepoint"
    clips = sorted(p.name for p in pipe.iterdir() if (p / "embed.npz").exists())
    rng = np.random.default_rng(42)
    fe = FaceEmbedder(cfg, ("adaface",))
    same = []
    for c in rng.choice(clips, 20, replace=False):
        z, dz = np.load(pipe / c / "embed.npz"), np.load(pipe / c / "detections.npz")
        for j in rng.choice(len(z["det_id"]), 5, replace=False):
            d = int(z["det_id"][j])
            fr = int(dz["frame_idx"][d])
            src = ROOT / "Dataset/ChokePoint" / c
            im = cv2.imread(str(dict(list_frames(src))[fr]))
            r = fe.embed_crops([norm_crop(im, dz["kps"][d])])
            same.append(bool(np.array_equal(r["emb"]["adaface"][0], z["emb_adaface"][j]) and np.array_equal(r["mag_norm"][0], z["mag_norm"][j])))
    return save("check2_phase6_7_switches_reproduce", dict(lambda0_per_clip_sha256=hb, phase7_per_clip_sha256=ha, lambda0_byte_identical=bool(ha == hb),
                                                         default_embedder_identical_faces=int(sum(same)), n_faces=len(same), passed=bool(ha == hb and all(same))))


def check4():
    from src.video.identify import LOW, UNKNOWN, decide
    cases = [(22.99, 0.9, 7, 0.3, LOW), (23.0, 0.9, 7, 0.3, 7), (23.0, 0.2, 7, 0.3, UNKNOWN), (float("nan"), 0.9, 7, 0.3, LOW), (40.0, 0.3, 7, 0.3, 7), (5.0, 0.99, 7, 0.0, LOW)]
    got = [decide(h, s, t, thr, 23.0) for h, s, t, thr, _ in cases]
    rng = np.random.default_rng(42)
    h = rng.uniform(10, 60, 10000)
    lab = [decide(x, 1.0, 1, 0.0, 23.0) for x in h]
    withheld_iff_below = bool(all((label == LOW) == (x < 23.0) for x, label in zip(h, lab)))
    return save("check4_quality_bar_unit_test", dict(cases_passed=bool(got == [c[-1] for c in cases]), random_heights_withheld_iff_below_23px=withheld_iff_below, n_random=len(h),
                                                     passed=bool(got == [c[-1] for c in cases] and withheld_iff_below)))


def check5():
    from src.video.index import GalleryIndex
    rng = np.random.default_rng(42)
    res = {}
    for n in (100, 1000, 10000):
        g = rng.normal(size=(n, 512)).astype(np.float32)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        q = rng.normal(size=(2000, 512)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        _, rows = GalleryIndex(g, "flat").search(q, 1)
        exact = (q @ g.T).argmax(1)
        res[f"synthetic_{n}"] = bool(np.array_equal(rows[:, 0], exact))
    return save("check5_faiss_flat_equals_exact_search", dict(**res, passed=bool(all(res.values())), note="the Part E probes are checked again in scripts/phase9_retrieval.py"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["1a", "1", "2", "4", "5"])
    a = ap.parse_args()
    r = {"1a": check1a, "1": check1, "2": check2, "4": check4, "5": check5}[a.check]()
    sys.exit(0 if r.get("passed") else 1)
