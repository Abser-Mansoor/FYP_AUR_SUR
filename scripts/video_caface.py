"""Phase 7 CAFace arm (REPORT_phase7.md section 0.6): the WebFace4M AdaFace backbone and the CAFace aggregator of github.com/mk-minchul/CAFace (commit e94ad02), inference only.

    envs/caface/Scripts/python.exe scripts/video_caface.py --selftest            # load the checkpoint, run a few ChokePoint crops, check the interface
Run with envs/caface (torch 2.14.1+cu126). The repository code is at models/caface/CAFace_src (cloned at the pinned commit; it is not part of this repository).

Interface (read from demo/inference.py and demo/model_loader.py): backbone(img[B,3,112,112], BGR, (x/255 - 0.5)/0.5) -> (feature [B,512] L2-normalised, norm [B,1], style) with
`return_style=hparam.style_index` ([3, 5]); `aggregator.compress_intermediate(style)` -> [B, C]; fused feature = aggregator(features * norms [1,T,512], compressed [1,T,C])['aggregated'],
chunked by 512 frames with `aggregator.use_memory = True` and `memory_count` reset (as `aggregator_caface_fuse` in demo/inference.py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models/caface/CAFace_src"
CKPT = ROOT / "models/caface/CAFace_AdaFaceWebFace4M.ckpt"
sys.path.append(str(ROOT))  # for src.align
for p in (SRC, SRC / "caface"):
    sys.path.insert(0, str(p))


class _Wrapper(torch.nn.Module):
    def __init__(self, fr_model, aggregator_model):
        super().__init__()
        self.model = fr_model
        self.aggregator = aggregator_model


def load_caface(device="cuda"):
    from omegaconf import OmegaConf
    from caface.model import build_model
    from caface.nets import aggregator as agg_mod
    from caface.nets import fusion_net
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    hp = OmegaConf.create(ckpt["hyper_parameters"])
    backbone = build_model(hp["arch"])
    fusion = fusion_net.build_fusion_net(hp["decoder_name"], hp)
    aggregator = agg_mod.build_aggregator(model_name=hp.aggregator_name, fusion_net=fusion, style_index=",".join(str(i) for i in hp.style_index),
                                          input_maker_outdim=fusion.config["intermediate_outdim"], use_memory=hp.use_memory, normemb_dim=fusion.config["norm_dim"])
    w = _Wrapper(backbone, aggregator)
    r = w.load_state_dict(ckpt["state_dict"], strict=False)
    assert len(r.missing_keys) == 0 and list(r.unexpected_keys) == ["center.weight"], (r.missing_keys, r.unexpected_keys)
    w.model.to(device).eval()
    w.aggregator.to(device).eval()
    return w.aggregator, w.model, hp


def to_tensor(bgr_crops, device="cuda"):
    """List of aligned 112 x 112 BGR uint8 crops -> [B,3,112,112] float, normalised as the AdaFace family (x/255 - 0.5)/0.5, channel order unchanged (BGR in, as the demo's dataset)."""
    x = torch.from_numpy(np.stack(bgr_crops)).to(device).permute(0, 3, 1, 2).contiguous().float() / 255.0
    return (x - 0.5) / 0.5


@torch.no_grad()
def embed(crops, backbone, aggregator, hp, device="cuda", batch=64):
    """-> feature [N,512] (L2-normalised), norm [N], compressed style [N,C] (float32 numpy)."""
    F, Nn, I = [], [], []
    for s in range(0, len(crops), batch):
        feat, norm, style = backbone(to_tensor(crops[s:s + batch], device), return_style=hp.style_index)
        F.append(feat.float().cpu().numpy())
        Nn.append(norm.float().reshape(-1).cpu().numpy())
        I.append(aggregator.compress_intermediate(style).float().cpu().numpy())
    return np.concatenate(F), np.concatenate(Nn), np.concatenate(I)


@torch.no_grad()
def fuse(feat, norm, interm, aggregator, device="cuda", max_element=512):
    """CAFace fusion of one track (as demo/inference.py `aggregator_caface_fuse`): returns the L2-normalised fused feature [512]. A track of one frame returns its own feature."""
    if len(feat) == 1:
        return feat[0] / (np.linalg.norm(feat[0]) + 1e-8)
    f = torch.tensor(feat * norm[:, None]).float().to(device).unsqueeze(0)
    it = torch.tensor(interm).float().to(device).unsqueeze(0)
    aggregator.use_memory = True
    fused = None
    for k, split in enumerate(np.array_split(np.arange(f.shape[1]), int(np.ceil(f.shape[1] / max_element)))):
        sf, si = f[:, split, :], it[:, split, :]
        bad = sf.isnan().any(0).any(-1)
        sf, si = sf[:, ~bad, :], si[:, ~bad, :]
        if k == 0:
            aggregator.memory_count = 0
        fused = aggregator(sf, si)["aggregated"].squeeze(0).detach().cpu().numpy()
    aggregator.memory_count = 0
    return fused / (np.linalg.norm(fused) + 1e-8)


def selftest(a):
    import cv2
    agg, model, hp = load_caface()
    print("loaded; arch", hp["arch"], "intermediate_type", hp.intermediate_type, "style_index", list(hp.style_index))
    imgs = sorted((ROOT / "outputs/video/chokepoint_sample").rglob("*.png"))[:6] if (ROOT / "outputs/video/chokepoint_sample").exists() else []
    crops = [cv2.resize(cv2.imread(str(p)), (112, 112)) for p in imgs] or [np.random.default_rng(i).integers(0, 255, (112, 112, 3), dtype=np.uint8) for i in range(6)]
    feat, norm, interm = embed(crops, model, agg, hp)
    print("feature", feat.shape, "norm", norm.shape, "style", interm.shape, "norms", np.round(norm, 2).tolist())
    print("feature unit norm:", bool(np.allclose(np.linalg.norm(feat, axis=1), 1, atol=1e-4)))
    fused = fuse(feat, norm, interm, agg)
    print("fused", fused.shape, "unit", round(float(np.linalg.norm(fused)), 4), "cos(fused, mean feature)", round(float(fused @ (feat.mean(0) / np.linalg.norm(feat.mean(0)))), 4))
    one = fuse(feat[:1], norm[:1], interm[:1], agg)
    print("length-1 track equals its own feature:", bool(np.allclose(one, feat[0], atol=1e-5)))


DATA = ROOT / "Dataset/ChokePoint"
PIPE = ROOT / "outputs/video/pipeline/chokepoint"
OUT = ROOT / "outputs/video/caface"
SIZES = (16, 12)


def degrade(crop, s):
    """Phase 6 protocol E: area downsampling to s x s, bilinear upsampling back to 112 x 112."""
    import cv2
    return cv2.resize(cv2.resize(crop, (s, s), interpolation=cv2.INTER_AREA), (112, 112), interpolation=cv2.INTER_LINEAR)


def embed_chokepoint(a):
    """CAFace backbone on the matched ChokePoint detections: native, and the every-5th-matched-detection subsample degraded to 16 and 12 px (the Phase 6 E subsample).
    Per clip: features and norms, and the CAFace-fused feature of every GT track (clip, person id) at each size. Cache: outputs/video/caface/emb/<clip>.npz."""
    import cv2
    import pandas as pd
    from src.align import norm_crop
    agg, model, hp = load_caface()
    m = pd.read_csv(ROOT / "outputs/video/baseline/matches.csv")
    (OUT / "emb").mkdir(parents=True, exist_ok=True)
    clips = a.clips or sorted(m["clip"].unique())[: a.limit]
    for k, clip in enumerate(clips, 1):
        f = OUT / "emb" / f"{clip}.npz"
        if f.exists():
            continue
        s = m[m["clip"] == clip].sort_values(["frame", "det_id"]).reset_index(drop=True)
        kps = np.load(PIPE / clip / "detections.npz")["kps"]
        crops = []
        for fno, g in s.groupby("frame", sort=True):
            bgr = cv2.imread(str(DATA / clip / f"{int(fno):08d}.jpg"))
            crops += [norm_crop(bgr, kps[int(d)]) for d in g.det_id]  # groupby keeps the (frame, det_id) order of s
        pid = s.pid.to_numpy()
        res = dict(det_id=s.det_id.to_numpy(), pid=pid, frame=s.frame.to_numpy())

        def run(cr, pids, tag):
            feat, norm, interm = embed(cr, model, agg, hp)
            tp = np.unique(pids)
            res[f"feat{tag}"], res[f"norm{tag}"] = feat, norm
            res[f"track_pids{tag}"] = tp
            res[f"fused{tag}"] = np.stack([fuse(feat[pids == p_], norm[pids == p_], interm[pids == p_], agg) for p_ in tp])

        run(crops, pid, "112")
        sub = np.arange(0, len(s), 5)
        res["sub_idx"] = sub
        for size in SIZES:
            run([degrade(crops[i], size) for i in sub], pid[sub], str(size))
        tmp = f.with_name(f.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, **res)
        tmp.replace(f)
        print(f"[{k}/{len(clips)}] {clip}: {len(s)} faces, tracks {len(res['track_pids112'])}", flush=True)


def embed_ytf(a):
    """CAFace backbone on the accepted YouTube Faces frames (the detections of scripts/video_ytf_embed.py: kps, frame numbers). Per video: features, norms and the CAFace-fused feature."""
    import cv2
    import pandas as pd
    from src.align import norm_crop
    agg, model, hp = load_caface()
    fr = pd.read_csv(ROOT / "outputs/video/ytf/frames.csv", usecols=["video_idx", "video"]).drop_duplicates("video_idx").set_index("video_idx").video
    src = ROOT / "outputs/video/ytf/embed"
    (OUT / "ytf").mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npz"), key=lambda p: int(p.stem))[: a.limit]
    for k, fpath in enumerate(files, 1):
        v = int(fpath.stem)
        f = OUT / "ytf" / f"{v}.npz"
        if f.exists():
            continue
        z = np.load(fpath)
        if len(z["frame_no"]) == 0:
            continue
        person, vid = fr[v].split("/")
        crops = []
        for fno, kp in zip(z["frame_no"], z["kps"]):
            bgr = cv2.imread(str(ROOT / "Dataset/YouTubeFaces/frame_images_DB" / person / vid / f"{vid}.{int(fno)}.jpg"))
            crops.append(norm_crop(bgr, kp))
        feat, norm, interm = embed(crops, model, agg, hp)
        tmp = f.with_name(f.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, frame_no=z["frame_no"], feat=feat, norm=norm, fused=fuse(feat, norm, interm, agg))
        tmp.replace(f)
        if k % 100 == 0:
            print(f"[{k}/{len(files)}] video {v}", flush=True)


def check5_features(a):
    """Acceptance check 5, step 1 (envs/caface): 100 seeded matched ChokePoint faces; the aligned crops and the CAFace backbone features are saved."""
    import cv2
    import pandas as pd
    from src.align import norm_crop
    m = pd.read_csv(ROOT / "outputs/video/baseline/matches.csv")
    pick = m.iloc[np.random.default_rng(42).choice(len(m), 100, replace=False)].sort_values(["clip", "frame"])
    crops = []
    for r in pick.itertuples():
        kps = np.load(PIPE / r.clip / "detections.npz")["kps"]
        crops.append(norm_crop(cv2.imread(str(DATA / r.clip / f"{int(r.frame):08d}.jpg")), kps[int(r.det_id)]))
    agg, model, hp = load_caface()
    feat, norm, _ = embed(crops, model, agg, hp)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "check5_cafaceside.npz", crops=np.stack(crops), feat=feat, norm=norm)
    print("saved", feat.shape)


def check5_compare(a):
    """Acceptance check 5, step 2 (envs/video): the same crops through our Recognizer path with the standalone AdaFace WebFace4M checkpoint (no flip), cosine with the CAFace backbone features."""
    import json
    import yaml
    from src.recognizer import Recognizer
    z = np.load(OUT / "check5_cafaceside.npz")
    cfg = yaml.safe_load(open(ROOT / "configs/video.yaml"))
    rec = Recognizer("adaface", ROOT / "models/caface/adaface_ir101_webface4m.ckpt", cfg["preprocess"]["input_size"], cfg["preprocess"]["interp"], False, 64, "cuda")
    ref = rec.get_embedding(list(z["crops"]))
    cos = (ref * z["feat"]).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(z["feat"], axis=1))
    res = dict(n_faces=int(len(cos)), cosine_min=float(cos.min()), cosine_median=float(np.median(cos)), cosine_max=float(cos.max()), n_above_0_9999=int((cos > 0.9999).sum()),
               note="CAFace backbone (inside CAFace_AdaFaceWebFace4M.ckpt) against the standalone adaface_ir101_webface4m.ckpt through src/recognizer.py (flip off)")
    p = ROOT / "outputs/video/agg7/checks.json"
    allres = json.loads(p.read_text()) if p.exists() else {}
    allres["check5_caface_backbone_vs_standalone_adaface_webface4m"] = res
    p.write_text(json.dumps(allres, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--part", choices=["chokepoint", "ytf", "check5", "check5-ref"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips or videos (smoke test)")
    ap.add_argument("--clips", nargs="+", default=None, help="ChokePoint: only these clips")
    a = ap.parse_args()
    if a.selftest:
        selftest(a)
    elif a.part:
        {"chokepoint": embed_chokepoint, "ytf": embed_ytf, "check5": check5_features, "check5-ref": check5_compare}[a.part](a)
    else:
        sys.exit("use --selftest or --part chokepoint|ytf")
