"""Image loading, preprocessing and batched embedding extraction."""
import cv2
import numpy as np
import torch

from .align import norm_crop

INTERP = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}


def load_bgr(path):
    # np.fromfile + imdecode handles any Windows path
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"unreadable image: {path}")
    return img


def to_112(bgr, interp="bicubic", kps=None):
    """Aligned crop when landmarks are given, else direct stretch-resize (Phase 1 convention)."""
    if kps is not None:
        return norm_crop(bgr, kps)
    return cv2.resize(bgr, (112, 112), interpolation=INTERP[interp])


def to_tensor(bgr112, channel, norm):
    img = bgr112[:, :, ::-1] if channel == "RGB" else bgr112
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    if norm == "pm1":
        x = (x - 0.5) / 0.5
    elif norm != "01":
        raise ValueError(norm)
    return x


@torch.no_grad()
def extract(embed, paths, cfg, landmarks=None, batch=256, device="cuda"):
    """Embed images (paths or preloaded BGR arrays). cfg keys: channel, norm, interp, flip.
    Returns raw float32 [N,512]."""
    out = []
    for s in range(0, len(paths), batch):
        xs = []
        for i in range(s, min(s + batch, len(paths))):
            kps = None if landmarks is None else landmarks[i]
            img = paths[i] if isinstance(paths[i], np.ndarray) else load_bgr(paths[i])
            xs.append(to_tensor(to_112(img, cfg["interp"], kps), cfg["channel"], cfg["norm"]))
        x = torch.stack(xs).to(device)
        e = embed(x)
        if cfg.get("flip"):
            e = e + embed(torch.flip(x, dims=[3]))
        out.append(e.float().cpu().numpy())
    return np.concatenate(out)
