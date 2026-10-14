"""Contact sheets: random crops from each end of a measure, for visual inspection."""
import cv2
import numpy as np

from ..preprocess import load_bgr

TILE = 96
COLS = 10


def _tile(path, label):
    im = load_bgr(path)
    t = cv2.resize(im, (TILE, TILE), interpolation=cv2.INTER_NEAREST)  # nearest: show the real pixels
    cv2.rectangle(t, (0, 0), (TILE, 13), (0, 0, 0), -1)
    cv2.putText(t, label, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def contact_sheet(df, col, end, q, n, seed, fmt="{:.1f}"):
    """n random crops from the lowest (end='low') or highest (end='high') fraction q of `col`."""
    v = df[col]
    ok = v.notna()
    order = np.argsort(v[ok].to_numpy(), kind="stable")
    idx = np.flatnonzero(ok.to_numpy())[order]
    k = max(int(q * len(idx)), n)
    pool = idx[:k] if end == "low" else idx[-k:]
    rng = np.random.default_rng(seed)
    pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
    tiles = [_tile(df["path"].iloc[i], fmt.format(v.iloc[i])) for i in pick]
    while len(tiles) % COLS:
        tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
    return np.vstack([np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)])


def save_png(path, img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))
