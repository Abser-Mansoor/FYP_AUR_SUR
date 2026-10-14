"""Per-crop measures. All take a BGR uint8 array at native resolution."""
import cv2
import numpy as np


def luminance(bgr):
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def laplacian_variance(bgr, size=112):
    """Resize to size x size (bilinear, as in Phase 1), grayscale, variance of the Laplacian."""
    g = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def fft_high_freq(bgr, cutoff=0.5):
    """Share of spectral energy above `cutoff` x Nyquist, at native resolution (no resizing).

    Hann-windowed grayscale, mean removed. NaN for crops with no AC energy (flat images)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = g.shape
    win = np.outer(np.hanning(h) if h > 2 else np.ones(h), np.hanning(w) if w > 2 else np.ones(w))
    g = (g - g.mean()) * win
    p = np.abs(np.fft.fft2(g)) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2) / 0.5  # 1.0 = Nyquist along an axis
    total = p.sum()
    if total <= 1e-9:
        return float("nan")
    return float(p[r > cutoff].sum() / total)


def crop_measures(bgr, lap_size=112, fft_cutoff=0.5):
    h, w = bgr.shape[:2]
    return dict(h=h, w=w, min_side=min(h, w), luminance=luminance(bgr),
                lap_var=laplacian_variance(bgr, lap_size), fft_hf=fft_high_freq(bgr, fft_cutoff))
