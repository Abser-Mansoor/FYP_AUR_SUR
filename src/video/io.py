"""Frame sources for the video pipeline: an image sequence (ChokePoint, YTF frame folders) or a video file.

    for frame_idx, timestamp, bgr in iter_frames(path, fps=30): ...

`frame_idx` is the number in the file name for an image sequence (ChokePoint frames are named 00000233.jpg, so the index matches the ground truth)
and the 0-based decode order for a video file. `timestamp` is frame_idx / fps in seconds.
"""
from pathlib import Path

import cv2

from src.sr.common import read_bgr

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


def list_frames(folder):
    """Sorted [(frame_idx, path)] of an image-sequence folder; names must be (or end in) the frame number."""
    out = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            digits = "".join(ch for ch in p.stem if ch.isdigit())
            out.append((int(digits) if digits else len(out), p))
    return out


def iter_frames(source, fps=30.0, limit=None):
    """Yields (frame_idx, timestamp_s, bgr). `source` is a folder of images or a video file."""
    source = Path(source)
    n = 0
    if source.is_dir():
        for idx, p in list_frames(source):
            if limit is not None and n >= limit:
                return
            yield idx, idx / fps, read_bgr(p)
            n += 1
        return
    if source.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"not a folder or a known video file: {source}")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {source}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    idx = 0
    try:
        while limit is None or n < limit:
            ok, bgr = cap.read()
            if not ok:
                break
            yield idx, idx / video_fps, bgr
            idx += 1
            n += 1
    finally:
        cap.release()
