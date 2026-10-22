"""Phase 5 check: on a 1280x720 frame with ONE face fully inside one tile, every detector and mode returns exactly one detection
with score >= 0.5 (tiling must not duplicate it).   envs/video/Scripts/python.exe scripts/phase5_verify_tiling.py
Writes outputs/video/verify_tiling.json; exit code 1 on failure."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.video import detect as D  # noqa: E402


def main():
    face = cv2.resize(cv2.imread(str(ROOT / "outputs/sr/lfw/hr112/00000.png")), (220, 220), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((720, 1280, 3), 90, np.uint8)
    canvas[60:280, 60:280] = face
    res = {}
    for name, det in (("scrfd", D.SCRFDDetector(ROOT / "models/detector/det_10g.onnx")),
                      ("retinaface", D.RetinaFaceDetector(ROOT / "models/detector/retinaface/Resnet50_Final.pth"))):
        for mode in D.MODES:
            d, _ = D.detect_mode(det, canvas, mode)
            res[f"{name}__{mode}"] = int((d[:, 4] >= 0.5).sum())
    res["pass"] = all(v == 1 for k, v in res.items())
    (ROOT / "outputs/video").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/video/verify_tiling.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
