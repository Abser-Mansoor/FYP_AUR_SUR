"""Write stored T4 scrfd__full640 detections in the official WIDER prediction txt format (x y w h score, full precision)."""
import sys, os
from pathlib import Path
import numpy as np
ROOT = Path(r"D:/FYP_AUR_SUR")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import phase5_merge as M
names, dets, errors, sec = M.load(ROOT / "outputs/video/detstudy", "wider", "scrfd__full640")
out = Path(sys.argv[1]); nl = chr(10)
for n, d in zip(names, dets):
    ev, f = n.split("/", 1)
    (out / ev).mkdir(parents=True, exist_ok=True)
    lines = [f + ".jpg", str(len(d))] + ["%.9g %.9g %.9g %.9g %.9g" % (x1, y1, x2 - x1, y2 - y1, s) for x1, y1, x2, y2, s in d.tolist()]
    (out / ev / (f + ".txt")).write_text(nl.join(lines) + nl)
print("wrote", len(names))
