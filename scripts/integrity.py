"""Phase 0 dataset integrity scan: decode every image, record size, flag
corrupt files and exact (byte-level) duplicates. Read-only on datasets."""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/diagnostics"
EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def scan(p):
    b = p.read_bytes()
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return str(p), None, None, None, hashlib.md5(b).hexdigest()
    ch = 1 if img.ndim == 2 else img.shape[2]
    return str(p), img.shape[1], img.shape[0], ch, hashlib.md5(b).hexdigest()


def main(name, root):
    files = [p for p in Path(root).rglob("*") if p.is_file()]
    imgs = [p for p in files if p.suffix.lower() in EXT]
    other = Counter(p.suffix.lower() for p in files if p.suffix.lower() not in EXT)
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(scan, imgs, chunksize=256))
    with open(OUT / f"integrity_{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "w", "h", "channels", "md5"])
        w.writerows(rows)
    corrupt = [r[0] for r in rows if r[1] is None]
    by_md5 = defaultdict(list)
    for r in rows:
        by_md5[r[4]].append(r[0])
    dups = [v for v in by_md5.values() if len(v) > 1]
    # duplicates split by top-level subfolder, to spot cross-split leakage
    rel = lambda s: Path(s).relative_to(root).parts  # noqa: E731
    cross = [v for v in dups if len({rel(x)[:2] for x in v}) > 1]
    ok = [r for r in rows if r[1] is not None]
    W, H = np.array([r[1] for r in ok]), np.array([r[2] for r in ok])
    per_dir = Counter(rel(r[0])[:2] if len(rel(r[0])) > 2 else rel(r[0])[:1] for r in rows)
    summary = {
        "dataset": name, "n_files": len(files), "n_images": len(imgs), "non_image_ext": dict(other),
        "ext": dict(Counter(p.suffix.lower() for p in imgs)), "channels": dict(Counter(r[3] for r in ok)),
        "n_corrupt": len(corrupt), "corrupt_examples": corrupt[:20],
        "n_duplicate_groups": len(dups), "n_duplicate_files": sum(len(v) for v in dups),
        "n_cross_folder_duplicate_groups": len(cross), "cross_folder_examples": cross[:10],
        "width": {q: float(np.percentile(W, q)) for q in (0, 5, 25, 50, 75, 95, 100)},
        "height": {q: float(np.percentile(H, q)) for q in (0, 5, 25, 50, 75, 95, 100)},
        "width_mean": float(W.mean()), "height_mean": float(H.mean()),
        "images_per_folder": {"/".join(k): v for k, v in sorted(per_dir.items()) if v > 0 and len(k) <= 2 and len(per_dir) < 50},
    }
    (OUT / f"integrity_{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "images_per_folder"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], ROOT / sys.argv[2])
