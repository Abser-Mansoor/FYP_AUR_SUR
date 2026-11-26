"""Phase 9 B3 visual audit (REPORT_phase9.md section 0.4). The sheet (outputs/video/mevid/audit/sheet.csv) is written by `scripts/mevid_errors.py --stage dev`.

    envs/video/Scripts/python.exe scripts/mevid_audit.py --montages          # one PNG per audit id in outputs/video/mevid/audit/montages/ (LOCAL ONLY: MEVID images, gitignored)
    envs/video/Scripts/python.exe scripts/mevid_audit.py --agreement labels_umar.csv labels_abser.csv

Montage: 5 evenly spaced usable faces of the tracklet (the chip cropped around the face box with 20% margin, scaled to 112 px high), then gallery photos A and B (the best usable front
face of each person, same scaling). A and B are the true and the predicted person in a random order (the key is in key.csv, not shown to the labelers). Labelers copy labels_TEMPLATE.csv
to labels_<name>.csv and put 1 under every tag that applies.
Agreement: Cohen's kappa per tag over all audited tracklets; shares of each tag among the wrong tracklets (a tag counts if both labelers set it) and among the correct controls.
Outputs: audit/agreement.csv, audit/taxonomy.csv (tracked: counts only).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
AUD = ROOT / "outputs/video/mevid/audit"
M = ROOT / "outputs/video/mevid"
TAGS = ["not_face_or_back", "profile", "blur", "occlusion", "low_light_or_ir", "tiny", "wrong_person_in_tracklet", "gallery_mismatch", "look_alike", "nothing_visible"]


def face_crop(im, box, h_out=112):
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    x1, y1, x2, y2 = int(max(0, x1 - 0.2 * w)), int(max(0, y1 - 0.2 * h)), int(min(im.shape[1], x2 + 0.2 * w)), int(min(im.shape[0], y2 + 0.2 * h))
    c = im[y1:y2, x1:x2]
    s = h_out / max(c.shape[0], 1)
    return cv2.resize(c, (max(1, round(c.shape[1] * s)), h_out), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)


def montages():
    import mevid_embed as ME
    sheet = pd.read_csv(AUD / "sheet.csv")
    tr = pd.read_csv(M / "tracklets.csv").set_index("tracklet")
    gf = pd.read_csv(M / "gallery_faces.csv")
    best = gf[gf.usable48].sort_values("face_h", ascending=False).groupby("person").head(1).set_index("person")
    names = {s: ME.load_names(s) for s in ("train", "test")}
    out = AUD / "montages"
    out.mkdir(parents=True, exist_ok=True)
    for r in sheet.itertuples():
        t = tr.loc[r.tracklet]
        z, e = np.load(M / "emb" / f"det_{r.tracklet}.npz"), np.load(M / "emb" / f"emb_{r.tracklet}.npz")
        sel = e["sel"][np.unique(np.round(np.linspace(0, len(e["sel"]) - 1, 5)).astype(int))]
        tiles = [face_crop(cv2.imread(str(ME.chip_path(t.split, names[t.split][int(z["chip_idx"][q])]))), z["box"][q]) for q in sel]
        for tag, p in (("A", r.photo_a_person), ("B", r.photo_b_person)):
            b = best.loc[p]
            im = face_crop(cv2.imread(str(ME.PHOTOS / b.file)), np.array([b.x1, b.y1, b.x2, b.y2]))
            cv2.putText(im, tag, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(np.pad(im, ((0, 0), (12, 0), (0, 0))))
        cv2.imwrite(str(out / f"{r.audit_id}.png"), np.concatenate(tiles, axis=1))
    print("montages:", len(sheet), "->", out)


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def agreement(f1, f2):
    key = pd.read_csv(AUD / "key.csv").set_index("audit_id")
    a, b = (pd.read_csv(AUD / f).set_index("audit_id").loc[key.index] for f in (f1, f2))
    a, b = a[TAGS].fillna(0).astype(int), b[TAGS].fillna(0).astype(int)
    rows, tax = [], []
    for t in TAGS:
        rows.append(dict(tag=t, kappa=kappa(a[t], b[t]), share_labeler1=a[t].mean(), share_labeler2=b[t].mean()))
        both = (a[t] == 1) & (b[t] == 1)
        tax.append(dict(tag=t, wrong_n=int(key.wrong.sum()), wrong_share_both=float(both[key.wrong == 1].mean()), correct_n=int((key.wrong == 0).sum()), correct_share_both=float(both[key.wrong == 0].mean())))
    pd.DataFrame(rows).to_csv(AUD / "agreement.csv", index=False, float_format="%.4f")
    pd.DataFrame(tax).to_csv(AUD / "taxonomy.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(tax).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--montages", action="store_true")
    ap.add_argument("--agreement", nargs=2)
    a = ap.parse_args()
    montages() if a.montages else agreement(*a.agreement)
