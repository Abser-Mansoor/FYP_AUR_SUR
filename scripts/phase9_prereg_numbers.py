"""Numbers that define the Phase 9 quality bar (REPORT_phase9.md section 0.1, D2 and D2a), computed from Phase 8 outputs only (no Phase 9 data).

    envs/video/Scripts/python.exe scripts/phase9_prereg_numbers.py
Input: outputs/video/mevid/tracklets.csv (mean usable-face height) and results/closed_records_adaface_bar48.csv (the 3,188 Phase 8 tracklets, MS1MV2 R1 outcome).
Output: outputs/video/phase9/prereg_numbers.json: the 25th percentile of the mean face height, and coverage and MS1MV2 R1 error of the kept tracklets at 20-24 px.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
r = pd.read_csv(ROOT / "outputs/video/mevid/results/closed_records_adaface_bar48.csv")
h = pd.read_csv(ROOT / "outputs/video/mevid/tracklets.csv").set_index("tracklet").loc[r.tracklet].mean_face_h.to_numpy()
err = 1 - r.ok_R1.to_numpy(float)
out = dict(n_tracklets=int(len(h)), p25_mean_face_height_px=float(np.percentile(h, 25)), median_mean_face_height_px=float(np.median(h)), overall_error_ms1mv2_r1=float(err.mean()),
           at_px={str(t): dict(coverage=float((h >= t).mean()), error_kept_ms1mv2_r1=float(err[h >= t].mean())) for t in (20, 21, 22, 23, 24)})
p = ROOT / "outputs/video/phase9/prereg_numbers.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1, sort_keys=True))
print(json.dumps(out, indent=1, sort_keys=True))
