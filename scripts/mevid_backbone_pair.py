"""EXPLORATORY (not pre-registered in REPORT_phase8.md section 0): paired comparison of the two backbones on the same MEVID tracklets.

    envs/video/Scripts/python.exe scripts/mevid_backbone_pair.py

Backbone A = AdaFace R100 MS1MV2 (the pipeline default), B = AdaFace R100 WebFace4M (the CAFace backbone). Same eligible tracklets of enrolled persons, same gallery bar, same R1 rule.
From the per-tracklet outcomes already written by scripts/mevid_openset.py (results/closed_records_<model>_bar<bar>.csv): paired per tracklet, B minus A, for the single-frame
accuracy, the R1 accuracy and the aggregation gain (R1 minus single frame); cluster bootstrap over persons (1,000 draws, seed 42). Output: results/backbone_pair.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "outputs/video/mevid/results"
SEED, N = 42, 1000


def draws(values, pids, rng):
    u = np.unique(pids)
    s = np.array([values[pids == i].sum() for i in u])
    c = np.array([(pids == i).sum() for i in u], float)
    idx = rng.integers(0, len(u), (N, len(u)))
    return s[idx].sum(1) / c[idx].sum(1)


rows = []
for bar in (48, 64):
    a = pd.read_csv(R / f"closed_records_adaface_bar{bar}.csv").set_index("tracklet")
    b = pd.read_csv(R / f"closed_records_webface4m_bar{bar}.csv").set_index("tracklet")
    common = a.index.intersection(b.index)
    assert len(common) == len(a) == len(b), (len(common), len(a), len(b))
    a, b = a.loc[common], b.loc[common]
    pid = a.person.to_numpy()
    gain_a, gain_b = a.ok_R1 - a.single, b.ok_R1 - b.single
    for name, x in (("single frame", b.single - a.single), ("R1 tracklet", b.ok_R1 - a.ok_R1), ("aggregation gain (R1 minus single)", gain_b - gain_a)):
        d = draws(x.to_numpy(), pid, np.random.default_rng(SEED))
        rows.append(dict(gallery_bar=bar, quantity=name, n_tracklets=len(common), n_persons=int(len(np.unique(pid))), mean_A=float({"single frame": a.single, "R1 tracklet": a.ok_R1}.get(name, gain_a).mean()),
                         mean_B=float({"single frame": b.single, "R1 tracklet": b.ok_R1}.get(name, gain_b).mean()), B_minus_A=float(x.mean()), ci_lo=float(np.quantile(d, 0.025)), ci_hi=float(np.quantile(d, 0.975))))
out = pd.DataFrame(rows)
out.to_csv(R / "backbone_pair.csv", index=False, float_format="%.5f")
print(out.round(4).to_string(index=False))
