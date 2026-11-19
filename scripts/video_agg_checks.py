"""Phase 7 acceptance checks 1 and 2 on synthetic tracks (the rules in src/video/aggregate.py). Checks 3 to 5 are added by the scripts that have the data.

    envs/video/Scripts/python.exe scripts/video_agg_checks.py
Writes outputs/video/agg7/checks.json (key `unit`; other keys are kept).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.video import aggregate as A  # noqa: E402

OUT = ROOT / "outputs/video/agg7/checks.json"


def tracks(rng, n):
    e = A.l2n(rng.normal(size=(n, 512)))
    q = rng.uniform(18, 30, n)
    q[: n // 3] = q[0]  # equal qualities, to exercise the tie-break
    return e, q, np.arange(100, 100 + n)


def determinism(a_dir, b_dir):
    """Check 4 (aggregation part): every column that exists in both runs' result CSVs is identical."""
    import pandas as pd
    res = {}
    for f in ("chokepoint", "predicted", "curves_frames", "curves_quality", "chokepoint_thin_gallery", "predicted_track_stats"):
        a, b = pd.read_csv(Path(a_dir) / f"{f}.csv"), pd.read_csv(Path(b_dir) / f"{f}.csv")
        cols = [c for c in a.columns if c in b.columns]
        res[f] = dict(rows=[len(a), len(b)], columns_compared=len(cols), identical=bool(len(a) == len(b) and a[cols].equals(b[cols])))
    out = dict(first_run_dir=str(a_dir), second_run_dir=str(b_dir), files=res, all_identical=all(v["identical"] for v in res.values()),
               note="the second run added the rule-minus-R1 columns (only columns present in both runs are compared)")
    allres = json.loads(OUT.read_text()) if OUT.exists() else {}
    allres["check4_determinism_aggregation"] = out
    OUT.write_text(json.dumps(allres, indent=1))
    print(json.dumps(out, indent=1))


def main():
    rng = np.random.default_rng(42)
    res = {"check1_length1": {}, "check1_topk_N_equals_mean": {}, "check2_shuffle_invariance": {}}
    e, q, idx = tracks(rng, 1)
    for name, rule in A.RULES.items():
        res["check1_length1"][name] = bool(np.allclose(rule(e, q, idx), A.l2n(e[0]), atol=1e-12))
    T = A.l2n(rng.normal(size=(5, 512)))
    res["check1_length1"]["R5"] = bool(np.allclose(A.score_mean(e, T), e[0] @ T.T))
    res["check1_length1"]["R6"] = bool(A.vote(e, T) == int((e[0] @ T.T).argmax()))
    for n in (3, 7, 25):
        e, q, idx = tracks(rng, n)
        res["check1_topk_N_equals_mean"][str(n)] = bool(np.allclose(A.r3_topk(e, q, idx, k=n), A.r1_mean(e, q, idx), atol=1e-12) and np.allclose(A.r3_topk(e, q, idx, k=n + 5), A.r1_mean(e, q, idx), atol=1e-12))
    for n in (3, 7, 25):
        e, q, idx = tracks(rng, n)
        ok = {}
        for name, rule in A.RULES.items():
            base = rule(e, q, idx)
            ok[name] = all(np.array_equal(base, rule(e[p], q[p], idx[p])) for p in (rng.permutation(n) for _ in range(20)))
        ok["R5"] = all(np.array_equal(A.score_mean(e, T, idx), A.score_mean(e[p], T, idx[p])) for p in (rng.permutation(n) for _ in range(20)))
        ok["R6"] = all(A.vote(e, T, idx) == A.vote(e[p], T, idx[p]) for p in (rng.permutation(n) for _ in range(20)))
        res["check2_shuffle_invariance"][str(n)] = ok
    flat = [v for d in res.values() for x in d.values() for v in (x.values() if isinstance(x, dict) else [x])]
    res["all_pass"] = bool(all(flat))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allres = json.loads(OUT.read_text()) if OUT.exists() else {}
    allres["unit"] = res
    OUT.write_text(json.dumps(allres, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--determinism":
        determinism(sys.argv[2], sys.argv[3])
    else:
        main()
