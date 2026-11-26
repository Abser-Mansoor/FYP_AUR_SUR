"""Phase 9 Part E: retrieval table, FAISS Flat against HNSW (REPORT_phase9.md section 0.6 E). CPU. Run after section 0b (uses the frozen configuration's templates).

    envs/video/Scripts/python.exe scripts/phase9_retrieval.py

Gallery sizes 10^2, 10^3, 10^4: the 170 MEVID templates (for 10^2: 100 of them, seed 42) padded with WebFace4M embeddings of TinyFace distractors (outputs/video/phase9/tinyface_wf.npz,
role "pad"). Probes: the 3,188 tracklet templates of the frozen configuration (probes of persons not in a 10^2 gallery are counted as misses for Rank-1). Reported: Rank-1 against the
MEVID identity, agreement of the HNSW top-1 with the Flat top-1, median query latency (batch 1, 1 thread), build time. Also acceptance check 5 on the real probes.
Output: outputs/video/phase9/retrieval.csv.
"""
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.video.index import GalleryIndex  # noqa: E402

P9 = ROOT / "outputs/video/phase9"


def main():
    import mevid_errors as E
    faiss.omp_set_num_threads(1)
    frozen = json.loads((E.P9 / "frozen_config.json").read_text())
    cfg = frozen["config"]
    tr, frames, g = E.load()
    key = ("resmatch_flip" if cfg["flip"] else "resmatch") if cfg["gallery"] == "resmatch" else ("feat_flip_native" if cfg["flip"] else "feat_native")
    T = E.templates(g, key)
    fkey = "wf_flip" if cfg["flip"] else "wf"
    Q = []
    for r in tr.itertuples():
        f = frames[r.tracklet]
        k = E.frame_sel(f, cfg)
        Q.append(E.A.RULES[cfg["rule"]](f[fkey][k], f["q"][k], f["idx"][k]))
    Q = np.ascontiguousarray(np.stack(Q), np.float32)
    truth = tr.person.to_numpy()
    tz = np.load(P9 / "tinyface_wf.npz")
    pad = E.l2n(tz["feat_flip" if cfg["flip"] else "feat"][tz["role"] == "pad"]).astype(np.float32)
    rng = np.random.default_rng(42)
    rows, check5 = [], []
    for size in (100, 1000, 10000):
        if size < len(T):
            keep = np.sort(rng.choice(len(T), size, replace=False))
            G, gid = T[keep], g["ids"][keep]
        else:
            G, gid = np.concatenate([T, pad[: size - len(T)]]), np.concatenate([g["ids"], -np.ones(size - len(T), int)])
        res = {}
        for kind in ("flat", "hnsw"):
            t0 = time.perf_counter()
            idx = GalleryIndex(G, kind)
            build = time.perf_counter() - t0
            lat, top = [], []
            for qv in Q:
                t = time.perf_counter()
                _, i = idx.search(qv[None], 1)
                lat.append(1000 * (time.perf_counter() - t))
                top.append(int(i[0, 0]))
            res[kind] = np.array(top)
            rows.append(dict(gallery_size=size, index=kind, n_probes=len(Q), rank1=float(np.mean(gid[res[kind]] == truth)), build_s=build, median_query_ms=float(np.median(lat)), p95_query_ms=float(np.quantile(lat, 0.95))))
        rows[-1]["hnsw_top1_agrees_with_flat"] = float(np.mean(res["hnsw"] == res["flat"]))
        check5.append(bool(np.array_equal(res["flat"], (Q @ G.T).argmax(1))))
    d = pd.DataFrame(rows)
    d.to_csv(P9 / "retrieval.csv", index=False, float_format="%.5f")
    print(d.round(5).to_string(index=False))
    import phase9_checks as C
    C.save("check5_faiss_flat_equals_exact_search_real_probes", dict(per_gallery_size=dict(zip(["100", "1000", "10000"], check5)), passed=bool(all(check5))))


if __name__ == "__main__":
    main()
