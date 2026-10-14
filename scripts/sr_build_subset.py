"""Build the fixed identity-stratified subsets (tuning + eval) shared by all SR arms.

    envs/diag/Scripts/python.exe scripts/sr_build_subset.py [--config configs/sr.yaml]

Identities are drawn at random (seeded) inside strata of gallery-image count (tertiles), and ALL their probe /
gallery images are kept, so identification protocols stay valid. Distractors / unmated queries / verification pairs
are random draws. Tuning and eval identities are disjoint. Output: <out>/subset/<name>_images.csv and _pairs.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import datasets  # noqa: E402


def pick_ids(ids_with_counts, n, rng, exclude=()):
    """Stratified (tertiles of image count) random identity sample, without replacement."""
    s = ids_with_counts[~ids_with_counts.index.isin(exclude)]
    q = pd.qcut(s.rank(method="first"), 3, labels=False)
    per = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    out = []
    for k in range(3):
        pool = s.index[q == k].to_numpy()
        out += rng.choice(pool, min(per[k], len(pool)), replace=False).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sr.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    rng = np.random.default_rng(cfg["subset"]["seed"])
    tf = datasets.tinyface(ROOT / cfg["paths"]["tinyface"])
    sf = datasets.survface_identification(ROOT / cfg["paths"]["survface"])
    vpaths, pairs = datasets.survface_verification(ROOT / cfg["paths"]["survface"])
    bins = {(ds, s): pd.read_csv(ROOT / cfg["paths"]["bins"] / f"{ds}_{s}.csv", keep_default_na=False).set_index("name")
            for ds, s in [("tinyface", "probe"), ("tinyface", "gallery_match"), ("tinyface", "distractor"),
                          ("survface", "gallery"), ("survface", "mated_probe"), ("survface", "unmated_probe"), ("survface", "verification")]}
    out_dir = ROOT / cfg["paths"]["out"] / "subset"
    out_dir.mkdir(parents=True, exist_ok=True)

    tf_g = pd.Series(tf["gallery_match"][1]).value_counts()
    tf_g = tf_g[tf_g.index.isin(tf["probe"][1])]
    sf_g = pd.Series(sf["gallery"][1]).value_counts()
    sf_g = sf_g[sf_g.index.isin(sf["mated_probe"][1])]
    used_tf, used_sf, used_unm, used_dis, used_pair, used_vimg = set(), set(), set(), set(), set(), set()
    names = {}
    for ds, s in [("tinyface", "probe"), ("tinyface", "gallery_match"), ("tinyface", "distractor")]:
        names[(ds, s)] = [p.name for p in tf[s][0]]
    for s in ("gallery", "mated_probe", "unmated_probe"):
        names[("survface", s)] = [p.name for p in sf[s][0]]
    summary = []
    for part in ("tuning", "eval"):
        c = cfg["subset"][part]
        rows = []
        ids_tf = pick_ids(tf_g, c["tinyface_identities"], rng, used_tf); used_tf |= set(ids_tf)
        ids_sf = pick_ids(sf_g, c["survface_identities"], rng, used_sf); used_sf |= set(ids_sf)

        def add(ds, split, idx, pid=None):
            b = bins[(ds, split)]
            for i in idx:
                n = names[(ds, split)][i]
                r = b.loc[n]
                rows.append(dict(part=part, dataset=ds, split=split, name=n, path=r["path"], id=-100 if pid is None else int(pid[i]),
                                 h=int(r["h"]), w=int(r["w"]), sharp_bin_sz=r["sharp_bin_sz"], lum_bin=r["lum_bin"], lum_bin10=r["lum_bin10"]))
        for split in ("probe", "gallery_match"):
            ids = tf[split][1]
            add("tinyface", split, np.flatnonzero(np.isin(ids, ids_tf)), ids)
        pool = np.array([i for i in range(len(names[("tinyface", "distractor")])) if i not in used_dis])
        d = rng.choice(pool, c["tinyface_distractors"], replace=False); used_dis |= set(d.tolist())
        add("tinyface", "distractor", sorted(d.tolist()))
        for split in ("gallery", "mated_probe"):
            ids = sf[split][1]
            add("survface", split, np.flatnonzero(np.isin(ids, ids_sf)), ids)
        pool = np.array([i for i in range(len(names[("survface", "unmated_probe")])) if i not in used_unm])
        u = rng.choice(pool, c["survface_unmated"], replace=False); used_unm |= set(u.tolist())
        add("survface", "unmated_probe", sorted(u.tolist()))
        # verification pairs: random positive and negative pairs, images added to the list
        prs = []
        for k in ("pos", "neg"):
            pool = np.array([i for i in range(len(pairs[k])) if (k, i) not in used_pair and pairs[k][i][0] not in used_vimg and pairs[k][i][1] not in used_vimg])
            sel = rng.choice(pool, c["survface_ver_pairs"], replace=False)
            used_pair |= {(k, int(i)) for i in sel}
            for i in sorted(sel.tolist()):
                prs.append(dict(part=part, kind=k, a=pairs[k][i][0], b=pairs[k][i][1]))
        pr = pd.DataFrame(prs)
        vimgs = sorted(set(pr["a"]) | set(pr["b"]))
        used_vimg |= set(vimgs)
        vb = bins[("survface", "verification")]
        for n in vimgs:
            r = vb.loc[n]
            rows.append(dict(part=part, dataset="survface", split="verification", name=n, path=r["path"], id=-100, h=int(r["h"]), w=int(r["w"]),
                             sharp_bin_sz=r["sharp_bin_sz"], lum_bin=r["lum_bin"], lum_bin10=r["lum_bin10"]))
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{part}_images.csv", index=False)
        pr.to_csv(out_dir / f"{part}_pairs.csv", index=False)
        summary.append((part, df.groupby(["dataset", "split"]).size().to_dict(), len(df)))
        print(part, "images:", len(df)); print(df.groupby(["dataset", "split"]).size().to_string())
        print(" identities: tinyface", len(ids_tf), "survface", len(ids_sf))
    print("tuning and eval image overlap (by dataset/split/name):",
          len(set(map(tuple, pd.read_csv(out_dir / "tuning_images.csv")[["dataset", "split", "name"]].values)) &
              set(map(tuple, pd.read_csv(out_dir / "eval_images.csv")[["dataset", "split", "name"]].values))))


if __name__ == "__main__":
    main()
