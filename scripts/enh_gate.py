"""Phase 4 gated variants: enhance a crop only if its RAW measure is below a global threshold tau.

    envs\\score\\Scripts\\python.exe scripts\\enh_gate.py --config configs\\enhance.yaml [--parts tuning eval]

Per gated family (config `gates`): the family's selected arm is the one with the best tuning score among the family's arms
(selection_<arm>.json); tau candidates are the configured quantiles of the pooled TUNING distribution of the raw measure
(outputs/stratify/measures/*.csv: luminance for low-light, lap_var for deblur); one tau is chosen on tuning with the Phase 3 selection
rule. Gated embedding caches mix, per crop, the enhanced embedding (measure < tau) and the direct-feed embedding, so no restoration is
rerun (a crop's embedding does not depend on other crops, so this equals gating the image). They are scored through
E.evaluate_variant into <results>/tuning_gated.csv and eval_gated.csv. The eval part needs the selected arm's eval embeddings
(sr_pipeline.py arm, without --tune-only) and is skipped, with a message, while they are missing.
Gating sanity (tuning): tau = 0 equals direct, tau = inf equals the full arm (np.array_equal) -> <results>/gate_sanity.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.sr import eval as E  # noqa: E402
import sr_pipeline as P  # noqa: E402

NAMES = E.NAMES


def family_arm(res, arms):
    """Selected arm of a family: best tuning score among its arms."""
    best = None
    for a in arms:
        sel = json.loads((res / f"selection_{a}.json").read_text())[0]
        if best is None or sel["score"] > best[2]:
            best = (a, sel["setting"], sel["score"])
    return best


def measure_of(d, key):
    """Raw per-crop measure for the rows of d (joined on dataset, split, name)."""
    m = pd.concat([pd.read_csv(f).assign(_k=f.stem) for f in sorted((ROOT / "outputs/stratify/measures").glob("*.csv"))])
    m = m.rename(columns={"name": "_name"})
    k = d["dataset"] + "_" + d["split"]
    j = pd.DataFrame({"_k": k.to_numpy(), "_name": d["name"].to_numpy()}).merge(m[["_k", "_name", key]], on=["_k", "_name"], how="left")
    assert len(j) == len(d)
    return j[key].to_numpy(float)


def emb_path(part, arm, setting, n):
    return ROOT / "outputs/sr/emb" / part / arm / setting / "seed0" / f"{part}__{n}.npy"


def gated_cache(part, family, q, base, tau, enh):
    """Write the gated embeddings (enhanced rows where enh, direct elsewhere); returns the setting name."""
    setting = f"q{round(q * 100)}__{base[0]}_{base[1]}"
    for n in NAMES:
        e = np.load(emb_path(part, base[0], base[1], n))
        dd = np.load(emb_path(part, "direct", "resize", n))
        out = np.where(enh[:, None], e, dd)
        f = emb_path(part, f"gate_{family}", setting, n)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, out)
    return setting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/enhance.yaml")
    ap.add_argument("--parts", nargs="+", default=["tuning", "eval"])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    res = ROOT / cfg["paths"].get("results", "outputs/sr/results")
    S = cfg["bootstrap"]["seed"]
    Bs = {"tuning": cfg["bootstrap"]["tuning_n"], "eval": cfg["bootstrap"]["eval_n"]}
    recs = {n: None for n in NAMES}  # cached embeddings only: no model is loaded
    dt, _ = E.load_subset(ROOT, "tuning")
    sanity, taus, rows_by_part = {}, [], {p: [] for p in a.parts}
    for fam, g in cfg["gates"].items():
        base = family_arm(res, g["arms"])
        mt = measure_of(dt, g["measure"])
        n_nan = int(np.isnan(mt).sum())
        qs = {q: float(np.nanquantile(mt, q)) for q in g["quantiles"]}
        print(f"[{fam}] selected arm {base[0]}__{base[1]} (tuning score {base[2]:.4f}); measure {g['measure']}; NaN {n_nan}; tau {qs}", flush=True)
        # gating sanity (tuning): tau = 0 -> direct, tau = inf -> full arm
        for tag, tau in (("tau0", 0.0), ("tauinf", np.inf)):
            enh = np.nan_to_num(mt, nan=np.inf) < tau  # a crop without a measure is never enhanced (n_nan is printed above)
            ok = {}
            for n in NAMES:
                e = np.load(emb_path("tuning", base[0], base[1], n))
                dd = np.load(emb_path("tuning", "direct", "resize", n))
                gated = np.where(enh[:, None], e, dd)
                ok[n] = bool(np.array_equal(gated, dd if tag == "tau0" else e))
            sanity[f"{fam}_{tag}"] = ok
        for part in a.parts:
            d, pairs = E.load_subset(ROOT, part)
            if not all(emb_path(part, base[0], base[1], n).exists() for n in NAMES):
                print(f"[{fam}] {part}: embeddings of {base[0]}__{base[1]} missing, skipped (run the eval step first)", flush=True)
                continue
            m = mt if part == "tuning" else measure_of(d, g["measure"])
            # eval uses the tau chosen on tuning: tuning gate selection is read back from file once it exists
            sel_file = res / f"selection_gate_{fam}.json"
            if part == "eval" and not sel_file.exists():
                print(f"[{fam}] eval: tau not selected yet (run the tuning part first)", flush=True)
                continue
            if part == "tuning":
                use = [(q, tau) for q, tau in qs.items()]
            else:
                chosen = json.loads(sel_file.read_text())[0]
                use = [(chosen["q"], chosen["tau"])]
            dc, rows = {}, []
            cache = ROOT / "outputs/sr/emb" / part
            rows += E.evaluate_variant(recs, d, pairs, ROOT, "direct", "resize", 0, part, None, set(), cache, Bs[part], S, dc)
            for r in rows:
                r["status"] = "OK"
            for q, tau in use:
                enh = np.nan_to_num(m, nan=np.inf) < tau
                setting = gated_cache(part, fam, q, base, tau, enh)
                new = E.evaluate_variant(recs, d, pairs, ROOT, f"gate_{fam}", setting, 0, part, None, set(), cache, Bs[part], S, dc)
                for r in new:
                    r.update(status="OK", family=fam, quantile=q, tau=tau, measure=g["measure"], base_arm=base[0], base_setting=base[1],
                             frac_enhanced=float(enh.mean()))
                rows += new
                taus.append(dict(family=fam, part=part, quantile=q, tau=tau, measure=g["measure"], base=f"{base[0]}__{base[1]}",
                                 frac_enhanced=float(enh.mean()), n=len(enh)))
                print(f"[{fam}] {part} q={q} tau={tau:.4g} enhanced {enh.mean():.1%}", flush=True)
            df = pd.DataFrame(rows)
            rows_by_part[part] += rows
            if part == "tuning":
                g_, best = P.selection(df)
                g_.to_csv(res / f"tuning_gate_selection_{fam}.csv", index=False)
                bi = best.iloc[0]
                q_sel = round(int(bi["setting"].split("__")[0][1:]) / 100, 2)
                (res / f"selection_gate_{fam}.json").write_text(json.dumps(
                    [dict(bi.to_dict(), q=q_sel, tau=qs[q_sel], base_arm=base[0], base_setting=base[1], measure=g["measure"])], indent=2))
                print(f"[{fam}] selected gate {bi['setting']} (score {bi['score']:.4f})", flush=True)
    for part, rows in rows_by_part.items():
        if rows:
            out = res / f"{part}_gated.csv"
            new = pd.DataFrame(rows)
            if out.exists():  # families are processed in turn; keep earlier families' rows, drop the repeated direct rows
                old = pd.read_csv(out)
                new = pd.concat([old[~old["arm"].isin(new["arm"].unique())], new], ignore_index=True)
            new = new.drop_duplicates(subset=["arm", "setting", "seed", "part", "recognizer", "protocol"], keep="last")
            new.to_csv(out, index=False)
    tau_f = res / "gate_tau.csv"
    t_new = pd.DataFrame(taus)
    if tau_f.exists() and len(t_new):  # keep the rows of parts not run this time
        old = pd.read_csv(tau_f)
        t_new = pd.concat([old[~old["part"].isin(t_new["part"].unique())], t_new], ignore_index=True)
    t_new.to_csv(tau_f, index=False)
    (res / "gate_sanity.json").write_text(json.dumps(sanity, indent=2))
    print("gating sanity:", json.dumps(sanity))
    bad = [k for k, v in sanity.items() if not all(v.values())]
    if bad:
        sys.exit(f"GATING SANITY FAILED: {bad}")


if __name__ == "__main__":
    main()
