"""MEVID open set and aggregation (Phase 8 Part B; protocol: REPORT_phase8.md section 0, committed before any number). CPU only, from the caches of scripts/mevid_embed.py.

    envs/video/Scripts/python.exe scripts/mevid_openset.py --model adaface        # primary rows (MS1MV2)
    envs/video/Scripts/python.exe scripts/mevid_openset.py --model webface4m      # secondary rows (CAFace backbone): single frame and R1 only
    envs/video/Scripts/python.exe scripts/mevid_openset.py --selftest             # synthetic data: checks 3 and 4 and the pipeline end to end

Definitions (REPORT_phase8.md section 0.3): eligible tracklet = >= 5 usable faces; gallery template of a person = normalised mean of the normalised embeddings of the person's usable front
photos (>= 48 px, or >= 64 px in the sensitivity run); enrolled = at least one usable photo. Closed set: gallery = all enrolled persons, probes = eligible tracklets of enrolled persons.
Open set: P = enrolled persons with an eligible tracklet, shuffled with default_rng(42) and cut into 5 folds; in fold k the group k is not enrolled (it is removed from the gallery) and its
tracklets are the non-enrolled probes; threshold t_f = the (1 - f) quantile of their top-1 similarities; TPIR1 = share of mated probes whose top-1 is the true person with similarity >= t_f;
TPIR20 uses the Phase 7 arithmetic (distance sqrt(2 - 2 cos), top-20 list). Cluster bootstrap over persons, 1,000 draws, seed 42. Outputs: outputs/video/mevid/results/*.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.video import aggregate as A  # noqa: E402
from video_phase7_bench import tpir_at  # noqa: E402

OUT = ROOT / "outputs/video/mevid"
SEED, N_BOOT, N_FOLDS, FPIRS = 42, 1000, 5, [0.3, 0.1, 0.01]
BARS = {48: "usable48", 64: "usable64"}
HEIGHT_BINS = [(16, 24), (24, 32), (32, 48), (48, 10 ** 6)]
RULE_Q = ["R1", "R2", "R3_k1", "R3_k3", "R3_k5", "R3_k10", "R4", "R4*"]  # embedding rules; R5 and R6 are score-level


def l2n(x):
    x = np.asarray(x, np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def draws_mean(values, pids, rng, n=N_BOOT):
    """Cluster bootstrap over persons of the tracklet-weighted mean of `values`."""
    u = np.unique(pids)
    s = np.array([values[pids == i].sum() for i in u])
    c = np.array([(pids == i).sum() for i in u], float)
    idx = rng.integers(0, len(u), (n, len(u)))
    return s[idx].sum(1) / c[idx].sum(1)


def ci(d, lo=0.025):
    return float(np.quantile(d, lo)), float(np.quantile(d, 1 - lo))


# ----------------------------------------------------------------------------- data
class Data:
    """tr: DataFrame of eligible tracklets (tracklet, person, camera, split, mean_face_h); frames[tracklet] = dict(emb [K,512] unit, q [K], idx [K], h [K]);
    gal: DataFrame of gallery faces (person, face_h) and gal_emb [G,512]."""

    def __init__(self, tr, frames, gal, gal_emb):
        self.tr, self.frames, self.gal, self.gal_emb = tr.reset_index(drop=True), frames, gal.reset_index(drop=True), l2n(gal_emb)
        self.person_of = dict(zip(self.tr.tracklet, self.tr.person))


def load_real(model):
    tr = pd.read_csv(OUT / "tracklets.csv")
    tr = tr[tr.detected.fillna(False).astype(bool) & tr.eligible.fillna(False).astype(bool)].reset_index(drop=True)
    frames = {}
    for r in tr.itertuples():
        z = np.load(OUT / "emb" / f"emb_{r.tracklet}.npz")
        if model == "adaface":
            emb, q = z["emb_adaface"], z["mag_norm"]
        else:
            c = np.load(OUT / "emb" / f"cf_{r.tracklet}.npz")
            emb, q = c["feat"], c["norm"]
        frames[r.tracklet] = dict(emb=l2n(emb), q=np.asarray(q, np.float64), idx=z["chip_idx"], h=z["face_h"])
    g = np.load(OUT / "gallery.npz" if model == "adaface" else OUT / "gallery_cf.npz", allow_pickle=True)
    gal = pd.DataFrame(dict(person=g["person"], face_h=g["face_h"]))
    return Data(tr, frames, gal, g["emb_adaface"] if model == "adaface" else g["feat"])


def gallery_templates(d, bar):
    """-> (person ids, templates [P,512]) of the persons with at least one gallery face of height >= bar."""
    ok = d.gal.face_h.to_numpy() >= bar
    ids = np.array(sorted(d.gal.person[ok].unique()))
    T = np.stack([l2n(d.gal_emb[ok & (d.gal.person.to_numpy() == p)].mean(0)) for p in ids])
    return ids, T


# ----------------------------------------------------------------------------- closed set
def closed_records(d, ids, T, rules):
    """One row per eligible tracklet of an enrolled person: single (mean frame correctness) and ok_<rule> (correctness of the aggregated decision)."""
    enrolled = set(ids.tolist())
    rows = []
    for r in d.tr.itertuples():
        if r.person not in enrolled:
            continue
        f = d.frames[r.tracklet]
        row = dict(tracklet=r.tracklet, person=r.person, camera=r.camera, mean_face_h=r.mean_face_h, n_faces=len(f["emb"]), single=float((ids[(f["emb"] @ T.T).argmax(1)] == r.person).mean()))
        for rule in rules:
            if rule == "R5":
                win = ids[int(np.argmax(A.score_mean(f["emb"], T, f["idx"])))]
            elif rule == "R6":
                win = ids[A.vote(f["emb"], T, f["idx"])]
            else:
                win = ids[int(np.argmax(T @ A.RULES[rule](f["emb"], f["q"], f["idx"])))]
            row[f"ok_{rule}"] = float(win == r.person)
        rows.append(row)
    return pd.DataFrame(rows)


def closed_table(rec, rules, label):
    pid = rec.person.to_numpy()
    out = []
    for rule in rules:
        delta = rec[f"ok_{rule}"].to_numpy() - rec.single.to_numpy()
        d = draws_mean(delta, pid, np.random.default_rng(SEED))
        lo, hi = ci(d)
        blo, bhi = ci(d, 0.05 / max(len(rules) - 1, 1) / 2)  # Bonferroni over the comparisons with R1
        row = dict(setting=label, rule=rule, n_tracklets=len(rec), n_persons=int(rec.person.nunique()), single_rank1=float(rec.single.mean()), agg_rank1=float(rec[f"ok_{rule}"].mean()),
                   delta=float(delta.mean()), delta_ci_lo=lo, delta_ci_hi=hi, delta_bonf_lo=blo, delta_bonf_hi=bhi)
        if "ok_R1" in rec and rule != "R1":
            dv = rec[f"ok_{rule}"].to_numpy() - rec["ok_R1"].to_numpy()
            dd = draws_mean(dv, pid, np.random.default_rng(SEED))
            mb = max(len(rules) - 1, 1)  # Bonferroni over the comparisons with R1
            row.update(minus_R1=float(dv.mean()), minus_R1_ci_lo=ci(dd)[0], minus_R1_ci_hi=ci(dd)[1], minus_R1_bonf_lo=ci(dd, 0.05 / mb / 2)[0], minus_R1_bonf_hi=ci(dd, 0.05 / mb / 2)[1])
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- open set
def make_folds(P):
    return np.array_split(np.random.default_rng(SEED).permutation(np.array(sorted(P))), N_FOLDS)


def probe_arrays(d, tracklets, ids, T, kind):
    """kind 'R1': one R1 template per tracklet; 'single': every usable face is a probe. -> dict(person, sim [top-1 cosine], top_id, dist_true [distance to the true person if in the top 20 else 1e5])."""
    X, P = [], []
    for t in tracklets:
        f = d.frames[t]
        person = d.person_of[t]
        if kind == "R1":
            X.append(A.r1_mean(f["emb"], f["q"], f["idx"])[None])
            P.append(person)
        else:
            X.append(f["emb"])
            P += [person] * len(f["emb"])
    if not X:
        return dict(person=np.zeros(0, int), sim=np.zeros(0), top_id=np.zeros(0, int), dist_true=np.zeros(0))
    X, P = np.concatenate(X), np.array(P)
    S = X @ T.T
    order = np.argsort(-S, axis=1, kind="stable")[:, :20]
    top = ids[order[:, 0]]
    sim = S[np.arange(len(S)), order[:, 0]]
    dist_true = np.full(len(S), 1e5)
    for i in range(len(S)):
        hit = np.flatnonzero(ids[order[i]] == P[i])
        if len(hit):
            dist_true[i] = np.sqrt(max(2 - 2 * S[i, order[i][hit[0]]], 0.0))
    return dict(person=P, sim=sim, top_id=top, dist_true=dist_true)


def fold_arrays(d, bar, kind):
    """Per fold: mated and non-enrolled probe arrays (see probe_arrays)."""
    ids, T_all = gallery_templates(d, bar)
    enrolled = set(ids.tolist())
    elig = d.tr[d.tr.person.isin(enrolled)]
    P = sorted(set(elig.person.unique()))  # enrolled persons with an eligible tracklet
    folds = make_folds(P)
    res = []
    for k, held in enumerate(folds):
        held = set(held.tolist())
        keep = np.array([p not in held for p in ids])
        ids_k, T_k = ids[keep], T_all[keep]
        mated_t = elig[~elig.person.isin(held)].tracklet.tolist()
        non_t = elig[elig.person.isin(held)].tracklet.tolist()
        res.append(dict(fold=k, held=sorted(held), mated=probe_arrays(d, mated_t, ids_k, T_k, kind), non=probe_arrays(d, non_t, ids_k, T_k, kind), n_gallery=len(ids_k)))
    return res, ids, T_all, P


def fold_metrics(m, non, f):
    t = np.quantile(non["sim"], 1 - f)
    return float(np.mean((m["top_id"] == m["person"]) & (m["sim"] >= t))) if len(m["sim"]) else np.nan


def fold_tpir20(m, non):
    s1 = np.sqrt(np.maximum(2 - 2 * non["sim"], 0.0))
    return tpir_at(s1, m["dist_true"])


def open_set_point(folds):
    """Fold-averaged TPIR1 at FPIR 0.3, 0.1, 0.01 and TPIR20 at the same values."""
    t1 = np.array([[fold_metrics(x["mated"], x["non"], f) for f in FPIRS] for x in folds])
    t20 = np.array([fold_tpir20(x["mated"], x["non"]) for x in folds])
    return t1.mean(0), t20.mean(0), t1


def open_set_boot(folds_r1, folds_sf, n=N_BOOT):
    """Cluster bootstrap over persons for the fold-averaged TPIR1 and TPIR20 of the R1 tracklet probes and of the single-frame probes, and their difference (same resamples)."""
    rng = np.random.default_rng(SEED)

    def groups(arr):
        order = np.argsort(arr["person"], kind="stable")
        pe = arr["person"][order]
        cuts = np.flatnonzero(np.r_[True, pe[1:] != pe[:-1], True])
        return {int(pe[a]): order[a:b] for a, b in zip(cuts[:-1], cuts[1:])}, order

    prep = []
    for fr1, fsf in zip(folds_r1, folds_sf):
        prep.append(dict(g1m=groups(fr1["mated"])[0], g1n=groups(fr1["non"])[0], gsm=groups(fsf["mated"])[0], gsn=groups(fsf["non"])[0], r1=fr1, sf=fsf))
    out = {"r1_t1": [], "sf_t1": [], "r1_t20": [], "sf_t20": []}
    for _ in range(n):
        v = {k: [] for k in out}
        for p in prep:
            for tag, gm, gn, arr in (("r1", p["g1m"], p["g1n"], p["r1"]), ("sf", p["gsm"], p["gsn"], p["sf"])):
                pm, pn = np.array(sorted(gm)), np.array(sorted(gn))
                im = np.concatenate([gm[int(q)] for q in pm[rng.integers(0, len(pm), len(pm))]]) if len(pm) else np.zeros(0, int)
                inn = np.concatenate([gn[int(q)] for q in pn[rng.integers(0, len(pn), len(pn))]]) if len(pn) else np.zeros(0, int)
                m = {k: arr["mated"][k][im] for k in arr["mated"]}
                non = {k: arr["non"][k][inn] for k in arr["non"]}
                v[f"{tag}_t1"].append([fold_metrics(m, non, f) for f in FPIRS])
                v[f"{tag}_t20"].append(fold_tpir20(m, non))
        for k in out:
            out[k].append(np.nanmean(np.array(v[k], float), axis=0))
    return {k: np.array(v) for k, v in out.items()}


def open_set_tables(d, bar, label):
    fr1, ids, T, P = fold_arrays(d, bar, "R1")
    fsf, _, _, _ = fold_arrays(d, bar, "single")
    (r1_t1, r1_t20, r1_folds), (sf_t1, sf_t20, sf_folds) = open_set_point(fr1), open_set_point(fsf)
    boot = open_set_boot(fr1, fsf)
    rows, h2 = [], None
    for metric, i_set in (("TPIR1", 0), ("TPIR20", 1)):
        for j, f in enumerate(FPIRS):
            for tag, pt, key in (("single frame", (sf_t1, sf_t20)[i_set], "sf"), ("R1 tracklet", (r1_t1, r1_t20)[i_set], "r1")):
                d_ = boot[f"{key}_t{'1' if i_set == 0 else '20'}"][:, j]
                rows.append(dict(setting=label, metric=metric, fpir=f, probe=tag, value=float(pt[j]), ci_lo=ci(d_)[0], ci_hi=ci(d_)[1]))
            diff = boot[f"r1_t{'1' if i_set == 0 else '20'}"][:, j] - boot[f"sf_t{'1' if i_set == 0 else '20'}"][:, j]
            rows.append(dict(setting=label, metric=metric, fpir=f, probe="R1 minus single frame", value=float((r1_t1, r1_t20)[i_set][j] - (sf_t1, sf_t20)[i_set][j]), ci_lo=ci(diff)[0], ci_hi=ci(diff)[1]))
    per_fold = [dict(setting=label, fold=x["fold"], n_gallery=x["n_gallery"], n_held_persons=len(x["held"]), n_mated_tracklets=len(x["mated"]["person"]), n_nonenrolled_tracklets=len(x["non"]["person"]),
                     **{f"TPIR1@{f}_R1": float(v) for f, v in zip(FPIRS, r1_folds[k])}) for k, x in enumerate(fr1)]
    return pd.DataFrame(rows), pd.DataFrame(per_fold), (fr1, ids, T, P)


def no_photo_row(d, bar, fr1, ids, T):
    """Eligible tracklets of persons without a usable gallery photo, as non-enrolled probes against the full gallery; false positive rate at the fold thresholds t_f (mean over folds)."""
    enrolled = set(ids.tolist())
    nop = d.tr[~d.tr.person.isin(enrolled)].tracklet.tolist()
    if not nop:
        return pd.DataFrame([dict(setting=f"bar {bar}", n_tracklets=0)])
    arr = probe_arrays(d, nop, ids, T, "R1")
    rows = []
    for f in FPIRS:
        fp = [float(np.mean(arr["sim"] >= np.quantile(x["non"]["sim"], 1 - f))) for x in fr1]
        rows.append(dict(setting=f"bar {bar}", fpir_target=f, n_tracklets=len(nop), n_persons=int(d.tr[d.tr.tracklet.isin(nop)].person.nunique()), false_positive_rate_mean_over_folds=float(np.mean(fp))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- run
def run(d, model, out):
    out.mkdir(parents=True, exist_ok=True)
    rules = RULE_Q + ["R5", "R6"] if model == "adaface" else ["R1"]
    tables = {"closed": [], "open": [], "folds": [], "no_photo": [], "camera": [], "height": [], "enrolment": []}
    for bar in (48, 64):
        label = f"{model}, gallery bar {bar} px"
        ids, T = gallery_templates(d, bar)
        rec = closed_records(d, ids, T, rules)
        rec.to_csv(out / f"closed_records_{model}_bar{bar}.csv", index=False, float_format="%.5f")
        tables["closed"].append(closed_table(rec, rules, label))
        if bar == 48:  # per camera and by face height (closed set, R1 against single frame)
            for cam, g in rec.groupby("camera"):
                if g.person.nunique() >= 3:
                    tables["camera"].append(closed_table(g, ["R1"], f"camera {cam}").assign(model=model))
            for lo, hi in HEIGHT_BINS:
                g = rec[(rec.mean_face_h >= lo) & (rec.mean_face_h < hi)]
                if g.person.nunique() >= 3:
                    tables["height"].append(closed_table(g, ["R1"], f"mean face height [{lo}, {hi}) px").assign(model=model))
        o, f_, (fr1, ids_, T_, P) = open_set_tables(d, bar, label)
        tables["open"].append(o)
        tables["folds"].append(f_)
        tables["no_photo"].append(no_photo_row(d, bar, fr1, ids_, T_).assign(model=model))
        tables["enrolment"].append(pd.DataFrame([dict(model=model, bar=bar, persons_enrolled=len(ids), persons_in_open_set_population=len(P), eligible_tracklets=len(d.tr),
                                                        eligible_tracklets_of_enrolled_persons=int(d.tr.person.isin(set(ids.tolist())).sum()), persons_without_usable_photo_with_eligible_tracklets=int(d.tr[~d.tr.person.isin(set(ids.tolist()))].person.nunique()))]))
    for k, v in tables.items():
        if v:
            pd.concat(v, ignore_index=True).to_csv(out / f"{k}_{model}.csv", index=False, float_format="%.5f")
    return {k: pd.concat(v, ignore_index=True) for k, v in tables.items() if v}


# ----------------------------------------------------------------------------- self-test on synthetic data (acceptance checks 3 and 4 and the pipeline)
def synthetic(n_persons=40, seed=0):
    rng = np.random.default_rng(seed)
    centres = l2n(rng.normal(size=(n_persons, 512)))
    rows, frames = [], {}
    for p in range(n_persons):
        for t in range(4):
            k = int(rng.integers(5, 30))
            emb = l2n(centres[p] + 0.6 * rng.normal(size=(k, 512)) / np.sqrt(512) * 8)
            name = f"te{p:03d}{t}"
            rows.append(dict(tracklet=name, person=p + 1, camera=int(rng.choice([505, 507, 639])), split="test", mean_face_h=float(rng.uniform(16, 60))))
            frames[name] = dict(emb=emb, q=rng.uniform(20, 28, k), idx=np.sort(rng.choice(1000, k, replace=False)), h=rng.uniform(16, 60, k))
    gal_rows, gal_emb = [], []
    for p in range(n_persons - 6):  # the last 6 persons have no usable photo
        for _ in range(2):
            gal_rows.append(dict(person=p + 1, face_h=float(rng.uniform(40, 120))))
            gal_emb.append(l2n(centres[p] + 0.2 * rng.normal(size=512) / np.sqrt(512) * 8))
    return Data(pd.DataFrame(rows), frames, pd.DataFrame(gal_rows), np.array(gal_emb))


def check_rules(d):
    """Acceptance check 3 on the data in `d`: a length-1 tracklet equals the single frame for every rule; frame order does not change R1 to R5 (and the vote)."""
    f = d.frames[d.tr.tracklet.iloc[0]]
    f1 = {k: v[:1] for k, v in f.items()}
    p = np.random.default_rng(1).permutation(len(f["emb"]))
    return dict(length1_equals_single_frame=bool(all(np.allclose(r(f1["emb"], f1["q"], f1["idx"]), f1["emb"][0], atol=1e-9) for r in A.RULES.values())),
                frame_order_invariance=bool(all(np.array_equal(r(f["emb"], f["q"], f["idx"]), r(f["emb"][p], f["q"][p], f["idx"][p])) for r in A.RULES.values())
                                            and np.array_equal(A.score_mean(f["emb"], f["emb"][:3], f["idx"]), A.score_mean(f["emb"][p], f["emb"][:3], f["idx"][p]))))


def check_folds(d, bar=48):
    """Acceptance check 4: every person of P is held out exactly once; the gallery and the non-enrolled sets are disjoint in every fold."""
    fr1, ids, T, P = fold_arrays(d, bar, "R1")
    held = [h for x in fr1 for h in x["held"]]
    disj = True
    for x in fr1:
        keep = {int(p) for p in ids} - set(x["held"])
        disj &= keep.isdisjoint(set(x["held"])) and not (set(x["non"]["person"].tolist()) & keep)
    return dict(each_person_held_out_exactly_once=bool(sorted(held) == sorted(P) and len(held) == len(set(held))), gallery_and_nonenrolled_disjoint=bool(disj), persons_in_population=len(P), folds=len(fr1))


def selftest():
    import tempfile
    d = synthetic()
    checks = {**check_rules(d), **check_folds(d)}
    res = run(d, "adaface", Path(tempfile.mkdtemp()))
    h1 = res["closed"][(res["closed"].rule == "R1") & res["closed"].setting.str.contains("48")].iloc[0]
    checks["synthetic_R1_not_worse_than_single_frame"] = bool(h1.agg_rank1 >= h1.single_rank1)
    print(json.dumps(checks, indent=1))
    sys.exit(0 if all(v for v in checks.values() if isinstance(v, bool)) else 1)


def write_checks(d):
    p = OUT / "checks.json"
    allres = json.loads(p.read_text()) if p.exists() else {}
    allres["check3_aggregation_rules_on_real_tracklet"] = check_rules(d)
    allres["check4_open_set_folds"] = check_folds(d)
    p.write_text(json.dumps(allres, indent=1, sort_keys=True))
    print(json.dumps({k: allres[k] for k in ("check3_aggregation_rules_on_real_tracklet", "check4_open_set_folds")}, indent=1, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["adaface", "webface4m"], default="adaface")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    d = load_real(a.model)
    res = run(d, a.model, OUT / "results")
    if a.model == "adaface":
        write_checks(d)
    prim = res["closed"][(res["closed"].rule == "R1") & res["closed"].setting.str.contains("bar 48")]
    print(prim.round(4).to_string(index=False))
    print(res["open"][(res["open"].metric == "TPIR1") & (res["open"].fpir == 0.1) & res["open"].setting.str.contains("bar 48")].round(4).to_string(index=False))
