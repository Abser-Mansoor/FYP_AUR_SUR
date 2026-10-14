"""Phase 3 pipeline: restore -> embed -> score -> select. Resumable; every failure is logged as a result.

    envs/diag/Scripts/python.exe scripts/sr_pipeline.py direct                      # once: direct-feed embeddings + metrics
    envs/diag/Scripts/python.exe scripts/sr_pipeline.py arm --arms invsr            # tune, select, eval, seed study for one arm
Several `arm` workers run in parallel on the one GPU (one process per arm). Restored outputs go to
outputs/sr/restored/<part>/<arm>__<setting>/seed<k>/, scores to outputs/sr/results/*.csv.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recognizer import Recognizer  # noqa: E402
from src.sr import eval as E  # noqa: E402
from src.utils import seed_all  # noqa: E402

log = logging.getLogger("sr")
RES = ROOT / "outputs/sr/results"


def vid(v):
    """Variant dict -> (arm, setting string)."""
    a = v["arm"]
    if a == "esrgan":
        return a, "native" if v["pre"] == 0 else f"pre{v['pre']}"
    if a == "invsr":
        return a, f"pre{v['pre']}_st{v['steps']}"
    if a == "tadsr":
        return a, f"pre{v['pre']}_t{v['t']}"
    return a, f"pre{v['pre']}"


def out_dir(part, v, seed):
    a, s = vid(v)
    return ROOT / "outputs/sr/restored" / part / f"{a}__{s}" / f"seed{seed}"


def n_inputs(part):
    return len(list((ROOT / f"outputs/sr/subset/{part}_input").glob("*.png")))


def restore(cfg, part, v, seed, limit=None):
    """Run one variant's restoration in its own venv (resumable)."""
    a, s = vid(v)
    arm = cfg["arms"][a]
    od = out_dir(part, v, seed)
    t = od / "timing.json"
    n_in = limit or n_inputs(part)
    if t.exists():
        j = json.loads(t.read_text())
        if j.get("n_ok", 0) + j.get("n_failed", 0) >= n_in:
            return od
    rel = str(od.relative_to(ROOT)).replace("\\", "/")
    cmd = [str(ROOT / arm["python"]), "-W", "ignore", str(ROOT / arm["runner"]), "--input-dir", f"outputs/sr/subset/{part}_input"]
    tmp = ["--out-root", "outputs/sr/restored/_tmp"]
    if a == "bicubic":
        cmd += tmp + ["--sizes", str(v["pre"]), "--out-template", rel]
    elif a == "esrgan":
        cmd += tmp + ["--pre-sizes", str(v["pre"]), "--out-template", rel]
    elif a == "invsr":
        cmd += ["--out-dir", rel, "--pre-size", str(v["pre"]), "--steps", str(v["steps"]), "--seed", str(seed)]
    elif a == "tadsr":
        cmd += tmp + ["--pre-size", str(v["pre"]), "--timesteps", str(v["t"]), "--seed", str(seed), "--out-template", rel]
    else:
        cmd += tmp + ["--pre-sizes", str(v["pre"]), "--seed", str(seed), "--out-template", rel, "--sd-path", cfg["osdface_sd_path"]]
    if limit:
        cmd += ["--limit", str(limit)]
    log.info(f"restore {part} {a}__{s} seed{seed} ...")
    logf = od.parent / f"seed{seed}_stdout.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):  # a crash (e.g. GPU memory pressure from other workers) resumes where it stopped
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        logf.write_text((r.stdout or "")[-8000:] + "\n--- stderr ---\n" + (r.stderr or "")[-8000:], encoding="utf-8")
        if r.returncode == 0:
            break
        log.error(f"restore FAILED {a}__{s} seed{seed} attempt {attempt}: returncode {r.returncode}; see {logf}")
        time.sleep(60)
    return od


def make_recs(cfg):
    return {n: Recognizer(n, ROOT / w, (112, 112), "bilinear", True, 256, "cuda") for n, w in cfg["recognizers"].items()}


def score(cfg, recs, part, items, results_csv, B):
    """items: list of (variant dict, seed). Scores direct feed + every item; incremental CSV."""
    d, pairs = E.load_subset(ROOT, part)
    cache = ROOT / "outputs/sr/emb" / part
    S = cfg["bootstrap"]["seed"]
    dc = {}
    all_rows = E.evaluate_variant(recs, d, pairs, ROOT, "direct", "resize", 0, part, None, set(), cache, B, S, dc)
    for r in all_rows:
        r["status"] = "OK"
    restore_splits = {"probe", "mated_probe", "unmated_probe", "verification"}
    if cfg.get("restore_gallery", True):
        restore_splits |= {"gallery_match", "distractor", "gallery"}
    for v, seed in items:
        a, s = vid(v)
        od = out_dir(part, v, seed)
        tinfo = E.timing_info(od)
        done = (tinfo.get("n_ok") or 0) + (tinfo.get("n_failed") or 0)
        if not tinfo or done < len(d) or tinfo.get("n_failed"):
            msg = f"INCOMPLETE/FAILED restoration: n_ok={tinfo.get('n_ok')} n_failed={tinfo.get('n_failed')} of {len(d)}"
            log.warning(f"{a}__{s} seed{seed}: {msg}")
            all_rows.append(dict(arm=a, setting=s, seed=seed, part=part, status=msg, **tinfo))
            if not tinfo or (tinfo.get("n_ok") or 0) < len(d):
                pd.DataFrame(all_rows).to_csv(results_csv, index=False)
                continue
        rows = E.evaluate_variant(recs, d, pairs, ROOT, a, s, seed, part, od, restore_splits, cache, B, S, dc)
        for r in rows:
            r.update(tinfo)
            r["status"] = "OK"
        all_rows += rows
        pd.DataFrame(all_rows).to_csv(results_csv, index=False)
        log.info(f"scored {part} {a}__{s} seed{seed}")
    df = pd.DataFrame(all_rows)
    df.to_csv(results_csv, index=False)
    return df


def selection(df):
    d = df[(df["status"] == "OK") & df["recognizer"].isin(E.NON_LEAKY) & (df["arm"] != "direct")]
    t = d[d["protocol"] == "tinyface"][["arm", "setting", "seed", "recognizer", "Rank-1"]]
    v = d[d["protocol"] == "survface_verification"][["arm", "setting", "seed", "recognizer", "AUC"]]
    m = t.merge(v, on=["arm", "setting", "seed", "recognizer"])
    m["score"] = (m["Rank-1"] + m["AUC"]) / 2
    g = m.groupby(["arm", "setting"]).agg(score=("score", "mean"), rank1=("Rank-1", "mean"), auc=("AUC", "mean")).reset_index()
    g = g.sort_values(["arm", "score"], ascending=[True, False])
    return g, g.groupby("arm").head(1)


def find_variant(cfg, arm, setting):
    return next(v for v in cfg["tuning_variants"] if vid(v) == (arm, setting))


def score_free(cfg, part, items, csv, B):
    """Score with freshly loaded recognizers and release them. A lock directory lets only one worker score at a time
    (a stale lock older than 2 h is removed)."""
    lock = ROOT / "outputs/sr/.score_lock"
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 7200:
                    os.rmdir(lock)
            except OSError:
                pass
            time.sleep(15)
    try:
        recs = make_recs(cfg)
        try:
            return score(cfg, recs, part, items, csv, B)
        finally:
            del recs
            torch.cuda.empty_cache()
    finally:
        os.rmdir(lock)


def run_arm(cfg, arm):
    """Tune -> select -> evaluate -> seed study for one arm. Resumable; writes per-arm CSVs."""
    variants = [v for v in cfg["tuning_variants"] if v["arm"] == arm]
    log.info(f"[{arm}] tuning sweep: {[vid(v)[1] for v in variants]}")
    for v in variants:
        restore(cfg, "tuning", v, 0)
    df = score_free(cfg, "tuning", [(v, 0) for v in variants], RES / f"tuning_{arm}.csv", cfg["bootstrap"]["tuning_n"])
    g, best = selection(df)
    g.to_csv(RES / f"tuning_selection_{arm}.csv", index=False)
    (RES / f"selection_{arm}.json").write_text(json.dumps(best.to_dict("records"), indent=2))
    log.info(f"[{arm}] selection table:\n" + g.to_string(index=False))
    if best.empty:
        log.error(f"[{arm}] no scored variant; stopping this arm")
        return
    sel = best.iloc[0]
    v = find_variant(cfg, arm, sel["setting"])
    if v["pre"] == 256 and arm in ("invsr", "tadsr"):
        msg = f"[{arm}] selected {sel['setting']} (pre-size 256, 7-10 h on the eval subset). STOPPED: needs your approval."
        (RES / f"needs_approval_{arm}.txt").write_text(msg)
        log.warning(msg)
        return
    log.info(f"[{arm}] selected {sel['setting']} (score {sel['score']:.4f}); evaluating on the eval subset")
    restore(cfg, "eval", v, 0)
    score_free(cfg, "eval", [(v, 0)], RES / f"eval_{arm}.csv", cfg["bootstrap"]["eval_n"])
    log.info(f"[{arm}] eval scored")
    if cfg["arms"][arm]["stochastic"]:
        for sd in (1, 2):
            restore(cfg, "tuning", v, sd)
        score_free(cfg, "tuning", [(v, sd) for sd in (0, 1, 2)], RES / f"seed_study_{arm}.csv", cfg["bootstrap"]["tuning_n"])
        log.info(f"[{arm}] seed study scored")
    log.info(f"[{arm}] ARM COMPLETE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["direct", "arm"])
    ap.add_argument("--arms", nargs="+", default=[], help="arms handled sequentially by this worker")
    ap.add_argument("--config", default="configs/sr.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    RES.mkdir(parents=True, exist_ok=True)
    tag = "direct" if a.cmd == "direct" else "_".join(a.arms)
    logging.basicConfig(level=logging.INFO, format=f"[%(asctime)s] [{tag}] %(message)s",
                        handlers=[logging.FileHandler(ROOT / "outputs/sr/run.log"), logging.StreamHandler(sys.stdout)])
    seed_all(cfg["seed"], True, True)
    if a.cmd == "direct":
        # compute the direct-feed embeddings/per-sample files once, so parallel workers only read them
        recs = make_recs(cfg)
        for part, B in (("tuning", cfg["bootstrap"]["tuning_n"]), ("eval", cfg["bootstrap"]["eval_n"])):
            d, pairs = E.load_subset(ROOT, part)
            rows = E.evaluate_variant(recs, d, pairs, ROOT, "direct", "resize", 0, part, None, set(), ROOT / "outputs/sr/emb" / part,
                                      B, cfg["bootstrap"]["seed"], {})
            pd.DataFrame(rows).to_csv(RES / f"direct_{part}.csv", index=False)
            log.info(f"direct feed scored on {part}")
        return
    for arm in a.arms:
        run_arm(cfg, arm)


if __name__ == "__main__":
    main()
