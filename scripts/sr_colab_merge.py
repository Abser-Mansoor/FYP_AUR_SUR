"""Merge finished remote jobs (Colab) into the local Phase 3 pipeline, validate, then score.

    envs/diag/Scripts/python.exe scripts/sr_colab_merge.py                      # merge + score every job with a DONE marker
    envs/diag/Scripts/python.exe scripts/sr_colab_merge.py --dry-run            # validate only, change nothing
    envs/diag/Scripts/python.exe scripts/sr_colab_merge.py --shared <dir>       # other jobs root (tests)

Checks per job: chunk names cover the subset's `file` column exactly and in order; every embedding file has shape
(n, 512), finite values and unit-scale norms; xcheck.json (if present) passed. Then it rebuilds
outputs/sr/emb/<part>/<arm>/<setting>/seed<k>/<part>__<recognizer>.npy (the cache path eval.get_embeddings reads),
writes a timing.json (remote device recorded), copies the chunk thumbnails, and scores the variants with the
existing code (sr_pipeline.score_free). OSDFace tuning jobs also run the selection rule once both are present.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import sr_pipeline as P  # noqa: E402

RECS = ("arcface", "adaface", "magface", "curricularface")
RES = ROOT / "outputs/sr/results"


def variant_of(job):
    a = dict(arm=job["arm"], pre=job["args"]["pre"])
    if "steps" in job["args"]:
        a["steps"] = job["args"]["steps"]
    return a


def validate_and_load(jdir, job, files):
    """Returns ({rec: [N,512]}, timing dict) or raises ValueError with the reason."""
    chunk_dirs = sorted((jdir / "chunks").glob("[0-9]*"))
    names, embs, timings = [], {r: [] for r in RECS}, []
    for cd in chunk_dirs:
        if not (cd / "embedded").exists():
            raise ValueError(f"chunk {cd.name} has no 'embedded' marker")
        n = json.loads((cd / "names.json").read_text())
        names += n
        for r in RECS:
            e = np.load(cd / f"emb_{r}.npy")
            if e.shape != (len(n), 512):
                raise ValueError(f"chunk {cd.name} {r}: shape {e.shape}, expected {(len(n), 512)}")
            if not np.isfinite(e).all():
                raise ValueError(f"chunk {cd.name} {r}: non-finite values")
            nm = np.linalg.norm(e, axis=1)
            if not (0.5 < nm.min() and nm.max() < 2.5):
                raise ValueError(f"chunk {cd.name} {r}: embedding norms outside 0.5-2.5 ({nm.min():.2f}-{nm.max():.2f})")
            embs[r].append(e)
        timings.append(json.loads((cd / "timing.json").read_text()) if (cd / "timing.json").exists() else {})
    if names != files:
        raise ValueError(f"chunk names do not match the subset (got {len(names)} names, expected {len(files)}; first mismatch "
                         f"at {next((i for i, (a, b) in enumerate(zip(names, files)) if a != b), 'length')})")
    return {r: np.concatenate(v) for r, v in embs.items()}, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", default="G:/My Drive/FYP_SR_SHARED/jobs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/sr.yaml"))
    # group jobs into variants (part, arm, setting, seed); a variant may be split into crop ranges (shards)
    groups = {}
    for jdir in sorted(Path(a.shared).glob("*/")):
        if (jdir / "job.json").exists():
            job = json.loads((jdir / "job.json").read_text())
            v = variant_of(job)
            groups.setdefault((job["part"], *P.vid(v), job["seed"]), []).append((jdir, job, v))
    done = []
    for (part, arm, setting, seed), members in sorted(groups.items()):
        members.sort(key=lambda m: (m[1].get("range") or [0])[0])
        label = f"{part}/{arm}/{setting}/seed{seed}"
        pending = [m[1]["job_id"] for m in members if not (m[0] / "DONE").exists()]
        if pending:
            print(f"{label}: waiting for {len(pending)} of {len(members)} job(s): {pending}")
            continue
        files = pd.read_csv(ROOT / "outputs/sr/subset" / f"{part}_images.csv")["file"].tolist()
        try:
            embs, timings, covered = {r: [] for r in RECS}, [], 0
            for jdir, job, v in members:
                lo, hi = job.get("range") or [0, len(files)]
                if lo != covered:
                    raise ValueError(f"shards leave a gap or overlap at crop {covered} (next shard starts at {lo})")
                e, t = validate_and_load(jdir, job, files[lo:hi])
                for r in RECS:
                    embs[r].append(e[r])
                timings += t
                covered = hi
            if covered != len(files):
                raise ValueError(f"shards cover {covered} of {len(files)} crops")
            emb = {r: np.concatenate(x) for r, x in embs.items()}
        except ValueError as e:
            print(f"{label}: INVALID - {e}")
            continue
        xinfo = [json.loads((m[0] / "xcheck.json").read_text()) if (m[0] / "xcheck.json").exists() else {} for m in members]
        xpass = all(x.get("pass", False) for x in xinfo if x) and all(xinfo)
        if not xpass:
            print(f"{label}: WARNING a cross-device check is missing or did not pass")
        print(f"{label}: valid ({len(files)} crops from {len(members)} job(s))")
        if a.dry_run:
            continue
        v = members[0][2]
        tag = ROOT / "outputs/sr/emb" / part / arm / setting / f"seed{seed}"
        tag.mkdir(parents=True, exist_ok=True)
        for r in RECS:
            np.save(tag / f"{part}__{r}.npy", emb[r])
        med = [t["sec_per_crop_median"] for t in timings if t.get("sec_per_crop_median")]
        devs = sorted({json.loads((m[0] / "device.json").read_text()).get("gpu") for m in members if (m[0] / "device.json").exists()})
        od = P.out_dir(part, v, seed)
        od.mkdir(parents=True, exist_ok=True)
        (od / "timing.json").write_text(json.dumps(dict(
            model=arm, n_ok=len(files), n_failed=0, n_timed=len(files), remote=True, device=devs, jobs=[m[1]["job_id"] for m in members],
            sec_per_crop_median=float(np.median(med)) if med else None, sec_per_crop_mean=float(np.mean(med)) if med else None,
            peak_vram_mb=max([t.get("peak_vram_mb") or 0 for t in timings] or [0]), xcheck_pass=xpass,
            note="restored on remote GPU(s); PNGs not copied, embeddings merged from the shared folder"), indent=1))
        for jdir, job, _ in members:
            th = ROOT / "outputs/sr/thumbs" / job["job_id"]
            th.mkdir(parents=True, exist_ok=True)
            for z in (jdir / "chunks").glob("*/thumbs.zip"):
                shutil.copy2(z, th / f"{z.parent.name}.zip")
        done.append((part, arm, seed, v))
    if a.dry_run or a.no_score or not done:
        return
    B_t, B_e = cfg["bootstrap"]["tuning_n"], cfg["bootstrap"]["eval_n"]
    # OSDFace tuning sweep (seed 0) -> table + selection
    tun = [(v, 0) for part, arm, seed, v in done if part == "tuning" and arm == "osdface" and seed == 0]
    if tun:
        df = P.score_free(cfg, "tuning", tun, RES / "tuning_osdface.csv", B_t)
        g, best = P.selection(df)
        g.to_csv(RES / "tuning_selection_osdface.csv", index=False)
        (RES / "selection_osdface.json").write_text(json.dumps(best.to_dict("records"), indent=2))
        print("OSDFace tuning table:\n" + g.to_string(index=False))
    # InvSR pre-128 sensitivity check (not the rule-selected setting)
    sens = [(v, seed) for part, arm, seed, v in done if part == "eval" and arm == "invsr"]
    if sens:
        P.score_free(cfg, "eval", sens, RES / "eval_invsr_sensitivity.csv", B_e)
        print("scored InvSR sensitivity ->", RES / "eval_invsr_sensitivity.csv")
    # OSDFace at the selected setting: eval subset, and the seed study (seeds 0-2 on tuning) once seeds 1 and 2 exist
    sel_f = RES / "selection_osdface.json"
    if sel_f.exists():
        sel = json.loads(sel_f.read_text())[0]["setting"]
        ev = [(v, seed) for part, arm, seed, v in done if part == "eval" and arm == "osdface" and P.vid(v)[1] == sel]
        if ev:
            P.score_free(cfg, "eval", ev, RES / "eval_osdface.csv", B_e)
            print("scored OSDFace eval ->", RES / "eval_osdface.csv")
        seeds = sorted({seed for part, arm, seed, v in done if part == "tuning" and arm == "osdface" and P.vid(v)[1] == sel})
        if seeds == [0, 1, 2]:
            v = next(v for part, arm, seed, v in done if part == "tuning" and arm == "osdface" and P.vid(v)[1] == sel)
            P.score_free(cfg, "tuning", [(v, sd) for sd in seeds], RES / "seed_study_osdface.csv", B_t)
            print("scored OSDFace seed study ->", RES / "seed_study_osdface.csv")


if __name__ == "__main__":
    main()
