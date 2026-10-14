"""Regenerate the STATUS block of CONTEXT.md from the files on disk and mirror CONTEXT.md to the shared Drive folder.

    envs/diag/Scripts/python.exe scripts/update_context.py [--no-drive]

Reads only: outputs/sr/restored (image counts), outputs/sr/results (CSV / selection files), the Drive jobs folder
(chunk markers, DONE, xcheck.json, device.json). It never writes numbers by hand.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SHARED = Path("G:/My Drive/FYP_SR_SHARED")
BEGIN, END = "<!-- STATUS:BEGIN -->", "<!-- STATUS:END -->"
sys.path.insert(0, str(ROOT / "scripts"))


def vid(v):
    a = v["arm"]
    if a == "esrgan":
        return a, "native" if v["pre"] == 0 else f"pre{v['pre']}"
    if a == "invsr":
        return a, f"pre{v['pre']}_st{v['steps']}"
    if a == "tadsr":
        return a, f"pre{v['pre']}_t{v['t']}"
    return a, f"pre{v['pre']}"


def n_png(d):
    return len(list(d.glob("*.png"))) if d.exists() else 0


def variant_done(part, arm, setting, seed, n):
    d = ROOT / "outputs/sr/restored" / part / f"{arm}__{setting}" / f"seed{seed}"
    t = d / "timing.json"
    if t.exists():
        try:
            j = json.loads(t.read_text())
            if (j.get("n_ok") or 0) >= n:
                return True, n
        except ValueError:
            pass
    k = n_png(d)
    return k >= n, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-drive", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/sr.yaml"))
    n = {p: sum(1 for _ in open(ROOT / f"outputs/sr/subset/{p}_images.csv", encoding="utf-8")) - 1 for p in ("tuning", "eval")}
    res = ROOT / "outputs/sr/results"
    lines = [f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/update_context.py` (read-only scan of the files below).", ""]

    # --- per-arm local pipeline state
    lines += ["### Phase 3 arms (outputs on the lab PC)", "",
              "| Arm | Tuning variants restored | Selected setting | Eval scored | Seed study scored |", "|---|---|---|---|---|"]
    for arm in ("bicubic", "esrgan", "invsr", "tadsr", "osdface"):
        vs = [v for v in cfg["tuning_variants"] if v["arm"] == arm]
        done = sum(variant_done("tuning", *vid(v), 0, n["tuning"])[0] for v in vs)
        sel = res / f"selection_{arm}.json"
        selected = "-"
        if sel.exists():
            try:
                j = json.loads(sel.read_text())
                selected = j[0]["setting"] if j else "(none scored)"
            except ValueError:
                selected = "(unreadable)"
        lines.append(f"| {arm} | {done} of {len(vs)} | {selected} | {'yes' if (res / f'eval_{arm}.csv').exists() else 'no'} | "
                     f"{('yes' if (res / f'seed_study_{arm}.csv').exists() else 'no') if cfg['arms'][arm]['stochastic'] else 'n/a (deterministic)'} |")
    extra = [p.name for p in res.glob("eval_*_sensitivity.csv")]
    if extra:
        lines += ["", "Extra result files: " + ", ".join(extra)]
    for f in res.glob("needs_approval_*.txt"):
        lines += ["", f"**NEEDS SUPERVISOR APPROVAL:** {f.read_text().strip()}"]

    # --- remote (Drive) jobs
    lines += ["", "### Colab / remote jobs (Drive `FYP_SR_SHARED/jobs`)", ""]
    jobs = sorted((SHARED / "jobs").glob("*/job.json")) if SHARED.exists() else []
    if not jobs:
        lines.append("Drive folder not visible from this machine, or no jobs published yet.")
    else:
        lines += ["| Job | Machine | Chunks done | State | Cross-device check | GPU |", "|---|---|---|---|---|---|"]
        for jp in jobs:
            job, jd = json.loads(jp.read_text()), jp.parent
            total = math.ceil(n[job["part"]] / 1000)
            done = len(list((jd / "chunks").glob("*/embedded"))) if (jd / "chunks").exists() else 0
            failed = len(list((jd / "chunks").glob("*/FAILED.txt"))) if (jd / "chunks").exists() else 0
            state = "DONE" if (jd / "DONE").exists() else ("running or paused" if done else "not started")
            if failed:
                state += f" ({failed} chunk(s) FAILED)"
            xc = "-"
            if (jd / "xcheck.json").exists():
                try:
                    xj = json.loads((jd / "xcheck.json").read_text())
                    xc = "pass" if xj.get("pass") else "FAIL"
                except ValueError:
                    xc = "unreadable"
            dev = "-"
            if (jd / "device.json").exists():
                try:
                    dev = json.loads((jd / "device.json").read_text()).get("gpu", "-")
                except ValueError:
                    pass
            lines.append(f"| {job['job_id']} | {job.get('machine', '-')} | {done} / {total} | {state} | {xc} | {dev} |")

    # --- local worker
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'sr_pipeline.py' }).Count"],
                           capture_output=True, text=True, timeout=60)
        running = int((r.stdout or "0").strip() or 0) >= 1
    except Exception:
        running = None
    lines += ["", f"Lab-PC worker (`SR_worker`, TADSR): {'running' if running else 'not running' if running is not None else 'unknown'}."]
    block = "\n".join(lines)

    p = ROOT / "CONTEXT.md"
    s = p.read_text(encoding="utf-8")
    i, j = s.index(BEGIN), s.index(END)
    p.write_text(s[:i + len(BEGIN)] + "\n" + block + "\n" + s[j:], encoding="utf-8")
    print("updated", p)
    if not a.no_drive and SHARED.exists():
        shutil.copy2(p, SHARED / "CONTEXT.md")
        print("mirrored to", SHARED / "CONTEXT.md")


if __name__ == "__main__":
    main()
