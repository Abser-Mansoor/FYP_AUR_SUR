# Phase 4 handoff (written 2026-10-23, about 21:35)

Give this file to Claude at the start of the next session and say "continue Phase 4 from this handoff". The runbook is `research/Sprint_Notes/Phase4_restoration_runbook.md`; this file records where the work stopped.

**Machine:** Raghib's laptop, repo `D:\PROJECTS\FYP\FYP_AUR_SUR`, local branch `phase_4` (renamed from `phase4-enhance`; tracks `origin/phase_4`, which equalled `main`). **Nothing is pushed yet.** Venvs: `envs\score`, `envs\zerodce`, `envs\nafnet` (all torch 2.6.0+cu124, Quadro RTX 3000 works). `uv` is run as `python -m uv`. Drive bundle path used: `g:\.shortcut-targets-by-id\1jj9zaEy9AHTi9VIn4AAbz37lf4domATL\FYP_SR_SHARED\phase4` (the `G:\My Drive\...` path does not exist here).

## Done and verified
- Step 0 (bundle) by Abser; Step 1 setup, unpack with SHA-256 check: done.
- Cross-device check passed (`outputs/enhance/results/xcheck.json`, min cosine 0.999987 over the 4 recognizers).
- Harness generalised, Phase 3 behaviour unchanged (the Phase 3 `sr_report_tables.py` rerun matched the committed file except the "Remote device" column, which needs Colab `timing.json` files that are not on this laptop).
- Runners: `run_photometric.py`, `run_zerodce.py`, `run_nafnet.py`; vendored code + weights + SHA-256 in `MODEL_INVENTORY.md` and `src/sr/vendor/SOURCES.md`.
- Harness sanity: `noop` embeddings equal direct (`np.array_equal`, 4 recognizers, tuning and eval); noop paired Rank-1 delta exactly 0. Resume test passed (CLAHE and NAFNet). Smoke run (75 crops) and montage: `outputs/enhance/smoke/`.
- Tuning sweep (all 16 variants, 0 failures) and selections: bicubic pre128, gamma adaptive, clahe c2_t2, unsharp a0.5_s0.5, zerodce s1, nafnet pre0 (native). noop scored 0.5728 on tuning; every selected arm scored at or below it.
- Gates chosen on tuning (`enh_gate.py`): low-light = gamma-adaptive, enhance if luminance < 60.17 (q10); deblur = unsharp a0.5_s0.5, enhance if lap_var < 3.369 (q50). Gating sanity (tau 0 = direct, tau inf = full arm) passed.
- Pre-registered rules committed in `REPORT_phase4.md` (commit d420a69) **before** any eval scoring.
- Eval subset (22,898 crops) restored and scored for bicubic, gamma, clahe, unsharp, zerodce, nafnet (all 0 failed), and the gated eval (`eval_gated.csv`; gates enhanced 8.8% and 55.6% of eval crops). **No eval numbers have been read or interpreted yet.**

## UPDATE 21:50: the LFW calibration was stopped on purpose (Raghib ended the session); nothing is running now.
The probes-only analysis finished and is committed. The calibration has cached embeddings in `outputs/sr/lfw/emb_enh/` (hr112 and the first strengths; partial files were removed), so rerunning `enh_lfw_check.py calibrate` only computes the missing strengths (about 4.3 min per strength, 16 strengths in total). `lfw_enh_calibration.csv` does not exist yet. The rest of the list below still applies; skip the "check first" part.

## Running when this was written (check first)
Two jobs were started at ~21:30; they may have finished, or died if the laptop slept or the session was interrupted (earlier background runs died that way):
1. **FINISHED at ~21:40, results committed**: `scripts\sr_extra_analysis.py --config configs\enhance.yaml --workers 4` (CPU; writes `probes_only.csv`, `per_bin.csv`, `extra_tables.md` in `outputs/enhance/results/` and injects the EXTRA block into `REPORT_phase4.md`).
2. `scripts\enh_lfw_check.py calibrate` (GPU; writes `lfw_enh_calibration.csv`, which must exist before any LFW arm is restored).

Check: `Get-Process python`, `nvidia-smi`, and whether those output files exist. If a file is missing, rerun that command (all steps resume). Do not run two GPU-heavy jobs at once (6 GB VRAM).

## Remaining work, in order
1. Confirm the two jobs above finished; rerun if not.
2. `envs\score\Scripts\python.exe scripts\enh_lfw_check.py restore` then `... score` (or `all`). Restores 5 arms on the dark / gauss / motion sets at 112 px, then scores 4 recognizers with paired bootstrap. Not yet run or tested beyond a syntax check, so expect to debug; `enh_lfw_check.py` has never been executed past `calibrate`.
3. `envs\score\Scripts\python.exe scripts\sr_report_tables.py --config configs\enhance.yaml` (injects TABLES, stop-rule, cost and LFW tables into `REPORT_phase4.md`; not yet run, so debug if needed).
4. Figures: `scripts\sr_failure_cases.py --config configs\enhance.yaml`, `scripts\sr_sample_figs.py --config configs\enhance.yaml`, `scripts\sr_brief_strips.py --config configs\enhance.yaml` (new code paths in `scripts/enh_samples.py`; untested).
5. Write `REPORT_phase4.md` sections 1-4, 7, 8, 10, 11 (currently placeholders) with the mechanical stop-rule verdict from the generated stop-rule table. Every number must come from `outputs/enhance/results/*.csv` or the injected tables. State in Limitations: lap_var gate size-confounded, dark-bin size, synthetic LFW degradations, multiplicity (8 tests; Bonferroni CIs are in `probes_only.csv`), different GPU from Phase 3.
6. `scripts\phase4_upload.py` (new, untested): zips results, report, thumbnails (default 1,000 seeded crops per selected variant, not all 22,898; mention this deviation from "thumbnails of the selected variants") to the Drive `phase4` folder with a SHA-256 manifest.
7. Commit everything on `phase_4`, **ask Raghib before pushing** and opening the PR to `main` for Abser. Then STOP for supervisor approval (end-of-W2 checkpoint).

## Open items and cautions
- Uncommitted: the new `eval_*.csv`, `eval_gated.csv` (commit them) and anything produced by the two running jobs.
- Raghib has not yet been asked about pushing `phase_4` to GitHub.
- `Timing`/runners: NAFNet loads all crops into memory before the first output appears (about 4-5 min of silence on the eval subset); that is normal.
- Tool quirks in this environment: long `sleep` in a tool call is blocked (use short checks or `run_in_background`); in Bash heredocs a literal backslash-n gets turned into a real newline, so write patch scripts to files with the Write tool or use `chr(92)`; background jobs launched from a tool call can be killed on session interruption.
- Files changed this phase: `scripts/{sr_pipeline,sr_extra_analysis,sr_report_tables,sr_failure_cases,sr_sample_figs,sr_brief_strips}.py`, `src/sr/{eval,common}.py`; new `scripts/{enh_xcheck,enh_gate,enh_lfw_check,enh_tables,enh_samples,phase4_upload}.py`, `src/sr/run_{photometric,zerodce,nafnet}.py`, `configs/enhance.yaml`, `REPORT_phase4.md`.
