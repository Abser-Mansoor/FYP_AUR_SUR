# Phase 4 runbook: restoration test, low-light and deblur

**Status:** approved 2026-10-22. Raghib runs this on his own laptop; the code changes are not made yet.

**Runs on:** Raghib's Windows 10/11 laptop (Quadro RTX 3000, 6 GB VRAM).
**Window:** W1–W2, 26 Oct – 8 Nov 2026, at Raghib's own pace.
**Owner:** Raghib. **Support:** Abser prepares the data bundle on the lab PC.
**Parent plan:** `research/Sprint_Notes/Sprint_2_implementation_plan.md`, Phase 4.

## Context
- **Why.**
  - Phase 3 closed generative SR as a negative result.
  - The supervisor brief (section 6) asks for one cheap, **non-generative** low-light and deblur test before restoration is closed or kept as an optional gated stage.
  - The video track uses the lab PC GPU, so Phase 4 runs on Raghib's laptop, which also has a GPU.
- **Question.** Does any low-light or deblur method give a **probes-only** recognition gain with a 95% CI above zero?
  - Yes: the winner becomes the gate arm for Phase 8.
  - No: close the restoration engine and record that.
- **Approach.** Reuse the Phase 3 harness unchanged in behaviour: the same subsets, recognizers, selection rule, probes-only and per-bin analysis, and LFW check. Phase 4 numbers then compare directly with Phase 3.

**Standing rules:**
- inference only, no training;
- seed 42;
- every number comes from a script output;
- failures are logged as results, never dropped;
- datasets are read-only;
- one venv per model repo;
- config-driven;
- Windows-native: PowerShell, `num_workers=0`, no symlinks.

**One device per reported number.** Every Phase 4 number is computed on the RTX 3000, **including the direct-feed baseline**. Paired deltas therefore never mix devices. The lab-PC direct embeddings are used only for a cross-device check.

## Where the GPU helps (RTX 3000, 6 GB)

| Step | Device | Why |
|---|---|---|
| Scoring: embedding 22,898 crops × 4 recognizers × flip | **GPU** (required in practice) | Hours on CPU, minutes on GPU. Use batch 128 (256 may exceed 6 GB with 4 IR-100/101 models loaded). |
| NAFNet-GoPro-width64 (native and pre-128) | **GPU**, batch 32 native, 8 at pre-128 | The largest model. Expected 10×+ faster than CPU; measured in the smoke run. |
| Zero-DCE++ | GPU (CPU also fine) | About 10k parameters, so the speedup is small. Use GPU for consistency. |
| gamma / CLAHE / unsharp / noop | CPU (OpenCV) | Already milliseconds per crop |

**Laptop settings for long runs:**
- plugged in;
- Windows power mode "Best performance";
- sleep disabled while plugged in;
- the NVIDIA driver supports CUDA 12.4 (`nvidia-smi` shows driver ≥ 550).

Turing (sm_75) is supported by torch 2.6.0+cu124.

## Step 0: Abser prepares the data bundle on the lab PC (once, before W1)
New script `scripts/phase4_bundle.py`, run in `envs/diag`. It writes to `G:\My Drive\FYP_SR_SHARED\phase4\`, with a `MANIFEST.json` holding the SHA-256 of every file:
- `subset_inputs.zip`: copied from `bundle/` (tuning and eval crops, about 90 MB);
- `recognizers.zip`: copied from `bundle/` (4 weights, about 1.19 GB);
- `phase4_data.zip`:
  - `outputs/stratify/measures/*.csv` (raw luminance and `lap_var`; gitignored);
  - `outputs/sr/lfw/hr112/` (7,701 aligned 112 px LFW crops) and `data/lfw/lfw_home/pairs.txt`;
  - `outputs/sr/smoke/input/` and `manifest.csv` (75 smoke crops);
  - `outputs/sr/emb/{tuning,eval}/direct/resize/seed0/*.npy` (lab-PC direct embeddings, used only for the cross-device check).

The code comes from GitHub (`Abser-Mansoor/FYP_AUR_SUR`). Abser commits and pushes the Phase 4 code changes, or Raghib makes them on a branch `phase4-enhance`.

## Step 1: Raghib's laptop setup (W1 Mon)
Run in PowerShell. Install `git` and `uv` first if missing.

```powershell
git clone https://github.com/Abser-Mansoor/FYP_AUR_SUR.git D:\FYP_AUR_SUR ; cd D:\FYP_AUR_SUR
# Drive for desktop must show G:\My Drive\FYP_SR_SHARED (add shortcut to My Drive if it is shared with you)
python scripts\phase4_unpack.py --src "G:\My Drive\FYP_SR_SHARED\phase4"   # SHA-256 check, then unzip into the repo layout
uv venv envs\score --python 3.11
uv pip install --python envs\score\Scripts\python.exe torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python envs\score\Scripts\python.exe numpy opencv-python pandas scipy pyyaml matplotlib pillow
envs\score\Scripts\python.exe -c "import torch;print(torch.cuda.get_device_name(0))"   # expect Quadro RTX 3000
```

- `scripts/phase4_unpack.py` is new and stdlib-only. It refuses to unpack if any hash mismatches.
- **`envs/score`** replaces the lab's `envs/diag` for scoring, the OpenCV arms and the analysis scripts.
- **`envs/zerodce`** and **`envs/nafnet`** are separate uv venvs: Python 3.11, torch 2.6.0+cu124 plus the minimal packages for the vendored model code. One venv per model repo.

**Cross-device check (must pass before anything else):**
- `scripts/enh_xcheck.py` embeds the direct feed of the tuning subset on the RTX 3000;
- it compares the result with the lab-PC `.npy` files;
- per-crop cosine must exceed 0.9999 for all 4 recognizers, the same threshold as the Phase 3 Colab check;
- it writes `outputs/enhance/results/xcheck.json`;
- a failure means STOP: report it to Abser.

## Arms and tuning variants (`configs/enhance.yaml`)

| Family | Arm | Tuning variants | Venv | Device | Embedding rule |
|---|---|---|---|---|---|
| control | `noop` | copy of the input | score | CPU | bilinear (direct rule) |
| control | `bicubic` | pre-128 (comparator for NAFNet pre-128; existing runner) | score | CPU | area |
| low-light | `gamma` | γ ∈ {0.5, 0.7}; adaptive γ = log(0.45)/log(mean luminance/255), clamped to [0.3, 1] | score | CPU | bilinear |
| low-light | `clahe` | L channel of Lab; clip ∈ {1, 2} × tiles ∈ {2×2, 4×4} | score | CPU | bilinear |
| low-light | `zerodce` | Zero-DCE++ `Epoch99.pth`, scale_factor 1, native size | zerodce | GPU | bilinear |
| deblur | `unsharp` | σ ∈ {0.5, 1.0} × amount ∈ {0.5, 1.0} | score | CPU | bilinear |
| deblur | `nafnet` | NAFNet-GoPro-width64: native (reflect-padded to a multiple of 16, cropped back) and pre-128 (bicubic, then NAFNet) | nafnet | GPU | native: bilinear; pre-128: area |

**Gated variants.** These are built from cached embeddings, with no new restoration. Enhanced and direct crops share the bilinear rule, so gating at embedding level equals gating the image exactly.
- **Low-light gate:** enhance only if raw luminance is below τ_L.
- **Deblur gate:** enhance only if raw `lap_var` is below τ_S.
- **Candidate τ:** the 10th, 20th, 30th and 50th percentiles of the pooled tuning distribution, from `outputs/stratify/measures/*.csv`.
- It is one global threshold, as a deployable gate would use, not the per-split Phase 2 bins.
- Caveat: `lap_var` is size-confounded (Phase 2). The report states this.

**Selection.** Unchanged from Phase 3: the best mean over AdaFace, MagFace and CurricularFace of [TinyFace Rank-1, SurvFace verification AUC], on the tuning subset (2,800 crops).
- One setting is chosen per arm.
- One τ is chosen per gated family, on that family's selected arm.

## Pre-registered decision rules
These rules are copied into `REPORT_phase4.md` **before** the eval subset is scored.
1. **Stop rule.**
   - Restoration closes if no selected arm or gated arm gives a probes-only TinyFace Rank-1 paired gain whose 95% CI is above zero for at least 2 of the 3 non-leaky recognizers.
   - Otherwise the winner is the arm with the highest mean probes-only gain, and it becomes the Phase 8 gate arm.
   - Up to 10 tests run. The report states this, and Bonferroni-adjusted CIs are shown as a sensitivity check.
2. **LFW check.**
   - An arm "recovers identity" if the paired CI of AUC(cross-enhanced) − AUC(cross-degraded) is above zero for at least 2 of the 3 non-leaky recognizers.
   - Each arm is also compared post-hoc against no-op.
3. **Harness sanity.**
   - `noop` embeddings must equal direct-feed embeddings exactly (`np.array_equal`), on the same device.
   - If they do not, STOP.

## Code changes (repo `FYP_AUR_SUR`, branch `phase4-enhance`)

**Harness generalisation.** Phase 3 behaviour must stay byte-identical.
- **`scripts/sr_pipeline.py`:**
  - `RES` comes from `cfg["paths"]["results"]` (default `outputs/sr/results`).
  - `make_recs` takes its batch size from `cfg["batch_size"]` (default 256; `enhance.yaml` sets 128).
  - `vid()` gets a generic fallback: `arm` plus the sorted remaining keys, for example `gamma__g0.7` or `clahe__c2_t4`. The existing arms keep their strings.
  - `restore()` gets a branch for `generic: true` arms: `--input-dir`, `--out-dir rel`, `--params '<variant json>'`, `--device`.
  - The variant's `interp` is passed to scoring.
- **`src/sr/eval.py`:**
  - `get_embeddings(..., interp="area")`: restored rows use `interp`, and direct rows stay bilinear.
  - `evaluate_variant` passes it through.
- **`src/sr/common.py`:** `Timing` calls `torch.cuda.synchronize` and reads peak VRAM only when CUDA is used. A pure-time path serves the OpenCV runners.

**New runners.** They follow the `src/sr/run_bicubic.py` pattern:
- resume by skipping existing outputs;
- atomic `write_png`;
- `timing.json` with `n_ok`, `n_failed` and `device`;
- every failure caught and logged.

The runners:
- `src/sr/run_photometric.py` (score venv) handles `noop`, `gamma`, `clahe` and `unsharp`.
- `src/sr/run_zerodce.py` uses vendored `src/sr/vendor/zerodce/`: the model file plus licence at a pinned commit. Weights SHA-256 go in `MODEL_INVENTORY.md`.
- `src/sr/run_nafnet.py` uses vendored `src/sr/vendor/nafnet/NAFNet_arch.py` (pinned commit and licence), runs batched on GPU, and catches out-of-memory errors. On OOM it halves the batch and retries; each retry is logged.

**Config.** `configs/enhance.yaml` is a copy of `configs/sr.yaml`, plus:
- `paths.results: outputs/enhance/results`;
- `batch_size: 128`;
- the arms above (`python`, `runner`, `generic: true`, `stochastic: false`, `interp`, `device`);
- `tuning_variants`;
- `gates: {lowlight: {measure: luminance, quantiles: [0.1, 0.2, 0.3, 0.5]}, deblur: {measure: lap_var, quantiles: [0.1, 0.2, 0.3, 0.5]}}`.

Restored outputs and embedding caches stay under `outputs/sr/restored|emb/<part>/<arm>__<setting>`, with new arm names. Direct-feed embeddings are recomputed on the RTX 3000 under `outputs/sr/emb/<part>/direct/` (the lab copies are unpacked under `outputs/enhance/xcheck/`).

**New and extended scripts:**
- **`scripts/enh_xcheck.py`** (new): the cross-device check above.
- **`scripts/enh_gate.py`** (new):
  - selects τ on tuning;
  - writes gated embedding caches, mixing enhanced and direct rows per crop;
  - scores them via `E.evaluate_variant` into `tuning_gated.csv` and `eval_gated.csv`.
- **`scripts/sr_extra_analysis.py`:**
  - add `--config`; the arms and gated variants come from the config's selection files;
  - add a `dark` slice (`lum_bin == "dark"`, 20%) and its complement, next to `dark10` and `low_sharpness`.
- **`scripts/enh_lfw_check.py`** (new). It reuses `lfw_pairs`, `unique_images`, `sims`, `auc`, `tar_at_far` and the bootstrap from `scripts/sr_lfw_check.py`.
  - Degradations of `hr112`:
    - dark: I·s, then Poisson–Gaussian noise (seeded);
    - Gaussian blur σ;
    - 11 px linear-motion blur at a seeded angle.
  - Calibration (direct only): the strength whose mean non-leaky cross-degraded AUC is closest to 0.80, within 0.70–0.90. It is written to `lfw_enh_calibration.csv` before any arm runs.
  - The low-light arms run on the dark set and the deblur arms on the blur sets, all with bilinear interpolation.
- **`scripts/sr_report_tables.py`:** add `--config`, and inject tables into `REPORT_phase4.md` between the `TABLES`, `EXTRA` and `LFW` markers.
- **`scripts/sr_failure_cases.py`, `scripts/sr_sample_figs.py`, `scripts/sr_brief_strips.py`:** add `--config` / `--arms`.
  - Samples are seeded (42).
  - Dark examples come from `dark10` and blur examples from the low `sharp_bin_sz` bin.
  - Labels show dataset, method and the problem seen, as in the supervisor brief appendix.

**Run commands** (each step resumable; rerunning a step continues where it stopped):
```powershell
$py = "envs\score\Scripts\python.exe"
& $py scripts\enh_xcheck.py                                                   # must pass
& $py scripts\sr_pipeline.py direct --config configs\enhance.yaml             # direct feed on this GPU
& $py scripts\sr_pipeline.py arm --config configs\enhance.yaml --arms noop    # sanity: must equal direct
& $py scripts\sr_pipeline.py arm --config configs\enhance.yaml --arms bicubic gamma clahe unsharp zerodce nafnet
& $py scripts\enh_gate.py --config configs\enhance.yaml
& $py scripts\sr_extra_analysis.py --config configs\enhance.yaml
& $py scripts\enh_lfw_check.py all
& $py scripts\sr_report_tables.py --config configs\enhance.yaml
```
Long steps can run in a scheduled task with the Phase 3 settings: restart on failure and a logon trigger. The setup is the same as `SR_worker`.

## Schedule (flexible within W1–W2)

| When | Work |
|---|---|
| Before W1 | Abser: `phase4_bundle.py` to Drive; pushes the code skeleton, or Raghib starts the branch |
| W1 day 1 | Setup, venvs, unpack, **cross-device check** |
| W1 days 2–3 | Harness changes; `noop` sanity; runners; vendoring with SHA-256; smoke run on 75 crops with montage; s/crop on CPU vs GPU |
| W1 days 4–5 | Tuning sweep, all variants; `selection_<arm>.json`; gate τ selection; write the pre-registered rules into the report draft |
| W2 days 1–2 | Eval subset (22,898 crops): restore the selected variants; score; gated eval |
| W2 day 3 | Probes-only and per-bin; LFW calibration and check |
| W2 day 4 | Visual samples, failure sheets, tables |
| W2 day 5 | `REPORT_phase4.md` with the stop-rule verdict; upload; **STOP** for supervisor approval (end-of-W2 checkpoint) |

**Runtimes.** No runtimes are measured on the RTX 3000 yet; the day-2 smoke run measures them. Until then, the rough expectation is:
- the OpenCV arms take seconds per subset;
- Zero-DCE++ takes minutes on GPU;
- NAFNet takes minutes (native) to under an hour (pre-128) on GPU;
- each scoring pass takes a few minutes per variant.

## Sync back (end of each working session, and at the end)
- `scripts/phase4_upload.py` (new) zips `outputs/enhance/results/`, `REPORT_phase4.md`, the thumbnails of the selected variants (256 px JPEGs, Phase 3 style) and the logs into `G:\My Drive\FYP_SR_SHARED\phase4\results_<date>.zip`, with a SHA-256 manifest.
- Code changes: commit and push to `phase4-enhance`, then open a PR to `main` for Abser to review.
- Full-size restored PNGs stay on the laptop and are not uploaded.

## Deliverable: `REPORT_phase4.md` (REPORT_phase3 structure)
1. Summary and stop-rule verdict.
2. What was run, including device, cross-device check and runtimes.
3. Findings per arm.
4. Gated variants.
5. Probes-only and per-bin (`dark`, `dark10`, `low_sharpness`).
6. LFW synthetic check (dark, Gaussian blur, motion blur).
7. Labelled visual samples.
8. Cost: s/crop on CPU and GPU.
9. Full tables (generated).
10. Limitations:
    - `lap_var` gate is size-confounded;
    - dark-bin size;
    - synthetic LFW degradations;
    - multiplicity;
    - a different GPU from Phase 3.
11. Decision for Phase 8: the gate arm with its τ, or "restoration closed".

## Verification
- **Cross-device:** lab-PC direct embeddings vs RTX 3000, cosine > 0.9999 per crop (`xcheck.json`).
- **Control:**
  - `noop` embeddings equal direct (`np.array_equal`) for all 4 recognizers on both parts;
  - `noop` paired Rank-1 delta is exactly 0.
- **Phase 3 unchanged:**
  - on the lab PC, rerun `sr_report_tables.py` with `configs/sr.yaml`;
  - the output must be byte-identical to the current `report_tables.md`;
  - Abser checks this when reviewing the PR.
- **Resume:** interrupt a runner halfway, rerun it, and check the outputs equal an uninterrupted run (hashes of 100 files).
- **Gating sanity:**
  - τ = 0 equals direct;
  - τ = ∞ equals the full arm (embedding equality).
- **LFW:** the calibration file is written before arm scoring (timestamps). The direct ceiling condition gives AUC ≈ 0.999.
- **Numbers:** every number in `REPORT_phase4.md` comes from `outputs/enhance/results/*.csv` or the injected tables.

## After approval (Claude, on the lab PC)
- Save this runbook as `research/Sprint_Notes/Phase4_restoration_runbook.md` and copy it to `G:\My Drive\FYP_SR_SHARED\phase4\`.
- Save a memory that every plan, local or distributed, is also saved as a markdown file in `research/Sprint_Notes/`.
