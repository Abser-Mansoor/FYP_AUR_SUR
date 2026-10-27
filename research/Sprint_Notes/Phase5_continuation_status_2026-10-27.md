# Phase 5 continuation: status and decisions needed from Abser (2026-10-27)

Written by Claude on the lab PC for Abser, with Umar. Branch `phase_5`, nothing committed yet. Every number below comes from a file in `outputs/video/detstudy/` (named next to the number).

## 1. Where we are

Parts A, B and C of `Phase_5_continuation.md` are done. Part C stopped at sanity gate 1 (WIDER AP), as the plan requires, so Part D (report, CONTEXT.md, commits, push) and Part E (SR asymmetry study) have not started. Nothing is committed. `MODEL_INVENTORY.md` is modified and `research/Sprint_Notes/Phase5_continuation_implementation_plan.md` is new.

| Part | State |
|---|---|
| A. Setup | Done. Branch `phase_5` already existed at the same commit as `main` (5fc60b0). The repo git identity pairs one member's name with another's email, so commits will use Umar's identity through `git -c`, without changing config. |
| B1. CAFace and FaceMoE hashes | Done. All six downloaded files match `model_downloads.json`. The SHA-256 values and sizes are written into section 4 of `MODEL_INVENTORY.md`. |
| B2. Verification files from 22 Oct | Read and all pass (SCRFD, tiling, tracker smoke). |
| C1 to C5. Merge of the T4 study | Done. Chunk counts are 102 (WIDER), 30 (ChokePoint) and 230 (crops), as expected. The merge exits 0. Two merges give byte-identical result files. The T4 cross-device check passes (100 crops, 90 detected, 0 mismatches, max difference 0.0031 px). |
| C6 gate 2 (recall rises with face size) | Pass for all six configurations. |
| C6 gate 3 (errors) | Pass. Zero per-item errors in every configuration. |
| C6 gate 1 (SCRFD full640 AP within 1 point of published) | **Fail, STOP.** See section 2. |

Side finding, unrelated to the gate: the ChokePoint sample has no faces below 32 px, so its small-face recall bins are empty (NaN in `chokepoint_bins.csv`). ChokePoint cannot separate the detectors on small faces.

## 2. The gate failure

SCRFD `full640` AP from `results/wider_ap.csv`: easy 0.9485, medium 0.9346, hard 0.8128 (3,226 images, 0 errors). The plan compared these with 95.16 / 93.87 / 83.05, giving a hard gap of -1.77 points against a limit of 1.0.

## 3. What we checked after the STOP

Nothing in `src/`, configs, chunks or vendored code was changed. All diagnostics are in `outputs/video/detstudy/_diag/` and are not reported numbers. They were run on the lab PC CPU only, to respect the rule against re-running the study on the lab GPU.

**3.1 The reference row is wrong.** `det_10g.onnx` has keypoints (the code asserts nine outputs), so it is SCRFD_10G_KPS. In the official README that row is 95.40 / 94.01 / 82.80. The row we used (95.16 / 93.87 / 83.05) is the model without keypoints. Against the KPS row the gaps are -0.55, -0.55 and -1.52. Hard is still over the 1.0 limit, but the gap is smaller than first reported.

**3.2 Our settings versus the official insightface test script** (`detection/scrfd/tools/test_widerface.py`, mode 0, and `configs/scrfd/scrfd_10g_bnkps.py`):

| Item | Ours (`src/align.py`, `src/video/detect.py`, `configs/video.yaml`) | Official |
|---|---|---|
| Score floor | 0.02, all detections kept for AP | 0.02 |
| NMS IoU | 0.4 | 0.45 |
| Resize | keep ratio to 640, new size truncated with `int()` | keep ratio to 640, size rounded |
| Padding | black (0) bottom and right, applied before normalisation | normalise first, then pad with 0, which is mid-grey |
| Scale, flip | single scale, no flip | single scale, no flip |
| Normalisation | (x - 127.5) / 128, RGB | same |

**3.3 Score floor test.** Stored T4 detections re-scored with a higher floor (`thr_sensitivity.json`): hard AP is 0.8128 at 0.02, 0.8121 at 0.05, 0.8084 at 0.1 and 0.7535 at 0.3. A fresh CPU run over all 3,226 images with our settings and a floor of 0.001 (`lowthr_study.json`) gives 0.9485 / 0.9346 / 0.8128, identical to the 0.02 result and identical to the T4 result. So the floor is not the cause, and the CPU reproduces the T4 numbers exactly.

**3.4 Official-settings test on the first 500 images only** (`lowthr_official_lim500.json`, `lowthr_study_lim500.json`):

| Variant | Easy | Medium | Hard |
|---|---|---|---|
| Our settings | 0.9623 | 0.9542 | 0.8432 |
| Official settings (grey pad, NMS 0.45, rounded resize) | 0.9644 | 0.9567 | 0.8626 |

The official settings raise hard AP by 1.95 points and easy and medium by about 0.2 points. This is the same pattern as the original gap, which is mostly on hard. The absolute values on 500 images are not comparable to the full-set values; only the difference between the two rows means anything. The run changed three things at once, so it does not say which of them matters.

## 4. Conclusion so far

The score floor is not the cause. The evaluator reproduces itself and gives the same numbers on the T4 and the CPU, but it has not been compared against the official WIDER evaluation tool, so it is not cleared. The most likely cause is the difference in preprocessing and NMS from the official test script, which is a difference in the detector wrapper rather than in the evaluator. This is an inference from a 500-image subset, not a confirmed result.

## 5. Decisions needed from Abser

1. **Which reference row is correct for the gate?** Our recommendation: the KPS row (95.40 / 94.01 / 82.80), because the weights have keypoints. Please confirm, so the plan's gate text can be corrected.
2. **How to resolve the hard-AP gap.** The options are:
   - **(a)** Adopt the official preprocessing (grey padding after normalisation, NMS 0.45, rounded resize) in the SCRFD wrapper. This means a change to `src/align.py` or `src/video/detect.py`, a new bundle for Umar, and a full re-run of the study on the Colab T4. The gate would then be rechecked against the KPS row. The earlier phases that use the same `SCRFD` class would also need an impact check (section 6).
   - **(b)** Keep the current wrapper, record the gap as a known deviation from the official operating point, and say so in the limitations. The selection result (currently `scrfd__full1280`) would stand but would carry this caveat.
   - **(c)** Before choosing, run the official-settings variant on all 3,226 images on the CPU (about 1.7 hours) so the gap is measured on the full set, and compare the evaluator against the official WIDER evaluation tool.

   Our recommendation is (c) first, then (a) if the full-set gap closes to within the limit. We have not changed anything.
3. **Is a CPU diagnostic run on the lab PC acceptable under the "one device per reported number" rule?** We treated it as acceptable because none of its numbers is reported. Please confirm, or tell us to discard `_diag/`.

## 6. Open points

- The `SCRFD` class in `src/align.py` is shared with earlier phases. If the preprocessing is changed in that file, earlier landmark and alignment results could change. The safer route is to add the official preprocessing as an option in `src/video/detect.py` only. This needs Abser's decision.
- `selection.json` names `scrfd__full1280` (recall 0.7611 on 16 to 32 px, 88.4 ms median on the T4), but we are not using it until the gate is resolved. A change of preprocessing could change this choice.
- Still on hold until Abser rules: ChokePoint protocol (D1), Umar's overlay spot check (D2), `.gitignore` block (D3), `REPORT_phase5.md` (D4), `CONTEXT.md` (D5), the four commits and the push (D6), and Part E.
- The dataset access requests in `docs/requests/` have not been sent by anyone yet.

## 7. Files

- STOP note: `outputs/video/detstudy/STOP_note_C6_gate1.md`
- Merge outputs: `outputs/video/detstudy/results/`
- Merge hash files: `outputs/video/detstudy/merge_hashes_run1.txt` and `merge_hashes_run2.txt`
- Diagnostics: `outputs/video/detstudy/_diag/` (`thr_sensitivity.py`, `lowthr_cpu.py`, their json and log outputs)
- Plan: `research/Sprint_Notes/Phase5_continuation_implementation_plan.md`
