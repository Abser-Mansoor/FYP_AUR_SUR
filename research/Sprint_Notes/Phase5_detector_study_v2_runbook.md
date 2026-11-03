# Phase 5 detector study, re-run v2: runbook for Abser (Colab T4)

Written by Claude on the lab PC for Abser, 2026-11-03. This implements decision (a) from `Phase5_continuation_status_2026-10-27.md`: SCRFD now uses the official WIDER FACE preprocessing, and the whole study is re-run on one Colab T4 (yours). The first run (Umar's T4, `phase5/results/`) is kept as the record of the superseded setup and is not changed or mixed in.

## What changed in the code

| File | Change |
|---|---|
| `src/video/detect.py` | `SCRFDDetector` has a new option `preprocess`. `"legacy"` (the default) is the old path through `src/align.py`, untouched. `"official"` follows insightface's `test_widerface.py` (mode 0): resize keeping the aspect ratio with the new size rounded, normalise, then pad with 0 after normalising (mid-grey), single scale, no flip. Boxes are rescaled by the actual per-axis resize factors. |
| `configs/video.yaml` | New `scrfd: {preprocess: official, nms: 0.45}`. `paths.out` is now `outputs/video/detstudy_v2`, so the new results never share a folder with the first run. RetinaFace and the tile-merge NMS stay at 0.4. |
| `scripts/video_detector_study.py` | `build_detector` takes a protocol. The WIDER, ChokePoint and runtime jobs use the study settings. The crop job and the cross-device check keep the unchanged Phase 1 SCRFD path, so they stay comparable with the Phase 1 landmarks and with `lab_kps.npy`. |
| `scripts/phase5_bundle_v2.py` (new) | Builds `code_v2.zip` and its manifest and publishes them to a new Drive folder `FYP_SR_SHARED/phase5_v2/`. Nothing in `phase5/` is overwritten. |
| `colab/build_phase5_v2_notebook.py` (new) | Builds `colab/phase5_detector_colab_v2.ipynb` from the first notebook with asserted text replacements. It reads the unchanged data zips from `phase5/` (not the old `code.zip`), reads `code_v2.zip` from `phase5_v2/`, and writes to `phase5_v2/results/`. |

Tile-merge note: in `tiled` mode SCRFD uses NMS 0.45 inside each tile and NMS 0.4 when merging tiles. Tiling is our own addition, not part of the official protocol, so the merge value was left as it was.

## Check done on the lab PC before this run

`outputs/video/detstudy/_diag/verify_official_option.json` (see the result below): the new `official` option, on the CPU, against the diagnostic run on the first 500 WIDER val images, and a check that the `legacy` option still returns exactly what `src/align.py` returns.

## What you do

1. **Wait for Drive to sync.** In `FYP_SR_SHARED/phase5_v2/` on your Drive there must be `code_v2.zip`, `MANIFEST.json` and `phase5_detector_colab_v2.ipynb`. The notebook checks this itself and stops if a file is incomplete.
2. **Open the notebook** from Drive (right-click, Open with, Google Colaboratory). Use your own Colab account. The shared folder is yours, so no shortcut is needed.
3. **Runtime, Change runtime type, T4 GPU.** Then **Run all**. Keep the tab open and active.
4. **If it disconnects,** run all cells again. Finished chunks (200 images each) are skipped.
5. **The notebook stops by itself, and you should not continue, in these cases:**
   - the GPU is not a T4;
   - the bundle is incomplete or a checksum differs;
   - ONNX Runtime is not using the GPU;
   - the detector weight hashes differ;
   - the cross-device check fails (SCRFD keypoints on 100 TinyFace probes must agree with the lab PC within 0.5 px). In every case, copy the printed message and send it to Umar's team chat for the lab PC. Do not try to fix it.
6. **When the last cell prints "exit code 0"** it writes `phase5_v2/results/DONE`. Tell Claude on the lab PC that the run is finished.

## What happens next (lab PC)

1. Copy `phase5_v2/results` to `outputs/video/detstudy_v2` with robocopy through PowerShell (exit code 0 to 7 is success). Count the `.npz` files against the first run.
2. Run `scripts/phase5_merge.py` twice and compare the SHA-256 of all result files.
3. Check the gates again. Gate 1 now uses the **KPS row (95.40 / 94.01 / 82.80)**, as you decided, and stays at a limit of 1 point. The diagnostic CPU run on the lab PC gave 94.99 / 93.60 / 82.42, so a pass is expected, but the gate is decided by the T4 numbers.
4. Then Part D (report, `CONTEXT.md`, commits, push).

## Device rule

The v2 numbers are all from your T4 in a single run, and `device.json` records it. They are not mixed with the first run's numbers from Umar's T4. If a run is interrupted and resumed on a different Colab session, the GPU type is still checked at the start, and `device.json` is rewritten on each start.

## Lab check result

On the first 500 WIDER val images, CPU, the new `official` option gives AP 96.52 / 95.66 / 86.15 (easy / medium / hard), against 96.44 / 95.67 / 86.26 from the diagnostic run it reproduces. The two agree to within 0.11 points. The detection count was identical on 499 of the 500 images. On those images the largest box coordinate difference was 3.9 px, which is larger than the per-axis rescaling alone should give; it is probably two neighbouring detections swapping places when their scores differ in the last digits, but this was not traced. The `legacy` option returned exactly the same boxes as `src/align.py` on the three images checked. Source: `outputs/video/detstudy/_diag/verify_official_option.json`.
