# STOP: Phase 5 continuation, step C6 gate 1 (WIDER AP) failed

Date: 2026-10-27, lab PC. Written for Abser (post this in the team chat). Nothing was changed to "fix" it. The report was not written.

## What failed
Gate C6.1: SCRFD `full640` easy/medium/hard AP must be within 1 point of the published SCRFD-10GF values 95.16 / 93.87 / 83.05.

| Set | Ours (`results/wider_ap.csv`) | Published | Difference |
|---|---|---|---|
| easy | 94.85 | 95.16 | -0.31 |
| medium | 93.46 | 93.87 | -0.41 |
| **hard** | **81.28** | 83.05 | **-1.77** (limit 1.0) |

Exact values: easy 0.9484639341384087, medium 0.9346212597882659, hard 0.8127933049179324 (`scrfd__full640`, 3,226 images, 0 errors).

## Command and state
- Command: `envs\video\Scripts\python.exe scripts\phase5_merge.py` (exit 0, no exception). Run twice.
- Inputs: `outputs/video/detstudy/chunks/` copied from `G:\My Drive\FYP_SR_SHARED\phase5\results` (robocopy exit code 3 = success). File counts matched the plan: 102 wider, 30 chokepoint, 230 crops.
- Outputs: `outputs/video/detstudy/results/` (8 files plus `device.json`, `xcheck.json`).
- Branch `phase_5` (same commit as `main`, 5fc60b0). Nothing committed.

## Other checks, all computed from the result files
- **C4 determinism: passed.** The 8 result files have identical SHA-256 hashes across two merges (`merge_hashes_run1.txt`, `merge_hashes_run2.txt`).
- **C5:** `device.json` and `xcheck.json` copied into `results/`.
- **T4 cross-device check (`xcheck.json`): passed.** Max difference 0.003 px, 0 detection mismatches, 100 crops.
- **Gate 2 (recall rises with face size): passed.** 0 decreases between neighbouring bins in all 6 configurations.
- **Gate 3 (errors): passed.** `n_errors` = 0 for every configuration; `problems.txt` is empty.
- B2 verification files (from 22 Oct) all pass.

## Observations that may help the check (not conclusions)
- The easy and medium AP are close to the published values. Only hard is off, and the gap is in the direction of fewer small faces found at 640 px.
- A 300-image smoke run on 22 Oct on the lab PC gave 82.03 for hard (subset, indicative only).
- Hard AP depends on the operating point (input size, score threshold, NMS threshold, test-time scale). The plan notes that a larger gap means the evaluation port or the operating point differs.
- **Unrelated to the gate:** the ChokePoint sample has no faces below 32 px, so the `<10` to `24-32` bins are empty (NaN) in `chokepoint_bins.csv`. The ChokePoint recall tables cannot discriminate small-face performance.

## Question for Abser
Is the 1.77-point gap on hard acceptable as a known difference of operating point (and recorded as a limitation), or must the evaluation be checked first? Until then the study is not used and `REPORT_phase5.md` is not written. The selection result stays unused: `selection.json` currently names `scrfd__full1280` (recall 0.761 on 16-32 px, 88.4 ms median on the T4).

## Waiting
Everything in Parts D and E is on hold. The CAFace/FaceMoE weight download (B1) runs independently of this and continues.
