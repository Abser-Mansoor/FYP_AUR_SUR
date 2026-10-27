# Implementation plan: Phase 5 continuation (Parts A–E)

**Status:** draft, 2026-10-27, written by Claude on the lab PC. It is waiting for the go-ahead from Umar.
**Implements:** `research/Sprint_Notes/Phase_5_continuation.md` (Abser, 2026-10-27). That file is the authority. This one only adds what was found on the PC and how each step will be done.

## What was checked before planning (read-only)

| Item | State on the lab PC | Effect on the plan |
|---|---|---|
| `main` | Up to date with `origin/main`; includes `9ceed1d` (Phase 4) and `5fc60b0` (this plan + CONTEXT) | A1 is fine |
| Branch `phase_5` | **Already existed locally**, pointing at the same commit as `main` (5fc60b0), with no extra commits | Switched to it (same result as `git checkout -b`). Not a STOP |
| Git identity | `user.name = Abser-Mansoor`, `user.email = raghibrizwan007@gmail.com` (name and email belong to different members) | A2: Umar decides; commits use `git -c` only |
| `envs/video` | torch 2.14.0+cu126, onnxruntime 1.30.0, imports OK | A3 passes |
| `outputs/video/model_downloads.json` | **Missing**; `models/caface/` and `models/facemoe/` do not exist | B1 must run `scripts/phase5_fetch_models.py` (network, about 1–2 GB, size not measured). `huggingface_hub` may need installing in `envs/video` first |
| `verify_scrfd.json`, `verify_tiling.json`, `tracker_smoke.json` | All present (from 2026-10-22) | B2: read and report, no GPU run needed |
| Drive `phase5/results/` | `DONE`, `chunks/`, `device.json`, `runtime.json`, `xcheck.json`; chunk counts 6×17 wider, 6×5 chokepoint, 2×115 crops | Matches the C2 table before copying |
| Local `outputs/video/detstudy/` | Holds `_smoke/` and a lab-PC `device.json` from the 2026-10-22 smoke test | robocopy overwrites `device.json` with the T4 one. `_smoke/` is not read by the merge and stays ignored by git |
| Ground truth for the merge | `outputs/video/chokepoint_sample/gt_boxes.csv`, `outputs/video/gt/*.json` present | C3 can run |
| `gh` CLI | **Not installed** | D6: push only; Umar sends Abser the branch name (or opens the PR in the browser) |
| Phase 3 embedding caches (Part E) | `outputs/sr/emb/eval/{direct,bicubic,esrgan,invsr,osdface,tadsr}` present | Variant folders are checked in E before any analysis |

## Part A: setup (about 10 min)
1. **A1 (done):** `git pull` reported "Already up to date"; on branch `phase_5`.
2. **A2 [UMAR]:** show him the identity above. If he wants his name on the commits, he gives the name and email, and every commit uses `git -c user.name=... -c user.email=...`. No config file is changed.
3. **A3 (done):** environment check passed.

## Part B: close 5b (about 15–40 min; the download dominates)
- **B1. Model hashes.**
  1. `envs\video\Scripts\python.exe -c "import huggingface_hub"`; if it fails, install `huggingface_hub` into `envs/video` with uv (the only package change; it is recorded in `MODEL_INVENTORY.md`).
  2. Run `scripts\phase5_fetch_models.py` in the background. It is resumable and downloads only.
  3. If gdown hits the Google Drive quota for a CAFace file, the json will say "DOWNLOAD FAILED". That is logged as a result, and its row in `MODEL_INVENTORY.md` says so. It is **not** a STOP, because nothing is run with these weights in Phase 5. A hash mismatch on a file that did download **is** a STOP.
  4. Recompute SHA-256 with `hashlib`, compare with the json, then write the hashes into the CAFace and FaceMoE rows of `MODEL_INVENTORY.md` §4.
- **B2. Verification jsons.** Read the three files and report pass/fail in a small table. Nothing is rerun, because all three exist.

## Part C: merge the T4 study (about 30 min)
1. **C1.** `robocopy "G:\My Drive\FYP_SR_SHARED\phase5\results" outputs\video\detstudy /E`. Exit code 0–7 means success.
2. **C2 [UMAR]:** count the `.npz` files per folder locally (expected 102 / 30 / 230) and show him the table.
3. **C3.** `envs\video\Scripts\python.exe scripts\phase5_merge.py`. CPU only: it reads chunks and ground truth, and runs no inference.
4. **C4.** Record the SHA-256 of every file in `results/` in `outputs/video/detstudy/merge_hashes_run1.txt`, run the merge again, compare with run 2. Any difference is a STOP.
5. **C5.** Copy `device.json` and `xcheck.json` into `results/`.
6. **C6 gates**, checked by a short read-only script so the numbers come straight from the CSVs:
   - SCRFD `full640` AP is within 1.0 point of 95.16 / 93.87 / 83.05;
   - recall per bin rises broadly with face height. Defined as: the >64 px bin is higher than the <10 px bin, and there is at most one decrease between neighbouring bins. The definition is written in the report;
   - per-item errors are at most 1% per configuration.
7. **C7 [UMAR]:** show `selection.json`. Point out that RetinaFace `full1280` (107.7 ms) and `tiled` (124.4 ms) are over the 100 ms cap, which `runtime.json` on Drive confirms.

## Part D: protocol, gitignore, report, PR (about 2–3 h)
- **D1.** Read the Zenodo README (already downloaded: `Dataset/ChokePoint/archives/README.txt`) and fetch `arma.sourceforge.net/chokepoint`.
  - The local README has **no protocol section**. It covers environment, recording time, camera and sequence naming only.
  - If the web page is unreachable or has no G1/G2 definition, write "protocol not found" and make it an open decision.
  - Update the ChokePoint row: status and download date from the log or file timestamps, plus the licence note.
- **D2 [UMAR]:** he opens 5 overlays. His answer goes into §4 of the report word for word.
- **D3.** Add the `.gitignore` block after the `outputs/enhance` lines. Check with `git status --ignored` and `git add -n .` that only the listed files are new: no `.npz`, frames, overlays, `chokepoint_eyes.csv`, sample folder or bundle.
- **D4. `REPORT_phase5.md`.**
  - The 10 sections from the Phase 5 plan, in the style of `REPORT_phase4.md`.
  - The tables are pasted from `detector_tables.md` between `<!-- TABLES:BEGIN/END -->` markers.
  - The selection, AP gate, false positives per frame for the chosen configuration, and error counts come from the CSVs and json files.
  - §9 and §10 are exactly as listed in the continuation plan.
  - The report ends with **STOP: awaiting supervisor approval (together with Phase 4).**
- **D5. `CONTEXT.md`.** Hand-edit §3 and add a dated §11 entry. The STATUS block is left alone. Then copy the file to `G:\My Drive\FYP_SR_SHARED\CONTEXT.md`; that is the only Drive write.
- **D6.** Four commits as named in the plan, using Umar's identity if he chooses it. Then `git push -u origin phase_5`. `gh` is missing, so there is no PR from the PC. Umar sends Abser the branch name, or opens the PR in the browser from the link git prints. Nobody merges it.
- **D7 [UMAR]:** a 5-line chat message drafted for Umar.

## Part E: SR asymmetry study (Wed 29 or Thu 30 Oct; starts only after the Phase 5 push)
- **Step 0.**
  1. Create branch `extra/sr_asymmetry` from `main`.
  2. Create `REPORT_extra_sr_asymmetry.md` with the hypotheses, tests and decision criteria copied verbatim.
  3. Commit it, and have Umar confirm the commit before any number is computed.
- **Variant check.** List `outputs/sr/emb/eval/<arm>/<setting>/seed0/eval__<rec>.npy` for the 6 variants × 3 non-leaky recognizers, plus `direct/resize/seed0`. Check row counts against `eval_images.csv` (22,898). Missing variants are recorded and skipped; no restorer is run.
- **Script `scripts/extra_sr_asymmetry.py`** (CPU, seed 42, `--config configs/sr.yaml`):
  - It reuses the loading and probes-only mixing of `scripts/sr_extra_analysis.py`, the protocols in `src/eval_protocols.py` (`tinyface_identification`, `survface_openset`, `survface_verification`) and `src/metrics.py` (`bootstrap`, `paired_delta`, `separation`, `impostor_shift`).
  - **E0:** SurvFace closed-set Rank-1 reuses `tinyface_identification` with the SurvFace gallery and an empty distractor set. TinyFace verification draws 1,000 genuine and 1,000 impostor probe–gallery pairs (seed 42), saved to `asymmetry/tinyface_pairs.csv` before scoring.
  - **H2–H6** as specified. H3 mixes embeddings per split: distractors at direct feed. H5 parses the camera from the SurvFace file name, and the kNN camera accuracy uses NumPy cosine with leave-one-out. H6 recomputes TPIR with the direct-feed threshold. **H7** is written as deferred and not run.
  - Bonferroni-adjusted CIs are shown next to the plain ones.
  - The script runs twice, and the output hashes must be identical.
- **Report** `REPORT_extra_sr_asymmetry.md` with the 6 sections. Then push the branch. There is no PR from the PC (no `gh`), and it STOPs for supervisor review.

## Risks noticed in advance
- **CAFace Google Drive quota:** handled in B1 as a logged result, not a STOP.
- **Drive "online-only" files:** robocopy forces the download. If it fails (exit ≥ 8), check Drive for Desktop and rerun C1, as the continuation plan says.
- **Identity mismatch** in the repo git config (name from one member, email from another): Umar's choice in A2. It is mentioned to him; the config is not changed.
- **ChokePoint protocol** may not be documented anywhere we can reach: then it is an open decision for Phase 6, not a split of our own.
