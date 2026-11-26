# Phase 9 plan: final pipeline, error investigation, quality bar, FPS, retrieval and demo

**Status:** approved 2026-11-26 (Umar, with the decisions Abser gave the same day).

**Window:** W6–W7 (from 2026-11-26).
**Executor:** Claude Code on the lab PC.
- Umar commits and pushes himself; Claude gives the commands, as in Phase 8.
- Umar runs the T4 timings on Colab.
- A second team member labels the audit sheet.

## Context
- Phase 8 (branch `phase_8`) answered the open-set, quality, tracker and gate questions on MEVID.
- Abser decided the Phase 9 settings on 2026-11-26:
  - D1: ByteTrack with the face term.
  - D3: R4* primary, R2 as a row.
  - D4: WebFace4M.
  - D5: same detector mode.
  - D6: FPIR 0.1.
  - D7: deferred.
- The quality bar (D2) needs one more experiment. On MEVID, MS1MV2 with R1 has a closed-set error of 50.5%. Rejecting the smallest quarter still leaves about 40%.
- Abser asked for three things on D2:
  - an absolute pixel threshold;
  - an expected coverage, with an action fixed in advance in case coverage is off;
  - the full error-vs-coverage curve.
- Abser also asked for **an investigation of the high error and a reduction of it**.
- The Sprint 2 Phase 9 items still apply:
  - end-to-end runs with the final configuration;
  - FPS by stage (lab PC and T4);
  - Flat vs HNSW retrieval;
  - `scripts/video_demo.py`.
- **Demo footage:** MEVA clips (the source video of MEVID: CC BY 4.0, consented actors), with the MEVID check-in photos as the gallery. Campus clips stay blocked, and T2 stays deferred.
- **Untested combination:** Phase 8 measured R2 and R4* only with MS1MV2, and WebFace4M only with R1. Phase 9 measures them together before freezing the configuration.
- **Standing rules:**
  - inference only;
  - seed 42;
  - every number from a script;
  - pre-register before computing; failures are logged;
  - one GPU job at a time;
  - no MEVID or MEVA frame, chip or embedding in git (the repo is public).

## Part 0: decisions and pre-registration (`REPORT_phase9.md` §0, committed by Umar before any Phase 9 number)
| # | Setting | Value | Basis |
|---|---|---|---|
| D1 | Tracker | ByteTrack + face term: λ 0.5; gate: score ≥ 0.5 and native height ≥ 32 px | Phase 8 §14.1, decided |
| D2 | Quality bar | Tracklet mean face height **≥ 23 px**; below it, the identity is withheld ("low quality") | 25th percentile of the mean face height of the 3,188 Phase 8 tracklets = 22.7 px. This is the quarter cut, fixed in pixels |
| D2a | Expected coverage | MEVID about 75% (Phase 8: 22 px gives 0.769). **Demo floor Z = 60%.** If demo coverage is below 60%, report it and do not retune the threshold. The threshold also stays fixed above 60% | Abser |
| D2b | Error statement | The report states plainly that error at the bar is still high (MS1MV2 R1: about 40%; final configuration: measured) | Abser |
| D3 | Aggregation | R4* (min-max, λ 5) default; R2 and R1 as rows. **Fallback rule:** if R4* is below R2 on the development split with WebFace4M (paired CI excludes zero), R2 becomes the default | decided 2026-11-26 |
| D4 | Backbone | AdaFace R100 WebFace4M, standalone `models/caface/adaface_ir101_webface4m.ckpt` through `src/recognizer.py` (flip off, as in the Phase 8 rows). MagFace (MS1MV2) stays for the norm used by R4* and the quality signals | Phase 7 check 5: cosine ≥ 0.99999 with the CAFace backbone. The weights' licence terms (WebFace4M: research use) go into `MODEL_INVENTORY.md` |
| D5 | Detector | SCRFD det_10g, official preprocessing. ChokePoint `full1280`. MEVA mode per camera from a calibration clip (`scripts/video_mode_calibrate.py`) | carried forward |
| D6 | Open set | Threshold at FPIR 0.1 | Phase 8 §14.6 |
| D7 | YTF usage terms, T2, PaSC | deferred (ISP and VPN blocking) | Abser |

**MEVID split for Part B:**
- **Development:** the 104 train-split persons (2,297 Phase 8 tracklets).
- **Confirmation:** the 54 test-split persons (891 tracklets).
- **Gallery:** all enrolled persons (170) for both splits.
- The confirmation split is read **once**, after §0b freezes the configuration. The script refuses to read it without `--confirm`.
- Phase 8 already looked at all of MEVID, so D3 and D4 are not held-out choices. The report states this.

**Hypotheses (confirmation split, one look; cluster bootstrap over persons, 1,000 draws):**
- **H1:** Rank-1 of the final configuration minus Phase 8 MS1MV2 R1 Rank-1, paired per tracklet; the CI is above zero.
- **H2:** the same difference for TPIR1 at FPIR 0.1 (5 folds within the confirmation persons); the CI is above zero.
- **Secondary:**
  - error and coverage at 23 px with CIs;
  - the full error-vs-coverage curve (px thresholds 16–48, and rejected fraction 0–0.5);
  - the residual-quality test (B5).

## Part A: final configuration in the pipeline (code)
- **`configs/video.yaml`:** a new `final:` block holding D1–D6.
  - Every switch can be set back to the Phase 6/7 values for check 2.
- **`src/video/pipeline.py` / `scripts/run_video.py`:** recognizer chosen from the config.
  - Add an aggregate stage (reuse `src/video/aggregate.py`, R4*).
  - Add the quality-bar flag and the search stage (reuse `src/video/index.py`).
  - The face term uses the active backbone's embeddings, through the existing hook (`src/video/track.py:TrackerAdapter(lam, min_h)`).
- **Face term on WebFace4M:** Phase 8 validated it with MS1MV2 cosines.
  - Rerun `scripts/video_tracker_stress.py --lam 0.5` with WebFace4M embeddings and report IDF1 for S0–S5.
  - If IDF1 at S1–S3 drops below the λ 0 values, raise it as a decision for Abser. No retuning.
- **WebFace4M embeddings** for all ChokePoint detections (GPU, detached, resumable): reuse the `run_video.py --stage embed` path.

## Part B: error investigation and reduction (MEVID, mostly CPU from cache)
New `scripts/mevid_errors.py`. It reuses `scripts/mevid_openset.py` (`load_real`, `gallery_templates`, `closed_records`, `open_set_boot`) and `scripts/mevid_quality.py` (`curve`, `area`, `reject_table`). Inputs are `emb/*.npz` (kps, det score, face_h, lap_var, lum, mag_norm) and `cf_*.npz` (WebFace4M).
1. **B1 baseline on development:** WebFace4M single frame, R1, R2 and R4*. This applies the D3 fallback rule.
2. **B2 error decomposition (development only):**
   - error by face height bin;
   - error by a yaw proxy from the 5 keypoints, |nose x − eye-midpoint x| / eye distance;
   - error by Laplacian variance, detector score, camera, number of faces, and the person's gallery face height;
   - the rank of the true identity: the share of errors with the true identity in the top 5, i.e. near misses;
   - per-person concentration of errors;
   - the most frequent confusion pairs.
3. **B3 visual audit (Phase 6 study C pattern):**
   - Sample: 120 seeded wrong tracklets above 23 px plus 40 correct controls, shuffled blind.
   - Montage: 5 faces next to the predicted and true gallery photos. Montages stay local only.
   - Labels: two labelers tag non-face or back of head, profile, blur, occlusion, low light or IR, tiny, wrong person in the tracklet, gallery mismatch (photo quality, appearance change), and look-alike.
   - Report Cohen's kappa. Output: `outputs/video/mevid/audit/{sheet,labels_<who>,agreement}.csv`.
4. **B4 candidate fixes:** inference only, chosen to target the causes, at most 5, each measured on development.
   - **F1 frame filter before aggregation:** drop faces with yaw proxy > 0.35 or detector score < 0.6; keep at least 3.
   - **F2 resolution-matched gallery:** add embeddings of each gallery face downscaled to 24 and 32 px (then the standard resize to 112) to the template. Small GPU job.
   - **F3 flip test-time augmentation** for WebFace4M: re-embed about 90k faces, about 1 h GPU.
   - **F4 score fusion** of MS1MV2 and WebFace4M (mean cosine). Adopted only if the Part D FPS cost stays acceptable.
   - **F5 cohort score normalisation for the open set:** subtract the mean top-10 similarity to a cohort of TinyFace embeddings (shared with Part E).
   - **Adoption rule (in §0):** a fix is adopted if its paired development CI against B1's default excludes zero (Bonferroni over the fixes). The adopted fixes are combined, and the combination is checked once on development.
   - **§0b:** commit the frozen configuration, then run `--confirm`.
5. **B5 does height miss failures that the MagFace norm sees?**
   - Within the tracklets kept by the 23 px bar, run the Part C AUC test for MagFace norm, yaw proxy and Laplacian (development, then confirmation).
   - If the MagFace norm helps in this residual on both splits, a second gate is proposed for Abser. It is not adopted in Phase 9.
   - Together with the B3 shares (blur, pose, occlusion among errors above the bar), this is the evidence for Abser's concern.

## Part C: end-to-end on ChokePoint with the final configuration
54 ground-truth clips:
- IDF1, HOTA and ID switches;
- open-set track-identity accuracy at FPIR 0.1 (reuse `scripts/mevid_tracker_identity.py:identity`);
- comparison with the Phase 6/7 MS1MV2 rows.

## Part D: FPS at 720p by stage
- Stages: detect, align + WebFace4M, MagFace norm, track, aggregate, search.
- Settings: single stream, batch 1, with and without tiling.
- Devices: lab PC RTX 4000 Ada (Claude), and T4 on Colab (Umar, with a runbook).
- Reuses `scripts/video_timings.py`.

## Part E: retrieval table
- FAISS `IndexFlatIP` vs `IndexHNSWFlat` (`src/video/index.py`): recall@1 and latency.
- Gallery sizes 10², 10³ and 10⁴: the MEVID gallery padded with WebFace4M embeddings of TinyFace distractors (GPU, about 10k images).
- Probes: MEVID tracklet templates.

## Part F: demo on MEVA
- **Data:**
  - 5–10 MEVA clips from the public bucket, from cameras that appear in MEVID.
  - Selection is seeded and fixed in §0 before any run.
  - Print the sizes before downloading.
  - Add a `DATASETS.md` row (CC BY 4.0, citation).
  - Clips go to `Dataset/MEVA/`, gitignored.
- **`scripts/video_demo.py`:**
  - Input: a video file.
  - Output: an annotated MP4 (local only) with boxes, track IDs, identity or "unknown", the quality-bar state and FPS.
- **Gallery:** MEVID front check-in photos ≥ 48 px (170 persons).
- **Reported:**
  - coverage per clip against the D2a floor;
  - counts of identified, unknown and low-quality tracks;
  - runtime.
- **Qualitative check:** a team member compares 30 seeded identified tracks with the check-in photos. This is labelled qualitative: MEVA has no face labels.

## Acceptance checks (`outputs/video/phase9_checks.json`; a failure is a STOP)
1. WebFace4M through `src/recognizer.py` in `envs/video` matches the Phase 8 `cf_*.npz` features on 100 MEVID faces (cosine > 0.9999).
2. The pipeline with Phase 6/7 switches (MS1MV2, R1, λ 0, no bar) reproduces the Phase 6/7 ChokePoint tables byte for byte.
3. The pipeline's R4* equals `src/video/aggregate.py` on cached tracklets.
4. Unit test: the identity is withheld exactly when the mean face height is below 23 px.
5. FAISS Flat top-1 equals exact NumPy search on every probe.
6. The development and confirmation persons are disjoint. The confirmation results come from one `--confirm` run after the §0b commit (checked with `git log`).
7. CPU reruns are byte-identical; a demo rerun on one clip gives identical tracks.

## Deliverable
`REPORT_phase9.md` contains:
- §0 and §0b;
- the B1–B5 results, with the audit taxonomy;
- the H1 and H2 confirmation;
- the error-vs-coverage curve with the 23 px point and a plain statement of the remaining error;
- the ChokePoint end-to-end table, the FPS tables and the retrieval table;
- the demo description (no frames);
- limitations and decisions for Phase 10;
- a STOP.

Tables are injected by `scripts/phase9_report_tables.py` (the Phase 8 pattern). `CONTEXT.md` gets §3, §4 and §11 updates, with a new dated Drive copy. At the end, the git commands for Umar.

## Order of work
| Day | Work |
|---|---|
| 1 | Draft §0 (Umar commits); Part A code; checks 1–4; MEVA clip selection and download; ChokePoint WebFace4M embedding overnight |
| 2 | B1, B2 and B5 (development); audit sheet to the labelers; F2 and F3 GPU jobs; TinyFace distractor embedding |
| 3 | F1–F5 on development; §0b (Umar commits); `--confirm`; Part C; lab-PC FPS; T4 runbook to Umar |
| 4 | Part E; demo runs; audit agreement; checks 5–7 |
| 5 | Report, tables, `CONTEXT.md`, commands for Umar; STOP |

## Risks
- **Fixes give nothing:** report the negative result. The final configuration is then WebFace4M + R4* (or R2) + the bar.
- **Demo coverage below 60%:** report it, as fixed in D2a.
- **MEVA clips are large:** download selected clips only.
- **The face term on WebFace4M behaves differently:** raise a decision; do not tune.
- **Colab T4 unavailable:** report the lab-PC times only and state why.
