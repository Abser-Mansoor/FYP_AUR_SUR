# Phase 8 plan: held-out open set on MEVID, frame-quality reject test, final tracker choice

**Status:** approved 2026-11-24 (Abser). Executor: Claude Code on the lab PC with the team. Part 0 decisions and section 0 of `REPORT_phase8.md` are committed before any Phase 8 number.

## Context
- Phase 7 showed that per-track aggregation helps where single frames fail: +0.098 Rank-1 at 12 px on ChokePoint, and +0.009 on YouTube Faces. The plain mean (R1) is the best rule.
- Three questions remain open:
  1. **Open set.** It was inconclusive: only 6 non-enrolled identities.
  2. **Frame quality.** The MagFace norm did not help, so a better quality signal is needed.
  3. **Tracker default.** ByteTrack is provisional.
- Abser's concern: confirming choices on the data that produced them is circular.
- **MEVID** answers that concern. The probe (branch `mevid_probe`, merged 2026-11-23) gave GO on the high-resolution gallery route:
  - consented actors, so no celebrity overlap with MS1MV2 or WebFace;
  - real low-resolution faces (median 32.5 px);
  - actor check-in photos as the gallery.
  - **Usage terms:** cite Davila et al., "MEVID", WACV 2023, arXiv 2211.04656 (Abser, 2026-11-24). Record this in `DATASETS.md`; the report cites it.
- Scope decided by Abser on 2026-11-24:
  - MEVID train package included (all 158 identities);
  - tracker test on ChokePoint with a rotated open set;
  - optional arms: the WebFace4M backbone on MEVID, and the quality reject test;
  - FaceMoE not included.
- Restoration gate (Objective 1): Phase 4 closed restoration, so per the Sprint 2 plan the gate is **reported as tested and closed, with no run**.
- Executor: Claude Code on the lab PC with the team. One device for all numbers: the lab PC, RTX 4000 Ada.
- Rules: inference only; numbers only from scripts; pre-register before computing; seed 42; failures logged.
- After approval, save this plan as `research/Sprint_Notes/Phase8_plan.md`.

## Part 0: MEVID decisions (Abser confirms in the §0 commit, before any embedding)
These come from the probe's detection data only. No recognition score exists yet.

| # | Decision | Recommended setting | Basis |
|---|---|---|---|
| M1 | Identities | All 158: train package (30.5 GB) plus test | We train nothing, so the split does not matter. The other people's check-in photos are larger (median height 1,844 px over all 2,898 photos, against about 500 px for test people) |
| M2 | Gallery photos | **Front view only** | All 158 front photos of test people have a face. Back-view "faces" are turned heads or false boxes: G3 counted one person only through a back photo |
| M3 | Gallery face size bar | **48 px**, with a 64 px sensitivity row | Test people with a front face of at least 48 / 56 / 64 px: 52 / 44 / 28 of 54. 48 px is still above the probe median of 32.5 px |
| M4 | Gallery detector | SCRFD (official preprocessing), photo resized so its longest side is **1280**; the largest front face with score ≥ 0.5 | Size 640 shrinks faces on 1,844 px photos by about 2.9×. 1280 keeps photo faces in SCRFD's range (Phase 6 T1: faces fail only when upscaled to about 550 px) |
| M5 | Gallery template | Plain mean (R1) of the person's usable front-face embeddings | Phase 7 rule |
| M6 | Extra gallery identities | Enrol the 14 people who have photos but no tracklets (172 against 158) | A realistic gallery: enrolled but never seen |
| M7 | Probe tracklet | Up to **50 chips** evenly spaced per tracklet; face rule as in the probe (score ≥ 0.5, box centre in the upper half, native height ≥ 16 px); tracklets with **≥ 5 usable faces** are probes | Probe rule. 50 chips bounds the run (8,092 tracklets × 50 ≈ 405k chips) |
| M8 | Cameras | No camera excluded. Results pooled and per camera; eligibility by camera reported | Excluding cameras after seeing the data would be a data-driven choice; ineligible tracklets drop out by M7 anyway |
| M9 | Open-set design | Among people who have both a usable gallery and an eligible tracklet: **5 rotated folds**, 20% held out per fold (seed 42). Non-enrolled probes = held-out people's tracklets. People with no usable gallery photo are a **separate** non-enrolled row | Holding people out by photo quality would mix "not enrolled" with "worse data" |
| M10 | Metrics | Closed-set Rank-1; open-set **TPIR1** at FPIR 0.3, 0.1 and 0.01 (primary), plus TPIR20 for continuity with Phases 1–7; cluster bootstrap over identities (1,000 draws) | With about 170 gallery identities, rank 20 is too lenient |
| M11 | Recognizers | AdaFace MS1MV2 (primary, the pipeline default); AdaFace WebFace4M (secondary, the backbone question) | Abser's choice |
| M12 | Detector size on chips | `det.detect(chip, size=640)`, as in the probe | Already used for eligibility |

**Primary MEVID hypotheses** (AdaFace MS1MV2, all eligible tracklets, gallery bar 48 px):
- **H1:** R1 tracklet Rank-1 minus single-frame expected Rank-1 (paired per tracklet), CI above zero. This replicates Phase 7 on real low resolution.
- **H2:** R1 minus single-frame on open-set TPIR1 at FPIR 0.1 (fold-averaged), CI above zero.
- **Secondary:** the 64 px gallery bar; per camera; face-height bins (real sizes now); the no-usable-photo non-enrolled row; R2–R6; the WebFace4M rows.

**Limitations to state:**
- check-in photos are probably from the recording days, so the enrolment-to-probe gap is short;
- tracklets are given (MEVID's own tracking), not produced by our tracker;
- the probe faces are body-chip crops.

## Part A: MEVID data and embedding
1. Download `mevid-v1-bbox-train.tgz` with `scripts/mevid_download.py` (parallel range requests). Extract with `scripts/mevid_extract.py`. Record SHA-256.
   - Update the `DATASETS.md` row: full dataset held; usage terms = cite arXiv 2211.04656.
2. New `scripts/mevid_embed.py`. It reuses:
   - `src/video/pipeline.py:build_detector`;
   - `src/video/embed.py:FaceEmbedder` (alignment from SCRFD keypoints, AdaFace, MagFace raw norm), with `return_raw=True` so the **AdaFace raw norm is saved** for Part C;
   - the face rule from `scripts/mevid_probe.py`.
   - Outputs (local only, never committed): `outputs/video/mevid/emb/` per tracklet (`det`, `kps`, `emb_adaface`, `adaface_norm`, `mag_norm`, `lum`, `lap_var`, `face_h`) and `gallery.npz`.
   - Resumable per tracklet; chunked; launched detached.
   - Index CSVs are tracked: `tracklets.csv` (eligibility and counts), `gallery_faces.csv`.
3. **WebFace4M rows:** embed the same aligned crops with the CAFace backbone (`scripts/video_caface.py` path, `envs/caface`).
4. **Cost** (estimates): detection of about 405k chips ≈ 3–4 h; embedding about 135k faces ≈ 2 h; WebFace4M ≈ 1 h. Run over 1–2 nights.

## Part B: MEVID open set and aggregation (CPU, from cache)
- New `scripts/mevid_openset.py`. It reuses:
  - `src/video/aggregate.py` (R1–R6);
  - from `scripts/video_aggregation.py`: `track_records`, `draws_mean`, `ci`, `open_set`;
  - the TPIR arithmetic in `scripts/video_phase7_bench.py:tpir_at`, extended to rank 1;
  - `src/eval_protocols.py`.
- Outputs `outputs/video/mevid/results/*.csv` (tracked). Tables are injected into `REPORT_phase8.md` by `scripts/phase8_report_tables.py` (the Phase 7 pattern).

## Part C: frame-quality reject test (Phase 7 decision 2)
- **Question:** which signal best predicts identification errors, so the pipeline can reject or flag unreliable tracks?
- **Data:**
  - MEVID (primary: real low resolution);
  - ChokePoint 12 px and 16 px from cache (secondary: synthetic). There, face size is not a valid signal, because the degraded faces keep their native detection.
- **Signals per tracklet:**
  - number of usable faces;
  - self-consistency (mean cosine of each face to the track's leave-one-out mean);
  - mean face height (MEVID only);
  - mean Laplacian variance;
  - mean luminance;
  - mean AdaFace raw norm;
  - mean MagFace norm (reference).
- **Metric:** error against rejection for closed-set R1 (error rate as the lowest-quality 0–50% of tracklets are rejected), and the area under that curve.
- **A signal "helps" if** its area is below that of random rejection, with the CI of the difference excluding zero (paired bootstrap over identities, Bonferroni over the signals).
- **Output:** the quality signal for the Phase 9 demo's quality bar, or "none is reliable".

## Part D: final tracker choice on ChokePoint (CPU from cache plus small new code)
- **Face-aware association:** add a quality-gated face-embedding cost to **both** trackers, so the comparison is fair:
  - `cost = (1 − λ)·(1 − IoU) + λ·(1 − cos)`, applied only when both faces pass the Part C signal (fallback: native height ≥ 32 px and self-consistency, fixed before the run);
  - λ = 0.5, fixed, not tuned.
  - Implemented in `src/video/track.py:TrackerAdapter` as an option. The vendored trackers stay unchanged apart from a cost hook (record it in `VENDOR.md`).
- **Conditions:** S0 to S5 from `scripts/video_tracker_stress.py` (Phase 7 Addendum 2), with and without the face term.
- **Metrics:**
  - IDF1 and ID switches (TrackEval, vendored);
  - **downstream open-set track-identity accuracy**: each predicted track gets its R1 identity or "unknown" against a gallery with **all 29 identities rotated** as non-enrolled across 5 folds of about 6, at the FPIR 0.1 threshold.
- **Decision rule** (registered in §0): the tracker with the higher mean track-identity accuracy over S0–S5 (with the face term) wins if the CI of the per-clip difference excludes zero. Otherwise ByteTrack stays the default.
- **Caveat:** this is within ChokePoint, not held out (MEVID has no frame coordinates). Stated in the report.

## Part E: restoration gate (Objective 1 closure)
No run. `REPORT_phase8.md` states that the gate stage is closed by the Phase 4 stop rule (0 of 3 non-leaky recognizers for every arm), and that the pipeline keeps the hook off by default.

## Acceptance checks
1. Tracklet index counts equal the README (8,092 tracklets, 158 identities) and the gallery photo counts equal the inventory.
2. On the 200 probe-sampled tracklets, `mevid_embed.py` reproduces the probe's face boxes exactly (same detector and rule).
3. Aggregation: a length-1 tracklet equals single-frame for every rule; frame order does not change R1–R5 (as Phase 7).
4. Open-set folds: every eligible identity is held out exactly once; gallery and non-enrolled sets are disjoint in each fold.
5. Face-aware term with λ = 0 reproduces the Phase 7 S0–S5 tracker results byte-for-byte.
6. Reruns of every CPU part are byte-identical.

## Deliverable
`REPORT_phase8.md`:
- §0 pre-registration (Part 0 decisions, hypotheses H1 and H2, Part C and D rules), committed before any Phase 8 number;
- summary; MEVID results (primary, secondary, per camera, face-height bins); backbone rows; quality reject curves; tracker decision; gate closure;
- limitations; decisions for Phase 9 (detector mode, tracker, quality bar, backbone);
- STOP for the supervisor.
- MEVID is cited.
- Update `CONTEXT.md` (§3, §4, §11; also bring the stale §3 rows for Phases 6 and 7 up to date) and mirror it to Drive.

## Order of work (lab days Tue–Thu, Mon/Fri if available)

| When | Work |
|---|---|
| Day 1 | Start the train-package download (parallel); write and commit §0 with the Part 0 decisions (Abser confirms); `mevid_embed.py` plus check 2 on the probe sample |
| Day 1 night | MEVID detection and embedding (detached, resumable) |
| Day 2 | Part D code plus check 5, tracker runs (CPU); WebFace4M embedding overnight |
| Day 3 | Part B and Part C runs, checks 1, 3, 4, 6 |
| Day 4 | Tables, `REPORT_phase8.md`, `CONTEXT.md`, PR, then STOP |

## Verification
- `git log` shows the §0 commit before any `outputs/video/mevid/results` file.
- Checks 1–6 are written to `outputs/video/mevid/checks.json`.
- No MEVID image or embedding is tracked (`git status --ignored`).
- Report tables are injected by the table script, never typed by hand.

## Risks
- **Train download is slow** (0.2–0.3 MB/s per connection): run it detached with parallel ranges from day 1. Fallback: the test split alone (54 identities), recorded as a deviation before any score.
- **Train photos differ** from the test inventory (no faces, a different naming scheme): the gallery rule still applies; report the coverage per split.
- **Few enrolled identities at 48 px:** the 64 px row shows the sensitivity. The CIs are reported as they come.
- **The face-aware term changes tracker behaviour unexpectedly:** check 5 guards the λ = 0 baseline; ByteTrack stays the default unless the rule says otherwise.
