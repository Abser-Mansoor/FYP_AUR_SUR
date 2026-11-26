# Phase 9: final pipeline, error investigation, quality bar, FPS, retrieval and demo

Plan: `research/Sprint_Notes/Phase9_plan.md` (approved 2026-11-26). Executor: Claude Code on the lab PC (RTX 4000 Ada). All numbers come from scripts. Seed 42. Inference only.

## 0. Pre-registered protocol (committed before any Phase 9 number)

This section is committed before any Phase 9 result is computed.
- Allowed before this commit (they produce no metric): code, the MEVA clip download, and embedding runs.
- §0b fixes the final configuration after the development runs, and it is committed before the confirmation split is read.

### 0.1 Decisions (Abser, 2026-11-26)
| # | Setting | Value |
|---|---|---|
| D1 | Tracker | ByteTrack with the Phase 8 face term: cost (1 − λ)·(1 − IoU) + λ·(1 − cos), λ = 0.5. Applied when both faces have score ≥ 0.5 and native box height ≥ 32 px |
| D2 | Quality bar | A track (tracklet) whose **mean face height is below 23 px** gets no identity; it is shown as "low quality". Mean face height = the mean native box height of its usable faces (score ≥ 0.5, height ≥ 16 px; Phase 8 rule) |
| D2a | Expected coverage and action | Expected coverage on MEVID: about 74%. 23 px is the quarter cut of Phase 8: the 25th percentile of the mean face height of the 3,188 tracklets is 22.66 px, and 23 px keeps 73.7% (`scripts/phase9_prereg_numbers.py` → `outputs/video/phase9/prereg_numbers.json`, from Phase 8 outputs only). **Demo coverage floor: 60%.** Below 60%, the coverage is reported and the threshold is **not** retuned. The threshold also stays at 23 px when coverage is above 60% |
| D2b | Error statement | The report gives the error of the kept tracks at 23 px with its CI, and states plainly that it is still high if it is. MS1MV2 R1 at 23 px, from the Phase 8 outcomes: 39.5% of the kept tracklets wrong, against 50.5% overall |
| D3 | Aggregation rule | R4*: norm pooling, min-max, λ = 5 (`src/video/aggregate.py:r4s_np_star`). Quality q = MagFace (MS1MV2) raw norm. R2 and R1 are reported rows. **Fallback:** if R4* is below R2 on the development split with WebFace4M (paired 95% CI of R4* − R2 entirely below zero), R2 becomes the default |
| D4 | Backbone | AdaFace IR-101 WebFace4M, standalone checkpoint `models/caface/adaface_ir101_webface4m.ckpt`, through `src/recognizer.py`. Flip off: this is how the Phase 8 WebFace4M rows were made. The MagFace network stays, for q. **All Phase 9 WebFace4M features are re-embedded in `envs/video`** (`scripts/phase9_embed.py`: `wf_<tracklet>.npz`, `gallery_wf.npz`), because the Phase 8 features were made from crops warped by a different OpenCV (0.7, check 1) |
| D5 | Detector | SCRFD det_10g, official preprocessing, NMS 0.45. ChokePoint: `full1280`. MEVA: the mode per camera from `scripts/video_mode_calibrate.py` on the camera's calibration clip (0.6) |
| D6 | Open-set operating point | Threshold at FPIR 0.1 |
| D7 | Deferred | YouTube Faces usage terms, appendix T2, PaSC (access blocked by ISP and VPN filtering) |

### 0.2 MEVID split (Part B)
- **Data:** the 3,188 Phase 8 tracklets: eligible, of persons enrolled at the 48 px gallery bar.
- **Development:** tracklets of MEVID train-split persons (2,297 tracklets).
- **Confirmation:** tracklets of test-split persons (891 tracklets).
- **Gallery:** all 170 persons enrolled at 48 px, in both splits.
- **Confirmation access:** `scripts/mevid_errors.py` reads the confirmation tracklets only with `--confirm`. That flag is run once, after the §0b commit (acceptance check 6).
- **What is not held out:** Phase 8 used all of MEVID, so D3 and D4 were chosen with the confirmation persons in view. Only the Phase 9 fixes (0.4) are chosen without them.

### 0.3 Hypotheses (confirmation split, one look)
- **H1 (closed set):** Rank-1 of the frozen final configuration minus Rank-1 of the Phase 8 baseline (AdaFace MS1MV2, flip on, R1).
  - Paired per tracklet; the baseline is read from `outputs/video/mevid/results/closed_records_adaface_bar48.csv`.
  - Cluster bootstrap over persons, 1,000 draws, seed 42.
  - Supported if the 95% CI is above zero.
- **H2 (open set):** fold-averaged TPIR1 at FPIR 0.1 of the final configuration minus that of the baseline.
  - Folds: 5 folds of the confirmation persons that have an eligible tracklet (numpy `default_rng(42)` permutation, `array_split`). In fold k the held-out persons are removed from the gallery; train persons always stay enrolled.
  - Threshold per fold: the 0.9 quantile of the held-out tracklets' top-1 similarities (Phase 8 arithmetic).
  - Bootstrap: both configurations use the **same** person resamples (paired). The Phase 8 implementation drew them separately, which gave a wider, conservative CI. The paired form is used from Phase 9 on.
  - Supported if the 95% CI is above zero.
- **Secondary (confirmation; development rows reported too):**
  - coverage and error at 23 px, with CIs;
  - the error-vs-coverage curve over thresholds 16 to 48 px (step 1 px) and over rejected fractions 0 to 0.5 (step 0.01);
  - the residual-quality test (0.5).

### 0.4 Part B: error investigation and fixes (development split only)
**B1 baseline.** On development: single frame, R1, R2 and R4* with WebFace4M (q = MagFace norm), and the same rows with MS1MV2 for reference. The D3 fallback is applied here. The resulting rule is "the default".

**B2 decomposition** (descriptive, no test). Error of the default against:
- face height bins: 16–20, 20–24, 24–32, 32–48 and ≥ 48 px;
- yaw proxy y = |x_nose − (x_left eye + x_right eye)/2| / |x_right eye − x_left eye| (SCRFD 5-point order), tracklet mean, quartiles;
- Laplacian variance, quartiles;
- detector score, quartiles;
- camera;
- number of usable faces;
- the person's best gallery face height.

It also reports:
- the rank of the true identity among the 170 (share of errors with the true identity at ranks 2–5);
- the concentration of errors over persons (share of errors from the worst 10% of persons);
- the 10 most frequent (true, predicted) pairs.

**B3 visual audit.**
- **Sample:** 120 wrong tracklets with mean face height ≥ 23 px, drawn with numpy `default_rng(42)`, plus 40 correct ones from the same height range as controls. Shuffled, with outcomes hidden.
- **Sheet:** per tracklet, 5 evenly spaced face crops, the true person's gallery photo and the predicted person's gallery photo, unlabelled as "true" or "predicted". The sheet is local only.
- **Tags (several allowed):**
  - not a face or back of head;
  - profile (more than about 45°);
  - blur;
  - occlusion;
  - low light or IR;
  - tiny;
  - wrong person in the tracklet;
  - gallery mismatch (poor photo, or appearance change such as hair, glasses or hat);
  - look-alike;
  - nothing visible.
- **Labelers and agreement:** two team members label independently. Agreement is Cohen's kappa per tag.
- **Use:** the audit describes causes and does not select fixes.

**B4 fixes.** Fixed before any B1 number. All are inference only.

| Fix | Definition | Measured on |
|---|---|---|
| F1 frame filter | Before aggregation, drop faces with yaw proxy > 0.35 or detector score < 0.6. If fewer than 3 faces remain, the tracklet is used unfiltered | closed-set Rank-1 |
| F2 resolution-matched gallery | For each usable gallery face, the photo is resized (INTER_AREA) so that the face box height is 24 px and 32 px, then aligned with the scaled keypoints (`norm_crop`) and embedded. The template is the normalised mean of the native, 24 px and 32 px embeddings | closed-set Rank-1 |
| F3 flip | WebFace4M embeddings with the flip sum (probes and gallery) | closed-set Rank-1 |
| F4 fusion | Similarity = 0.5·cos(WebFace4M) + 0.5·cos(MS1MV2), each with the default rule | closed-set Rank-1 |
| F5 cohort normalisation | Probe score s' = s − mean of the probe's 10 highest cosines to a cohort of 2,000 TinyFace distractors (`Gallery_Distractor`, seeded, disjoint from the Part E padding set; resize-only, as Phases 1–3). It does not change the closed-set decision | open-set TPIR1 at FPIR 0.1 (5 folds of the development persons, built as in H2) |

**Adoption rule:**
- F1–F4: paired difference to the default, cluster bootstrap over persons (1,000 draws). A fix is adopted if the Bonferroni CI (level 1 − 0.05/4) is entirely above zero.
- F5: adopted if its 95% CI is above zero.
- F4 needs one more condition: the lab-PC end-to-end rate at 720p with both backbones must be at least 10 frames per second (the demo floor of Phase 8 §14.1), in the slowest detector mode that the calibration assigns to the demo cameras (`scripts/phase9_fps.py` → `fps_condition.json`). Otherwise it is reported but not adopted.
- The adopted fixes are combined and measured once on development. If the combination is below the best single adopted fix, only that fix is kept.
- §0b records the frozen configuration.

### 0.5 Residual-quality test (Abser's concern: height cannot see blur, pose or occlusion)
- **Data:** the tracklets kept by the 23 px bar.
- **Method:** the Phase 8 Part C test with the default configuration's errors. This is the area under the error-vs-rejected-fraction curve over 0–0.5, minus the random area, with a paired cluster bootstrap and Bonferroni over the signals.
- **Signals:** MagFace norm, mean yaw proxy (negated, so that higher is better), Laplacian variance, and mean face height (as the reference).
- **Order:** development first, then confirmation.
- **Outcome:** if the MagFace norm helps on both splits, a second gate is proposed to Abser for after Phase 9. It is not adopted in Phase 9.

### 0.6 Parts C–F
- **C. ChokePoint end-to-end** (the clips with ground truth and the Phase 6 detections; identity on the SC clips as in Phase 8 Part D). These are not held out; they show the final configuration works end to end.
  - Inputs: WebFace4M embeddings of every detection with score ≥ 0.5 (`embed_webface4m.npz` next to the Phase 6 `embed.npz`, same `det_id` order).
  - Tracker: the stress conditions S0–S5 of `scripts/video_tracker_stress.py`, with the face term on WebFace4M cosines (flip off; if F3 is adopted, only the identity templates use the flip).
  - Comparison arm: the Phase 8 configuration (MS1MV2, R1, MS1MV2 face term) on the same folds; paired per-clip difference over S0–S5, cluster bootstrap over clips.
  - Reported: IDF1, HOTA, ID switches; open-set track-identity accuracy at FPIR 0.1 (the Phase 8 Part D definition, `scripts/mevid_tracker_identity.py`), with tracks templated by the default rule; the share of tracks under the 23 px bar.
  - Rule: if IDF1 under S1–S3 with the WebFace4M face term is below the λ = 0 IDF1, this goes to Abser as a decision. λ is not retuned.
- **D. FPS at 720p.**
  - Frames: 300 frames of each MEVA demo clip, resized to 1280×720.
  - Settings: single stream, batch 1.
  - Stages: decode, detect (`full640`, `full1280`, `tiled`), align + WebFace4M, align + MS1MV2 (for F4), MagFace norm, track, aggregate, search. Median ms per frame, per face or per track; end-to-end ms per frame from the medians.
  - Devices: lab PC; T4 on Colab (Umar). If no T4 is available, the reason is stated.
- **E. Retrieval.**
  - FAISS `IndexFlatIP` against `IndexHNSWFlat` (M 32, efConstruction 200, efSearch 64).
  - Galleries: 10², 10³ and 10⁴ entries: the 170 MEVID templates, plus WebFace4M embeddings of seeded TinyFace distractors to fill up. For 10², 100 MEVID templates are chosen at random (seed 42).
  - Probes: all 3,188 tracklet templates of the frozen configuration (§0b).
  - Reported: Rank-1 against MEVID ground truth, agreement of the HNSW top-1 with the Flat top-1, and median query latency (batch 1, 1 thread).
- **F. MEVA demo.**
  - Clips: 8 cameras that have drop-5 clips and ≥ 50 eligible MEVID tracklets (G329, G330, G340, G505, G507, G508, G509, G639). For each camera, 2 clips are drawn with numpy `default_rng(42)` from the alphabetically sorted list: the first is the demo clip, the second the calibration clip. The list is in `outputs/video/phase9/meva_clips.txt` (SHA-256 of the files in `meva_sha256.txt`). The selection used only the S3 file listing.
  - Gallery: the 170 persons (MEVID front check-in photos, ≥ 48 px), templated as in the frozen configuration.
  - Open-set threshold: the mean over the 5 development folds of the FPIR 0.1 threshold of the frozen configuration (`frozen_config.json`, `demo_threshold`).
  - Detector mode per camera: `scripts/video_mode_calibrate.py` on the camera's calibration clip (rule: 5th percentile of face height ≥ 32 px → `full640`, else `tiled`). If the calibration frames contain no face: `tiled`.
  - Pipeline order: detect → embed (WebFace4M + MagFace norm for score ≥ 0.5) → ByteTrack with the face term → identify. Offline per clip.
  - Output: an annotated MP4 per clip (local only).
  - **Coverage, primary** (matched to the MEVID eligibility rule): among tracks with ≥ 5 usable faces (score ≥ 0.5, height ≥ 16 px), the share with mean usable-face height ≥ 23 px, against the 60% floor.
  - **Coverage, broad:** among all tracks with at least one face of score ≥ 0.5; a track with no usable face counts as below the bar.
  - Also reported per clip: counts of identified / unknown / low-quality tracks, and runtime per stage.
  - Qualitative check: one team member compares 30 seeded identified tracks with the check-in photos. MEVA has no face-identity labels, so this is not a measured accuracy.

### 0.7 Acceptance checks (`outputs/video/phase9_checks.json`; a failure is a STOP)
1. WebFace4M through `src/recognizer.py` (alias `adaface_wf4m`) is the network that made the Phase 8 `cf_*.npz` features: cosine > 0.9999 on 100 seeded MEVID faces **given identical crops**.
   - **Result before this commit (amended check).** As first written, the check compared crops re-made in `envs/video` with the Phase 8 features, and it failed: cosine minimum 0.948, median 0.99965.
   - **Cause.** `envs/caface` has OpenCV 5.1.0 and `envs/video` has 4.13.0. `norm_crop`'s warp of the same tiny chips differs by a mean of 0.13 grey levels (maximum 5).
   - **Model equivalence.** Given the crops made in `envs/caface`, the pipeline recognizer reproduces the Phase 8 features: cosine minimum 0.9999996.
   - **Consequence.** The network is the same; the crops differ. Phase 9 therefore re-embeds every WebFace4M feature through the pipeline path (0.1, D4). The Phase 8 WebFace4M rows stand as reported, with this caveat.
   - The numbers above come from a first diagnostic run. The amended check is run formally by `scripts/phase9_checks.py --check 1a` (`envs/caface`) and `--check 1` (`envs/video`).
2. The pipeline set to the Phase 6/7 switches (MS1MV2, R1, λ 0, no bar) reproduces the Phase 7 ChokePoint tracker table byte for byte, and `FaceEmbedder` with its defaults gives exactly the stored Phase 6 embeddings and MagFace norms on 100 seeded ChokePoint faces.
3. The pipeline's track templates equal `src/video/aggregate.py` on cached tracklets.
4. The identity is withheld exactly when the mean face height is below 23 px (unit test).
5. The FAISS Flat top-1 equals exact NumPy search for every probe.
6. The development and confirmation persons are disjoint, and `git log` shows the §0b commit before the confirmation result files.
7. CPU reruns are byte-identical; a demo rerun on one clip gives identical tracks.

### 0.8 Limitations stated in advance
- D3 and D4 were chosen on all of MEVID (0.2).
- MEVID tracklets come from MEVID's own tracking, and its faces are body-chip crops.
- The MEVA demo clips come from the same actors and cameras as MEVID, and they have no identity labels.
- ChokePoint is not held out.
- TinyFace distractors are low-resolution crops, so the large-gallery retrieval table measures speed and approximation more than realistic confusion.
