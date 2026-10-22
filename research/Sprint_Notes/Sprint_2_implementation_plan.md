# Plan: Sprint 2: video recognition pipeline (main track) plus restoration test (parallel track)

**Status:** approved 2026-10-22. Companion to `Sprint_2_video_notes.md` (literature, datasets) and `supervisor_brief_2026-10-20.pdf`.

## Context
- **Decision.** The supervisor brief of 20 Oct 2026 closed SR as a negative result. It made the video pipeline the main deliverable for the next 8 weeks: detection, alignment, recognition, tracking, per-track aggregation and retrieval.
- **Parallel track.** One cheap, non-generative test of low-light and deblur methods runs alongside. Its winner may become an optional gated stage, off by default.
- **Sources.**
  - `research/Sprint_Notes/Sprint_2_video_notes.md` lists papers, techniques, datasets and a feasibility ranking.
  - Brief sections 6–8 give the restoration test, the pipeline, the 4 experiments and the per-member split.
- **Format.** The plan keeps the Phase 0–3 working style:
  - each phase has a goal, tasks, deliverables and acceptance checks;
  - each phase ends in `REPORT_phaseN.md` and a STOP for approval;
  - phases are sized to 1–2 weeks.

**Standing rules carried over:**
- inference only, no training;
- fixed seeds (42);
- never invent numbers, and every number comes from a script output;
- log failures as results;
- datasets are read-only;
- one venv per model repo;
- config-driven;
- one GPU type per reported number;
- one GPU job at a time on the lab PC, with resumable runs (the PC reboots);
- Windows-native (PowerShell, `num_workers=0`, no symlinks);
- the Linux home PC is for editing only.

**Calendar** (Week 1 starts Mon 26 Oct 2026):

| Week | Dates |
|---|---|
| W1 | 26 Oct – 1 Nov |
| W2 | 2 – 8 Nov |
| W3 | 9 – 15 Nov |
| W4 | 16 – 22 Nov |
| W5 | 23 – 29 Nov |
| W6 | 30 Nov – 6 Dec |
| W7 | 7 – 13 Dec |
| W8 | 14 – 20 Dec |

**Assumption, not yet confirmed (Sprint 2 notes, open question 1):** ChokePoint and YouTube Faces are the reported video benchmarks. COX, FANVID and SCface are used only if they arrive by the end of W3.

## Overview

| Phase | Track | Weeks | Owner | Output |
|---|---|---|---|---|
| 4 | Restoration test: low-light + deblur (CPU) | W1–W2 | Raghib | REPORT_phase4.md; go/no-go for the gate |
| 5 | Video foundations: data, environments, detector study | W1–W2 | Abser (data, env), Umar (detector) | REPORT_phase5.md |
| 6 | Core pipeline and tracker validation | W2–W3 | Abser (pipeline), Umar (embedding, index) | REPORT_phase6.md |
| 7 | Aggregation experiments (headline result) | W4–W5 | Umar (protocol, metrics), Abser (runs) | REPORT_phase7.md |
| 8 | Restoration gate + robustness extensions | W5–W6 | Raghib (gate), Abser/Umar (extensions) | REPORT_phase8.md |
| 9 | Integration, FPS, demo | W6–W7 | Abser (demo), Umar (speed tables) | REPORT_phase9.md + demo video |
| 10 | Consolidation and write-up | W8 | All | thesis chapters, slides, freeze |

**Supervisor checkpoints (every two weeks):**
- end of W2: Phases 4 and 5;
- end of W3: Phase 6;
- end of W5: Phase 7;
- end of W7: Phases 8 and 9;
- W8: final.

A STOP only blocks the track it belongs to. Other members keep preparing.

**GPU schedule:**
- Phase 4 runs on CPU.
- Lab PC GPU: video phases only.
- Colab T4s: overflow, such as the detector study on WIDER FACE and FaceMoE placement. They reuse the chunked, resumable pattern of `scripts/sr_colab_job.py`.

---

## Phase 4: restoration test, low-light and deblur (W1–W2, Raghib, parallel)
**Goal.** Answer brief section 6. Does any cheap non-generative enhancement give a **probes-only** recognition gain with a CI above zero? If yes, it becomes the gate arm for Phase 8. If no, close the restoration engine and record the result.

**Arms (all CPU):**
- **Low-light:**
  - gamma (sweep γ ∈ {0.5, 0.7}, plus an adaptive γ that maps mean luminance to a target);
  - CLAHE on the L channel of Lab (clip ∈ {1, 2}, tiles ∈ {2×2, 4×4} for tiny crops);
  - Zero-DCE++ (official repo `Li-Chongyi/Zero-DCE_extension`, `Epoch99.pth`, about 10k parameters).
- **Deblur:**
  - unsharp mask (σ ∈ {0.5, 1.0}, amount ∈ {0.5, 1.0});
  - NAFNet GoPro width-64 (official `megvii-research/NAFNet`). It runs at native size padded to a multiple of 16, plus a pre-128 bicubic variant.
- **No-op control:** γ = 1.0 and amount = 0. They must reproduce direct feed exactly. This is the harness sanity check.
- **Gated variant** of the tuning winner (Sprint 1 Task 5):
  - apply the enhancement only when the Phase 2 measure flags the crop (luminance for low-light, `lap_var` for deblur);
  - the threshold is chosen on the tuning subset.

**Protocol:**
- The existing Phase 3 subsets: tuning 2,800 and eval 22,898.
- The same selection rule: mean over AdaFace, MagFace and CurricularFace of [TinyFace Rank-1, SurvFace verification AUC], computed on tuning.
- Reported on eval:
  - both sides restored and probes-only (`scripts/sr_extra_analysis.py`);
  - per-bin results on `dark10`, `dark` (20%, for more power than the n=98 bin) and `sharp_bin_sz` low.
- **LFW synthetic check** (`scripts/sr_lfw_check.py` pattern), with two new degradations:
  - dark: intensity scaled down, plus Poisson–Gaussian noise;
  - blur: Gaussian or linear-motion kernel.
  - The degradation level is calibrated on direct feed only, to cross-direct AUC ≈ 0.80, as in Phase 3.
  - Same pre-registered rule: paired CI above zero for at least 2 of the 3 non-leaky recognizers.
  - Bicubic or no-op is the fair comparator.
- **Stop rule (fixed now):** no arm or gated arm gives a probes-only TinyFace Rank-1 gain with a 95% CI above zero for at least 2 of the 3 non-leaky recognizers → the restoration engine is closed.

**Code changes:**
- New `configs/enhance.yaml`, a copy of `configs/sr.yaml` with the new arms and tuning variants. Phase 3's config stays frozen.
- New runners `src/sr/run_photometric.py` (gamma, CLAHE, unsharp; `--method`) and `src/sr/run_zerodce.py` and `src/sr/run_nafnet.py`. Each has its own venv (`envs/zerodce`, `envs/nafnet`). They reuse `src/sr/common.py` (`list_images`, `read_bgr`, `write_png`, `Timing`).
- `scripts/sr_pipeline.py`:
  - generalise `vid()` / `find_variant()` to the new parameter keys;
  - add a per-arm `interp` key, passed through to `src/sr/eval.py:get_embeddings`. Enhancement arms output native-size crops, so they must use the **direct-feed bilinear rule**, not the `area` rule used for 512 px SR outputs. Otherwise interpolation, not enhancement, would drive the deltas.
- Visual sheets with `scripts/sr_failure_cases.py`, `scripts/sr_sample_figs.py` and `scripts/sr_brief_strips.py`, pointed at the new arms.

**Deliverable.** `REPORT_phase4.md`: tables, probes-only, per-bin, LFW, visual samples, stop-rule verdict, and the gate arm or closure statement.

**Acceptance:**
- the no-op arms reproduce direct feed bit-for-bit (embeddings equal);
- every arm has timing;
- the stop-rule verdict is stated against the pre-registered rule.

---

## Phase 5: video foundations (W1–W2, Abser + Umar)
**Goal.** Get the data, environments and model weights in place, and measure where detection fails by face size.

**5a. Data (Abser, W1):**
- **ChokePoint** (Zenodo 815657):
  - download into `Dataset/ChokePoint/` (read-only after the checksum);
  - parse `groundtruth.tar.xz` (eye positions per frame per identity) into `outputs/video/gt/chokepoint_*.csv`;
  - read and record the official protocol (G1/G2 verification; gallery and probe sequences) **before** defining splits.
- **YouTube Faces:** registration request. If the portal is down by the end of W2, fall back to VoxCeleb2 cropped face tracks for the aggregation-scale study.
- **WIDER FACE val:** for the detector size study.
- **Access requests and emails:** FANVID authors; COX Face DB and SCface requests; an IJB-S question to the supervisor; a consent form for campus demo clips. Check whether university ethics approval is needed.
- Record each dataset's licence, size, SHA-256 and access date in a new `DATASETS.md`.

**5b. Environments and models (Abser + Umar, W1):**
- **`envs/video`:** a `--system-site-packages` venv like `envs/diag`, plus `faiss-cpu`, `lap`, `scipy`, `pandas`. It reuses the working onnxruntime-gpu SCRFD and torch recognizers.
  - If `faiss-cpu` has no Windows/py3.13 wheel, use exact NumPy inner product. It is exact, and galleries are small. HNSW would then be reported via `hnswlib`.
- **Trackers:** vendor only the tracker modules, at pinned commits with licence files:
  - ByteTrack `yolox/tracker/byte_tracker.py` (MIT) into `src/video/vendor/bytetrack/`;
  - BoT-SORT `tracker/bot_sort.py` and `gmc.py` into `src/video/vendor/botsort/` (check the licence first).
- **Detectors:**
  - SCRFD `det_10g.onnx` (already present, `models/detector/`);
  - RetinaFace R50, an official insightface or `biubug6/Pytorch_Retinaface` checkpoint.
  - SFE-DETR: check for code in W1, and drop it if it is absent.
- **Extra models:** CAFace (`mk-minchul/CAFace`, MIT; its own venv `envs/caface`) and FaceMoE (`Kartik-3004/FaceMoE`, MIT; `envs/facemoe`).
  - Download only, and record the SHA-256 in `MODEL_INVENTORY.md`.
  - Check whether CAFace needs its own AdaFace WebFace4M backbone rather than our MS1MV2 checkpoint. If it does, the CAFace row gets its own single-frame baseline.

**5c. Detector size study (Umar, W1–W2):**
- Compare SCRFD and RetinaFace, each three ways:
  - full frame;
  - tiled (SAHI-style, 2×2 with overlap);
  - 2× upscaled frame.
- Data:
  - WIDER FACE val (face-size bins in px: <10, 10–16, 16–24, 24–32, 32–64, >64);
  - ChokePoint (boxes derived from the eye labels);
  - TinyFace and SurvFace crops (the padded-crop protocol of `src/align.py:detect_landmarks`, as in Phase 1).
- Metric: recall at IoU 0.5 per size bin, at a fixed score threshold, plus false positives per frame. Runtime per frame.
- Pick the detector and its mode for Phase 6 with a rule fixed in advance: the highest recall on 16–32 px at ≤ 0.1 false positives per frame, then speed.
- Code:
  - new `src/video/detect.py`, which wraps `src/align.py:SCRFD.detect`, adds RetinaFace, and adds tiling plus cross-tile NMS (reuse `align._nms`);
  - new `scripts/video_detector_study.py`.

**Deliverable.** `REPORT_phase5.md`: data inventory and licences, access status, environment and pins, the detector-by-size table, and the chosen detector.

**Acceptance:**
- the SCRFD wrapper on full frames reproduces the Phase 1 landmarks on 100 TinyFace probes (same keypoints within 0.5 px);
- ChokePoint ground-truth counts match the dataset page (64,204 face crops; 48 sequences).

---

## Phase 6: core pipeline and tracker validation (W2–W3, Abser + Umar)
**Goal.** Build an end-to-end pipeline that works on ChokePoint: frames → detections → aligned faces → AdaFace embedding + MagFace norm → tracks → per-track identity. Validate the trackers against ground-truth face tracks.

**Code (new package `src/video/`):**
- `io.py`: frame source, either an image sequence (ChokePoint ships JPEG frames) or `cv2.VideoCapture`. Each frame keeps its index and timestamp.
- `detect.py`: from Phase 5.
- `embed.py`:
  - `norm_crop` (`src/align.py`), then `Recognizer.get_embedding` (`src/recognizer.py`);
  - the MagFace norm is computed as in `src/stratify/measure_split.py`, which takes the raw norm before L2 normalisation;
  - add a `return_raw` option to `Recognizer` rather than duplicating the code;
  - per-face luminance and Laplacian variance via `src/stratify/measures.py:crop_measures`.
- `track.py`: one adapter interface over ByteTrack and BoT-SORT, taking detections (box, score) per frame and returning track ids.
- `index.py`: gallery enrolment and search; FAISS `IndexFlatIP` or the NumPy fallback; an HNSW option.
- `pipeline.py`: orchestration. It writes per-sequence Parquet/CSV (`frame, det_id, box, kps, score, track_id, quality`) plus embeddings as `.npy`.
  - Resumable per sequence, with atomic writes (reuse `src/sr/common.py:write_png`-style temp-then-rename).
- `configs/video.yaml` and `scripts/run_video.py` (`detect | track | embed | all --dataset chokepoint`).

**Face-track ground truth.** ChokePoint's eye labels per identity per frame become ground-truth boxes, using a fixed eye-to-box rule. The rule is checked visually on 20 frames.

**Tracker validation:**
- Compare ByteTrack and BoT-SORT on predicted detections.
- Metrics: IDF1, HOTA, MOTA and ID switches, computed with TrackEval (pip `trackeval`, MIT) against the ground-truth face tracks.
- Report the 2 crowded sequences separately.

**Single-frame identification baseline:**
- Enrolment gallery from the official gallery sequences.
- Every probe-frame embedding is matched independently.
- Report Rank-1 and open-set TPIR@FPIR, with logic reused from `src/eval_protocols.py`.

**Deliverable.** `REPORT_phase6.md`: tracker table, single-frame baseline, stage timings, failure examples (ID switch montages).

**Acceptance:**
- **identity check:** feeding ground-truth boxes as detections gives IDF1 ≈ 1 for both trackers;
- **embedding check:** a face run through the video path has cosine > 0.9999 to the same face through the Phase 1 aligned path;
- reruns are byte-identical (deterministic seeds).

---

## Phase 7: aggregation experiments, the headline result (W4–W5, Umar + Abser)
**Goal.** Experiments 1 and 2 of the brief. Does per-track aggregation beat single-frame identification, and which rule is best? This is the new thesis result: tracking uses real information where SR invents it.

**Rules** (all inference-only, on AdaFace embeddings unless noted):
- per-frame majority vote (baseline);
- plain mean;
- MagFace-norm-weighted mean;
- top-k by quality (k ∈ {1, 3, 5, 10});
- norm pooling (arXiv 2409.09832);
- CAFace (pretrained; own backbone if needed, see 5b).

**Design:**
- **Two track sources:**
  - (a) ground-truth tracks, which isolate aggregation from tracker errors;
  - (b) predicted tracks from Phase 6, which give real-system numbers.
- **Datasets:**
  - ChokePoint, using the official protocol;
  - YouTube Faces (or VoxCeleb2), using the standard 5,000-pair verification protocol. Its frames are pre-aligned, so they feed straight into the recognizer.
- **Selection.** Hyperparameters (k, quality threshold) are chosen on a tuning split (a disjoint subset of identities or sequences), then reported on eval. This mirrors the Phase 3 tuning/eval split.
- **Low-resolution stress test (diagnostic only, per policy D8):** downsample ChokePoint frames so faces reach the QMUL range (about 16–24 px). Repeat (a) to show whether aggregation gains grow as single frames fail. FANVID replaces this if it is released.
- **Metrics:**
  - closed-set Rank-1;
  - open-set TPIR20@FPIR=0.3 with a "not enrolled" outcome;
  - TAR@FAR for YTF;
  - bootstrap CIs over identities/tracks (`src/metrics.py:bootstrap`, 1,000 draws, seed 42);
  - paired deltas against single-frame (`src/metrics.py:paired_delta`).
- **Curves:**
  - accuracy against the number of frames used;
  - accuracy against face-size and quality bins (the Phase 2 measures).

**Code:**
- `src/video/aggregate.py`: one function per rule, with a common signature `(emb[N,512], quality[N]) -> emb[512]`.
- `scripts/video_aggregation.py`: experiment runner and table generator. The tables are injected into the report between markers, as Phase 3's `sr_report_tables.py` does.

**Deliverable.** `REPORT_phase7.md`.

**Acceptance:**
- the mean of a single frame equals the single-frame result exactly;
- the result is independent of track ordering (shuffling frames inside a track does not change it, except for top-k ties, which are broken deterministically).

---

## Phase 8: restoration gate and robustness extensions (W5–W6)
**Goal.** Experiment 3 of the brief: does the gate help? Then optional extensions, in priority order. Stop adding extensions when W6 ends.

1. **Restoration gate (Raghib, required; it closes proposal Objective 1).**
   - If Phase 4 produced a winner: apply it per face crop before embedding, gated by the Phase 4 threshold.
   - Compare gate on vs off on the Phase 7 setup (single-frame and best aggregation rule; ChokePoint plus the low-resolution stress test).
   - If Phase 4 closed restoration: report the gate as tested and closed, with no run.
   - Code: a `gate` stage in `src/video/pipeline.py` that calls the Phase 4 runner in-process for photometric methods.
2. **Face-aware association (Abser):** add a quality-gated face-embedding term to the BoT-SORT cost (the FaceQSORT idea). Report the change in IDF1 and ID switches on ChokePoint.
3. **FaceMoE as a 5th recognizer (Umar, Colab):**
   - its own 120 px preprocessing;
   - first place it on the Phase 1 TinyFace split and the Phase 3 eval subset, to check that its claims transfer;
   - then add one row to the Phase 7 table.
   - Needs supervisor approval (open question 2).
4. **Person-then-face tracking on 5–10 selected MEVA clips (Abser, only if time remains):**
   - a COCO person detector (official YOLO/RT-DETR weights) with ByteTrack, plus faces attached inside body boxes;
   - track-continuity metrics and qualitative results only, because MEVA has no face labels;
   - needs supervisor approval (open question 3).

**Deliverable.** `REPORT_phase8.md`.

---

## Phase 9: integration, FPS and demo (W6–W7, Abser + Umar)
- **End-to-end runs** on all reported sequences with the final configuration.
- **FPS at 720p, broken down by stage** (detect, align + embed, track, aggregate, search):
  - on the RTX 4000 Ada and on a T4, each reported separately;
  - single stream, batch 1;
  - with and without tiling.
- **Retrieval table:** FAISS Flat against HNSW, comparing recall@1 and latency at gallery sizes 10², 10³, 10⁴ (padding the gallery with distractor embeddings from TinyFace).
- **Demo:** `scripts/video_demo.py`. Input is a video file; output is an annotated MP4 with boxes, track IDs, identity or "unknown", and a quality bar.
  - It runs on the campus clips recorded with consent. These are for the demo only and never feed reported metrics.

**Deliverable.** `REPORT_phase9.md` and the demo video.

## Phase 10: consolidation (W8, all)
- **Thesis chapters:**
  - SR (Phase 3, negative result);
  - restoration test (Phase 4);
  - video pipeline (Phases 5–9);
  - limitations (no IJB-S or BRIAR; ChokePoint is small; synthetic low-resolution stress test).
- Update `CONTEXT.md` and `MODEL_INVENTORY.md`.
- **Reproduce-from-scratch check:** re-run one ChokePoint sequence end to end from a clean checkout. The tables must match.
- Final slides and a code freeze tag.

---

## Parallel-work map

| Week | Abser | Raghib | Umar |
|---|---|---|---|
| W1 | Data downloads and requests, `envs/video`, tracker vendoring | Phase 4 arms and runners; tuning sweep (CPU) | Detector wrappers; model SHA checks |
| W2 | ChokePoint GT parsing; pipeline skeleton | Phase 4 eval, probes-only, per-bin, LFW; REPORT_phase4 | Detector size study; REPORT_phase5 (with Abser) |
| W3 | Pipeline, trackers, validation | Phase 3/4 thesis chapter | Embedding and index modules; single-frame baseline |
| W4–W5 | Aggregation runs, predicted tracks | Gate stage implementation | Aggregation protocol, metrics, CAFace |
| W6 | Face-aware association; MEVA (optional) | Gate experiment; REPORT_phase8 | FaceMoE (optional); FPS tables |
| W7 | Demo, end-to-end runs | Results tables and figures | Retrieval table; REPORT_phase9 |
| W8 | All: write-up, slides, freeze | | |

## Risks and fallbacks
- **YouTube Faces portal down:** VoxCeleb2 cropped tracks for the scale study. ChokePoint stays the surveillance benchmark.
- **FANVID, COX or SCface not arrived by W3:** future work. No schedule depends on them.
- **ChokePoint is small** (54 identities): CIs will be wide. Report them honestly; the low-resolution stress test and YTF add power.
- **Lab PC reboots:** every runner is resumable per sequence or chunk. Long jobs go to Colab using the Phase 3 chunked pattern.
- **`faiss-cpu` wheel missing on Windows:** exact NumPy search plus `hnswlib`.
- **CAFace needs a different backbone:** report it as its own row with its own single-frame baseline.
- **Licences:** BoT-SORT or another vendored module may have an incompatible licence. In that case use ByteTrack only and cite BoT-SORT.

## Verification
Each phase lists its acceptance checks above. Across all phases:
- **No-op and identity controls:**
  - γ = 1 and amount = 0 equal direct feed;
  - ground-truth boxes give IDF1 ≈ 1;
  - aggregating one frame equals the single-frame result;
  - video-path embeddings equal Phase 1 embeddings (cos > 0.9999).
- **Determinism:** rerun one sequence or chunk per phase and compare outputs byte for byte.
- **Report numbers:** every number in a report comes from a generated CSV, and tables are injected between markers, not typed.
- **Reports for review:** each REPORT_phaseN.md is reviewed at the next supervisor checkpoint before the phase it gates begins.
