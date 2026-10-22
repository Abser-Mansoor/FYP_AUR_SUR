# Phase 5 plan: video foundations (data, environment, detector study)

**Status:** approved 2026-10-22.

**Window:** W1–W2, 22 Oct – 8 Nov 2026. Lab PC work can start now.
**Who:**
- Claude on the lab PC: downloads, environment, code, smoke tests, merge, report.
- Abser: sends the emails and handles registrations, which Claude cannot do.
- Umar: runs the detector study on a Colab T4.

**Parent plan:** `research/Sprint_Notes/Sprint_2_implementation_plan.md`, Phase 5.

## Context
- The supervisor brief makes the video pipeline the main deliverable. Before any pipeline exists (Phase 6), we need four things:
  1. video data with identity labels;
  2. a video environment with trackers vendored;
  3. the extra model weights checked;
  4. evidence on **where face detection fails by face size**.
- Detection is the first stage, and Phase 1 already showed SCRFD finds only 31% of TinyFace distractors (median 18×14 px).
- The intended outcome is:
  - ChokePoint downloaded and parsed into ground truth;
  - a detector wrapper with tiling and upscaling modes;
  - a detector-by-size table that picks the Phase 6 detector with a rule fixed in advance;
  - the data inventory and access requests sent;
  - `REPORT_phase5.md`, then STOP.

**Standing rules:**
- inference only;
- seed 42;
- numbers come only from script outputs;
- failures are logged as results;
- datasets are read-only after their checksum;
- one venv per model repo;
- config-driven;
- Windows-native on the lab PC;
- one GPU job at a time on the lab PC.

**One GPU type per reported number:** every detector-study number comes from the Colab T4. Lab PC GPU runs are smoke tests and cross-checks only.

## 5a. Data (Claude on the lab PC; Abser for requests)

**ChokePoint** (Zenodo 815657, free for non-commercial research):
1. Query the Zenodo API for the file list, sizes and md5. Print them before downloading; the notes estimate about 12 GB, and D: has 1.5 TB free.
2. Download into `Dataset/ChokePoint/archives/`. Verify each md5, then extract into `Dataset/ChokePoint/`. Never edit the dataset after that.
3. Download the protocol and README pages from arma.sourceforge.net/chokepoint. Record the official G1/G2 gallery and probe split in `DATASETS.md` **before** any split is defined.
4. `scripts/video_chokepoint_gt.py` parses `groundtruth` (per frame, per person: left and right eye) into `outputs/video/gt/chokepoint_eyes.csv` (`seq, frame, person_id, lx, ly, rx, ry`).
   - **Acceptance:** the counts match the dataset page (48 sequences; 64,204 labelled faces). Any mismatch is reported as found, not fixed.
5. **Eye-to-box rule** (needed for detection recall and the Phase 6 track ground truth). It is fitted from data, not guessed:
   - run SCRFD on 300 seeded frames with large, high-confidence faces;
   - regress the box centre offset and box size against the inter-eye distance d (median ratios);
   - write the rule to `outputs/video/gt/eye_box_rule.json`;
   - check it visually on 20 overlay frames in `outputs/video/gt/overlays/`.
6. **Detector-study sample:** 1,000 frames, seeded and stratified over the 48 sequences, including the 2 crowded ones. They go to `outputs/video/detstudy/chokepoint_sample/` with their GT boxes.

**WIDER FACE val:**
- the val images (3,226) and `wider_face_split` annotations, from the official source or the CUHK Hugging Face mirror;
- record the SHA-256;
- extract to `Dataset/WIDER_FACE/`.

**Requests.** Claude drafts these in `docs/requests/`; Abser sends them or acts on them by hand:
- YouTube Faces portal registration (Abser, by hand);
- an email to the FANVID authors asking about the release date;
- a COX Face DB request (via vipl.ict.ac.cn release agreement);
- an SCface request (Prof. Grgic, signed agreement, all 3 members named);
- an IJB-S question to the supervisor;
- a consent form for the campus demo clips, plus a note asking whether university ethics approval is needed.

**`DATASETS.md`** (new, repo root) records for each dataset:
- source URL, licence, size, SHA-256 or md5, access date;
- protocol notes and status (held / requested / blocked).

## 5b. Environment and models (Claude on the lab PC)

**`envs/video`:**
- `--system-site-packages` on the same system Python 3.13 as `envs/diag`. It reuses the working torch 2.14+cu126 and onnxruntime-gpu 1.30, so SCRFD and the recognizers run unchanged.
- Add: `lap`, `pandas`, `scipy`, `faiss-cpu`.
  - If `faiss-cpu` has no Windows py3.13 wheel, fall back to exact NumPy search plus `hnswlib`, and log the reason.
  - `trackeval` is installed in Phase 6.

**Trackers.** Vendor only the tracker files, at the commit current on download day, with licence files. Record the commit hash and the SHA-256 of each file in `MODEL_INVENTORY.md`.
- ByteTrack (MIT): `yolox/tracker/{byte_tracker,basetrack,kalman_filter,matching}.py` go to `src/video/vendor/bytetrack/`.
- BoT-SORT: `tracker/{bot_sort,gmc,...}.py` go to `src/video/vendor/botsort/`. Check the licence first. If it is incompatible, keep ByteTrack only and cite BoT-SORT.
- Import smoke test: run both trackers on synthetic boxes (2 objects crossing) and check that the track IDs stay stable.

**Detector weights:**
- SCRFD `det_10g.onnx`: already present. Add its SHA-256 to `MODEL_INVENTORY.md`.
- RetinaFace R50: `biubug6/Pytorch_Retinaface` (MIT), `Resnet50_Final.pth`, downloaded with gdown from the README link. Vendor only `models/retinaface.py`, `models/net.py`, `layers/functions/prior_box.py`, `utils/box_utils.py`, `utils/nms/py_cpu_nms.py` and `data/config.py` into `src/video/vendor/retinaface/`.
- SFE-DETR: search GitHub and the paper for code. If none exists, record "no code, dropped" in `REPORT_phase5.md`.

**Extra models** (download and SHA-256 only; no runs in Phase 5):
- CAFace (`mk-minchul/CAFace`, MIT): both Google Drive checkpoints. Record which backbone it needs. If it needs AdaFace WebFace4M, download that backbone too, and record that the CAFace row will need its own single-frame baseline.
- FaceMoE (`Kartik-3004/FaceMoE`, MIT): Hugging Face `kartiknarayan/FaceMoE`. Record the input size (120 px) and the preprocessing from its code.

## 5c. Detector size study (code by Claude; T4 run by Umar)

**Code (new package `src/video/`):**
- **`detect.py`.** `Detector` interface: `detect(bgr) -> (dets[N,5], kps[N,5,2])` in original image coordinates.
  - **`SCRFDDetector`** wraps `src/align.py:SCRFD.detect` unchanged, with a configurable input size and threshold.
  - **`RetinaFaceDetector`** runs the vendored torch model (fp32, official decode and NMS settings from its `detect.py`: confidence 0.02 for scoring, NMS 0.4, top-k 5,000, keep 750).
  - **Modes** for each detector:
    - `full640`: letterboxed to 640 (SCRFD's published operating point);
    - `full1280`: 2× input size, the "upscaled frame" mode;
    - `tiled`: 2×2 tiles with 20% overlap, each at 640, mapped back and merged with `src/align.py:_nms` at 0.4.
- **`scripts/video_detector_study.py`.** It is chunked and resumable, and every chunk writes a CSV, following the `sr_colab_job.py` pattern.
  - **WIDER FACE val:**
    - recall at IoU 0.5 per face-height bin (<10, 10–16, 16–24, 24–32, 32–64, >64 px), at score thresholds 0.5 and 0.3;
    - false positives per image;
    - the official easy/medium/hard AP, using a vendored numpy port of the WIDER evaluation.
  - **ChokePoint sample:** recall per bin against the eye-rule GT boxes; false positives per frame.
  - **TinyFace/SurvFace eval crops** (22,898, from `phase4/subset_inputs.zip`): the Phase 1 padded-crop detection rate per native crop size. Reuse `src/align.py:detect_landmarks` for SCRFD and a matching path for RetinaFace.
  - **Runtime:** median ms per image at 720p (sampled ChokePoint frames upscaled to 1280×720), per detector and mode.
  - `src/stratify/measures.py` is not needed here.
- **`colab/build_phase5_notebook.py`** generates `colab/phase5_detector_colab.ipynb`, following the `colab/build_notebook.py` pattern:
  1. mount Drive and assert a T4;
  2. copy and unzip the bundle;
  3. `pip install onnxruntime-gpu` plus the minimal packages; record torch, onnxruntime, CUDA and driver versions in `device.json`;
  4. SHA-256 check of the weights;
  5. **cross-check:** SCRFD keypoints on 100 TinyFace probes against the lab PC values (max difference ≤ 0.5 px), written to `xcheck.json`; on failure, STOP;
  6. run the study chunks, writing results to `FYP_SR_SHARED/phase5/results/`. A rerun resumes.
- **`scripts/phase5_bundle.py`** (lab PC) writes to `FYP_SR_SHARED/phase5/`:
  - `code.zip` (`src/`, `scripts/video_detector_study.py`, `configs/video.yaml`);
  - `detectors.zip` (`det_10g.onnx`, `Resnet50_Final.pth`);
  - `wider_val.zip`;
  - `chokepoint_sample.zip`;
  - `xcheck.zip` (100 probes plus the lab keypoints);
  - `MANIFEST.json` (SHA-256 of each file).
  - The eval crops are reused from `phase4/subset_inputs.zip`.
- **`scripts/phase5_merge.py`** (lab PC) validates chunk coverage, then writes `outputs/video/detstudy/results/*.csv` and `detector_tables.md`.
- **`configs/video.yaml`** (new) holds paths, detector modes, thresholds, size bins, the chunk size and seed 42.

**Selection rule (fixed now, written in the report before the T4 run):**
- the Phase 6 detector and mode has the highest WIDER recall on the 16–32 px bins at threshold 0.5;
- among those with a median 720p runtime ≤ 100 ms per frame on the T4;
- ties (within 0.01 recall) go to the faster one.

## Order of work

| When | Who | Work |
|---|---|---|
| Now (22–23 Oct) | Claude | Zenodo file list → ChokePoint and WIDER val downloads in the background (resumable, md5/SHA checks); `envs/video`; tracker vendoring; weight downloads and SHA-256; request drafts |
| 24–27 Oct | Claude | `detect.py` + RetinaFace vendor; GT parser; eye-to-box rule; ChokePoint sample; detector-study script; **lab smoke tests** (below); bundle and notebook to Drive |
| 24–27 Oct | Abser | Send the requests; register for YTF |
| 28 Oct – 2 Nov | Umar | Colab notebook: cross-check, then full study (resumable) |
| 3–6 Nov | Claude | Merge; tables; `DATASETS.md`; `MODEL_INVENTORY.md`; `REPORT_phase5.md` |
| End of W2 | Team | **STOP:** supervisor checkpoint together with Phase 4 |

## Deliverable: `REPORT_phase5.md` (same style as REPORT_phase0–3)
1. Summary and the chosen detector.
2. Data inventory, licences, access status.
3. Environment, pins and vendored commits.
4. ChokePoint GT parse and eye-to-box rule (with overlays).
5. Detector-by-size tables: WIDER, ChokePoint, TinyFace/SurvFace crops.
6. Runtime.
7. Cross-device check.
8. SFE-DETR, CAFace and FaceMoE status.
9. Limitations.
10. Decisions for Phase 6.

## Verification
- **SCRFD wrapper equals Phase 1:** on the lab PC, the `SCRFDDetector` padded-crop path gives keypoints identical to `outputs/cache/phase1/landmarks_tinyface_probe.npz` for 100 seeded probes (max difference 0 on the same device).
- **WIDER eval sanity:** SCRFD `full640` easy/medium/hard AP lies within 1 point of the published SCRFD-10GF numbers (95.16 / 93.87 / 83.05). The comparison is made on the lab PC first (smoke, 300 images, indicative only), then in full on the T4. A larger gap is reported and investigated before the study is used.
- **Tiling sanity:** on an image with one large face, `tiled` returns one detection after merging, not duplicates.
- **Trackers:** the synthetic crossing test keeps IDs for both trackers.
- **ChokePoint:** label counts match the dataset page. The eye-to-box overlays are viewed (20 frames).
- **Cross-device:** T4 SCRFD keypoints are within 0.5 px of the lab PC's on 100 probes (`xcheck.json`).
- **Merge:** chunk coverage is exact (every image once) and the values are finite. A rerun of the merge gives byte-identical tables.

## After approval
- Save this plan as `research/Sprint_Notes/Phase5_video_foundations_plan.md`.
- Copy the Colab part, as a runbook for Umar, to `G:\My Drive\FYP_SR_SHARED\phase5\`.
