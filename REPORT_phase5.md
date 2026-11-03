# REPORT — Phase 5: video foundations (data, environment, detector size study)

**Status:** complete, pending supervisor approval (end-of-W2 checkpoint, together with Phase 4). The detector selection rule was fixed in `configs/video.yaml` before the first study run (see section 1 and section 5.1).

## 1. Summary and chosen detector
**Chosen for Phase 6: SCRFD `det_10g`, full frame upscaled to 1280 px (`scrfd__full1280`), with the official WIDER preprocessing** (`SCRFDDetector(preprocess="official")`, NMS 0.45; the settings are in `configs/video.yaml`).

- **Selection rule (fixed before the run):** highest pooled recall on the 16-24 and 24-32 px bins at score 0.5 among the configurations whose median 720p runtime on the T4 is at most 100 ms; a difference of 0.01 or less in recall counts as a tie and goes to the faster configuration. Source: `selection.json`.
- **Result:** `scrfd__full1280` has pooled recall 0.763 at 88.3 ms. `scrfd__tiled` has 0.761 at 96.0 ms, which is inside the tie margin, so the faster one wins. RetinaFace `full1280` has the higher recall (0.792) but runs at 105.8 ms, over the cap, and RetinaFace `tiled` runs at 122.1 ms. `scrfd__full640` and RetinaFace `full640` are fast (24.8 ms and 30.4 ms) but find too few small faces (0.630 and 0.467).
- **False positives of the chosen configuration at score 0.5:** 0.307 per image on WIDER FACE val and 0.102 per frame on the ChokePoint sample (`wider_bins.csv`, `chokepoint_bins.csv`). The Sprint 2 plan asked for at most 0.1 per frame; the ChokePoint value is just above that. Section 9 explains why the runtime rule was applied.
- **Detection of small faces is the weak point of every configuration.** For the chosen one on WIDER val, recall is 0.20 for faces under 10 px, 0.55 for 10-16 px, 0.73 for 16-24 px and 0.82 for 24-32 px. Upscaling to 1280 raises small-face recall but lowers recall on faces above 64 px (0.903, against 0.944 at 640 px).
- **This report describes the second run of the study.** The first run (Colab T4, 2026-10-24) failed the WIDER AP sanity gate: SCRFD `full640` hard AP was 81.28 against the published 82.80 for this model. The cause was the preprocessing in our detector wrapper, not the evaluator (section 7). The study was re-run on a Colab T4 with the official preprocessing and the gate passes (82.42 on hard). Both runs choose the same detector. The first run's tables are kept in `outputs/video/detstudy/results/` as a record and are not used for any claim here.

## 2. Data inventory, licences, access
Full per-dataset table: `DATASETS.md`. Summary:

- **Held:** ChokePoint (downloaded and extracted 2026-10-22, 25 files, 11.71 GB, per-file md5 checked; CC BY-NC 4.0 on Zenodo, non-commercial research terms on the NICTA page), WIDER FACE val (3,226 images, 2026-10-22), TinyFace and QMUL-SurvFace (Phase 0).
- **Requests:** on 2026-11-03 Abser reported that the dataset access requests have been sent, by him and by the supervisor, and that responses are awaited. The drafts are in `docs/requests/` (YouTube Faces registration, FANVID authors, COX Face DB, SCface, IJB-S question to the supervisor, campus clip consent). **Which individual requests were sent was not stated, so the per-dataset statuses in `DATASETS.md` are unchanged.** BRIAR is blocked (no academic route found).
- **Licence handling:** the GitHub repository is public and ChokePoint is non-commercial, so no ChokePoint frame, overlay or eye-label file is committed (`.gitignore`).
- **ChokePoint protocol:** found on the dataset page and recorded in `DATASETS.md` (section "ChokePoint protocol"). It is a pair-verification protocol with groups G1 and G2 of eight sequences each. It does not define a tracking or identification split. The Zenodo `README.txt` has no protocol section.

## 3. Environment, pins, vendored commits
- **Lab PC (`envs/video`):** Python 3.13.5, torch 2.14.0+cu126, numpy 2.5.4, onnxruntime-gpu 1.30.0, opencv 4.13.0, scipy 1.18.1, pandas 3.0.6, lap 0.5.13, faiss-cpu 1.15.1, gdown 6.4.2. `torch` is imported before onnxruntime creates a CUDA session, because loading the CUDA 13 cuDNN first breaks torch's CUDA 12 cuDNN. Source: `MODEL_INVENTORY.md` section 4.
- **Colab T4 (second run):** Tesla T4, torch 2.11.0+cu130, CUDA 13.0, onnxruntime 1.31.1, Python 3.13.16 (`outputs/video/detstudy_v2/results/device.json`). The first run's T4 had onnxruntime 1.31.0 (`outputs/video/detstudy/results/device.json`).
- **Vendored code** (`src/video/vendor/VENDOR.md`, with upstream SHA-256 and the patches): ByteTrack at commit d1bf0191adff59bc8fcfeaa0b33d3d1642552a99, BoT-SORT at 251985436d6712aaf682aaaf5f71edb4987224bd (re-ID off), RetinaFace (biubug6/Pytorch_Retinaface) at b984b4b775b2c4dced95c1eadd195a5c7d32a60b. All MIT. The patches are import changes, `np.float` replaced by `float`, and a NumPy replacement for `cython_bbox`.
- **Weights:** SCRFD `det_10g.onnx` SHA-256 5838f7fe...5b91 and RetinaFace `Resnet50_Final.pth` SHA-256 6d1de9c2...a16d (full values in `MODEL_INVENTORY.md`). The Colab notebook checks both against the lab-PC hashes before running.
- **Tracker smoke test** (`outputs/video/tracker_smoke.json`): two synthetic objects crossing, 0 identity switches for ByteTrack and for BoT-SORT.
- **Tiling check** (`outputs/video/verify_tiling.json`): on an image with one large face, every configuration returns one detection after merging.

## 4. ChokePoint ground truth and the eye-to-box rule
Source: `outputs/video/gt/chokepoint_gt_summary.json`, `outputs/video/gt/eye_box_rule.json`. Counts are reported as found and were not adjusted to match the dataset page.

| Quantity | Parsed | Dataset page |
|---|---|---|
| Annotation files (clips) | 72 | 48 sequences |
| Labelled person-frame entries | 72,289 | 64,204 face images |
| Distinct identities | 29 | 25 (portal 1) + 29 (portal 2) |
| Entries missing an eye (skipped) | 2 (both in P2E_S3_C1.1) | - |

- The P2E_S5 and P2L_S5 clips (the crowded ones) have no ground-truth file, so they are not used for recall.
- Median inter-eye distance over all entries is 27.0 px (25th to 75th percentile 20.0 to 36.9 px).
- **Eye-to-box rule** (the ground-truth boxes for recall are produced by it): box centre at the eye midpoint plus (-0.026, +0.259) times the inter-eye distance d, box width 2.31 d and height 3.00 d. It was fitted on SCRFD detections with score 0.9 or higher, matched to eye labels, and **only 12 pairs matched** (`n_pairs` in the json). The interquartile range of the offsets is 0.22 d horizontally and 0.07 d vertically.
- **Overlay check.** The plan asks Umar to open five overlays and say whether the boxes sit on the faces. **That answer has not been given yet.** Claude opened five overlays from `outputs/video/gt/overlays/` (files 00, 05, 10, 15 and 19) on 2026-11-03. In all five the red eye points sit on the eyes. In four the green box covers the face. In file 19 (`P2E_S4_C1.1_00003114`) the person's head is turned and the box covers only the right half of the face, with the left cheek outside it. One image in five is a small sample, but it shows that a fixed offset from the eye midpoint can misplace the box on turned heads. Umar's own check is still open (section 10).

## 5. Detector-by-size tables
The tables are pasted from `outputs/video/detstudy_v2/results/detector_tables.md` (second run, one Colab T4). Face-height bins are in pixels. `nan` means the sample has no faces in that bin.

<!-- TABLES:BEGIN -->
### WIDER FACE val: official AP

| config | easy | medium | hard | n_images | n_errors |
|---|---|---|---|---|---|
| scrfd__full640 | 0.950 | 0.936 | 0.824 | 3226 | 0 |
| scrfd__full1280 | 0.920 | 0.925 | 0.892 | 3226 | 0 |
| scrfd__tiled | 0.933 | 0.931 | 0.886 | 3226 | 0 |
| retinaface__full640 | 0.944 | 0.910 | 0.665 | 3226 | 0 |
| retinaface__full1280 | 0.955 | 0.943 | 0.872 | 3226 | 0 |
| retinaface__tiled | 0.943 | 0.932 | 0.840 | 3226 | 0 |

### WIDER FACE val: recall by face height, score >= 0.5

| config | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | FP/image |
|---|---|---|---|---|---|---|---|
| retinaface__full1280 | 0.139 | 0.556 | 0.754 | 0.851 | 0.911 | 0.962 | 0.750 |
| retinaface__full640 | 0.000 | 0.036 | 0.358 | 0.641 | 0.834 | 0.947 | 0.377 |
| retinaface__tiled | 0.054 | 0.448 | 0.713 | 0.821 | 0.905 | 0.958 | 0.824 |
| scrfd__full1280 | 0.202 | 0.554 | 0.727 | 0.820 | 0.876 | 0.903 | 0.307 |
| scrfd__full640 | 0.015 | 0.199 | 0.556 | 0.747 | 0.864 | 0.944 | 0.203 |
| scrfd__tiled | 0.156 | 0.524 | 0.723 | 0.821 | 0.881 | 0.923 | 0.341 |

### WIDER FACE val: recall by face height, score >= 0.3

| config | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | FP/image |
|---|---|---|---|---|---|---|---|
| retinaface__full1280 | 0.210 | 0.642 | 0.802 | 0.880 | 0.925 | 0.967 | 1.416 |
| retinaface__full640 | 0.000 | 0.077 | 0.448 | 0.703 | 0.861 | 0.956 | 0.791 |
| retinaface__tiled | 0.104 | 0.549 | 0.768 | 0.858 | 0.923 | 0.965 | 1.564 |
| scrfd__full1280 | 0.467 | 0.757 | 0.841 | 0.897 | 0.924 | 0.931 | 1.891 |
| scrfd__full640 | 0.087 | 0.489 | 0.753 | 0.857 | 0.915 | 0.964 | 1.444 |
| scrfd__tiled | 0.395 | 0.740 | 0.841 | 0.897 | 0.925 | 0.951 | 2.131 |

### ChokePoint sample: recall by face height, score >= 0.5

| config | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | FP/frame |
|---|---|---|---|---|---|---|---|
| retinaface__full1280 | nan | nan | nan | nan | 1.000 | 0.981 | 0.115 |
| retinaface__full640 | nan | nan | nan | nan | 1.000 | 0.988 | 0.094 |
| retinaface__tiled | nan | nan | nan | nan | 0.997 | 0.973 | 0.140 |
| scrfd__full1280 | nan | nan | nan | nan | 1.000 | 0.978 | 0.102 |
| scrfd__full640 | nan | nan | nan | nan | 1.000 | 0.979 | 0.093 |
| scrfd__tiled | nan | nan | nan | nan | 1.000 | 0.984 | 0.097 |

### ChokePoint sample: recall by face height, score >= 0.3

| config | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | FP/frame |
|---|---|---|---|---|---|---|---|
| retinaface__full1280 | nan | nan | nan | nan | 1.000 | 0.981 | 0.130 |
| retinaface__full640 | nan | nan | nan | nan | 1.000 | 0.988 | 0.099 |
| retinaface__tiled | nan | nan | nan | nan | 0.997 | 0.973 | 0.171 |
| scrfd__full1280 | nan | nan | nan | nan | 1.000 | 0.978 | 0.133 |
| scrfd__full640 | nan | nan | nan | nan | 1.000 | 0.979 | 0.111 |
| scrfd__tiled | nan | nan | nan | nan | 1.000 | 0.984 | 0.135 |

### TinyFace / SurvFace crops: detection rate by native crop size (longest side, px)

| detector | dataset | all | <16 | 16-24 | 24-32 | 32-48 | >48 |
|---|---|---|---|---|---|---|---|
| scrfd | tinyface | 0.561 | 0.100 | 0.381 | 0.563 | 0.909 | 0.522 |
| scrfd | survface | 0.425 | 0.256 | 0.326 | 0.496 | 0.579 | 0.715 |
| retinaface | tinyface | 0.589 | 0.138 | 0.421 | 0.586 | 0.930 | 0.413 |
| retinaface | survface | 0.435 | 0.308 | 0.389 | 0.486 | 0.531 | 0.676 |

### Runtime (median ms per 720p frame, batch 1)

| config | median_ms | p95_ms | n |
|---|---|---|---|
| scrfd__full640 | 24.787 | 26.648 | 100 |
| scrfd__full1280 | 88.296 | 93.345 | 100 |
| scrfd__tiled | 96.015 | 108.602 | 100 |
| retinaface__full640 | 30.394 | 31.226 | 100 |
| retinaface__full1280 | 105.764 | 109.888 | 100 |
| retinaface__tiled | 122.075 | 126.945 | 100 |

### Selected detector (Phase 6)

```

{
 "config": "scrfd__full1280",
 "recall": 0.7629861421546714,
 "median_ms": 88.29566050007998,
 "candidates": [
  {
   "config": "scrfd__full640",
   "recall": 0.629682610639249,
   "median_ms": 24.78697549986464
  },
  {
   "config": "scrfd__full1280",
   "recall": 0.7629861421546714,
   "median_ms": 88.29566050007998
  },
  {
   "config": "scrfd__tiled",
   "recall": 0.7606616003576219,
   "median_ms": 96.01534500052367
  },
  {
   "config": "retinaface__full640",
   "recall": 0.467411712114439,
   "median_ms": 30.3943909998452
  },
  {
   "config": "retinaface__full1280",
   "recall": 0.7916852928028609,
   "median_ms": 105.7636294999611
  },
  {
   "config": "retinaface__tiled",
   "recall": 0.7546714349575324,
   "median_ms": 122.07454650024374
  }
 ],
 "rule": {
  "score": 0.5,
  "bins": [
   "16-24",
   "24-32"
  ],
  "max_ms": 100,
  "tie_recall": 0.01
 }
}

```
<!-- TABLES:END -->

### 5.1 Reading the tables
- **WIDER FACE sanity gate (SCRFD `full640`, official AP).** The reference is the SCRFD-10G-KPS row of the insightface README (95.40 / 94.01 / 82.80), because `det_10g.onnx` has keypoints. Ours is 95.03 / 93.64 / 82.42, which is -0.37, -0.37 and -0.38 points (limit: 1.0 point). The gate passes. The first run, with our original preprocessing, was 94.85 / 93.46 / 81.28, a hard gap of -1.52, which failed (section 7).
- **Small faces.** On WIDER, `full1280` and `tiled` find far more faces under 24 px than `full640`. The two are close: pooled recall on 16-24 and 24-32 px is 0.763 for SCRFD `full1280` and 0.761 for SCRFD `tiled`. At score 0.3 SCRFD finds 0.47 of the faces under 10 px at `full1280`, at the price of 1.89 false positives per image.
- **RetinaFace.** At 640 px it finds almost nothing below 16 px (recall 0.000 below 10 px and 0.036 for 10-16 px). At 1280 px it has the highest recall of all configurations for 16-24 px and 24-32 px, but it exceeds the runtime cap.
- **Large faces.** `full1280` loses recall on faces above 64 px compared with `full640` (SCRFD: 0.903 against 0.944).
- **ChokePoint sample.** Every ground-truth face in the sample is 32 px or larger, so the bins below 32 px are empty and recall is 0.97 to 1.00 for every configuration. The sample cannot tell the detectors apart. The decision rests on WIDER FACE.
- **TinyFace and SurvFace crops.** This is the Phase 1 padded-crop protocol, run on the crops at their native size. Detection rates are low: 0.56 (SCRFD) and 0.59 (RetinaFace) on TinyFace, 0.43 and 0.44 on SurvFace. Rates below 16 px are 0.10 and 0.14 on TinyFace. Bins with few crops (above 48 px: 46 TinyFace and 179 SurvFace crops) are not reliable. These rates are not comparable with frame-level recall.

## 6. Runtime
Median milliseconds per 1280x720 frame, batch 1, after 10 warm-up frames, 100 frames from the ChokePoint sample resized to 1280x720, on the Colab T4 (table in section 5, source `runtime.csv`). SCRFD: 24.8 ms at `full640`, 88.3 ms at `full1280`, 96.0 ms at `tiled`. RetinaFace: 30.4 ms, 105.8 ms and 122.1 ms. The 100 ms cap excludes RetinaFace `full1280` and `tiled`. Timings depend on the Colab session; they were measured once.

## 7. Cross-device check and how the AP gate was resolved
- **Cross-device check** (`results/xcheck.json`): SCRFD padded-crop keypoints on 100 seeded TinyFace probes, T4 against the lab PC. 90 detected on both, 0 detection mismatches, maximum difference 0.0031 px (limit 0.5 px). The result is identical for both runs. This check uses the unchanged Phase 1 SCRFD path, not the study's official-preprocessing path.
- **Wrapper check** (`outputs/video/verify_scrfd.json`): the study wrapper equals `src/align.py:detect_landmarks` exactly (0.0 px), and equals the Phase 1 cache within 0.005 px, on 100 probes (0 mismatches).
- **How the first AP failure was traced** (files in `outputs/video/detstudy/_diag/`; none of these runs is a reported study number):
  1. The reference row was wrong: 95.16 / 93.87 / 83.05 belongs to the model without keypoints. Our weights are the KPS model (95.40 / 94.01 / 82.80), so the first hard gap was -1.52, not -1.77, but still over the limit.
  2. The score floor is not the cause: on the lab-PC CPU, with our original settings, a floor of 0.001 gives the same AP as 0.02 (hard 0.8128), and it equals the T4 result (`lowthr_study.json`, `thr_sensitivity.json`).
  3. The evaluator is not the cause: the unmodified reference evaluation script of the WIDER FACE Python port (wondervictor/WiderFace-Evaluation), run on our stored detections, gives exactly our values (easy 0.9485, medium 0.9346, hard 0.8128; `refeval_results.txt`). It was also run on the official-preprocessing detections and agrees with our evaluator (hard 0.8242). Its compiled IoU function was replaced by a NumPy one with the same +1 pixel convention as the official MATLAB `boxoverlap.m` (`refeval_shim_bbox.py`).
  4. The preprocessing is the cause: our wrapper padded with black before normalising, truncated the resized size and used NMS 0.4, while the official script normalises before padding (mid-grey), rounds the size and uses NMS 0.45. With the official settings on all 3,226 images, on the CPU, hard AP rises from 0.8128 to 0.8242 (+1.14 points; `lowthr_official.json`). The three changes were made together, so their separate effects are not known.
  5. The second T4 run agrees with the lab-PC CPU on the same settings: hard AP 0.82419 (T4) and 0.82424 (CPU, `lowthr_official.json`).
- **Change from the first run to the second** (`wider_ap.csv` of both runs; points): SCRFD `full640` +0.18 / +0.17 / +1.14 (easy / medium / hard), `full1280` +0.17 / +0.23 / +0.45, `tiled` -0.02 / -0.01 / +0.26. RetinaFace is unchanged (0.00 at two decimals in every cell), which is also a consistency check between the two Colab sessions.
- **Determinism:** the merge was run twice on the second run's chunks and all 7 result files have identical SHA-256 (`outputs/video/detstudy_v2/merge_hashes_run1.txt` and `run2.txt`); the same was true for the first run.

## 8. SFE-DETR, CAFace and FaceMoE
- **SFE-DETR:** dropped. The paper gives no code link ("data ... available from the corresponding author upon reasonable request") and the model is trained on SCUT-HEAD, which would need training.
- **CAFace:** both checkpoints and the AdaFace WebFace4M backbone are downloaded, and every file's SHA-256 matches `outputs/video/model_downloads.json` (values in `MODEL_INVENTORY.md` section 4). CAFace is trained on an AdaFace R100 WebFace4M backbone, not on our AdaFace, so its Phase 7 row needs its own single-frame baseline with that backbone. Nothing was run with these weights in Phase 5.
- **FaceMoE:** two checkpoints downloaded and hashed (`swin4m_exp_3_k_2_briar_full`, `swin4m_exp_3_k_2_tinyface_full`). Input size is 120 px and it needs its own preprocessing. It stays an optional fifth recognizer.

## 9. Limitations
- **Selection rule difference.** The Sprint 2 plan said "highest recall on 16-32 px at no more than 0.1 false positives per frame". The Phase 5 plan and `configs/video.yaml` use "median 720p runtime of at most 100 ms on the T4". The config rule was fixed before the run and is the one applied. The chosen configuration has 0.102 false positives per frame on the ChokePoint sample and 0.307 per image on WIDER val at score 0.5.
- **The study was run twice.** The first run's SCRFD results used a preprocessing that differs from the official one (section 7) and fail the AP gate. They are superseded. The second run changed only the SCRFD settings; RetinaFace and the tile-merge NMS (0.4) were not changed, so in `tiled` mode SCRFD uses NMS 0.45 inside each tile and 0.4 when merging. Tiling is our own addition, not part of the official protocol. The first run used one Colab T4 session and the second another; the numbers in this report all come from the second.
- **ChokePoint counts differ from the dataset page:** 72 annotation files against "48 sequences", 72,289 entries against 64,204 face images, 29 identities against 25 + 29. They are reported as found. The P2E_S5 and P2L_S5 clips have no ground truth.
- **The ChokePoint "ground truth" boxes come from a fitted eye-to-box rule, not from hand-drawn boxes.** The rule was fitted on SCRFD detections with only 12 matched pairs, so the ChokePoint recall of SCRFD is partly circular and should not be used to rank the two detectors. The five-overlay check found one box misplaced on a turned head.
- **The ChokePoint sample has no face under 32 px,** so it says nothing about small faces. WIDER FACE carries the detector choice.
- **The cross-device check covers SCRFD only,** and only the unchanged Phase 1 path. RetinaFace numbers are T4-only, so they are internally consistent. The agreement of the T4 with the lab-PC CPU on the official-preprocessing path rests on the AP values (0.82419 against 0.82424), not on a keypoint comparison.
- **The crop detection rate uses the Phase 1 padded-crop protocol** and is not comparable to frame-level recall.
- **Runtime** is one measurement of 100 frames on a shared Colab T4, so it can vary between sessions.
- **The Phase 6 detector is chosen on WIDER FACE,** which has different image types from surveillance video. Recall on real surveillance footage is not measured in this phase.

## 10. Decisions for Phase 6
- **Detector:** SCRFD `full1280` with `preprocess="official"` and NMS 0.45 (`configs/video.yaml`), if the supervisor agrees. `scrfd__tiled` is the alternative within the tie margin at 96.0 ms.
- **ChokePoint protocol:** recorded in `DATASETS.md`. It is a verification protocol (G1 and G2). Whether to use it or to define a tracking-specific evaluation is open.
- **YouTube Faces:** no registration confirmed. Fallback to VoxCeleb2 cropped face tracks if there is no access by 8 Nov. Abser reports that requests were sent on 2026-11-03; which ones needs to be confirmed.
- **CAFace** needs its own AdaFace WebFace4M single-frame baseline.
- **FaceMoE** needs 120 px preprocessing.
- **SFE-DETR** is dropped (no code).
- **Umar's overlay check** (section 4) is still to be recorded.

**STOP: awaiting supervisor approval (together with Phase 4).**
