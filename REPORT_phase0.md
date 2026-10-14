# REPORT — Phase 0: Reconnaissance + Phase 1 baseline post-mortem

Date: 2026-10-06. Project root: `D:/FYP_AUR_SUR` (the prompts and the old run's log say `D:/FYP`; that path does not exist).
All numbers below were produced by scripts in this repo. Sources are cited per section. Nothing was hand-entered.

---

## 1. System

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4000 Ada Generation, 20475 MiB VRAM, driver 596.71, CUDA (driver) 13.2, WDDM. About 2.4 GB in use by desktop apps at idle. |
| CPU | Intel Core i7-14700, 20 cores / 28 threads |
| RAM | 31.7 GB |
| Disk D: free | about 1591 GB |
| Python | 3.13.5 (the only installed version). Global site-packages has torch 2.14.0+cu126 (CUDA OK), numpy 2.3.5, scipy 1.17.1, opencv 4.13, scikit-learn 1.8, PyYAML |
| venvs | `envs/diag` (`--system-site-packages` + onnxruntime-gpu 1.30.0, for alignment) |
| git | 2.51.2.windows.1 (the project is not a git repo) |
| Network | github.com, pypi.org, figshare: reachable. **huggingface.co: NOT reachable** (TCP 443 fails). Any HF-only checkpoint needs a mirror. |

Risk: Python 3.13 is newer than most diffusion and SR repos support (often 3.8–3.11). Later phases will likely need an extra Python install for per-repo venvs.

## 2. Datasets (read-only)

### 2.1 QMUL-SurvFace — `Dataset/QMUL-SurvFace-v1/QMUL-SurvFace`
| Folder | Content |
|---|---|
| `training_set/` | 5319 identity folders, 220,888 images. File name format: `[PersonID]_[CameraID]_[ImageName].jpg` |
| `Face_Identification_Test_Set/gallery` | 60,294 images. `gallery_img_ID_pairs.mat` (`gallery_set`, `gallery_ids`) defines the order and IDs. **3000 unique gallery IDs**. (The readme says "from 5,319 test identities", but the .mat has 3000. The eval script comment confirms "only 3000 ids are in gallery".) |
| `Face_Identification_Test_Set/mated_probe` | 60,423 images, `mated_probe_img_ID_pairs.mat` |
| `Face_Identification_Test_Set/unmated_probe` | 121,736 images (no labels; distractor queries) |
| `Face_Verification_Test_Set/verification_images` | 10,051 images (+ one stray `.DS_Store`). `positive_pairs_names.mat` and `negative_pairs_names.mat` hold 5320×2 each, referencing 9,913 unique images |
| `Face_Identification_Evaluation/` | `test_open_set_face_identification.m`, `construct_template.m` |
| `Face_Verification_Evaluation/` | `test_face_verification.m` |

**Protocol (confirmed from the .m files):**
- *Open-set identification:*
  - Gallery = per-ID **mean-feature templates** (3000).
  - Queries = mated + unmated probes.
  - **Euclidean** distance.
  - FPIR thresholds come from the unmated top-1 distance (1001 steps). TPIR counts a mated hit if the true ID is in the **top-20** and its distance is below the threshold.
  - Report TPIR20@FPIR = 0.3 / 0.2 / 0.1 and AUC.
- *Verification:*
  - Euclidean distance on pairs, thresholds 0.2:0.001:2.0.
  - Report TAR@FAR = 0.3 / 0.1 / 0.01 / 0.001, AUC, and 10-fold mean accuracy.

### 2.2 TinyFace — `Dataset/tinyface/tinyface`
| Folder | Content |
|---|---|
| `Training_Set/` | 2570 identity folders, 7,804 images |
| `Testing_Set/Probe` | 3,728 images, `probe_img_ID_pairs.mat` |
| `Testing_Set/Gallery_Match` | 4,443 images, `gallery_match_img_ID_pairs.mat` |
| `Testing_Set/Gallery_Distractor` | 153,428 images |
| `Face_Identification_Evaluation/` | `test_face_identification.m`, `compute_AP.m` (+ copies of the two .mat files, and an empty `features/`) |

**Protocol (confirmed):** closed-set identification. Gallery = Gallery_Match + Distractors (157,871), Euclidean distance. Report CMC Rank-1/5/10/20 and mAP (trapezoid AP from `compute_AP.m`).

**Important:** TinyFace filename prefixes are **not** the IDs. 8151 of 8171 test images have a prefix that differs from the `.mat` ID. Always take labels from the `.mat` files.

Also present: `Dataset/QMUL-SurvFace-v1.zip`, `Dataset/tinyface.zip` (the original archives).

### 2.3 Integrity (`scripts/integrity.py` → `outputs/diagnostics/integrity_*.{csv,json}`)
| | SurvFace | TinyFace |
|---|---|---|
| Images | 473,392 `.jpg`, all 3-channel | 169,403 `.jpg`, all 3-channel |
| Corrupt / undecodable | **0** | **0** |
| Width px (min / 5% / 50% / 95% / max) | 3 / 9 / 22 / 30 / 106 | 2 / 6 / 15 / 32 / 119 |
| Height px (min / 5% / 50% / 95% / max) | 6 / 11 / 27 / 35 / 124 | 2 / 8 / 18 / 38 / 145 |
| Mean W×H | 20.0 × 24.5 | 16.8 × 20.4 |
| Byte-identical duplicate groups | 10,395 | 300 |

Duplicates broken down by split:
- **SurvFace:**
  - Most groups are verification images that also appear in the identification test set (4358 unmated, 3075 gallery, 2615 mated). This is expected: verification is a sampled subset of the test set.
  - **125 images appear in both `mated_probe` and `unmated_probe`** (protocol noise).
  - 87 unmated probes also appear in `training_set`.
  - 133 groups are within `training_set` (e.g. IDs 10324 and 10328 hold identical images).
  - None of this affects inference-only evaluation except the 125 mated/unmated overlaps.
- **TinyFace:**
  - **143 probes are byte-identical to a Gallery_Match image**, which is about 3.8% of probes and gives free Rank-1 hits.
  - This is part of the official protocol, so keep it, but note it when interpreting results.

## 3. Recognizer checkpoints (`models/weights/`)

| Recognizer | File | Backbone (verified by `strict=True` load) | Training data | Source |
|---|---|---|---|---|
| ArcFace | `arcface_iresnet100.pth` (sha256 a566a62357f0…) | insightface `iresnet100` | **UNKNOWN**. insightface publishes r100 ArcFace on MS1MV2, MS1MV3 and Glint360K at the same file size. **Please tell me where it came from.** | insightface `arcface_torch` model zoo |
| AdaFace | `adaface_ir101_ms1mv2.ckpt` (4a26839460d8…) | AdaFace `IR_101` | MS1MV2 (head has 85,742 classes = MS1MV2) | github.com/mk-minchul/AdaFace |
| MagFace | `magface_iresnet100_MS1MV2.pth` (cfeba792dada…) | MagFace `iresnet100` | MS1MV2 (sharded fc 8×10718 ≈ 85.7k classes) | github.com/IrvingMeng/MagFace |
| CurricularFace | `CurricularFace_Backbone.pth` (031ea0f7ba5f…) | CurricularFace `IR_101` (face.evoLVe style) | MS1MV2 per the repo README (not independently verifiable from the weights) | github.com/HuangYG123/CurricularFace |

- `official_adaface_ir101.ckpt` is a byte-identical duplicate of the AdaFace file and can be deleted.
- **Confounds:**
  - (a) Two backbone families: insightface iresnet100 (ArcFace, MagFace) vs IR_101 (AdaFace, CurricularFace).
  - (b) ArcFace training data is unknown and may not be MS1MV2.
  - (c) Each repo has its own input convention (below), so the "same preprocessing" can't be literally identical across models.

**Official input conventions** (from each repo's inference code; confirmed on LFW, §5.2):

| Model | Channel | Normalization |
|---|---|---|
| ArcFace | RGB | (x/255 − 0.5)/0.5 |
| AdaFace | BGR | (x/255 − 0.5)/0.5 |
| MagFace | BGR | **x/255 only** |
| CurricularFace | RGB | (x/255 − 0.5)/0.5 |

Backbone code is vendored unmodified in `src/models/vendor/` (see `SOURCES.md`). All four load with `strict=True`: 0 missing and 0 unexpected keys.

## 4. Proposed layout and Phase 1 convention

```
D:/FYP_AUR_SUR/
  Dataset/            read-only datasets
  data/lfw/           LFW (HR control)
  models/weights/     recognizer checkpoints;  models/detector/ SCRFD det_10g.onnx
  configs/*.yaml      all paths and parameters (no symlinks)
  src/                datasets.py, preprocess.py, align.py, eval_protocols.py, models/
  scripts/            entry points (diagnose.py, integrity.py, rescore_old_cache.py, ...)
  envs/               one venv per model repo
  outputs/{cache,results,diagnostics,figures}
```

**Phase 1 preprocessing convention (proposed, based on §5):**
- Direct resize to 112×112 (stretch, no pad), **bilinear**.
- Each model's **official** channel and normalization.
- **Flip-test fusion on** (sum of the embeddings of the image and its horizontal flip).
- Alignment: **decision needed** (see §5.5 and §6). Resize-only was the original convention, but it costs about 45 Rank-1 points.
- `num_workers=0`, seed 42.
- Cache keys must include channel, normalization, interpolation, flip and alignment.

## 5. Post-mortem of the earlier Phase 1 run (code wiped)

Old results (`outputs/results/baseline_results.csv`):

| Metric | Result |
|---|---|
| TinyFace Rank-1 | 0.13–0.24 |
| SurvFace TPIR20@FPIR=0.3 | 0.014–0.042 |
| SurvFace verification AUC | 0.66–0.70 |

### 5.1 The evaluation code was correct
`scripts/rescore_old_cache.py` re-scores the old cached embeddings with `src/eval_protocols.py`, a line-by-line port of the official .m files:
- **Identical** results for all Rank-k, TPIR and open-set AUC values (diff 0.000000).
- mAP is +0.0016 and verification mean accuracy is up to +0.0016. The old code did not use the official trapezoid AP or the threshold sweep. These are negligible.
- Name order, .mat label mapping and set sizes were all correct.

### 5.2 The models and loading are correct (HR control)
- **LFW**: 6000 pairs, 10-fold, aligned with SCRFD + the ArcFace template. Detection rate 99.95%.
- Results file: `outputs/diagnostics/lfw_control.csv`.

| Model (official preprocessing) | LFW accuracy |
|---|---|
| ArcFace | 0.9985 |
| AdaFace | 0.9985 |
| MagFace | 0.9982 |
| CurricularFace | 0.9982 |

- **MagFace with [-1,1] normalization drops to 0.889 (RGB) / 0.941 (BGR).** It is the only model sensitive to normalization.
- The others stay at 0.997–0.999 under any channel or normalization on HR images.

### 5.3 What the old run actually did (`scripts/fingerprint_old_cache.py`)
I re-embedded 64 TinyFace probes under candidate pipelines and compared them with the old cache:
- **ArcFace, AdaFace, CurricularFace:** the old cache is reproduced exactly (cosine 0.99996–1.00000) by **PIL bicubic resize + official channel/normalization + no flip**. Their old pipeline was correct, and their old numbers are valid resize-only results.
- **MagFace: the old embeddings are invalid.**
  - Cosine to every candidate pipeline is about 0.0006.
  - It is not a row permutation (0/3728 rows have a match anywhere).
  - It is not a random-init network, the MagFace weights in the ArcFace class, or a copy of another model's cache.
  - The weights file hash matches the one in the old log, so the old code built or loaded MagFace wrongly in some way that can't be recovered.
  - **Discard the old MagFace numbers.**

### 5.4 Preprocessing grid (TinyFace probes vs Gallery_Match only, no distractors; resize-only)
`outputs/diagnostics/tinyface_grid.csv`: 96 runs (4 models × channel × normalization × interpolation × flip).

Rank-1 values, all resize-only:

| Model | Old run (from cache) | Official, bicubic, no flip | **Official, best interp + flip** | Worst config |
|---|---|---|---|---|
| ArcFace | 0.2538 | 0.2479 | **0.2841** (bilinear) | 0.2122 |
| AdaFace | 0.2878 | 0.2902 | **0.3160** (bicubic) | 0.1958 |
| MagFace | 0.1634 (invalid) | 0.2758 | **0.3085** (bilinear) | 0.1792 |
| CurricularFace | 0.2457 | 0.2468 | **0.2910** (bilinear) | 0.1835 |

Findings:
- The official channel and normalization is the best setting for every model.
- Flip fusion adds +2.6 to +4.4 points. Bilinear vs bicubic is within about ±1 point.
- With correct settings, resize-only tops out at about **0.28–0.32 Rank-1 against the match-only gallery**.
- So for the three valid models, preprocessing settings explain only a few points. The main remaining gap is alignment (§5.5) plus genuine low resolution.

### 5.5 Alignment ablation (diagnostic only)
`outputs/diagnostics/tinyface_align.csv`. Same subset and metric as §5.4. Settings: official channel/normalization, bicubic.

**How the alignment works:**
- SCRFD `det_10g` finds 5 landmarks on each crop. The crop is first upsampled to 112 px on its long side and padded by 50%.
- A Umeyama similarity transform then maps those landmarks to the ArcFace 112×112 template.
- **Detection succeeded on 94.65%** of the 8171 images. The other 5.35% fall back to plain resize.

Rank-1 (mAP in brackets):

| Model | Resize | Resize + flip | **Aligned** | **Aligned + flip** |
|---|---|---|---|---|
| ArcFace | 0.2479 (0.179) | 0.2774 (0.206) | 0.7213 (0.669) | **0.7302 (0.679)** |
| AdaFace | 0.2902 (0.221) | 0.3160 (0.245) | 0.7264 (0.671) | **0.7312 (0.679)** |
| MagFace | 0.2758 (0.205) | 0.3002 (0.226) | 0.7082 (0.650) | **0.7167 (0.662)** |
| CurricularFace | 0.2468 (0.183) | 0.2817 (0.210) | 0.6998 (0.648) | **0.7108 (0.661)** |

- **Alignment alone adds +42 to +47 Rank-1 points for every model. It is by far the largest factor.**
- After alignment the four models are within 2 points of each other.
- These numbers use the match-only gallery. The full protocol, with 153k distractors, will be lower; the old run lost about 4 points to distractors. Published TinyFace results for aligned MS1MV2 IR-101 models are around 0.6–0.7 Rank-1, which is the same range. I haven't verified exact paper values here.
- Caveat: with failures falling back to resize, aligned and unaligned crops are mixed for 5% of images.

### 5.6 Conclusion: causes of the poor results
1. **MagFace: a real bug.** The old embeddings don't come from the MagFace network with its weights. The old MagFace numbers must be discarded and rerun.
2. **All models: no face alignment (dominant cause, about −45 Rank-1 points).**
   - The recognizers were trained on 5-point-aligned crops. Stretch-resizing loose LR bounding boxes puts the eyes and mouth in the wrong places.
   - This is a consequence of the resize-only protocol choice, not a code bug. But it means the resize-only baseline sits about 45 points below what the same models reach with standard alignment.
   - Missing flip fusion costs another 1–4 points.
3. **The datasets themselves:** median face size is 15×18 px (TinyFace) and 22×27 px (SurvFace). Even aligned, there is a gap to HR performance; LFW shows the models are fine at HR.
   - SurvFace has not been re-run aligned yet. Its old numbers (TPIR20 ≈ 0.02–0.04) are expected to rise substantially but stay low.
4. **Not causes:** the evaluation code, ID mapping, checkpoint loading, and channel/normalization for ArcFace, AdaFace and CurricularFace.

## 6. Open questions for you
1. Where did `arcface_iresnet100.pth` come from (MS1MV2, MS1MV3 or Glint360K)? This decides whether ArcFace is comparable to the three MS1MV2 models.
2. **Decision on alignment, given §5.5:**
   - (a) Keep resize-only as the official baseline. Accept that it sits about 45 points below the aligned result. Report aligned as a diagnostic.
   - (b) Make aligned + flip the baseline.
   - (c) Report both as two baselines. **Recommended:** later SR, deblur and low-light arms can be measured against both, and alignment interacts with restoration because SR output may make detection easier.
3. If alignment is used: should the 5.35% detection failures fall back to resize (current), or be excluded? Exclusion breaks the official protocol, so I recommend fallback and logging the rate.
4. Next run: the full official protocols (TinyFace with distractors, SurvFace open-set and verification) for all 4 models, with the corrected pipeline, resize and aligned. Estimate: about 1.5–2 h of GPU time, mostly SurvFace (250k images). Detection on SurvFace adds roughly 15–20 min.

**STOP: awaiting approval before Phase 1.**
