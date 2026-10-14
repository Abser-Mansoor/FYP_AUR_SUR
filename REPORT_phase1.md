# REPORT — Phase 1: direct-feed baseline (inference only)

Reproduce: `envs/diag/Scripts/python.exe scripts/run_baseline.py --config configs/baseline.yaml`
Config: `configs/baseline.yaml` · Results: `outputs/results/phase1/` (`baseline_results.csv/.json`, `sanity.csv/.json`, `per_sample/*.npz`) · Logs: `phase1/run.log`, `phase1_resume.log`

## 1. Summary
- All 4 recognizers (ArcFace, AdaFace, MagFace, CurricularFace) were evaluated zero-shot on TinyFace (closed-set identification) and QMUL-SurvFace (open-set identification and verification) with one resize-only preprocessing rule. All 16 result rows have status OK and all 4 sanity checks pass.
- Zero-shot accuracy is low: TinyFace Rank-1 is 24.5–27.5%, SurvFace TPIR20@FPIR=0.3 is 2.6–4.1%, SurvFace verification AUC is 0.66–0.69.
- These numbers are not an implementation bug. The evaluation code reproduces the official MATLAB metrics, the four models score 99.8–99.9% on high-resolution LFW (Phase 0 report, section 5.2), and the sanity checks pass.
- A diagnostic arm with 5-point alignment lifts TinyFace Rank-1 from about 25% to 67–70% for every model. Missing alignment is the main reason the resize-only baseline is weak on TinyFace. On SurvFace, where only 41–51% of images could be aligned, the gain is small (verification AUC +0.02 to +0.04, open-set TPIR unchanged). These arms are not the official baseline.
- Relative to the earlier (code-wiped) Phase 1 run, ArcFace reproduced the old log exactly. The earlier run's TinyFace Rank-1 range was 13–24% (per the Phase 0 post-mortem); the new range is 24.5–27.5%, a modest gain for the other models that I did not trace per model. What is new is the confirmed cause and the aligned diagnostic.

## 2. Protocol
| Item | Setting |
|---|---|
| Seed / determinism | 42; cuDNN deterministic, benchmark off, TF32 on |
| Preprocessing | resize crop to 112×112, bilinear (`cv2.INTER_LINEAR`); each model's official channel order and normalization; embedding = crop + horizontal flip, summed, L2-normalized |
| Batch size | 256 |
| Datasets | TinyFace (3728 probes, 4443 gallery matches, 153,428 distractors); SurvFace (60,294 gallery images / 3000 IDs, 60,423 mated and 121,736 unmated queries, verification pairs) |
| Metrics | TinyFace: mAP, Rank-1/5/10/20. SurvFace 1:N: TPIR20@FPIR 0.3/0.2/0.1, AUC. SurvFace 1:1: TAR@FAR 0.3/0.1/0.01/0.001, AUC, 10-fold accuracy |
| Confidence intervals | 95% bootstrap over per-sample scores, n=1000, seed 42 |
| Hardware | RTX 4000 Ada (20 GB), torch 2.14.0+cu126, Python 3.13.5 |

Per-model preprocessing (from the log):
| Model | Channel | Normalization | Weights SHA-256 (prefix) |
|---|---|---|---|
| ArcFace (iresnet100) | RGB | [-1,1] | a566a62357f0 |
| AdaFace (IR-101, MS1MV2) | BGR | [-1,1] | 4a26839460d8 |
| MagFace (iresnet100, MS1MV2) | BGR | [0,1] | cfeba792dada |
| CurricularFace (IR-101) | RGB | [-1,1] | 031ea0f7ba5f |

## 3. Results: official baseline (resize-only, direct feed)
### 3.1 TinyFace closed-set identification
| Model | mAP | Rank-1 [95% CI] | Rank-5 | Rank-10 | Rank-20 |
|---|---|---|---|---|---|
| ArcFace | 0.1755 | 0.2452 [0.232, 0.259] | 0.3055 | 0.3313 | 0.3589 |
| AdaFace | 0.2049 | 0.2749 [0.261, 0.289] | 0.3361 | 0.3637 | 0.3927 |
| MagFace | 0.1995 | 0.2744 [0.260, 0.288] | 0.3340 | 0.3616 | 0.3965 |
| CurricularFace | 0.1777 | 0.2524 [0.238, 0.266] | 0.3095 | 0.3326 | 0.3557 |

### 3.2 SurvFace open-set identification (1:N)
| Model | TPIR20@FPIR=0.3 [95% CI] | @0.2 | @0.1 | AUC |
|---|---|---|---|---|
| ArcFace | 0.0412 [0.039, 0.043] | 0.0261 | 0.0126 | 0.0800 |
| AdaFace | 0.0257 [0.024, 0.027] | 0.0131 | 0.0038 | 0.0636 |
| MagFace | 0.0392 [0.038, 0.041] | 0.0260 | 0.0136 | 0.0768 |
| CurricularFace | 0.0267 [0.025, 0.028] | 0.0165 | 0.0072 | 0.0597 |

### 3.3 SurvFace verification (1:1)
| Model | TAR@FAR=0.3 | 0.1 | 0.01 | 0.001 | AUC [95% CI] | Accuracy |
|---|---|---|---|---|---|---|
| ArcFace | 0.5549 | 0.3058 | 0.0825 | 0.0229 | 0.6790 [0.669, 0.689] | 0.6285 |
| AdaFace | 0.5361 | 0.2602 | 0.0483 | 0.0064 | 0.6625 [0.653, 0.673] | 0.6208 |
| MagFace | 0.5590 | 0.3246 | 0.0865 | 0.0256 | 0.6854 [0.676, 0.696] | 0.6340 |
| CurricularFace | 0.5374 | 0.2885 | 0.0756 | 0.0227 | 0.6667 [0.657, 0.677] | 0.6176 |

The four models differ by only a few points and many of the confidence intervals overlap. Do not rank them on these numbers alone. AdaFace and CurricularFace are lowest on SurvFace 1:N and at low FAR.

## 4. Diagnostic arm: 5-point aligned TinyFace (NOT the baseline)
SCRFD (`det_10g`) landmarks, similarity warp to the ArcFace 112×112 template, fallback to the plain resize when no face is found.

| Model | Rank-1 resize → aligned | mAP resize → aligned |
|---|---|---|
| ArcFace | 0.245 → 0.695 [0.680, 0.709] | 0.176 → 0.640 |
| AdaFace | 0.275 → 0.695 [0.678, 0.709] | 0.205 → 0.637 |
| MagFace | 0.274 → 0.675 [0.659, 0.689] | 0.200 → 0.616 |
| CurricularFace | 0.252 → 0.674 [0.659, 0.689] | 0.178 → 0.619 |

Detection rate: probes 94.7%, gallery matches 94.6%, **distractors 31.4%**. Most of the 153k distractors therefore fall back to plain resize, so the aligned gallery is a mix of aligned and unaligned images.

### 4.1 Why the distractor detection rate is low (checked, not a bug)
- On 300 random images per split, the CUDA and CPU detectors agreed on 100% of images, and both matched the cached rate (probe 0.950, gallery 0.953, distractor 0.327 on the sample). The provider is not the cause.
- The distractors are much smaller: median size 18×14 px, against 32×32 px for probes and gallery matches. SCRFD finds few faces at that size.
- Sweeping the upsample size (112–320 px) and detection threshold (0.5–0.1) on 400 distractors raised detection to at most 58% (320 px, threshold 0.1). That setting also accepts lower-confidence detections, so landmark quality is unverified. On probes the same sweep gave 79–98%. I kept the original setting (112 px, threshold 0.3) for all splits.
- Sensitivity of the aligned result to the mixed gallery (cached embeddings, Rank-1; probes and gallery matches aligned in all columns):

| Model | Distractors as reported (mixed) | Distractors left unaligned | Only the detected distractors | No distractors |
|---|---|---|---|---|
| ArcFace | 0.695 | 0.710 | 0.696 | 0.730 |
| AdaFace | 0.695 | 0.705 | 0.699 | 0.731 |
| MagFace | 0.675 | 0.697 | 0.678 | 0.718 |
| CurricularFace | 0.674 | 0.691 | 0.676 | 0.713 |

The aligned Rank-1 stays between 0.67 and 0.73 in every variant, against 0.25–0.27 with resize (0.28–0.31 with no distractors). The mixed gallery therefore moves the result by up to 2 points and does not affect the conclusion.

### 4.2 Aligned SurvFace (diagnostic, `scripts/run_aligned_survface.py`)
Same detector settings. Detection rates: gallery 40.5%, mated queries 42.5%, unmated queries 42.5%, verification 50.6%. About half of the images keep the plain-resize fallback, so this is a partial alignment. Results are in `outputs/results/phase1/aligned_survface_results.csv/.json` (the baseline tables are untouched).

| Model | Ver. AUC resize → aligned [95% CI] | TAR@FAR=0.1 | TPIR20@FPIR=0.3 [95% CI] | 1:N AUC |
|---|---|---|---|---|
| ArcFace | 0.679 → 0.720 [0.711, 0.731] | 0.306 → 0.357 | 0.0412 → 0.0408 [0.039, 0.042] | 0.080 → 0.096 |
| AdaFace | 0.663 → 0.703 [0.694, 0.713] | 0.260 → 0.323 | 0.0257 → 0.0284 [0.027, 0.030] | 0.064 → 0.083 |
| MagFace | 0.685 → 0.715 [0.705, 0.724] | 0.325 → 0.346 | 0.0392 → 0.0395 [0.038, 0.041] | 0.077 → 0.091 |
| CurricularFace | 0.667 → 0.689 [0.679, 0.699] | 0.289 → 0.314 | 0.0267 → 0.0293 [0.028, 0.031] | 0.060 → 0.075 |

Alignment gives a modest verification gain (AUC +0.02 to +0.04; the aligned and resize CIs touch or overlap for MagFace and CurricularFace) and almost no change in open-set TPIR20@FPIR=0.3 (+0.000 to +0.003). The 1:N AUC rises by 0.015 to 0.019. SurvFace stays hard even where alignment is applied, so its low zero-shot numbers are not mainly an alignment artefact.

## 5. Verification
| Check | Result |
|---|---|
| Sanity: identical image cosine ≥ 0.99999 | pass, all 4 |
| Sanity: determinism (max abs diff between two runs) | 0.0, all 4 |
| Sanity: shuffled-label TinyFace Rank-1 near chance | 0.0003–0.0008 vs chance 0.00002, all 4 |
| Sanity: shuffled-label SurvFace verification AUC near 0.5 | 0.494–0.498, all 4 |
| Sanity: self-compare (drift 1.0, paired delta 0) | pass, all 4 |
| Evaluation code vs official MATLAB logic | reproduced the old logged numbers exactly (Phase 0 report, section 5.1) |
| High-resolution control (LFW 10-fold, official preprocessing) | ArcFace 0.9985, AdaFace 0.9985, MagFace 0.9982, CurricularFace 0.9982 (Phase 0 report, section 5.2) |
| Result rows with status OK | 16 of 16 |

The shuffled-label check uses `shuffled_*` columns in `sanity.csv`. The LFW control and the old-log re-score were produced in Phase 0, not re-run here.

## 6. Interpretation
1. Zero-shot recognizers trained on aligned 112×112 faces perform poorly on tiny, unaligned crops. Alignment alone adds about 40 points of Rank-1 on TinyFace (diagnostic).
2. SurvFace is much harder than TinyFace. Partial alignment (41–51% of images) lifts verification AUC by 0.02–0.04 but leaves open-set TPIR20@FPIR=0.3 at 2.8–4.1%, which fits genuinely low-resolution surveillance images.
3. Open-set identification is the weakest result (TPIR20@FPIR=0.3 of 2.6–4.1%), so any Phase 2 improvement there will be easy to see.
4. On TinyFace direct feed, the "genuine" similarity is lower than the "impostor" similarity (negative d'). The impostor score is the best match among 150k+ gallery images, so it is biased high. Treat d' on this protocol as a descriptive statistic only.

## 7. Limitations and notes
- **Distractor detection rate of 31%** is explained (section 4.1): the distractors are tiny (median 18×14 px). The mixed gallery changes aligned Rank-1 by at most about 2 points.
- **Aligned arms are partial.** TinyFace aligns 95% of probes and gallery matches but 31% of distractors; SurvFace aligns 41–51% of images. Both use the plain-resize fallback elsewhere. Neither is a baseline.
- **Detector on GPU:** the aligned SurvFace log records `CUDAExecutionProvider`. TinyFace distractor landmarks were checked against CPU on a sample (100% agreement).
- **Small recomputation difference:** MagFace direct Rank-1 recomputed from cached embeddings is 0.2741 against 0.2744 in the table, which is one probe out of 3728. The other three models matched to four decimals.
- **Training-set confound:** ArcFace's training data is unknown (MS1MV3 or Glint360K); AdaFace and MagFace are MS1MV2. Cross-model differences cannot be attributed to the architecture.
- **Detector provider:** a CUDA 13 runtime was installed mid-run so the face detector could use the GPU. The TinyFace baseline log does not record the active provider, but the CPU/GPU comparison in section 4.1 shows identical detections, so results do not depend on it.
- **ArcFace direct numbers came from cached embeddings** (identical to the 10-06 log); the other three were computed fresh in this run.
- **Warning from torch/ORT:** torch uses CUDA 12.6 and onnxruntime-gpu uses CUDA 13. No failure was observed.

## 8. Status
Phase 1 (direct-feed baseline) is complete: 4 recognizers × 2 datasets, all 16 rows OK, all sanity checks pass. Both diagnostic arms (aligned TinyFace and aligned SurvFace) are complete, and the distractor detection rate is explained. No open anomalies remain; section 7 lists the known limitations.

**Stopping here for your approval before Phase 2.**
