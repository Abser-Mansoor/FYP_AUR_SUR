# REPORT — Phase 2: stratification (Task 4)

Reproduce: `envs/diag/Scripts/python.exe scripts/run_stratify.py --config configs/stratify.yaml`
Code: `src/stratify/` (`measures.py`, `measure_split.py`, `binning.py`, `sheets.py`, `analysis.py`), `scripts/run_stratify.py`, `configs/stratify.yaml`
Outputs: `outputs/stratify/` — `bins/*.csv` (per-crop bin assignments, used by all later tasks), `measures/*.csv`, `thresholds.json`, `contact_sheets/*.png` (50), `analysis/*` (CSVs, `dark_flags.json`, trend plots), `run.log`

## 1. Summary
- All 413,965 crops in 7 splits (TinyFace probe, gallery_match, distractor; SurvFace gallery, mated_probe, unmated_probe, verification) were measured and binned. No image was unreadable. Rerunning the binning step gives byte-identical CSVs.
- **Dark bin (luminance):** large enough everywhere (745 to 30,685 crops per split), and accuracy is lower than a size-matched random subset for all 4 recognizers on all 3 protocols. **One flag:** the TinyFace dark bin reaches luminance 83/255, so it is "dimmer than average", not truly dark (section 6).
- **Low-sharpness bin (Laplacian variance):**
  - **TinyFace: valid.** Rank-1 drops from 0.29–0.35 (sharp) to 0.11–0.13 (low) and the effect holds inside native-size groups.
  - **SurvFace verification: weakly valid.** The low bin is below random only for AdaFace.
  - **SurvFace open-set identification: not valid.** The low-sharpness bin is *not* worse than random for any recognizer, and for MagFace it is better. On SurvFace the score mostly tracks crop size, and it disagrees with the second sharpness measure. **Do not use this bin as a "deblurring" subset for SurvFace identification without a change** (section 7).
- The FFT high-frequency measure is a consistent degradation indicator on SurvFace (accuracy rises monotonically with its quintile), but it is strongly tied to crop size (ρ ≈ −0.6), so it needs the same caution.

**Follow-up applied (section 11):** I added a size-controlled sharpness bin and a strict 10% dark bin. The size-controlled bin fixes the SurvFace identification anomaly (below random for all 4 models). The strict dark bin passes all three dark-bin rules on TinyFace.

## 2. Method
All numbers come from `configs/stratify.yaml`; seeds are fixed (42) and inference-only code is used.

| Measure | Definition |
|---|---|
| `luminance` | mean of the grayscale crop at native size (0–255) |
| `lap_var` | resize to 112×112 (bilinear), grayscale, `cv2.Laplacian(gray, CV_64F).var()` |
| `fft_hf` | share of non-DC spectral energy above 0.5×Nyquist, Hann window, computed at **native** resolution (so it is not affected by upsampling) |
| `magface_norm` | L2 norm of the raw MagFace embedding (Phase 1 preprocessing, no flip); computed for the 4 query splits only |
| `h, w, min_side` | native crop size (confounder) |

**Bins.** Quantiles of the observed distribution, never fixed thresholds. Rank-based with a stable sort, so exactly floor(q·n) crops enter an extreme bin and reruns are identical.
- `dark` = darkest 20% of luminance, else `normal`.
- `low_sharpness` = lowest 20% of `lap_var`; `sharp` = top 20%; the rest `mid`.
- Quintile columns `lum_q`, `sharp_q`, `fft_q`, `mag_q` (0 = lowest) and `size_q` (value-based, duplicate edges merged because most crops share a size).
- Thresholds are computed **per split** (the configured option), and each is saved in `thresholds.json`. A pooled-per-dataset option exists but was not run.

**Validation reuses the Phase 1 direct-feed per-sample scores (no re-embedding).**
- Queries are restricted to the slice and the gallery is unchanged. TinyFace: probes. SurvFace 1:N: both mated and unmated queries. SurvFace 1:1: a pair is `dark` / `low_sharpness` if *either* image is, and `normal` / `sharp` only if *both* are; quintile slices use the worse image's quintile.
- 95% bootstrap CIs (n=1000, seed 42). The random baseline draws 1000 random subsets of the same size (same number of mated, unmated, positive and negative items) and reports the 2.5–97.5% range.

## 3. Bin counts and thresholds
| Split | n | Dark bin n | Dark: luminance ≤ | Low-sharpness n | Low: lap_var ≤ | Sharp: lap_var ≥ | median min-side (px) |
|---|---|---|---|---|---|---|---|
| tinyface/probe | 3,728 | 745 | 83.2 | 745 | 4.18 | 17.62 | 32 |
| tinyface/gallery_match | 4,443 | 888 | 83.6 | 888 | 4.14 | 17.82 | 32 |
| tinyface/distractor | 153,428 | 30,685 | 63.8 | 30,685 | 1.84 | 6.27 | 13 |
| survface/gallery | 60,294 | 12,058 | 72.5 | 12,058 | 2.10 | 4.47 | 21 |
| survface/mated_probe | 60,423 | 12,084 | 74.8 | 12,084 | 2.12 | 4.61 | 23 |
| survface/unmated_probe | 121,736 | 24,347 | 74.2 | 24,347 | 2.18 | 4.95 | 22 |
| survface/verification | 9,913 | 1,982 | 64.6 | 1,982 | 2.41 | 6.95 | 19 |

Unreadable crops: 0 in every split. `fft_hf` is undefined (flat image) for 6 TinyFace distractors; they get no FFT bin and are never dropped from the other bins.
Thresholds differ between splits (for example the dark limit is 63.8 for distractors and 83.2 for probes), so "dark" is relative to each split, not a common absolute level.

## 4. Visual inspection (contact sheets)
50 random crops from each end of each measure, in `outputs/stratify/contact_sheets/<dataset>_<split>__<measure>_<end>.png`; each tile shows its score. I inspected these:
- `tinyface_probe__lap_var_LOW`: visibly blurry or smeared faces; `…_HIGH`: crisp faces with glasses, teeth, and edges. The score matches what the eye sees.
- `tinyface_probe__luminance_DARK`: moderately dim faces (luminance 40–83); some are very dark but many are only dull, matching the flag in section 6.
- `survface_mated_probe__lap_var_LOW`: tiny, heavily pixelated crops (the blocky upsampled look); `…_HIGH`: crops with more edge detail, some of it JPEG-block noise. Noise inflates this score, as expected for a lack-of-detail measure.
- `survface_mated_probe__luminance_DARK`: dim faces (luminance 47–74), not near-black.

The other 44 sheets (FFT, MagFace norm, remaining splits) were generated but not individually reviewed.

## 5. Does accuracy fall from sharp to low-sharpness? (direct-feed, Phase 1 scores)
Random range = 95% range of a size-matched random subset. "Below random" means the low-sharpness value is under that range.

### 5.1 TinyFace, Rank-1 (probes)
| Model | Full | Normal-lum | **Dark** | **Low-sharp** | Sharp | Random range at n=745 |
|---|---|---|---|---|---|---|
| ArcFace | 0.245 | 0.257 | 0.197 | **0.128** | 0.286 | 0.217–0.273 |
| AdaFace | 0.275 | 0.288 | 0.222 | **0.132** | 0.354 | 0.247–0.303 |
| MagFace | 0.274 | 0.289 | 0.215 | **0.134** | 0.344 | 0.244–0.305 |
| CurricularFace | 0.252 | 0.268 | 0.189 | **0.109** | 0.334 | 0.224–0.281 |

Dark and low-sharpness are below random for all 4 models. Across sharpness quintiles Rank-1 rises almost monotonically (for example AdaFace 0.131, 0.265, 0.302, 0.322, 0.354).

### 5.2 SurvFace open-set identification, TPIR20@FPIR=0.3
| Model | Full | Normal-lum | **Dark** | **Low-sharp** | Sharp | Random range (n≈12k mated) |
|---|---|---|---|---|---|---|
| ArcFace | 0.0412 | 0.0445 | 0.0290 | **0.0442** | 0.0434 | 0.0377–0.0444 |
| AdaFace | 0.0257 | 0.0269 | 0.0208 | **0.0223** | 0.0258 | 0.0233–0.0286 |
| MagFace | 0.0392 | 0.0427 | 0.0276 | **0.0498** | 0.0370 | 0.0360–0.0423 |
| CurricularFace | 0.0267 | 0.0281 | 0.0218 | **0.0257** | 0.0288 | 0.0241–0.0293 |

Dark is below random for all 4. **Low-sharpness is inside the random range for 3 models and above it for MagFace.** Across sharpness quintiles there is no monotone trend (ArcFace 0.044, 0.042, 0.032, 0.031, 0.043).

### 5.3 SurvFace verification, AUC
| Model | Full | Normal-lum | **Dark** | **Low-sharp** | Sharp |
|---|---|---|---|---|---|
| ArcFace | 0.679 | 0.700 | 0.637 | 0.687 | 0.698 |
| AdaFace | 0.663 | 0.682 | 0.619 | 0.644 | 0.701 |
| MagFace | 0.685 | 0.702 | 0.651 | 0.677 | 0.739 |
| CurricularFace | 0.667 | 0.685 | 0.624 | 0.656 | 0.712 |

Dark is below random for all 4. Low-sharpness is below random only for AdaFace; the quintile trend rises (AdaFace 0.644 → 0.701) but flattens above Q3. The numbers of positive pairs differ between bins (1,699 low vs 540 sharp for ArcFace) because of the pair rule.

### 5.4 Luminance trend
Accuracy is not monotone in luminance. On TinyFace it rises from Q1 to Q3–Q4 and falls again at Q5 (ArcFace 0.197, 0.247, 0.271, 0.272, 0.239), so the brightest quintile is also worse than the middle. On SurvFace verification it rises from the darkest to the brightest quintile (small dips for CurricularFace); SurvFace 1:N peaks at Q4.

## 6. Dark-bin check (pre-registered rules, `analysis/dark_flags.json`)
| Rule | Result |
|---|---|
| Too small (< 300 queries) | **Pass** everywhere (smallest dark bin: 745 TinyFace probes) |
| CI too wide (half-width > 5 points) | **Pass**: max half-width 0.030 (TinyFace Rank-1), 0.003 (SurvFace TPIR), 0.019 (SurvFace AUC) |
| Not really dark (upper luminance > 80 of 255) | **FLAG: tinyface/probe (83.2) and tinyface/gallery_match (83.6).** Pass: distractor 63.8, SurvFace 64.6–74.8 |

**Verdict:** the dark bin is large enough and measurably harder than a random subset on every protocol, so it is usable as a "low-luminance" subset. But on TinyFace its top end (83/255) is a dim face, not a night-time one, and on SurvFace the top is 65–75. Later low-light work should treat this as "darkest 20% of each split", not "genuinely low-light", and I recommend not claiming a low-light result from the TinyFace dark bin without saying so.

## 7. Second measure, confounders and the SurvFace anomaly
### 7.1 Agreement between sharpness measures (bottom 20% sets)
| Split | Spearman lap_var vs fft_hf | Jaccard | Cohen κ | lap_var vs MagFace norm (Spearman) |
|---|---|---|---|---|
| tinyface/probe | 0.66 | 0.52 | 0.60 | 0.05 |
| tinyface/distractor | −0.19 | 0.07 | −0.10 | n/a |
| survface/mated_probe | −0.02 | 0.11 | 0.00 | 0.42 |
| survface/unmated_probe | 0.02 | 0.12 | 0.03 | 0.41 |
| survface/verification | −0.05 | 0.11 | −0.01 | 0.25 |

The two measures agree on TinyFace probes and gallery (ρ 0.66) and **do not agree on SurvFace (ρ ≈ 0)**. MagFace norm agrees moderately with `lap_var` on SurvFace queries (ρ ≈ 0.4) but not on TinyFace. MagFace-norm-low is not below random on TinyFace (ArcFace 0.252 against a random range of 0.219–0.274), so the norm is not a useful quality score there.

### 7.2 Confounders
- **Native size dominates on SurvFace:** `lap_var` correlates +0.47 to +0.57 with min-side and `fft_hf` correlates −0.58 to −0.62 (smaller crops have a higher share of high-frequency energy). On TinyFace probes both correlations are weak (|ρ| ≤ 0.15) because almost all probes are 32 px.
- **Luminance** correlates with `lap_var` at 0.34 on TinyFace probes (dark crops look less sharp) and weakly elsewhere.
- **Smallest SurvFace crops are the most accurate**, which I cannot explain: the smallest native-size quintile (`size_Q1`) gives TPIR20@FPIR=0.3 of 0.072 for ArcFace against 0.020–0.036 for the other quintiles. Because small crops tend to have low `lap_var` (ρ ≈ +0.47), they probably make up much of the low-sharpness bin, which would explain why that bin is not worse than random; I did not count the overlap. This could be a camera or identity cohort effect; I have not tested it.

### 7.3 Sharpness within native-size groups
- **TinyFace** (probes, 287 and 458 per end): in both size tertiles, low-sharpness Rank-1 (0.108–0.140) is far below top-sharpness (0.277–0.371), with non-overlapping CIs for every model. The TinyFace effect is not a size artefact.
- **SurvFace 1:N** (ArcFace and MagFace shown in `analysis/within_size.csv`): in the two larger size tertiles (21–24 px and 25–78 px) low-sharpness is clearly worse (ArcFace 0.021 and 0.018 against 0.046 and 0.040). In the smallest tertile (5–20 px) it is not (0.037 against 0.032). The pooled result is therefore a size mix effect (a Simpson-style confound), and the sharpness score is meaningful on SurvFace only after controlling for size.

## 8. Baselines requested
- **Full unstratified set:** the "Full" columns above (identical to Phase 1).
- **Size-matched random subset:** random ranges in sections 5.1–5.3 and `analysis/random_baseline.csv` (all slices, all models).
- **Agreement with a second sharpness measure:** section 7.1 (`analysis/agreement.csv`).

## 9. Issues and limits
1. **The Laplacian low-sharpness subset is not a valid blur proxy for SurvFace 1:N**, and only weakly valid for SurvFace verification. It mostly measures lack of detail at tiny sizes.
2. **TinyFace dark bin is "dim", not "dark"** (flag in section 6).
3. **Only luminance and sharpness were stratified.** The size-controlled sharpness bin was added in section 11; an intersection of `lap_var` and `fft_hf` and dataset-pooled thresholds were not tested.
4. **Pair-level verification slices** use the worse-image rule, which gives different positive and negative counts per slice.
5. **MagFace norm** was computed for 4 query splits only and without flip.
6. Contact sheets: 44 of the 50 sheets were generated but not individually reviewed.

## 10. Decisions for Tasks 2 and 3 (resolved: recommendations adopted, see section 11)
1. SurvFace deblurring subset: size-controlled low-sharpness bin (`sharp_bin_sz`); TinyFace keeps the plain Laplacian bin.
2. Low-light subset: darkest 20% (`lum_bin`) kept, plus a strict darkest-10% bin (`lum_bin10`); use the 10% bin for TinyFace.
3. Thresholds: per-split quantiles kept.

## 11. Follow-up: size-controlled sharpness and strict dark bin
New columns in `outputs/stratify/bins/*.csv`: `lum_bin10` (`dark10` = darkest 10%), `size_group` (native min-side tertile, duplicate edges merged) and `sharp_bin_sz` (`low_sharpness` / `mid` / `sharp` = lowest 20% / top 20% of `lap_var` **ranked inside each size group**). All earlier columns are unchanged. New slices: `dark10`, `low_sharpness_sz`, `sharp_sz`, `not_low_sharpness_sz` in `analysis/metrics_by_slice.csv` and `random_baseline.csv`.

### 11.1 Size-controlled low-sharpness (`low_sharpness_sz`)
| Protocol (metric) | Model | Full | Low-sharp (plain) | **Low-sharp (size-controlled)** | Random range for that n | Below random? |
|---|---|---|---|---|---|---|
| SurvFace 1:N (TPIR20@FPIR=0.3) | ArcFace | 0.0412 | 0.0442 | **0.0324** | 0.0377–0.0442 | yes |
| | AdaFace | 0.0257 | 0.0223 | **0.0150** | 0.0231–0.0286 | yes |
| | MagFace | 0.0392 | 0.0498 | **0.0356** | 0.0361–0.0422 | yes (barely) |
| | CurricularFace | 0.0267 | 0.0257 | **0.0194** | 0.0242–0.0294 | yes |
| SurvFace 1:1 (AUC) | ArcFace | 0.679 | 0.687 | **0.692** | 0.664–0.693 | no |
| | AdaFace | 0.663 | 0.644 | **0.647** | 0.649–0.676 | yes |
| | MagFace | 0.685 | 0.677 | **0.675** | 0.671–0.699 | no |
| | CurricularFace | 0.667 | 0.656 | **0.663** | 0.651–0.682 | no |
| TinyFace (Rank-1) | all 4 | 0.245–0.275 | 0.109–0.134 | **0.110–0.133** | 0.217–0.305 | yes, all 4 |

- **SurvFace 1:N:** controlling for size removes the anomaly. Low sharpness is now below random for all 4 models (MagFace only just: 0.0356 against a lower bound of 0.0361). The ArcFace CI is 0.030–0.036 and the CurricularFace CI 0.017–0.022, so the 1:N loss is clear for those two.
- **SurvFace 1:1:** still weak. Only AdaFace is below random. The sharp-versus-low gap exists for AdaFace (0.714 against 0.647), MagFace (0.746 against 0.675) and CurricularFace (0.725 against 0.663) but not for ArcFace (0.687 against 0.692). Pair counts are uneven by the pair rule (499 positive and 212 negative pairs for `sharp_sz` against 1,699 and 1,913 for `low_sharpness_sz`), so verification slices are less reliable.
- **TinyFace:** no change, as expected (almost all probes are 32 px).

### 11.2 Strict dark bin (`dark10`)
| Split / check | Result | Rule |
|---|---|---|
| TinyFace probe: n | 372 | pass (≥ 300) |
| TinyFace probe: upper luminance | 68.5 | pass (≤ 80); the 20% bin failed at 83.2 |
| TinyFace Rank-1 CI half-width | 0.036–0.039 | pass (≤ 0.05) |
| SurvFace mated / unmated queries: n | 6,042 / 12,173 | pass |
| SurvFace upper luminance | 55.1–66.9 | pass |

Accuracy in the 10% bin against a size-matched random subset:
- **TinyFace Rank-1:** below random for all 4 (ArcFace 0.175 against 0.207–0.288; AdaFace 0.177; MagFace 0.186; CurricularFace 0.156).
- **SurvFace 1:N TPIR20@FPIR=0.3:** below random for ArcFace, AdaFace and MagFace; CurricularFace is *not* (0.0275, inside 0.0232–0.0311).
- **SurvFace verification AUC:** below random for all 4.

The strict bin is therefore the better low-light subset for TinyFace. It is not a perfect one for SurvFace identification with CurricularFace. Its CI on TinyFace is wider than the 20% bin because it has half as many probes.

### 11.3 What is still unresolved
- The unexplained high accuracy of the smallest SurvFace crops (section 7.2) is not addressed by the size-controlled bin; it only prevents it from masking the sharpness effect.
- Verification-pair slices remain weak and unevenly sized.
- `lap_var` still responds to noise and JPEG blocking, and the size-controlled bin was only checked numerically, not by new contact sheets.

**STOP: awaiting your approval before Phase 3.**
