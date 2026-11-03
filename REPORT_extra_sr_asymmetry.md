# REPORT (extra): why the SR penalty differs between SurvFace and TinyFace

**Status:** pre-registration only. This file was committed before any analysis number was computed and before `scripts/extra_sr_asymmetry.py` exists. It contains only the hypotheses, tests and decision criteria, copied verbatim from `research/Sprint_Notes/Phase_5_continuation.md` (Part E, "Analyses"). The results sections are added in a later commit, and this section is not edited afterwards.

**Question (supervisor request, 2026-10-27):** why does restoration change recognition by different amounts on SurvFace and TinyFace? Either outcome is reported as found.

**Fixed before the run (from the same Part E):** no new restoration and no new inference; only cached Phase 3 embeddings, per-crop measures and pairs are used; CPU only; seed 42; non-leaky recognizers AdaFace, MagFace and CurricularFace (ArcFace only in an appendix, marked as leaky for OSDFace); variants bicubic `pre512`, Real-ESRGAN `native`, InvSR `pre64_st1` and `pre128_st1`, TADSR `pre128_t250`, OSDFace `pre128`, each in two modes (both sides restored, and probes only). A missing cache is recorded and skipped, never filled by running a restorer.

### Analyses (all bootstraps: 1,000 draws, seed 42, paired on the same items)

**E0. Common scale (do this first).** For every variant × recognizer × mode, compute the same metrics on both datasets:
- **Closed-set Rank-1:**
  - TinyFace: probes against gallery plus distractors (the Phase 3 protocol);
  - SurvFace: mated queries against the 200-identity gallery (new for SurvFace).
- **Verification AUC and d′:**
  - SurvFace: the existing 1,000 positive and 1,000 negative pairs;
  - TinyFace: 1,000 genuine and 1,000 impostor probe–gallery pairs, seeded. Save the pair list to `outputs/sr/results/asymmetry/tinyface_pairs.csv` before scoring.
- For each metric, report:
  - the paired change (restored − direct) with its 95% CI;
  - the relative change (Δ / direct).
- Output: `asym_common_scale.csv` and an asymmetry table in the report.
- **Decision:** if, on the matched metrics, the dataset difference in Δ falls inside the CIs for most variants, the asymmetry was a protocol artefact (**H1 supported**). Report that, run H4 for completeness, and stop. Otherwise continue with H2–H6.

**H2. Input information (crop size, sharpness, brightness).**
- Within each dataset, relate the per-crop change in genuine similarity (restored vs direct, mated pairs) to native `min_side`, `lap_var` and luminance. Use binned means with CIs; no fitted causal model.
- Then compare the datasets on **matched strata**: coarsened exact matching on `size_q` × `sharp_bin_sz`, using only strata present in both datasets.
- **Supported if** the dataset gap in Δ genuine similarity shrinks by at least 50% on matched strata.

**H3. Distractor composition (TinyFace only).**
- Recompute both-sides TinyFace Rank-1 with the 4,000 distractors kept at direct feed, and only probes and gallery matches restored.
- Split the impostor-score shift into gallery-match impostors and distractor impostors.
- **Supported if** keeping distractors at direct feed changes the TinyFace Rank-1 Δ by more than the half-width of its CI.

**H4. Genuine vs impostor decomposition.**
- Per dataset, variant and recognizer: Δ mean genuine score, Δ mean impostor score, Δ d′.
- This shows whether restoration makes faces more alike ("homogenises"): both score distributions rise, and separation falls.
- **Supported** (as the mechanism) if the arm with the larger penalty on a dataset shows a larger Δ impostor relative to Δ genuine there.

**H5. Camera domain (SurvFace).**
- SurvFace file names carry the camera (`<id>_cam<k>_<n>.jpg`).
- Split mated query–gallery comparisons into same-camera and cross-camera. Compare their Δ mated score.
- Also compute camera-ID predictability from the embeddings: leave-one-out kNN (k = 5) camera accuracy, direct vs restored, on the SurvFace gallery.
- **Supported if** restoration raises camera predictability, or hurts cross-camera pairs more than same-camera pairs (CI excludes zero).

**H6. Open-set threshold (SurvFace).**
- Split the TPIR@FPIR = 0.3 change into two parts:
  1. **Threshold shift:** the threshold moves because unmated scores change. Compute TPIR with the direct-feed threshold applied to restored mated scores.
  2. **Mated-score change:** the remainder.
- Report both parts. **Supported** (as the main driver) if the threshold-shift part is larger than the mated part.

**H7. Quality inflation (deferred; not run in this pass).**
- The question: do restored crops gain MagFace norm, so they look "higher quality", on the dataset with the larger penalty, without a matching gain in genuine similarity?
- The cached embeddings cannot answer it. They are L2-normalised (`src/recognizer.py`, `get_embedding`), so the raw norm is lost.
- Answering it would need new MagFace inference on the restored PNGs. That breaks this study's "no new inference" constraint. Also, OSDFace restorations are not stored at full size on the PC.
- Write H7 in the pre-registration as "deferred: needs re-embedding; decision for Abser". Do not run it.

**Multiplicity.** Show Bonferroni-adjusted CIs over the hypothesis tests next to the plain ones (as `REPORT_phase4.md` does). Verdicts use the plain CIs; the report mentions any verdict that changes under Bonferroni.

**H7 status in this pre-registration: deferred; needs re-embedding; decision for Abser.** It is not run in this pass.

---

## Operationalisation addendum (committed after the pre-registration and before any number was computed)

The pre-registration above is not edited. The choices below turn its tests into code. No analysis number existed when they were written; the only data looked at were file shapes, the column names of the per-crop measure files, and the value ranges of `size_q`.

1. **Cells.** A cell is one variant (6) × non-leaky recognizer (AdaFace, MagFace, CurricularFace) × mode (both sides restored, probes only) = 36 cells. ArcFace is left out; the plan allows it only in an appendix. Cache check done before writing this: all 7 embedding folders exist for all 4 recognizers, 22,898 rows each, no zero or NaN rows.
2. **Verdict rule (all hypotheses).** A hypothesis is *supported* if its pre-registered criterion is met in more than half of its applicable cells, *not supported* if met in fewer than a third, and *inconclusive* otherwise. The counts are always shown. This is the reading of "most variants" in the E0 decision rule too.
3. **E0 matched metrics.** Closed-set Rank-1 and verification AUC, absolute change (restored minus direct) with a paired 95% bootstrap CI; relative change (change divided by direct) and d′ are reported but do not decide. The dataset difference is the TinyFace change minus the SurvFace change; its CI is the percentile interval of the difference of the two datasets' bootstrap draws (draw k minus draw k; the datasets are independent). H1 is supported if that CI contains zero in more than half of the 36 cells for **both** matched metrics. If H1 is supported, only H4 is run afterwards (for completeness) and the report stops there; otherwise H2–H6 are run.
4. **SurvFace closed-set Rank-1.** The 4,476 mated probe images against the SurvFace gallery images, reusing `tinyface_identification` with no distractors. It is not an official SurvFace protocol.
5. **TinyFace verification pairs.** Seed 42. 1,000 distinct probes each get one random gallery-match image of the same identity (genuine); another 1,000 distinct probes each get one random gallery-match image of a different identity (impostor). Saved to `outputs/sr/results/asymmetry/tinyface_pairs.csv` before scoring. Scored with the same function as the SurvFace pairs (Euclidean distance on normalised embeddings, thresholds 0.2 to 2.0, `survface_verification_from_scores`). Probes-only mode restores the first image of each pair (the probe) and keeps the second as direct feed, as in `scripts/sr_extra_analysis.py`.
6. **Bootstrap.** 1,000 draws, seed 42. Index sets depend only on the sample size, so every variant and recognizer shares the same resamples. Bonferroni-adjusted CIs use a family of 36 (the cells) and are shown next to the plain ones. Verdicts use the plain CIs.
7. **H2.** The per-probe genuine similarity is the mean cosine between the probe and all gallery images of the same identity (TinyFace: gallery-match images; SurvFace: gallery images), the same definition for both datasets. The change is restored minus direct, mode "both". Binned means use `min_side` bins (up to 12, 13–16, 17–24, 25–32, above 32 px), and `lap_var` and `luminance` quintiles pooled over both datasets. **Matched strata** are `size_q` × `sharp_bin_sz`, using only strata present in both datasets, as pre-registered. Note: `size_q` is a quintile within each split, so the same `size_q` is a different pixel range in the two datasets (for example, size_q 0 is 12–30 px for TinyFace probes and 5–12 px for SurvFace mated probes). The matched strata are therefore matched by rank, not by pixel size. A **sensitivity** version with pooled pixel bins × `sharp_bin_sz` is also computed and reported, but the verdict uses the pre-registered strata. The unmatched gap is the TinyFace mean change minus the SurvFace mean change over all mated probes; the matched gap is the average over common strata of the stratum gap, weighted by the stratum's total number of probes. Criterion: the absolute matched gap is at most half the absolute unmatched gap (per cell, mode "both").
8. **H3.** Mode "both" only. The hybrid keeps the 4,000 distractors at direct feed and restores probes and gallery matches. The change in the Rank-1 change (hybrid minus standard both-sides) is compared with the half-width of the standard both-sides change's paired CI. The impostor shift (best cosine to any gallery image of another identity, per probe) is split into gallery-match impostors and distractor impostors.
9. **H4.** On the verification pair sets, per dataset: change in mean genuine cosine, in mean impostor cosine, and in d′. For each (dataset, recognizer, mode) group of 6 variants, the criterion is met if the Spearman correlation across variants between the penalty (minus the change in verification AUC) and (impostor change minus genuine change) is above zero. With only 6 variants this is descriptive.
10. **H5.** Every mated SurvFace probe is compared with every gallery image of the same identity; same-camera and cross-camera comparisons are separated by the camera in the file name. The change in cosine is bootstrapped over comparisons (the comparisons share queries, so the CI is somewhat narrow). The camera predictability is the leave-one-out kNN (k = 5, cosine, majority vote, ties to the smaller camera number) accuracy on the SurvFace gallery. The criterion is met in a cell if the kNN accuracy rises with a CI above zero, or if the cross-camera change minus the same-camera change has a CI entirely below zero. Only the "both" mode changes the gallery, so only "both" cells count.
11. **H6.** The threshold is the one that the official open-set code picks for FPIR 0.3. Mated-score part = TPIR with the direct threshold applied to restored mated scores, minus direct TPIR; threshold-shift part = restored TPIR (with its own threshold) minus that mixed value. The two sum to the total change. The criterion is met if the absolute threshold-shift part is larger than the absolute mated-score part. This follows the stated logic (the threshold moves because the unmated scores change, and fixing the direct threshold isolates the mated-score change); the plan's wording lists the parts in the other order.
12. **Devices.** The OSDFace embeddings were computed on a Colab T4 and merged into the cache; every other embedding, including the direct-feed baseline, is from the lab PC. The OSDFace changes therefore compare T4 restored embeddings with lab-PC direct embeddings. Phase 3's cross-device check is the only evidence that this is harmless.


---

# Results (added after the run; the sections above are unchanged)

Numbers come from `outputs/sr/results/asymmetry/` (written by `scripts/extra_sr_asymmetry.py`, committed before the run). The script ran twice and all 8 output files have identical SHA-256 (`determinism_hashes.txt`). Check against Phase 3: the direct-feed and both-sides-restored TinyFace Rank-1 and SurvFace verification AUC computed here equal the Phase 3 result files to 6 decimals in all 18 variant × recognizer cells and for the 3 direct-feed baselines (0 mismatches). CPU only, seed 42, 1,000 bootstrap draws.

## 1. Summary
- **By the pre-registered rule, H1 is supported: the difference between SurvFace and TinyFace in the Phase 3 tables is largely a protocol artefact.** On a common scale, the dataset difference in the change caused by restoration has a CI that contains zero in 26 of 36 cells for closed-set Rank-1 (72%) and in 20 of 36 cells for verification AUC (56%). Under the Bonferroni adjustment these become 32 and 27 of 36. Per the decision rule, H2, H3, H5 and H6 were not run.
- **The change in absolute Rank-1 is about the same on both datasets** (mean over the 36 cells: TinyFace -0.052, SurvFace -0.057). In relative terms SurvFace loses more (mean -29% against -16%), because its direct-feed Rank-1 is lower (for example 0.18 against 0.34 for AdaFace).
- **The AUC support is marginal and has a direction.** 20 of 36 is just above the pre-registered "more than half". In the 16 cells where the gap is significant, TinyFace always does better than SurvFace (mean gap +0.022 AUC over all 36 cells). These cells come from the four generative arms (Real-ESRGAN, InvSR, OSDFace, TADSR; 9 with both sides restored, 7 probes-only), never from bicubic, and 14 of the 16 are MagFace or CurricularFace.
- **H4 is supported as a mechanism on both datasets:** across the 6 variants, the arms with the larger penalty raise impostor similarity more relative to genuine similarity (Spearman 0.77 to 1.00 in all 12 dataset × recognizer × mode groups). Restoration makes faces more alike, on both datasets. With 6 variants per group this is descriptive.
- **H7 is deferred** (needs new MagFace inference on the restored crops; decision for Abser).

## 2. The pre-registered plan
Unchanged: see the pre-registration and the operationalisation addendum at the top of this file.

## 3. Common-scale table (E0)
Closed-set Rank-1 (TinyFace: probes against gallery matches plus 4,000 distractors; SurvFace: 4,476 mated probe images against the 4,486-image gallery, no distractors), and verification AUC and d' on 1,000 genuine and 1,000 impostor pairs per dataset (SurvFace: the existing pairs; TinyFace: the seeded pair list `tinyface_pairs.csv`). The change is restored minus direct feed with its paired 95% bootstrap CI; the last column says whether the CI of the dataset gap contains zero. The CSV files also hold the relative changes and the Bonferroni-adjusted CIs (`asym_common_scale.csv`, `asym_dataset_gap.csv`).

#### Common scale (E0): change (restored minus direct) with 95% CI [Bonferroni over 36 cells in brackets in the CSV]

**Closed-set Rank-1**

| Variant | Recognizer | Mode | TinyFace direct -> restored | TinyFace change | SurvFace direct -> restored | SurvFace change | Gap (TinyFace - SurvFace) | CI contains 0 |
|---|---|---|---|---|---|---|---|---|
| bicubic: pre512 | adaface | both | 0.336 -> 0.342 | +0.006 [-0.007, +0.020] | 0.183 -> 0.191 | +0.007 [-0.001, +0.015] | -0.001 [-0.017, +0.015] | yes |
| bicubic: pre512 | adaface | probes | 0.336 -> 0.336 | +0.000 [-0.010, +0.012] | 0.183 -> 0.187 | +0.004 [-0.004, +0.011] | -0.004 [-0.016, +0.010] | yes |
| esrgan: native | adaface | both | 0.336 -> 0.343 | +0.007 [-0.016, +0.029] | 0.183 -> 0.141 | -0.042 [-0.055, -0.030] | +0.049 [+0.022, +0.075] | no |
| esrgan: native | adaface | probes | 0.336 -> 0.252 | -0.084 [-0.107, -0.063] | 0.183 -> 0.088 | -0.095 [-0.107, -0.083] | +0.011 [-0.015, +0.036] | yes |
| invsr: pre64_st1 | adaface | both | 0.336 -> 0.242 | -0.094 [-0.120, -0.071] | 0.183 -> 0.084 | -0.099 [-0.111, -0.087] | +0.005 [-0.022, +0.032] | yes |
| invsr: pre64_st1 | adaface | probes | 0.336 -> 0.190 | -0.146 [-0.171, -0.123] | 0.183 -> 0.047 | -0.137 [-0.149, -0.125] | -0.010 [-0.037, +0.018] | yes |
| invsr: pre128_st1 | adaface | both | 0.336 -> 0.267 | -0.069 [-0.093, -0.046] | 0.183 -> 0.102 | -0.081 [-0.092, -0.069] | +0.012 [-0.015, +0.040] | yes |
| invsr: pre128_st1 | adaface | probes | 0.336 -> 0.208 | -0.128 [-0.152, -0.105] | 0.183 -> 0.040 | -0.144 [-0.156, -0.132] | +0.016 [-0.010, +0.043] | yes |
| tadsr: pre128_t250 | adaface | both | 0.336 -> 0.367 | +0.031 [+0.008, +0.054] | 0.183 -> 0.169 | -0.015 [-0.025, -0.003] | +0.046 [+0.021, +0.070] | no |
| tadsr: pre128_t250 | adaface | probes | 0.336 -> 0.313 | -0.023 [-0.043, -0.001] | 0.183 -> 0.151 | -0.032 [-0.043, -0.022] | +0.010 [-0.014, +0.035] | yes |
| osdface: pre128 | adaface | both | 0.336 -> 0.409 | +0.073 [+0.048, +0.098] | 0.183 -> 0.218 | +0.035 [+0.023, +0.048] | +0.038 [+0.011, +0.067] | no |
| osdface: pre128 | adaface | probes | 0.336 -> 0.267 | -0.069 [-0.094, -0.048] | 0.183 -> 0.077 | -0.106 [-0.119, -0.093] | +0.037 [+0.009, +0.064] | no |
| bicubic: pre512 | magface | both | 0.336 -> 0.317 | -0.019 [-0.031, -0.005] | 0.233 -> 0.232 | -0.002 [-0.010, +0.006] | -0.017 [-0.032, -0.001] | no |
| bicubic: pre512 | magface | probes | 0.336 -> 0.323 | -0.013 [-0.024, -0.003] | 0.233 -> 0.232 | -0.002 [-0.009, +0.006] | -0.011 [-0.025, +0.003] | yes |
| esrgan: native | magface | both | 0.336 -> 0.288 | -0.048 [-0.069, -0.026] | 0.233 -> 0.166 | -0.068 [-0.081, -0.055] | +0.020 [-0.006, +0.046] | yes |
| esrgan: native | magface | probes | 0.336 -> 0.217 | -0.119 [-0.143, -0.094] | 0.233 -> 0.104 | -0.129 [-0.142, -0.118] | +0.011 [-0.017, +0.038] | yes |
| invsr: pre64_st1 | magface | both | 0.336 -> 0.216 | -0.119 [-0.143, -0.097] | 0.233 -> 0.094 | -0.140 [-0.152, -0.127] | +0.020 [-0.004, +0.045] | yes |
| invsr: pre64_st1 | magface | probes | 0.336 -> 0.153 | -0.183 [-0.207, -0.158] | 0.233 -> 0.064 | -0.169 [-0.182, -0.157] | -0.014 [-0.042, +0.014] | yes |
| invsr: pre128_st1 | magface | both | 0.336 -> 0.239 | -0.097 [-0.119, -0.074] | 0.233 -> 0.144 | -0.090 [-0.103, -0.078] | -0.007 [-0.032, +0.019] | yes |
| invsr: pre128_st1 | magface | probes | 0.336 -> 0.195 | -0.141 [-0.165, -0.114] | 0.233 -> 0.124 | -0.109 [-0.122, -0.098] | -0.032 [-0.059, -0.004] | no |
| tadsr: pre128_t250 | magface | both | 0.336 -> 0.331 | -0.005 [-0.026, +0.017] | 0.233 -> 0.214 | -0.019 [-0.029, -0.009] | +0.014 [-0.010, +0.039] | yes |
| tadsr: pre128_t250 | magface | probes | 0.336 -> 0.298 | -0.038 [-0.061, -0.015] | 0.233 -> 0.216 | -0.018 [-0.027, -0.008] | -0.020 [-0.045, +0.006] | yes |
| osdface: pre128 | magface | both | 0.336 -> 0.362 | +0.026 [+0.003, +0.050] | 0.233 -> 0.223 | -0.010 [-0.022, +0.003] | +0.036 [+0.011, +0.063] | no |
| osdface: pre128 | magface | probes | 0.336 -> 0.237 | -0.099 [-0.124, -0.074] | 0.233 -> 0.148 | -0.085 [-0.099, -0.071] | -0.014 [-0.044, +0.015] | yes |
| bicubic: pre512 | curricularface | both | 0.298 -> 0.284 | -0.014 [-0.029, +0.001] | 0.166 -> 0.167 | +0.002 [-0.007, +0.010] | -0.016 [-0.031, +0.002] | yes |
| bicubic: pre512 | curricularface | probes | 0.298 -> 0.292 | -0.006 [-0.019, +0.007] | 0.166 -> 0.169 | +0.003 [-0.004, +0.011] | -0.009 [-0.024, +0.006] | yes |
| esrgan: native | curricularface | both | 0.298 -> 0.301 | +0.003 [-0.019, +0.025] | 0.166 -> 0.150 | -0.015 [-0.027, -0.004] | +0.019 [-0.007, +0.044] | yes |
| esrgan: native | curricularface | probes | 0.298 -> 0.198 | -0.100 [-0.124, -0.075] | 0.166 -> 0.082 | -0.083 [-0.095, -0.072] | -0.016 [-0.044, +0.011] | yes |
| invsr: pre64_st1 | curricularface | both | 0.298 -> 0.218 | -0.080 [-0.100, -0.057] | 0.166 -> 0.079 | -0.086 [-0.097, -0.076] | +0.007 [-0.017, +0.033] | yes |
| invsr: pre64_st1 | curricularface | probes | 0.298 -> 0.150 | -0.148 [-0.172, -0.123] | 0.166 -> 0.050 | -0.115 [-0.127, -0.104] | -0.033 [-0.059, -0.005] | no |
| invsr: pre128_st1 | curricularface | both | 0.298 -> 0.218 | -0.080 [-0.102, -0.056] | 0.166 -> 0.097 | -0.069 [-0.080, -0.057] | -0.011 [-0.035, +0.013] | yes |
| invsr: pre128_st1 | curricularface | probes | 0.298 -> 0.180 | -0.118 [-0.140, -0.092] | 0.166 -> 0.072 | -0.094 [-0.105, -0.083] | -0.024 [-0.050, +0.004] | yes |
| tadsr: pre128_t250 | curricularface | both | 0.298 -> 0.322 | +0.024 [+0.003, +0.049] | 0.166 -> 0.143 | -0.022 [-0.032, -0.012] | +0.047 [+0.024, +0.073] | no |
| tadsr: pre128_t250 | curricularface | probes | 0.298 -> 0.255 | -0.042 [-0.068, -0.017] | 0.166 -> 0.151 | -0.015 [-0.025, -0.005] | -0.028 [-0.056, +0.001] | yes |
| osdface: pre128 | curricularface | both | 0.298 -> 0.383 | +0.085 [+0.061, +0.110] | 0.166 -> 0.209 | +0.044 [+0.031, +0.057] | +0.041 [+0.015, +0.069] | no |
| osdface: pre128 | curricularface | probes | 0.298 -> 0.241 | -0.057 [-0.081, -0.031] | 0.166 -> 0.111 | -0.055 [-0.066, -0.042] | -0.003 [-0.031, +0.027] | yes |

**Verification AUC**

| Variant | Recognizer | Mode | TinyFace direct -> restored | TinyFace change | SurvFace direct -> restored | SurvFace change | Gap (TinyFace - SurvFace) | CI contains 0 |
|---|---|---|---|---|---|---|---|---|
| bicubic: pre512 | adaface | both | 0.813 -> 0.817 | +0.004 [-0.000, +0.009] | 0.652 -> 0.657 | +0.004 [-0.005, +0.013] | -0.000 [-0.010, +0.010] | yes |
| bicubic: pre512 | adaface | probes | 0.813 -> 0.813 | +0.000 [-0.003, +0.004] | 0.652 -> 0.651 | -0.001 [-0.008, +0.006] | +0.001 [-0.006, +0.009] | yes |
| esrgan: native | adaface | both | 0.813 -> 0.811 | -0.002 [-0.017, +0.015] | 0.652 -> 0.672 | +0.020 [-0.001, +0.041] | -0.022 [-0.048, +0.004] | yes |
| esrgan: native | adaface | probes | 0.813 -> 0.785 | -0.028 [-0.042, -0.014] | 0.652 -> 0.633 | -0.019 [-0.037, -0.002] | -0.009 [-0.031, +0.013] | yes |
| invsr: pre64_st1 | adaface | both | 0.813 -> 0.768 | -0.045 [-0.063, -0.028] | 0.652 -> 0.587 | -0.066 [-0.092, -0.041] | +0.021 [-0.009, +0.051] | yes |
| invsr: pre64_st1 | adaface | probes | 0.813 -> 0.737 | -0.076 [-0.092, -0.059] | 0.652 -> 0.589 | -0.063 [-0.084, -0.041] | -0.013 [-0.040, +0.014] | yes |
| invsr: pre128_st1 | adaface | both | 0.813 -> 0.781 | -0.032 [-0.049, -0.014] | 0.652 -> 0.605 | -0.047 [-0.070, -0.023] | +0.015 [-0.013, +0.043] | yes |
| invsr: pre128_st1 | adaface | probes | 0.813 -> 0.767 | -0.046 [-0.063, -0.030] | 0.652 -> 0.602 | -0.050 [-0.069, -0.031] | +0.004 [-0.020, +0.028] | yes |
| tadsr: pre128_t250 | adaface | both | 0.813 -> 0.850 | +0.037 [+0.023, +0.051] | 0.652 -> 0.649 | -0.003 [-0.019, +0.012] | +0.040 [+0.020, +0.061] | no |
| tadsr: pre128_t250 | adaface | probes | 0.813 -> 0.817 | +0.003 [-0.010, +0.017] | 0.652 -> 0.637 | -0.016 [-0.027, -0.004] | +0.019 [+0.001, +0.037] | no |
| osdface: pre128 | adaface | both | 0.813 -> 0.848 | +0.034 [+0.018, +0.051] | 0.652 -> 0.704 | +0.051 [+0.026, +0.077] | -0.017 [-0.049, +0.014] | yes |
| osdface: pre128 | adaface | probes | 0.813 -> 0.787 | -0.026 [-0.044, -0.011] | 0.652 -> 0.627 | -0.025 [-0.049, +0.001] | -0.001 [-0.032, +0.027] | yes |
| bicubic: pre512 | magface | both | 0.771 -> 0.768 | -0.002 [-0.007, +0.002] | 0.673 -> 0.671 | -0.002 [-0.008, +0.004] | -0.001 [-0.008, +0.007] | yes |
| bicubic: pre512 | magface | probes | 0.771 -> 0.769 | -0.002 [-0.005, +0.001] | 0.673 -> 0.672 | -0.001 [-0.006, +0.003] | -0.001 [-0.007, +0.004] | yes |
| esrgan: native | magface | both | 0.771 -> 0.778 | +0.007 [-0.008, +0.022] | 0.673 -> 0.671 | -0.002 [-0.020, +0.017] | +0.008 [-0.014, +0.032] | yes |
| esrgan: native | magface | probes | 0.771 -> 0.756 | -0.015 [-0.028, -0.002] | 0.673 -> 0.637 | -0.035 [-0.052, -0.019] | +0.021 [+0.001, +0.042] | no |
| invsr: pre64_st1 | magface | both | 0.771 -> 0.747 | -0.024 [-0.041, -0.005] | 0.673 -> 0.594 | -0.079 [-0.101, -0.058] | +0.055 [+0.027, +0.084] | no |
| invsr: pre64_st1 | magface | probes | 0.771 -> 0.727 | -0.044 [-0.061, -0.028] | 0.673 -> 0.580 | -0.092 [-0.112, -0.073] | +0.048 [+0.022, +0.074] | no |
| invsr: pre128_st1 | magface | both | 0.771 -> 0.772 | +0.001 [-0.016, +0.018] | 0.673 -> 0.628 | -0.045 [-0.067, -0.025] | +0.046 [+0.020, +0.071] | no |
| invsr: pre128_st1 | magface | probes | 0.771 -> 0.741 | -0.030 [-0.046, -0.014] | 0.673 -> 0.616 | -0.056 [-0.073, -0.039] | +0.026 [+0.003, +0.049] | no |
| tadsr: pre128_t250 | magface | both | 0.771 -> 0.825 | +0.054 [+0.041, +0.068] | 0.673 -> 0.646 | -0.026 [-0.039, -0.013] | +0.080 [+0.062, +0.099] | no |
| tadsr: pre128_t250 | magface | probes | 0.771 -> 0.776 | +0.005 [-0.009, +0.019] | 0.673 -> 0.652 | -0.021 [-0.030, -0.011] | +0.026 [+0.009, +0.042] | no |
| osdface: pre128 | magface | both | 0.771 -> 0.809 | +0.038 [+0.019, +0.057] | 0.673 -> 0.676 | +0.003 [-0.021, +0.028] | +0.035 [+0.004, +0.068] | no |
| osdface: pre128 | magface | probes | 0.771 -> 0.765 | -0.006 [-0.024, +0.012] | 0.673 -> 0.623 | -0.050 [-0.072, -0.027] | +0.044 [+0.015, +0.076] | no |
| bicubic: pre512 | curricularface | both | 0.785 -> 0.782 | -0.003 [-0.008, +0.003] | 0.647 -> 0.644 | -0.002 [-0.011, +0.006] | -0.000 [-0.011, +0.010] | yes |
| bicubic: pre512 | curricularface | probes | 0.785 -> 0.782 | -0.003 [-0.007, +0.001] | 0.647 -> 0.642 | -0.004 [-0.011, +0.002] | +0.002 [-0.007, +0.009] | yes |
| esrgan: native | curricularface | both | 0.785 -> 0.799 | +0.015 [-0.002, +0.031] | 0.647 -> 0.632 | -0.015 [-0.038, +0.007] | +0.030 [+0.003, +0.058] | no |
| esrgan: native | curricularface | probes | 0.785 -> 0.764 | -0.020 [-0.037, -0.005] | 0.647 -> 0.598 | -0.049 [-0.069, -0.030] | +0.029 [+0.004, +0.053] | no |
| invsr: pre64_st1 | curricularface | both | 0.785 -> 0.767 | -0.018 [-0.039, +0.001] | 0.647 -> 0.567 | -0.080 [-0.106, -0.054] | +0.062 [+0.031, +0.095] | no |
| invsr: pre64_st1 | curricularface | probes | 0.785 -> 0.728 | -0.056 [-0.076, -0.038] | 0.647 -> 0.561 | -0.086 [-0.110, -0.064] | +0.029 [-0.001, +0.061] | yes |
| invsr: pre128_st1 | curricularface | both | 0.785 -> 0.791 | +0.006 [-0.012, +0.024] | 0.647 -> 0.597 | -0.049 [-0.073, -0.027] | +0.055 [+0.026, +0.083] | no |
| invsr: pre128_st1 | curricularface | probes | 0.785 -> 0.741 | -0.043 [-0.063, -0.024] | 0.647 -> 0.596 | -0.050 [-0.070, -0.031] | +0.007 [-0.021, +0.032] | yes |
| tadsr: pre128_t250 | curricularface | both | 0.785 -> 0.831 | +0.046 [+0.030, +0.062] | 0.647 -> 0.615 | -0.031 [-0.047, -0.016] | +0.077 [+0.054, +0.100] | no |
| tadsr: pre128_t250 | curricularface | probes | 0.785 -> 0.777 | -0.008 [-0.025, +0.008] | 0.647 -> 0.618 | -0.029 [-0.042, -0.015] | +0.021 [-0.001, +0.041] | yes |
| osdface: pre128 | curricularface | both | 0.785 -> 0.842 | +0.057 [+0.038, +0.074] | 0.647 -> 0.672 | +0.025 [+0.000, +0.053] | +0.032 [-0.000, +0.064] | yes |
| osdface: pre128 | curricularface | probes | 0.785 -> 0.762 | -0.022 [-0.044, -0.004] | 0.647 -> 0.600 | -0.046 [-0.074, -0.016] | +0.024 [-0.010, +0.058] | yes |

**Verification d'**

| Variant | Recognizer | Mode | TinyFace direct -> restored | TinyFace change | SurvFace direct -> restored | SurvFace change | Gap (TinyFace - SurvFace) | CI contains 0 |
|---|---|---|---|---|---|---|---|---|
| bicubic: pre512 | adaface | both | 1.229 -> 1.246 | +0.017 [-0.003, +0.038] | 0.564 -> 0.573 | +0.009 [-0.022, +0.044] | +0.007 [-0.033, +0.046] | yes |
| bicubic: pre512 | adaface | probes | 1.229 -> 1.235 | +0.006 [-0.010, +0.022] | 0.564 -> 0.551 | -0.012 [-0.035, +0.012] | +0.018 [-0.010, +0.047] | yes |
| esrgan: native | adaface | both | 1.229 -> 1.215 | -0.014 [-0.079, +0.052] | 0.564 -> 0.639 | +0.075 [-0.002, +0.158] | -0.089 [-0.197, +0.016] | yes |
| esrgan: native | adaface | probes | 1.229 -> 1.114 | -0.115 [-0.181, -0.052] | 0.564 -> 0.468 | -0.096 [-0.167, -0.026] | -0.019 [-0.114, +0.076] | yes |
| invsr: pre64_st1 | adaface | both | 1.229 -> 1.008 | -0.221 [-0.292, -0.153] | 0.564 -> 0.304 | -0.260 [-0.351, -0.169] | +0.039 [-0.077, +0.160] | yes |
| invsr: pre64_st1 | adaface | probes | 1.229 -> 0.896 | -0.333 [-0.412, -0.258] | 0.564 -> 0.298 | -0.266 [-0.339, -0.193] | -0.068 [-0.171, +0.039] | yes |
| invsr: pre128_st1 | adaface | both | 1.229 -> 1.056 | -0.173 [-0.246, -0.102] | 0.564 -> 0.372 | -0.192 [-0.274, -0.102] | +0.019 [-0.092, +0.126] | yes |
| invsr: pre128_st1 | adaface | probes | 1.229 -> 1.025 | -0.204 [-0.276, -0.136] | 0.564 -> 0.349 | -0.215 [-0.286, -0.143] | +0.011 [-0.092, +0.104] | yes |
| tadsr: pre128_t250 | adaface | both | 1.229 -> 1.329 | +0.100 [+0.040, +0.169] | 0.564 -> 0.548 | -0.016 [-0.072, +0.041] | +0.116 [+0.031, +0.201] | no |
| tadsr: pre128_t250 | adaface | probes | 1.229 -> 1.266 | +0.037 [-0.023, +0.099] | 0.564 -> 0.492 | -0.071 [-0.113, -0.030] | +0.108 [+0.037, +0.179] | no |
| osdface: pre128 | adaface | both | 1.229 -> 1.385 | +0.156 [+0.084, +0.228] | 0.564 -> 0.747 | +0.183 [+0.087, +0.275] | -0.027 [-0.144, +0.098] | yes |
| osdface: pre128 | adaface | probes | 1.229 -> 1.141 | -0.088 [-0.162, -0.016] | 0.564 -> 0.481 | -0.083 [-0.182, +0.010] | -0.005 [-0.127, +0.121] | yes |
| bicubic: pre512 | magface | both | 1.066 -> 1.052 | -0.014 [-0.031, +0.003] | 0.647 -> 0.643 | -0.004 [-0.026, +0.019] | -0.011 [-0.039, +0.017] | yes |
| bicubic: pre512 | magface | probes | 1.066 -> 1.056 | -0.010 [-0.022, +0.001] | 0.647 -> 0.644 | -0.002 [-0.018, +0.014] | -0.008 [-0.029, +0.011] | yes |
| esrgan: native | magface | both | 1.066 -> 1.081 | +0.015 [-0.051, +0.085] | 0.647 -> 0.635 | -0.012 [-0.079, +0.063] | +0.027 [-0.075, +0.126] | yes |
| esrgan: native | magface | probes | 1.066 -> 0.994 | -0.072 [-0.124, -0.017] | 0.647 -> 0.499 | -0.147 [-0.207, -0.086] | +0.075 [-0.009, +0.155] | yes |
| invsr: pre64_st1 | magface | both | 1.066 -> 0.930 | -0.136 [-0.209, -0.063] | 0.647 -> 0.347 | -0.299 [-0.383, -0.218] | +0.164 [+0.056, +0.271] | no |
| invsr: pre64_st1 | magface | probes | 1.066 -> 0.869 | -0.197 [-0.268, -0.127] | 0.647 -> 0.308 | -0.339 [-0.406, -0.267] | +0.141 [+0.044, +0.237] | no |
| invsr: pre128_st1 | magface | both | 1.066 -> 1.033 | -0.033 [-0.111, +0.039] | 0.647 -> 0.459 | -0.188 [-0.264, -0.111] | +0.155 [+0.045, +0.258] | no |
| invsr: pre128_st1 | magface | probes | 1.066 -> 0.922 | -0.144 [-0.211, -0.075] | 0.647 -> 0.430 | -0.217 [-0.274, -0.155] | +0.073 [-0.023, +0.170] | yes |
| tadsr: pre128_t250 | magface | both | 1.066 -> 1.260 | +0.194 [+0.125, +0.264] | 0.647 -> 0.547 | -0.100 [-0.143, -0.052] | +0.294 [+0.210, +0.379] | no |
| tadsr: pre128_t250 | magface | probes | 1.066 -> 1.075 | +0.009 [-0.053, +0.070] | 0.647 -> 0.565 | -0.082 [-0.117, -0.044] | +0.091 [+0.021, +0.160] | no |
| osdface: pre128 | magface | both | 1.066 -> 1.213 | +0.146 [+0.066, +0.233] | 0.647 -> 0.656 | +0.009 [-0.083, +0.104] | +0.137 [+0.019, +0.266] | no |
| osdface: pre128 | magface | probes | 1.066 -> 1.046 | -0.021 [-0.100, +0.068] | 0.647 -> 0.443 | -0.204 [-0.293, -0.121] | +0.183 [+0.071, +0.315] | no |
| bicubic: pre512 | curricularface | both | 1.122 -> 1.102 | -0.020 [-0.044, +0.006] | 0.545 -> 0.540 | -0.005 [-0.039, +0.027] | -0.014 [-0.053, +0.031] | yes |
| bicubic: pre512 | curricularface | probes | 1.122 -> 1.109 | -0.013 [-0.032, +0.007] | 0.545 -> 0.533 | -0.012 [-0.036, +0.013] | -0.001 [-0.032, +0.031] | yes |
| esrgan: native | curricularface | both | 1.122 -> 1.159 | +0.038 [-0.032, +0.111] | 0.545 -> 0.492 | -0.053 [-0.135, +0.027] | +0.090 [-0.014, +0.206] | yes |
| esrgan: native | curricularface | probes | 1.122 -> 1.028 | -0.094 [-0.161, -0.020] | 0.545 -> 0.348 | -0.196 [-0.263, -0.131] | +0.102 [+0.009, +0.205] | no |
| invsr: pre64_st1 | curricularface | both | 1.122 -> 0.999 | -0.122 [-0.199, -0.045] | 0.545 -> 0.251 | -0.294 [-0.388, -0.198] | +0.172 [+0.051, +0.285] | no |
| invsr: pre64_st1 | curricularface | probes | 1.122 -> 0.862 | -0.260 [-0.347, -0.186] | 0.545 -> 0.209 | -0.336 [-0.417, -0.255] | +0.076 [-0.037, +0.183] | yes |
| invsr: pre128_st1 | curricularface | both | 1.122 -> 1.066 | -0.056 [-0.133, +0.018] | 0.545 -> 0.355 | -0.190 [-0.275, -0.105] | +0.134 [+0.022, +0.244] | no |
| invsr: pre128_st1 | curricularface | probes | 1.122 -> 0.900 | -0.222 [-0.309, -0.145] | 0.545 -> 0.346 | -0.199 [-0.261, -0.128] | -0.023 [-0.135, +0.067] | yes |
| tadsr: pre128_t250 | curricularface | both | 1.122 -> 1.244 | +0.123 [+0.056, +0.190] | 0.545 -> 0.422 | -0.122 [-0.178, -0.068] | +0.245 [+0.159, +0.334] | no |
| tadsr: pre128_t250 | curricularface | probes | 1.122 -> 1.072 | -0.050 [-0.124, +0.021] | 0.545 -> 0.432 | -0.113 [-0.162, -0.065] | +0.063 [-0.022, +0.148] | yes |
| osdface: pre128 | curricularface | both | 1.122 -> 1.326 | +0.205 [+0.119, +0.287] | 0.545 -> 0.660 | +0.115 [+0.017, +0.206] | +0.089 [-0.045, +0.217] | yes |
| osdface: pre128 | curricularface | probes | 1.122 -> 1.023 | -0.099 [-0.190, -0.008] | 0.545 -> 0.388 | -0.157 [-0.259, -0.066] | +0.058 [-0.077, +0.203] | yes |

## 4. Hypotheses
- **H1 (protocol artefact): supported** by the pre-registered rule (Rank-1 gap CI contains zero in 26 of 36 cells, AUC in 20 of 36; d' in 23 of 36 is shown for information and does not decide). The 10 Rank-1 cells where it does not contain zero: 7 favour TinyFace (Real-ESRGAN, TADSR and OSDFace with both sides restored, and OSDFace probes-only for AdaFace) and 3 favour SurvFace (bicubic MagFace both sides; InvSR probes-only for MagFace and CurricularFace). All 16 AUC cells where it does not contain zero favour TinyFace.
- **H2, H3, H5, H6: not run.** The decision rule for E0 says to stop after H4 when H1 is supported. The code for them exists in the script and was tested on a small forced run that is not a result. During that test one data property was found: every mated SurvFace probe is compared with gallery images from a different camera (0 same-camera comparisons out of 340,677 per recognizer), so the pre-registered same-camera against cross-camera test of H5 could not be run on mated pairs even if H5 were run. The script therefore also computes a supplementary gallery-against-gallery comparison, which has not been run for the reported results.
- **H4 (genuine / impostor decomposition): supported** (`h4_genuine_impostor.csv`, `h4_spearman.csv`). In every dataset, recognizer and mode group, the variants with the larger loss in verification AUC are the ones where the mean impostor cosine rises more than the mean genuine cosine.
- **H7 (quality inflation): deferred**, not run.

## 5. What this means for the SR closure claim
The claim that restoration does not help recognition on these crops holds on both datasets, and the SurvFace-versus-TinyFace difference in the earlier tables is mostly due to measuring them with different protocols: once both are scored with the same closed-set and verification metrics, the absolute change is similar in size and the dataset difference is inside the bootstrap error in most cells. The caveat is that on verification AUC the generative arms with both sides restored help TinyFace slightly more than SurvFace, and SurvFace's lower direct-feed baseline makes the same absolute loss a larger relative loss. The mechanism visible on both datasets is that restoration pushes impostor similarity up relative to genuine similarity. None of this changes the closure of restoration; it only narrows the claim: the penalty is not specific to SurvFace.

## 6. Limitations
- **Subset sizes and identity counts:** 800 TinyFace identities (1,155 probes, 5,373 gallery and distractor images) against 200 SurvFace identities (4,476 mated probes, 4,486 gallery images). The SurvFace closed-set Rank-1 is not an official SurvFace protocol, and TinyFace Rank-1 includes 4,000 distractors while SurvFace has none, so the Rank-1 levels are not comparable across datasets; only the changes are.
- **The H1 rule is marginal for AUC (20 of 36 cells), and "most" was operationalised as "more than half" after the pre-registration was committed** (addendum, before any number). A stricter reading would not support H1 for AUC.
- **H2, H3, H5 and H6 were not run, by the decision rule.** The explanation for the residual SurvFace gap (crop size, distractors, camera, threshold) is therefore untested. Running them would be an extra, non-pre-registered step.
- **TinyFace verification pairs** are one seeded sample of 1,000 genuine and 1,000 impostor pairs; another seed would give somewhat different numbers.
- **Observational, not causal:** the H4 correlation rests on 6 variants per group.
- **Devices:** the OSDFace embeddings were computed on a Colab T4, all other embeddings (including the direct-feed baseline) on the lab PC. OSDFace changes therefore compare T4 restored embeddings with lab-PC direct embeddings; the Phase 3 cross-device check is the only evidence that this is harmless.
- **ArcFace** was left out (leaky for OSDFace; the plan allows it only as an appendix).
- **Multiple tests:** 36 cells per metric. The plain CIs decide; the Bonferroni-adjusted CIs (family of 36) would turn the AUC count into 27 of 36 and the Rank-1 count into 32 of 36, which strengthens H1.

**STOP: awaiting supervisor review.**


---

## Second addendum: exploratory extension (committed before the extension was run)

After the results above, Abser asked for two further things: to run H2, H3, H5 and H6 although the pre-registered stop rule (H1 supported) says to stop, and to run H7 for the four variants that have full-size restored PNGs on the lab PC. This is **exploratory**: it was not part of the pre-registered plan, the stop rule is deliberately overridden, and the verdicts below it describe what the data show; they do not reopen the verdict on H1. The first-pass results and verdicts above are not changed. No number of this extension existed when this addendum was written.

1. **H2, H3, H5 and H6** are run exactly as defined in the first addendum, with the same code (`scripts/extra_sr_asymmetry.py --exploratory-stage2`), in a separate output folder `outputs/sr/results/asymmetry_exploratory/` and marked `exploratory` in `verdicts.csv`. The same script re-computes E0 and H4 there; those files must equal the first-pass files (hash comparison). Known limit, found while testing the script: H5's same-camera against cross-camera test on mated comparisons has no same-camera comparisons, so only the kNN part and the supplementary gallery-against-gallery comparison can be evaluated. The script is run twice and the outputs must be byte-identical.
2. **H7 is new inference** and therefore breaks the "no new inference" constraint of the first plan; Abser authorised it. It is run by `scripts/extra_sr_asymmetry_h7.py` on the lab-PC GPU with the MagFace weights of Phase 1, on the full-size restored PNGs of bicubic `pre512`, Real-ESRGAN `native`, InvSR `pre64_st1` and TADSR `pre128_t250`. OSDFace (thumbnails only) and InvSR `pre128_st1` (no stored PNGs) are not covered. Definitions, fixed now:
   - Only the mated probes are embedded (1,155 TinyFace probes, 4,476 SurvFace mated probes); the norm is a property of the probe image.
   - The quality measure is the raw, un-normalised MagFace norm of the unflipped embedding, measured as in Phase 2 (`magface_norm_flip: false`). Direct crops use bilinear resizing to 112 px; restored 512-px outputs use INTER_AREA, as in Phase 3.
   - Preprocessing checks that must pass before any H7 number is used: (a) the flip-summed, normalised embedding of every embedded crop equals the cached Phase 3 MagFace embedding (minimum cosine reported; expected above 0.9999); (b) the direct-feed norm of the TinyFace probes equals the Phase 2 `magface_norm` of the same images (maximum relative difference reported).
   - For each variant and dataset: the change in norm (restored minus direct, per probe) with a paired 95% bootstrap CI (1,000 draws, seed 42), the relative change, and the change in genuine similarity (the H2 per-probe quantity, MagFace cached embeddings, both sides restored), also with its CI.
   - The **larger-penalty dataset** of a variant is the one with the more negative change in MagFace verification AUC (both sides restored) in `asym_common_scale.csv`.
   - **Criterion per variant:** on the larger-penalty dataset the CI of the mean norm change is entirely above zero, **and** the CI of the mean genuine-similarity change is not entirely above zero (no matching gain). The verdict rule is the one of the first addendum (supported if more than half of the 4 variants meet it, not supported if fewer than a third, else inconclusive).
   - The Spearman correlation between the per-probe norm change and the per-probe genuine-similarity change is reported for each variant and dataset as a descriptive number only.
   - The MagFace norm is a property of the MagFace model, so H7 uses MagFace only.


---

# Exploratory extension: results (H2, H3, H5, H6 and H7 on 4 variants)

This section follows the second addendum. It is exploratory: the pre-registered stop rule was deliberately overridden at Abser's request, and H7 needed new MagFace inference. The verdicts above (H1 supported, H4 supported) are unchanged. Files are in `outputs/sr/results/asymmetry_exploratory/`. The stage-2 script and the H7 script were each run twice and every output file is byte-identical (`determinism_hashes.txt`). The first-pass files in that folder (E0, H4, the TinyFace pair list) are identical to those in `outputs/sr/results/asymmetry/`, and the verdict table marks H2, H3, H5 and H6 as `exploratory`.

## Summary of the extension
- **None of H2, H3, H5 and H6 explains the residual SurvFace gap.** All four are "not supported" by the verdict rule of the first addendum (H2 1 of 18 cells, H3 0 of 18, H5 1 of 18, H6 4 of 36).
- **H7 is supported in 3 of 4 variants** (Real-ESRGAN, InvSR pre-64, TADSR, all on SurvFace, which is the larger-penalty dataset for them): MagFace's raw norm goes up while the genuine similarity goes down. The effect is small for InvSR and TADSR (+0.34 and +0.36 on a norm of about 21) and larger for Real-ESRGAN (+1.23, about +6%). It is not specific to the larger-penalty dataset: Real-ESRGAN raises the norm on TinyFace as much (+1.26).
- **Preprocessing checks for H7 passed:** the new flip-summed embeddings equal the cached Phase 3 MagFace embeddings (minimum cosine 0.9999998 for direct feed and all four variants), and the new direct-feed norms of the 1,155 TinyFace probes equal the Phase 2 measure (maximum relative difference 5.3e-05).

## H2 (input information): not supported
Per probe, the change in genuine similarity (restored minus direct) was compared between the datasets on matched strata (`h2_matched_strata.csv`, `h2_binned_delta_genuine.csv`). With the pre-registered strata (`size_q` × `sharp_bin_sz`, 6 strata present in both datasets) the matched gap is at most half the unmatched gap in 1 of 18 cells; the median shrinkage is -16%, so matching usually makes the gap larger. The sensitivity version with pooled pixel-size strata (8 common strata) meets the criterion in 3 of 18 cells (median shrinkage +1%). The unmatched gap itself is small to begin with (mean -0.0035 over the 18 cells), so "shrinks by half" has little to shrink. TinyFace probes are concentrated in one size bin (1,088 of 1,155 are 25 to 32 px) while SurvFace spans all bins, so the overlap between the datasets is limited, and the pre-registered strata are matched by rank within each dataset, not by pixel size (first addendum, point 7).

## H3 (distractor composition, TinyFace only): not supported
Keeping the 4,000 distractors at direct feed changes the TinyFace Rank-1 change by +0.0085 on average (over 18 cells), against a mean CI half-width of 0.0215; no cell exceeds its half-width (`h3_distractors.csv`). The change in the best impostor cosine per probe (restored minus direct, both sides restored) averages -0.051 for gallery-match impostors and -0.068 for distractor impostors (`h3_impostor_split.csv`); these are best-match scores, not the pair means of H4.

## H5 (camera domain): not supported
- **Camera predictability** (leave-one-out kNN, k = 5, on the SurvFace gallery): direct feed 0.720 on average; restoration changes it by -0.050 on average. The CI is above zero in 1 of 18 cells and below zero in 11 of 18, so restoration mostly makes the camera less predictable, not more (`h5_knn_camera.csv`).
- **Same-camera against cross-camera on mated comparisons: not testable.** Every mated query-gallery pair (340,677 per recognizer) is cross-camera (`h5_camera_pairs.csv`).
- **Supplementary (not part of the verdict):** gallery-against-gallery pairs of the same identity (171,985 same-camera, 80,046 cross-camera). The mean cosine change is -0.077 for same-camera and -0.054 for cross-camera pairs, so restoration hurts same-camera pairs more; the pre-registered direction (cross-camera hurt more) holds in 2 of 18 cells (`h5_supplementary_gallery_pairs.csv`).

## H6 (open-set threshold, SurvFace): not supported
The change in TPIR@FPIR=0.3 was split into a mated-score part (the direct-feed threshold applied to the restored scores) and a threshold-shift part (the remainder). Averaged over the cells, with both sides restored the total change is -0.021, the mated-score part is -0.052 and the threshold-shift part is +0.030; with probes only the total is -0.050, the mated-score part -0.098 and the threshold-shift part +0.048 (`h6_threshold_split.csv`). The threshold shift is larger in absolute size than the mated part in 4 of 36 cells. The mated scores drive the loss, and the threshold moves in the opposite direction (the threshold-shift part is positive in 32 of 36 cells, the mated part is negative in 34 of 36). The split follows the first addendum, point 11.

## H7 (quality inflation, 4 variants): supported in 3 of 4
| Variant | Dataset | Larger penalty (MagFace AUC change, both sides) | Mean norm direct -> restored | Norm change [95% CI] | Genuine-similarity change [95% CI] | Spearman (per probe) | Criterion met |
|---|---|---|---|---|---|---|---|
| bicubic: pre512 | tinyface | yes (-0.0024) | 22.21 -> 22.11 | -0.098 [-0.137, -0.061] | +0.0034 [+0.0017, +0.0050] | -0.03 | no |
| bicubic: pre512 | survface | no (-0.0018) | 21.28 -> 21.61 | +0.321 [+0.299, +0.345] | -0.0035 [-0.0042, -0.0027] | -0.06 | no |
| esrgan: native | tinyface | no (+0.0069) | 22.21 -> 23.46 | +1.257 [+1.117, +1.391] | -0.0356 [-0.0420, -0.0288] | +0.09 | no |
| esrgan: native | survface | yes (-0.0015) | 21.28 -> 22.51 | +1.231 [+1.160, +1.303] | -0.0071 [-0.0097, -0.0046] | -0.04 | yes |
| invsr: pre64_st1 | tinyface | no (-0.0236) | 22.21 -> 22.45 | +0.239 [+0.089, +0.387] | -0.0894 [-0.0967, -0.0817] | +0.06 | no |
| invsr: pre64_st1 | survface | yes (-0.0790) | 21.28 -> 21.62 | +0.339 [+0.242, +0.430] | -0.1045 [-0.1077, -0.1013] | -0.15 | yes |
| tadsr: pre128_t250 | tinyface | no (+0.0541) | 22.21 -> 22.00 | -0.212 [-0.332, -0.101] | -0.0845 [-0.0897, -0.0784] | +0.13 | no |
| tadsr: pre128_t250 | survface | yes (-0.0263) | 21.28 -> 21.64 | +0.361 [+0.317, +0.408] | -0.0492 [-0.0510, -0.0476] | +0.04 | yes |

Reading: a variant meets the criterion if, on the larger-penalty dataset, the norm CI is above zero and the genuine-similarity CI is not above zero. Real-ESRGAN, InvSR pre-64 and TADSR meet it on SurvFace. Bicubic does not: its larger-penalty dataset is TinyFace, where the norm falls slightly. The penalties that define the larger-penalty dataset are tiny for bicubic and Real-ESRGAN (differences of 0.0006 and 0.0084 in AUC change), so that choice is fragile there. Per probe, the norm change and the genuine-similarity change are only weakly related (Spearman -0.15 to +0.13), so a larger norm gain does not go with a larger loss of similarity for the same crop. Only OSDFace and InvSR pre-128 are not covered, because their full-size restorations are not on this PC.

## What the extension adds to section 5
The first-pass conclusion stands: the SurvFace-versus-TinyFace difference is mostly a protocol artefact (H1) and restoration makes faces more alike on both datasets (H4). The extension rules out four candidate explanations for the residual gap in this data (crop size and sharpness, distractor composition, camera domain, open-set threshold), and finds that restored crops tend to look higher quality to MagFace without being more similar to their own identity (H7). That quality inflation appears on both datasets and is strongest for Real-ESRGAN, so it does not explain why SurvFace would be hit harder; it is a general property of some restorers.

## Limitations of the extension
- Exploratory: the stop rule was overridden, so the five verdicts are not pre-registered conclusions; no correction was made for running them after H1.
- H7 covers 4 of 6 variants and MagFace only (the norm is a MagFace property), and uses new inference on the lab-PC GPU; OSDFace and InvSR pre-128 are missing.
- H2's matching has the limits described above (rank-matched strata, little size overlap between the datasets).
- H5's pre-registered same-camera test is impossible on mated pairs; the supplementary comparison uses gallery pairs, which are a different population.
- The H6 split is one of two reasonable decompositions (the other order would give different parts); the sum is fixed.
