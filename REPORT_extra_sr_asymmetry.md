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
