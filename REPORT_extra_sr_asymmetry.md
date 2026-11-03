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
