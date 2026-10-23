# REPORT — Phase 4: non-generative low-light and deblur test (draft)

**Status:** draft. Sections 1-8, 10 and 11 are filled after the eval subset is scored. The rules in section 0 were written and committed **before** the eval subset was scored (see the git history of this file).

## 0. Pre-registered decision rules (fixed before the eval subset is scored)
Question: does any low-light or deblur method give a **probes-only** recognition gain with a 95% CI above zero? Yes: the winner becomes the gate arm for Phase 8. No: restoration is closed.

1. **Stop rule.**
   - Restoration closes if no selected arm or gated arm gives a probes-only TinyFace Rank-1 paired gain whose 95% CI is above zero for at least 2 of the 3 non-leaky recognizers (AdaFace, MagFace, CurricularFace).
   - Otherwise the winner is the arm with the highest mean probes-only gain, and it becomes the Phase 8 gate arm.
   - Up to 10 tests run. Bonferroni-adjusted CIs are shown as a sensitivity check.
2. **LFW check.**
   - An arm "recovers identity" if the paired CI of AUC(cross-enhanced) − AUC(cross-degraded) is above zero for at least 2 of the 3 non-leaky recognizers.
   - Each arm is also compared post-hoc against no-op.
3. **Harness sanity.** `noop` embeddings must equal direct-feed embeddings exactly (`np.array_equal`), on the same device. If not, STOP.

**Selection (unchanged from Phase 3).** Best mean over AdaFace, MagFace and CurricularFace of [TinyFace Rank-1, SurvFace verification AUC] on the tuning subset (2,800 crops). One setting per arm; one τ per gated family, chosen on that family's selected arm. Gate τ candidates: the 10th, 20th, 30th and 50th percentiles of the pooled tuning distribution of raw luminance (low-light gate) or raw `lap_var` (deblur gate); one global threshold.

**Harness sanity (done before the sweep).** On the RTX 3000, `noop` embeddings equal direct embeddings for all 4 recognizers on both parts (`np.array_equal`), and the noop paired Rank-1 delta is exactly 0. Cross-device check (`outputs/enhance/results/xcheck.json`): per-crop cosine to the lab-PC direct embeddings is above 0.9999 for all 4 recognizers (minimum 0.999987).

## 1. Summary and stop-rule verdict
_To be filled._

## 2. What was run
_To be filled (device Quadro RTX 3000 6 GB, torch 2.6.0+cu124, cross-device check, runtimes)._

## 3. Findings per arm
_To be filled._

## 4. Gated variants
_To be filled._

## 5. Probes-only and per-bin (`dark`, `dark10`, `low_sharpness`)
<!-- TABLES:BEGIN -->
<!-- TABLES:END -->
<!-- EXTRA:BEGIN -->
<!-- EXTRA:END -->

## 6. LFW synthetic check (dark, Gaussian blur, motion blur)
<!-- LFW:BEGIN -->
<!-- LFW:END -->

## 7. Labelled visual samples
_To be filled._

## 8. Cost: s/crop on CPU and GPU
_To be filled. Smoke run (75 crops, median s/crop, CPU / GPU): Zero-DCE++ 0.0097 / 0.0109; NAFNet native 0.029 / 0.0109; NAFNet pre-128 0.331 / 0.0236._

## 9. Full tables (generated)
_Injected between the markers above._

## 10. Limitations
- The `lap_var` gate is size-confounded (Phase 2).
- Size of the dark bin.
- Synthetic LFW degradations.
- Multiplicity (up to 10 tests).
- A different GPU from Phase 3 (RTX 3000 laptop, not the lab PC).

## 11. Decision for Phase 8
_The gate arm with its τ, or "restoration closed"._
