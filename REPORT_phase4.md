# REPORT — Phase 4: non-generative low-light and deblur test

**Status:** complete, pending supervisor approval (end-of-W2 checkpoint). The rules in section 0 were written and committed **before** the eval subset was scored (see the git history of this file).

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
**Verdict (pre-registered stop rule, section 0): restoration is closed.** None of the 6 selected arms and 2 gated variants gives a probes-only TinyFace Rank-1 paired gain with a 95% CI above zero for any non-leaky recognizer (0 of 3 for every arm, with or without the Bonferroni adjustment over 8 tests). The stop-rule table at the top of section 5 has the per-recognizer gains.

- **Real degraded faces:** the gains are zero or negative. Mean probes-only gain over AdaFace, MagFace and CurricularFace: gate_deblur -0.0003, gate_lowlight -0.0040, unsharp -0.0049, bicubic -0.0063, gamma -0.0089, NAFNet -0.0271, Zero-DCE++ -0.0364, CLAHE -0.0436. The CI is entirely below zero for every recognizer for CLAHE, Zero-DCE++ and NAFNet.
- **Dark and blurry subsets:** no arm has a CI above zero, and every mean change is at or below zero, in the dark bin (236 probes), the darkest-10% bin (98 probes) or the low-sharpness bin (217 probes); see section 5. Dark10 is small, so its intervals are wide.
- **Tuning subset:** no selected arm scored above the no-op on the tuning score (noop 0.5728). The deblur gate at q50 reached 0.5763 on tuning; on the eval subset its probes-only gain is -0.0003 (CI includes zero).
- **LFW synthetic check (section 6):** on Gaussian blur (sigma 8) NAFNet recovers identity by the pre-registered rule (cross-AUC +0.073 to +0.092, 3 of 3 non-leaky recognizers). On motion blur it meets the rule formally, but the effect is +0.0006 to +0.0009 on a baseline of 0.9985, so it is a ceiling effect, not a finding. No low-light arm meets the rule (Zero-DCE++ 1 of 3, gamma and CLAHE 0 of 3).
- **Reading:** a deblurring network can undo synthetic Gaussian blur, but the same network lowers recognition on the real TinyFace and SurvFace probes, where the degradation is not Gaussian blur. Low-light methods do not help even on the synthetic dark set.

## 2. What was run
- **Device:** Raghib's laptop, Quadro RTX 3000 (6 GB), torch 2.6.0+cu124, recognizer batch 128. Every reported number, including the direct-feed baseline, was computed on this one device, so paired deltas never mix devices.
- **Cross-device check** (`xcheck.json`): direct-feed embeddings of the 2,800 tuning crops against the lab-PC embeddings; per-crop cosine above 0.9999 for all four recognizers (minimum 0.999996 ArcFace, 0.999990 AdaFace, 0.999998 MagFace, 0.999987 CurricularFace). The embeddings are not bit-identical across devices, as expected.
- **Harness sanity:** `noop` embeddings equal direct-feed embeddings (`np.array_equal`) for all four recognizers on both tuning and eval, and the noop paired Rank-1 delta is exactly 0.
- **Harness:** the Phase 3 harness with the changes listed in the runbook (results path and batch size from the config, generic arms, per-variant embedding rule). The Phase 3 report tables regenerate unchanged except for the "Remote device" column, which needs Colab timing files that are not on this laptop.
- **Arms and tuning:** 16 variants on the tuning subset (2,800 crops), 0 failed crops. Selected on the Phase 3 rule: bicubic pre-128, gamma adaptive, CLAHE clip 2 / 2x2 tiles, unsharp sigma 0.5 / amount 0.5, Zero-DCE++ scale_factor 1, NAFNet native. Each was run on the eval subset (22,898 crops) with 0 failed crops.
- **Gates:** chosen on tuning only. Low-light: gamma-adaptive where raw luminance is below 60.17 (q10; 8.8% of eval crops enhanced). Deblur: unsharp where raw lap_var is below 3.369 (q50; 55.6% of eval crops enhanced). Gating sanity (tau = 0 equals direct, tau = infinity equals the full arm, embedding equality) passed for both families.
- **Runtimes** are in section 8.

## 3. Findings per arm
Non-leaky recognizers (AdaFace, MagFace, CurricularFace); probes-only TinyFace Rank-1 gain, mean over the three (full CIs in section 5). For reference, mean direct-feed Rank-1 is 0.3232.

- **bicubic pre-128 (control):** -0.0063. Interpolation alone does nothing useful here.
- **gamma (adaptive):** -0.0089. Brightens dark crops but also exposes colour casts and block artefacts (section 7); mean probes-only SurvFace AUC is 0.651 against 0.657 for direct feed.
- **CLAHE (c2, 2x2):** -0.0436, the worst arm. Local contrast amplifies noise and compression blocks; 33 of 1,155 TinyFace probes lose a correct rank-1 match for at least 2 of 3 recognizers.
- **Zero-DCE++ (scale 1):** -0.0364 (22 lost matches). It lifts brightness but adds a pink cast at full resolution; it helps on synthetic dark only for ArcFace and MagFace.
- **unsharp (sigma 0.5, amount 0.5):** -0.0049. Almost no visible change at this size.
- **NAFNet (GoPro, native):** -0.0271 (30 lost matches). It visibly sharpens edges (section 7) but moves the embeddings of real probes away from their gallery images; on synthetic Gaussian blur it improves identity (section 6).
- **Tuning vs eval:** no selected arm beat the no-op on tuning and none does on eval. The tuning-set AUC gains of NAFNet native (0.672 against 0.663) and the stronger unsharp settings did not translate into Rank-1 gains.

## 4. Gated variants
- **Low-light gate** (gamma-adaptive below luminance 60.17): probes-only gain -0.0040 (AdaFace -0.0043 [-0.0087, -0.0009], MagFace -0.0043 [-0.0095, +0.0000], CurricularFace -0.0035 [-0.0078, +0.0000]). Gating roughly halves the loss of the full arm (-0.0089 to -0.0040) but does not turn it positive.
- **Deblur gate** (unsharp below lap_var 3.369): probes-only gain -0.0003; every interval includes or touches zero. It is the closest to the direct feed of all variants, because it changes little.
- Neither gate meets the stop rule. The lap_var gate is size-confounded (Phase 2): small crops tend to have low lap_var, so the gate partly selects by crop size (limitations, section 10).

## 5. Probes-only and per-bin (`dark`, `dark10`, `low_sharpness`)
<!-- TABLES:BEGIN -->
Eval subset: 22898 crops; tuning subset: 2800 crops. Values: point estimate [95% bootstrap CI, 1,000 draws]. † = leaky recognizer (OSDFace's identity loss uses ArcFace); conclusions rest on the other three.

**Stop rule: probes-only TinyFace Rank-1 paired gain (eval subset, non-leaky recognizers), 8 tests**

Gain = probes restored / enhanced, gallery and distractors direct, minus direct feed, paired on the same probes, 95% bootstrap CI. 'Bonferroni' is the same gain with the CI at 0.05 / number of tests (sensitivity check). Rule: closes restoration unless some arm has the CI above zero for at least 2 of the 3 recognizers.

| Arm | AdaFace | MagFace | CurricularFace | Mean gain | CI > 0 (of 3) | Bonferroni CI > 0 (of 3) |
|---|---|---|---|---|---|---|
| bicubic: pre128 | +0.0035 [-0.0078, +0.0156] | -0.0173 [-0.0286, -0.0061] | -0.0052 [-0.0173, +0.0069] | -0.0063 | 0 | 0 |
| gamma: gadaptive | -0.0113 [-0.0182, -0.0052] | -0.0069 [-0.0147, +0.0009] | -0.0087 [-0.0165, -0.0017] | -0.0089 | 0 | 0 |
| clahe: c2_t2 | -0.0468 [-0.0623, -0.0294] | -0.0381 [-0.0528, -0.0234] | -0.0459 [-0.0623, -0.0303] | -0.0436 | 0 | 0 |
| unsharp: a0.5_s0.5 | -0.0095 [-0.0182, -0.0009] | -0.0035 [-0.0121, +0.0044] | -0.0017 [-0.0087, +0.0061] | -0.0049 | 0 | 0 |
| zerodce: s1 | -0.0381 [-0.0545, -0.0208] | -0.0381 [-0.0528, -0.0234] | -0.0329 [-0.0502, -0.0165] | -0.0364 | 0 | 0 |
| nafnet: pre0 | -0.0294 [-0.0433, -0.0147] | -0.0286 [-0.0433, -0.0156] | -0.0234 [-0.0364, -0.0113] | -0.0271 | 0 | 0 |
| gate_lowlight: q10__gamma_gadaptive | -0.0043 [-0.0087, -0.0009] | -0.0043 [-0.0095, +0.0000] | -0.0035 [-0.0078, +0.0000] | -0.0040 | 0 | 0 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | -0.0009 [-0.0026, +0.0000] | +0.0000 [-0.0026, +0.0026] | +0.0000 [-0.0026, +0.0026] | -0.0003 | 0 | 0 |

**Verdict (mechanical): no arm meets the rule: restoration is closed.**

### TinyFace Rank-1 (closed-set identification)

| Arm | ArcFace | AdaFace | MagFace | CurricularFace |
|---|---|---|---|---|
| Direct feed (resize) | 0.302 [0.276, 0.328] | 0.336 [0.309, 0.363] | 0.336 [0.310, 0.362] | 0.298 [0.271, 0.325] |
| No-op (control): default | 0.302 [0.276, 0.328] | 0.336 [0.309, 0.363] | 0.336 [0.310, 0.362] | 0.298 [0.271, 0.325] |
| bicubic: pre128 | 0.298 [0.274, 0.323] | 0.345 [0.319, 0.371] | 0.318 [0.291, 0.343] | 0.283 [0.258, 0.307] |
| gamma: gadaptive | 0.300 [0.274, 0.328] | 0.325 [0.300, 0.352] | 0.334 [0.307, 0.362] | 0.289 [0.264, 0.315] |
| clahe: c2_t2 | 0.264 [0.240, 0.289] | 0.293 [0.268, 0.319] | 0.295 [0.270, 0.322] | 0.259 [0.235, 0.284] |
| unsharp: a0.5_s0.5 | 0.293 [0.268, 0.319] | 0.327 [0.300, 0.352] | 0.332 [0.305, 0.358] | 0.290 [0.266, 0.316] |
| zerodce: s1 | 0.273 [0.248, 0.297] | 0.307 [0.281, 0.334] | 0.313 [0.288, 0.339] | 0.275 [0.250, 0.302] |
| nafnet: pre0 | 0.281 [0.256, 0.305] | 0.299 [0.274, 0.324] | 0.304 [0.277, 0.331] | 0.285 [0.261, 0.310] |
| Low-light gate on the selected low-light arm: q10__gamma_gadaptive | 0.301 [0.276, 0.328] | 0.328 [0.301, 0.356] | 0.333 [0.308, 0.359] | 0.295 [0.270, 0.322] |
| Deblur gate on the selected deblur arm: q50__unsharp_a0.5_s0.5 | 0.301 [0.275, 0.327] | 0.334 [0.307, 0.360] | 0.334 [0.308, 0.360] | 0.296 [0.269, 0.323] |

### TinyFace mAP

| Arm | ArcFace | AdaFace | MagFace | CurricularFace |
|---|---|---|---|---|
| Direct feed (resize) | 0.222 [0.202, 0.241] | 0.262 [0.240, 0.284] | 0.250 [0.228, 0.271] | 0.224 [0.204, 0.245] |
| No-op (control): default | 0.222 [0.202, 0.241] | 0.262 [0.240, 0.284] | 0.250 [0.228, 0.271] | 0.224 [0.204, 0.245] |
| bicubic: pre128 | 0.222 [0.204, 0.242] | 0.267 [0.245, 0.289] | 0.238 [0.216, 0.258] | 0.212 [0.194, 0.232] |
| gamma: gadaptive | 0.223 [0.203, 0.243] | 0.256 [0.234, 0.277] | 0.247 [0.226, 0.269] | 0.219 [0.199, 0.239] |
| clahe: c2_t2 | 0.191 [0.173, 0.210] | 0.224 [0.204, 0.246] | 0.221 [0.201, 0.242] | 0.188 [0.170, 0.206] |
| unsharp: a0.5_s0.5 | 0.217 [0.198, 0.237] | 0.253 [0.231, 0.275] | 0.248 [0.227, 0.270] | 0.218 [0.199, 0.238] |
| zerodce: s1 | 0.201 [0.183, 0.218] | 0.236 [0.216, 0.257] | 0.232 [0.212, 0.254] | 0.210 [0.190, 0.230] |
| nafnet: pre0 | 0.202 [0.184, 0.220] | 0.228 [0.208, 0.248] | 0.225 [0.205, 0.245] | 0.207 [0.189, 0.226] |
| Low-light gate on the selected low-light arm: q10__gamma_gadaptive | 0.221 [0.201, 0.241] | 0.257 [0.235, 0.279] | 0.247 [0.226, 0.269] | 0.221 [0.201, 0.242] |
| Deblur gate on the selected deblur arm: q50__unsharp_a0.5_s0.5 | 0.221 [0.202, 0.241] | 0.260 [0.239, 0.282] | 0.248 [0.227, 0.270] | 0.224 [0.203, 0.245] |

### SurvFace verification TAR@FAR=0.1

| Arm | ArcFace | AdaFace | MagFace | CurricularFace |
|---|---|---|---|---|
| Direct feed (resize) | 0.305 [0.253, 0.338] | 0.268 [0.222, 0.313] | 0.310 [0.268, 0.343] | 0.279 [0.226, 0.328] |
| No-op (control): default | 0.305 [0.253, 0.338] | 0.268 [0.222, 0.313] | 0.310 [0.268, 0.343] | 0.279 [0.226, 0.328] |
| bicubic: pre128 | 0.316 [0.264, 0.354] | 0.287 [0.241, 0.317] | 0.293 [0.256, 0.338] | 0.265 [0.229, 0.319] |
| gamma: gadaptive | 0.298 [0.250, 0.335] | 0.242 [0.202, 0.289] | 0.322 [0.287, 0.354] | 0.294 [0.247, 0.330] |
| clahe: c2_t2 | 0.278 [0.241, 0.316] | 0.213 [0.163, 0.261] | 0.306 [0.265, 0.347] | 0.264 [0.227, 0.303] |
| unsharp: a0.5_s0.5 | 0.289 [0.253, 0.331] | 0.249 [0.209, 0.287] | 0.302 [0.267, 0.337] | 0.284 [0.246, 0.320] |
| zerodce: s1 | 0.308 [0.258, 0.339] | 0.214 [0.185, 0.262] | 0.329 [0.290, 0.371] | 0.271 [0.229, 0.300] |
| nafnet: pre0 | 0.275 [0.236, 0.314] | 0.211 [0.181, 0.250] | 0.296 [0.254, 0.327] | 0.251 [0.207, 0.301] |
| Low-light gate on the selected low-light arm: q10__gamma_gadaptive | 0.309 [0.256, 0.347] | 0.261 [0.206, 0.299] | 0.322 [0.286, 0.354] | 0.288 [0.229, 0.327] |
| Deblur gate on the selected deblur arm: q50__unsharp_a0.5_s0.5 | 0.295 [0.257, 0.341] | 0.258 [0.219, 0.296] | 0.303 [0.267, 0.337] | 0.293 [0.244, 0.329] |

### SurvFace verification TAR@FAR=0.01

| Arm | ArcFace | AdaFace | MagFace | CurricularFace |
|---|---|---|---|---|
| Direct feed (resize) | 0.077 [0.022, 0.105] | 0.054 [0.033, 0.098] | 0.082 [0.060, 0.147] | 0.069 [0.045, 0.106] |
| No-op (control): default | 0.077 [0.022, 0.105] | 0.054 [0.033, 0.098] | 0.082 [0.060, 0.147] | 0.069 [0.045, 0.106] |
| bicubic: pre128 | 0.084 [0.043, 0.125] | 0.066 [0.024, 0.102] | 0.105 [0.074, 0.142] | 0.083 [0.052, 0.109] |
| gamma: gadaptive | 0.082 [0.042, 0.108] | 0.050 [0.026, 0.080] | 0.086 [0.068, 0.126] | 0.068 [0.045, 0.089] |
| clahe: c2_t2 | 0.099 [0.048, 0.131] | 0.050 [0.027, 0.068] | 0.105 [0.072, 0.136] | 0.061 [0.039, 0.086] |
| unsharp: a0.5_s0.5 | 0.077 [0.020, 0.120] | 0.043 [0.018, 0.082] | 0.095 [0.068, 0.135] | 0.074 [0.051, 0.103] |
| zerodce: s1 | 0.084 [0.036, 0.131] | 0.026 [0.016, 0.051] | 0.103 [0.079, 0.153] | 0.068 [0.028, 0.115] |
| nafnet: pre0 | 0.079 [0.046, 0.112] | 0.049 [0.031, 0.090] | 0.096 [0.069, 0.133] | 0.075 [0.036, 0.103] |
| Low-light gate on the selected low-light arm: q10__gamma_gadaptive | 0.075 [0.028, 0.103] | 0.041 [0.026, 0.074] | 0.079 [0.058, 0.141] | 0.063 [0.042, 0.092] |
| Deblur gate on the selected deblur arm: q50__unsharp_a0.5_s0.5 | 0.077 [0.024, 0.125] | 0.052 [0.018, 0.092] | 0.092 [0.067, 0.135] | 0.077 [0.044, 0.110] |

### SurvFace verification AUC

| Arm | ArcFace | AdaFace | MagFace | CurricularFace |
|---|---|---|---|---|
| Direct feed (resize) | 0.663 [0.637, 0.686] | 0.652 [0.628, 0.676] | 0.673 [0.649, 0.695] | 0.647 [0.621, 0.670] |
| No-op (control): default | 0.663 [0.637, 0.686] | 0.652 [0.628, 0.676] | 0.673 [0.649, 0.695] | 0.647 [0.621, 0.670] |
| bicubic: pre128 | 0.671 [0.647, 0.694] | 0.661 [0.637, 0.684] | 0.671 [0.646, 0.693] | 0.645 [0.621, 0.669] |
| gamma: gadaptive | 0.663 [0.638, 0.687] | 0.639 [0.613, 0.663] | 0.671 [0.648, 0.694] | 0.640 [0.614, 0.663] |
| clahe: c2_t2 | 0.658 [0.634, 0.681] | 0.626 [0.602, 0.651] | 0.666 [0.642, 0.688] | 0.637 [0.612, 0.660] |
| unsharp: a0.5_s0.5 | 0.663 [0.639, 0.686] | 0.648 [0.623, 0.671] | 0.671 [0.647, 0.693] | 0.646 [0.622, 0.670] |
| zerodce: s1 | 0.669 [0.644, 0.691] | 0.624 [0.599, 0.649] | 0.668 [0.644, 0.691] | 0.639 [0.613, 0.663] |
| nafnet: pre0 | 0.647 [0.620, 0.668] | 0.630 [0.605, 0.654] | 0.662 [0.639, 0.684] | 0.629 [0.605, 0.653] |
| Low-light gate on the selected low-light arm: q10__gamma_gadaptive | 0.661 [0.636, 0.685] | 0.646 [0.620, 0.669] | 0.674 [0.650, 0.696] | 0.645 [0.619, 0.668] |
| Deblur gate on the selected deblur arm: q50__unsharp_a0.5_s0.5 | 0.662 [0.637, 0.686] | 0.650 [0.624, 0.673] | 0.671 [0.647, 0.693] | 0.646 [0.622, 0.670] |

### SurvFace open-set TPIR20@FPIR=0.3

| Arm | ArcFace | AdaFace | MagFace | CurricularFace |
|---|---|---|---|---|
| Direct feed (resize) | 0.1917 [0.179, 0.205] | 0.1318 [0.120, 0.145] | 0.1785 [0.165, 0.190] | 0.1171 [0.107, 0.128] |
| No-op (control): default | 0.1917 [0.179, 0.205] | 0.1318 [0.120, 0.145] | 0.1785 [0.165, 0.190] | 0.1171 [0.107, 0.128] |
| bicubic: pre128 | 0.1899 [0.178, 0.203] | 0.1280 [0.116, 0.141] | 0.1689 [0.155, 0.181] | 0.1108 [0.100, 0.122] |
| gamma: gadaptive | 0.1935 [0.181, 0.207] | 0.1153 [0.104, 0.128] | 0.1653 [0.154, 0.177] | 0.1162 [0.106, 0.127] |
| clahe: c2_t2 | 0.1584 [0.145, 0.171] | 0.1233 [0.111, 0.137] | 0.1600 [0.149, 0.172] | 0.1059 [0.097, 0.118] |
| unsharp: a0.5_s0.5 | 0.1926 [0.179, 0.206] | 0.1202 [0.108, 0.132] | 0.1754 [0.164, 0.188] | 0.1086 [0.099, 0.119] |
| zerodce: s1 | 0.2015 [0.189, 0.214] | 0.1233 [0.111, 0.136] | 0.1868 [0.173, 0.199] | 0.1316 [0.121, 0.144] |
| nafnet: pre0 | 0.1718 [0.158, 0.185] | 0.1173 [0.105, 0.130] | 0.1586 [0.146, 0.170] | 0.1115 [0.100, 0.122] |
| Low-light gate on the selected low-light arm: q10__gamma_gadaptive | 0.1908 [0.178, 0.205] | 0.1289 [0.117, 0.142] | 0.1745 [0.162, 0.187] | 0.1139 [0.103, 0.125] |
| Deblur gate on the selected deblur arm: q50__unsharp_a0.5_s0.5 | 0.1910 [0.178, 0.205] | 0.1180 [0.107, 0.130] | 0.1756 [0.164, 0.189] | 0.1072 [0.099, 0.118] |

### Paired change vs direct feed on the same crops (eval subset; mean [95% CI])

Drift = cosine between the restored and the direct-feed embedding of the same crop (1 = unchanged). Rank-1 delta = paired change in TinyFace rank-1 hit rate. Impostor shift > 0 = impostors look more similar.

| Arm | Recognizer | Drift (TinyFace probes) | TinyFace Rank-1 delta | Impostor shift (TinyFace) | d' TinyFace | d' SurvFace 1:1 |
|---|---|---|---|---|---|---|
| No-op (control): default | arcface | 1.000 [1.000, 1.000] | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] | -0.50 | 0.60 |
| No-op (control): default | adaface | 1.000 [1.000, 1.000] | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] | -0.41 | 0.56 |
| No-op (control): default | magface | 1.000 [1.000, 1.000] | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] | -0.44 | 0.65 |
| No-op (control): default | curricularface | 1.000 [1.000, 1.000] | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] | -0.47 | 0.54 |
| bicubic: pre128 | arcface | 0.967 [0.966, 0.969] | -0.004 [-0.017, 0.009] | -0.003 [-0.004, -0.001] | -0.52 | 0.64 |
| bicubic: pre128 | adaface | 0.965 [0.964, 0.967] | +0.010 [-0.003, 0.023] | -0.006 [-0.008, -0.005] | -0.39 | 0.59 |
| bicubic: pre128 | magface | 0.976 [0.975, 0.977] | -0.018 [-0.032, -0.006] | +0.002 [0.000, 0.003] | -0.46 | 0.64 |
| bicubic: pre128 | curricularface | 0.947 [0.945, 0.949] | -0.015 [-0.029, 0.000] | -0.004 [-0.006, -0.002] | -0.48 | 0.54 |
| gamma: gadaptive | arcface | 0.987 [0.985, 0.989] | -0.002 [-0.012, 0.008] | +0.002 [0.002, 0.003] | -0.50 | 0.60 |
| gamma: gadaptive | adaface | 0.978 [0.975, 0.981] | -0.011 [-0.020, -0.003] | +0.005 [0.004, 0.007] | -0.42 | 0.51 |
| gamma: gadaptive | magface | 0.992 [0.990, 0.993] | -0.002 [-0.011, 0.008] | +0.004 [0.003, 0.004] | -0.44 | 0.64 |
| gamma: gadaptive | curricularface | 0.984 [0.982, 0.986] | -0.009 [-0.020, 0.002] | +0.008 [0.007, 0.010] | -0.48 | 0.52 |
| clahe: c2_t2 | arcface | 0.861 [0.857, 0.865] | -0.038 [-0.055, -0.021] | -0.006 [-0.009, -0.003] | -0.62 | 0.59 |
| clahe: c2_t2 | adaface | 0.870 [0.866, 0.875] | -0.043 [-0.062, -0.025] | +0.011 [0.007, 0.014] | -0.54 | 0.46 |
| clahe: c2_t2 | magface | 0.910 [0.907, 0.913] | -0.041 [-0.057, -0.023] | +0.007 [0.005, 0.009] | -0.54 | 0.63 |
| clahe: c2_t2 | curricularface | 0.846 [0.842, 0.851] | -0.039 [-0.056, -0.021] | -0.000 [-0.004, 0.003] | -0.60 | 0.51 |
| unsharp: a0.5_s0.5 | arcface | 0.992 [0.991, 0.992] | -0.010 [-0.019, 0.000] | +0.000 [-0.001, 0.001] | -0.52 | 0.60 |
| unsharp: a0.5_s0.5 | adaface | 0.988 [0.987, 0.988] | -0.009 [-0.020, 0.002] | -0.000 [-0.001, 0.001] | -0.44 | 0.54 |
| unsharp: a0.5_s0.5 | magface | 0.994 [0.994, 0.994] | -0.003 [-0.012, 0.005] | -0.001 [-0.001, 0.000] | -0.45 | 0.64 |
| unsharp: a0.5_s0.5 | curricularface | 0.986 [0.986, 0.987] | -0.008 [-0.017, 0.003] | -0.001 [-0.002, 0.000] | -0.48 | 0.54 |
| zerodce: s1 | arcface | 0.896 [0.893, 0.899] | -0.029 [-0.046, -0.014] | +0.010 [0.008, 0.013] | -0.57 | 0.63 |
| zerodce: s1 | adaface | 0.872 [0.868, 0.876] | -0.029 [-0.047, -0.010] | +0.027 [0.024, 0.030] | -0.48 | 0.45 |
| zerodce: s1 | magface | 0.926 [0.924, 0.928] | -0.023 [-0.037, -0.008] | +0.018 [0.016, 0.020] | -0.50 | 0.63 |
| zerodce: s1 | curricularface | 0.879 [0.875, 0.882] | -0.023 [-0.038, -0.006] | +0.028 [0.025, 0.030] | -0.53 | 0.52 |
| nafnet: pre0 | arcface | 0.910 [0.903, 0.915] | -0.022 [-0.039, -0.006] | -0.009 [-0.011, -0.006] | -0.58 | 0.54 |
| nafnet: pre0 | adaface | 0.906 [0.899, 0.912] | -0.037 [-0.055, -0.020] | -0.002 [-0.005, 0.001] | -0.50 | 0.47 |
| nafnet: pre0 | magface | 0.933 [0.927, 0.938] | -0.032 [-0.048, -0.016] | -0.007 [-0.010, -0.004] | -0.50 | 0.61 |
| nafnet: pre0 | curricularface | 0.901 [0.894, 0.908] | -0.013 [-0.029, 0.002] | -0.009 [-0.012, -0.006] | -0.54 | 0.48 |
| Low-light gate on the selected low-light arm: q10__gamma_gadaptive | arcface | 0.995 [0.993, 0.996] | -0.001 [-0.006, 0.003] | +0.001 [-0.000, 0.001] | -0.50 | 0.60 |
| Low-light gate on the selected low-light arm: q10__gamma_gadaptive | adaface | 0.991 [0.988, 0.994] | -0.008 [-0.014, -0.003] | +0.001 [0.000, 0.002] | -0.41 | 0.54 |
| Low-light gate on the selected low-light arm: q10__gamma_gadaptive | magface | 0.997 [0.995, 0.998] | -0.003 [-0.009, 0.003] | +0.001 [0.000, 0.001] | -0.44 | 0.65 |
| Low-light gate on the selected low-light arm: q10__gamma_gadaptive | curricularface | 0.994 [0.992, 0.996] | -0.003 [-0.009, 0.003] | +0.002 [0.002, 0.003] | -0.48 | 0.54 |
| Deblur gate on the selected deblur arm: q50__unsharp_a0.5_s0.5 | arcface | 0.999 [0.999, 0.999] | -0.001 [-0.004, 0.003] | -0.000 [-0.001, 0.000] | -0.50 | 0.60 |
| Deblur gate on the selected deblur arm: q50__unsharp_a0.5_s0.5 | adaface | 0.998 [0.998, 0.998] | -0.002 [-0.005, 0.002] | -0.000 [-0.001, 0.000] | -0.41 | 0.55 |
| Deblur gate on the selected deblur arm: q50__unsharp_a0.5_s0.5 | magface | 0.999 [0.999, 0.999] | -0.002 [-0.006, 0.003] | -0.000 [-0.001, 0.000] | -0.44 | 0.64 |
| Deblur gate on the selected deblur arm: q50__unsharp_a0.5_s0.5 | curricularface | 0.998 [0.998, 0.998] | -0.002 [-0.006, 0.002] | -0.000 [-0.001, 0.000] | -0.47 | 0.54 |

### Cost (s/crop; eval restoration on the device shown, smoke = 75 crops on CPU vs GPU)

| Arm | Setting | Eval device | Eval s/crop median | Eval s/crop mean | Peak VRAM (MB) | Failed | Smoke CPU s/crop | Smoke GPU s/crop |
|---|---|---|---|---|---|---|---|---|
| bicubic | pre128 | cpu | n/a | 0.0121 | 0 | 0 | (CPU only) | (CPU only) |
| gamma | gadaptive | cpu | 0.0117 | 0.0122 | 0 | 0 | (CPU only) | (CPU only) |
| clahe | c2_t2 | cpu | 0.0117 | 0.0100 | 0 | 0 | (CPU only) | (CPU only) |
| unsharp | a0.5_s0.5 | cpu | 0.0092 | 0.0076 | 0 | 0 | (CPU only) | (CPU only) |
| zerodce | s1 | cuda | 0.0144 | 0.0154 | 13 | 0 | 0.0097 | 0.0110 |
| nafnet | pre0 | cuda | 0.0046 | 0.0047 | 459 | 0 | 0.0293 | 0.0109 |

### Tuning sweeps (tuning subset; score = mean over AdaFace, MagFace, CurricularFace of [TinyFace Rank-1, SurvFace verification AUC])

**noop** (selected: default)

| Setting | Score | Rank-1 | AUC |
|---|---|---|---|
| default | 0.5728 | 0.4826 | 0.6631 |

**bicubic** (selected: pre128)

| Setting | Score | Rank-1 | AUC |
|---|---|---|---|
| pre128 | 0.5701 | 0.4726 | 0.6676 |

**gamma** (selected: gadaptive)

| Setting | Score | Rank-1 | AUC |
|---|---|---|---|
| gadaptive | 0.5672 | 0.4826 | 0.6517 |
| g0.7 | 0.5649 | 0.4826 | 0.6472 |
| g0.5 | 0.5532 | 0.4677 | 0.6386 |

**clahe** (selected: c2_t2)

| Setting | Score | Rank-1 | AUC |
|---|---|---|---|
| c2_t2 | 0.5451 | 0.4378 | 0.6525 |
| c1_t2 | 0.5436 | 0.4378 | 0.6493 |
| c2_t4 | 0.5107 | 0.3881 | 0.6334 |
| c1_t4 | 0.5106 | 0.3881 | 0.6332 |

**unsharp** (selected: a0.5_s0.5)

| Setting | Score | Rank-1 | AUC |
|---|---|---|---|
| a0.5_s0.5 | 0.5692 | 0.4726 | 0.6657 |
| a1.0_s0.5 | 0.5630 | 0.4577 | 0.6682 |
| a0.5_s1.0 | 0.5558 | 0.4428 | 0.6688 |
| a1.0_s1.0 | 0.5503 | 0.4279 | 0.6728 |

**zerodce** (selected: s1)

| Setting | Score | Rank-1 | AUC |
|---|---|---|---|
| s1 | 0.5431 | 0.4378 | 0.6484 |

**nafnet** (selected: pre0)

| Setting | Score | Rank-1 | AUC |
|---|---|---|---|
| pre0 | 0.5548 | 0.4378 | 0.6719 |
| pre128 | 0.5489 | 0.4378 | 0.6599 |

**gate_lowlight** (selected: q10__gamma_gadaptive)

| Setting | Score | Rank-1 | AUC |
|---|---|---|---|
| q10__gamma_gadaptive | 0.5720 | 0.4876 | 0.6565 |
| q20__gamma_gadaptive | 0.5720 | 0.4876 | 0.6564 |
| q30__gamma_gadaptive | 0.5702 | 0.4876 | 0.6529 |
| q50__gamma_gadaptive | 0.5656 | 0.4776 | 0.6536 |

**gate_deblur** (selected: q50__unsharp_a0.5_s0.5)

| Setting | Score | Rank-1 | AUC |
|---|---|---|---|
| q50__unsharp_a0.5_s0.5 | 0.5763 | 0.4876 | 0.6651 |
| q30__unsharp_a0.5_s0.5 | 0.5742 | 0.4826 | 0.6658 |
| q20__unsharp_a0.5_s0.5 | 0.5736 | 0.4826 | 0.6645 |
| q10__unsharp_a0.5_s0.5 | 0.5731 | 0.4826 | 0.6636 |

<!-- TABLES:END -->
<!-- EXTRA:BEGIN -->
### Probes-only restoration vs both sides restored (eval subset)

Probes / queries restored, gallery, distractors and templates left as direct feed. Delta = probes-only minus direct feed, paired on the same probes (TinyFace); SurvFace cells show the unpaired CI. Verification: first image of each pair restored, second direct (roles arbitrary).

**TinyFace Rank-1**

| Arm | Recognizer | Direct feed | Both sides restored | Probes-only restored | Probes-only minus direct (paired) |
|---|---|---|---|---|---|
| bicubic: pre128 | arcface | 0.302 [0.276, 0.328] | 0.298 [0.274, 0.323] | 0.303 [0.278, 0.328] | +0.001 [-0.011, 0.013] |
| bicubic: pre128 | adaface | 0.336 [0.309, 0.363] | 0.345 [0.319, 0.371] | 0.339 [0.310, 0.366] | +0.003 [-0.008, 0.016] |
| bicubic: pre128 | magface | 0.336 [0.310, 0.362] | 0.318 [0.291, 0.343] | 0.319 [0.292, 0.345] | -0.017 [-0.029, -0.006] |
| bicubic: pre128 | curricularface | 0.298 [0.271, 0.325] | 0.283 [0.258, 0.307] | 0.293 [0.267, 0.320] | -0.005 [-0.017, 0.007] |
| gamma: gadaptive | arcface | 0.302 [0.276, 0.328] | 0.300 [0.274, 0.328] | 0.296 [0.272, 0.322] | -0.006 [-0.013, 0.000] |
| gamma: gadaptive | adaface | 0.336 [0.309, 0.363] | 0.325 [0.300, 0.352] | 0.325 [0.299, 0.352] | -0.011 [-0.018, -0.005] |
| gamma: gadaptive | magface | 0.336 [0.310, 0.362] | 0.334 [0.307, 0.362] | 0.329 [0.303, 0.355] | -0.007 [-0.015, 0.001] |
| gamma: gadaptive | curricularface | 0.298 [0.271, 0.325] | 0.289 [0.264, 0.315] | 0.289 [0.264, 0.314] | -0.009 [-0.016, -0.002] |
| clahe: c2_t2 | arcface | 0.302 [0.276, 0.328] | 0.264 [0.240, 0.289] | 0.261 [0.237, 0.287] | -0.041 [-0.058, -0.024] |
| clahe: c2_t2 | adaface | 0.336 [0.309, 0.363] | 0.293 [0.268, 0.319] | 0.289 [0.265, 0.313] | -0.047 [-0.062, -0.029] |
| clahe: c2_t2 | magface | 0.336 [0.310, 0.362] | 0.295 [0.270, 0.322] | 0.298 [0.273, 0.323] | -0.038 [-0.053, -0.023] |
| clahe: c2_t2 | curricularface | 0.298 [0.271, 0.325] | 0.259 [0.235, 0.284] | 0.252 [0.227, 0.275] | -0.046 [-0.062, -0.030] |
| unsharp: a0.5_s0.5 | arcface | 0.302 [0.276, 0.328] | 0.293 [0.268, 0.319] | 0.298 [0.273, 0.324] | -0.004 [-0.012, 0.003] |
| unsharp: a0.5_s0.5 | adaface | 0.336 [0.309, 0.363] | 0.327 [0.300, 0.352] | 0.326 [0.300, 0.352] | -0.010 [-0.018, -0.001] |
| unsharp: a0.5_s0.5 | magface | 0.336 [0.310, 0.362] | 0.332 [0.305, 0.358] | 0.332 [0.305, 0.359] | -0.003 [-0.012, 0.004] |
| unsharp: a0.5_s0.5 | curricularface | 0.298 [0.271, 0.325] | 0.290 [0.266, 0.316] | 0.296 [0.269, 0.323] | -0.002 [-0.009, 0.006] |
| zerodce: s1 | arcface | 0.302 [0.276, 0.328] | 0.273 [0.248, 0.297] | 0.272 [0.248, 0.298] | -0.030 [-0.047, -0.015] |
| zerodce: s1 | adaface | 0.336 [0.309, 0.363] | 0.307 [0.281, 0.334] | 0.298 [0.273, 0.324] | -0.038 [-0.055, -0.021] |
| zerodce: s1 | magface | 0.336 [0.310, 0.362] | 0.313 [0.288, 0.339] | 0.298 [0.273, 0.325] | -0.038 [-0.053, -0.023] |
| zerodce: s1 | curricularface | 0.298 [0.271, 0.325] | 0.275 [0.250, 0.302] | 0.265 [0.241, 0.290] | -0.033 [-0.050, -0.016] |
| nafnet: pre0 | arcface | 0.302 [0.276, 0.328] | 0.281 [0.256, 0.305] | 0.281 [0.256, 0.306] | -0.021 [-0.035, -0.006] |
| nafnet: pre0 | adaface | 0.336 [0.309, 0.363] | 0.299 [0.274, 0.324] | 0.306 [0.282, 0.333] | -0.029 [-0.043, -0.015] |
| nafnet: pre0 | magface | 0.336 [0.310, 0.362] | 0.304 [0.277, 0.331] | 0.307 [0.281, 0.333] | -0.029 [-0.043, -0.016] |
| nafnet: pre0 | curricularface | 0.298 [0.271, 0.325] | 0.285 [0.261, 0.310] | 0.274 [0.249, 0.300] | -0.023 [-0.036, -0.011] |
| gate_lowlight: q10__gamma_gadaptive | arcface | 0.302 [0.276, 0.328] | 0.301 [0.276, 0.328] | 0.301 [0.275, 0.328] | -0.001 [-0.004, 0.002] |
| gate_lowlight: q10__gamma_gadaptive | adaface | 0.336 [0.309, 0.363] | 0.328 [0.301, 0.356] | 0.332 [0.305, 0.359] | -0.004 [-0.009, -0.001] |
| gate_lowlight: q10__gamma_gadaptive | magface | 0.336 [0.310, 0.362] | 0.333 [0.308, 0.359] | 0.332 [0.306, 0.358] | -0.004 [-0.010, 0.000] |
| gate_lowlight: q10__gamma_gadaptive | curricularface | 0.298 [0.271, 0.325] | 0.295 [0.270, 0.322] | 0.294 [0.268, 0.320] | -0.003 [-0.008, 0.000] |
| gate_deblur: q50__unsharp_a0.5_s0.5 | arcface | 0.302 [0.276, 0.328] | 0.301 [0.275, 0.327] | 0.302 [0.276, 0.328] | +0.000 [0.000, 0.000] |
| gate_deblur: q50__unsharp_a0.5_s0.5 | adaface | 0.336 [0.309, 0.363] | 0.334 [0.307, 0.360] | 0.335 [0.308, 0.361] | -0.001 [-0.003, 0.000] |
| gate_deblur: q50__unsharp_a0.5_s0.5 | magface | 0.336 [0.310, 0.362] | 0.334 [0.308, 0.360] | 0.336 [0.310, 0.363] | +0.000 [-0.003, 0.003] |
| gate_deblur: q50__unsharp_a0.5_s0.5 | curricularface | 0.298 [0.271, 0.325] | 0.296 [0.269, 0.323] | 0.298 [0.271, 0.325] | +0.000 [-0.003, 0.003] |

**SurvFace open-set TPIR20@FPIR=0.3**

| Arm | Recognizer | Direct feed | Both sides restored | Probes-only restored | Probes-only minus direct (paired) |
|---|---|---|---|---|---|
| bicubic: pre128 | arcface | 0.1917 [0.1785, 0.2046] | 0.1899 [0.1783, 0.2026] | 0.1897 [0.1783, 0.2022] | - |
| bicubic: pre128 | adaface | 0.1318 [0.1202, 0.1452] | 0.1280 [0.1159, 0.1405] | 0.1289 [0.1177, 0.1401] | - |
| bicubic: pre128 | magface | 0.1785 [0.1653, 0.1903] | 0.1689 [0.1550, 0.1805] | 0.1649 [0.1537, 0.1796] | - |
| bicubic: pre128 | curricularface | 0.1171 [0.1066, 0.1278] | 0.1108 [0.1003, 0.1224] | 0.1182 [0.1068, 0.1289] | - |
| gamma: gadaptive | arcface | 0.1917 [0.1785, 0.2046] | 0.1935 [0.1810, 0.2067] | 0.1870 [0.1749, 0.2004] | - |
| gamma: gadaptive | adaface | 0.1318 [0.1202, 0.1452] | 0.1153 [0.1041, 0.1278] | 0.1075 [0.0954, 0.1195] | - |
| gamma: gadaptive | magface | 0.1785 [0.1653, 0.1903] | 0.1653 [0.1539, 0.1772] | 0.1669 [0.1530, 0.1781] | - |
| gamma: gadaptive | curricularface | 0.1171 [0.1066, 0.1278] | 0.1162 [0.1059, 0.1271] | 0.1032 [0.0938, 0.1135] | - |
| clahe: c2_t2 | arcface | 0.1917 [0.1785, 0.2046] | 0.1584 [0.1454, 0.1711] | 0.1061 [0.0947, 0.1169] | - |
| clahe: c2_t2 | adaface | 0.1318 [0.1202, 0.1452] | 0.1233 [0.1113, 0.1367] | 0.0561 [0.0485, 0.0652] | - |
| clahe: c2_t2 | magface | 0.1785 [0.1653, 0.1903] | 0.1600 [0.1486, 0.1723] | 0.1101 [0.0990, 0.1206] | - |
| clahe: c2_t2 | curricularface | 0.1171 [0.1066, 0.1278] | 0.1059 [0.0970, 0.1175] | 0.0536 [0.0471, 0.0614] | - |
| unsharp: a0.5_s0.5 | arcface | 0.1917 [0.1785, 0.2046] | 0.1926 [0.1792, 0.2058] | 0.1874 [0.1734, 0.2026] | - |
| unsharp: a0.5_s0.5 | adaface | 0.1318 [0.1202, 0.1452] | 0.1202 [0.1077, 0.1318] | 0.1260 [0.1144, 0.1392] | - |
| unsharp: a0.5_s0.5 | magface | 0.1785 [0.1653, 0.1903] | 0.1754 [0.1638, 0.1881] | 0.1747 [0.1611, 0.1886] | - |
| unsharp: a0.5_s0.5 | curricularface | 0.1171 [0.1066, 0.1278] | 0.1086 [0.0990, 0.1191] | 0.1200 [0.1099, 0.1305] | - |
| zerodce: s1 | arcface | 0.1917 [0.1785, 0.2046] | 0.2015 [0.1892, 0.2145] | 0.1709 [0.1597, 0.1841] | - |
| zerodce: s1 | adaface | 0.1318 [0.1202, 0.1452] | 0.1233 [0.1110, 0.1361] | 0.0632 [0.0547, 0.0733] | - |
| zerodce: s1 | magface | 0.1785 [0.1653, 0.1903] | 0.1868 [0.1731, 0.1988] | 0.1588 [0.1461, 0.1709] | - |
| zerodce: s1 | curricularface | 0.1171 [0.1066, 0.1278] | 0.1316 [0.1209, 0.1439] | 0.0529 [0.0456, 0.0617] | - |
| nafnet: pre0 | arcface | 0.1917 [0.1785, 0.2046] | 0.1718 [0.1582, 0.1852] | 0.1676 [0.1562, 0.1807] | - |
| nafnet: pre0 | adaface | 0.1318 [0.1202, 0.1452] | 0.1173 [0.1052, 0.1300] | 0.1041 [0.0920, 0.1135] | - |
| nafnet: pre0 | magface | 0.1785 [0.1653, 0.1903] | 0.1586 [0.1457, 0.1700] | 0.1586 [0.1441, 0.1705] | - |
| nafnet: pre0 | curricularface | 0.1171 [0.1066, 0.1278] | 0.1115 [0.1003, 0.1218] | 0.1211 [0.1101, 0.1327] | - |
| gate_lowlight: q10__gamma_gadaptive | arcface | 0.1917 [0.1785, 0.2046] | 0.1908 [0.1783, 0.2053] | 0.1919 [0.1785, 0.2047] | - |
| gate_lowlight: q10__gamma_gadaptive | adaface | 0.1318 [0.1202, 0.1452] | 0.1289 [0.1171, 0.1416] | 0.1300 [0.1182, 0.1428] | - |
| gate_lowlight: q10__gamma_gadaptive | magface | 0.1785 [0.1653, 0.1903] | 0.1745 [0.1618, 0.1870] | 0.1769 [0.1638, 0.1890] | - |
| gate_lowlight: q10__gamma_gadaptive | curricularface | 0.1171 [0.1066, 0.1278] | 0.1139 [0.1032, 0.1253] | 0.1139 [0.1041, 0.1256] | - |
| gate_deblur: q50__unsharp_a0.5_s0.5 | arcface | 0.1917 [0.1785, 0.2046] | 0.1910 [0.1781, 0.2046] | 0.1859 [0.1723, 0.2006] | - |
| gate_deblur: q50__unsharp_a0.5_s0.5 | adaface | 0.1318 [0.1202, 0.1452] | 0.1180 [0.1068, 0.1298] | 0.1265 [0.1151, 0.1390] | - |
| gate_deblur: q50__unsharp_a0.5_s0.5 | magface | 0.1785 [0.1653, 0.1903] | 0.1756 [0.1635, 0.1886] | 0.1740 [0.1613, 0.1874] | - |
| gate_deblur: q50__unsharp_a0.5_s0.5 | curricularface | 0.1171 [0.1066, 0.1278] | 0.1072 [0.0985, 0.1180] | 0.1191 [0.1097, 0.1296] | - |

**SurvFace verification AUC (one side restored)**

| Arm | Recognizer | Direct feed | Both sides restored | Probes-only restored | Probes-only minus direct (paired) |
|---|---|---|---|---|---|
| bicubic: pre128 | arcface | 0.663 [0.637, 0.686] | 0.671 [0.647, 0.694] | 0.663 [0.639, 0.687] | - |
| bicubic: pre128 | adaface | 0.652 [0.628, 0.676] | 0.661 [0.637, 0.684] | 0.653 [0.628, 0.676] | - |
| bicubic: pre128 | magface | 0.673 [0.649, 0.695] | 0.671 [0.646, 0.693] | 0.672 [0.647, 0.694] | - |
| bicubic: pre128 | curricularface | 0.647 [0.621, 0.670] | 0.645 [0.621, 0.669] | 0.643 [0.618, 0.667] | - |
| gamma: gadaptive | arcface | 0.663 [0.637, 0.686] | 0.663 [0.638, 0.687] | 0.660 [0.635, 0.683] | - |
| gamma: gadaptive | adaface | 0.652 [0.628, 0.676] | 0.639 [0.613, 0.663] | 0.642 [0.617, 0.666] | - |
| gamma: gadaptive | magface | 0.673 [0.649, 0.695] | 0.671 [0.648, 0.694] | 0.671 [0.647, 0.693] | - |
| gamma: gadaptive | curricularface | 0.647 [0.621, 0.670] | 0.640 [0.614, 0.663] | 0.640 [0.615, 0.663] | - |
| clahe: c2_t2 | arcface | 0.663 [0.637, 0.686] | 0.658 [0.634, 0.681] | 0.655 [0.630, 0.678] | - |
| clahe: c2_t2 | adaface | 0.652 [0.628, 0.676] | 0.626 [0.602, 0.651] | 0.641 [0.616, 0.664] | - |
| clahe: c2_t2 | magface | 0.673 [0.649, 0.695] | 0.666 [0.642, 0.688] | 0.660 [0.637, 0.683] | - |
| clahe: c2_t2 | curricularface | 0.647 [0.621, 0.670] | 0.637 [0.612, 0.660] | 0.629 [0.604, 0.654] | - |
| unsharp: a0.5_s0.5 | arcface | 0.663 [0.637, 0.686] | 0.663 [0.639, 0.686] | 0.663 [0.638, 0.687] | - |
| unsharp: a0.5_s0.5 | adaface | 0.652 [0.628, 0.676] | 0.648 [0.623, 0.671] | 0.649 [0.625, 0.672] | - |
| unsharp: a0.5_s0.5 | magface | 0.673 [0.649, 0.695] | 0.671 [0.647, 0.693] | 0.672 [0.648, 0.694] | - |
| unsharp: a0.5_s0.5 | curricularface | 0.647 [0.621, 0.670] | 0.646 [0.622, 0.670] | 0.647 [0.622, 0.670] | - |
| zerodce: s1 | arcface | 0.663 [0.637, 0.686] | 0.669 [0.644, 0.691] | 0.653 [0.629, 0.676] | - |
| zerodce: s1 | adaface | 0.652 [0.628, 0.676] | 0.624 [0.599, 0.649] | 0.632 [0.607, 0.656] | - |
| zerodce: s1 | magface | 0.673 [0.649, 0.695] | 0.668 [0.644, 0.691] | 0.661 [0.635, 0.684] | - |
| zerodce: s1 | curricularface | 0.647 [0.621, 0.670] | 0.639 [0.613, 0.663] | 0.627 [0.603, 0.651] | - |
| nafnet: pre0 | arcface | 0.663 [0.637, 0.686] | 0.647 [0.620, 0.668] | 0.656 [0.631, 0.679] | - |
| nafnet: pre0 | adaface | 0.652 [0.628, 0.676] | 0.630 [0.605, 0.654] | 0.640 [0.614, 0.664] | - |
| nafnet: pre0 | magface | 0.673 [0.649, 0.695] | 0.662 [0.639, 0.684] | 0.669 [0.645, 0.691] | - |
| nafnet: pre0 | curricularface | 0.647 [0.621, 0.670] | 0.629 [0.605, 0.653] | 0.637 [0.612, 0.659] | - |
| gate_lowlight: q10__gamma_gadaptive | arcface | 0.663 [0.637, 0.686] | 0.661 [0.636, 0.685] | 0.659 [0.634, 0.684] | - |
| gate_lowlight: q10__gamma_gadaptive | adaface | 0.652 [0.628, 0.676] | 0.646 [0.620, 0.669] | 0.647 [0.622, 0.671] | - |
| gate_lowlight: q10__gamma_gadaptive | magface | 0.673 [0.649, 0.695] | 0.674 [0.650, 0.696] | 0.673 [0.649, 0.695] | - |
| gate_lowlight: q10__gamma_gadaptive | curricularface | 0.647 [0.621, 0.670] | 0.645 [0.619, 0.668] | 0.645 [0.619, 0.667] | - |
| gate_deblur: q50__unsharp_a0.5_s0.5 | arcface | 0.663 [0.637, 0.686] | 0.662 [0.637, 0.686] | 0.663 [0.637, 0.686] | - |
| gate_deblur: q50__unsharp_a0.5_s0.5 | adaface | 0.652 [0.628, 0.676] | 0.650 [0.624, 0.673] | 0.651 [0.626, 0.674] | - |
| gate_deblur: q50__unsharp_a0.5_s0.5 | magface | 0.673 [0.649, 0.695] | 0.671 [0.647, 0.693] | 0.672 [0.648, 0.694] | - |
| gate_deblur: q50__unsharp_a0.5_s0.5 | curricularface | 0.647 [0.621, 0.670] | 0.646 [0.622, 0.670] | 0.647 [0.622, 0.671] | - |

### Per-bin breakdown of the both-sides-restored results (Phase 2 bins)

`low_sharpness` = size-controlled lowest 20% of Laplacian variance; `dark10` = darkest 10% (Phase 2 definitions, per split). n = number of probes (TinyFace) / mated queries (SurvFace) in the bin; cells with fewer than 20 items are omitted. TinyFace: Rank-1 change vs direct feed, paired [95% CI]. SurvFace: direct -> restored (unpaired CIs are in `per_bin.csv`).

**Slice `low_sharpness`**

| Arm | Recognizer | n probes | TinyFace Rank-1: direct -> restored (paired delta) | n mated | SurvFace TPIR20@FPIR=0.3: direct -> restored | n pos pairs | SurvFace AUC: direct -> restored |
|---|---|---|---|---|---|---|---|
| bicubic: pre128 | arcface | 217 | 0.138 -> 0.138 (+0.000 [-0.028, 0.028]) | 1201 | 0.2365 -> 0.2273 | 322 | 0.662 -> 0.669 |
| bicubic: pre128 | adaface | 217 | 0.147 -> 0.138 (-0.009 [-0.037, 0.014]) | 1201 | 0.1224 -> 0.1041 | 322 | 0.634 -> 0.643 |
| bicubic: pre128 | magface | 217 | 0.138 -> 0.143 (+0.005 [-0.014, 0.023]) | 1201 | 0.2298 -> 0.1982 | 322 | 0.646 -> 0.646 |
| bicubic: pre128 | curricularface | 217 | 0.106 -> 0.097 (-0.009 [-0.037, 0.014]) | 1201 | 0.1149 -> 0.0991 | 322 | 0.646 -> 0.640 |
| gamma: gadaptive | arcface | 217 | 0.138 -> 0.134 (-0.005 [-0.028, 0.023]) | 1201 | 0.2365 -> 0.2298 | 322 | 0.662 -> 0.667 |
| gamma: gadaptive | adaface | 217 | 0.147 -> 0.124 (-0.023 [-0.046, -0.005]) | 1201 | 0.1224 -> 0.0699 | 322 | 0.634 -> 0.624 |
| gamma: gadaptive | magface | 217 | 0.138 -> 0.134 (-0.005 [-0.018, 0.009]) | 1201 | 0.2298 -> 0.2132 | 322 | 0.646 -> 0.642 |
| gamma: gadaptive | curricularface | 217 | 0.106 -> 0.101 (-0.005 [-0.028, 0.018]) | 1201 | 0.1149 -> 0.0908 | 322 | 0.646 -> 0.640 |
| clahe: c2_t2 | arcface | 217 | 0.138 -> 0.078 (-0.060 [-0.097, -0.028]) | 1201 | 0.2365 -> 0.1932 | 322 | 0.662 -> 0.634 |
| clahe: c2_t2 | adaface | 217 | 0.147 -> 0.120 (-0.028 [-0.065, 0.005]) | 1201 | 0.1224 -> 0.1099 | 322 | 0.634 -> 0.585 |
| clahe: c2_t2 | magface | 217 | 0.138 -> 0.101 (-0.037 [-0.074, -0.005]) | 1201 | 0.2298 -> 0.2040 | 322 | 0.646 -> 0.627 |
| clahe: c2_t2 | curricularface | 217 | 0.106 -> 0.111 (+0.005 [-0.032, 0.032]) | 1201 | 0.1149 -> 0.0983 | 322 | 0.646 -> 0.604 |
| unsharp: a0.5_s0.5 | arcface | 217 | 0.138 -> 0.129 (-0.009 [-0.028, 0.000]) | 1201 | 0.2365 -> 0.2231 | 322 | 0.662 -> 0.662 |
| unsharp: a0.5_s0.5 | adaface | 217 | 0.147 -> 0.138 (-0.009 [-0.028, 0.005]) | 1201 | 0.1224 -> 0.0933 | 322 | 0.634 -> 0.622 |
| unsharp: a0.5_s0.5 | magface | 217 | 0.138 -> 0.134 (-0.005 [-0.014, 0.000]) | 1201 | 0.2298 -> 0.2065 | 322 | 0.646 -> 0.640 |
| unsharp: a0.5_s0.5 | curricularface | 217 | 0.106 -> 0.101 (-0.005 [-0.014, 0.000]) | 1201 | 0.1149 -> 0.0974 | 322 | 0.646 -> 0.645 |
| zerodce: s1 | arcface | 217 | 0.138 -> 0.106 (-0.032 [-0.065, -0.005]) | 1201 | 0.2365 -> 0.2814 | 322 | 0.662 -> 0.673 |
| zerodce: s1 | adaface | 217 | 0.147 -> 0.101 (-0.046 [-0.083, -0.009]) | 1201 | 0.1224 -> 0.1107 | 322 | 0.634 -> 0.614 |
| zerodce: s1 | magface | 217 | 0.138 -> 0.097 (-0.041 [-0.078, -0.005]) | 1201 | 0.2298 -> 0.2889 | 322 | 0.646 -> 0.647 |
| zerodce: s1 | curricularface | 217 | 0.106 -> 0.088 (-0.018 [-0.046, 0.005]) | 1201 | 0.1149 -> 0.1474 | 322 | 0.646 -> 0.624 |
| nafnet: pre0 | arcface | 217 | 0.138 -> 0.078 (-0.060 [-0.092, -0.028]) | 1201 | 0.2365 -> 0.2140 | 322 | 0.662 -> 0.636 |
| nafnet: pre0 | adaface | 217 | 0.147 -> 0.111 (-0.037 [-0.078, 0.005]) | 1201 | 0.1224 -> 0.1307 | 322 | 0.634 -> 0.607 |
| nafnet: pre0 | magface | 217 | 0.138 -> 0.101 (-0.037 [-0.074, 0.000]) | 1201 | 0.2298 -> 0.2007 | 322 | 0.646 -> 0.627 |
| nafnet: pre0 | curricularface | 217 | 0.106 -> 0.083 (-0.023 [-0.055, 0.009]) | 1201 | 0.1149 -> 0.1074 | 322 | 0.646 -> 0.606 |
| gate_lowlight: q10__gamma_gadaptive | arcface | 217 | 0.138 -> 0.143 (+0.005 [0.000, 0.018]) | 1201 | 0.2365 -> 0.2490 | 322 | 0.662 -> 0.667 |
| gate_lowlight: q10__gamma_gadaptive | adaface | 217 | 0.147 -> 0.129 (-0.018 [-0.037, -0.005]) | 1201 | 0.1224 -> 0.0991 | 322 | 0.634 -> 0.623 |
| gate_lowlight: q10__gamma_gadaptive | magface | 217 | 0.138 -> 0.134 (-0.005 [-0.018, 0.009]) | 1201 | 0.2298 -> 0.2190 | 322 | 0.646 -> 0.649 |
| gate_lowlight: q10__gamma_gadaptive | curricularface | 217 | 0.106 -> 0.111 (+0.005 [0.000, 0.014]) | 1201 | 0.1149 -> 0.1041 | 322 | 0.646 -> 0.642 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | arcface | 217 | 0.138 -> 0.134 (-0.005 [-0.014, 0.000]) | 1201 | 0.2365 -> 0.2215 | 322 | 0.662 -> 0.659 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | adaface | 217 | 0.147 -> 0.143 (-0.005 [-0.023, 0.009]) | 1201 | 0.1224 -> 0.0958 | 322 | 0.634 -> 0.624 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | magface | 217 | 0.138 -> 0.129 (-0.009 [-0.023, 0.000]) | 1201 | 0.2298 -> 0.2065 | 322 | 0.646 -> 0.640 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | curricularface | 217 | 0.106 -> 0.097 (-0.009 [-0.023, 0.000]) | 1201 | 0.1149 -> 0.0974 | 322 | 0.646 -> 0.644 |

**Slice `not_low_sharpness`**

| Arm | Recognizer | n probes | TinyFace Rank-1: direct -> restored (paired delta) | n mated | SurvFace TPIR20@FPIR=0.3: direct -> restored | n pos pairs | SurvFace AUC: direct -> restored |
|---|---|---|---|---|---|---|---|
| bicubic: pre128 | arcface | 938 | 0.340 -> 0.335 (-0.005 [-0.021, 0.009]) | 3275 | 0.1408 -> 0.1444 | 678 | 0.670 -> 0.680 |
| bicubic: pre128 | adaface | 938 | 0.380 -> 0.393 (+0.014 [-0.001, 0.028]) | 3275 | 0.1035 -> 0.1060 | 678 | 0.664 -> 0.672 |
| bicubic: pre128 | magface | 938 | 0.382 -> 0.358 (-0.023 [-0.038, -0.007]) | 3275 | 0.1340 -> 0.1316 | 678 | 0.690 -> 0.688 |
| bicubic: pre128 | curricularface | 938 | 0.342 -> 0.326 (-0.016 [-0.033, 0.002]) | 3275 | 0.0974 -> 0.1008 | 678 | 0.653 -> 0.655 |
| gamma: gadaptive | arcface | 938 | 0.340 -> 0.339 (-0.001 [-0.012, 0.010]) | 3275 | 0.1408 -> 0.1475 | 678 | 0.670 -> 0.668 |
| gamma: gadaptive | adaface | 938 | 0.380 -> 0.371 (-0.009 [-0.018, 0.001]) | 3275 | 0.1035 -> 0.1005 | 678 | 0.664 -> 0.651 |
| gamma: gadaptive | magface | 938 | 0.382 -> 0.381 (-0.001 [-0.013, 0.011]) | 3275 | 0.1340 -> 0.1298 | 678 | 0.690 -> 0.690 |
| gamma: gadaptive | curricularface | 938 | 0.342 -> 0.333 (-0.010 [-0.021, 0.002]) | 3275 | 0.0974 -> 0.1026 | 678 | 0.653 -> 0.647 |
| clahe: c2_t2 | arcface | 938 | 0.340 -> 0.307 (-0.033 [-0.052, -0.015]) | 3275 | 0.1408 -> 0.1264 | 678 | 0.670 -> 0.674 |
| clahe: c2_t2 | adaface | 938 | 0.380 -> 0.333 (-0.047 [-0.067, -0.026]) | 3275 | 0.1035 -> 0.0965 | 678 | 0.664 -> 0.649 |
| clahe: c2_t2 | magface | 938 | 0.382 -> 0.340 (-0.042 [-0.059, -0.022]) | 3275 | 0.1340 -> 0.1252 | 678 | 0.690 -> 0.688 |
| clahe: c2_t2 | curricularface | 938 | 0.342 -> 0.293 (-0.049 [-0.070, -0.030]) | 3275 | 0.0974 -> 0.0959 | 678 | 0.653 -> 0.658 |
| unsharp: a0.5_s0.5 | arcface | 938 | 0.340 -> 0.330 (-0.010 [-0.021, 0.001]) | 3275 | 0.1408 -> 0.1460 | 678 | 0.670 -> 0.671 |
| unsharp: a0.5_s0.5 | adaface | 938 | 0.380 -> 0.371 (-0.009 [-0.020, 0.004]) | 3275 | 0.1035 -> 0.1020 | 678 | 0.664 -> 0.662 |
| unsharp: a0.5_s0.5 | magface | 938 | 0.382 -> 0.378 (-0.003 [-0.014, 0.006]) | 3275 | 0.1340 -> 0.1362 | 678 | 0.690 -> 0.690 |
| unsharp: a0.5_s0.5 | curricularface | 938 | 0.342 -> 0.334 (-0.009 [-0.019, 0.003]) | 3275 | 0.0974 -> 0.1023 | 678 | 0.653 -> 0.654 |
| zerodce: s1 | arcface | 938 | 0.340 -> 0.311 (-0.029 [-0.048, -0.012]) | 3275 | 0.1408 -> 0.1417 | 678 | 0.670 -> 0.672 |
| zerodce: s1 | adaface | 938 | 0.380 -> 0.355 (-0.025 [-0.045, -0.003]) | 3275 | 0.1035 -> 0.1008 | 678 | 0.664 -> 0.634 |
| zerodce: s1 | magface | 938 | 0.382 -> 0.364 (-0.018 [-0.034, -0.002]) | 3275 | 0.1340 -> 0.1392 | 678 | 0.690 -> 0.682 |
| zerodce: s1 | curricularface | 938 | 0.342 -> 0.319 (-0.023 [-0.042, -0.005]) | 3275 | 0.0974 -> 0.1035 | 678 | 0.653 -> 0.652 |
| nafnet: pre0 | arcface | 938 | 0.340 -> 0.327 (-0.013 [-0.030, 0.005]) | 3275 | 0.1408 -> 0.1334 | 678 | 0.670 -> 0.656 |
| nafnet: pre0 | adaface | 938 | 0.380 -> 0.342 (-0.037 [-0.059, -0.018]) | 3275 | 0.1035 -> 0.0944 | 678 | 0.664 -> 0.645 |
| nafnet: pre0 | magface | 938 | 0.382 -> 0.351 (-0.031 [-0.049, -0.013]) | 3275 | 0.1340 -> 0.1264 | 678 | 0.690 -> 0.683 |
| nafnet: pre0 | curricularface | 938 | 0.342 -> 0.332 (-0.011 [-0.028, 0.007]) | 3275 | 0.0974 -> 0.0931 | 678 | 0.653 -> 0.644 |
| gate_lowlight: q10__gamma_gadaptive | arcface | 938 | 0.340 -> 0.338 (-0.002 [-0.007, 0.002]) | 3275 | 0.1408 -> 0.1414 | 678 | 0.670 -> 0.665 |
| gate_lowlight: q10__gamma_gadaptive | adaface | 938 | 0.380 -> 0.374 (-0.005 [-0.012, 0.001]) | 3275 | 0.1035 -> 0.1026 | 678 | 0.664 -> 0.659 |
| gate_lowlight: q10__gamma_gadaptive | magface | 938 | 0.382 -> 0.380 (-0.002 [-0.010, 0.005]) | 3275 | 0.1340 -> 0.1319 | 678 | 0.690 -> 0.690 |
| gate_lowlight: q10__gamma_gadaptive | curricularface | 938 | 0.342 -> 0.338 (-0.004 [-0.012, 0.002]) | 3275 | 0.0974 -> 0.0950 | 678 | 0.653 -> 0.652 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | arcface | 938 | 0.340 -> 0.340 (+0.000 [-0.004, 0.004]) | 3275 | 0.1408 -> 0.1453 | 678 | 0.670 -> 0.670 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | adaface | 938 | 0.380 -> 0.378 (-0.001 [-0.003, 0.000]) | 3275 | 0.1035 -> 0.0995 | 678 | 0.664 -> 0.664 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | magface | 938 | 0.382 -> 0.382 (+0.000 [-0.004, 0.004]) | 3275 | 0.1340 -> 0.1362 | 678 | 0.690 -> 0.689 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | curricularface | 938 | 0.342 -> 0.342 (+0.000 [-0.004, 0.003]) | 3275 | 0.0974 -> 0.1005 | 678 | 0.653 -> 0.654 |

**Slice `dark10`**

| Arm | Recognizer | n probes | TinyFace Rank-1: direct -> restored (paired delta) | n mated | SurvFace TPIR20@FPIR=0.3: direct -> restored | n pos pairs | SurvFace AUC: direct -> restored |
|---|---|---|---|---|---|---|---|
| bicubic: pre128 | arcface | 98 | 0.194 -> 0.173 (-0.020 [-0.051, 0.000]) | 325 | 0.0646 -> 0.0677 | 148 | 0.581 -> 0.580 |
| bicubic: pre128 | adaface | 98 | 0.204 -> 0.194 (-0.010 [-0.061, 0.041]) | 325 | 0.0338 -> 0.0400 | 148 | 0.628 -> 0.624 |
| bicubic: pre128 | magface | 98 | 0.276 -> 0.255 (-0.020 [-0.061, 0.020]) | 325 | 0.0523 -> 0.0462 | 148 | 0.628 -> 0.628 |
| bicubic: pre128 | curricularface | 98 | 0.245 -> 0.214 (-0.031 [-0.082, 0.010]) | 325 | 0.0369 -> 0.0462 | 148 | 0.601 -> 0.611 |
| gamma: gadaptive | arcface | 98 | 0.194 -> 0.184 (-0.010 [-0.071, 0.041]) | 325 | 0.0646 -> 0.0554 | 148 | 0.581 -> 0.577 |
| gamma: gadaptive | adaface | 98 | 0.204 -> 0.153 (-0.051 [-0.112, 0.000]) | 325 | 0.0338 -> 0.0338 | 148 | 0.628 -> 0.600 |
| gamma: gadaptive | magface | 98 | 0.276 -> 0.245 (-0.031 [-0.082, 0.020]) | 325 | 0.0523 -> 0.0646 | 148 | 0.628 -> 0.636 |
| gamma: gadaptive | curricularface | 98 | 0.245 -> 0.194 (-0.051 [-0.112, 0.010]) | 325 | 0.0369 -> 0.0400 | 148 | 0.601 -> 0.614 |
| clahe: c2_t2 | arcface | 98 | 0.194 -> 0.153 (-0.041 [-0.102, 0.020]) | 325 | 0.0646 -> 0.0585 | 148 | 0.581 -> 0.562 |
| clahe: c2_t2 | adaface | 98 | 0.204 -> 0.173 (-0.031 [-0.102, 0.041]) | 325 | 0.0338 -> 0.0277 | 148 | 0.628 -> 0.584 |
| clahe: c2_t2 | magface | 98 | 0.276 -> 0.214 (-0.061 [-0.133, 0.010]) | 325 | 0.0523 -> 0.0769 | 148 | 0.628 -> 0.615 |
| clahe: c2_t2 | curricularface | 98 | 0.245 -> 0.194 (-0.051 [-0.122, 0.020]) | 325 | 0.0369 -> 0.0400 | 148 | 0.601 -> 0.594 |
| unsharp: a0.5_s0.5 | arcface | 98 | 0.194 -> 0.163 (-0.031 [-0.071, 0.000]) | 325 | 0.0646 -> 0.0677 | 148 | 0.581 -> 0.578 |
| unsharp: a0.5_s0.5 | adaface | 98 | 0.204 -> 0.194 (-0.010 [-0.051, 0.020]) | 325 | 0.0338 -> 0.0338 | 148 | 0.628 -> 0.622 |
| unsharp: a0.5_s0.5 | magface | 98 | 0.276 -> 0.265 (-0.010 [-0.031, 0.000]) | 325 | 0.0523 -> 0.0523 | 148 | 0.628 -> 0.628 |
| unsharp: a0.5_s0.5 | curricularface | 98 | 0.245 -> 0.204 (-0.041 [-0.082, -0.010]) | 325 | 0.0369 -> 0.0554 | 148 | 0.601 -> 0.602 |
| zerodce: s1 | arcface | 98 | 0.194 -> 0.122 (-0.071 [-0.133, -0.020]) | 325 | 0.0646 -> 0.0492 | 148 | 0.581 -> 0.583 |
| zerodce: s1 | adaface | 98 | 0.204 -> 0.153 (-0.051 [-0.112, 0.010]) | 325 | 0.0338 -> 0.0369 | 148 | 0.628 -> 0.571 |
| zerodce: s1 | magface | 98 | 0.276 -> 0.235 (-0.041 [-0.102, 0.020]) | 325 | 0.0523 -> 0.0554 | 148 | 0.628 -> 0.608 |
| zerodce: s1 | curricularface | 98 | 0.245 -> 0.184 (-0.061 [-0.133, 0.010]) | 325 | 0.0369 -> 0.0554 | 148 | 0.601 -> 0.614 |
| nafnet: pre0 | arcface | 98 | 0.194 -> 0.163 (-0.031 [-0.092, 0.031]) | 325 | 0.0646 -> 0.0585 | 148 | 0.581 -> 0.582 |
| nafnet: pre0 | adaface | 98 | 0.204 -> 0.153 (-0.051 [-0.112, 0.000]) | 325 | 0.0338 -> 0.0308 | 148 | 0.628 -> 0.595 |
| nafnet: pre0 | magface | 98 | 0.276 -> 0.235 (-0.041 [-0.102, 0.020]) | 325 | 0.0523 -> 0.0615 | 148 | 0.628 -> 0.626 |
| nafnet: pre0 | curricularface | 98 | 0.245 -> 0.194 (-0.051 [-0.112, 0.010]) | 325 | 0.0369 -> 0.0646 | 148 | 0.601 -> 0.585 |
| gate_lowlight: q10__gamma_gadaptive | arcface | 98 | 0.194 -> 0.173 (-0.020 [-0.061, 0.020]) | 325 | 0.0646 -> 0.0554 | 148 | 0.581 -> 0.574 |
| gate_lowlight: q10__gamma_gadaptive | adaface | 98 | 0.204 -> 0.133 (-0.071 [-0.123, -0.020]) | 325 | 0.0338 -> 0.0185 | 148 | 0.628 -> 0.600 |
| gate_lowlight: q10__gamma_gadaptive | magface | 98 | 0.276 -> 0.245 (-0.031 [-0.082, 0.020]) | 325 | 0.0523 -> 0.0492 | 148 | 0.628 -> 0.634 |
| gate_lowlight: q10__gamma_gadaptive | curricularface | 98 | 0.245 -> 0.214 (-0.031 [-0.071, 0.010]) | 325 | 0.0369 -> 0.0400 | 148 | 0.601 -> 0.604 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | arcface | 98 | 0.194 -> 0.184 (-0.010 [-0.031, 0.000]) | 325 | 0.0646 -> 0.0677 | 148 | 0.581 -> 0.579 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | adaface | 98 | 0.204 -> 0.204 (+0.000 [0.000, 0.000]) | 325 | 0.0338 -> 0.0369 | 148 | 0.628 -> 0.625 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | magface | 98 | 0.276 -> 0.276 (+0.000 [0.000, 0.000]) | 325 | 0.0523 -> 0.0554 | 148 | 0.628 -> 0.629 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | curricularface | 98 | 0.245 -> 0.245 (+0.000 [0.000, 0.000]) | 325 | 0.0369 -> 0.0554 | 148 | 0.601 -> 0.602 |

**Slice `not_dark10`**

| Arm | Recognizer | n probes | TinyFace Rank-1: direct -> restored (paired delta) | n mated | SurvFace TPIR20@FPIR=0.3: direct -> restored | n pos pairs | SurvFace AUC: direct -> restored |
|---|---|---|---|---|---|---|---|
| bicubic: pre128 | arcface | 1057 | 0.312 -> 0.309 (-0.003 [-0.016, 0.011]) | 4151 | 0.2021 -> 0.2012 | 852 | 0.679 -> 0.690 |
| bicubic: pre128 | adaface | 1057 | 0.348 -> 0.360 (+0.011 [-0.002, 0.025]) | 4151 | 0.1433 -> 0.1404 | 852 | 0.657 -> 0.669 |
| bicubic: pre128 | magface | 1057 | 0.342 -> 0.324 (-0.018 [-0.032, -0.006]) | 4151 | 0.1872 -> 0.1790 | 852 | 0.682 -> 0.680 |
| bicubic: pre128 | curricularface | 1057 | 0.303 -> 0.289 (-0.013 [-0.029, 0.003]) | 4151 | 0.1236 -> 0.1168 | 852 | 0.656 -> 0.652 |
| gamma: gadaptive | arcface | 1057 | 0.312 -> 0.311 (-0.001 [-0.010, 0.009]) | 4151 | 0.2021 -> 0.2053 | 852 | 0.679 -> 0.680 |
| gamma: gadaptive | adaface | 1057 | 0.348 -> 0.341 (-0.008 [-0.015, 0.000]) | 4151 | 0.1433 -> 0.1270 | 852 | 0.657 -> 0.647 |
| gamma: gadaptive | magface | 1057 | 0.342 -> 0.342 (+0.001 [-0.009, 0.010]) | 4151 | 0.1872 -> 0.1754 | 852 | 0.682 -> 0.679 |
| gamma: gadaptive | curricularface | 1057 | 0.303 -> 0.298 (-0.005 [-0.014, 0.005]) | 4151 | 0.1236 -> 0.1219 | 852 | 0.656 -> 0.648 |
| clahe: c2_t2 | arcface | 1057 | 0.312 -> 0.274 (-0.038 [-0.056, -0.021]) | 4151 | 0.2021 -> 0.1679 | 852 | 0.679 -> 0.677 |
| clahe: c2_t2 | adaface | 1057 | 0.348 -> 0.304 (-0.044 [-0.061, -0.025]) | 4151 | 0.1433 -> 0.1349 | 852 | 0.657 -> 0.635 |
| clahe: c2_t2 | magface | 1057 | 0.342 -> 0.303 (-0.039 [-0.056, -0.022]) | 4151 | 0.1872 -> 0.1713 | 852 | 0.682 -> 0.675 |
| clahe: c2_t2 | curricularface | 1057 | 0.303 -> 0.265 (-0.038 [-0.057, -0.020]) | 4151 | 0.1236 -> 0.1132 | 852 | 0.656 -> 0.645 |
| unsharp: a0.5_s0.5 | arcface | 1057 | 0.312 -> 0.305 (-0.008 [-0.017, 0.002]) | 4151 | 0.2021 -> 0.2040 | 852 | 0.679 -> 0.680 |
| unsharp: a0.5_s0.5 | adaface | 1057 | 0.348 -> 0.340 (-0.009 [-0.019, 0.003]) | 4151 | 0.1433 -> 0.1298 | 852 | 0.657 -> 0.653 |
| unsharp: a0.5_s0.5 | magface | 1057 | 0.342 -> 0.339 (-0.003 [-0.012, 0.007]) | 4151 | 0.1872 -> 0.1848 | 852 | 0.682 -> 0.679 |
| unsharp: a0.5_s0.5 | curricularface | 1057 | 0.303 -> 0.298 (-0.005 [-0.014, 0.005]) | 4151 | 0.1236 -> 0.1137 | 852 | 0.656 -> 0.655 |
| zerodce: s1 | arcface | 1057 | 0.312 -> 0.287 (-0.026 [-0.043, -0.009]) | 4151 | 0.2021 -> 0.2146 | 852 | 0.679 -> 0.685 |
| zerodce: s1 | adaface | 1057 | 0.348 -> 0.322 (-0.026 [-0.045, -0.007]) | 4151 | 0.1433 -> 0.1311 | 852 | 0.657 -> 0.634 |
| zerodce: s1 | magface | 1057 | 0.342 -> 0.321 (-0.021 [-0.037, -0.006]) | 4151 | 0.1872 -> 0.1968 | 852 | 0.682 -> 0.681 |
| zerodce: s1 | curricularface | 1057 | 0.303 -> 0.284 (-0.019 [-0.034, -0.001]) | 4151 | 0.1236 -> 0.1371 | 852 | 0.656 -> 0.645 |
| nafnet: pre0 | arcface | 1057 | 0.312 -> 0.291 (-0.021 [-0.037, -0.004]) | 4151 | 0.2021 -> 0.1802 | 852 | 0.679 -> 0.660 |
| nafnet: pre0 | adaface | 1057 | 0.348 -> 0.312 (-0.036 [-0.055, -0.016]) | 4151 | 0.1433 -> 0.1270 | 852 | 0.657 -> 0.637 |
| nafnet: pre0 | magface | 1057 | 0.342 -> 0.310 (-0.031 [-0.048, -0.014]) | 4151 | 0.1872 -> 0.1669 | 852 | 0.682 -> 0.670 |
| nafnet: pre0 | curricularface | 1057 | 0.303 -> 0.293 (-0.009 [-0.026, 0.006]) | 4151 | 0.1236 -> 0.1159 | 852 | 0.656 -> 0.638 |
| gate_lowlight: q10__gamma_gadaptive | arcface | 1057 | 0.312 -> 0.313 (+0.001 [-0.002, 0.005]) | 4151 | 0.2021 -> 0.2038 | 852 | 0.679 -> 0.678 |
| gate_lowlight: q10__gamma_gadaptive | adaface | 1057 | 0.348 -> 0.346 (-0.002 [-0.006, 0.002]) | 4151 | 0.1433 -> 0.1419 | 852 | 0.657 -> 0.655 |
| gate_lowlight: q10__gamma_gadaptive | magface | 1057 | 0.342 -> 0.342 (+0.000 [-0.005, 0.005]) | 4151 | 0.1872 -> 0.1857 | 852 | 0.682 -> 0.682 |
| gate_lowlight: q10__gamma_gadaptive | curricularface | 1057 | 0.303 -> 0.303 (+0.000 [-0.005, 0.005]) | 4151 | 0.1236 -> 0.1233 | 852 | 0.656 -> 0.655 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | arcface | 1057 | 0.312 -> 0.312 (+0.000 [-0.004, 0.004]) | 4151 | 0.2021 -> 0.2033 | 852 | 0.679 -> 0.679 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | adaface | 1057 | 0.348 -> 0.346 (-0.002 [-0.006, 0.002]) | 4151 | 0.1433 -> 0.1284 | 852 | 0.657 -> 0.655 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | magface | 1057 | 0.342 -> 0.340 (-0.002 [-0.007, 0.003]) | 4151 | 0.1872 -> 0.1865 | 852 | 0.682 -> 0.680 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | curricularface | 1057 | 0.303 -> 0.301 (-0.002 [-0.006, 0.003]) | 4151 | 0.1236 -> 0.1130 | 852 | 0.656 -> 0.655 |

**Slice `dark`**

| Arm | Recognizer | n probes | TinyFace Rank-1: direct -> restored (paired delta) | n mated | SurvFace TPIR20@FPIR=0.3: direct -> restored | n pos pairs | SurvFace AUC: direct -> restored |
|---|---|---|---|---|---|---|---|
| bicubic: pre128 | arcface | 236 | 0.263 -> 0.254 (-0.008 [-0.030, 0.013]) | 671 | 0.0864 -> 0.0939 | 291 | 0.595 -> 0.598 |
| bicubic: pre128 | adaface | 236 | 0.297 -> 0.292 (-0.004 [-0.034, 0.025]) | 671 | 0.0760 -> 0.0745 | 291 | 0.605 -> 0.609 |
| bicubic: pre128 | magface | 236 | 0.305 -> 0.292 (-0.013 [-0.034, 0.013]) | 671 | 0.0894 -> 0.0790 | 291 | 0.619 -> 0.622 |
| bicubic: pre128 | curricularface | 236 | 0.267 -> 0.242 (-0.025 [-0.051, 0.000]) | 671 | 0.0447 -> 0.0522 | 291 | 0.584 -> 0.592 |
| gamma: gadaptive | arcface | 236 | 0.263 -> 0.246 (-0.017 [-0.055, 0.017]) | 671 | 0.0864 -> 0.0760 | 291 | 0.595 -> 0.592 |
| gamma: gadaptive | adaface | 236 | 0.297 -> 0.258 (-0.038 [-0.064, -0.008]) | 671 | 0.0760 -> 0.0775 | 291 | 0.605 -> 0.586 |
| gamma: gadaptive | magface | 236 | 0.305 -> 0.292 (-0.013 [-0.047, 0.021]) | 671 | 0.0894 -> 0.0849 | 291 | 0.619 -> 0.625 |
| gamma: gadaptive | curricularface | 236 | 0.267 -> 0.242 (-0.025 [-0.059, 0.008]) | 671 | 0.0447 -> 0.0462 | 291 | 0.584 -> 0.586 |
| clahe: c2_t2 | arcface | 236 | 0.263 -> 0.216 (-0.047 [-0.089, -0.008]) | 671 | 0.0864 -> 0.0700 | 291 | 0.595 -> 0.593 |
| clahe: c2_t2 | adaface | 236 | 0.297 -> 0.263 (-0.034 [-0.076, 0.004]) | 671 | 0.0760 -> 0.0596 | 291 | 0.605 -> 0.572 |
| clahe: c2_t2 | magface | 236 | 0.305 -> 0.275 (-0.030 [-0.068, 0.008]) | 671 | 0.0894 -> 0.1058 | 291 | 0.619 -> 0.622 |
| clahe: c2_t2 | curricularface | 236 | 0.267 -> 0.216 (-0.051 [-0.093, -0.013]) | 671 | 0.0447 -> 0.0656 | 291 | 0.584 -> 0.585 |
| unsharp: a0.5_s0.5 | arcface | 236 | 0.263 -> 0.250 (-0.013 [-0.034, 0.004]) | 671 | 0.0864 -> 0.0879 | 291 | 0.595 -> 0.596 |
| unsharp: a0.5_s0.5 | adaface | 236 | 0.297 -> 0.284 (-0.013 [-0.030, 0.004]) | 671 | 0.0760 -> 0.0790 | 291 | 0.605 -> 0.598 |
| unsharp: a0.5_s0.5 | magface | 236 | 0.305 -> 0.301 (-0.004 [-0.021, 0.013]) | 671 | 0.0894 -> 0.0924 | 291 | 0.619 -> 0.622 |
| unsharp: a0.5_s0.5 | curricularface | 236 | 0.267 -> 0.242 (-0.025 [-0.047, -0.008]) | 671 | 0.0447 -> 0.0507 | 291 | 0.584 -> 0.584 |
| zerodce: s1 | arcface | 236 | 0.263 -> 0.225 (-0.038 [-0.076, -0.004]) | 671 | 0.0864 -> 0.0611 | 291 | 0.595 -> 0.601 |
| zerodce: s1 | adaface | 236 | 0.297 -> 0.233 (-0.064 [-0.110, -0.021]) | 671 | 0.0760 -> 0.0715 | 291 | 0.605 -> 0.558 |
| zerodce: s1 | magface | 236 | 0.305 -> 0.275 (-0.030 [-0.068, 0.008]) | 671 | 0.0894 -> 0.0805 | 291 | 0.619 -> 0.612 |
| zerodce: s1 | curricularface | 236 | 0.267 -> 0.237 (-0.030 [-0.064, 0.000]) | 671 | 0.0447 -> 0.0566 | 291 | 0.584 -> 0.586 |
| nafnet: pre0 | arcface | 236 | 0.263 -> 0.229 (-0.034 [-0.072, 0.000]) | 671 | 0.0864 -> 0.0805 | 291 | 0.595 -> 0.599 |
| nafnet: pre0 | adaface | 236 | 0.297 -> 0.242 (-0.055 [-0.093, -0.017]) | 671 | 0.0760 -> 0.0477 | 291 | 0.605 -> 0.586 |
| nafnet: pre0 | magface | 236 | 0.305 -> 0.250 (-0.055 [-0.093, -0.017]) | 671 | 0.0894 -> 0.0894 | 291 | 0.619 -> 0.623 |
| nafnet: pre0 | curricularface | 236 | 0.267 -> 0.242 (-0.025 [-0.068, 0.013]) | 671 | 0.0447 -> 0.0537 | 291 | 0.584 -> 0.573 |
| gate_lowlight: q10__gamma_gadaptive | arcface | 236 | 0.263 -> 0.258 (-0.004 [-0.025, 0.013]) | 671 | 0.0864 -> 0.0849 | 291 | 0.595 -> 0.588 |
| gate_lowlight: q10__gamma_gadaptive | adaface | 236 | 0.297 -> 0.258 (-0.038 [-0.064, -0.013]) | 671 | 0.0760 -> 0.0656 | 291 | 0.605 -> 0.589 |
| gate_lowlight: q10__gamma_gadaptive | magface | 236 | 0.305 -> 0.288 (-0.017 [-0.042, 0.008]) | 671 | 0.0894 -> 0.0760 | 291 | 0.619 -> 0.622 |
| gate_lowlight: q10__gamma_gadaptive | curricularface | 236 | 0.267 -> 0.254 (-0.013 [-0.034, 0.004]) | 671 | 0.0447 -> 0.0417 | 291 | 0.584 -> 0.585 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | arcface | 236 | 0.263 -> 0.258 (-0.004 [-0.013, 0.000]) | 671 | 0.0864 -> 0.0879 | 291 | 0.595 -> 0.595 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | adaface | 236 | 0.297 -> 0.292 (-0.004 [-0.013, 0.000]) | 671 | 0.0760 -> 0.0745 | 291 | 0.605 -> 0.599 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | magface | 236 | 0.305 -> 0.305 (+0.000 [-0.013, 0.013]) | 671 | 0.0894 -> 0.0924 | 291 | 0.619 -> 0.621 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | curricularface | 236 | 0.267 -> 0.267 (+0.000 [0.000, 0.000]) | 671 | 0.0447 -> 0.0507 | 291 | 0.584 -> 0.583 |

**Slice `not_dark`**

| Arm | Recognizer | n probes | TinyFace Rank-1: direct -> restored (paired delta) | n mated | SurvFace TPIR20@FPIR=0.3: direct -> restored | n pos pairs | SurvFace AUC: direct -> restored |
|---|---|---|---|---|---|---|---|
| bicubic: pre128 | arcface | 919 | 0.312 -> 0.309 (-0.003 [-0.018, 0.012]) | 3805 | 0.2124 -> 0.2089 | 709 | 0.697 -> 0.709 |
| bicubic: pre128 | adaface | 919 | 0.346 -> 0.359 (+0.013 [0.000, 0.026]) | 3805 | 0.1443 -> 0.1453 | 709 | 0.673 -> 0.686 |
| bicubic: pre128 | magface | 919 | 0.344 -> 0.324 (-0.020 [-0.035, -0.004]) | 3805 | 0.1921 -> 0.1829 | 709 | 0.699 -> 0.695 |
| bicubic: pre128 | curricularface | 919 | 0.306 -> 0.294 (-0.012 [-0.028, 0.004]) | 3805 | 0.1309 -> 0.1240 | 709 | 0.676 -> 0.672 |
| gamma: gadaptive | arcface | 919 | 0.312 -> 0.314 (+0.002 [-0.007, 0.011]) | 3805 | 0.2124 -> 0.2145 | 709 | 0.697 -> 0.698 |
| gamma: gadaptive | adaface | 919 | 0.346 -> 0.342 (-0.004 [-0.013, 0.003]) | 3805 | 0.1443 -> 0.1293 | 709 | 0.673 -> 0.665 |
| gamma: gadaptive | magface | 919 | 0.344 -> 0.345 (+0.001 [-0.008, 0.011]) | 3805 | 0.1921 -> 0.1845 | 709 | 0.699 -> 0.694 |
| gamma: gadaptive | curricularface | 919 | 0.306 -> 0.301 (-0.004 [-0.015, 0.005]) | 3805 | 0.1309 -> 0.1306 | 709 | 0.676 -> 0.670 |
| clahe: c2_t2 | arcface | 919 | 0.312 -> 0.276 (-0.036 [-0.054, -0.018]) | 3805 | 0.2124 -> 0.1748 | 709 | 0.697 -> 0.690 |
| clahe: c2_t2 | adaface | 919 | 0.346 -> 0.300 (-0.046 [-0.065, -0.025]) | 3805 | 0.1443 -> 0.1382 | 709 | 0.673 -> 0.653 |
| clahe: c2_t2 | magface | 919 | 0.344 -> 0.300 (-0.044 [-0.062, -0.026]) | 3805 | 0.1921 -> 0.1766 | 709 | 0.699 -> 0.687 |
| clahe: c2_t2 | curricularface | 919 | 0.306 -> 0.270 (-0.036 [-0.056, -0.015]) | 3805 | 0.1309 -> 0.1154 | 709 | 0.676 -> 0.662 |
| unsharp: a0.5_s0.5 | arcface | 919 | 0.312 -> 0.304 (-0.009 [-0.018, 0.002]) | 3805 | 0.2124 -> 0.2152 | 709 | 0.697 -> 0.698 |
| unsharp: a0.5_s0.5 | adaface | 919 | 0.346 -> 0.338 (-0.008 [-0.020, 0.004]) | 3805 | 0.1443 -> 0.1306 | 709 | 0.673 -> 0.671 |
| unsharp: a0.5_s0.5 | magface | 919 | 0.344 -> 0.341 (-0.003 [-0.013, 0.007]) | 3805 | 0.1921 -> 0.1903 | 709 | 0.699 -> 0.695 |
| unsharp: a0.5_s0.5 | curricularface | 919 | 0.306 -> 0.303 (-0.003 [-0.014, 0.008]) | 3805 | 0.1309 -> 0.1217 | 709 | 0.676 -> 0.677 |
| zerodce: s1 | arcface | 919 | 0.312 -> 0.285 (-0.027 [-0.047, -0.010]) | 3805 | 0.2124 -> 0.2276 | 709 | 0.697 -> 0.700 |
| zerodce: s1 | adaface | 919 | 0.346 -> 0.326 (-0.020 [-0.038, 0.000]) | 3805 | 0.1443 -> 0.1372 | 709 | 0.673 -> 0.655 |
| zerodce: s1 | magface | 919 | 0.344 -> 0.323 (-0.021 [-0.037, -0.004]) | 3805 | 0.1921 -> 0.2066 | 709 | 0.699 -> 0.695 |
| zerodce: s1 | curricularface | 919 | 0.306 -> 0.285 (-0.021 [-0.038, -0.003]) | 3805 | 0.1309 -> 0.1438 | 709 | 0.676 -> 0.664 |
| nafnet: pre0 | arcface | 919 | 0.312 -> 0.294 (-0.018 [-0.037, -0.001]) | 3805 | 0.2124 -> 0.1858 | 709 | 0.697 -> 0.671 |
| nafnet: pre0 | adaface | 919 | 0.346 -> 0.313 (-0.033 [-0.052, -0.012]) | 3805 | 0.1443 -> 0.1338 | 709 | 0.673 -> 0.651 |
| nafnet: pre0 | magface | 919 | 0.344 -> 0.318 (-0.026 [-0.045, -0.009]) | 3805 | 0.1921 -> 0.1703 | 709 | 0.699 -> 0.683 |
| nafnet: pre0 | curricularface | 919 | 0.306 -> 0.296 (-0.010 [-0.026, 0.008]) | 3805 | 0.1309 -> 0.1230 | 709 | 0.676 -> 0.658 |
| gate_lowlight: q10__gamma_gadaptive | arcface | 919 | 0.312 -> 0.312 (+0.000 [-0.003, 0.003]) | 3805 | 0.2124 -> 0.2129 | 709 | 0.697 -> 0.697 |
| gate_lowlight: q10__gamma_gadaptive | adaface | 919 | 0.346 -> 0.346 (+0.000 [-0.003, 0.003]) | 3805 | 0.1443 -> 0.1427 | 709 | 0.673 -> 0.673 |
| gate_lowlight: q10__gamma_gadaptive | magface | 919 | 0.344 -> 0.345 (+0.001 [-0.002, 0.005]) | 3805 | 0.1921 -> 0.1905 | 709 | 0.699 -> 0.699 |
| gate_lowlight: q10__gamma_gadaptive | curricularface | 919 | 0.306 -> 0.306 (+0.000 [-0.005, 0.005]) | 3805 | 0.1309 -> 0.1304 | 709 | 0.676 -> 0.676 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | arcface | 919 | 0.312 -> 0.312 (+0.000 [-0.004, 0.004]) | 3805 | 0.2124 -> 0.2155 | 709 | 0.697 -> 0.696 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | adaface | 919 | 0.346 -> 0.345 (-0.001 [-0.005, 0.002]) | 3805 | 0.1443 -> 0.1298 | 709 | 0.673 -> 0.673 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | magface | 919 | 0.344 -> 0.342 (-0.002 [-0.007, 0.002]) | 3805 | 0.1921 -> 0.1905 | 709 | 0.699 -> 0.695 |
| gate_deblur: q50__unsharp_a0.5_s0.5 | curricularface | 919 | 0.306 -> 0.304 (-0.002 [-0.008, 0.003]) | 3805 | 0.1309 -> 0.1185 | 709 | 0.676 -> 0.677 |

<!-- EXTRA:END -->

## 6. LFW synthetic check (dark, Gaussian blur, motion blur)
<!-- LFW:BEGIN -->
Degradations of the sharp 112 px LFW crops (6,000 official pairs); strengths chosen on direct feed only, before any arm was run (`lfw_enh_calibration.csv`): the strength whose mean non-leaky cross-degraded AUC is closest to 0.80 inside 0.70-0.90. Motion blur is a fixed 11 px line at a seeded angle (not tuned).

| Degradation | Strength | Mean non-leaky cross AUC (direct) | In 0.70-0.90 |
|---|---|---|---|
| dark | 0.05 | 0.836 | yes |
| gauss | 8 | 0.711 | yes |
| motion | 11 | 0.999 | NO |

Ceiling (sharp, sharp), mean non-leaky AUC: 0.9993.

**Absolute verification AUC by condition**

| Degradation | Condition | Arm | ArcFace | AdaFace | MagFace | CurricularFace |
|---|---|---|---|---|---|---|
| dark | ceiling (HR, HR) | noop | 0.999 | 0.999 | 0.999 | 0.999 |
| dark | cross degraded (deg, HR) | noop | 0.832 | 0.893 | 0.817 | 0.798 |
| dark | symmetric degraded (deg, deg) | noop | 0.734 | 0.815 | 0.754 | 0.705 |
| dark | cross degraded (deg, HR) | gamma: gadaptive | 0.832 | 0.893 | 0.817 | 0.798 |
| dark | symmetric degraded (deg, deg) | gamma: gadaptive | 0.734 | 0.815 | 0.754 | 0.705 |
| dark | cross enhanced (enh, HR) | gamma: gadaptive | 0.824 | 0.840 | 0.814 | 0.734 |
| dark | symmetric enhanced (enh, enh) | gamma: gadaptive | 0.761 | 0.795 | 0.769 | 0.731 |
| dark | cross degraded (deg, HR) | clahe: c2_t2 | 0.832 | 0.893 | 0.817 | 0.798 |
| dark | symmetric degraded (deg, deg) | clahe: c2_t2 | 0.734 | 0.815 | 0.754 | 0.705 |
| dark | cross enhanced (enh, HR) | clahe: c2_t2 | 0.846 | 0.856 | 0.818 | 0.785 |
| dark | symmetric enhanced (enh, enh) | clahe: c2_t2 | 0.757 | 0.788 | 0.760 | 0.720 |
| dark | cross degraded (deg, HR) | zerodce: s1 | 0.832 | 0.893 | 0.817 | 0.798 |
| dark | symmetric degraded (deg, deg) | zerodce: s1 | 0.734 | 0.815 | 0.754 | 0.705 |
| dark | cross enhanced (enh, HR) | zerodce: s1 | 0.858 | 0.864 | 0.836 | 0.774 |
| dark | symmetric enhanced (enh, enh) | zerodce: s1 | 0.776 | 0.811 | 0.784 | 0.748 |
| gauss | ceiling (HR, HR) | noop | 0.999 | 0.999 | 0.999 | 0.999 |
| gauss | cross degraded (deg, HR) | noop | 0.785 | 0.746 | 0.721 | 0.665 |
| gauss | symmetric degraded (deg, deg) | noop | 0.871 | 0.828 | 0.849 | 0.774 |
| gauss | cross degraded (deg, HR) | unsharp: a0.5_s0.5 | 0.785 | 0.746 | 0.721 | 0.665 |
| gauss | symmetric degraded (deg, deg) | unsharp: a0.5_s0.5 | 0.871 | 0.828 | 0.849 | 0.774 |
| gauss | cross enhanced (enh, HR) | unsharp: a0.5_s0.5 | 0.785 | 0.746 | 0.721 | 0.665 |
| gauss | symmetric enhanced (enh, enh) | unsharp: a0.5_s0.5 | 0.871 | 0.828 | 0.849 | 0.774 |
| gauss | cross degraded (deg, HR) | nafnet: pre0 | 0.785 | 0.746 | 0.721 | 0.665 |
| gauss | symmetric degraded (deg, deg) | nafnet: pre0 | 0.871 | 0.828 | 0.849 | 0.774 |
| gauss | cross enhanced (enh, HR) | nafnet: pre0 | 0.857 | 0.823 | 0.795 | 0.757 |
| gauss | symmetric enhanced (enh, enh) | nafnet: pre0 | 0.895 | 0.860 | 0.873 | 0.828 |
| motion | ceiling (HR, HR) | noop | 0.999 | 0.999 | 0.999 | 0.999 |
| motion | cross degraded (deg, HR) | noop | 0.998 | 0.999 | 0.999 | 0.998 |
| motion | symmetric degraded (deg, deg) | noop | 0.995 | 0.988 | 0.992 | 0.989 |
| motion | cross degraded (deg, HR) | unsharp: a0.5_s0.5 | 0.998 | 0.999 | 0.999 | 0.998 |
| motion | symmetric degraded (deg, deg) | unsharp: a0.5_s0.5 | 0.995 | 0.988 | 0.992 | 0.989 |
| motion | cross enhanced (enh, HR) | unsharp: a0.5_s0.5 | 0.998 | 0.999 | 0.999 | 0.998 |
| motion | symmetric enhanced (enh, enh) | unsharp: a0.5_s0.5 | 0.995 | 0.988 | 0.991 | 0.989 |
| motion | cross degraded (deg, HR) | nafnet: pre0 | 0.998 | 0.999 | 0.999 | 0.998 |
| motion | symmetric degraded (deg, deg) | nafnet: pre0 | 0.995 | 0.988 | 0.992 | 0.989 |
| motion | cross enhanced (enh, HR) | nafnet: pre0 | 0.999 | 0.999 | 1.000 | 0.999 |
| motion | symmetric enhanced (enh, enh) | nafnet: pre0 | 0.999 | 0.999 | 1.000 | 0.999 |

**Pre-registered contrast: cross-enhanced minus cross-degraded AUC** (rule: CI above zero for at least 2 of the 3 non-leaky recognizers). noop is the degraded crop itself, so this is also the contrast against no-op.

| Degradation | Arm | ArcFace | AdaFace | MagFace | CurricularFace | Non-leaky positive | Rule |
|---|---|---|---|---|---|---|---|
| dark | gamma: gadaptive | -0.008 [-0.015, -0.000] | -0.053 [-0.059, -0.046] | -0.003 [-0.010, +0.005] | -0.064 [-0.073, -0.055] | 0 of 3 | not supported |
| dark | clahe: c2_t2 | +0.014 [+0.011, +0.017] | -0.037 [-0.041, -0.033] | +0.001 [-0.003, +0.005] | -0.013 [-0.017, -0.008] | 0 of 3 | not supported |
| dark | zerodce: s1 | +0.026 [+0.020, +0.032] | -0.029 [-0.033, -0.025] | +0.019 [+0.013, +0.026] | -0.024 [-0.032, -0.017] | 1 of 3 | not supported |
| gauss | unsharp: a0.5_s0.5 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0 of 3 | not supported |
| gauss | nafnet: pre0 | +0.073 [+0.067, +0.079] | +0.077 [+0.070, +0.083] | +0.074 [+0.068, +0.080] | +0.092 [+0.083, +0.100] | 3 of 3 | met |
| motion | unsharp: a0.5_s0.5 | -0.000 [-0.000, -0.000] | -0.000 [-0.000, +0.000] | -0.000 [-0.000, +0.000] | -0.000 [-0.000, -0.000] | 0 of 3 | not supported |
| motion | nafnet: pre0 | +0.001 [+0.000, +0.002] | +0.001 [+0.000, +0.001] | +0.001 [+0.000, +0.002] | +0.001 [+0.000, +0.001] | 3 of 3 | met |

**Same contrast with both images degraded (symmetric)**

| Degradation | Arm | ArcFace | AdaFace | MagFace | CurricularFace |
|---|---|---|---|---|---|
| dark | gamma: gadaptive | +0.028 [+0.020, +0.035] | -0.020 [-0.026, -0.013] | +0.015 [+0.007, +0.023] | +0.026 [+0.018, +0.033] |
| dark | clahe: c2_t2 | +0.024 [+0.021, +0.028] | -0.026 [-0.030, -0.022] | +0.006 [+0.001, +0.010] | +0.014 [+0.010, +0.019] |
| dark | zerodce: s1 | +0.043 [+0.037, +0.049] | -0.004 [-0.008, +0.001] | +0.030 [+0.023, +0.036] | +0.043 [+0.036, +0.050] |
| gauss | unsharp: a0.5_s0.5 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| gauss | nafnet: pre0 | +0.024 [+0.020, +0.028] | +0.032 [+0.027, +0.037] | +0.024 [+0.020, +0.028] | +0.054 [+0.047, +0.062] |
| motion | unsharp: a0.5_s0.5 | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | -0.000 [-0.001, -0.000] | -0.000 [-0.001, -0.000] |
| motion | nafnet: pre0 | +0.004 [+0.003, +0.006] | +0.011 [+0.009, +0.013] | +0.008 [+0.006, +0.010] | +0.010 [+0.008, +0.012] |

<!-- LFW:END -->

## 7. Labelled visual samples
Random, seeded (42) crops, identical for every method, never picked by outcome: dark examples from the darkest-10% bin (`outputs/enhance/samples/dark_tinyface.png`, `dark_survface.png`), blur examples from the low-sharpness bin (`blur_tinyface.png`, `blur_survface.png`). Each row is labelled with dataset, crop name, size and the problem seen. Failure sheets per arm (identity changed, lost matches, random) are in `outputs/enhance/failure_cases/`; strips for the brief are `docs/figures/enh_*.png`.

What the samples show (visual judgement, not a measured result):
- **gamma and Zero-DCE++** brighten the face but reveal colour casts, saturated pixels around the eyes and block patterns of the compressed source; Zero-DCE++ adds a pink tint.
- **CLAHE** raises contrast strongly and amplifies noise and block edges; it looks the most altered.
- **unsharp** is almost indistinguishable from the input at these sizes.
- **NAFNet** gives slightly crisper edges around eyes and mouth with no obvious invented detail in these samples, but it cannot recover the lost resolution.

## 8. Cost: s/crop on CPU and GPU
Eval-subset restoration timings and the 75-crop smoke CPU/GPU comparison are in the generated cost table in section 5 (the tables block). All arms cost milliseconds per crop at these sizes: the OpenCV arms about 0.008-0.012 s/crop on CPU; Zero-DCE++ about 0.010 s/crop on CPU and 0.011 on GPU (no speedup, the model has about 10k parameters); NAFNet native 0.029 s/crop on CPU against 0.011 on GPU in the smoke run (0.0047 s/crop measured in the batched eval run, peak VRAM 459 MB); NAFNet pre-128 0.331 s/crop on CPU against 0.024 on GPU in the smoke run. Cost is not a reason for or against any arm.

## 9. Full tables (generated)
_Injected between the markers above._

## 10. Limitations
- **lap_var gate is size-confounded** (Phase 2): the deblur gate partly selects small crops.
- **Dark-bin size:** 236 dark probes (98 in dark10), so their intervals are wide and a small gain could not be excluded there; the point estimates are still all at or below zero.
- **Synthetic LFW degradations:** dark = scaling plus Poisson-Gaussian noise, blur = Gaussian sigma 8; neither is the degradation of real TinyFace or SurvFace probes. The strengths come from a coarse grid (dark s = 0.05 gives cross-AUC 0.836, above the 0.80 target; no grid value fell between 0.05 and 0.03). The fixed 11 px motion blur barely degrades recognition (direct cross-AUC 0.9985, out of the 0.70-0.90 range), so the motion rows cannot discriminate arms; its formal "met" result for NAFNet is a ceiling effect (+0.0006 to +0.0009).
- **Multiplicity:** 8 tests on the stop rule (up to 10 allowed); the Bonferroni sensitivity CIs lead to the same verdict.
- **Different GPU from Phase 3:** a laptop RTX 3000, not the lab-PC RTX 4000 Ada; all Phase 4 numbers use one device, so the paired comparisons are internally consistent, but absolute values can differ slightly from Phase 3 (cross-device cosine above 0.9999).
- **LFW check scope:** only the arms' selected settings were run, at 112 px; the gated variants were not run on LFW; noop equals the degraded crop, so the contrast against noop equals the contrast against cross-degraded.
- **Thumbnails:** the uploaded thumbnails are a seeded sample, not every eval crop (see the upload manifest).

## 11. Decision for Phase 8
**Restoration closed.** No low-light or deblur arm, and no gated variant, gives a probes-only recognition gain with a 95% CI above zero, so none becomes the Phase 8 gate arm. Direct feed stays the default for Phase 8. The one positive signal (NAFNet on synthetic Gaussian blur) is recorded as a result but does not carry over to real probes,, so it does not reopen the engine.

*Pending supervisor approval at the end-of-W2 checkpoint.*
