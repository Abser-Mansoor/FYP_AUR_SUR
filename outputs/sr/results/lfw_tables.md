Low-resolution side: 10 px (chosen on direct feed only; at 24 px direct feed is at the ceiling, AUC 0.998-0.999). 6,000 LFW pairs. Cells: AUC; contrasts are paired differences with 95% CIs (positive and negative pairs resampled separately, 1,000 draws). † = leaky.

**Absolute verification AUC by condition**

| Condition | Arm | ArcFace | AdaFace | MagFace | CurricularFace |
|---|---|---|---|---|---|
| ceiling (HR, HR) | direct feed | 0.999 | 0.999 | 0.999 | 0.999 |
| cross direct (LRpx, HR) | direct feed | 0.866 | 0.875 | 0.835 | 0.828 |
| symmetric direct (LRpx, LRpx) | direct feed | 0.902 | 0.897 | 0.886 | 0.884 |
| cross restored (restored, HR) | Bicubic | 0.894 | 0.897 | 0.852 | 0.835 |
| cross restored (restored, HR) | Real-ESRGAN native | 0.715 | 0.695 | 0.706 | 0.671 |
| cross restored (restored, HR) | InvSR pre-64 | 0.732 | 0.715 | 0.736 | 0.696 |
| cross restored (restored, HR) | TADSR pre-128 / t250 | 0.895 | 0.887 | 0.866 | 0.841 |
| cross restored (restored, HR) | OSDFace pre-128 | 0.845 † | 0.847 | 0.845 | 0.839 |
| symmetric restored (restored, restored) | Bicubic | 0.918 | 0.916 | 0.901 | 0.893 |
| symmetric restored (restored, restored) | Real-ESRGAN native | 0.771 | 0.757 | 0.768 | 0.735 |
| symmetric restored (restored, restored) | InvSR pre-64 | 0.760 | 0.755 | 0.758 | 0.737 |
| symmetric restored (restored, restored) | TADSR pre-128 / t250 | 0.915 | 0.911 | 0.902 | 0.895 |
| symmetric restored (restored, restored) | OSDFace pre-128 | 0.884 † | 0.880 | 0.878 | 0.878 |

**Pre-registered contrast: cross-restored minus cross-direct AUC** (rule: CI above zero for at least 2 of the 3 non-leaky recognizers)

| Arm | ArcFace | AdaFace | MagFace | CurricularFace | Non-leaky positive | Rule |
|---|---|---|---|---|---|---|
| Bicubic | +0.027 [+0.022, +0.032] | +0.022 [+0.016, +0.028] | +0.017 [+0.012, +0.023] | +0.008 [+0.001, +0.014] | 3 of 3 | met |
| Real-ESRGAN native | -0.151 [-0.165, -0.139] | -0.180 [-0.194, -0.167] | -0.129 [-0.142, -0.117] | -0.157 [-0.171, -0.142] | 0 of 3 | not supported |
| InvSR pre-64 | -0.134 [-0.146, -0.121] | -0.160 [-0.173, -0.147] | -0.099 [-0.111, -0.087] | -0.132 [-0.146, -0.118] | 0 of 3 | not supported |
| TADSR pre-128 / t250 | +0.029 [+0.023, +0.035] | +0.012 [+0.005, +0.019] | +0.031 [+0.025, +0.038] | +0.014 [+0.006, +0.020] | 3 of 3 | met |
| OSDFace pre-128 | -0.022 [-0.033, -0.011] † | -0.028 [-0.039, -0.017] | +0.010 [-0.002, +0.022] | +0.011 [-0.002, +0.023] | 0 of 3 | not supported |

**Same contrast with both images low-resolution (symmetric restored minus symmetric direct)**

| Arm | ArcFace | AdaFace | MagFace | CurricularFace |
|---|---|---|---|---|
| Bicubic | +0.016 [+0.011, +0.020] | +0.019 [+0.014, +0.024] | +0.015 [+0.011, +0.019] | +0.009 [+0.004, +0.014] |
| Real-ESRGAN native | -0.131 [-0.143, -0.121] | -0.140 [-0.152, -0.127] | -0.117 [-0.128, -0.107] | -0.148 [-0.162, -0.137] |
| InvSR pre-64 | -0.142 [-0.154, -0.130] | -0.142 [-0.154, -0.130] | -0.128 [-0.139, -0.116] | -0.147 [-0.160, -0.134] |
| TADSR pre-128 / t250 | +0.013 [+0.008, +0.019] | +0.014 [+0.008, +0.020] | +0.016 [+0.010, +0.021] | +0.012 [+0.005, +0.018] |
| OSDFace pre-128 | -0.019 [-0.027, -0.010] | -0.017 [-0.026, -0.009] | -0.007 [-0.017, +0.001] | -0.006 [-0.015, +0.002] |

**Post-hoc: arm minus bicubic, cross condition** (added after seeing that bicubic met the rule; bicubic is the fair non-generative comparator)

| Arm | ArcFace | AdaFace | MagFace | CurricularFace |
|---|---|---|---|---|
| TADSR pre-128 / t250 | +0.001 [-0.002, +0.005] | -0.010 [-0.014, -0.006] | +0.014 [+0.010, +0.018] | +0.006 [+0.001, +0.011] |
| OSDFace pre-128 | -0.049 [-0.059, -0.040] † | -0.049 [-0.059, -0.039] | -0.007 [-0.018, +0.004] | +0.003 [-0.009, +0.015] |
| InvSR pre-64 | -0.162 [-0.173, -0.149] | -0.182 [-0.194, -0.170] | -0.116 [-0.128, -0.104] | -0.140 [-0.153, -0.125] |
| Real-ESRGAN native | -0.179 [-0.192, -0.166] | -0.202 [-0.215, -0.188] | -0.146 [-0.159, -0.134] | -0.165 [-0.178, -0.149] |

**Post-hoc: arm minus bicubic, symmetric condition** (added after seeing that bicubic met the rule; bicubic is the fair non-generative comparator)

| Arm | ArcFace | AdaFace | MagFace | CurricularFace |
|---|---|---|---|---|
| TADSR pre-128 / t250 | -0.002 [-0.006, +0.001] | -0.005 [-0.009, -0.001] | +0.001 [-0.003, +0.004] | +0.003 [-0.001, +0.007] |
| OSDFace pre-128 | -0.034 [-0.042, -0.026] † | -0.036 [-0.044, -0.029] | -0.023 [-0.032, -0.014] | -0.015 [-0.024, -0.007] |
| InvSR pre-64 | -0.158 [-0.169, -0.146] | -0.161 [-0.173, -0.149] | -0.143 [-0.154, -0.132] | -0.156 [-0.169, -0.144] |
| Real-ESRGAN native | -0.147 [-0.158, -0.135] | -0.159 [-0.171, -0.146] | -0.132 [-0.144, -0.122] | -0.157 [-0.170, -0.145] |
