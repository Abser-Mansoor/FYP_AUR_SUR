# Sprint 1 — Restoration Engine Feasibility & Evaluation Harness

**Project:** Identity-Preserving Adaptive Surveillance Restoration & Recognition Pipeline (Approach B)
**Last updated:** 2026-09-30
**Status:** Planning complete, execution not started
**Sprint 1 rule:** *No training.* Inference, evaluation and go/no-go decisions only.

> Legend: **[V]** verified from a primary source during Sprint 1 planning · **[R]** reported in earlier notes, re-check on first use · **[?]** unverified / to be confirmed · **TBD** to be filled in by the team.

---

## 1. Objective

1. Re-evaluate the feasibility, frontier status and problem-fit of the proposed restoration architectures.
2. Build the evaluation harness (including the no-restoration baseline) that every later result depends on.
3. Smoke-test each candidate restorer on QMUL crops and record a go/no-go decision per model.
4. Defer all fine-tuning and the ViT quality gate to the next sprint.

**Core research question:** Does diffusion-based restoration improve surveillance face recognition on real low-quality imagery without harming identity? An honest "no" on real data is a valid, publishable outcome.

---

## 2. Compute inventory

| System | Specs | Notes |
|---|---|---|
| Lab machine (university) | RTX 4000 Ada, **20 GB VRAM [R]**; 32 GB DDR5 system RAM; CPU model unconfirmed | Run `nvidia-smi` and `lscpu`, paste output here. The proposal's own Approach B minimum is 24 GB, so this machine is below spec. |
| Colab ×3 (one per member) | GPU varies per session (commonly 16 GB class) | Sessions are ephemeral. Persist datasets/checkpoints on Drive; use config-driven scripts and fixed seeds. |
| Kaggle kernels (optional) | Varies | Useful for the gate later; dataset licensing may restrict uploads. |

**Action:** record actual VRAM/RAM per system in this table before assigning models.

---

## 3. Decision log

| # | Decision | Status | Evidence / rationale |
|---|---|---|---|
| D1 | **SR: InvSR primary, TADSR optional second arm.** DreamSR and FluxSR excluded. | Decided | DreamSR is built on a Flux DiT + ControlNet and targets ultra-high-resolution, patch-wise inference **[V]**; FluxSR distills FLUX.1-dev **[R]**. Flux-scale backbones are not expected to fit 20 GB / Colab without quantization, and both are designed for far larger inputs than 20×16 crops. *Not benchmarked — excluded on scale and design mismatch, not measured failure.* |
| D2 | **InvSR:** SD-class backbone, 1–5 selectable sampling steps. | Decided | Step count enables a "how much diffusion before identity degrades" ablation (supplementary report on QMUL). |
| D3 | **TADSR** (CVPR 2026, Nankai): one-step SD distillation with a timestep-controlled fidelity/realism trade-off. | Decided | Timestep sweep = identity-preservation ablation. **Name collision:** `LearningHx/TAD-SR` is a *different* paper (code "coming soon"). Use `zty557/TADSR`. |
| D4 | **OSDFace added as face-specific all-in-one arm.** | Decided | CVPR 2025; one-step diffusion on SD 2.1 with a recognition-derived identity loss **[V]**. Runs on aligned face crops, i.e. a *parallel branch*, not a stage in the frame-level chain. **Leakage caveat:** identify which recognizer its identity loss used; weight results from *other* recognizers more heavily. |
| D5 | **ID-CDM excluded.** | Decided | Authors trained the distillation + consistency models on GoPro clean images (300K iterations, 4× 80 GB A800) **[V]**; no code or weights found **[V]**; ~5.8 s per 256×256 image **[V]**; generic natural-scene training, limited face evidence **[V]**; authors state degradation under severe blur+noise and low light **[V]**. Optional: email corresponding author for checkpoints. |
| D6 | **Deblurring: NAFNet or Restormer (working arm). FideDiff blocked.** | Decided / blocked | FideDiff (ICLR 2026, SJTU) is a single-step diffusion deblurrer with kernel estimation, but code/weights are announced, not released **[V]**. Monitor `github.com/xyLiu339/FideDiff`. Record which pretrained checkpoint (GoPro vs RealBlur) is used. |
| D7 | **Low-light: π-Diff has no public repo (none found as of 2026-09-30).** Use a diffusion method with released code: QuadPrior, LightenDiffusion or Diff-Retinex++. Pick one primary + one backup after smoke test. | Decided, pending smoke test | These are 2024-era methods, so the "frontier" claim for this stage is weaker than the proposal states. Check whether each is zero-reference or trained on LOL (a domain-gap factor). |
| D8 | **Synthetic-data policy.** Synthetic degradation is diagnostic only. | Decided | Idiap, IJCB 2026 (arXiv 2608.06580): the degradation setting best on synthetic benchmarks is not best on real native-LR TinyFace; learned SR pipelines did not beat direct feed. *Specific figures (e.g. 57.53 vs 65.12 mAP) come from a secondary summary — verify in the paper.* See §5.4. |
| D9 | **Single restoration per frame initially** (gate flags mutually exclusive). | Decided | Chaining SR + deblur + low-light multiplies ablation combinations and compounds artifacts. Revisit after Sprint 2. |
| D10 | **Geometric-artifact detection (Sarkar et al., "Shadows Don't Lie and Lines Can't Bend", CVPR 2024) deferred.** | Deferred | Cues (shadows, lines, perspective fields) do not exist in 20×16 crops; applicable only to frame-level branch. Identity-drift metrics are the priority check (§5.3). Revisit in FYP-II with classical-processing controls. |
| D11 | **PartialFC is a training strategy, not a recognizer.** Drop from ablation list unless a released checkpoint exists. | Proposed | Avoids a viva question. Needs supervisor agreement (see §9). |
| D12 | **Fine-tuning to QMUL crops deferred.** | Deferred | QMUL has no HR ground truth; fine-tuning would need an identity loss or self-supervision, on **training split only** (use official splits). |

---

## 4. Model & code inventory

Fill in on first download. Log the date and commit so results are reproducible.

| Model | Role | Repo | Weights confirmed | Commit / date | VRAM / s per crop | Go/no-go |
|---|---|---|---|---|---|---|
| InvSR | SR (diffusion, 1–5 steps) | github.com/zsyOAOA/InvSR | [R] | TBD | TBD | TBD |
| TADSR | SR (one-step) | github.com/zty557/TADSR | [R] | TBD | TBD | TBD |
| OSDFace | Face restoration (one-step) | github.com/jkwang28/OSDFace | Official: Drive/OneDrive; unofficial HF mirror exists (prefer official) | TBD | TBD | TBD |
| NAFNet / Restormer | Deblur (feed-forward) | TBD | [?] | TBD | TBD | TBD |
| FideDiff | Deblur (diffusion) | github.com/xyLiu339/FideDiff | **Not released** | — | — | Blocked |
| QuadPrior | Low-light (diffusion) | github.com/daooshee/QuadPrior | [?] | TBD | TBD | TBD |
| LightenDiffusion | Low-light (diffusion) | github.com/JianghaiSCU/LightenDiffusion | [?] | TBD | TBD | TBD |
| Diff-Retinex++ | Low-light (diffusion) | github.com/XunpengYi/Diff-Retinex-Plus | [?] | TBD | TBD | TBD |
| Real-ESRGAN | SR baseline | TBD | [?] | TBD | TBD | TBD |
| Zero-DCE++ | Low-light baseline | TBD | [?] | TBD | TBD | TBD |
| RetinaFace / SFE-DETR | Detection | TBD | SFE-DETR code availability **[?]** | TBD | TBD | TBD |

**Recognizers (measuring instruments, not trained):**

| Model | Backbone | Training data | Checkpoint source | Notes |
|---|---|---|---|---|
| ArcFace | TBD | TBD | TBD | |
| AdaFace | TBD | TBD | TBD | |
| MagFace | TBD | TBD | TBD | |
| CurricularFace | TBD | TBD | TBD | |

The names refer to *loss functions*. For a fair ablation the backbone and training data must match across all four. If they cannot be matched, report the mismatch as a confound.

---

## 5. Datasets & evaluation protocol

### 5.1 Datasets

| Dataset | Role | Notes |
|---|---|---|
| **QMUL-SurvFace** | Headline real surveillance benchmark | Native LR, uncooperative capture; low resolution, motion blur, pose, occlusion, poor illumination. Pre-cropped faces, **no HR ground truth, no full frames**. Confirm download/access route. |
| **QMUL-TinyFace** | Headline native-LR identification | ~20×16 px average faces. Pre-cropped, no ground truth. |
| SCface | Real surveillance probes at measured distances + HR mugshots; possible infrared | Strong lead from jury notes. Small; check license; infrared needs separate handling. |
| DARK-FACE | Low-light *detection* only | Real low-light images with face boxes, **no identity labels**. Cannot test enhancement-vs-identity. Check license. |
| WIDER FACE | Detection stage | |
| LOL / LOL-v2 / SICE, RealBlur / GoPro / HIDE | Module-level validation only | Not faces; no identity metric possible. |
| QMUL-GRID | **Future work** | Person re-ID; shifts scope to full-body multi-camera tracking. |

### 5.2 Protocols (verify against the original papers before coding)

- **SurvFace:** open-set identification with TAR@FAR (and verification), using the official train/test split **[?]**.
- **TinyFace:** closed-set identification, Rank-1 / Rank-20 / mAP **[?]**.
- The proposal applies Rank-1 and TAR@FAR to both datasets. Correct this if the original protocols differ.
- Keep the gallery fixed across all variants and bins. Report bootstrap confidence intervals.

### 5.3 Metrics

- **Identity (real data, headline):** Rank-1, TAR@FAR, mAP (TinyFace), per recognizer.
- **Embedding drift:** cosine similarity between restored and original embeddings.
- **Impostor-score shift:** whether impostor pairs score higher after restoration ("restoration makes strangers look alike").
- **Genuine/impostor separation:** change in the gap between distributions, per restorer and setting (InvSR steps, TADSR timestep).
- **Image metrics (PSNR / SSIM / LPIPS):** only computable on synthetic paired data; always labelled *synthetic-only*.
- **Cost:** VRAM and seconds per crop, measured (the proposal's throughput figures are estimates).
- **Stochasticity:** diffusion models are non-deterministic. Run multiple seeds; report mean ± std.

### 5.4 Synthetic-data policy

- Synthetic degradation is used **only** for controlled diagnostics (e.g. dose-response sweeps) and for the PSNR/SSIM/LPIPS numbers that need ground truth.
- Never use it for headline claims, gate thresholds, or choosing the "best" setting.
- Use degradation families the restorers were *not* trained on (avoid Real-ESRGAN-style pipelines) and report per family.
- Always report synthetic and real results side by side. If rankings flip, that is a finding.

### 5.5 Tiny-crop protocol (record all choices)

- SD-based VAEs downsample 8×, so 20×16 crops cannot be processed at native size. **Pre-upsampling size is a hyperparameter**; log it per experiment.
- Landmark alignment will likely fail on such crops. Decide the resize-only convention up front and apply it identically to every variant.
- Every restorer sees the same input crops and the same recognizer preprocessing.

### 5.6 Stratifying real data by measured degradation

**Low-light (luminance):** compute mean luminance per crop; split into dark / normal bins; compare identity metrics with and without enhancement per bin.

**Low-sharpness (Laplacian variance):**

1. Resize every crop to the same size (e.g. 112×112), convert to grayscale.
2. Compute `cv2.Laplacian(gray, cv2.CV_64F).var()`.
3. Split by **quantiles of the observed distribution** (not fixed thresholds from tutorials).
4. Evaluate per bin with bootstrap CIs.

**Caveats:**

- The score measures lack of detail, not blur specifically. Call it a *low-sharpness* subset.
- Confounders: noise inflates the score; low contrast, darkness and low-texture faces deflate it. Stratify by native crop size and luminance too.
- Luminance correlates with skin tone; sharpness with texture. Prefer **paired per-sample comparisons** (before vs after restoration on the same crop) over absolute bin accuracy, and report bin sizes.
- Validate the split: inspect ~50 random crops from each end; cross-check with a second measure (FFT high-frequency energy or feature norm from MagFace/AdaFace). Identity accuracy should fall from sharp to low-sharpness bins; if not, the score is not capturing recognition-relevant degradation.
- Check the dark bin actually contains enough samples before planning around it.

---

## 6. Tasks

**Proposed ownership (from proposal Table 1; team to confirm):** Umar → Task 0; Raghib → Tasks 1–3; Abser → Task 4, config system, pipeline scaffolding. Machine assignment TBD.

### Task 0 — Evaluation harness *(prerequisite for everything)*

- Dataset loaders, official splits, gallery/probe construction.
- Recognizer checkpoints loaded through one common interface (see §4).
- Common preprocessing per §5.5; metric code per §5.2–5.3; bootstrap CIs; seed control; results logging (W&B or TensorBoard).
- **Direct-feed baseline on QMUL-SurvFace and TinyFace for all four recognizers.** This is the reference for every later comparison.
- **Done when:** baseline numbers reproduce plausibly against published figures for at least one recognizer, and one command re-runs the full baseline.

### Task 1 — SR and face-restoration arms

- Load InvSR (sweep 1–5 steps), TADSR (timestep sweep) and OSDFace; run all on QMUL crops. No fine-tuning.
- Produce a comparison report: identity metrics, drift, impostor shift, VRAM, seconds per crop, failure cases with images.
- **Baselines:** direct feed, bicubic, Real-ESRGAN; optionally CodeFormer/GFPGAN.
- **Done when:** each model has a recorded go/no-go and per-recognizer numbers with CIs.

### Task 2 — Deblurring and the SR-vs-deblur question

- Run NAFNet/Restormer on QMUL crops **without SR** to test whether deblurring works on small crops.
- Test **all orderings**: direct · SR only · deblur only · deblur→SR · SR→deblur · gated (later).
- If SR alone matches deblur→SR on identity metrics, the deblurring stage is unnecessary at this resolution.
- **Baselines:** direct feed, unsharp mask, Real-ESRGAN (implicit deblurring).
- Analyze primarily on the low-sharpness bins from §5.6.

### Task 3 — Low-light enhancement

- Isolate the dark bin (§5.6); compare with and without enhancement, per bin, paired per sample.
- Smoke-test QuadPrior, LightenDiffusion and Diff-Retinex++; select one primary and one backup.
- **Baselines:** direct feed, gamma, CLAHE (on luminance), Zero-DCE++; optionally Retinexformer **[?]**.
- If the diffusion method cannot beat CLAHE on identity metrics, report that as a result.
- DARK-FACE is an *optional* extra for the detection stage only.

### Task 4 — Stratification

- Implement the luminance and Laplacian-variance splits with the validation checks in §5.6.
- **Baselines for this task:** full unstratified set; a size-matched random subset; agreement with a second sharpness measure.

### Later (not Sprint 1): gate baselines

Always restore · never restore · random gate at the same restore rate · Laplacian-threshold heuristic · learned ViT gate.

---

## 7. Experiment matrix (Sprint 1)

| Stage | Variants | Data | Primary metrics |
|---|---|---|---|
| Reference | direct feed | SurvFace, TinyFace | Rank-1, TAR@FAR, mAP |
| SR | bicubic, Real-ESRGAN, InvSR (1–5 steps), TADSR (timesteps), OSDFace | SurvFace, TinyFace | + drift, impostor shift, cost |
| Deblur | unsharp, NAFNet/Restormer, orderings with SR | low-sharpness bins | + paired deltas |
| Low-light | gamma, CLAHE, Zero-DCE++, chosen diffusion method | dark bin | + paired deltas |
| Synthetic diagnostics | dose-response sweeps (held-out degradation families) | paired synthetic set | PSNR/SSIM/LPIPS (synthetic-only), drift |

---

## 8. Risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | QMUL dataset access delayed | Request today; verify integrity and splits. |
| R2 | SD-based models fail on 20×16 crops | Treat pre-upsampling size as a swept hyperparameter; a documented failure is a result. |
| R3 | Weights unavailable for a candidate | Fallback arms already chosen (§3); log status in §4. |
| R4 | Colab session limits / lost state | Persist to Drive; config-driven runs; fixed seeds. |
| R5 | OSDFace identity-loss leakage inflates one recognizer | Identify the loss recognizer; emphasise the other recognizers. |
| R6 | SFE-DETR code unverified | Check early; RetinaFace is the fallback. |
| R7 | Diffusion stochasticity hides small effects | Multi-seed, CIs. |
| R8 | Confounded bins (skin tone, texture, noise) | Paired comparisons, bin sizes, second sharpness measure. |
| R9 | Dark bin too small to be informative | Check counts first; use SCface or a larger quantile. |
| R10 | Recognizer checkpoints not comparable | Match backbone/data or document as confound. |
| R11 | Restoration does not beat direct feed on real data | Expected possibility; frame the study as "when does it help or harm". |
| R12 | Dropped proposal modules not approved | Get supervisor sign-off (§9). |

---

## 9. Deviations from the proposal (needs supervisor sign-off)

- DreamSR, ID-CDM and π-Diff removed from the active pipeline (reasons in §3).
- FluxSR and TADSR treatment changed (FluxSR excluded, TADSR added as optional).
- OSDFace added as a face-specific arm.
- PartialFC dropped or reframed.
- "Each module is independently trained" should be reworded: only the quality gate is trained; the rest is pretrained inference.
- Baselines (Real-ESRGAN, Zero-DCE++, direct feed) made explicit in Approach B.

---

## 10. Proposal citation corrections

| Ref | Issue |
|---|---|
| [1] DreamSR | First author is Qingji Dong (ByteDance), not "Y. Dong". |
| [2] ID-CDM | Authors are Zhaohan Wang, Chengjun Chen, Chenggang Dai; *Complex & Intelligent Systems* vol. 12, 2026 (online Dec 2025). |
| [3] π-Diff | First author is Min Wang (Wang, Wang, Zhou, Wang); CVPR Workshops 2026, pp. 5105–5115. |
| [20] TADSR | Now a CVPR 2026 paper (Zhang et al.), not only arXiv. |
| [6], [7] QMUL datasets | Titles/authors/years in the proposal do not match what was seen (SurvFace: arXiv 1804.09691, 2018). **Verify both against the original papers.** |
| New | Add OSDFace (CVPR 2025), FideDiff (ICLR 2026), Idiap IJCB 2026 (arXiv 2608.06580), Sarkar et al. (CVPR 2024). |
| Prior README | Corrected: AdaFace is CVPR 2022, YOLOv11 has no peer-reviewed paper, NTIRE event deblurring needs event-camera hardware. |

---

## 11. Jury suggestions

1. **QMUL-GRID:** person re-identification dataset. Relevant only if scope expands to full-body multi-camera tracking. *Future work.*
2. **SCface:** real surveillance probes at measured distances with HR mugshots and infrared. Directly addresses distance-dependent degradation with real (non-synthetic) data and an identity reference. **Recommended for Sprint 2.** Verify size, license and infrared handling.

---

## 12. Exit criteria for Sprint 1

- [ ] Task 0 complete: direct-feed baseline on both QMUL datasets for all four recognizers, reproducible from one command.
- [ ] Each of InvSR, TADSR, OSDFace has a recorded go/no-go with measured VRAM and time per crop.
- [ ] Deblur experiment answers: does deblurring work on small crops, and does SR alone suffice?
- [ ] Dark bin sized; one primary and one backup low-light method selected.
- [ ] §4 tables filled (weights, commits, dates).
- [ ] Supervisor briefed on §9 deviations.

---

## 13. Open items for the next sprint

- **ViT quality gate:** the only component to be trained. Labels should reflect identity utility (does restoration help this crop?), not perceptual quality; hold out a validation split; do not tune thresholds on the QMUL test sets or on synthetic data. Baseline signals: MagFace/AdaFace feature norm, Laplacian variance.
- Optional identity-aware fine-tuning of InvSR (LoRA) on training-split crops.
- Deferred geometric-artifact study (D10).
