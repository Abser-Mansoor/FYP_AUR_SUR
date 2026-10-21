# Identity-Preserving Adaptive Surveillance Restoration & Recognition Pipeline
### Literature-Verified Stage-by-Stage Design, Two Candidate Approaches, and System Requirements

*Verification note: every paper below was re-checked against its arXiv / CVPR / ICCV / ICML / NTIRE / AIM listing on 22 July 2026. A few items from the original draft could not be confirmed as distinct formal publications and are flagged explicitly rather than cited as if verified.*

---

## 1. Stage-by-Stage Table

| Stage | Task | Recommended Model(s) | Key 2025–2026 Paper(s) (verified) | Why Chosen | Advantages | Disadvantages | Relevant Dataset(s) | Implementation Difficulty |
|---|---|---|---|---|---|---|---|---|
| 1 | Frame quality assessment | CNN/ViT-based blur, noise, exposure, and compression estimators | No single dominant 2025–2026 "surveillance QA" paper; this stage is typically assembled from classical + learned no-reference IQA (e.g., Laplacian variance for blur, NIQE/BRISQUE-style learned successors) | Prevents unnecessary restoration, which is the single biggest source of identity distortion in a naive pipeline | Cheap, fast, gates expensive stages | No single agreed-upon SOTA model; must be assembled/validated by the student | Custom-labelled subset of your CCTV data; RealBlur/GoPro for blur-score calibration | Medium — mostly engineering, some threshold-tuning |
| 2A | Super-resolution (SR) — production baseline | Real-ESRGAN | Wang et al., *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*, ICCVW 2021 | Mature, fast (GAN, single forward pass), open-source, widely used on CCTV-style low-res footage | Fast inference, small VRAM footprint, easy to fine-tune | Can hallucinate texture; weaker fidelity than diffusion SR on faces | RealSR, DIV2K+degradation pipeline, your own CCTV crops | Low |
| 2A | Super-resolution — frontier comparator | DreamSR (diffusion transformer, receptive-field enhanced) | Dong et al., **DreamSR: Towards Ultra-High-Resolution Image Super-Resolution via a Receptive-Field Enhanced Diffusion Transformer**, CVPR 2026 (arXiv:2605.15682) | Confirmed CVPR 2026 paper; addresses the specific failure mode of patch-wise diffusion SR (local over-generation) that is most damaging to small, cropped face regions | Sharper, more faithful local texture than earlier diffusion SR; explicitly targets the "patch hallucination" problem | Multi-step diffusion inference, heavier VRAM, slower | Its evaluation benchmarks + your CCTV faces | High |
| 2A | Super-resolution — efficient one-step alternative | FluxSR / TADSR / InvSR | FluxSR: Li et al., *One Diffusion Step to Real-World Super-Resolution via Flow Trajectory Distillation*, ICML 2025; TADSR: *Time-Aware One-Step Diffusion Network for Real-World Image Super-Resolution*, arXiv:2508.16557 (2025); InvSR: Yue et al., *Arbitrary-steps Image Super-resolution via Diffusion Inversion*, CVPR 2025 | All three confirmed as genuine 2025 one-step/few-step diffusion distillation papers — the realistic middle ground between Real-ESRGAN's speed and DreamSR's quality | 1–5 sampling steps instead of 20–50; near-diffusion quality at near-GAN speed | Younger codebases, less battle-tested on CCTV-specific degradations | Same as above; InvSR reports on RealSet80/ImageNet-Test | Medium-High |
| 2B | Blur detection (gate) | Laplacian variance / FFT spectral energy / lightweight CNN classifier | No dedicated 2025–2026 paper — this is a well-established classical technique, correctly described as such | Sub-millisecond, no training required for the classical version | Trivial to implement, interpretable threshold | Classical variance methods are noise-sensitive; CNN variant needs labelled data | Any blur/sharp paired set (GoPro, RealBlur) for threshold calibration | Low |
| 2C | Deblurring | ID-CDM (zero-shot consistency-distillation deblurring) | Wang, Chen & Dai, **Zero-Shot Realistic Image Deblurring with Consistency Model (ID-CDM)**, *Complex & Intelligent Systems*, 2025 | Confirmed 2025 publication; explicitly zero-shot (no deblurring-specific training needed) and evaluated on a real-world blur set, which matters because synthetic-Gaussian-trained deblurrers generalize poorly to CCTV motion blur | Avoids paired-data training entirely; targets real (not synthetic) blur; faster than full multi-step diffusion via consistency distillation | Still diffusion-based → slower than a CNN deblurrer; identity-preservation on faces not specifically validated in the paper | HIDE (synthetic), RealBlur (real-world) — as used in the paper itself | High |
| 2C | Deblurring — efficiency benchmark context | Efficient/event-based deblurring challenges | AIM 2025: Feijoo et al., **Efficient Real-World Deblurring using Single Images: AIM 2025 Challenge Report**, ICCVW 2025 (arXiv:2510.12788); NTIRE 2026: **The Second Challenge on Event-Based Image Deblurring at NTIRE 2026** | Both confirmed as real, recent challenge-report papers. AIM 2025 caps models at <5M parameters / <200 GMACs — directly relevant if you need edge/real-time deployment. The NTIRE 2026 challenge needs event-camera input, which ordinary CCTV does not have, so it is context/benchmarking material rather than a directly deployable stage | Gives you a credible "efficient deblurring" citation and parameter/GMAC budget to benchmark against | Event-based NTIRE work is **not applicable unless your camera has an event sensor** — flag this in your literature review so an examiner doesn't think you deployed it | RSBlur (AIM 2025), HighREV (NTIRE event challenge) | Medium (AIM baseline), N/A for event-based unless you have event-camera hardware |
| 2D | Low-light enhancement | π-Diff (recommended frontier) / Zero-DCE++ (baseline) | Wang et al., **π-Diff: Physically-Inspired Low-Light Image Enhancement with Structure-Preserving Diffusion Priors**, CVPR 2026 Workshop (LoViF); Li et al., *Zero-DCE++*, TPAMI 2021 | π-Diff is a training-free diffusion framework designed specifically to avoid hallucinated content, color distortion, structural degradation, and over-enhancement in extremely dark scenes. It aligns closely with the identity-preservation goals of the overall surveillance pipeline. Zero-DCE++ remains the lightweight practical baseline. | π-Diff: strong structure preservation, no training required, diffusion-quality enhancement. Zero-DCE++: extremely fast and lightweight. | π-Diff is slower than Zero-DCE++; workshop publication rather than main-conference paper. | LOL, LOL-v2, SICE; NTIRE 2026 Efficient Low-Light Enhancement Challenge | Medium |
| 3 | Face detection | RetinaFace (baseline) / SFE-DETR (frontier) | RetinaFace: Deng et al., CVPR 2020; Yang, Jiang & Song, **SFE-DETR: An Enhanced Transformer-Based Face Detector for Small Target Faces in Open Complex Scenes**, Sensors 2025/2026 | SFE-DETR targets small, degraded, dense, and partially occluded faces in open complex scenes. These characteristics closely match CCTV surveillance environments and provide a modern transformer-based comparison against RetinaFace. | Better small-face performance, transformer architecture, reduced parameter count, stronger degraded-image robustness. | Sensors venue rather than CVPR/ICCV-tier conference; newer ecosystem and fewer implementations. | WIDER FACE | Medium |
| 4 | Face alignment | 5-point / 68-point landmark-based similarity transform to 112×112 | Standard ArcFace-family preprocessing convention, not a separate paper | Well-defined, deterministic, and the de facto input contract for ArcFace/AdaFace-family recognizers | Nearly free computationally; improves recognition more than adding another restoration stage | Fails ungracefully on extreme pose/occlusion | Same as detection stage | Low |
| 5 | Face recognition (embedding) | ArcFace / AdaFace / MagFace / CurricularFace | ArcFace: CVPR 2019; AdaFace: CVPR 2022; MagFace: CVPR 2021; CurricularFace: CVPR 2020 | Provides a complete comparison between classical margin-based recognition, quality-adaptive recognition, quality-aware representation learning, and curriculum-based hard-sample optimization. | Strong ablation study covering the major modern face-recognition paradigms. | Requires additional experimentation and evaluation time. | MS1MV3, Glint360K, LFW, CFP-FP, AgeDB, IJB-B, IJB-C, QMUL-SurvFace, QMUL-TinyFace | Medium-High |
| 6 | Identity tracking across frames | ByteTrack or BoT-SORT (associate face/person boxes over time) | ByteTrack: Zhang et al., ECCV 2022; BoT-SORT: Aharon et al., arXiv 2022 | Both are the standard, still-dominant multi-object trackers layered on top of a detector; not 2025–2026 but appropriately cited as established infrastructure | Reduces redundant recognition calls; gives persistent IDs across frames | Track fragmentation under long occlusion; not identity-aware by itself (needs the recognizer's embedding to actually confirm identity, not just track continuity) | MOT17, MOT20 (generic benchmark); your own CCTV footage for qualitative evaluation | Medium |
| 7 | Database matching | Cosine similarity / ANN search (FAISS, HNSW) over embeddings | Not paper-specific; standard vector-search engineering | Sub-linear search at scale | Simple, well-documented libraries | Threshold selection (FAR/FRR trade-off) needs careful evaluation, not just "similarity > 0.5" | Your enrolled-identity gallery + QMUL-SurvFace as an open-set stress test | Low-Medium |
| — | Non-face object detection/tracking (traffic) | YOLOv11 / RT-DETR + ByteTrack/BoT-SORT | YOLOv11: Ultralytics, 2024 release (not a peer-reviewed paper — cite the Ultralytics documentation/GitHub, not a nonexistent "YOLOv11 paper"); RT-DETR: Zhao et al., CVPR 2024 | Independent branch from the face pipeline, as in your original design | Real-time, mature ecosystem | YOLOv11 has no formal peer-reviewed paper as of writing — this is a citation-accuracy issue you should flag explicitly to your supervisor | BDD100K, UA-DETRAC | Low |

---

## 2. Paper Summaries

**DreamSR (CVPR 2026).** <cite index="3-1">The paper targets a specific failure mode in patch-wise diffusion-based super-resolution: when large images are super-resolved patch by patch, the global prompt from the low-resolution image is misaligned with the incomplete semantic content of each local patch, causing over-generation and loss of fine texture.</cite> The authors introduce a dual-branch control mechanism and a receptive-field-enhancement training strategy so the model captures more local context per patch, improving fidelity on ultra-high-resolution outputs. For an FYP, the relevant takeaway is that DreamSR was explicitly designed to fix the kind of local hallucination that would be most damaging to a small cropped face region — making it a legitimate frontier comparator, not just a "bigger diffusion model."

**FluxSR (ICML 2025).** <cite index="11-1">This work proposes a one-step diffusion technique for real-world super-resolution based on flow matching, using the FLUX.1-dev model as both teacher and base model to distill a multi-step flow-matching process into a single sampling step.</cite> It adds a perceptual loss and an attention-diversification regularizer to suppress the transformer artifacts that typically appear when diffusion models are compressed to one step. It is a strong "frontier-but-fast" candidate if DreamSR's inference cost is a problem for your GPU budget.

**TADSR (arXiv, 2025).** <cite index="14-1">TADSR introduces a variable timestep into the student model along with a time-aware VAE encoder, allowing the one-step model to draw on the generative priors of a diffusion backbone at different noise levels, with a distillation strategy that yields a controllable trade-off between fidelity and realism.</cite> This tunability is useful for an FYP ablation: you can dial the model toward "fidelity" mode when identity preservation matters more than aesthetic sharpness.

**InvSR (CVPR 2025).** <cite index="24-1">InvSR reformulates super-resolution as diffusion inversion: a noise-predictor network estimates an optimal noise map that initializes an intermediate state of a pretrained diffusion model, from which sampling proceeds, and this design supports an arbitrary number of sampling steps (one to five) chosen at inference time.</cite> It is the most "flexible" of the three frontier SR options and pairs naturally with the adaptive/conditional philosophy of your proposed pipeline (skip restoration when not needed, tune step count when it is).

**ID-CDM (Complex & Intelligent Systems, 2025).** <cite index="29-1">The paper addresses two practical limits of diffusion-based deblurring: existing methods depend on paired blurry/clean training data whose synthetic blur does not match real-world blur, and multi-step diffusion sampling is computationally expensive and prone to hallucinating extra detail.</cite> <cite index="29-1">Its proposed Image Deblurring with Consistency Distillation Models (ID-CDM) restores real-scene blurred images without task-specific training, using a diffusion-coefficient-dominated score model distilled into a consistency model to cut sampling time while limiting artifact generation.</cite> This "zero-shot, real-blur-focused" framing is exactly what a CCTV deblurring stage needs, since your footage will not resemble the synthetic Gaussian-blur datasets most classical deblurrers were trained on.

**AIM 2025 Efficient Real-World Deblurring Challenge (ICCVW 2025).** <cite index="43-1">This challenge report reviews submissions to an efficient real-world deblurring competition built on a new test set derived from the RSBlur dataset, captured with a double-camera system, under strict efficiency constraints of under 5 million parameters and under 200 GMACs.</cite> <cite index="43-1">Seventy-one participants registered and four teams submitted valid solutions, with the top approach reaching roughly 31.1 dB PSNR.</cite> Use this as your citation for "deployable, edge-friendly deblurring," and as a parameter/compute budget target if you want to argue your pipeline could run in real time on modest hardware.

**NTIRE 2026 Second Challenge on Event-Based Image Deblurring.** <cite index="38-1">This is the second edition of an NTIRE workshop challenge that fuses event-camera data with conventional imagery to recover sharp frames from severe motion blur, following on from a first, well-attended 2025 edition.</cite> Important caveat for your FYP: this line of work assumes an **event camera** as an additional sensor. Standard CCTV footage has no event stream, so this is background/context for your literature review, not a directly deployable component unless your hardware setup includes an event sensor.

**AdaFace (CVPR 2022 — correcting the "2025–2026" framing in the original draft).** <cite index="50-1">AdaFace targets face recognition on low-quality datasets by introducing an adaptive margin function that approximates image quality using feature norms, so that the loss function's emphasis on hard, misclassified samples is adjusted according to how recognizable each sample actually is.</cite> <cite index="51-1">On the mixed-quality IJB-B and IJB-C benchmarks it reduced errors by roughly 11% and 9% respectively relative to the next-best prior method, while remaining competitive with prior state of the art on clean, high-quality benchmarks.</cite> This is not a 2025–2026 paper — it is a 2022 CVPR paper that remains a standard, strong baseline. It is the single most directly relevant recognizer for your project because its entire design goal (don't over-trust degraded low-quality faces) matches the surveillance use case.

**QMUL-SurvFace / QMUL-TinyFace (established surveillance-specific benchmarks, not 2025–2026 but essential and frequently overlooked).** <cite index="67-1">QMUL-SurvFace is built from real, uncooperative surveillance footage rather than artificially down-sampled high-resolution images, containing 463,507 face images across 15,573 identities collected from real surveillance scenes, and demonstrates that recognition accuracy drops sharply compared to standard celebrity-photo benchmarks — for example one baseline model dropped from roughly 65% Rank-1 on MegaFace to about 30% on QMUL-SurvFace.</cite> <cite index="65-1">QMUL-TinyFace complements this with 169,403 native low-resolution face images across 5,139 identities, averaging only about 20×16 pixels per face, specifically for large-gallery 1:N identification testing.</cite> These two datasets are the most important addition you should make relative to the original proposal — they let you report recognition accuracy on data that is actually representative of CCTV, not just synthetic-degradation benchmarks like RealSR/RealBlur.

---

## 3. Classical CNN/GAN vs. Diffusion/Foundation-Model Comparison

| Dimension | Classical CNN/GAN (Real-ESRGAN, RetinaFace, ArcFace, ByteTrack) | Diffusion/Foundation-Model (DreamSR, FluxSR/TADSR/InvSR, ID-CDM) |
|---|---|---|
| Inference speed | Milliseconds per frame, single forward pass | Multi-step: seconds per frame; one-step distilled variants: tens–hundreds of ms |
| VRAM | 2–6 GB typical | 8–24 GB depending on resolution and step count |
| Texture/detail quality | Can look "plasticky" or over-smoothed; GANs can hallucinate implausible detail | Generally higher perceptual quality, better texture realism |
| Identity preservation | Predictable, but limited detail recovery on very low-res faces | Risk of *inventing* facial detail not present in the source — a serious concern for evidentiary/forensic use, must be evaluated with downstream recognition accuracy, not just PSNR/SSIM |
| Maturity / reproducibility | Very mature, well-documented, easy for an FYP timeline | Younger codebases (2025–2026), less tooling, more debugging overhead |
| Best FYP fit | Practical/reproducible pipeline (Approach A) | Frontier/publication-level pipeline (Approach B) |

---

## 4. Dataset Table

| Purpose | Dataset(s) | Notes |
|---|---|---|
| Super-resolution training/eval | RealSR, DIV2K+degradation, RSBlur (also usable for SR-adjacent eval) | Standard SR benchmarks; none are surveillance-native |
| Deblurring | GoPro (synthetic), HIDE (synthetic), RealBlur (real-world) | ID-CDM and most deblurring papers evaluate on this pair |
| Low-light enhancement | LOL, LOL-v2, SICE | Standard low-light benchmarks |
| Face detection | WIDER FACE | Standard, includes small/occluded faces relevant to CCTV |
| Face recognition — clean | LFW, CFP-FP, AgeDB, CALFW | High-quality, near-frontal; use only as a sanity check, not your headline result |
| Face recognition — mixed quality | IJB-B, IJB-C | Where AdaFace's advantage is demonstrated |
| **Face recognition — surveillance-native (recommended headline eval)** | **QMUL-SurvFace, QMUL-TinyFace** | Real uncooperative surveillance imagery; this is what should anchor your "identity-aware evaluation" novelty claim |
| Object detection (traffic) | BDD100K, UA-DETRAC | Independent of the face branch |
| Multi-object tracking | MOT17, MOT20 | Standard tracking benchmarks for ByteTrack/BoT-SORT validation |

---

## Face Recognition Ablation Models

### MagFace (CVPR 2021)

MagFace introduces a quality-aware face representation where the feature magnitude itself acts as a quality indicator. Higher-quality images naturally produce embeddings with larger magnitudes, enabling simultaneous recognition and quality assessment.

Relevance:
- Most conceptually similar comparator to AdaFace.
- Strong candidate for surveillance-quality imagery.
- Provides direct comparison between quality-aware representation learning and quality-adaptive margin learning.

### PartialFC (CVPR 2022)

PartialFC is primarily a large-scale efficient training strategy rather than a new recognition loss function.

Relevance:
- Efficient training baseline.
- Useful if retraining recognition models on MS1MV3 or Glint360K.
- Included primarily for scalability comparisons.

---

## 5. Approach A — Practical FYP Pipeline (Single GPU, Reproducible)

```
CCTV Frame
   │
   ▼
Frame Quality Assessment (Laplacian variance + simple CNN classifier)
   │
   ▼
Conditional Restoration
   ├── Real-ESRGAN (SR, always fast enough to run)
   ├── Skip/Run gate → ID-CDM-lite or classical deblur (only if blur score high)
   └── Zero-DCE++ (only if low-light flag set)
   │
   ▼
RetinaFace (detection + 5-point landmarks)
   │
   ▼
Similarity-transform alignment → 112×112
   │
   ▼
AdaFace embedding (512-d)
   │
   ▼
Cosine similarity → FAISS gallery search
   │
   ▼
ByteTrack (persistent ID across frames)
```

**Rationale:** every component is mature, has a public PyTorch implementation, and can run comfortably on a single consumer/workstation GPU within a semester timeline.

**System requirements (Approach A):**

| Resource | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA RTX 3060 (12 GB VRAM) | RTX 4070/4080 (12–16 GB VRAM) |
| CPU | 6-core modern CPU | 8+ core |
| RAM | 16 GB | 32 GB |
| Storage | 100 GB SSD (datasets + checkpoints) | 500 GB NVMe SSD |
| CUDA/cuDNN | CUDA 11.8+ | CUDA 12.x |
| Framework | PyTorch 2.x | PyTorch 2.3+ with torch.compile |
| Expected throughput | 8–15 FPS end-to-end at 720p with SR+detection+recognition | 20–30 FPS with TensorRT-optimized RetinaFace/AdaFace |
| Cloud fallback | Single T4/L4 instance (e.g., Colab Pro, cloud GPU rental) | Single A10/RTX A4000 instance |

---

## 6. Approach B — Frontier / Publication-Level Pipeline

```
CCTV Frame
   │
   ▼
ViT-Based Frame Quality Assessment
   │
   ▼
Conditional Restoration
   ├── DreamSR / InvSR (Diffusion SR)
   ├── ID-CDM (Zero-Shot Deblurring)
   └── π-Diff (Low-Light Enhancement)
   │
   ▼
Face Detection
   ├── RetinaFace
   └── SFE-DETR
   │
   ▼
Face Alignment
   │
   ▼
Face Recognition Ablation
   ├── ArcFace
   ├── AdaFace
   ├── MagFace
   ├── CurricularFace
   └── PartialFC
   │
   ▼
FAISS / HNSW Search
   │
   ▼
BoT-SORT Tracking
   │
   ▼
Identity-Aware Evaluation
      ├── PSNR
      ├── SSIM
      ├── LPIPS
      ├── Rank-1 Accuracy
      ├── TAR@FAR
      └── QMUL-SurvFace / QMUL-TinyFace
```

**Rationale:** this is the version worth writing up as a genuine research contribution — the core novelty is the identity-aware, task-driven evaluation protocol (measuring recognition accuracy pre/post restoration, not just image-quality metrics), combined with a head-to-head of modular vs. diffusion-based restoration under an adaptive/conditional gate.

**System requirements (Approach B):**

| Resource | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA RTX 4090 (24 GB VRAM) | A100 40GB / A6000 48GB (university HPC cluster access strongly recommended) |
| CPU | 8-core | 16-core |
| RAM | 32 GB | 64 GB |
| Storage | 500 GB NVMe SSD | 1–2 TB NVMe SSD (diffusion checkpoints alone can be 5–15 GB each) |
| CUDA/cuDNN | CUDA 12.x | CUDA 12.4+ |
| Framework | PyTorch 2.3+, diffusers, xformers/flash-attention | Same, plus TensorRT or torch.compile for the recognition/detection stages to keep the pipeline usable |
| Expected throughput | 1–4 FPS with full multi-step diffusion SR | 5–10 FPS with one-step distilled variants (FluxSR/TADSR/InvSR at 1 step) on frames flagged as needing restoration |
| Cloud fallback | Single A100 40GB instance | Multi-GPU (2× A100) if you also want to run ablations in parallel |

---

## 7. Overall Comparison

| Criterion | Approach A (Practical) | Approach B (Frontier) |
|---|---|---|
| Accuracy (image quality) | Good | Better (diffusion detail, but risk of hallucination) |
| Accuracy (downstream recognition, degraded input) | Good, especially with AdaFace | Potentially better *if* restoration doesn't distort identity — this must be measured, not assumed |
| Runtime | Real-time-adjacent on a single consumer GPU | Not real-time without one-step distillation; needs a capable GPU |
| VRAM | 12 GB workable | 24 GB+ strongly preferred |
| Engineering complexity | Low-Medium — all components have mature repos | High — diffusion codebases from 2025–2026 are less stable, more dependency conflicts, more debugging |
| Publication/thesis strength | Solid, safe, demonstrable FYP | Stronger novelty claim, better fit for a paper/poster submission, higher risk of running out of time |
| Expected FYP difficulty | Achievable comfortably in one semester | Ambitious for a single semester without prior diffusion-model experience; realistic for a two-semester or thesis-extension timeline |

---

## 8. Research Question:

Can modern diffusion-based restoration methods
(DreamSR, InvSR, ID-CDM, π-Diff)
improve surveillance face-recognition accuracy on
real low-quality CCTV imagery without introducing
identity-altering hallucinations?

## Novelty:

Rather than evaluating restoration solely using
PSNR, SSIM, and LPIPS, the proposed system evaluates
restoration through downstream identity preservation
using ArcFace, AdaFace, MagFace, and CurricularFace
on surveillance-native benchmarks
(QMUL-SurvFace and QMUL-TinyFace).

This creates a task-aware evaluation framework
for surveillance restoration that directly measures
whether visual enhancement helps or harms
recognition performance.

---

## 9. Final Recommendation

For a standard one-semester FAST/NUCES FYP timeline, **build Approach A first as your working system**, then implement **one frontier SR model from Approach B (InvSR is the best single choice — it is CVPR 2025, has a public, well-documented repo, and its variable-step design lets you directly ablate "how much diffusion is enough" for identity preservation)** as a comparison arm rather than replacing the whole pipeline.

This gives you:
1. A complete, defensible, reproducible baseline system (Approach A) as your safety net.
2. One credible frontier comparison (Approach A's Real-ESRGAN vs. InvSR, evaluated with **both image-quality metrics and AdaFace recognition accuracy on QMUL-SurvFace/TinyFace**) as your actual research contribution.
3. A citation-accurate literature review — several items from the original proposal (AdaFace's date, "YOLOv11-face" as if it were a paper, the NTIRE event-deblurring line requiring event-camera hardware) needed correction, and getting this right will matter to your FYP evaluators.

If a two-semester or thesis-track timeline is available, extend Approach B to include the full DreamSR vs. ID-CDM vs. modular pipeline comparison as originally scoped — that is a genuinely publishable-scope study, not just an FYP demo.
