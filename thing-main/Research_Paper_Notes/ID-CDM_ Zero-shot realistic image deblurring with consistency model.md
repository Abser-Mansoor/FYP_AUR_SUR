# ID-CDM: Zero-shot realistic image deblurring with consistency model

**Authors:** Zhaohan Wang, Chengjun Chen, Chenggang Dai  
**Venue:** Published online: 2 December 2025

---

## General Overview

Image deblurring in real-world scenes is challenging because blur is caused by multiple factors (camera shake, out-of-focus, fast-moving objects). Current diffusion-based deblurring methods require paired blurry-clear datasets for training, which is problematic because we can't precisely model real-world blur causes. Additionally, diffusion models are computationally expensive and can generate distorted images during deblurring.

ID-CDM solves this by using a **consistency distillation model** that works under **zero-shot conditions** (trained only on clear images, never on blurred images). The key innovations are:
- A **diffusion coefficient-dominated score model (DCSM)** that removes the drift term from diffusion equations, reducing computational complexity
- **Integrating DCSM into a consistency model** to enable fast, high-quality deblurring
- Using **invertible linear transformations** to compress images into latent space, further reducing computation

---

## The problem it's solving

**Traditional methods** rely on estimating a blur kernel — but when the degraded model can't describe real data or estimation is poor, results are unsatisfactory.

**Deep learning methods** (CNNs, Transformers) can't effectively process high-frequency details in complex real-world blur and require paired datasets for training.

**Diffusion models** have three major problems:
1. **Computational cost:** 10-2000 iterative denoising steps per image
2. **Slow inference:** Impractical for real-time applications
3. **Artifact generation:** Creates additional artifacts that don't exist in the clear image

**Consistency models** reduce sampling time but often fail to recover fine details when directly applied to image restoration.

---

## The core idea — three innovations

### 1. Diffusion Coefficient-Dominated Score Model (DCSM)

**Standard diffusion models** have a forward process: `dx = f(x,t)dt + g(t)dw`

The drift term `f(x,t)` acts as a low-frequency bias without contributing to high-frequency detail recovery. Since deep networks can absorb such shifts during training, the drift term becomes redundant.

**DCSM's key simplification:** Set `f(x,t) ≈ 0` and `g(t) = ξ^t`

This removes the drift term entirely, simplifies the network architecture, and reduces computational complexity. The absence of the drift term actually accelerates training and improves stability.

### 2. Integration with Consistency Model

The **consistency model** maps any point at any time step directly to the starting point of the trajectory.

**How ID-CDM uses this:**
- DCSM provides accurate score estimates at each time step
- The consistency model learns to directly map noisy images to clean images
- While standard diffusion models need hundreds of steps, the consistency model enables **single-step generation** or **few-step iterative refinement** (10-60 steps)
- **40% acceleration** compared to EDM (from 9.6s to 5.8s per image)

### 3. Zero-shot Processing with Latent Space Compression

**Zero-shot aspect:** ID-CDM is trained ONLY on clear images. During inference, the blurred image is used as a guide/reference while preserving known information.

**Latent space compression:**
- **Invertible linear transformation A** maps images from pixel space to latent space using orthogonal projection (QR decomposition)
- Transformation is fully invertible: `A⁻¹ = A^T`
- **Binary mask Λ** preserves known information from the blurred image

---

## Why this actually pushes the model to deblur effectively

This is the part worth understanding slowly, since it's the whole point of the paper:

**DCSM's role:** During training on clear images, DCSM learns what a "natural" image looks like at different noise levels — general structure at high noise, fine details at low noise.

**The consistency model's role:** Maps any noisy version of an image directly to the clean version in a single step, enforcing consistency along the denoising path.

**Zero-shot deblurring process:**
1. Input blurred image y, map to latent space, apply mask to keep known information
2. Initialize with blurred image + small amount of noise
3. Single-step denoise using consistency model
4. Apply mask again (blend known information with generated clean image)
5. Iterative refinement: repeatedly add noise and denoise, each time re-applying the mask

**Why iterative refinement helps:** Each iteration corrects errors and improves detail, while the mask ensures we don't lose known information from the original blurred image.

**Physical analogy:** Imagine a blurry photo where you know the general shape and colors (low-frequency), but fine details are smeared. ID-CDM is like a restoration expert who takes the blurry photo, marks the known parts, generates a clean version, compares and corrects inconsistencies, and repeats until the image is clear.

---

## Method summary

**Stage 1: Pre-train DCSM distillation model**
- U-Net-like architecture with time-dependent score estimation
- Time encoding using Gaussian random features
- Trained on clear image dataset (GoPro training set)

**Stage 2: Train consistency model**
- Use DCSM to estimate scores at each time step
- Euler or Heun ODE solver to estimate data distribution
- Minimize discrepancy between two adjacent data points on the trajectory

**Stage 3: Zero-shot deblurring**
- Map blurred image to latent space using invertible linear transform
- Apply binary mask to preserve known information
- Iterative sampling (N=60 steps) with noise addition and denoising

---

## Key results

- **GoPro dataset:** PSNR 33.19 dB, SSIM 0.970 (highest SSIM among all methods)
- **RealBlur-R:** PSNR 36.34 dB (highest, excellent for low-light nighttime scenes)
- **LPIPS (perceptual quality):** 0.0768 (best among all methods)
- **Computational efficiency:** 127.5 G FLOPs, 5.8s per image (40% faster than EDM, significantly faster than other diffusion-based methods)
- **Robustness:** Smallest performance drop when Gaussian noise is added (σ=5,15,25)

**Ablation findings:**
- Removing drift term (DCSM) reduces FLOPs by ~20%
- Adding consistency model improves SSIM from 0.965 to 0.970
- Full ID-CDM is 40% faster than EDM + DDPM
- Optimal iteration steps: N=20 gives best balance (33.19 dB in 5.8s)

---

## Relevance to our FYP

**Stage 5 – Face recognition:**
- This paper's approach is the baseline option for deblurring before face recognition
- DCSM is a simplified diffusion model that could be used for face deblurring
- Zero-shot operation means we don't need paired blurry-clear face datasets

**Stage 7 – Database matching:**
- The paper uses invertible linear transformations to compress images — similar to how face embeddings work (low-dimensional representation)
- Iterative refinement (adding noise and denoising) is conceptually similar to how ArcFace training tightens clusters

**Connection to ArcFace:**
- Both methods rely on geometric interpretation of angles
- ArcFace uses angular margin on the embedding sphere; ID-CDM uses cosine similarity (angle-based) for zero-shot editing

**Why this matters for our project:**
- Understanding diffusion models (simplified version via DCSM)
- Zero-shot capabilities (models can work without paired training data)
- Computational efficiency (consistency models reduce sampling steps)
- Latent space compression (similar to how face recognition models work in embedding space)

**Core insight:** Working in a compressed latent space with geometric operations (angles, distances) enables both efficient computation and better generalization.
