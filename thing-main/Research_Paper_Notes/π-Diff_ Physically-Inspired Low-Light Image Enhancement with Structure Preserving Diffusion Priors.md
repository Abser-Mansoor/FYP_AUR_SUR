# π-Diff: Physically-Inspired Low-Light Image Enhancement with Structure Preserving Diffusion Priors

> **CVPRW 2026**

π-Diff is an **unsupervised, training-free framework for Low-Light Image Enhancement (LLIE)** that combines diffusion priors with a physically-inspired image signal processing (ISP) strategy.

Instead of using diffusion to directly generate the final enhanced image, π-Diff treats its output as a **noiseless structural proxy** and performs illumination and color restoration separately.

## Paper Information

* **Title:** π-Diff: Physically-Inspired Low-Light Image Enhancement with Structure Preserving Diffusion Priors
* **Conference:** CVPRW 2026
* **Authors:** Min Wang, Xue Wang, Guoqing Zhou, Qing Wang
* **Task:** Low-Light Image Enhancement
* **Framework:** Unsupervised & Training-Free

## Overview

Low-light images commonly suffer from:

* Poor exposure
* Noise
* Color distortion
* Low contrast
* Detail loss

Traditional diffusion-based LLIE methods often use diffusion to perform denoising, illumination, color restoration, and enhancement together. This can lead to **blurred details, hallucination, incorrect exposure, and color distortion**.

π-Diff solves this by **separating structural recovery from physical enhancement**.

## Key Idea

### Conventional Approach

```text
Low-Light Image
      ↓
Diffusion Model
      ↓
Enhanced Image
```

### π-Diff

```text
Low-Light Image
      ↓
Diffusion Prior
      ↓
Structural Proxy
      ↓
Intrinsic Decomposition
   ┌──┴──┐
   ↓     ↓
Illumination  Reflectance
   ↓
Adaptive Gain
   ↓
Gamma Correction
   └──┬──┘
      ↓
Enhanced Image
```

The diffusion output provides structural information such as **edges, details, and object boundaries**, while the physical recovery process handles brightness and color.

## Main Components

### 1. Diffusion Structural Prior

The diffusion model produces a clean structural representation rather than directly generating the final image.

### 2. Intrinsic Decomposition

The image is separated into:

* **Illumination:** lighting and exposure
* **Reflectance:** intrinsic scene colors

Conceptually:

```text
Image ≈ Reflectance × Illumination
```

### 3. Adaptive Light Gain

An adaptive gain compensates for insufficient illumination instead of applying uniform brightness enhancement.

### 4. Gamma Correction

Gamma correction remaps intensity values to produce more visually appropriate exposure.

### 5. Reflectance Consistency

Maintaining the reflectance map helps preserve natural colors and reduces color shifts and over-saturation.

## Why π-Diff Works

| Problem          | π-Diff Solution                  |
| ---------------- | -------------------------------- |
| Blurry details   | Structural diffusion proxy       |
| Hallucination    | Decoupled enhancement            |
| Color distortion | Reflectance preservation         |
| Under-exposure   | Adaptive gain + gamma correction |
| Detail loss      | Structure-preserving prior       |

## Comparison

| Method     | Main Approach                        | Issue                  |
| ---------- | ------------------------------------ | ---------------------- |
| GDP        | Direct diffusion enhancement         | Direct generation      |
| AGLLDiff   | Diffusion enhancement                | Possible hallucination |
| QuadPrior  | Stable Diffusion prior               | Direct generation      |
| **π-Diff** | Structural proxy + physical recovery | Sampling cost          |

The main difference is that π-Diff **does not ask diffusion to directly generate the enhanced image**.

## Strengths

* **Training-free:** No paired training data required.
* **Detail preservation:** Maintains fine structures and edges.
* **Physically interpretable:** Uses illumination, reflectance, gain, and gamma correction.
* **Reduced hallucination:** Diffusion is used primarily for structural information.
* **Good real-world performance:** Designed and evaluated for real low-light images.

## Limitations

* Diffusion sampling can be **slow**.
* Multiple sampling steps increase computational cost.
* Results can depend on the diffusion sampling process.
* Real-time deployment may be challenging.

## Main Contributions

1. Introduces a **structural proxy paradigm** for diffusion-based LLIE.
2. Uses **physics-inspired illumination enhancement** with adaptive gain and gamma correction.
3. Preserves **structure and intrinsic colors** through reflectance consistency.
4. Provides a **training-free, unsupervised** enhancement framework.
5. Achieves competitive performance on real-world LLIE benchmarks.

## Core Concept

> **π-Diff uses diffusion to recover clean structural information and then performs physically-inspired illumination enhancement instead of directly generating the final bright image.**

### Keywords

`Low-Light Enhancement` · `Diffusion Models` · `Image Restoration` · `Intrinsic Decomposition` · `Illumination` · `Reflectance` · `Gamma Correction` · `ISP` · `Training-Free`
