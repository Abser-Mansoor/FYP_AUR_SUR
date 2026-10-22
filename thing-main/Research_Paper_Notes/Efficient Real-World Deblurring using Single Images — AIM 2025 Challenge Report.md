# Efficient Real-World Deblurring using Single Images — AIM 2025

**Authors:** Daniel Feijoo, Paula Garrido-Mellado, Marcos V. Conde, Jaesung Rim, Alvaro Garcia, Sunghyun Cho, Radu Timofte
**Venue:** ICCV Workshops — AIM 2025

---

## 1. Overview

The **AIM 2025 Challenge on Efficient Real-World Deblurring using Single Images** focuses on removing motion blur from real-world images while keeping the models computationally efficient.

Motion blur commonly occurs in **low-light conditions**, where cameras use longer exposure times. This problem is especially common in smartphones because their smaller sensors collect less light.

The main objective of the challenge is therefore:

> **Achieve high-quality image deblurring while keeping the model small and fast enough for mobile and edge devices.**

### Challenge Statistics

* **71** participants registered
* **4** teams submitted valid solutions
* Best PSNR: **31.13 dB**
* Maximum parameters: **< 5 million**
* Maximum computation: **< 200 GMACs**

---

## 2. Problem

Real-world image deblurring is difficult because real camera blur is more complex than synthetic blur.

Real images can contain:

* Camera and object motion
* Sensor noise
* Saturated pixels
* ISP processing effects
* Continuous motion patterns
* Different exposure conditions

Large deep-learning models can improve restoration quality, but they often require significant:

* Computation
* Memory
* Processing time

Therefore, the challenge focuses on both **restoration quality and efficiency**.

---

## 3. Dataset — RSBlur

The challenge uses the **RSBlur** dataset containing real-world blurred and sharp image pairs.

A dual-camera setup is used:

```text
Incoming Light
      ↓
Beam Splitter
   ↙       ↘
Short     Long
Exposure  Exposure
   ↓         ↓
 Sharp     Blurred
 Image      Image
```

The **short-exposure camera** captures the sharp ground-truth image, while the **long-exposure camera** captures the real motion-blurred image.

### Dataset

| Split            | Images |
| ---------------- | -----: |
| Training         |  8,887 |
| Validation       |  1,120 |
| Original Test    |  3,360 |
| Challenge Test   |    420 |
| Challenge Scenes |     84 |

The new challenge test set contains **different scenes**, helping prevent overfitting to the original dataset.

---

## 4. Evaluation Metrics

The challenge evaluates restoration using three metrics:

### PSNR

Measures pixel-level reconstruction quality.

**Higher is better.**

### SSIM

Measures structural similarity between the restored and sharp image.

**Higher is better.**

### LPIPS

Measures perceptual similarity.

**Lower is better.**

The overall score combines these metrics:

$$
Score = \lambda_1 PSNR + \lambda_2 SSIM + \lambda_3 LPIPS
$$

---

## 5. Computational Constraints

Models must satisfy strict efficiency requirements.

| Constraint            |           Limit |
| --------------------- | --------------: |
| Parameters            |        **< 5M** |
| MACs                  |      **< 200G** |
| Evaluation Resolution | **1920 × 1200** |

These restrictions encourage models that can potentially run on **mobile and edge hardware**.

---

# 6. Challenge Results

| Rank | Team          | Method                | Params |    MACs |       PSNR |  SSIM |
| ---: | ------------- | --------------------- | -----: | ------: | ---------: | ----: |
|   🥇 | MiVideoDeblur | **NAFRepLocal**       |  4.76M | 198.25G | **31.130** | 0.843 |
|   🥈 | MAILab        | **RestormerL**        |  1.41M | 199.39G |     31.100 | 0.840 |
|   🥉 | IPIU          | **Data Augmentation** |  4.35M | 146.33G |     30.492 | 0.832 |
|    4 | XD2025PBL     | **SA-NAFNet**         |  4.51M |  172.2G |     30.189 | 0.819 |
|    — | Baseline      | NAFNet-C16-L28        |  4.35M | 146.33G |     30.173 | 0.826 |

The winning method improved PSNR by roughly **1 dB compared with the baseline** while remaining within the computational limits.

---

# 7. NAFRepLocal — Winner

**Team:** MiVideoDeblur, Xiaomi Inc.

NAFRepLocal is based on **NAFNet** and introduces several improvements.

### Architecture

```text
Input
  ↓
Encoder
32 → 64 → 128 → 256
  ↓
Middle NAFBlock
512 Channels
  ↓
Decoder
256 → 128 → 64 → 32
  ↓
Output
```

### Main Improvements

#### SCA-L + SCA-G

The method combines local and global channel attention.

* **SCA-L:** captures local information
* **SCA-G:** captures global information

This helps the model recover both local details and larger image structures.

#### Reparameterization

The model can use more complex structures during training and convert them into simpler structures for inference.

```text
Complex Training Structure
          ↓
Reparameterization
          ↓
Simpler Inference Structure
          ↓
Faster Deployment
```

#### EMA

**Exponential Moving Average (EMA)** keeps a smoothed version of the model weights, helping produce more stable final parameters.

---

## 8. Progressive Training

NAFRepLocal uses four training stages.

| Stage | Resolution | Steps |  Time |
| ----: | ---------- | ----: | ----: |
|     1 | 512×512    |  400K |  ~37h |
|     2 | 1024×1024  |  400K | ~137h |
|     3 | 1024×1024  |  400K | ~132h |
|     4 | 1024×1024  |   50K |   ~8h |

Training uses:

* **AdamW**
* Cosine learning-rate scheduling
* PSNR loss
* Random flipping

The total training process takes approximately **314 hours**.

---

# 9. RestormerL — 2nd Place

RestormerL is a lightweight version of the **Restormer Transformer** architecture.

The team reduced the model size by:

* Reducing transformer blocks
* Reducing channel dimensions
* Removing the refinement block
* Replacing GELU with SiLU

### Architecture Changes

| Component  | Original        | RestormerL     |
| ---------- | --------------- | -------------- |
| Blocks     | [4,6,6,8]       | [2,2,2,4]      |
| Channels   | [48,96,192,384] | [16,32,64,128] |
| Refinement | Yes             | No             |
| Activation | GELU            | SiLU           |

Despite having only **1.41M parameters**, RestormerL achieved **31.10 dB PSNR**, almost matching the winner.

It uses progressive learning with increasing patch sizes:

```text
256×256
   ↓
384×384
   ↓
512×512
```

---

# 10. IPIU — 3rd Place

The IPIU approach is based on NAFNet and focuses heavily on **data augmentation**.

### Main Techniques

* Layer Normalization
* GELU
* Random cropping
* Scaling
* Horizontal/vertical flipping
* Test-Time Augmentation (TTA)

During inference, the model processes transformed versions of the image and combines the results.

This method achieved:

* **4.35M parameters**
* **146.33 GMACs**
* **30.49 dB PSNR**

The main lesson is that **strong augmentation can significantly improve a relatively simple architecture**.

---

# 11. SA-NAFNet — 4th Place

SA-NAFNet adds **Spatial Attention** to NAFNet.

The method also uses multiple losses:

```text
L1 Loss
   +
Perceptual Loss
   +
Edge Loss
   ↓
Better Restoration
```

### Training Strategy

The model is trained in multiple stages:

1. Pixel-level training using L1 loss
2. Structural training using L1 + perceptual + edge losses
3. Final refinement using all losses

This helps the model focus first on accurate pixels and later on structural and perceptual quality.

---

# 12. Important Technical Concepts

## NAFNet

**NAFNet — Nonlinear Activation Free Network** is designed for efficient image restoration.

Instead of relying heavily on conventional activation functions such as ReLU, it uses components such as:

* SimpleGate
* Simplified Channel Attention (SCA)
* Lightweight convolutional blocks

NAFNet provides a strong balance between **quality and computational efficiency**.

---

## Simplified Channel Attention

SCA helps the network identify important feature channels.

### SCA-G

Uses global information:

```text
Feature Map
    ↓
Global Pooling
    ↓
Channel Information
    ↓
Attention
```

### SCA-L

Uses local information to focus on nearby image structures.

Using both approaches allows the model to capture **global context and local details**.

---

## Reparameterization

Reparameterization separates the training architecture from the inference architecture.

```text
Training
Complex Structure
      ↓
Better Learning

Inference
Simplified Structure
      ↓
Lower Computation
      ↓
Faster Inference
```

This is particularly useful for mobile and edge deployment.

---

## MACs

**MACs (Multiply-Accumulate Operations)** measure computational complexity.

A model with fewer MACs generally requires less computation.

The challenge limits models to:

> **Less than 200 GMACs for 1920×1200 images.**

---

# 13. Common Findings

The top-performing approaches share several important ideas:

### 1. NAFNet is highly effective

Most of the top solutions use NAFNet or a modified version of it.

### 2. Progressive Training

Starting with smaller images and gradually increasing resolution helps training.

### 3. Attention

Local, global, and spatial attention help recover important image details.

### 4. Data Augmentation

Cropping, flipping, and scaling improve generalization.

### 5. Multiple Losses

Combining pixel, perceptual, and edge losses can improve visual quality.

### 6. Efficiency Matters

The challenge demonstrates that a model does not need to be extremely large to achieve strong deblurring performance.

---

# 14. Relevance to Our FYP

Deblurring is important for our **face recognition pipeline** because motion blur can damage important facial features.

A possible pipeline is:

```text
Camera Image
     ↓
Blurred Face
     ↓
Efficient Deblurring
     ↓
Face Detection
     ↓
Face Recognition
     ↓
Face Embedding
     ↓
Database Matching
```

A sharper face can provide better facial features for the recognition model.

### Main Lessons for Our FYP

* Use **NAFNet or a lightweight NAFNet variant** as a potential baseline.
* Maintain a low parameter count for efficient inference.
* Use progressive training for high-resolution images.
* Apply strong data augmentation.
* Consider attention mechanisms for detail recovery.
* Evaluate both restoration quality and computational cost.
* Place deblurring before face recognition when blur significantly affects recognition.

---

# 15. Final Conclusion

The AIM 2025 Challenge demonstrates that **real-world image deblurring can be both accurate and computationally efficient**.

The winning **NAFRepLocal** approach achieved **31.13 dB PSNR** while using only **4.76M parameters** and **198.25 GMACs**.

The main takeaway for our FYP is:

> **Instead of using a very large deblurring network, we should focus on a lightweight architecture with efficient attention, progressive training, strong augmentation, and suitable loss functions.**

For our application, an efficient deblurring stage can potentially improve the quality of blurred faces before **face recognition and database matching**.
