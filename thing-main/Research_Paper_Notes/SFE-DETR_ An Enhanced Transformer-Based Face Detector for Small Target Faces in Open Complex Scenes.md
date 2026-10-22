# SFE-DETR: An Enhanced Transformer-Based Face Detector for Small Target Faces in Open Complex Scenes

> **SFE-DETR — Lightweight and Efficient Transformer-Based Detector for Small Faces**

SFE-DETR is a lightweight end-to-end object detection framework designed specifically for **small face detection in complex and crowded environments** such as surveillance systems, UAV imagery, and dense crowd scenes.

It improves small-face detection by redesigning the **backbone, attention mechanism, and feature pyramid network**, achieving better accuracy while using fewer parameters than the RT-DETR-R18 baseline.

---

## 📌 Problem

Small face detection in open and complex scenes is challenging because faces often suffer from:

* 🔹 Extremely small scale
* 🔹 Heavy occlusion
* 🔹 Motion blur
* 🔹 Poor illumination
* 🔹 Complex backgrounds
* 🔹 Dense crowds

Although detectors such as **YOLO** and **DETR** provide strong general-purpose detection performance, they can struggle with tiny faces or introduce significant computational overhead.

SFE-DETR addresses these issues with a lightweight architecture specifically optimized for small-face detection.

---

## 🚀 Main Contributions

SFE-DETR introduces three major components:

### 1. ISD-Net

A redesigned backbone consisting of two important modules:

#### IS Module — Shallow Layers

* Enhances local-global feature interaction.
* Preserves fine-grained spatial details.
* Helps retain information from tiny faces.

#### DR Module — Deep Layers

* Expands the receptive field.
* Uses re-parameterization for efficient feature extraction.
* Reduces the number of model parameters.

---

### 2. MHMSA

**Multi-Head Multi-Scale Self-Attention**

MHMSA improves the model's ability to focus on small faces while suppressing irrelevant background information.

It uses a dual-branch structure containing:

* Multi-scale convolutions
* Channel attention

This allows the network to capture features at different spatial scales and improve small-object representation.

---

### 3. SFE-FPN

A feature pyramid network designed specifically for small-face detection.

It includes:

#### SPDConv

* Preserves high-resolution **P2 spatial information**.
* Helps maintain details of extremely small faces.

#### CSPO-Fusion

Combines:

* Local feature branches
* Large-receptive-field branches
* Global feature branches
* Frequency-domain attention

This provides robust multi-scale feature fusion while keeping the computational cost low.

---

ISD-Net (Backbone): Keeps tiny faces from disappearing as the image shrinks, and makes the "eyes" lighter and faster.

MHMSA (Attention): Forces the "brain" to actually pay attention to the tiny dots in the corner, instead of just staring at the big obvious objects.

SFE-FPN (Mixer) : Combines all the zoom levels (P2, P3, P4, P5) perfectly, preserving tiny details while removing background clutter, without making the computer slow.

---

## 📊 Key Results

### SCUT-HEAD Dataset

| Metric         | RT-DETR-R18 |   SFE-DETR |
| -------------- | ----------: | ---------: |
| **mAP50**      |       92.6% |  **94.7%** |
| **AP-s**       |       39.0% |  **42.1%** |
| **Parameters** |      19.9 M | **14.3 M** |
| **GFLOPS**     |        57.0 |   **55.2** |
| **FPS**        |    **64.2** |       61.7 |

### Improvements

* **+2.1% mAP50**
* **+3.1% AP-s**
* **28.1% fewer parameters**
* Lower computational cost
* Maintains real-time detection performance

Although SFE-DETR is slightly slower than RT-DETR-R18, it provides significantly better small-face detection accuracy with a much lighter model.

---

## 🌍 WIDER FACE Results

On the challenging **WIDER FACE Hard** subset, SFE-DETR achieves:

> ### **86.3%**

| Model        |   Hard AP |
| ------------ | --------: |
| **SFE-DETR** | **86.3%** |
| YOLO5Face    |     85.2% |
| YOLO8Face    |     84.7% |
| RT-DETR-R18  |     83.7% |

SFE-DETR outperforms the comparable YOLO and RT-DETR variants while maintaining a lightweight architecture.

---

## 🔬 Ablation Study

Each proposed component contributes to the overall performance.

| Component   | mAP50 Improvement | AP-s Improvement | Parameter Impact |
| ----------- | ----------------: | ---------------: | ---------------: |
| **ISD-Net** |             +0.9% |                — |       **-29.6%** |
| **MHMSA**   |             +0.5% |            +1.4% |                — |
| **SFE-FPN** |             +1.0% |            +1.9% |              +2% |

The SFE-FPN provides particularly strong improvements while introducing only a small parameter increase compared with a naive P2 feature addition, which requires approximately **7.5%** more parameters.

---

## 👁️ Visual Analysis

Visualization results demonstrate that SFE-DETR produces stronger responses around small faces.

Compared with the baseline:

* SFE-DETR focuses more strongly on small faces.
* Background distractions are reduced.
* Large faces receive less unnecessary attention.
* Dense crowd scenes are handled more effectively.
* Detection remains robust under occlusion and blur.

Heatmap visualizations further demonstrate the model's improved ability to identify small-face regions.

---

## ✅ Strengths

### High Small-Face Accuracy

Designed specifically for tiny and difficult-to-detect faces.

### Lightweight Architecture

Uses only **14.3M parameters**, significantly fewer than RT-DETR-R18.

### Real-Time Performance

Achieves approximately **61.7 FPS**, making it suitable for real-time applications.

### End-to-End Detection

Uses a Transformer-based end-to-end detection pipeline without requiring **NMS (Non-Maximum Suppression)**.

### Strong Generalization

Demonstrates competitive performance on both **SCUT-HEAD** and **WIDER FACE**.

### Edge-Friendly

The reduced parameter count and computational requirements make the architecture suitable for resource-constrained devices.

---

## ⚠️ Limitations

Despite its advantages, SFE-DETR has some limitations:

* Slightly lower FPS compared with the RT-DETR-R18 baseline.
* Performance may require further improvement in extremely dynamic environments.
* Extreme weather conditions may still affect detection quality.
* Additional optimization may be required for highly resource-constrained edge devices.

---

## 🔮 Future Work

Future improvements could focus on:

* Cross-module architectural optimization.
* Further reduction of computational complexity.
* Better adaptation to extreme dynamic scenes.
* Robust detection under rain, fog, snow, and other extreme weather conditions.
* Further optimization for embedded and mobile hardware.
* Exploring more efficient attention mechanisms.

---

## 🧠 Core Takeaway

SFE-DETR demonstrates that **small-face detection can be significantly improved without making the detector heavier**.

By redesigning the:

**Backbone → Attention → Feature Pyramid**

the model achieves a strong balance between **accuracy, efficiency, and real-time performance**.

On SCUT-HEAD, SFE-DETR improves mAP50 from **92.6% to 94.7%** and AP-s from **39.0% to 42.1%**, while reducing parameters by **28.1%**.

> **SFE-DETR = Better Small-Face Detection + Fewer Parameters + Real-Time Efficiency**

---

## 📚 Summary

| Category                 | SFE-DETR             |
| ------------------------ | -------------------- |
| **Task**                 | Small Face Detection |
| **Architecture**         | End-to-End DETR      |
| **Backbone**             | ISD-Net              |
| **Attention**            | MHMSA                |
| **Feature Pyramid**      | SFE-FPN              |
| **Dataset**              | SCUT-HEAD            |
| **Additional Benchmark** | WIDER FACE           |
| **mAP50**                | 94.7%                |
| **AP-s**                 | 42.1%                |
| **Parameters**           | 14.3M                |
| **GFLOPS**               | 55.2                 |
| **FPS**                  | 61.7                 |
| **WIDER FACE Hard**      | 86.3%                |
| **NMS**                  | Not Required         |

---

## ⭐ Conclusion

SFE-DETR is a **lightweight, efficient, and small-face-focused Transformer detector** that addresses the limitations of conventional YOLO and DETR-based approaches in complex scenes.

Its combination of **ISD-Net, MHMSA, and SFE-FPN** enables improved feature representation, stronger small-face localization, and efficient multi-scale detection while maintaining real-time performance.
