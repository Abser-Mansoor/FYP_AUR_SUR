# CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition

**Authors:** Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, Feiyue Huang

**Venue:** CVPR 2020

---

## General Overview

1. ArcFace-style margin losses treat every training face the same, all the way through training — they never ask "is this particular face easy or hard for the model right now?"
2. Separately, "hard sample mining" methods try to focus training on the images the model currently gets wrong — but doing this too aggressively, too early, can make training unstable or even fail to converge.
3. CurricularFace's core contribution: borrow the idea of **curriculum learning** ("easy things first, hard things later," the way humans learn) and build it directly into the face-recognition loss function, fully automatically — no manual scheduling or hand-tuned hyperparameters required.
4. It keeps ArcFace's usual margin for the *correct* identity exactly the same. The novelty is entirely in how it treats a face's similarity scores to the *wrong* identities, especially for faces the model is currently misclassifying.
5. An automatically-updating value called `t` tracks "how far along is training" and steadily shifts the loss's attention from *easy* samples (early training) to *hard, misclassified* samples (later training).
6. Fixes a real, demonstrated failure case: training an ArcFace loss on a small, lightweight network (MobileFaceNet) can literally fail to converge, while CurricularFace trains successfully on the exact same setup.
7. Adds essentially zero extra computational cost (0.378s vs. ArcFace's 0.370s per training step) and no extra cost at all during actual deployment/inference, since only the training loss changes.

---

## The problem it's solving

Two separate lines of prior work each had a real flaw:
- **Margin-based losses** (SphereFace, CosFace, ArcFace) push all identities apart with a fixed margin, but never account for how difficult any individual training face currently is.
- **Hard-sample mining methods** (OHEM, Focal Loss, MV-Arc-Softmax) do focus on the model's current mistakes, but emphasizing hard/confusing examples too heavily from the very start of training — before the model has learned even the basics — can cause training to become unstable, or fail to converge entirely.

Nobody had combined "margin" and "mining" in a way that avoided both problems at once.

## The core idea — automatic curriculum learning

Mirror how people actually learn: master the easy material first, then gradually shift attention to the hard material. CurricularFace does this **automatically and adaptively** — it doesn't need a pre-planned schedule (traditional curriculum learning requires sorting the whole dataset by difficulty in advance); instead, it re-evaluates "which stage of training are we in" and "which specific samples are hard right now" continuously, on the fly, every mini-batch.

## How it actually works (the mechanism)

1. **The correct-identity margin stays exactly like ArcFace.** No change there.
2. **For each face, check if the model currently gets it right or wrong.** "Wrong" (a hard sample) specifically means some other identity's score beats this face's own margin-boosted correct-identity score.
3. **Easy samples:** their similarity scores to the wrong identities are left completely untouched — same as plain ArcFace.
4. **Hard samples:** their similarity scores to the wrong identities get multiplied by an adjustable factor, built from two ingredients:
   - **`t`** — a single automatically-computed number representing overall training progress. It's calculated after every batch, by averaging how close correct-identity scores are across that batch, then smoothed over time (an exponential moving average) so a single noisy batch doesn't throw it off. It starts near 0 and climbs toward 1 as training progresses and the model improves.
   - **The angle to the wrong identity** — so that even among hard samples in the same batch, the more confusing ones (closer to a wrong identity) get proportionally extra emphasis, not just a flat "hard" penalty.
5. **Early in training** (`t` near 0): the multiplying factor is small, so hard samples' wrong-identity scores are actually *shrunk* — the loss goes easy on them, letting the model spend its energy mastering the clear-cut, easy faces first.
6. **Later in training** (`t` near 1): the same factor grows, so hard samples' wrong-identity scores get *boosted* — the loss now leans in hard on exactly the still-confusing faces to squeeze out the remaining accuracy.

## Why this beats the obvious alternative (MV-Arc-Softmax)

A prior method already tried "boost the hard samples," but with a **fixed, manually-chosen** multiplier applied from the very first training step. That has two problems this paper avoids: (1) a badly-chosen fixed value can break training outright, and finding a good one requires manual trial and error; (2) emphasizing hard samples before the model has learned anything is exactly the instability risk described above. CurricularFace's `t` computes itself from training progress, and — crucially — starts low instead of high, so hard samples are eased into rather than front-loaded.

## Method summary

- Uses the same underlying architecture as ArcFace (e.g., ResNet50/ResNet100 backbone, 112×112 aligned face crops, normalized embeddings and class centers).
- Correct-identity score: identical to ArcFace's angular margin, `cos(θ_yi + m)`.
- Wrong-identity scores: left alone for easy samples; multiplied by `(t + cos θj)` for hard (misclassified) samples.
- `t` is updated every batch using an exponential moving average of the batch's average correct-identity similarity (momentum parameter 0.99, starting at `t = 0`).
- Trained with standard SGD; margin `m = 0.5`, scale `s = 64` — the same standard settings as ArcFace.

## Key results

- Trained on CASIA-WebFace (~0.5M images) and MS1MV2 (~5.8M images, 85K identities); evaluated on LFW, CFP-FP, CPLFW, AgeDB, CALFW, IJB-B, IJB-C, and MegaFace.
- Outperformed both ArcFace and MV-Arc-Softmax across nearly every benchmark. Gains were small on LFW (already near-saturated at ~99.8% for everyone) but clearly larger on harder tests: pose-variation benchmarks (CFP-FP, CPLFW) and large-scale identification (MegaFace).
- Extra training cost is negligible (0.378s vs. ArcFace's 0.370s per iteration), and there is zero extra cost at inference time, since only the training loss changes — the deployed model runs identically to an ArcFace-trained one.
- Standout robustness result: on **MobileFaceNet** (a small, efficiency-focused network), plain ArcFace's training diverged (loss became NaN), while CurricularFace trained successfully to good accuracy on the exact same setup — direct evidence that easing into hard samples helps small/lightweight models train reliably.

## Relevance to our FYP (Stage 5 – face recognition)

- Not a replacement for ArcFace or AdaFace — it's a training-time technique layered on top of a margin-based loss (the paper explicitly says the correct-identity margin can be swapped for any margin-based method).
- Conceptual stepping stone toward MagFace and AdaFace: all three papers are trying to answer "how do I know if this particular face is easy/hard or low/high quality, and how should that change training?" — CurricularFace answers it using "currently misclassified or not," while MagFace and AdaFace use embedding magnitude as the quality signal instead.
- The MobileFaceNet convergence result is worth remembering if the project ever needs a lightweight, efficient recognition model for real-time or edge deployment.
