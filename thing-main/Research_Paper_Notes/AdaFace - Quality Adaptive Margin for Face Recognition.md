# AdaFace: Quality Adaptive Margin for Face Recognition

**Authors:** Minchul Kim, Anil K. Jain, Xiaoming Liu

**Venue:** CVPR 2022

---

## General Overview

1. Margin-based losses (ArcFace, CurricularFace) and hard-sample mining both judge a face only by whether the model currently gets it right or wrong — never by whether the image is actually good enough to be gotten right in the first place.
2. Some low-quality faces are merely *hard* (still contain usable clues); others are truly **unidentifiable** — even a human couldn't recognize them. Blindly emphasizing "hard" samples forces the network to also emphasize these unidentifiable images, causing it to latch onto irrelevant shortcuts (clothing color, resolution) instead of real identity cues.
3. Core contribution: emphasis on a hard sample should depend on its **image quality**, not just on training progress (CurricularFace) or a fixed rule (ArcFace/MagFace) — emphasize hard-but-high-quality faces (real learnable signal), de-emphasize hard-but-low-quality faces (likely unidentifiable).
4. Two supporting findings: (a) different margin *types* (positive angular, negative angular, additive) don't just move the decision boundary — they change how much training emphasis a sample gets depending on its difficulty; (b) feature norm (embedding magnitude) reliably correlates with actual measured image quality (verified against the BRISQUE quality metric), separately from "difficulty."
5. AdaFace combines both findings: it computes a per-batch, per-face quality score from feature norm, and uses it to continuously slide the margin between known margin types — becoming ArcFace-like for low-quality faces (de-emphasize very hard ones) and a negative-margin-like behavior for high-quality faces (keep emphasizing hard ones).
6. Because it doesn't blindly punish unidentifiable images, AdaFace can benefit from aggressive data augmentation during training, whereas CurricularFace's accuracy gets *worse* under the same augmentations.
7. Validated on 9 datasets across 3 quality tiers, with the largest gains on the datasets closest to real surveillance video (IJB-S, TinyFace) — directly relevant to our project.

---

## The problem it's solving

Prior methods (ArcFace, CurricularFace, MagFace) treat "hard sample" as a single category, regardless of *why* it's hard. But a face can be hard for two very different reasons: it might be a genuinely learnable but tricky case (odd pose, partial occlusion, poor lighting — a **hard-but-recognizable** image), or it might be so degraded that no real identity information remains (an **unidentifiable** image). Forcing a model to fit unidentifiable images just as hard as genuinely hard-but-recognizable ones teaches it to exploit irrelevant shortcuts to lower its training loss, which hurts real-world accuracy — especially on low-quality data like surveillance footage, where unidentifiable images are common.

## Two findings the paper establishes first

1. **Margin type controls training emphasis, not just the decision boundary.** By analyzing the gradient math behind each margin function, the paper shows: a plain additive margin (CosFace) doesn't change emphasis at all — only the boundary shifts. A positive angular margin (ArcFace, and by extension MagFace) makes gradient emphasis peak near the decision boundary but *shrink* for the very hardest samples — i.e., ArcFace already quietly backs off on the hardest, most likely-unidentifiable cases. A **negative** angular margin does the opposite: emphasis keeps rising the harder a sample gets, with no pullback.
2. **Feature norm (embedding magnitude) really does track image quality.** Testing 1,534 real training images, the correlation between feature norm and an independent image-quality score (BRISQUE) reached about 0.52 — higher and more consistent (visible from early training onward) than using "is this sample currently classified correctly" as a stand-in for quality. This confirms magnitude tracks *quality* specifically, distinct from *difficulty*.

## The core idea, in one sentence

Combine both findings into one adaptive margin: **use each face's own feature-norm-based quality score to decide which margin "type" (ArcFace-like or negative-margin-like) to apply**, so hard-but-high-quality faces get emphasized harder while hard-but-low-quality (likely unidentifiable) faces get eased off, exactly as needed.

## How it actually works (the mechanism)

1. **Compute an "image quality indicator" for free**, from each face's own raw embedding magnitude before normalization: subtract the current training batch's average magnitude, divide by its spread, then clip the result to between -1 and 1. A score near -1 means "among the lowest quality in this batch," near +1 means "among the highest." This uses running batch statistics (via an exponential moving average, momentum 0.99) so it stays stable even with small batches, and gradients are stopped from flowing back through it so the network can't "cheat" by shrinking its own magnitudes.
2. **Feed that quality score into the margin function itself.** AdaFace's margin has two small pieces — one angular, one additive — both built directly from that quality score, so the margin's *character* continuously shifts as quality moves from low to high.
3. **The key trick: this slide passes exactly through the margin types studied earlier.** At the lowest quality score (-1), AdaFace's formula becomes mathematically identical to **ArcFace**. At the middle (0), it becomes identical to **CosFace**. At the highest quality score (+1), it becomes a **negative angular margin** — the type that (per finding #1) keeps boosting emphasis on hard samples rather than backing off.
4. **Net effect:** a low-quality face that's hard to classify is treated like it would be under ArcFace — gently eased off once very hard, since it's likely unidentifiable. A high-quality face that's hard to classify gets pushed *harder*, since a clear image that's still confusing almost certainly holds real, learnable identity information worth extracting.

## Why this beats a fixed schedule (CurricularFace) or a fixed rule (MagFace)

CurricularFace decides how much to emphasize hard samples using one shared value for the *whole batch*, based only on overall training progress — every face in a batch gets the same treatment regardless of its own quality. MagFace only ever uses a positive-angular-margin type of behavior (scaled by magnitude), so — per finding #1 — it never stops de-emphasizing the very hardest samples, no matter how high-quality they are. AdaFace is the only one of the three that decides emphasis **per individual face**, using that face's own quality, and is the only one able to actively boost emphasis on hard samples when quality is genuinely high.

## Method summary

- Same backbone, same 112×112 aligned face inputs, same training datasets (MS1MV2, MS1MV3, WebFace4M) as ArcFace/CurricularFace/MagFace — this is purely a change to the loss function.
- Hyperparameters: margin `m = 0.4` (best overall balance between high- and low-quality benchmarks) and concentration `h = 0.33` (chosen so roughly 68%, one standard deviation's worth, of a batch's quality scores land inside the useful -1 to 1 range).
- Because it doesn't blindly punish unidentifiable images, AdaFace benefits from on-the-fly data augmentation (random cropping, rescaling/blurring, photometric jitter) during training — these augmentations *hurt* CurricularFace's accuracy but *help* AdaFace's, since AdaFace can automatically down-weight any augmentation-induced low-quality garbage.
- Extra training cost: about 1% (0.3229s vs. ArcFace's 0.3193s per iteration). Zero extra cost at inference — deployment is still plain cosine-similarity comparison of the final embedding.

## Key results

- **High-quality benchmarks** (LFW, CFP-FP, CPLFW, AgeDB, CALFW): roughly tied with the best prior methods — expected, since these are already near-saturated for everyone.
- **Mixed-quality benchmarks** (IJB-B, IJB-C): meaningfully better — reduces errors of the next-best method by roughly 11% and 9% respectively.
- **Low-quality benchmarks (most relevant to our project):** IJB-S is explicitly a **surveillance video benchmark**, including a "Surveillance-to-Surveillance" protocol (matching a face from one surveillance video against a gallery built from other surveillance video — close to our actual CCTV use case). On that protocol, CurricularFace scores 19.5% Rank-1; AdaFace scores 23.7% with the same training data, and 35.1% when trained on the larger WebFace4M dataset — nearly double CurricularFace's number. TinyFace (tiny, low-resolution faces) shows a similarly clear win.

## Limitations the authors flag

- Doesn't specially handle **mislabeled** training images (a different problem from low quality). Since AdaFace deliberately emphasizes hard-but-high-quality samples, a clear photo that happens to be mislabeled could get emphasized even more strongly than under a standard loss — worth being aware of for any custom training data.
- Their training data (MS1MV*) derives from a dataset later withdrawn over consent concerns; the authors recommend newer datasets (e.g., WebFace4M) for future work.

## Relevance to our FYP (Stage 5 – face recognition)

- This is the model our project's readme recommends deploying, and it's the only one of the four Stage 5 papers explicitly validated on real surveillance video (IJB-S) — and it wins there by the widest margin of any comparison in the reading list.
- Directly builds on MagFace's "use the norm as a quality signal" idea, but goes further: MagFace only ever de-emphasizes the hardest samples (like ArcFace), while AdaFace can actively emphasize hard-but-high-quality samples too, which MagFace's mechanism structurally cannot do.
- Since AdaFace is robust to *moderate* quality variation by design, it raises a genuinely open question for our own architecture: how much do Stages 1–2 (quality check and restoration) actually add on top of AdaFace's built-in robustness? Worth testing directly rather than assuming either way.
