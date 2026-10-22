# MagFace: A Universal Representation for Face Recognition and Quality Assessment

**Authors:** Qiang Meng, Shichao Zhao, Zhida Huang, Feng Zhou

**Venue:** CVPR 2021

---

## General Overview

1. ArcFace and CurricularFace use one fixed margin for every face, regardless of how good or bad that particular face image is — a crystal-clear photo and a blurry photo of the same person are held to the exact same standard, which allows inconsistent, unpredictable clustering within an identity.
2. MagFace's core contribution: stop throwing away a face embedding's **magnitude** (its raw length, discarded by normalization in ArcFace/CurricularFace) and instead use it as a built-in, free quality signal.
3. It designs the loss so that high-quality, easy faces are rewarded for having **large magnitude** and are pulled tightly toward their identity's center, while low-quality, ambiguous faces are pushed toward the origin with **small magnitude** — forming a cone-shaped structure per identity instead of an unpredictable scatter.
4. This is achieved with two small additions on top of an ArcFace-style loss: a magnitude-aware margin `m(a)` (stricter for high-magnitude samples) and a regularizer `g(a)` (always rewards larger magnitude) — the tension between the two automatically settles each face at the "right" magnitude for its own quality, with mathematical proof that this always converges reliably.
5. MagFace is a strict generalization of ArcFace: set the margin to a constant and turn the regularizer off, and the loss becomes exactly ArcFace.
6. No quality labels, no extra network, no extra computation at inference — the magnitude comes for free from the same embedding already being computed.
7. Demonstrated useful for three separate tasks: face recognition (same task as ArcFace), face quality assessment (using magnitude as a quality score), and face clustering (grouping unlabeled faces).

---

## The problem it's solving

ArcFace and similar margin losses apply the **same fixed margin** to every training face, no matter its quality. This means a clear, high-quality photo and a blurry, ambiguous photo of the same identity are both simply required to land "somewhere in the allowed zone" around that identity's center — with nothing enforcing that the clear photo should sit closer to the center than the blurry one. In practice, this produces inconsistent within-class structure: a genuinely easy face might end up near the boundary just by chance, and the model can end up overfitting as it tries to force hard, noisy samples into the same tight allowed zone as everyone else. This hurts performance specifically in unconstrained, real-world conditions (like surveillance footage) where quality varies a lot.

## The core idea, in one sentence

Keep using the face embedding's **magnitude** (its raw length, normally discarded) as a natural, mathematically-guaranteed quality signal — design the loss so easy, high-quality faces are rewarded for large magnitude and pulled close to their class center, while ambiguous, low-quality faces are pushed toward the origin with small magnitude, kept safely away from corrupting the tight cluster the good faces form.

## How it actually works (the mechanism)

1. **Magnitude-aware margin, `m(a)`.** Instead of one fixed margin `m` for everyone, the margin now depends on `a`, the magnitude of that specific face's own embedding (before normalization). As `a` grows, the required margin grows too — meaning high-magnitude samples are held to a stricter, tighter standard and must sit even closer to their class center than a fixed-margin loss would demand.
2. **Regularizer, `g(a)`.** On its own, the tightening margin above would let a face "cheat" by simply keeping its magnitude small (an easier margin to satisfy). The regularizer closes this loophole by always rewarding larger magnitude, for every sample, no matter what.
3. **The tension between the two settles each face automatically.** For a genuinely easy, well-matched face, going big (large magnitude) and tight (close to center) is a net win under both loss terms. For a genuinely ambiguous face, trying to force a large magnitude while also being held to a tight margin creates a bigger competing penalty — so the loss-minimizing outcome naturally settles at a smaller magnitude for that face instead. Nothing is manually labeled or hand-tuned per sample; it falls out of the optimization itself.
4. **Proven mathematically, not just observed.** The paper proves two properties: (a) for every face there is exactly one best magnitude value that training converges to (no instability), and (b) that best magnitude value always increases as the face's angle to its own class center shrinks (i.e., as it becomes easier to recognize). Magnitude reliably tracks recognizability by mathematical guarantee.

## The resulting shape: a cone, not a scatter

Put together, each identity forms a **cone**: near the tip (large magnitude), clean and high-quality faces sit tightly packed close to the class center direction; near the wide base (small magnitude, close to the origin), blurry or ambiguous faces of that same identity are spread out and kept at a safe distance, so they can't drag the "true" center around or blur the decision boundary with neighboring identities.

## Relationship to ArcFace

MagFace is a strict generalization of ArcFace. Set the margin function `m(a)` to output the same fixed value regardless of `a`, and turn the regularizer `g(a)` off entirely, and the MagFace loss reduces to exactly the ArcFace loss. In short: ArcFace + a magnitude-aware margin + one small extra regularizing term.

## Method summary

- Same backbone and setup as ArcFace (ResNet100, 112×112 aligned faces, MS1MV2 training set: 5.8M images, 85K identities).
- Correct-identity term: `cos(θ_yi + m(a_i))`, where `m(a_i)` grows with the sample's own magnitude `a_i` instead of being a fixed constant.
- Adds a regularization term `λ_g · g(a_i)`, where `g` is a decreasing convex function of magnitude, penalizing small magnitudes.
- Magnitude is bounded within a fixed range (`la = 10` to `ua = 110` in their experiments); `m(a)` is chosen as a linear function and `g(a)` as a hyperbola.
- At test time, nothing changes from ArcFace: still plain cosine-similarity comparison of 512-D features. The magnitude is simply read off as a bonus quality score, with no extra computation.
- For identities with multiple images (e.g., IJB-B/IJB-C), a variant called "MagFace+" combines multiple images by weighting each one's contribution by its own magnitude, instead of averaging them equally.

## Key results

- **Face recognition:** on easier, near-saturated benchmarks (LFW, CFP-FP, AgeDB-30, CALFW, CPLFW), gains over ArcFace were small (a fraction of a percent), as expected since these tests are already close to maxed out. On harder, large-scale benchmarks (IJB-B, IJB-C, built from real unconstrained video/photos), improvements over ArcFace were more meaningful — roughly 2–3.6% depending on the exact threshold, with MagFace+ adding a further boost by weighting multi-image identities by quality.
- **Face quality assessment:** the magnitude, entirely for free, functions as a genuinely good quality score. Grouping thousands of real images into buckets by magnitude and averaging the faces in each bucket showed low-magnitude buckets look blurry/non-frontal and high-magnitude buckets look sharp/frontal. Outperformed several dedicated quality-estimation methods (classic image metrics like Brisque/Niqe/Piqe, and learned methods like FaceQNet and SER-FIQ, the latter requiring 100 extra forward passes per image) at identifying which faces actually hurt verification accuracy — with essentially zero extra computational cost.
- **Face clustering:** across four different clustering algorithms (K-means, hierarchical clustering, DBSCAN, and a graph-neural-network method) and three dataset sizes, MagFace features consistently clustered better than ArcFace features, since ambiguous faces no longer sit on the boundary between two people's clusters.

## Relevance to our FYP (Stage 5 – face recognition)

- Same family as ArcFace/CurricularFace — a margin-based loss tackling "how do I handle faces of different quality," but the specific mechanism (magnitude = quality, for free) is the direct conceptual ancestor of AdaFace, the model our project actually recommends deploying.
- AdaFace borrows this exact "use the embedding's norm as a quality signal" idea and adapts it further specifically for low-quality, surveillance-style faces — reading MagFace first should make AdaFace's design click quickly.
- The quality-assessment angle (magnitude as a free quality score, no extra network needed) is conceptually related to the kind of frame/face quality checks discussed for Stage 1 of our pipeline, though it uses a different mechanism than what's currently proposed there.
