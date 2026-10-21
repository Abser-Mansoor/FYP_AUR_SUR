# ArcFace: Additive Angular Margin Loss for Deep Face Recognition

**Authors:** Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou

**Venue:** CVPR 2019

---

## General Overview

1. Face recognition needs faces of the *same* person to map close together and faces of *different* people to map far apart in embedding space.
2. Older training losses (plain softmax, or earlier margin tricks like SphereFace/CosFace) didn't enforce this separation strongly or cleanly enough.
3. ArcFace's core contribution: an **Additive Angular Margin Loss** — it adds a small, fixed extra angle onto the angle between a face and its own identity's "center" during training, before scoring how well the network did.
4. This margin is added directly to the *angle*, which has a clean geometric meaning and matches exactly how faces are compared at test time (cosine similarity = angle-based comparison).
5. Trained on large-scale datasets (millions of images, tens of thousands of identities) and became the standard baseline nearly every later face-recognition paper (CurricularFace, MagFace, AdaFace, etc.) compares itself against.
6. Code released as the open-source **InsightFace** project — still widely used for pretrained face-recognition models.

---

## The problem it's solving

Standard classification training (softmax) is built to separate categories like "cat vs. dog" — it doesn't specifically try to make same-identity faces *tightly clustered*. Earlier attempts to fix this (SphereFace: multiplies the angle; CosFace: subtracts a margin from the cosine value) worked, but were mathematically awkward and harder to optimize stably.

## The core idea — angular margin

Instead of multiplying an angle or subtracting from a cosine score, ArcFace **adds a fixed value directly to the angle itself** before computing the loss. This is why it's called "Arc" (angle/arc) + "Face."

Why angles specifically? Because at test time — including in our project's Stage 7 database matching — two face embeddings are compared using **cosine similarity**, which is fundamentally a measure of the angle between them. Training with an angular margin means the training objective and the real-world usage objective are the exact same kind of measurement, with nothing lost in translation.

## Why this actually pushes same-identity faces together and different identities apart

This is the part worth understanding slowly, since it's the whole point of the paper:

1. **Every identity has a "center" direction.** During training, each identity in the dataset is assigned a learnable center point (a direction) on the embedding sphere — think of it as that identity's "ideal" representative face.

2. **The margin makes the network's job artificially harder.** Normally, the network's score for "is this the right identity?" is based on how small the angle is between the face and its own center — smaller angle, higher score. ArcFace adds an extra fixed angle (`m`) on top of the real angle *before* computing that score. So even if a face is already reasonably close to its center, the network is told "not close enough yet."

3. **To fix that penalty, the network pushes the embedding even closer to its own class center than it would under ordinary softmax training.** Since this happens for every single training image, across every identity, all same-identity faces get pulled tighter and tighter toward one shared point, over many training steps. This is what creates **tight, compact clusters per person**.

4. **The identity centers themselves are also being trained — and they get pushed apart.** Because the loss is computed relative to *all* identities at once (softmax compares the correct identity's score against every other identity's score), tightening each identity's cluster while keeping the margin requirement satisfied for every identity simultaneously naturally forces the different identity centers to spread further apart from each other. Two centers that are too close would make it very easy to accidentally satisfy another identity's margin — so the training signal pushes them apart to keep telling identities cleanly apart.

5. **A simple physical analogy:** imagine several groups of marbles of different colors, all inside the same bowl. If you use a magnet to pull each color's marbles into a tighter and tighter ball, the *empty space between different-colored groups* automatically grows — you don't have to separately push the groups apart, tightening each group does it for you. That's effectively what's happening in the embedding sphere: tightening intra-identity clusters is what creates inter-identity separation.

6. **Net result after training on millions of faces across thousands of identities:** each identity forms a small, tight cluster on the sphere, and there is now a "buffer zone" (worth roughly the margin `m`) between the decision boundaries of any two identities — so faces that are slightly blurry, poorly lit, or at an odd angle are still more likely to land on the correct side of the boundary, because that boundary now has breathing room instead of sitting right at the edge of each cluster.

## Method summary

- Extract an embedding from a face using a CNN backbone (e.g., ResNet).
- Normalize both the embedding and each identity's center vector to length 1 — this puts everything on the surface of a hypersphere, so only *direction* (angle) matters, not magnitude.
- Compute the angle between the embedding and its true identity's center.
- Add the fixed margin `m` to that angle.
- Convert back to cosine, scale, and feed into the standard softmax loss.
- Backpropagate as usual — the extra margin is what does all the work described above.

## Key results

- Outperformed prior state-of-the-art (softmax, SphereFace, CosFace) across major face-recognition benchmarks (LFW, YTF, MegaFace, and others) at the time of publication.
- Simpler to implement and more stable to train than the margin tricks it replaced.
- Became the default baseline that essentially all later face-recognition papers compare against — including the other three papers in this reading list (CurricularFace, MagFace, AdaFace).

## Relevance to our FYP (Stage 5 – face recognition)

- This is the baseline option listed for Stage 5 in our pipeline; AdaFace (the recommended option) is a direct descendant of this same angular-margin idea, adapted for low-quality/surveillance faces.
- Understanding the angular-margin mechanism here is what makes CurricularFace (adjusts *how much* margin to apply as training progresses), MagFace (uses embedding length as a quality signal), and AdaFace (uses that quality signal to adjust the margin per sample) click quickly — each is one additional idea layered on top of exactly this mechanism.
- Directly relevant to Stage 7 (database matching), since that stage uses the same cosine-similarity comparison this training method is built around.
