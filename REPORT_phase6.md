# REPORT — Phase 6: core video pipeline and tracker validation

**Status:** in progress. This file starts with the pre-registered protocol of the single-frame identification baseline. It was committed before any baseline number was computed (see the git history of this file). Plan: `research/Sprint_Notes/Phase6_core_pipeline_plan.md`.

## 0. Pre-registered protocol: single-frame identification baseline on ChokePoint
Question: how well does one frame, identified on its own, recognise a person on ChokePoint? This is the baseline that Phase 7's per-track aggregation must beat. No hyperparameter is tuned, so there is no tuning split.

1. **Detections.** The Phase 5 detector (SCRFD, `full1280`, official preprocessing) on every frame; faces with score 0.5 or higher are aligned and embedded (`configs/video.yaml`, `src/video/pipeline.py`). Frames are never subsampled.
2. **Groups.** The official ChokePoint groups G1 and G2 (`DATASETS.md`, section "ChokePoint protocol"). A clip belongs to a group if its name without the `.1` or `.2` suffix is in the group's sequence list. The crowded sequences P2E_S5 and P2L_S5 are in neither group and are not used.
3. **Matching a detection to an identity.** A detection is assigned to a ground-truth identity if its box overlaps that identity's ground-truth box in the same frame with IoU 0.5 or more (ground-truth boxes come from the fitted eye-to-box rule). One-to-one, highest IoU first. Detections that match no ground-truth box (false positives) are not used for identification and are counted and reported.
4. **Gallery (enrolment).** One group enrols and the other probes, in both directions (G1 enrols and G2 probes, then the roles swap). The template of an identity is the mean of the L2-normalised embeddings of all its matched detections in the enrolling group (`survface_openset` in `src/eval_protocols.py` builds it as a plain mean of normalised embeddings, and normalises it).
5. **Cameras.** Primary condition: single camera (SC), the camera listed for each sequence in the protocol tables, for both enrolment and probing. Secondary condition: multiple cameras (MC), all three cameras of the same sessions.
6. **Probes.** Every matched detection in the probing group is one probe. Frames of one pass are strongly correlated, so all uncertainty is computed over identities, not over frames.
7. **Metrics.**
   - **Closed set:** Rank-1 over the probes of enrolled identities, with all identities enrolled.
   - **Open set:** 20% of the identities (seeded, `numpy.random.default_rng(42)`, drawn once from the sorted identity list) are left out of the gallery; their probes are the unmated probes, and probes of the enrolled identities are the mated probes. TPIR@FPIR=0.3 with `survface_openset` (the same arithmetic as the Phase 1 and 3 open-set protocol).
   - **Reported:** the two directions separately and their mean, for SC and MC.
   - **CI:** cluster bootstrap over identities, 1,000 draws, seed 42 (identities are resampled with replacement, with all their probes), each direction, then averaged.
8. **Recognizers.** AdaFace is the primary row. MagFace and CurricularFace are added as rows. ArcFace is not used.
9. **Failure handling.** An identity with no matched detection in the enrolling group cannot be enrolled; its probes are dropped from the closed set and treated as unmated in the open set. The numbers of dropped identities and probes are reported.
10. **Device.** Lab PC, RTX 4000 Ada, for every number.
