# REPORT — Phase 6: core video pipeline and tracker validation

**Status:** results complete except the open items in section 10 (see section 1). This file starts with the pre-registered protocol of the single-frame identification baseline. It was committed before any baseline number was computed (see the git history of this file). Plan: `research/Sprint_Notes/Phase6_core_pipeline_plan.md`.

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

## 0b. Pre-registered addendum: Phase 7 benchmark protocols (written after the §0 baseline, before any number below is computed)
**Why.** The §0 baseline is at ceiling: AdaFace, MagFace and CurricularFace all reach Rank-1 of 0.9987 to 0.9998 (SC and MC), and the smallest face-height bin present (32 to 48 px, 5.8% of probes) is at 0.9971. ChokePoint has no probes under 32 px (median 81 px, 5th percentile 48 px by the eye-to-box rule). Phase 7's aggregation cannot be judged against a baseline with no room to improve. Decisions taken by Abser on 2026-11-12: run B, D and E as the Phase 7 benchmark; run C as an audited study in parallel with Phase 7. Nothing in this section changes §0 or its numbers.

**Common to B, D, E.** Same detections, matching (IoU 0.5), groups (G1/G2, both directions), SC and MC conditions, left-out identities (the same six, rng 42), metrics (closed-set Rank-1, open-set TPIR20) and cluster bootstrap over identities (1,000 draws, seed 42) as §0. No parameter is tuned. Lab PC, RTX 4000 Ada.

**B. Thin gallery (single-frame and five-frame enrolment).** The template of an identity is the mean of k randomly chosen matched embeddings of that identity in the enrolling group, k = 1 and k = 5 (all frames of the identity if it has fewer than k). 20 independent draws of the frames (`numpy.random.default_rng(42)`, draw index r seeds the r-th draw). Probes are unchanged. Reported: the mean over the 20 draws, and the cluster-bootstrap CI over identities computed on the draw-averaged metric; the spread across draws (min, max) is reported next to it. Recognizers: AdaFace, MagFace, CurricularFace (embeddings are cached; no new GPU work).

**D. Stricter operating points.** For every row of §0 and B and E, TPIR20 is also read at FPIR = 0.1 and FPIR = 0.01 (§0 reports 0.3). Same scores, same bootstrap. The 6 left-out identities give wide CIs at 0.01; this is reported as it comes, with the number of unmated probes. Read-out only: it adds no run.

**E. Degraded probes.** Enrolment stays at native quality. Each probe is the aligned 112 x 112 crop (`norm_crop`, as the pipeline), downsampled to s x s with `cv2.INTER_AREA` and upsampled back to 112 x 112 with bilinear interpolation, for s in {56, 32, 24, 16, 12}; s = 112 is the native row already in §0. No blur, noise or compression is added, so this is a controlled resolution loss, not real surveillance degradation; this limitation is stated wherever E is quoted. To bound the run, probes are subsampled before degradation: every 5th matched detection per clip in frame order (deterministic), the same subset for every s. Recognizer: AdaFace (primary) for all s; MagFace and CurricularFace are not run for E. Reported per s: Rank-1, TPIR20 at FPIR 0.3, 0.1, 0.01, the number of probes, and the mean MagFace raw norm of the degraded probes as a quality read-out (the pipeline already computes it).

**C. Unmatched detections as unmated probes (audited, parallel to Phase 7).** The detections that match no ground-truth box (56,571 at score 0.5 and above) contain real faces without an annotation and reflections in the glass door, so their identity labels are unknown. C therefore needs a manual audit before it can support any claim: a seeded sample of unmatched detections is labelled by a team member as {not a face / reflection, unlabelled enrolled person, unlabelled person not enrolled, unclear}, from crops shown without the recogniser's prediction. The sample size, the labelling sheet and the analysis plan are written in a separate pre-registration note before the labelling starts. Until the audit is done, no C number is reported as a result; §1 to §N below do not use C.

**Status of the other open items (decisions of 2026-11-12).** Official YouTube Faces: transfer by home-PC download, upload and lab-PC download (USB unavailable); expected on the lab PC by Tuesday 2026-11-17. Appendix T2 (custom video): carried into Phase 7 as an incomplete part of Phase 6 because of the open consent questions. Appendix T1 on the Kaggle crops: complete (Colab T4, results in `outputs/video/appendix/ytf/`).

---

## 1. Summary
**Status:** results complete except the items in section 10; pending supervisor review at the end-of-W4 checkpoint. Plan: `research/Sprint_Notes/Phase6_core_pipeline_plan.md`. Sections 0 and 0b were committed before the numbers they govern.

- **Built and checked.** An end-to-end path on ChokePoint: frames, SCRFD detection (Phase 5 choice), 5-point alignment, AdaFace embedding with the MagFace raw norm as a quality read-out, ByteTrack and BoT-SORT tracking, TrackEval metrics (vendored). All five acceptance checks pass (section 3). The run covers 78 clips and 181,599 frames; four corrupt dataset JPEGs are logged, not hidden.
- **Trackers.** On the 72 ground-truth clips both trackers give HOTA about 0.49 to 0.50 and IDF1 about 0.70 to 0.71; ID switches are 3 (ByteTrack) and 4 (BoT-SORT) in total. The two trackers are within each other's confidence intervals on every metric, so this report does not name a winner (section 4).
- **False positives are an annotation and scene effect, not detector quality.** 56,571 detections match no ground-truth box. In the frames we looked at, they are faces of people the eye annotation does not label and the reflection of the subject in a glass door. MOTA and DetA therefore understate the detector and are not reported as a quality score here (section 4).
- **The single-frame baseline is at ceiling.** All three recognizers reach Rank-1 of 0.9987 to 0.9998 on the pre-registered protocol (section 5). A baseline with no room to improve cannot be used to judge Phase 7's aggregation, so the benchmark for Phase 7 was redefined before any further number (section 0b; results in section 6).
- **Detector mode finding (Appendix T1).** On cropped YouTube Faces frames `full1280` finds the face in 5.0% of frames and `full640` in 98.1%, because `full1280` upscales faces beyond what SCRFD detects. ChokePoint is not affected (faces 48 to 174 px at 800 x 600), but footage with large faces would be (section 8).

## 2. What was built
`src/video/` : `io.py` (image sequences and video files; unreadable frames are logged and kept in the sequence), `embed.py` (alignment, recognizers, MagFace norm, luminance, Laplacian variance), `track.py` (one adapter over ByteTrack and BoT-SORT, 5-column input), `pipeline.py` (per clip: detections.npz, tracks.csv, embed.npz; resumable, atomic writes), `mot_eval.py` and `vendor/trackeval/` (HOTA, CLEAR, Identity; upstream commit 12c8791b303e0a0b50f753af204249e622d0281a, numpy-2 patches in `VENDOR.md`). `src/recognizer.py` gained `get_embedding(..., return_raw=True)`; the default output is unchanged. Scripts: `run_video.py`, `video_tracker_eval.py`, `video_single_frame.py`, `video_phase7_bench.py`, `video_pipeline_check.py`, `video_fp_overlay.py`, `video_timings.py`, `video_appendix_ytf.py`, `phase6_report_tables.py`. Configuration: `configs/video.yaml` (`pipeline`, `tracking`, `recognizers`). Two changes to the plan were made before any run and are recorded in the config: ChokePoint has 72 ground-truth clips (not 54), and the trackers receive detections from score 0.1 (not 0.5) so that ByteTrack's second association has its low-score detections; faces are embedded from score 0.5.

## 3. Acceptance checks
1. Identity (ground-truth boxes as detections): IDF1 >= 0.99 for both trackers on all 72 clips (`outputs/video/tracker_eval/gt_check.json`).
2. Embedding: the video path and the Phase 1 aligned path agree (cosine > 0.9999 on 100 seeded faces), and the default `get_embedding` output is unchanged (`outputs/video/embed_check.json`).
3. TrackEval: the vendored metrics reproduce the hand-built toy case with known values (`outputs/video/trackeval_check.json`).
4. Determinism: one clip rerun end to end is byte-identical (detections, keypoints, tracks.csv, embeddings, raw norm, luminance, Laplacian variance). One clip, one machine and software stack: it does not show identical results on another machine.
5. Coverage: every listed frame of every clip has an entry; four unreadable images are logged (P2E_S1_C2.1 frame 359; P2E_S1_C3.2 frames 33, 211, 399).
Details: `outputs/video/phase6_checks.json` (tables below).

## 4. Tracker validation on ChokePoint
Ground truth: eye labels turned into boxes with the Phase 5 rule, identity = person id (`video_tracker_eval.py`). Both trackers run on the same detections (re-ID off, no camera-motion compensation, vendored defaults, 30 fps).
- **Result.** See the tracker table. Identity metrics are the useful ones here: AssA and IDF1 and the ID-switch count. The ground-truth tracks are short and clean and only 906 ground-truth faces are missed (under 1%), so ID switches are rare for both trackers; this says the footage is easy for association, not that the trackers are proven on hard footage.
- **No winner.** The point differences between ByteTrack and BoT-SORT are small and the bootstrap CIs overlap on every metric.
- **The false-positive count (about 57,000) does not measure the detector.** It is concentrated in the portal-2 clips, where several people walk through together (P2E 18,334 and P2L 28,756 of 57,614 ByteTrack FP, against FN of 45 and 18). The median score of these detections is 0.8 and 86% are at or above 0.5. Scoring only detections at 0.5 and above removes about 4,500 FP and moves MOTA from 0.19 to 0.25 (second row of the tracker table), so the extras are mostly confident detections. Overlays of six seeded frames of P2E_S3_C1.1 (`outputs/video/overlays_fp/`, local only) were opened for two frames before this text was written: they show (a) the reflection of the walking person in the glass door, detected at about 0.7, and (b) the subject's own face detected at about 0.7 in frames where the annotation has no box (no ground-truth row within 10 frames). 28% of the at-or-above-0.5 detections in that clip are centred left of x = 230, where the door is. Two frames of one clip is a small sample; the split between reflections and unannotated faces is not quantified. This is the question that study C (section 0b) is meant to answer with a manual audit.
- **Consequence for reporting.** MOTA, DetA and the FP column are shown in the table but are not quoted as detector or tracker quality; HOTA also includes DetA and is shown with this caveat. A per-camera ignore region for the glass door was considered and not used (it needs a pre-registered rule per camera); Abser chose reporting with the caveat on 2026-11-12.

## 5. Single-frame identification baseline (section 0, run as written)
- **Counts.** 29 identities, 6 left out of the gallery in the open set (3, 13, 14, 19, 21, 26; rng 42). 71,344 of 72,289 ground-truth faces matched to a detection at IoU 0.5; no identity and no probe was dropped. SC probes: about 14,000 per direction; MC: about 34,000 to 37,000.
- **Result.** See the baseline table: Rank-1 0.9987 to 0.9998 and TPIR20@FPIR 0.3 of 0.9986 to 0.9998 for the three recognizers, SC above MC, AdaFace first (the CIs overlap).
- **Controls.** With the gallery labels shuffled Rank-1 is 0.0 (chance 0.035); the G1 and G2 clips do not overlap; single camera G1 to G2 had 1 error in 14,179 probes.
- **Why it is this high.** ChokePoint faces are near frontal with a median height of 81 px (5th percentile 48 px, none under 32 px) and the same 29 people appear in both groups. By face height (AdaFace, SC): 32 to 48 px 0.9971 (5.8% of probes), 48 to 64 px 0.9998, 64 to 96 px 0.9999, 96 px and up 1.0000 (an exploratory cut, not pre-registered).
- **What it means.** The baseline cannot be beaten in any meaningful way on this protocol. It still serves as a check that the pipeline preserves identity end to end, and as the reference row against the harder protocols in section 6.

## 6. Phase 7 benchmark: B, D and E (section 0b)
Run as written in section 0b (`scripts/video_phase7_bench.py`; tables in section 12). Same detections, matching, groups, left-out identities and cluster bootstrap over identities as section 0.

- **D (stricter operating points).** Reading the baseline at FPIR 0.1 and 0.01 does not create room by itself: at FPIR 0.01 TPIR20 is still 0.974 to 0.988 (AdaFace, MagFace, CurricularFace; SC and MC). With 6 left-out identities the 0.01 point is the least certain number in the report.
- **B (thin gallery).** One frame per identity lowers the baseline a little and shows a spread across draws. Rank-1 with k = 1 is 0.9951 (SC) and 0.9939 (MC) for AdaFace, 0.9930 and 0.9891 for MagFace and 0.9904 and 0.9862 for CurricularFace; the worst single draw is 0.952 (CurricularFace, MC). TPIR20 at FPIR 0.01 falls to 0.87 to 0.94. With five frames per identity the numbers are back to 0.997 to 0.999. So a single enrolment frame costs about 0.5 to 1.3 points of Rank-1 and about 5 to 11 points of TPIR at FPIR 0.01, and AdaFace is the most robust of the three, but the headroom is small on this footage (frontal faces, 29 people). The CI of the mean is over identities and the spread over the 20 draws is shown next to it.
- **E (degraded probes, AdaFace).** The probe side is the only knob that moves the result a lot. Rank-1 stays at 0.9993 to 1.0000 down to 24 px (SC), then falls: 0.9936 at 16 px and **0.8387 at 12 px** (SC; MC 0.9946 and 0.8539). TPIR20 at FPIR 0.01 falls from 0.99 at 56 px to 0.90 at 16 px and 0.37 at 12 px. The mean MagFace norm of the probes falls steadily with the side (25.98 at 56 px to 22.09 at 12 px, SC), so the quality read-out tracks the degradation. The two directions agree (12 px: 0.850 and 0.827 SC). This is a synthetic resolution loss (area downsampling, bilinear upsampling, no blur, noise or compression) on faces that were detected at their native size; it says where single-frame recognition fails for this recognizer, not how a real 12 px face in a surveillance frame would behave.
- **What Phase 7 should use.** The informative region is probes at 16 px and 12 px (and the thin gallery with k = 1). At 12 px a single frame is wrong for about 16% of probes, so per-track aggregation has real room to improve there; at 24 px and above it has none. Phase 7's primary comparison should therefore be the single-frame rows against aggregated rows at 16 px and 12 px, plus B with k = 1, each pre-registered before it is run. C (unmatched detections as unmated probes) is not part of these numbers; it waits for its audit.

## 7. Stage timings
See the timings table. Lab PC, RTX 4000 Ada, ONNX Runtime CUDA provider, 800 x 600 frames, `full1280`. Detect (about 80 ms per frame pooled) is the dominant stage and includes JPEG decoding; embedding (about 55 ms per face pooled) includes decoding, alignment, flip-sum for AdaFace and the MagFace norm pass; tracking is under 1 ms per frame for both trackers together. The lab PC is shared, and the clip-level spread (detect 50 to 105 ms between the 10th and 90th percentile) is real. These are not real-time figures for a 30 fps stream at 1280 px: detection alone is about 12 fps.

## 8. Appendix T: detector tests beyond WIDER FACE
**T1. YouTube Faces (Kaggle package "YouTube Faces with Facial Keypoints", Colab T4).** The held copy is a processed subset (2,194 videos, 828 people, cropped frames of about 120 to 300 px with a face box and landmarks), not the original full-frame database. The test used a seeded sample (rng 42): 500 people, one video each, 10 evenly spaced frames, 5,000 frames, with the dataset's tight face boxes as ground truth. See the T1 table. `full640` finds the face (IoU >= 0.5) in 98.1% (CI 97.1 to 98.9) and `full1280` in 5.0% (3.4 to 6.7). `full1280` upscales a 90 px face to about 550 px and SCRFD does not detect faces that large: its recall falls from 0.151 at 32 to 64 px to 0.000 above 128 px. The effect is the same as the lower recall of `full1280` on faces above 64 px in the Phase 5 WIDER table. This test says nothing about small faces in a wide scene (the crops are tight) and the ground truth is the dataset's own labelling. Extra detections per frame are not false positives (the dataset labels only the subject). Numbers are from one Colab T4 and are not pooled with the lab-PC numbers.
- **Consequence.** The Phase 5 choice (`full1280`) is right for ChokePoint-like footage and wrong for footage with large faces (close-up video, webcams). Phase 7 should decide whether the detector mode follows the frame or face size (for example `full640` when the frame already shows large faces) or stays fixed; this is raised as decision 3 in section 10.
- **Official YouTube Faces.** Not yet held. The official host is unreachable from the lab network and VPNs are not allowed, so the database is being downloaded on a home PC, uploaded and downloaded again (USB unavailable); expected on the lab PC by Tuesday 2026-11-17. The full-frame T1 test, and the Phase 7 verification protocol on YouTube Faces (3,425 videos, 1,595 people, 5,000 pairs in 10 folds, to be verified against these counts on arrival), wait for it.

**T2. Custom video of consenting volunteers.** Not done. Consent questions are open (written supervisor approval, signed consent forms); Abser decided on 2026-11-12 to carry T2 into Phase 7 as an incomplete part of Phase 6. No recording has been made. The labelling tool (`scripts/video_label_boxes.py`) is not written.

## 9. Limitations
- One dataset (ChokePoint), one recording site, 29 identities, frontal faces: the baseline saturates and the tracking is easy. Conclusions about harder footage need other data.
- Ground-truth boxes come from the fitted eye-to-box rule, not from annotated boxes; the chin is cut off on strongly turned heads (Phase 5). Faces without an eye annotation count as false positives (section 4).
- The tracker comparison has no winner and its CIs are over 72 clips of 29 identities; ID switches are very rare (3 and 4), so the tracker choice is not settled by this data.
- Check 4 (determinism) covers one clip on one machine.
- Frames are 800 x 600; no real-time claim.
- B and E are designed to add difficulty, not to model a real watch list or real degradation (E is a controlled resolution loss; no blur, noise or compression).
- Appendix T1 uses cropped frames and a 5,000-frame sample; it is not the official full-frame benchmark.
- The Colab T4 and the lab PC give different timings; they are never pooled.

## 10. Decisions and open items for the checkpoint
1. **Phase 7 benchmark.** B, D and E as the benchmark (decided 2026-11-12); C as an audited study in parallel with Phase 7. The audit protocol (sample size, labelling sheet, analysis) is to be written and committed before labelling starts. Who labels and when is open.
2. **Official YouTube Faces** by Tuesday 2026-11-17; then the full-frame T1 and the YTF verification protocol. Record source, licence, SHA-256 and the verified counts in `DATASETS.md` on arrival.
3. **Detector mode for large faces.** Fixed `full1280` (current) or size-dependent. To be decided with the YouTube Faces numbers.
4. **Appendix T2** (custom video): carried into Phase 7, gated on written supervisor approval and signed consent.
5. **Tracker choice** for Phase 7: no winner on this data. Keep both through Phase 7, or choose on cost (both under 1 ms per frame) and simplicity.
6. **Third dataset request** (name still to be identified); FANVID and SCface replies awaited.

## 11. Future work (FYP II): VLM integration
No implementation in FYP I. Where a vision-language model could add value: (1) description search when the face is unusable ("person in a red jacket near door 2"), text-to-person retrieval over track crops with CLIP/SigLIP-style embeddings, which complements face identity on tiny faces; (2) natural-language queries over the track database, with a model turning a question into index filters (time, camera, attributes, identity); (3) per-track summaries and alert explanations for an operator, not per frame, to keep latency down. **Hard constraint from the SR lesson:** generative models hallucinate, which is why SR was closed. VLM output is never identity evidence; identity stays with the face recognizer and its scores; VLM text is descriptive, shown with its source crops and checked against the detector and tracker output. Candidates are open-weight, inference-only on the lab PC (20 GB, quantised if needed), for example small Qwen-VL or InternVL-family models, Florence-2 for captioning and grounding, and CLIP/SigLIP for retrieval; versions and licences are to be checked at FYP II start. Evaluation: text-based person search benchmarks (CUHK-PEDES, ICFG-PEDES, RSTPReid, subject to access), attribute accuracy on our own consented clips, hallucination rate judged against the tracker ground truth, latency per track. Risks: privacy (attribute search is surveillance-sensitive), demographic bias in descriptions, compute, and the same proof standard as Phases 3 to 5 (pre-registered tests, every number from a script).

## 12. Result tables
<!-- TABLES:BEGIN -->

#### Tracker validation, pooled over the 72 ground-truth clips (cluster bootstrap over clips, 1,000 draws, seed 42)

| tracker | scoring | HOTA | AssA | IDF1 | DetA | MOTA | IDSW (total) | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| bytetrack | all detections tracked (floor 0.1) | 0.497 (0.459-0.535) | 0.571 (0.532-0.608) | 0.709 (0.669-0.751) | 0.437 (0.399-0.478) | 0.190 (0.027-0.346) | 3 | 906 | 57614 |
| botsort | all detections tracked (floor 0.1) | 0.491 (0.453-0.529) | 0.564 (0.526-0.600) | 0.703 (0.663-0.745) | 0.431 (0.393-0.471) | 0.165 (-0.001-0.322) | 4 | 871 | 59494 |
| bytetrack | scored on detections >= 0.5 only | 0.510 (0.471-0.550) | 0.582 (0.541-0.621) | 0.725 (0.685-0.767) | 0.452 (0.413-0.494) | 0.251 (0.096-0.400) | 3 | 980 | 53129 |
| botsort | scored on detections >= 0.5 only | 0.504 (0.465-0.543) | 0.575 (0.536-0.613) | 0.718 (0.678-0.760) | 0.445 (0.407-0.487) | 0.226 (0.068-0.376) | 4 | 945 | 54991 |

#### ByteTrack by group (mean over clips; FP and FN are sums)

| group | clips | HOTA | IDF1 | DetA | FN | FP | IDSW |
|---|---|---|---|---|---|---|---|
| P1E | 12 | 0.605 | 0.839 | 0.58 | 500.0 | 3507.0 | 2.0 |
| P1L | 12 | 0.609 | 0.842 | 0.598 | 343.0 | 7017.0 | 0.0 |
| P2E | 24 | 0.473 | 0.703 | 0.445 | 45.0 | 18334.0 | 1.0 |
| P2L | 24 | 0.373 | 0.575 | 0.332 | 18.0 | 28756.0 | 0.0 |

#### Acceptance checks 4 and 5

- Coverage: 78 clips (72 ground-truth, 6 crowded), 181599 frames, failures: 0; unreadable frames logged: {'P2E_S1_C2.1': [359], 'P2E_S1_C3.2': [33, 211, 399]}.
- Determinism (P2L_S4_C3.2): byte-identical = True (detections, keypoints, tracks.csv, AdaFace embeddings, raw norm, luminance, Laplacian variance).

#### Single-frame baseline (pre-registered section 0; mean of the two directions; CI is the cluster bootstrap over identities)

| recognizer | condition | Rank-1 (95% CI) | TPIR20@FPIR0.3 (95% CI) |
|---|---|---|---|
| adaface | SC | 0.9998 (0.9995-1.0000) | 0.9998 (0.9979-1.0000) |
| adaface | MC | 0.9995 (0.9991-0.9999) | 0.9992 (0.9968-0.9999) |
| magface | SC | 0.9997 (0.9994-0.9999) | 0.9993 (0.9948-0.9999) |
| magface | MC | 0.9992 (0.9985-0.9997) | 0.9986 (0.9934-0.9996) |
| curricularface | SC | 0.9992 (0.9981-1.0000) | 0.9993 (0.9946-0.9999) |
| curricularface | MC | 0.9987 (0.9976-0.9997) | 0.9988 (0.9926-0.9998) |

#### D: the baseline read at stricter FPIR (mean of the two directions)

| recognizer | condition | TPIR20@0.3 | @0.1 | @0.01 |
|---|---|---|---|---|
| adaface | SC | 0.9998 | 0.9989 | 0.9883 |
| adaface | MC | 0.9992 | 0.9977 | 0.9863 |
| magface | SC | 0.9993 | 0.9962 | 0.9789 |
| magface | MC | 0.9986 | 0.9954 | 0.9743 |
| curricularface | SC | 0.9993 | 0.9970 | 0.9785 |
| curricularface | MC | 0.9988 | 0.9957 | 0.9749 |

#### B: thin gallery (k frames per identity, 20 draws)

| recognizer | cond | k | Rank-1 | TPIR20@0.3 | @0.1 | @0.01 | Rank-1 over draws (min-max, mean of directions) |
|---|---|---|---|---|---|---|---|
| adaface | SC | 1 | 0.9951 | 0.9975 | 0.9852 | 0.9382 | 0.9814-0.9994 |
| adaface | SC | 5 | 0.9993 | 0.9990 | 0.9969 | 0.9852 | 0.9985-0.9998 |
| adaface | MC | 1 | 0.9939 | 0.9956 | 0.9788 | 0.9183 | 0.9731-0.9989 |
| adaface | MC | 5 | 0.9986 | 0.9981 | 0.9953 | 0.9826 | 0.9972-0.9994 |
| magface | SC | 1 | 0.9930 | 0.9936 | 0.9775 | 0.9003 | 0.9825-0.9981 |
| magface | SC | 5 | 0.9987 | 0.9980 | 0.9943 | 0.9739 | 0.9970-0.9996 |
| magface | MC | 1 | 0.9891 | 0.9898 | 0.9663 | 0.8660 | 0.9571-0.9976 |
| magface | MC | 5 | 0.9981 | 0.9972 | 0.9925 | 0.9680 | 0.9963-0.9990 |
| curricularface | SC | 1 | 0.9904 | 0.9950 | 0.9765 | 0.9073 | 0.9756-0.9976 |
| curricularface | SC | 5 | 0.9974 | 0.9978 | 0.9937 | 0.9747 | 0.9946-0.9994 |
| curricularface | MC | 1 | 0.9861 | 0.9904 | 0.9637 | 0.8698 | 0.9519-0.9969 |
| curricularface | MC | 5 | 0.9966 | 0.9969 | 0.9919 | 0.9693 | 0.9923-0.9982 |

#### E: degraded probes (AdaFace; enrolment native; every 5th matched detection)

| cond | probe side (px) | Rank-1 | TPIR20@0.3 | @0.1 | @0.01 | mean MagFace norm |
|---|---|---|---|---|---|---|
| SC | 56 | 1.0000 | 0.9998 | 0.9991 | 0.9910 | 25.98 |
| SC | 32 | 1.0000 | 0.9998 | 0.9986 | 0.9873 | 24.17 |
| SC | 24 | 0.9997 | 0.9993 | 0.9986 | 0.9728 | 23.12 |
| SC | 16 | 0.9936 | 0.9914 | 0.9797 | 0.8990 | 22.57 |
| SC | 12 | 0.8387 | 0.7858 | 0.6329 | 0.3675 | 22.09 |
| SC | 112 | 0.9998 | 0.9998 | 0.9991 | 0.9889 |  |
| MC | 56 | 0.9997 | 0.9993 | 0.9984 | 0.9890 | 25.68 |
| MC | 32 | 0.9995 | 0.9991 | 0.9979 | 0.9842 | 24.02 |
| MC | 24 | 0.9993 | 0.9989 | 0.9980 | 0.9702 | 23.04 |
| MC | 16 | 0.9946 | 0.9917 | 0.9805 | 0.9187 | 22.42 |
| MC | 12 | 0.8539 | 0.8070 | 0.6514 | 0.3605 | 21.92 |
| MC | 112 | 0.9995 | 0.9993 | 0.9982 | 0.9869 |  |

#### Stage timings (lab PC, RTX 4000 Ada, 800 x 600; detect and embed include JPEG decoding; track is both trackers together)

| stage | unit | n_clips | n_units | median_ms_per_unit | pooled_ms_per_unit | p10_clip | p90_clip |
|---|---|---|---|---|---|---|---|
| detect | frame | 78 | 181599 | 77.98 | 80.37 | 49.59 | 104.7 |
| track | frame | 78 | 181599 | 0.39 | 0.53 | 0.31 | 0.86 |
| embed | face | 78 | 148658 | 51.74 | 55.31 | 31.91 | 108.81 |

#### Appendix T1: SCRFD on the YouTube Faces Kaggle crops (Colab T4)

| mode | face height (px) | n | recall IoU 0.5 (95% CI) | recall IoU 0.3 | centre hit | extra/frame | median ms |
|---|---|---|---|---|---|---|---|
| full1280 | all | 5000 | 0.050 (0.034-0.067) | 0.050 | 0.050 | 0.030 | 66.8 |
| full1280 | 32-64 | 1024 | 0.151 (0.094-0.211) | 0.153 | 0.153 | 0.006 | 65.1 |
| full1280 | 64-96 | 1965 | 0.044 (0.023-0.067) | 0.045 | 0.045 | 0.034 | 66.7 |
| full1280 | 96-128 | 997 | 0.006 (0.001-0.013) | 0.006 | 0.006 | 0.032 | 68.9 |
| full1280 | >128 | 1014 | 0.000 (0.000-0.000) | 0.000 | 0.000 | 0.045 | 67.5 |
| full640 | all | 5000 | 0.981 (0.971-0.989) | 0.995 | 0.995 | 0.026 | 17.1 |
| full640 | 32-64 | 1024 | 0.977 (0.964-0.987) | 1.000 | 1.000 | 0.001 | 17.0 |
| full640 | 64-96 | 1965 | 0.988 (0.977-0.996) | 1.000 | 1.000 | 0.043 | 17.2 |
| full640 | 96-128 | 997 | 0.989 (0.978-0.997) | 1.000 | 1.000 | 0.016 | 17.2 |
| full640 | >128 | 1014 | 0.964 (0.929-0.991) | 0.974 | 0.974 | 0.031 | 17.1 |

<!-- TABLES:END -->

---
**STOP.** Phase 6 ends here for the end-of-W4 checkpoint. Phase 7 starts after the decisions in section 10.
