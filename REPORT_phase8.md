# REPORT — Phase 8: held-out open set on MEVID, frame-quality reject test, final tracker choice

**Status:** in progress. This file starts with the pre-registered protocol (section 0). It is committed on branch `phase_8` before any embedding or any Phase 8 number exists (see the git history of this file). Plan: `research/Sprint_Notes/Phase8_plan.md` (approved 2026-11-24, Abser). **Section 0 uses the settings recommended in the plan (Abser, 2026-11-25, relayed by Umar).**

## 0. Pre-registered protocol
Rules for the whole phase: inference only; every number comes from a script output; seed 42; failures are logged; one device for all numbers (lab PC, RTX 4000 Ada); no MEVID image or embedding is ever committed; no recognition score exists before this section is committed.

### 0.1 Data and counts (facts from the intake, recorded before any score)
- **MEVID** (consented actors; CC BY 4.0). **Usage terms: cite Davila et al., "MEVID", WACV 2023, arXiv 2211.04656** (Abser, 2026-11-24); the report cites it and `DATASETS.md` records it.
- Held from the probe (2026-11-23): annotations, actor check-in photos (2,898 files, 172 persons) and the test body chips (1,186,251 chips, 1,754 tracklets, 54 persons, 113 outfits, 316 query and 1,438 gallery tracklets). Phase 8 adds the train package (`mevid-v1-bbox-train.tgz`, 32,684,603,246 bytes), so all 158 identities (104 train + 54 test) are used.
- **Count note found before any score (explained):** the annotation files give 6,339 train and 1,754 test tracklets, **8,093 rows in total**, while the plan's acceptance check 1 states 8,092. The extra row is **one empty train tracklet** (`tr01889`, index 1,889 of `track_train_info.txt`: start 1,128,477, end 1,128,476, person 78, outfit 30, camera 341, zero chips); the other 8,092 tracklets all have at least one chip, which equals the README. The empty row is kept in the index, skipped by detection and never eligible; check 1 reports both numbers (8,093 rows, 8,092 with chips).
- Probe results that Part 0 relies on (from `outputs/video/mevid_probe/`, detection only): test persons with a front-photo face of at least 48 / 56 / 64 px: 52 / 44 / 28 of 54 (detector at size 640); all 158 front photos of test persons contain a detected face; photo height median 500 px for test persons and 1,904 px for the other persons; 14 persons have photos but no tracklet; probe body-chip faces: median height 32.5 px.

### 0.2 Decisions M1 to M12 (the plan's Part 0, confirmed by Abser on 2026-11-25 as the recommended settings)
| # | Decision | Setting |
|---|---|---|
| M1 | Identities | All 158 (train package plus test); nothing is trained, so the split does not matter |
| M2 | Gallery photos | Front view only (`f`, `F` in the file name) |
| M3 | Gallery face-size bar | **48 px** native face height; a 64 px sensitivity row |
| M4 | Gallery detector | SCRFD `det_10g` (official preprocessing), the photo resized so that its longest side is **1280**; the largest front face with score >= 0.5 |
| M5 | Gallery template | Plain mean (R1) of the person's usable front-face embeddings, then L2-normalised |
| M6 | Extra gallery identities | The 14 persons with photos but no tracklets are enrolled too (172 against 158) |
| M7 | Probe tracklet | Up to **50 chips** evenly spaced; face rule as in the probe (score >= 0.5, box centre in the upper half of the chip, native height >= 16 px, SCRFD size 640); a tracklet with **>= 5 usable faces** is an eligible probe |
| M8 | Cameras | None excluded; results pooled and per camera; eligibility per camera reported |
| M9 | Open-set design | Among persons with both a usable gallery and an eligible tracklet: **5 rotated folds**, 20% held out per fold (seed 42); non-enrolled probes are the held-out persons' tracklets. Persons with no usable gallery photo form a **separate** non-enrolled row |
| M10 | Metrics | Closed-set Rank-1; open-set **TPIR1** at FPIR 0.3, 0.1, 0.01 (primary) and TPIR20 for continuity; cluster bootstrap over persons, 1,000 draws |
| M11 | Recognizers | AdaFace MS1MV2 (primary, the pipeline default); AdaFace WebFace4M (secondary, the CAFace backbone) |
| M12 | Detector size on chips | `det.detect(chip, size=640)`, as in the probe |

### 0.3 Exact definitions (so that nothing is decided after a number)
- **Embedding.** The Phase 6 path: 5-point alignment from the SCRFD keypoints (`norm_crop`, 112 px), AdaFace MS1MV2 with bilinear resize and flip-sum (`src/recognizer.py`), embeddings L2-normalised. Saved per face: AdaFace raw norm (unflipped), MagFace raw norm, luminance, Laplacian variance, native face height.
- **Usable gallery photo:** a front photo whose largest face (score >= 0.5, SCRFD at size 1280) has native height >= 48 px (>= 64 px in the sensitivity row). A person is **enrolled** if at least one photo is usable. Template per M5. A person without a usable photo is not enrolled.
- **Tracklet unit.** MEVID's own tracklet; the usable faces of its (up to 50) sampled chips; **eligible** if it has >= 5 usable faces. The 50 chips are `numpy.round(linspace(start, end, min(50, n)))` made unique; the 200 probe-sampled tracklets of the probe use the same rule with 20 chips only for check 2.
- **Closed set.** Gallery = all enrolled persons (no hold-out). Probes = eligible tracklets of enrolled persons. Single-frame accuracy of a tracklet = the mean over its usable faces of the frame-level correctness (cosine to every template, correct if the arg max is the true person); aggregated accuracy = correctness of the tracklet's R1 template (normalised mean of its usable-face embeddings). Rank-1 = the mean over tracklets.
- **Open set (M9, M10).** Population P = persons that are enrolled and have >= 1 eligible tracklet. P is shuffled with `numpy.random.default_rng(42)` and cut into 5 groups of near-equal size; in fold k the group k is held out. Gallery of fold k = all enrolled persons minus the held-out group (this includes enrolled persons without eligible tracklets and the 14 extra persons). Mated probes = eligible tracklets of enrolled persons of the gallery that are in P; **non-enrolled probes** = eligible tracklets of the held-out group. Similarity = cosine between L2-normalised probe template and gallery templates. For a target FPIR f, the threshold t_f is the (1 - f) quantile of the top-1 similarities of the fold's non-enrolled probes. TPIR1 at f = the share of mated probes whose top-1 gallery person is correct **and** whose top-1 similarity >= t_f; FPIR is by construction f on the held-out probes. TPIR20 as the Phase 3 `survface_openset` arithmetic (top-20 list, distance sqrt(2 - 2 cos)). Fold results are averaged. Probes: for a tracklet, the R1 template; for the single-frame reference every usable face is a probe (frame-level), as in Phases 6 and 7.
- **No-usable-photo row.** The eligible tracklets of persons without a usable photo are non-enrolled probes against the full gallery (all enrolled persons). Reported: their false-positive rate at the thresholds t_f of each fold (mean over folds), next to the held-out-group rate (which is f).
- **Bootstrap.** Cluster bootstrap over persons, 1,000 draws, seed 42: persons are resampled with replacement (with all their tracklets) within each role of each fold; every statistic is recomputed per draw.
- **Rules R1 to R6** are those of Phase 7 section 0.2 (same definitions), with the **MagFace raw norm** as quality for R2 to R4 so that they stay comparable with Phase 7; the AdaFace raw norm is used only in Part C. WebFace4M rows: the same crops embedded with the CAFace backbone (`scripts/video_caface.py`, `envs/caface`); single frame and R1 only.
- **Implementation notes fixed before any embedding.** (1) Only the **eligible** tracklets (>= 5 usable faces) are embedded, since only they are probes; the detection stage runs on every tracklet first. (2) Of the train package only the chips that are read (up to 50 evenly spaced per train tracklet, about 317,000 of 3.6 million) are extracted (`scripts/mevid_extract.py --train-needed-only`); the data read are identical to a full extraction. (3) `scripts/mevid_embed.py` stages: `index`, `check2`, `detect`, `eligibility`, `gallery` use the detector only; `embed` is the first stage that produces an embedding and refuses to run without an explicit confirmation flag. (4) The WebFace4M embedding (`scripts/video_caface.py --part mevid`) runs on the same faces in the same order after the AdaFace one.
- **Face-height bins for the secondary table:** by the tracklet's mean usable-face height: [16, 24), [24, 32), [32, 48), >= 48 px.

### 0.4 Primary hypotheses (AdaFace MS1MV2, all eligible tracklets, gallery bar 48 px)
- **H1:** the R1 tracklet Rank-1 minus the single-frame expected Rank-1, paired per tracklet, mean difference with the cluster-bootstrap 95% CI **above zero**. (Replicates Phase 7 on real low resolution.)
- **H2:** R1 minus single frame on open-set **TPIR1 at FPIR 0.1**, fold-averaged, 95% CI **above zero**.
- **Secondary (reported as they come, none selected):** the 64 px gallery bar; per camera; face-height bins; the no-usable-photo row; R2 to R6; the WebFace4M rows; TPIR1 at FPIR 0.3 and 0.01; TPIR20.

### 0.5 Part C: frame-quality reject test (Phase 7 decision 2)
- **Question:** which signal best predicts identification errors, so that the pipeline can reject or flag unreliable tracks.
- **Data:** MEVID (primary, real low resolution): the eligible tracklets of enrolled persons, closed-set R1, gallery bar 48 px. ChokePoint 12 px and 16 px (secondary, synthetic): the Phase 7 GT tracks, SC, both directions pooled, R1, from cache. On ChokePoint face height is **not** a valid signal (degraded faces keep the native detection) and neither Laplacian variance, luminance nor the AdaFace raw norm exist for the degraded faces, so only the number of usable faces, self-consistency and the MagFace norm are evaluated there.
- **Signals per tracklet** (quality is "higher is better" for every one): number of usable faces; self-consistency (the mean cosine of each face to the leave-one-out mean of the tracklet's other faces); mean face height (MEVID only); mean Laplacian variance; mean luminance (as the plan lists it, higher = better); mean AdaFace raw norm; mean MagFace raw norm (reference).
- **Exploratory extra row (outside the Bonferroni count and the verdict):** the luminance variant -|mean luminance - 128|, because both too dark and too bright faces may hurt and the plan gives no direction for luminance; reported on MEVID only and never counted as "helps".
- **Metric:** the closed-set R1 error rate as the lowest-quality fraction q of tracklets is rejected, for q on a 0.01 grid from 0 to 0.5 (ties broken by tracklet index); **AUC** = the area under that curve (trapezoid). Random rejection leaves the expected error unchanged, so its area is the overall error rate times 0.5.
- **A signal "helps"** if its area is smaller than the random-rejection area and the 95% CI of the difference excludes zero (paired cluster bootstrap over persons, 1,000 draws, **Bonferroni** over the seven signals on MEVID and the three on ChokePoint; the exploratory row is not counted).
- **Output:** the signal for the Phase 9 quality bar, or "none is reliable".

### 0.6 Part D: final tracker choice on ChokePoint
- **Face-aware association.** Both trackers get the same added cost: `cost = (1 - lambda) * (1 - IoU) + lambda * (1 - cos)`, lambda = **0.5**, fixed, not tuned, applied to a track-detection pair only when both faces pass the **gate**. **Gate (the plan's fallback, fixed now):** a detection is gated in if its detector score >= 0.5 (it has an embedding) and its native box height >= 32 px; the track side uses the track's latest gated-in embedding. The plan's fallback also names "self-consistency" but gives no threshold, and self-consistency is a tracklet-level quantity (Part C); a threshold cannot be set without data, so it is **not** used in this run. The fallback applies (the Part C signal would need MEVID results first); the Part C result informs Phase 9 and does not change this run. Implemented as an option in `src/video/track.py:TrackerAdapter`; the vendored trackers change only by a cost hook, recorded in `VENDOR.md`.
- **Conditions:** S0 to S5 of the Phase 7 stress test (`scripts/video_tracker_stress.py`, Addendum 2), each with and without the face term (lambda = 0 is the Phase 7 baseline).
- **Cost placement (fixed now).** The face term is added to the cost that each tracker uses for its first association and for its association of unconfirmed tracks: `(1 - lambda)` times the tracker's own IoU distance (score-fused, as in the vendored code) plus `lambda * (1 - cos)`; the second association (low-score detections) is unchanged. The track side keeps the embedding of its latest gated-in matched detection.
- **Metrics:** IDF1 and ID switches (vendored TrackEval) and the **downstream open-set track-identity accuracy**. Definition (the plan's wording, with the enrolment details written out): the 29 ChokePoint identities are split into 5 folds of 5 to 6 (seed 42); in fold k the held-out group is not enrolled. Gallery templates come from the enrolling group of the Phase 6 protocol (G1 enrols and G2 probes, and the reverse; SC), as the mean of the enrolled person's matched embeddings. Probes are the predicted tracks (label = majority person id of the matched detections, tracks with < 50% matched detections not scored, as Phase 7); a probe's output is its R1 template's top-1 enrolled identity if the similarity is >= t_0.1 (the threshold at FPIR 0.1 on the held-out group's tracks of the same condition, tracker and direction), otherwise "unknown". A track is correct if it is an enrolled person's track identified correctly, or a held-out person's track output as "unknown". Accuracy = the share of correct tracks, per clip, averaged over folds and directions.
- **Decision rule (fixed now):** the tracker with the higher mean track-identity accuracy over S0 to S5 **with the face term** wins if the 95% CI (cluster bootstrap over clips, 1,000 draws, seed 42) of the per-clip difference excludes zero; otherwise **ByteTrack stays the default**.
- **Caveat stated here:** the test is within ChokePoint (not held out; MEVID has no frame coordinates).

### 0.7 Part E: restoration gate (Objective 1)
No run. The gate stage is closed by the Phase 4 stop rule (0 of 3 non-leaky recognizers for every arm); the pipeline keeps the hook off by default.

### 0.8 Acceptance checks (written to `outputs/video/mevid/checks.json`)
1. Tracklet index counts equal the README (8,092 tracklets, 158 identities) and the gallery photo counts equal the probe inventory. Known before any score: 8,093 rows in the annotation files, one of them empty (`tr01889`); 8,092 tracklets have chips; both numbers are reported.
2. On the 200 probe-sampled tracklets, `mevid_embed.py` reproduces the probe's face boxes exactly (same detector and rule).
3. A tracklet of length 1 equals the single frame for every rule; frame order does not change R1 to R5.
4. Open-set folds: every person in P is held out exactly once; the gallery and the non-enrolled probes are disjoint in each fold.
5. The face term with lambda = 0 reproduces the Phase 7 S0 to S5 tracker results byte for byte.
6. Reruns of every CPU part are byte-identical.

### 0.8b Confirmation and interpretations
Abser (2026-11-25, relayed by Umar): use the settings recommended in the Phase 8 plan. Where the plan is explicit, section 0 follows it. Where the plan is silent, the following interpretations were made and are stated here before any number: (1) the direction of the luminance signal is the plan's literal one (higher = better), with the distance-to-mid-gray variant as an exploratory row outside the verdict; (2) the face-term gate is the plan's fallback without the self-consistency condition, whose threshold the plan does not give; (3) the accuracy of the downstream track identification is defined per section 0.6 (enrolling group of the Phase 6 protocol, held-out group rotated over 5 folds, threshold at FPIR 0.1 from the held-out group's tracks); (4) the quality signals available on ChokePoint are the three that exist for degraded faces; (5) only eligible tracklets are embedded, and only the needed train chips are extracted (section 0.3). Any of these can be changed by a dated amendment before the corresponding result exists, with both versions reported.

### 0.9 Limitations and risks stated in advance
- The check-in photos are probably from the recording days, so the enrolment-to-probe gap is short.
- Tracklets are MEVID's own, not produced by our tracker; the probe faces are crops from body boxes.
- At the 48 px bar fewer persons are enrolled than the 158; the 64 px row shows the sensitivity. CIs are reported as they come.
- The train photos may differ from the test inventory (no faces, other file names): the gallery rule is applied unchanged and the coverage is reported per split.
- Train download speed (0.2 to 0.3 MB/s per connection): parallel ranges, detached. Fallback: the test split alone, recorded as a deviation before any score.
- The face term could change tracker behaviour unexpectedly: check 5 guards the lambda = 0 baseline; ByteTrack stays the default unless the rule says otherwise.

---

## 1. Summary
**Status:** results complete; pending supervisor review. Protocol: section 0, committed (`55c8741`) before any embedding. Numbers below come from `outputs/video/mevid/results/` and `outputs/video/mevid/checks.json`; tables are in section 13.

- **MEVID is a real test: single frames fail, and combining frames helps (both primary hypotheses supported).** On 3,188 eligible tracklets of 151 enrolled persons (AdaFace MS1MV2, gallery bar 48 px) closed-set Rank-1 is 0.347 for a single frame and 0.495 for the R1 tracklet template: **H1 +0.147 (95% CI +0.133 to +0.164)**. In the open set (5 rotated folds, 151 persons) TPIR1 at FPIR 0.1 is 0.329 for single frames and 0.374 for R1: **H2 +0.045 (+0.002 to +0.092)**; the lower bound is only just above zero. At FPIR 0.3 the gain is +0.050 (+0.015 to +0.088); at FPIR 0.01 it is +0.041 with a CI that includes zero (-0.017 to +0.122), so the strict operating point is inconclusive.
- **Aggregation helps at every face size and on almost every camera.** By the tracklet's mean face height: +0.088 for [16, 24) px, +0.164 for [24, 32), +0.174 for [32, 48) and +0.175 for 48 px and above (all CIs above zero); single-frame Rank-1 rises from 0.127 to 0.511 over the same bins.
- **On real faces the plain mean is not the best rule.** Against R1 (Bonferroni over nine comparisons) R4* (min-max norm pooling) is +0.0235 (CI +0.0081 to +0.0410) and R2 (norm-weighted mean) +0.0091 (+0.0045 to +0.0146); R3 with k = 10 and the vote are nominally higher but their corrected CIs reach zero; norm pooling with lambda 50 and the single highest-norm frame are worse. Phase 7 (synthetic degradation) found no rule better than the mean.
- **The MagFace norm and the face height predict errors on real faces (Part C).** Rejecting the lowest-quality half of the tracklets lowers the closed-set error from 50.5% to 34.4% (MagFace norm) or 34.5% (mean face height); six of seven pre-registered signals beat random rejection after Bonferroni correction (luminance does not). On the synthetic ChokePoint probes none of the available signals helps, which is why the Phase 7 decision "the MagFace norm is not useful" does not carry over to real faces.
- **The face-aware tracker term removes the frame-rate fragility (Part D).** With the face term (lambda 0.5) the IDF1 under every-10th-frame subsampling rises from 0.25 to 0.71 (BoT-SORT) and from 0.24 to 0.81 (ByteTrack) and the ID switches nearly vanish. By the pre-registered rule the downstream open-set track-identity accuracy does not separate the trackers (ByteTrack minus BoT-SORT -0.0036, CI -0.0106 to +0.0002), so **ByteTrack stays the default**.
- **Backbone (exploratory pairing).** The WebFace4M AdaFace is far stronger on real low-resolution faces: single-frame Rank-1 0.483 against 0.347 and R1 0.675 against 0.495; the aggregation gain is larger too (+0.192 against +0.147).
- **Restoration gate (Part E):** closed by the Phase 4 stop rule, no run.
- **Checks:** all six pass (section 3); the one count discrepancy (8,093 against 8,092 tracklets) is an empty annotation row.

## 2. What was built
- `scripts/mevid_embed.py`: stages `index`, `check2`, `detect`, `eligibility`, `gallery` (detector only) and `embed` (the first stage that produces an embedding; refuses to run without a flag; run only after section 0 was committed).
- `scripts/mevid_extract.py` (selective extraction of the train chips that are read), `scripts/mevid_download.py` (parallel range download of the train package).
- `scripts/mevid_openset.py` (Part B), `scripts/mevid_quality.py` (Part C), `scripts/mevid_tracker_identity.py` and `scripts/video_tracker_stress.py --lam` (Part D), `scripts/mevid_backbone_pair.py` (exploratory), `scripts/phase8_report_tables.py`; `scripts/video_caface.py --part mevid` (WebFace4M rows).
- `src/video/track.py:TrackerAdapter` (option `lam`, `min_h`) and a cost hook in the two vendored trackers (`src/video/vendor/VENDOR.md`).
- Deviations, all stated before any score: the parallel download (the S3 host gives about 0.2 to 0.3 MB/s per connection); only the train chips that are read were extracted (303,244 of 3.6 million); only eligible tracklets were embedded; one empty annotation row is skipped.

## 3. Acceptance checks (`outputs/video/mevid/checks.json`)
1. **Counts:** persons 158 and check-in photos 2,898 (172 persons) equal the README and the probe inventory; the annotation file has 8,093 tracklet rows, of which one (`tr01889`, start 1,128,477, end 1,128,476, zero chips) is empty, so **8,092 tracklets have chips, equal to the README and the plan**.
2. **Probe boxes reproduced:** the 3,949 probe chips (1,489 faces) give identical boxes and scores (maximum difference 0.0 after rounding to three decimals).
3. **Rules:** a length-1 tracklet equals the single frame for every rule and frame order does not change R1 to R5, on a real tracklet (and on synthetic data in the self-test).
4. **Folds:** every one of the 151 persons is held out exactly once; the gallery and the non-enrolled probes are disjoint in all 5 folds.
5. **Lambda 0:** the modified trackers with the face term at 0 reproduce the Phase 7 per-clip stress results byte for byte (the same SHA-256, 864 rows).
6. **Reruns:** every CPU part was run twice and compared byte for byte: the Part B tables for both backbones, the per-tracklet records, Part C (MEVID and ChokePoint), the backbone pairing, the face-term tracker run (per-clip table and all track files) and the per-clip track-identity table are identical. Two files differ in form only: the first-run copy of `closed_adaface.csv` predates the added Bonferroni columns (all shared columns equal), and the identity summary and decision file were first built from the 5-decimal CSV and then from unrounded values (differences below 2e-7; the decision is unchanged). The GPU embedding stages were not repeated (about 6 hours); they are deterministic in the Phase 6 and 7 checks. Result: pass.

## 4. MEVID intake: eligibility and enrolment
- **Tracklets:** 3,190 of 8,092 tracklets are eligible (at least 5 usable faces among up to 50 evenly spaced chips), 39.4%; 893 of 1,754 test and 2,297 of 6,338 train tracklets; 153 of 158 persons have one; 90,103 usable faces were embedded. The mean face height of an eligible tracklet has a median of 33.5 px (5th to 95th percentile 17.5 to 82.6).
- **Cameras:** eligibility varies widely (24 cameras; for example 464 of 553 tracklets on camera 507, 1 of 417 on camera 336).
- **Gallery:** 1,454 front photos of 172 persons; at the 48 px bar 170 persons are enrolled (52 of 54 test, all 104 train, all 14 extra), at the 64 px bar 148 (30 of 54 test). The test persons' check-in photos are small (median front-face height 56 px) against the train persons' (211 px). The open-set population (enrolled and with an eligible tracklet) has 151 persons at 48 px and 129 at 64 px.

## 5. MEVID results (AdaFace MS1MV2)
- **H1 and H2** as in section 1. The 64 px gallery bar gives the same picture (closed set +0.145, CI +0.128 to +0.163, 129 persons).
- **Open set.** TPIR1 at FPIR 0.3 / 0.1 / 0.01: single frame 0.389 / 0.329 / 0.134; R1 0.439 / 0.374 / 0.175 (TPIR20 in the table). The per-fold TPIR1 at FPIR 0.01 ranges from 0.04 to 0.28, which is why that operating point cannot be judged here.
- **People without a usable photo behave like held-out ones.** At the 64 px bar 24 persons (298 tracklets) have no usable photo; their false-positive rate at the fold thresholds is 0.302 / 0.107 / 0.015 for target FPIR 0.3 / 0.1 / 0.01, which is the target. At the 48 px bar only 2 persons are in this row (too few to say anything).
- **By camera.** Among the 21 cameras with at least 3 persons the R1 gain is positive on 20 and has a CI above zero on 16; it is near zero where single frames are almost always wrong (camera 339: 0.034 to 0.038) and cannot be judged where only 3 to 4 persons are present (cameras 341, 436).
- **Real faces are hard.** Even aggregated, the closed-set Rank-1 is 0.495: there is room for improvement, unlike on native ChokePoint (Phase 6) or YouTube Faces (Phase 7).

## 6. Rules on real faces
Against R1, mean over tracklets, 95% CI Bonferroni-corrected over the nine comparisons (gallery bar 48 px):
- **Better than R1:** R4* +0.0235 (+0.0081 to +0.0410), R2 +0.0091 (+0.0045 to +0.0146). At the 64 px bar also R3 with k = 5 (+0.018), k = 10 (+0.024), R4* (+0.030), R2 (+0.012) and the vote (+0.020).
- **Not distinguishable:** R3 with k = 3, k = 5 and k = 10 and the vote R6 at 48 px (corrected CIs reach zero).
- **Worse:** R4 norm pooling with lambda 50 (-0.0358, -0.0603 to -0.0117) and R3 with k = 1 (-0.1001, -0.1335 to -0.0657).
- **R5 equals R1** for Rank-1 by construction, as in Phase 7.
- The comparison with Phase 7 (synthetic 12 px: no rule beats the mean) shows that the rule ranking depends on the data: on real faces the MagFace norm carries information about frame reliability. R1 stays the reference rule of the primary hypotheses; R4* and R2 are candidates for the pipeline (decision for Phase 9).

## 7. Backbone rows (WebFace4M)
- **AdaFace R100 WebFace4M (the CAFace backbone), same crops, tracklets and gallery protocol:** closed-set single frame 0.483, R1 0.675 (difference +0.192, CI +0.177 to +0.209); open-set TPIR1 at FPIR 0.1: 0.456 for single frames and 0.543 for R1 (+0.087, +0.016 to +0.161).
- **Exploratory pairing (not pre-registered), WebFace4M minus MS1MV2, paired per tracklet, bootstrap over persons:** single frame +0.135 (+0.120 to +0.154), R1 +0.180 (+0.155 to +0.209), aggregation gain +0.045 (+0.026 to +0.064). The two models share the architecture and the loss; they differ in training data, so the difference is a training-data effect (and the WebFace4M weights come from a different training run), not a controlled one.
- Embedding the MS1MV2 and WebFace4M faces is on the same aligned crops; only R1 and the single frame are run for WebFace4M.

## 8. Part C: frame-quality reject test
- **MEVID (3,188 eligible tracklets of enrolled persons, closed-set R1; overall error 0.5053).** Area under the error-versus-rejected-fraction curve over [0, 0.5], difference to random rejection (random = 0.2527), Bonferroni CI over seven signals: mean MagFace norm -0.0477 (-0.0570 to -0.0391), mean face height -0.0475 (-0.0563 to -0.0371), number of usable faces -0.0396 (-0.0478 to -0.0313), mean Laplacian variance -0.0335 (-0.0418 to -0.0245), mean AdaFace raw norm -0.0167 (-0.0279 to -0.0041), self-consistency -0.0082 (-0.0160 to -0.0002); mean luminance -0.0006 (-0.0116 to +0.0110) does not help, nor does the exploratory variant -|luminance - 128| (-0.0007). Rejecting the lowest half leaves an error of 0.344 (MagFace norm), 0.345 (face height), 0.346 (usable faces), 0.383 (Laplacian), 0.438 (AdaFace norm), 0.489 (self-consistency).
- **ChokePoint, synthetic 12 and 16 px (secondary).** At 12 px (432 tracks, error 0.067) none of the three available signals helps (usable faces -0.0055, CI -0.0199 to +0.0055; self-consistency +0.0016; MagFace norm +0.0022); at 16 px R1 makes no errors, so there is nothing to reject.
- **Reading.** The best signals are the MagFace norm and the face height, which tie; the face height needs no recognition network (it comes from the detector), the MagFace norm needs one forward pass. The AdaFace norm is weaker than the MagFace norm in this setting, and self-consistency adds little.

## 9. Part D: final tracker choice (ChokePoint, face-aware association)
- **Without the face term**, under frame subsampling BoT-SORT has the higher downstream open-set track-identity accuracy (for example 0.962 against 0.788 at every 10th frame), but the numbers of scored tracks differ widely (2,097 against 299), so the accuracy over scored tracks is not comparable there; the IDF1 table (Phase 7) shows the opposite order.
- **With the face term (lambda 0.5, gate: detector score >= 0.5 and native height >= 32 px)** the two trackers behave alike and the fragility disappears: IDF1 under S1 / S2 / S3 (every 3rd / 5th / 10th frame) is 0.754 / 0.784 / 0.806 for ByteTrack (0.733 / 0.558 / 0.242 without the term) and 0.712 / 0.716 / 0.706 for BoT-SORT (0.633 / 0.427 / 0.247 without); ID switches per clip fall from 29 to 0.04 (BoT-SORT, S1) and from 1.8 to 0.01 (ByteTrack, S1). Under dropout (S4, S5) and on the original frames the term changes IDF1 little. The scored-track counts become equal (432 to 464 per condition).
- **Decision rule (section 0.6): the downstream open-set track-identity accuracy with the face term, averaged over S0 to S5, per-clip difference ByteTrack minus BoT-SORT, is -0.0036 (95% CI -0.0106 to +0.0002, 24 clips); the CI includes zero, so ByteTrack stays the default.** The result is borderline and driven by one condition (S5, 20% dropped detections: 0.884 for BoT-SORT against 0.862 for ByteTrack); IDF1 favours ByteTrack in every stress condition.
- **Caveat:** the test is within ChokePoint, not held out.

## 10. Part E: restoration gate
No run. The gate stage is closed by the Phase 4 stop rule (0 of 3 non-leaky recognizers for every arm); the pipeline keeps the hook off by default.

## 11. Timings
Lab PC (RTX 4000 Ada), shared machine, wall-clock, including image decoding: SCRFD detection of the up to 50 chips per tracklet took roughly 30 to 45 minutes for the 1,754 test tracklets and about 65 minutes for the 6,338 train tracklets (stopped once at the empty row and resumed; both ran next to other jobs); the gallery photos (1,454, size 1280) 354 s; the AdaFace embedding of the 90,103 usable faces about 40 minutes (with CPU analyses running alongside); the WebFace4M embedding about 30 minutes; the Part B to D analyses minutes to about 15 minutes each on the CPU. The 32.7 GB train package was fetched with 32 parallel connections (0.2 to 0.3 MB/s each at first, up to about 40 MB/s in total), then joined and verified, in roughly an hour altogether; the selective extraction of the needed chips took about 40 minutes.

## 12. Limitations
- The check-in photos are probably from the recording days, so the gap between enrolment and probe is short; the test persons' photos are small (median 56 px) so the 48 px bar enrols them at the edge of what the detector gives.
- MEVID tracklets are MEVID's own, not produced by our tracker; the faces are crops from body boxes; 39% of the tracklets are eligible, with very different shares per camera (the result describes the eligible tracklets).
- The open-set comparison at FPIR 0.01 is inconclusive (large fold-to-fold variation); the H2 lower bound is only just above zero.
- Part D is within ChokePoint (not held out), uses the plan's fallback gate, and its accuracy is over scored tracks (counts reported); the borderline decision is driven by one condition.
- The backbone pairing is exploratory; WebFace4M and MS1MV2 differ in training data and in the training run.
- One empty annotation row, selective extraction and the parallel download are deviations of procedure that do not change the data read.

## 13. Result tables
<!-- TABLES:BEGIN -->

#### Counts (acceptance check 1)

```
{
 "checkin_persons": 172,
 "checkin_photos": 2898,
 "checkin_photos_equal_probe_inventory": true,
 "empty_tracklets": [
  {
   "end": 1128476,
   "person": 78,
   "start": 1128477,
   "tracklet": "tr01889"
  }
 ],
 "persons_equal_readme": true,
 "persons_test": 54,
 "persons_total": 158,
 "persons_train": 104,
 "readme_counts": {
  "identities": 158,
  "tracklets": 8092
 },
 "tracklets_equal_readme": false,
 "tracklets_test": 1754,
 "tracklets_total": 8093,
 "tracklets_train": 6339,
 "tracklets_with_at_least_one_chip": 8092,
 "tracklets_with_chips_equal_readme": true
}
```

#### Tracklet eligibility (>= 5 usable faces of up to 50 chips)

| split | tracklets | eligible | persons_with_eligible |
|---|---|---|---|
| test | 1754 | 893 | 54 |
| train | 6338 | 2297 | 99 |

By camera:

| split | camera | tracklets | eligible | share_eligible |
|---|---|---|---|---|
| test | 301 | 1 | True | 1.0 |
| test | 329 | 66 | 42 | 0.636 |
| test | 330 | 24 | 10 | 0.417 |
| test | 336 | 365 | 1 | 0.003 |
| test | 340 | 50 | 13 | 0.26 |
| test | 436 | 185 | 3 | 0.016 |
| test | 505 | 34 | 23 | 0.676 |
| test | 507 | 553 | 464 | 0.839 |
| test | 508 | 34 | 32 | 0.941 |
| test | 509 | 148 | 144 | 0.973 |
| test | 639 | 294 | 160 | 0.544 |
| train | 299 | 509 | 136 | 0.267 |
| train | 300 | 51 | 28 | 0.549 |
| train | 326 | 164 | 131 | 0.799 |
| train | 328 | 206 | 0 | 0.0 |
| train | 329 | 136 | 92 | 0.676 |
| train | 330 | 433 | 126 | 0.291 |
| train | 331 | 258 | 202 | 0.783 |
| train | 336 | 52 | 0 | 0.0 |
| train | 339 | 608 | 80 | 0.132 |
| train | 340 | 206 | 80 | 0.388 |
| train | 341 | 672 | 4 | 0.006 |
| train | 419 | 318 | 198 | 0.623 |
| train | 420 | 479 | 355 | 0.741 |
| train | 421 | 511 | 334 | 0.654 |
| train | 423 | 169 | 18 | 0.107 |
| train | 424 | 433 | 79 | 0.182 |
| train | 436 | 37 | 0 | 0.0 |
| train | 505 | 99 | 59 | 0.596 |
| train | 506 | 552 | 196 | 0.355 |
| train | 508 | 222 | 170 | 0.766 |
| train | 509 | 8 | 4 | 0.5 |
| train | 638 | 215 | 5 | 0.023 |

#### Enrolment (persons with a usable gallery photo)

| model | bar | persons_enrolled | persons_in_open_set_population | eligible_tracklets | eligible_tracklets_of_enrolled_persons | persons_without_usable_photo_with_eligible_tracklets |
|---|---|---|---|---|---|---|
| adaface | 48 | 170 | 151 | 3190 | 3188 | 2 |
| adaface | 64 | 148 | 129 | 3190 | 2892 | 24 |

#### Primary hypotheses H1 and H2 (AdaFace MS1MV2, gallery bar 48 px)

| hypothesis | tracklets | persons | single | R1 | difference_95CI | above_zero |
|---|---|---|---|---|---|---|
| H1: closed-set Rank-1, R1 minus single frame (paired per tracklet) | 3188 | 151 | 0.3473 | 0.4947 | +0.1473 (+0.1326 to +0.1640) | yes |
| H2: open-set TPIR1 at FPIR 0.1, R1 minus single frame (fold-averaged) |  |  | 0.3286 | 0.3740 | +0.0454 (+0.0018 to +0.0915) | yes |

#### Closed set: rules against the single frame and against R1 (AdaFace MS1MV2)

| setting | rule | tracklets | persons | single | aggregated | difference_vs_single | minus_R1 | minus_R1_Bonferroni_CI |
|---|---|---|---|---|---|---|---|---|
| adaface, gallery bar 48 px | R1 | 3188 | 151 | 0.3473 | 0.4947 | +0.1473 (+0.1326 to +0.1640) |  |  |
| adaface, gallery bar 48 px | R2 | 3188 | 151 | 0.3473 | 0.5038 | +0.1564 (+0.1409 to +0.1738) | +0.0091 (+0.0053 to +0.0128) | +0.0045 to +0.0146 |
| adaface, gallery bar 48 px | R3_k1 | 3188 | 151 | 0.3473 | 0.3946 | +0.0473 (+0.0314 to +0.0640) | -0.1001 (-0.1233 to -0.0773) | -0.1335 to -0.0657 |
| adaface, gallery bar 48 px | R3_k3 | 3188 | 151 | 0.3473 | 0.4828 | +0.1354 (+0.1194 to +0.1525) | -0.0119 (-0.0273 to +0.0026) | -0.0302 to +0.0092 |
| adaface, gallery bar 48 px | R3_k5 | 3188 | 151 | 0.3473 | 0.5003 | +0.1530 (+0.1368 to +0.1708) | +0.0056 (-0.0076 to +0.0179) | -0.0112 to +0.0243 |
| adaface, gallery bar 48 px | R3_k10 | 3188 | 151 | 0.3473 | 0.5129 | +0.1655 (+0.1486 to +0.1842) | +0.0182 (+0.0064 to +0.0295) | -0.0011 to +0.0352 |
| adaface, gallery bar 48 px | R4 | 3188 | 151 | 0.3473 | 0.4589 | +0.1116 (+0.0956 to +0.1298) | -0.0358 (-0.0542 to -0.0175) | -0.0603 to -0.0117 |
| adaface, gallery bar 48 px | R4* | 3188 | 151 | 0.3473 | 0.5182 | +0.1709 (+0.1540 to +0.1897) | +0.0235 (+0.0122 to +0.0350) | +0.0081 to +0.0410 |
| adaface, gallery bar 48 px | R5 | 3188 | 151 | 0.3473 | 0.4947 | +0.1473 (+0.1326 to +0.1640) | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| adaface, gallery bar 48 px | R6 | 3188 | 151 | 0.3473 | 0.5066 | +0.1593 (+0.1427 to +0.1753) | +0.0119 (+0.0034 to +0.0211) | -0.0005 to +0.0256 |
| adaface, gallery bar 64 px | R1 | 2892 | 129 | 0.3471 | 0.4920 | +0.1450 (+0.1284 to +0.1629) |  |  |
| adaface, gallery bar 64 px | R2 | 2892 | 129 | 0.3471 | 0.5041 | +0.1571 (+0.1392 to +0.1762) | +0.0121 (+0.0078 to +0.0168) | +0.0065 to +0.0183 |
| adaface, gallery bar 64 px | R3_k1 | 2892 | 129 | 0.3471 | 0.3918 | +0.0447 (+0.0283 to +0.0615) | -0.1003 (-0.1243 to -0.0780) | -0.1330 to -0.0698 |
| adaface, gallery bar 64 px | R3_k3 | 2892 | 129 | 0.3471 | 0.4855 | +0.1384 (+0.1206 to +0.1556) | -0.0066 (-0.0226 to +0.0079) | -0.0286 to +0.0140 |
| adaface, gallery bar 64 px | R3_k5 | 2892 | 129 | 0.3471 | 0.5097 | +0.1626 (+0.1451 to +0.1797) | +0.0176 (+0.0044 to +0.0308) | +0.0005 to +0.0361 |
| adaface, gallery bar 64 px | R3_k10 | 2892 | 129 | 0.3471 | 0.5159 | +0.1689 (+0.1492 to +0.1875) | +0.0239 (+0.0116 to +0.0364) | +0.0081 to +0.0417 |
| adaface, gallery bar 64 px | R4 | 2892 | 129 | 0.3471 | 0.4588 | +0.1118 (+0.0938 to +0.1299) | -0.0332 (-0.0533 to -0.0155) | -0.0599 to -0.0079 |
| adaface, gallery bar 64 px | R4* | 2892 | 129 | 0.3471 | 0.5221 | +0.1751 (+0.1556 to +0.1938) | +0.0301 (+0.0176 to +0.0420) | +0.0143 to +0.0471 |
| adaface, gallery bar 64 px | R5 | 2892 | 129 | 0.3471 | 0.4920 | +0.1450 (+0.1284 to +0.1629) | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| adaface, gallery bar 64 px | R6 | 2892 | 129 | 0.3471 | 0.5118 | +0.1647 (+0.1458 to +0.1832) | +0.0197 (+0.0103 to +0.0308) | +0.0073 to +0.0348 |

#### Open set: TPIR1 and TPIR20, fold-averaged (AdaFace MS1MV2)

| setting | metric | fpir | probe | value | CI |
|---|---|---|---|---|---|
| adaface, gallery bar 48 px | TPIR1 | 0.3 | single frame | 0.3888 | 0.3631 to 0.4149 |
| adaface, gallery bar 48 px | TPIR1 | 0.3 | R1 tracklet | 0.4387 | 0.4140 to 0.4627 |
| adaface, gallery bar 48 px | TPIR1 | 0.3 | R1 minus single frame | 0.0499 | 0.0147 to 0.0883 |
| adaface, gallery bar 48 px | TPIR1 | 0.1 | single frame | 0.3286 | 0.2861 to 0.3576 |
| adaface, gallery bar 48 px | TPIR1 | 0.1 | R1 tracklet | 0.3740 | 0.3413 to 0.4020 |
| adaface, gallery bar 48 px | TPIR1 | 0.1 | R1 minus single frame | 0.0454 | 0.0018 to 0.0915 |
| adaface, gallery bar 48 px | TPIR1 | 0.01 | single frame | 0.1339 | 0.1023 to 0.1903 |
| adaface, gallery bar 48 px | TPIR1 | 0.01 | R1 tracklet | 0.1751 | 0.1489 to 0.2477 |
| adaface, gallery bar 48 px | TPIR1 | 0.01 | R1 minus single frame | 0.0412 | -0.0174 to 0.1215 |
| adaface, gallery bar 48 px | TPIR20 | 0.3 | single frame | 0.4129 | 0.3858 to 0.4380 |
| adaface, gallery bar 48 px | TPIR20 | 0.3 | R1 tracklet | 0.4669 | 0.4415 to 0.4917 |
| adaface, gallery bar 48 px | TPIR20 | 0.3 | R1 minus single frame | 0.0540 | 0.0190 to 0.0926 |
| adaface, gallery bar 48 px | TPIR20 | 0.1 | single frame | 0.3396 | 0.2980 to 0.3688 |
| adaface, gallery bar 48 px | TPIR20 | 0.1 | R1 tracklet | 0.3852 | 0.3481 to 0.4134 |
| adaface, gallery bar 48 px | TPIR20 | 0.1 | R1 minus single frame | 0.0456 | 0.0009 to 0.0933 |
| adaface, gallery bar 48 px | TPIR20 | 0.01 | single frame | 0.1388 | 0.1059 to 0.1963 |
| adaface, gallery bar 48 px | TPIR20 | 0.01 | R1 tracklet | 0.1793 | 0.1376 to 0.2443 |
| adaface, gallery bar 48 px | TPIR20 | 0.01 | R1 minus single frame | 0.0406 | -0.0348 to 0.1099 |
| adaface, gallery bar 64 px | TPIR1 | 0.3 | single frame | 0.3808 | 0.3548 to 0.4051 |
| adaface, gallery bar 64 px | TPIR1 | 0.3 | R1 tracklet | 0.4349 | 0.4087 to 0.4610 |
| adaface, gallery bar 64 px | TPIR1 | 0.3 | R1 minus single frame | 0.0541 | 0.0193 to 0.0910 |
| adaface, gallery bar 64 px | TPIR1 | 0.1 | single frame | 0.3183 | 0.2794 to 0.3442 |
| adaface, gallery bar 64 px | TPIR1 | 0.1 | R1 tracklet | 0.3644 | 0.3161 to 0.3911 |
| adaface, gallery bar 64 px | TPIR1 | 0.1 | R1 minus single frame | 0.0461 | -0.0053 to 0.0959 |
| adaface, gallery bar 64 px | TPIR1 | 0.01 | single frame | 0.1109 | 0.0870 to 0.1815 |
| adaface, gallery bar 64 px | TPIR1 | 0.01 | R1 tracklet | 0.1522 | 0.1327 to 0.2406 |
| adaface, gallery bar 64 px | TPIR1 | 0.01 | R1 minus single frame | 0.0413 | -0.0248 to 0.1241 |
| adaface, gallery bar 64 px | TPIR20 | 0.3 | single frame | 0.4055 | 0.3782 to 0.4288 |
| adaface, gallery bar 64 px | TPIR20 | 0.3 | R1 tracklet | 0.4616 | 0.4351 to 0.4883 |
| adaface, gallery bar 64 px | TPIR20 | 0.3 | R1 minus single frame | 0.0561 | 0.0218 to 0.0939 |
| adaface, gallery bar 64 px | TPIR20 | 0.1 | single frame | 0.3301 | 0.2899 to 0.3565 |
| adaface, gallery bar 64 px | TPIR20 | 0.1 | R1 tracklet | 0.3754 | 0.3241 to 0.4045 |
| adaface, gallery bar 64 px | TPIR20 | 0.1 | R1 minus single frame | 0.0453 | -0.0081 to 0.0971 |
| adaface, gallery bar 64 px | TPIR20 | 0.01 | single frame | 0.1152 | 0.0910 to 0.1862 |
| adaface, gallery bar 64 px | TPIR20 | 0.01 | R1 tracklet | 0.1466 | 0.1243 to 0.2273 |
| adaface, gallery bar 64 px | TPIR20 | 0.01 | R1 minus single frame | 0.0314 | -0.0362 to 0.1096 |

#### Open set per fold

| setting | fold | n_gallery | n_held_persons | n_mated_tracklets | n_nonenrolled_tracklets | TPIR1@0.3_R1 | TPIR1@0.1_R1 | TPIR1@0.01_R1 |
|---|---|---|---|---|---|---|---|---|
| adaface, gallery bar 48 px | 0 | 139 | 31 | 2595 | 593 | 0.4343 | 0.3499 | 0.0894 |
| adaface, gallery bar 48 px | 1 | 140 | 30 | 2432 | 756 | 0.4548 | 0.4005 | 0.2533 |
| adaface, gallery bar 48 px | 2 | 140 | 30 | 2511 | 677 | 0.4305 | 0.3771 | 0.2756 |
| adaface, gallery bar 48 px | 3 | 140 | 30 | 2637 | 551 | 0.4346 | 0.3724 | 0.2192 |
| adaface, gallery bar 48 px | 4 | 140 | 30 | 2577 | 611 | 0.4393 | 0.3698 | 0.038 |
| adaface, gallery bar 64 px | 0 | 122 | 26 | 2434 | 458 | 0.4359 | 0.3603 | 0.1339 |
| adaface, gallery bar 64 px | 1 | 122 | 26 | 2253 | 639 | 0.4314 | 0.3555 | 0.1025 |
| adaface, gallery bar 64 px | 2 | 122 | 26 | 2274 | 618 | 0.4547 | 0.3835 | 0.2265 |
| adaface, gallery bar 64 px | 3 | 122 | 26 | 2315 | 577 | 0.4026 | 0.34 | 0.2652 |
| adaface, gallery bar 64 px | 4 | 123 | 25 | 2292 | 600 | 0.4498 | 0.3826 | 0.0327 |

#### People without a usable gallery photo as non-enrolled probes

| setting | fpir_target | n_tracklets | n_persons | false_positive_rate_mean_over_folds | model |
|---|---|---|---|---|---|
| bar 48 | 0.3 | 2 | 2 | 0.0 | adaface |
| bar 48 | 0.1 | 2 | 2 | 0.0 | adaface |
| bar 48 | 0.01 | 2 | 2 | 0.0 | adaface |
| bar 64 | 0.3 | 298 | 24 | 0.302 | adaface |
| bar 64 | 0.1 | 298 | 24 | 0.1074 | adaface |
| bar 64 | 0.01 | 298 | 24 | 0.0148 | adaface |

#### By camera (closed set, R1 against single frame)

| setting | rule | n_tracklets | n_persons | single_rank1 | agg_rank1 | delta | delta_ci_lo | delta_ci_hi | delta_bonf_lo | delta_bonf_hi | model |
|---|---|---|---|---|---|---|---|---|---|---|---|
| camera 299 | R1 | 136 | 52 | 0.1779 | 0.2941 | 0.1162 | 0.0474 | 0.1918 | 0.0474 | 0.1918 | adaface |
| camera 300 | R1 | 28 | 20 | 0.1997 | 0.2857 | 0.086 | -0.0253 | 0.1721 | -0.0253 | 0.1721 | adaface |
| camera 326 | R1 | 131 | 48 | 0.505 | 0.7405 | 0.2355 | 0.184 | 0.2858 | 0.184 | 0.2858 | adaface |
| camera 329 | R1 | 134 | 65 | 0.3404 | 0.5298 | 0.1894 | 0.1357 | 0.2517 | 0.1357 | 0.2517 | adaface |
| camera 330 | R1 | 136 | 39 | 0.2408 | 0.375 | 0.1342 | 0.0428 | 0.2186 | 0.0428 | 0.2186 | adaface |
| camera 331 | R1 | 202 | 33 | 0.3522 | 0.5346 | 0.1825 | 0.1304 | 0.2455 | 0.1304 | 0.2455 | adaface |
| camera 339 | R1 | 80 | 48 | 0.0341 | 0.0375 | 0.0034 | -0.0268 | 0.0418 | -0.0268 | 0.0418 | adaface |
| camera 340 | R1 | 93 | 51 | 0.2794 | 0.4194 | 0.1399 | 0.0679 | 0.2043 | 0.0679 | 0.2043 | adaface |
| camera 341 | R1 | 4 | 4 | 0.0833 | 0.0 | -0.0833 | -0.25 | 0.0 | -0.25 | 0.0 | adaface |
| camera 419 | R1 | 198 | 80 | 0.5825 | 0.7576 | 0.1751 | 0.1395 | 0.2111 | 0.1395 | 0.2111 | adaface |
| camera 420 | R1 | 355 | 90 | 0.3187 | 0.5042 | 0.1855 | 0.1538 | 0.2158 | 0.1538 | 0.2158 | adaface |
| camera 421 | R1 | 334 | 63 | 0.252 | 0.3862 | 0.1342 | 0.0886 | 0.1839 | 0.0886 | 0.1839 | adaface |
| camera 423 | R1 | 18 | 14 | 0.3019 | 0.5556 | 0.2537 | 0.0777 | 0.4364 | 0.0777 | 0.4364 | adaface |
| camera 424 | R1 | 79 | 35 | 0.0884 | 0.1646 | 0.0762 | 0.0167 | 0.1508 | 0.0167 | 0.1508 | adaface |
| camera 436 | R1 | 3 | 3 | 0.2603 | 0.3333 | 0.073 | -0.0667 | 0.2857 | -0.0667 | 0.2857 | adaface |
| camera 505 | R1 | 82 | 36 | 0.4087 | 0.6463 | 0.2377 | 0.147 | 0.3202 | 0.147 | 0.3202 | adaface |
| camera 506 | R1 | 196 | 42 | 0.134 | 0.2092 | 0.0752 | 0.0393 | 0.1336 | 0.0393 | 0.1336 | adaface |
| camera 507 | R1 | 464 | 38 | 0.5756 | 0.7565 | 0.1809 | 0.1342 | 0.2255 | 0.1342 | 0.2255 | adaface |
| camera 508 | R1 | 200 | 37 | 0.272 | 0.445 | 0.173 | 0.1123 | 0.2587 | 0.1123 | 0.2587 | adaface |
| camera 509 | R1 | 148 | 19 | 0.7654 | 0.8378 | 0.0724 | 0.0311 | 0.1327 | 0.0311 | 0.1327 | adaface |
| camera 639 | R1 | 160 | 45 | 0.0656 | 0.0938 | 0.0281 | -0.0017 | 0.0653 | -0.0017 | 0.0653 | adaface |

#### By mean face height (closed set, R1 against single frame)

| setting | rule | n_tracklets | n_persons | single_rank1 | agg_rank1 | delta | delta_ci_lo | delta_ci_hi | delta_bonf_lo | delta_bonf_hi | model |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean face height [16, 24) px | R1 | 928 | 138 | 0.1268 | 0.2144 | 0.0876 | 0.0623 | 0.1166 | 0.0623 | 0.1166 | adaface |
| mean face height [24, 32) px | R1 | 550 | 121 | 0.2963 | 0.46 | 0.1637 | 0.1317 | 0.1952 | 0.1317 | 0.1952 | adaface |
| mean face height [32, 48) px | R1 | 867 | 139 | 0.4567 | 0.6309 | 0.1742 | 0.1478 | 0.2006 | 0.1478 | 0.2006 | adaface |
| mean face height [48, 1000000) px | R1 | 843 | 134 | 0.5108 | 0.6856 | 0.1748 | 0.1469 | 0.2 | 0.1469 | 0.2 | adaface |

#### WebFace4M backbone: closed set

| setting | rule | tracklets | persons | single | aggregated | difference_vs_single | minus_R1 | minus_R1_Bonferroni_CI |
|---|---|---|---|---|---|---|---|---|
| webface4m, gallery bar 48 px | R1 | 3188 | 151 | 0.4826 | 0.6747 | +0.1921 (+0.1770 to +0.2091) |  |  |
| webface4m, gallery bar 64 px | R1 | 2892 | 129 | 0.4807 | 0.6802 | +0.1995 (+0.1822 to +0.2178) |  |  |

#### WebFace4M backbone: open set

| setting | metric | fpir | probe | value | CI |
|---|---|---|---|---|---|
| webface4m, gallery bar 48 px | TPIR1 | 0.3 | single frame | 0.5307 | 0.5045 to 0.5566 |
| webface4m, gallery bar 48 px | TPIR1 | 0.3 | R1 tracklet | 0.6227 | 0.6020 to 0.6459 |
| webface4m, gallery bar 48 px | TPIR1 | 0.3 | R1 minus single frame | 0.0920 | 0.0607 to 0.1262 |
| webface4m, gallery bar 48 px | TPIR1 | 0.1 | single frame | 0.4556 | 0.3873 to 0.4880 |
| webface4m, gallery bar 48 px | TPIR1 | 0.1 | R1 tracklet | 0.5430 | 0.4752 to 0.5754 |
| webface4m, gallery bar 48 px | TPIR1 | 0.1 | R1 minus single frame | 0.0874 | 0.0161 to 0.1609 |
| webface4m, gallery bar 48 px | TPIR1 | 0.01 | single frame | 0.1560 | 0.1210 to 0.2310 |
| webface4m, gallery bar 48 px | TPIR1 | 0.01 | R1 tracklet | 0.1987 | 0.1737 to 0.3087 |
| webface4m, gallery bar 48 px | TPIR1 | 0.01 | R1 minus single frame | 0.0427 | -0.0249 to 0.1572 |
| webface4m, gallery bar 48 px | TPIR20 | 0.3 | single frame | 0.5579 | 0.5336 to 0.5822 |
| webface4m, gallery bar 48 px | TPIR20 | 0.3 | R1 tracklet | 0.6532 | 0.6336 to 0.6735 |
| webface4m, gallery bar 48 px | TPIR20 | 0.3 | R1 minus single frame | 0.0954 | 0.0660 to 0.1292 |
| webface4m, gallery bar 48 px | TPIR20 | 0.1 | single frame | 0.4696 | 0.4021 to 0.5014 |
| webface4m, gallery bar 48 px | TPIR20 | 0.1 | R1 tracklet | 0.5611 | 0.4871 to 0.5921 |
| webface4m, gallery bar 48 px | TPIR20 | 0.1 | R1 minus single frame | 0.0915 | 0.0147 to 0.1624 |
| webface4m, gallery bar 48 px | TPIR20 | 0.01 | single frame | 0.1601 | 0.1237 to 0.2348 |
| webface4m, gallery bar 48 px | TPIR20 | 0.01 | R1 tracklet | 0.1946 | 0.1606 to 0.2927 |
| webface4m, gallery bar 48 px | TPIR20 | 0.01 | R1 minus single frame | 0.0344 | -0.0369 to 0.1409 |
| webface4m, gallery bar 64 px | TPIR1 | 0.3 | single frame | 0.5209 | 0.4930 to 0.5436 |
| webface4m, gallery bar 64 px | TPIR1 | 0.3 | R1 tracklet | 0.6231 | 0.5985 to 0.6436 |
| webface4m, gallery bar 64 px | TPIR1 | 0.3 | R1 minus single frame | 0.1022 | 0.0682 to 0.1355 |
| webface4m, gallery bar 64 px | TPIR1 | 0.1 | single frame | 0.4442 | 0.3825 to 0.4729 |
| webface4m, gallery bar 64 px | TPIR1 | 0.1 | R1 tracklet | 0.5412 | 0.4654 to 0.5737 |
| webface4m, gallery bar 64 px | TPIR1 | 0.1 | R1 minus single frame | 0.0970 | 0.0204 to 0.1612 |
| webface4m, gallery bar 64 px | TPIR1 | 0.01 | single frame | 0.1311 | 0.1046 to 0.2182 |
| webface4m, gallery bar 64 px | TPIR1 | 0.01 | R1 tracklet | 0.1859 | 0.1518 to 0.3051 |
| webface4m, gallery bar 64 px | TPIR1 | 0.01 | R1 minus single frame | 0.0548 | -0.0295 to 0.1571 |
| webface4m, gallery bar 64 px | TPIR20 | 0.3 | single frame | 0.5470 | 0.5213 to 0.5696 |
| webface4m, gallery bar 64 px | TPIR20 | 0.3 | R1 tracklet | 0.6506 | 0.6268 to 0.6702 |
| webface4m, gallery bar 64 px | TPIR20 | 0.3 | R1 minus single frame | 0.1035 | 0.0701 to 0.1343 |
| webface4m, gallery bar 64 px | TPIR20 | 0.1 | single frame | 0.4581 | 0.3961 to 0.4867 |
| webface4m, gallery bar 64 px | TPIR20 | 0.1 | R1 tracklet | 0.5540 | 0.4757 to 0.5895 |
| webface4m, gallery bar 64 px | TPIR20 | 0.1 | R1 minus single frame | 0.0959 | 0.0170 to 0.1606 |
| webface4m, gallery bar 64 px | TPIR20 | 0.01 | single frame | 0.1349 | 0.1089 to 0.2230 |
| webface4m, gallery bar 64 px | TPIR20 | 0.01 | R1 tracklet | 0.1847 | 0.1389 to 0.2836 |
| webface4m, gallery bar 64 px | TPIR20 | 0.01 | R1 minus single frame | 0.0498 | -0.0506 to 0.1376 |

#### Backbone comparison, exploratory (not pre-registered): paired per tracklet, B = WebFace4M minus A = MS1MV2

| gallery_bar | quantity | n_tracklets | n_persons | mean_A | mean_B | B_minus_A | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|
| 48 | single frame | 3188 | 151 | 0.3473 | 0.4826 | 0.1352 | 0.1201 | 0.1539 |
| 48 | R1 tracklet | 3188 | 151 | 0.4947 | 0.6747 | 0.18 | 0.1546 | 0.2088 |
| 48 | aggregation gain (R1 minus single) | 3188 | 151 | 0.1473 | 0.1921 | 0.0448 | 0.0262 | 0.0639 |
| 64 | single frame | 2892 | 129 | 0.3471 | 0.4807 | 0.1336 | 0.1162 | 0.1514 |
| 64 | R1 tracklet | 2892 | 129 | 0.492 | 0.6802 | 0.1881 | 0.1604 | 0.2158 |
| 64 | aggregation gain (R1 minus single) | 2892 | 129 | 0.145 | 0.1995 | 0.0545 | 0.0346 | 0.0743 |

#### Part C: frame-quality reject test (AUC of error against rejected fraction, difference to random rejection)

MEVID (primary)

| dataset | signal | tracklets | overall_error | AUC | AUC_random | difference | CI95 | Bonferroni_CI | helps |
|---|---|---|---|---|---|---|---|---|---|
| MEVID (gallery bar 48 px) | n_usable_faces | 3188 | 0.5053 | 0.2130 | 0.2527 | -0.0396 | -0.0452 to -0.0335 | -0.0478 to -0.0313 | yes |
| MEVID (gallery bar 48 px) | self_consistency | 3188 | 0.5053 | 0.2444 | 0.2527 | -0.0083 | -0.0145 to -0.0022 | -0.0160 to -0.0002 | yes |
| MEVID (gallery bar 48 px) | mean_face_height | 3188 | 0.5053 | 0.2052 | 0.2527 | -0.0475 | -0.0538 to -0.0401 | -0.0563 to -0.0371 | yes |
| MEVID (gallery bar 48 px) | mean_laplacian_variance | 3188 | 0.5053 | 0.2192 | 0.2527 | -0.0335 | -0.0400 to -0.0268 | -0.0418 to -0.0245 | yes |
| MEVID (gallery bar 48 px) | mean_luminance | 3188 | 0.5053 | 0.2521 | 0.2527 | -0.0006 | -0.0089 to +0.0077 | -0.0116 to +0.0110 | no |
| MEVID (gallery bar 48 px) | mean_adaface_raw_norm | 3188 | 0.5053 | 0.2360 | 0.2527 | -0.0167 | -0.0249 to -0.0071 | -0.0279 to -0.0041 | yes |
| MEVID (gallery bar 48 px) | mean_magface_norm | 3188 | 0.5053 | 0.2049 | 0.2527 | -0.0477 | -0.0544 to -0.0417 | -0.0570 to -0.0391 | yes |
| MEVID (gallery bar 48 px) | luminance_closeness_to_midgray | 3188 | 0.5053 | 0.2520 | 0.2527 | -0.0007 | -0.0090 to +0.0078 | -0.0116 to +0.0110 | exploratory (not in the verdict) |

ChokePoint 12 and 16 px (secondary)

| dataset | signal | tracklets | overall_error | AUC | AUC_random | difference | CI95 | Bonferroni_CI | helps |
|---|---|---|---|---|---|---|---|---|---|
| ChokePoint 12 px (SC, GT tracks, both directions pooled) | n_usable_faces | 432 | 0.0671 | 0.0281 | 0.0336 | -0.0055 | -0.0162 to +0.0037 | -0.0199 to +0.0055 | no |
| ChokePoint 12 px (SC, GT tracks, both directions pooled) | self_consistency | 432 | 0.0671 | 0.0352 | 0.0336 | +0.0016 | -0.0074 to +0.0093 | -0.0093 to +0.0114 | no |
| ChokePoint 12 px (SC, GT tracks, both directions pooled) | mean_magface_norm | 432 | 0.0671 | 0.0358 | 0.0336 | +0.0022 | -0.0087 to +0.0120 | -0.0109 to +0.0135 | no |
| ChokePoint 16 px (SC, GT tracks, both directions pooled) | n_usable_faces | 432 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 to +0.0000 | +0.0000 to +0.0000 | no |
| ChokePoint 16 px (SC, GT tracks, both directions pooled) | self_consistency | 432 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 to +0.0000 | +0.0000 to +0.0000 | no |
| ChokePoint 16 px (SC, GT tracks, both directions pooled) | mean_magface_norm | 432 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 to +0.0000 | +0.0000 to +0.0000 | no |

#### Part D: tracker stress conditions with and without the face term (IDF1, HOTA, ID switches)

| lam | cond | tracker | IDF1 | HOTA | IDSW | n_tracks |
|---|---|---|---|---|---|---|
| 0.0 | S0 | botsort | 0.6991 | 0.4777 | 0.0556 | 1891 |
| 0.0 | S0 | bytetrack | 0.7058 | 0.4843 | 0.0417 | 1798 |
| 0.0 | S1 | botsort | 0.6326 | 0.4847 | 29.4583 | 10273 |
| 0.0 | S1 | bytetrack | 0.7328 | 0.5249 | 1.8472 | 1775 |
| 0.0 | S2 | botsort | 0.4268 | 0.3929 | 71.2361 | 13498 |
| 0.0 | S2 | bytetrack | 0.5583 | 0.4023 | 3.2083 | 1574 |
| 0.0 | S3 | botsort | 0.247 | 0.3204 | 65.3056 | 10478 |
| 0.0 | S3 | bytetrack | 0.2416 | 0.1787 | 1.2222 | 750 |
| 0.0 | S4 | botsort | 0.675 | 0.4491 | 0.2917 | 2118 |
| 0.0 | S4 | bytetrack | 0.682 | 0.4551 | 0.1528 | 1835 |
| 0.0 | S5 | botsort | 0.6441 | 0.4168 | 0.9444 | 2412 |
| 0.0 | S5 | bytetrack | 0.6516 | 0.4221 | 0.2917 | 1891 |
| 0.5 | S0 | botsort | 0.6992 | 0.4777 | 0.0 | 1854 |
| 0.5 | S0 | bytetrack | 0.7058 | 0.4843 | 0.0 | 1780 |
| 0.5 | S1 | botsort | 0.7124 | 0.5021 | 0.0417 | 4795 |
| 0.5 | S1 | bytetrack | 0.7543 | 0.5295 | 0.0139 | 1700 |
| 0.5 | S2 | botsort | 0.7156 | 0.5163 | 0.2083 | 4777 |
| 0.5 | S2 | bytetrack | 0.7842 | 0.5555 | 0.0278 | 1579 |
| 0.5 | S3 | botsort | 0.7061 | 0.529 | 1.3333 | 3995 |
| 0.5 | S3 | bytetrack | 0.8056 | 0.5765 | 0.0556 | 1407 |
| 0.5 | S4 | botsort | 0.6748 | 0.4494 | 0.2361 | 2074 |
| 0.5 | S4 | bytetrack | 0.6818 | 0.4554 | 0.0833 | 1793 |
| 0.5 | S5 | botsort | 0.6457 | 0.4172 | 0.6389 | 2328 |
| 0.5 | S5 | bytetrack | 0.6529 | 0.4225 | 0.0556 | 1817 |

#### Part D: downstream open-set track-identity accuracy and the decision

| lambda_ | condition | ByteTrack | BoT_SORT | difference |
|---|---|---|---|---|
| 0.0 | S0 | 0.8638 | 0.8629 | +0.0009 |
| 0.0 | S1 | 0.9033 | 0.9184 | -0.0151 |
| 0.0 | S2 | 0.9126 | 0.9542 | -0.0416 |
| 0.0 | S3 | 0.7883 | 0.9620 | -0.1737 |
| 0.0 | S4 | 0.8636 | 0.8629 | +0.0007 |
| 0.0 | S5 | 0.8643 | 0.8633 | +0.0009 |
| 0.5 | S0 | 0.8637 | 0.8628 | +0.0009 |
| 0.5 | S1 | 0.8981 | 0.8980 | +0.0000 |
| 0.5 | S2 | 0.8981 | 0.8985 | -0.0004 |
| 0.5 | S3 | 0.8981 | 0.8995 | -0.0014 |
| 0.5 | S4 | 0.8636 | 0.8631 | +0.0005 |
| 0.5 | S5 | 0.8625 | 0.8840 | -0.0214 |

lambda 0: mean per-clip difference ByteTrack minus BoT-SORT over S0 to S5 = -0.0380 (95% CI -0.0576 to -0.0210, 24 clips); CI excludes zero: True.

lambda 0.5: mean per-clip difference ByteTrack minus BoT-SORT over S0 to S5 = -0.0036 (95% CI -0.0106 to +0.0002, 24 clips); CI excludes zero: False.

**Decision by the rule fixed in section 0: bytetrack (default).**

#### Acceptance checks

- `check1_counts`: {"all_pass_when_the_empty_row_is_excluded": true, "checkin_equal_probe_inventory": true, "checkin_persons": 172, "checkin_photos": 2898, "empty_tracklets": [{"end": 1128476, "person": 78, "start": 1128477, "tracklet": "tr01889"}], "persons": 158, "persons_equal_readme": true, "readme_tracklets": 8092, "tracklet_rows": 8093, "tracklets_with_chips": 8092, "tra
- `check2_probe_boxes_reproduced`: {"chips": 3949, "faces_here": 1489, "faces_in_probe": 1489, "identical": true, "max_abs_difference_in_box_or_score_after_rounding_to_3_decimals": 0.0, "same_face_flag_for_every_chip": true, "seconds": 118.2}
- `check3_aggregation_rules_on_real_tracklet`: {"frame_order_invariance": true, "length1_equals_single_frame": true}
- `check4_open_set_folds`: {"each_person_held_out_exactly_once": true, "folds": 5, "gallery_and_nonenrolled_disjoint": true, "persons_in_population": 151}
- `check5_lambda0_reproduces_phase7_tracker_results`: {"byte_identical": true, "lambda0_per_clip_sha256": "5097c822846692c8476c9dca7551d0e1428c4585fe900725178dc5ca48f2ad3a", "phase7_per_clip_sha256": "5097c822846692c8476c9dca7551d0e1428c4585fe900725178dc5ca48f2ad3a", "rows_lambda0": 864, "rows_phase7": 864}
- `check6_cpu_parts_rerun_identical`: {"byte_identical": ["backbone_pair", "camera_*", "closed_records_* (4)", "closed_webface4m", "enrolment_*", "folds_*", "height_*", "no_photo_*", "open_adaface", "open_webface4m", "quality_mevid(+curves)", "quality_chokepoint(+curves)", "tracker_identity_per_clip", "tracker_face/lam0.5/per_clip.csv", "tracker_face/lam0.5/tracks/*"], "not_byte_identical_explai
- `check6_partB_adaface_rerun_identical`: {"all_identical": true, "files": {"camera_adaface.csv": true, "closed_adaface.csv": true, "closed_records_adaface_bar48.csv": true, "closed_records_adaface_bar64.csv": true, "enrolment_adaface.csv": true, "folds_adaface.csv": true, "height_adaface.csv": true, "no_photo_adaface.csv": true, "open_adaface.csv": true}, "note": "the second run added the Bonferron

<!-- TABLES:END -->

## 14. Decisions for Phase 9
1. **Tracker: ByteTrack stays the default** by the pre-registered rule, and the **face term (lambda 0.5, gate: score >= 0.5 and native height >= 32 px) should be adopted** in the pipeline: it removes the fragility at reduced frame rates for both trackers (IDF1 at every 10th frame 0.81 and 0.71 against 0.24 and 0.25) and costs nothing on the original frames. The evidence is within ChokePoint (not held out) and uses the plan's fallback gate; the demo should keep a frame rate of at least 10 frames per second regardless. For Abser to confirm.
2. **Quality bar for the demo: mean face height (from the detector, no extra network) or the MagFace norm; they tie** (rejecting the lowest half of the tracklets takes the closed-set error from 50.5% to 34.5% and 34.4%; rejecting the lowest quarter to 40.3%). The rejected fraction is a policy choice (it trades coverage for accuracy) and the threshold must be set in Phase 9 and pre-registered. Luminance and self-consistency are not worth using.
3. **Aggregation rule: R1 is the reference; R4* (+0.0235) and R2 (+0.0091) beat it on real faces** (Bonferroni CIs above zero) and the MagFace norm they use is now supported as a quality signal. R4* needs the MagFace norm of every face (one more network pass). Phase 7 found no rule better than R1 on synthetic and cropped data, so the benefit is specific to real low resolution and was found on MEVID itself (not held out); a confirmation on the Phase 9 demo data is advisable. Decision for Abser.
4. **Backbone: WebFace4M is much stronger on real low-resolution faces** (exploratory pairing: single frame +0.135, R1 +0.180 Rank-1). The choice of backbone for the final pipeline is a decision with the supervisor; the licence terms of the WebFace4M weights are to be checked (MODEL_INVENTORY.md records the source).
5. **Detector mode** is configured per camera from a calibration clip (`scripts/video_mode_calibrate.py`, Phase 7 decision 5). Phase 8 gives no new evidence; MEVID chips were detected at size 640.
6. **Open-set operating point:** at FPIR 0.1 the TPIR1 of R1 is 0.374 (MS1MV2) and 0.543 (WebFace4M) on real faces; FPIR 0.01 is not reliable at this data size. The operating point of the demo is to be set at FPIR 0.1 or higher unless more non-enrolled identities become available.
7. **Still deferred from Phase 7:** the YouTube Faces usage terms in `DATASETS.md`; appendix T2 (custom video).

---
**STOP.** Phase 8 ends here for the supervisor. Phase 9 starts after the decisions in section 14.
