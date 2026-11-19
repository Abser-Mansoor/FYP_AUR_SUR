# REPORT — Phase 7: per-track aggregation (ChokePoint + YouTube Faces, plus a CAFace arm)

**Status:** results complete, pending supervisor review (see section 1). This file starts with the pre-registered protocol (section 0), committed on branch `phase_7` before any Phase 7 number was computed (see the git history of this file). Plan: `research/Sprint_Notes/Phase7_aggregation_plan.md` (approved 2026-11-17). Day-1 intake notes (YouTube Faces layout, norm pooling, CAFace interface) are in section 0.9.

## 0. Pre-registered protocol
Question: does combining the embeddings of a track beat a single frame, and which rule is best? All numbers: lab PC (RTX 4000 Ada), inference only, seed 42, every number from a script output. Headroom (Phase 6, section 6): single frames are at ceiling on native ChokePoint; the ChokePoint part is judged on probes degraded to 12 px and 16 px and on a one-frame gallery. YouTube Faces (YTF) is a second dataset with the standard video verification protocol.

### 0.1 Unit of analysis (ChokePoint)
- **GT track:** all matched detections (IoU >= 0.5, Phase 6 matching) of one (clip, person id).
- **Predicted track:** one (clip, track id) of ByteTrack or BoT-SORT, using all its detections at score >= 0.5 (the embedding floor), including unmatched ones, because a deployed system cannot tell which are annotated. Its label is the majority person id of its matched detections; **track purity** (share of that id among its matched detections) is reported. Tracks with fewer than 50% matched detections are counted and reported but not scored (Phase 6 study C: most of them are unannotated faces and reflections).
- **Paired comparison per track:** single-frame accuracy = the mean over the track's frames of frame-level correctness (the expected accuracy of one random frame); aggregated accuracy = correctness of the track's one aggregated template. Delta per track; **cluster bootstrap over identities** (1,000 draws, seed 42). Frame-level rows from Phase 6 (E) are shown as reference.
- **Groups, directions, gallery:** as Phase 6 section 0: G1 enrols and G2 probes, and the reverse; SC and MC; gallery = mean of all matched native AdaFace embeddings of the enrolling group; the six left-out identities (rng 42) for the open set; results are the mean of the two directions.

### 0.2 Rules
Per track: input is `emb[N,512]` (L2-normalised AdaFace, MS1MV2) and quality `q[N]` (MagFace raw norm of the same face); output is one L2-normalised template. Ties in any ordering are broken by frame index.
| Rule | Definition |
|---|---|
| R0 | single frame (reference) |
| **R1** | **plain mean (primary rule)** |
| R2 | mean weighted by `q` |
| R3 | mean of the top-k frames by `q`, k in {1, 3, 5, 10} (all reported, none selected; k = N when the track has fewer frames) |
| R4 | norm pooling (Template-based Multi-Domain Face Recognition, arXiv 2409.09832, Eq. 4-6): template = sum_i c_i e_i with c_i = softmax(lambda * l_i), l_i = q_i / max_j q_j; lambda = 50 (the value the paper uses for AdaFace). **The norm is the MagFace raw norm, not AdaFace's own norm** (not cached); the row is labelled so |
| R4* | the paper's min-max variant for narrow norm distributions (Eq. 7): l_i = (q_i - min q) / (max q - min q), lambda = 5 (middle of the paper's 3 to 10 range for AdaFace-type norms) |
| R5 | score-level mean: the mean cosine of the track's frames to each gallery template; identity = argmax |
| R6 | majority vote of per-frame Rank-1 identities (ties to the higher mean score); closed set only |
Sensitivity (descriptive, not selected): R4 with lambda in {10, 50, 100}. **No hyperparameter is tuned**: with 29 identities a tuning split is not credible; every k and the lambda values above are fixed here.
**Multiplicity:** Bonferroni-adjusted CIs (level 1 - 0.05/9) are reported next to the plain 95% CIs for the nine comparisons with R1: R2, R3 (four k), R4, R4*, R5, R6.
R4 uses the paper's formula as written in the paper; whether the paper L2-normalises the features before pooling is not stated in the text we could read, so ours uses L2-normalised embeddings with the norm weights above and says so.

### 0.3 Primary hypothesis (ChokePoint)
Condition: AdaFace, GT tracks, SC, probes degraded to 12 px (Phase 6 protocol E: area downsampling then bilinear upsampling of the aligned crop). Test: R1 aggregated Rank-1 minus single-frame Rank-1, paired per track, mean of the two directions. **"Aggregation helps" if the 95% CI of the difference is above zero.**
Secondary conditions (reported as they come): 16 px; native; MC; predicted tracks (both trackers); B with k = 1 gallery and aggregated probes; open-set TPIR20 at FPIR 0.3, 0.1 and 0.01 (unmated tracks come from the 6 left-out identities, so these CIs will be wide).
**Degraded tracks** use the Phase 6 E subsample (every 5th matched detection per clip in frame order), so the 12 and 16 px tracks have about every 5th frame; the single-frame row then equals E exactly and no new GPU work is needed. For predicted tracks at degraded sizes only the subsampled matched detections exist; this is a limitation.

### 0.4 Curves (descriptive)
Accuracy against the number of frames used, n in {1, 2, 4, 8, 16, all}: random frame subsets per track (20 seeded draws); and accuracy against the track's mean MagFace norm (quartiles).

### 0.5 YouTube Faces protocol
Official splits: `meta_data/meta_and_splits.mat` (`Splits`: 500 pairs x 3 x 10 folds = 5,000 pairs, 250 same-person pairs per fold; the three columns are 1-based video index 1, video index 2, same-person flag; checked: label equality equals the flag for all 5,000 pairs). Unsupervised setting, no training.
- **Frames per video:** up to 50 evenly spaced frames among the video's labelled frames (`<person>.labeled_faces.txt`; boxes are centre x, centre y, width, height, checked on overlays). The subject is the SCRFD detection with IoU >= 0.5 to the labelled box; frames with no such detection are skipped and counted (per video and in total). **Interpretation note:** the plan's wording was "evenly spaced over the frames where the subject is found"; to avoid running the detector on all 621,126 frames, the spacing is taken over the labelled frames and the skipped frames are counted.
- **Detector mode: `full640`** (Phase 6 Appendix T1: `full1280` finds 5.0% and `full640` 98.1% of cropped YTF faces). This resolves Phase 6 decision 3 as "the mode follows face size; YTF uses `full640`, ChokePoint keeps `full1280`".
- **Pair score:** cosine of the two videos' templates (R1 to R4*); for R5 the mean cosine over all pairs of frames of the two videos; R6 is not used on YTF.
- **Single-frame reference:** one random frame per video, 20 seeded draws; the mean and range are reported.
- **Metrics:** verification accuracy with the threshold chosen on the nine other folds (mean +/- standard error over the ten folds); AUC; TAR@FAR 0.01 and 0.1 pooled over folds; paired difference against the single-frame reference per pair, with a bootstrap over pairs grouped by identity (1,000 draws, seed 42).
- **Primary YTF hypothesis:** R1 minus single-frame accuracy, 95% CI above zero.

### 0.6 CAFace arm (never pooled with the MS1MV2 AdaFace rows)
CAFace (github.com/mk-minchul/CAFace, commit e94ad0206d0a04a098594747138e4da5435f5ec7; `CAFace_AdaFaceWebFace4M.ckpt`) has its own backbone, AdaFace R100 trained on WebFace4M, and needs that backbone's **intermediate style features** (layers 3 and 5, `intermediate_type: style`; not only the 512-d embedding) for its aggregator, so the cached MS1MV2 embeddings cannot be used: the faces are re-embedded with the CAFace backbone. Its own rows: single frame, plain mean and the CAFace aggregator, from that backbone; CAFace is compared only against those rows. Runs: ChokePoint GT tracks (native, 16 px, 12 px, same subsample as 0.3) and YTF. Inference only; the center file (`center_WebFace4MAdaFace_webface4m_subset.pth`) is a training artefact and is not loaded.
If CAFace cannot run inference-only from the official weights, the arm is dropped with the reason recorded.

### 0.7 Not in Phase 7
Study C as unmated probes (needs a separately pre-registered identity-level labelling); Appendix T2 (custom video, consent); MagFace and CurricularFace rows (AdaFace only; they may be added from cache for native ChokePoint if time remains).

### 0.8 Acceptance checks (written to `outputs/video/agg7/checks.json`)
1. A track of length 1 gives the single-frame result for every rule; top-k with k = N equals the plain mean.
2. Shuffling the frames inside a track does not change R1 to R5.
3. The ChokePoint single-frame reference reproduces Phase 6 E (12 and 16 px) and section 0 (native) to the last digit.
4. YTF counts match the official ones (3,425 videos, 1,595 people, 5,000 pairs, 10 folds), or the mismatch is reported; every pair has both templates, or its failure is logged.
5. CAFace: no official example ships (the repository's `demo/run.sh` points at `examples/example1/probe.mp4`, which is not in the repository); instead, the CAFace backbone's single-frame features are compared with the standalone AdaFace WebFace4M checkpoint (`adaface_ir101_webface4m.ckpt`) through our own recognizer path on 100 seeded faces (cosine, reported as found), and the fusion code runs on a length-1 track.

### 0.9 Day-1 intake notes
- **YouTube Faces held (2026-11-18).** Archives in the team's Downloads folder; listing of `frame_images_DB.tar.gz`: 3,425 video folders, 1,595 person folders, 621,126 JPEG frames (the official frame count), 3,083 text files; `meta_data.tar.gz` holds `meta_and_splits.mat`; `headpose_DB` lists 3,425 files. SHA-256 of every archive and the extraction log are in `outputs/video/ytf/`. Frames are 256x192 to 384x288 pixels.
- **Norm pooling.** The paper (arXiv 2409.09832, read from the arXiv HTML page) defines the weights as softmax of lambda times the max-normalised feature norm (Eq. 5 and 6), lambda 50 for AdaFace and 10 for ArcFace and CosFace in its tables, and a min-max variant (Eq. 7) with lambda 3 to 10 for narrow norm distributions such as AdaFace. A sparse variant (sparsemax, Eq. 8-9) is not used here.
- **CAFace.** Checked from the repository code (`demo/main.py`, `demo/inference.py`, `demo/model_loader.py`): the aggregator takes the backbone's features times their norms and a compressed intermediate map or style feature; the center file is not read at inference (`load_caface` expects exactly one unexpected key, `center.weight`, in the checkpoint). The checkpoint (`CAFace_AdaFaceWebFace4M.ckpt`: keys `state_dict` and `hyper_parameters`; arch `ir_101`, aggregator `style_norm_srm`, style index [3, 5], memory on; the state dict holds backbone, aggregator and `center.weight`) loads with `torch.load(..., weights_only=False)` without pytorch-lightning. The repository pins PyTorch 1.10 and pytorch-lightning 1.7.1; the arm runs from a separate environment (`envs/caface`, torch 2.14.1+cu126, CUDA available) with current PyTorch, which is a deviation recorded here.
- **Addendum 1 (2026-11-19, written after the Phase 7 rows of both backbones were known; the comparison below is therefore exploratory and is labelled so wherever it is quoted).** Abser asked for a controlled comparison of the two backbones (decision 6 of section 12). Both are AdaFace R100 (same architecture and loss); they differ in training data: A = MS1MV2 (the weights of all other rows), B = WebFace4M (the CAFace backbone). Same crops, same GT tracks and the same E subsample (same matched detections), same gallery protocol (native, mean template per identity), same rules: for each of single frame and R1, the paired per-track difference B minus A, and the difference between the aggregation gains (R1 minus single frame) of B and A, at native, 16 and 12 px, SC and MC, mean of the two directions, cluster bootstrap over identities (1,000 draws, seed 42); on YouTube Faces the same on the 4,986 pairs with the same random frames for both backbones (20 seeded draws), bootstrap over persons. No other rule is run for B (its norm is AdaFace's own norm, not the MagFace norm used for A). Script: `video_aggregation.py --part backbone`; outputs `agg7/backbone_chokepoint.csv` and `agg7/backbone_ytf.csv`.
- **Addendum 2 (2026-11-19, written before the stress test was run; Abser approved the test and its decision rule): tracker stress test.** Purpose: a tiebreak for the tracker default (ByteTrack against BoT-SORT). Phase 6 gave no pooled difference; a paired per-clip analysis of the Phase 6 results (section 12, item 3) shows ByteTrack ahead by 0.007 on HOTA and IDF1; neither says how the trackers behave when conditions get worse. Inputs: the Phase 6 detections (SCRFD `full1280`, score from 0.1 up) of the 72 ground-truth clips and the Phase 6 ground truth; both trackers with the Phase 6 settings (`src/video/track.py` defaults, re-ID off, camera-motion compensation off, `match_iou` 0.3). Conditions: S0 original (reference only, not used in the decision); S1, S2, S3 keep every 3rd, 5th, 10th frame (tracker frame rate set to 30 divided by k, so a lost track survives about one second; the ground truth is scored on the kept frames only); S4 and S5 drop each detection independently with probability 0.1 and 0.2 (all frames kept; `numpy.random.default_rng([42, clip index, 100 p])`, one mask per clip and condition shared by both trackers). Metrics per clip and condition: IDF1, HOTA, AssA, DetA, ID switches and the number of tracks, from the same TrackEval code as Phase 6. **Primary statistic:** the per-clip IDF1 difference ByteTrack minus BoT-SORT, averaged over S1 to S5, then the mean over the 72 clips with a cluster bootstrap over clips (1,000 draws, seed 42). **Decision rule (fixed now):** if the 95% CI of that mean excludes zero, the tracker with the higher IDF1 becomes the provisional Phase 7 default; otherwise ByteTrack (simpler, no camera-motion state, same cost). Per-condition rows are reported descriptively. The choice stays provisional until the Phase 8 open-set test (face-aware association on open-set data: PaSC if the licence is obtained, otherwise ChokePoint with unmated identities), whose own decision rule is registered in Phase 8 before it runs. Script: `scripts/video_tracker_stress.py`; outputs `outputs/video/agg7/tracker_stress/`. Limitations: synthetic subsampling and dropout, detections from `full1280`, 72 clips of 29 identities. The script stores the SHA-256 of this paragraph with its results.
- **Confirmations (Abser, 2026-11-19), given before any Phase 7 run:** (1) the YouTube Faces detector mode is `full640` (ChokePoint keeps `full1280`); (2) "evenly spaced over the frames" means up to 50 evenly spaced frames among the video's labelled frames (section 0.5); (3) R4 (lambda 50), R4* (lambda 5), the lambda sensitivity set and the Bonferroni correction over nine comparisons as written in section 0.2; (4) the primary hypotheses of sections 0.3 and 0.5 as written. The YouTube Faces usage terms are not yet recorded in `DATASETS.md` (internet bandwidth problem); they will be added later.
- **Extraction note.** The first extraction of `frame_images_DB.tar.gz` was cut off when a session ended (682 of 1,595 person folders). The last folder being written (`Gerard_Kleisterlee`) was deleted and the archive was extracted again with `--skip-old-files`; the final count of frames (621,126 expected) is checked by `video_ytf_prepare.py` before any YouTube Faces run.

---

## 1. Summary
**Status:** results complete; pending supervisor review. Protocol: section 0, committed (fc175bf) and confirmed by Abser on 2026-11-19 before any Phase 7 number.
- **Aggregation helps where single frames fail (primary hypothesis supported).** ChokePoint, AdaFace, GT tracks, SC, probes degraded to 12 px: Rank-1 0.835 for a single frame and 0.933 for the plain mean of a track, difference **+0.098 (95% CI +0.071 to +0.125)**. The same holds in MC (+0.087), with a one-frame gallery (+0.111) and on predicted ByteTrack and BoT-SORT tracks (+0.099 each). At 16 px (+0.006) and native size (+0.0002) it removes the last errors; the native baseline is at ceiling.
- **YouTube Faces (second primary hypothesis supported, small).** Verification accuracy 0.9567 for a single random frame and 0.9657 for the plain mean, difference **+0.0090 (95% CI +0.0063 to +0.0117)**, 4,986 of 5,000 official pairs scored. The single frame is already at 95.7%, so the gain is under one point.
- **The plain mean is the best rule.** No rule is shown to beat it on either dataset. Taking only the highest-norm frame is worse; norm pooling, norm weighting and top-k selection are no better; score-level averaging equals the mean for Rank-1 by construction. The MagFace norm is not a useful frame-quality signal here.
- **CAFace (own WebFace4M backbone, never pooled with the MS1MV2 rows).** Its learned aggregator is not better than the plain mean of its own features (YouTube Faces -0.0002, CI -0.0017 to +0.0012; ChokePoint 12 px SC -0.0069, CI -0.0160 to 0.0000). The backbone changes the single-frame result far more than the aggregation does.
- **Curves.** At 12 px accuracy rises from 0.832 (1 frame) to 0.916 (4) and 0.930 (8) and then plateaus near 0.93: the remaining errors are identity-level and averaging does not repair them.
- **Open set is inconclusive** (6 unmated identities).
- **Tracker default (provisional): ByteTrack.** The pre-registered stress test (frame subsampling and detection dropout) gives a mean IDF1 difference of +0.048 for ByteTrack (95% CI +0.035 to +0.061; better in 58 of 72 clips); with the vendored defaults BoT-SORT fragments tracks at reduced frame rates, ByteTrack loses detections at 6 and 3 frames per second. A Phase 8 open-set test decides finally (section 7b).
- **Checks:** all five pass (section 3); determinism confirmed for the aggregation outputs and one YouTube Faces video.
- **Main limits:** 29 ChokePoint identities; the 12 and 16 px probes are synthetic resolution loss on every 5th frame; YouTube Faces is celebrity video (section 11).

## 2. What was built
- `src/video/aggregate.py`: the embedding rules R1 to R4* (one signature, frames put in frame-index order first, so the result does not depend on the order frames are passed in) and the score-level rules R5 and R6. Pure NumPy.
- `scripts/video_tracker_stress.py` (tracker stress test, Addendum 2) and `scripts/video_mode_calibrate.py` (detector mode from a calibration clip, decision 5).
- `scripts/video_aggregation.py` (`--part chokepoint | predicted | curves | thin | ytf | caface | caface-ytf | backbone`): GT and predicted tracks, paired differences with cluster bootstrap over identities, curves, YouTube Faces verification, CAFace rows. All ChokePoint parts run on CPU from cached embeddings.
- `scripts/video_ytf_prepare.py` and `scripts/video_ytf_embed.py`: the YouTube Faces frame choice, SCRFD `full640` detection of the subject and AdaFace embedding (also saves the keypoints, so the CAFace arm aligns the same faces).
- `scripts/video_caface.py` (`envs/caface`): the CAFace backbone and aggregator, ChokePoint and YouTube Faces embedding, acceptance check 5.
- `scripts/video_agg_checks.py`, `scripts/phase7_report_tables.py`; `configs/video.yaml` (block `ytf`); environment `envs/caface` (torch 2.14.1+cu126, timm 0.9.16, einops, omegaconf).

## 3. Acceptance checks
Written to `outputs/video/agg7/checks.json`; see the checks entry in section 12.
1. A track of length 1 gives the single-frame result for every rule, and top-k with k = N equals the plain mean (synthetic tracks): pass.
2. Shuffling the frames inside a track does not change R1 to R5 or R6 (bitwise, 20 shuffles of tracks of 3, 7 and 25 frames, ties included): pass.
3. The single-frame reference (frame-level R0 through the Phase 6 functions) reproduces Phase 6 section 0 (native) and E (16 and 12 px) to the stored 5 decimals for Rank-1 and TPIR20 at FPIR 0.3, 0.1, 0.01, both directions, SC and MC (12 rows): pass.
4. Counts and determinism. The YouTube Faces counts are in section 8. Determinism: the ChokePoint aggregation was run twice (with a code change between the runs that added the rule-minus-R1 columns); every column of the first run's six result files is identical in the second run. One YouTube Faces video (video 1) rerun end to end gives identical arrays (section 8).
5. CAFace: the CAFace backbone (inside `CAFace_AdaFaceWebFace4M.ckpt`) and the standalone AdaFace WebFace4M checkpoint, both on the same 100 seeded aligned faces, give a cosine of at least 0.9999998 on every face (median 1.0000): the backbone is that model, and our crops and normalisation match the CAFace code path. The fusion code gives a length-1 track its own feature (checked).

## 4. ChokePoint: primary and secondary results (GT tracks, AdaFace)
Single-frame accuracy here is the expected accuracy of one random frame of a track, averaged over tracks (every track counts once); it differs slightly from the frame-weighted Phase 6 E rows (SC 12 px: 0.8347 against 0.8387) because long tracks weigh less. Differences are aggregated minus single frame, paired per track, mean of the two directions, 95% CI from the cluster bootstrap over identities.
- **Primary hypothesis supported.** AdaFace, GT tracks, SC, probes at 12 px: single frame 0.8347, R1 (plain mean) 0.9329, **difference +0.0982 (95% CI +0.0712 to +0.1251)**; the CI is above zero. The Bonferroni CI (nine comparisons) is +0.0614 to +0.1426. Both directions agree (+0.0904 for G1 to G2, +0.1059 for G2 to G1).
- **Where it helps.** At 16 px the difference is +0.0064 (+0.0029 to +0.0105; 0.9936 to 1.0000), at native size +0.0002 (0.0000 to +0.0004; 0.9998 to 1.0000). Aggregation only has room where single frames fail (Phase 6 section 6); at native and 16 px it removes the last errors.
- **MC.** 12 px: 0.8518 to 0.9390, +0.0872 (+0.0639 to +0.1114); 16 px +0.0050 (+0.0023 to +0.0079); native +0.0004 (+0.0001 to +0.0008).
- **One-frame gallery (B, k = 1, 20 gallery draws).** The gain is at least as large: SC 12 px 0.6304 to 0.7418, +0.1114 (+0.0971 to +0.1255); 16 px +0.0399 (+0.0318 to +0.0491); native +0.0037 (+0.0027 to +0.0047); MC 12 px +0.1056 (+0.0943 to +0.1162).
- **Open set (TPIR20, 6 left-out identities).** At 12 px SC the point estimates rise from 0.786 to 0.917 (FPIR 0.3), 0.633 to 0.845 (0.1) and 0.368 to 0.652 (0.01); the CIs overlap (at 0.01: 0.160 to 0.699 for single frames, 0.500 to 0.956 for R1; MC 0.360 to 0.516). With six unmated identities the open-set comparison cannot show a significant difference; it is reported as found. At native size R1 gives TPIR 1.000 at all three FPIR values (SC); at 16 px 1.000 (SC) and 0.994 at FPIR 0.01 (MC).

## 5. Rules compared (pre-registered comparison with R1, Bonferroni over nine comparisons)
Tables in section 12 (differences against the single frame, and against R1). At 12 px SC (mean of the two directions):
- **No rule beats the plain mean R1.** The differences to R1 are zero or negative for every rule; the Bonferroni 99.4% CIs exclude zero (rule worse than R1) for R3 with k = 1 (-0.116, CI -0.168 to -0.065), R3 with k = 3 (-0.039, -0.084 to -0.002), R4 norm pooling with lambda 50 (-0.039, -0.074 to -0.003) and, as a sensitivity row, lambda 100 (-0.065, -0.111 to -0.028). R3 with k = 5 and k = 10, R4* (min-max, lambda 5), R6 (vote) and R4 with lambda 10 are not distinguishable from R1.
- **R2 and R5 are not independent evidence.** R5 (mean score to each gallery template) gives the same Rank-1 as R1 by construction: the mean of cosines to unit-norm templates is the dot product of the (unnormalised) mean embedding with each template, so the arg max is the same as for the normalised mean; they differ only in open-set scores and on YouTube Faces. R2 (weights proportional to the MagFace norm) is identical to R1 in the SC decisions at 12 px and within 0.0000 of it in MC: the norm varies little between frames of one track (about 22 to 26), so the weights are almost uniform.
- **Quality-based selection does not help.** Taking the single highest-norm frame (R3 k = 1) is no better than a random frame (-0.018, CI -0.040 to +0.006 against the single frame); using more frames (k = 3, 5, 10) climbs towards the plain mean (+0.059, +0.094, +0.096). Norm pooling concentrates the weight on few frames when lambda is large and loses accordingly (lambda 100: +0.033; lambda 50: +0.059); with lambda 10 it equals the mean (+0.1005). The MagFace norm of the degraded crops therefore is not a useful frame-quality signal for this recognizer here (see also the quartile table).
- Same pattern at 16 px (all rules except k = 1 and lambda 100 give the same gain) and in MC.

## 6. Curves (descriptive, SC)
- **Frames used (R1, 20 seeded random subsets per track):** at 12 px accuracy rises from 0.832 (1 frame) to 0.887 (2), 0.916 (4), 0.930 (8), 0.934 (16) and 0.933 (all); most of the gain comes within 4 to 8 frames and then stops. At 16 px: 0.993 (1), 0.999 (2), 1.000 (4 and more); native: 0.9999 (1), 1.000 from 2 frames.
- **The plateau at about 0.93 for 12 px is systematic:** the remaining errors are tracks whose frames agree on the wrong identity, so more frames do not repair them; averaging removes frame-level noise, not identity-level confusion.
- **Track quality:** by quartile of the track's mean MagFace norm (12 px, SC; 108 tracks each), single-frame accuracy is 0.836, 0.852, 0.862 and 0.790 and R1 accuracy 0.954, 0.926, 0.944 and 0.907 (lowest to highest norm): the norm does not order accuracy, and the highest-norm quartile is the worst.

## 7. Predicted tracks (ByteTrack and BoT-SORT) against GT tracks
- Same result on predicted tracks: at 12 px SC, ByteTrack 0.8399 to 0.9387, **+0.0988 (+0.0704 to +0.1263)**; BoT-SORT 0.8385 to 0.9378, **+0.0993 (+0.0705 to +0.1271)**; MC +0.0931 and +0.0939 (both CIs above zero). At 16 px +0.0060 for both; native +0.0067 and +0.0084. The aggregation gain does not depend on the tracker (the differences are 0.0005 or less at 12 px). On the tracking metrics themselves a paired per-clip comparison of the Phase 6 results (section 12, item 3) shows ByteTrack slightly but consistently ahead, which the unpaired pooled CIs of Phase 6 could not show.
- **Track composition.** A predicted track uses all its detections at score >= 0.5, matched or not. The scored tracks are pure (mean purity 1.000: all matched detections of a scored track carry one identity). Tracks with fewer than 50% matched detections are not scored: for ByteTrack (native, SC) 140 of 328 tracks in the G1 to G2 probe clips (27.9% of their detections) and 69 of 285 (8.4%) in G2 to G1; for BoT-SORT 160 of 342 (31.0%) and 85 of 301 (8.7%). These are the unannotated faces and reflections of Phase 6 study C; they are counted, not scored, and this is a limitation of the predicted-track result (it says nothing about those tracks).
- At degraded sizes only the subsampled matched detections of a track exist (section 0.3), so the predicted-track rows at 16 and 12 px describe the matched detections of each scored track.

## 7b. Tracker stress test (Addendum 2)
Run as pre-registered (`scripts/video_tracker_stress.py`; the SHA-256 of the protocol paragraph is stored in `outputs/video/agg7/tracker_stress/decision.json`). Check: condition S0 reproduces the Phase 6 per-clip results exactly (maximum absolute difference 0.0 on IDF1, HOTA, AssA, DetA and ID switches, 144 rows).
- **Decision statistic.** The mean over clips of the per-clip IDF1 difference ByteTrack minus BoT-SORT, averaged over S1 to S5, is **+0.0482 (95% CI +0.0351 to +0.0609)**; ByteTrack is better in 58 of the 72 clips and BoT-SORT in 14. The CI excludes zero and ByteTrack has the higher IDF1, so by the rule fixed in advance **ByteTrack is the provisional default**.
- **By condition** (mean IDF1 ByteTrack against BoT-SORT; predicted tracks; ID switches in total): S0 original 0.706 against 0.699 (1,798 against 1,891 tracks; 3 against 4); S1 every 3rd frame 0.733 against 0.633 (1,775 against 10,273; 133 against 2,121); S2 every 5th frame 0.558 against 0.427 (1,574 against 13,498; 231 against 5,129); S3 every 10th frame 0.242 against 0.247 (750 against 10,478; 88 against 4,702; BoT-SORT's IDF1 is marginally higher and its HOTA is 0.142 higher, because ByteTrack's detection accuracy collapses, DetA 0.146 against 0.472, while BoT-SORT makes 14 times as many tracks); S4 10% of detections dropped 0.682 against 0.675 (1,835 against 2,118; 11 against 21); S5 20% dropped 0.652 against 0.644 (1,891 against 2,412; 21 against 68).
- **Reading: two different failures at reduced frame rates.** With the vendored defaults BoT-SORT breaks the same people into many tracks as soon as the frame rate drops (about six times as many tracks as ByteTrack at every 3rd frame, nine times at every 5th; 16 to 53 times as many ID switches), so its association accuracy falls (AssA 0.527 against 0.570 at every 3rd frame, 0.371 against 0.457 at every 5th). ByteTrack keeps its tracks together but outputs only detections that belong to a confirmed track, and at every 5th and 10th frame most new tracks are never confirmed: its detection accuracy falls to 0.365 and 0.146 (BoT-SORT 0.467 and 0.472). The pre-registered statistic (IDF1) is dominated by the identity errors and favours ByteTrack at every 3rd and 5th frame (+0.100, +0.132); HOTA, which also counts missed detections, favours ByteTrack by 0.040 and 0.009 there and BoT-SORT by 0.142 at every 10th frame (3 frames per second), where both trackers are unusable (IDF1 about 0.24). Under detection dropout the two stay close, with ByteTrack slightly ahead (IDF1 +0.007) and with fewer tracks and ID switches. For the face pipeline both failures cost frames per track (fragmented tracks give short tracks, lost detections give no track), so neither tracker should be used at 6 frames per second or less without a new test.
- **Cause not identified.** An exploratory run tested one explanation: BoT-SORT's `proximity_thresh` gate (IoU of at least 0.5 to associate, against about 0.2 for ByteTrack). With the gate relaxed to 0.2 every result is identical to the default, because the gate only acts when Re-ID is on and Re-ID is off here; the explanation is rejected. The Kalman state parametrisation (BoT-SORT uses width and height, ByteTrack aspect ratio and height) is another candidate and was not tested. The result is therefore a statement about the two trackers with the vendored default parameters and our settings, not about the methods.
- **Status.** The choice is provisional. The Phase 8 open-set test (face-aware association on open-set data) decides finally, with its own rule registered in Phase 8 before it runs.

## 8. YouTube Faces
YouTube Faces (official database, held since 2026-11-18; archive SHA-256 values in `outputs/video/ytf/archives_sha256.txt`; the usage terms are still to be recorded in `DATASETS.md`).
- **Counts (check 4):** 3,425 videos, 1,595 people, 621,126 frames in `frame_images_DB` (all extracted and counted), 5,000 pairs in 10 folds of 500 with 250 same-person pairs each (`meta_and_splits.mat`): all equal to the official numbers. The box convention (centre x, centre y, width, height) was checked on 10 seeded overlays: the boxes sit on the faces.
- **Frames:** 171,021 frames selected by the pre-registered rule (202 videos have fewer than 50 labelled frames). SCRFD `full640` found the subject (best IoU with the labelled box at least 0.5, score at least 0.5) in 167,622 of them (98.0%); 163 frames had no detection at 0.5 or above, 3,236 had detections that did not overlap the labelled box, none was unreadable. Eight videos have no accepted frame at all, so 14 of the 5,000 pairs cannot be scored; **4,986 pairs are scored** and the failure is logged (every fold keeps about 499 pairs).
- **Primary hypothesis supported.** The single-frame reference (one random frame per video, 20 seeded draws) gives a verification accuracy of 0.9567 (mean of the draws, range 0.9535 to 0.9613; standard error over folds 0.0030). R1 (cosine of the plain-mean templates) gives 0.9657 (standard error 0.0022); the paired difference is **+0.0090 (95% CI +0.0063 to +0.0117)** (Bonferroni +0.0054 to +0.0127), above zero. AUC 0.9785 to 0.9827; TAR at FAR 0.01 0.9251 to 0.9435; TAR at FAR 0.1 0.9537 to 0.9607 (single frame to R1).
- **The gain is real but small: under one accuracy point.** The single frame is already at 95.7% on this protocol (full video frames; labelled face boxes 40 to 163 px at the 5th to 95th percentile, median 83 px), so, as on native ChokePoint, there is little room; the large effect of section 4 needs probes that single frames fail on.
- **Rules.** All rules except R3 with k = 1 have a difference to the single frame above zero; the order is R4 and R4* (0.9669 each), R3 with k = 10 (0.9665), R1 (0.9657), R4 with lambda 10 (0.9653), R3 with k = 5 (0.9649), R2 (0.9643), R4 with lambda 100 (0.9643), R3 with k = 3 and R5 (0.9633), R3 with k = 1 (0.9597). Against R1: R3 with k = 1 is worse (-0.0060, CI -0.0096 to -0.0024; Bonferroni -0.0110 to -0.0009); norm pooling is nominally better (+0.0012 for R4 and R4*) but the CIs include zero (R4 -0.0012 to +0.0040), so no rule is shown to beat the plain mean; R2 (-0.0014) and R5 (-0.0024) are not better. R5 here uses the dot product of the unnormalised mean embeddings as pair score and is not equal to R1.
- **Determinism (check 4):** video 1 (AJ_Cook/0) was rerun end to end (detection, subject matching, alignment, embedding) into a separate folder; every array is identical to the first run (boxes, keypoints, scores, IoU values, AdaFace embeddings, MagFace norms, luminance, Laplacian variance and the counts). One video, one machine and software stack.

## 9. CAFace arm
CAFace (github.com/mk-minchul/CAFace, commit e94ad02; checkpoint `CAFace_AdaFaceWebFace4M.ckpt`) run inference-only from `envs/caface` (torch 2.14.1 instead of the repository's pinned 1.10, section 0.9). Its backbone is AdaFace R100 trained on WebFace4M, a different model from the MS1MV2 AdaFace of all other rows, so **its rows are never pooled or ranked against them**; CAFace is compared only against single-frame and plain-mean rows from its own backbone. Check 5 passed (backbone equals the standalone AdaFace WebFace4M checkpoint, cosine at least 0.9999998 on 100 faces; a length-1 track returns its own feature). Tables in section 13.
- **ChokePoint GT tracks (CAFace backbone).** 12 px SC: single frame 0.9706; plain mean of the backbone features (R1) 1.0000, difference +0.0294 (95% CI +0.0185 to +0.0415); CAFace aggregator 0.9931, difference +0.0225 (+0.0122 to +0.0347); **CAFace minus R1: -0.0069 (CI -0.0160 to 0.0000)**. 12 px MC: single 0.9775, R1 1.0000 (+0.0225, +0.0160 to +0.0297), CAFace 0.9961 (+0.0186), CAFace minus R1 -0.0039 (-0.0093 to 0.0000). At 16 px single 0.9991 and both aggregators 1.0000 (+0.0009, +0.0002 to +0.0017, SC); native 0.9999 to 1.0000.
- **YouTube Faces (CAFace backbone, same 4,986 pairs).** Single frame 0.9613 (standard error 0.0030); R1 0.9695, difference +0.0082 (+0.0060 to +0.0104); CAFace aggregator 0.9693, difference +0.0080 (+0.0056 to +0.0107); **CAFace minus R1: -0.0002 (-0.0017 to +0.0012)**. AUC 0.9811, 0.9846, 0.9844; TAR at FAR 0.01 0.9336, 0.9475, 0.9483; TAR at FAR 0.1 0.9583, 0.9671, 0.9663.
- **Reading.** The learned aggregator gives the same gain as the plain mean on YouTube Faces and a slightly smaller one on 12 px ChokePoint (the upper end of the CI is zero); it is not better than averaging the backbone features. Applying CAFace at degraded resolution and on these datasets without any retraining is a use outside what it was trained for, so this says that the pretrained aggregator does not help here, not that learned aggregation cannot.
- **Backbone comparison (exploratory, Addendum 1; same architecture and loss, different training data: A = MS1MV2, B = WebFace4M).** Paired on the same crops and tracks. At 12 px SC the single frame of B is better by +0.136 (95% CI +0.090 to +0.184; 0.835 against 0.971), at 16 px by +0.0055 (+0.0024 to +0.0091), at native size by +0.0001; MC 12 px +0.126 (+0.083 to +0.171). The aggregation gain (R1 minus single frame) is smaller for B (12 px SC: -0.069, CI -0.093 to -0.045; 16 px -0.0055), because B's single frames leave little to gain: R1 reaches 1.000 for B and 0.933 for A at 12 px. On YouTube Faces B is better by +0.0032 for single frames (+0.0014 to +0.0050) and +0.0038 for R1 (+0.0006 to +0.0070); the aggregation gains do not differ (+0.0006, CI -0.0019 to +0.0032). So the training data of the backbone moves the single-frame result at low resolution far more than aggregation does, and the size of the aggregation effect in section 4 belongs to the MS1MV2 AdaFace; with the WebFace4M backbone the same conclusion (aggregation helps, plain mean is enough) holds with less room.

## 10. Timings
Lab PC (RTX 4000 Ada), shared machine; numbers are wall-clock and include JPEG decoding.
- **YouTube Faces detect and embed (SCRFD `full640` + AdaFace + MagFace norm):** 6,482 s for 171,021 selected frames in 3,425 videos (about 38 ms per selected frame; median 1.76 s per video of up to 50 frames; sum of the per-video times 6,415 s). During its first half-hour CPU-only aggregation reruns ran in parallel.
- **CAFace backbone and fusion (`envs/caface`):** ChokePoint, 72 clips, the 71,344 matched native faces plus two degraded subsamples of every 5th matched face, with all fusions: about 23 minutes (11:52 to 12:15); YouTube Faces, 167,622 accepted frames (decoding, alignment, backbone, fusion): about 24 minutes (12:16 to 12:40), roughly 114 frames per second.
- **ChokePoint aggregation, curves and bootstraps:** CPU only, minutes per part from the caches (not timed separately). The per-track aggregation itself (one NumPy mean per track) costs microseconds; the 1,000-draw bootstraps dominate.
- The Phase 6 per-frame figures (detect about 80 ms at 1280 px, embed about 55 ms per face) are not repeated here.

## 11. Limitations
- ChokePoint: 29 identities, one site, frontal faces; the primary effect is measured on **synthetically degraded** probes (area downsampling then bilinear upsampling, no blur, noise or compression), and the degraded tracks use every 5th matched frame, so a 12 px track has about a fifth of its frames. The result says that combining frames helps when single frames fail for lack of resolution, not how large the gain is on real 12 px surveillance faces.
- GT tracks come from the eye-label boxes (Phase 5) and exclude unannotated faces; predicted tracks are scored only when at least half of their detections are matched (section 7).
- The open-set comparison rests on 6 left-out identities and cannot discriminate.
- Rules R2, R4, R4* use the MagFace norm as quality, not AdaFace's own norm (not cached), and the norm of degraded crops carries little information here; the negative result for quality-weighted rules is a result for this proxy.
- YouTube Faces are celebrity videos, not surveillance footage; the CAFace arm uses another backbone and is never pooled with the MS1MV2 AdaFace rows.
- Study C and appendix T2 (consent) are not part of this phase; per-track results cover the detector's output only through the annotated faces and tracks described above.
- All numbers come from the lab PC (RTX 4000 Ada); determinism was checked for the aggregation outputs (section 3), and for one YouTube Faces video (section 8).

## 12. Decisions and open items for Phase 8
Abser's answers of 2026-11-19 are marked.
1. **Rule for the Phase 8 gate experiment: R1, the plain mean. DECIDED (Abser, 2026-11-19).** No rule beats it on ChokePoint (12 px SC and MC, 16 px) or on YouTube Faces, and CAFace's learned aggregator does not either (sections 5, 8, 9).
2. **Frame quality. DECIDED (Abser, 2026-11-19: follow the recommendation).** The MagFace norm did not help: norm-weighted rules are no better than the mean, the single highest-norm frame is worse than a random one, and the norm of a track does not order its accuracy (sections 5 and 6). Phase 8 pre-registers an error-versus-reject test (error rate against the fraction of lowest-quality frames or tracks rejected) of: track length (at least n frames), embedding self-consistency (cosine to the track's leave-one-out mean), sharpness and illumination (Laplacian variance, luminance), and the model's own norm (the WebFace4M AdaFace norm is already cached; the MS1MV2 AdaFace norm needs about one hour of GPU). Detector score and face size need real low-resolution footage (synthetic degradation keeps the native detection) and a dedicated quality network needs extra weights; both stay out unless time remains.
3. **Tracker default. PROVISIONAL: ByteTrack (Abser approved the stress test and its rule, 2026-11-19).** Result in section 7b: mean IDF1 difference over the five stress conditions +0.0482 (95% CI +0.0351 to +0.0609), ByteTrack better in 58 of 72 clips; the earlier paired per-clip comparison of the Phase 6 results also favours ByteTrack by 0.007 on HOTA and IDF1 (71 of 72 clips; the Phase 6 statement "no winner" came from unpaired pooled CIs). Abser added a further test for Phase 8 on open-set data (both trackers with the face-aware association term) to make the final choice with strong evidence; its rule is registered in Phase 8 before it runs. The Phase 8 plan builds the face-aware association on the BoT-SORT cost, so it needs ByteTrack with the same term for a fair comparison.
4. **Open-set evidence. OPEN, in progress (Abser).** Inconclusive here (6 unmated identities). UCCS was removed from the web; PaSC (data from the University of Notre Dame under a licence, supporting materials at Colorado State) is blocked by the firewall on the lab network, and Abser will pursue it from home. Whether PaSC supports the Phase 8 open-set test, and with which protocol (it is a verification protocol with genuine and impostor pairs), is decided when the licence terms and the data are in hand.
5. **Detector mode. DECIDED (Abser, 2026-11-19: follow the recommendation).** A mode fixed per camera, chosen from a calibration clip: `scripts/video_mode_calibrate.py` merges SCRFD `full640` and `full1280` detections on about 100 frames and recommends `full640` if the 5th percentile of the face height is at least 32 px, otherwise `tiled` (the compromise for scenes whose face sizes are unknown or mixed). The basis: on the annotated ChokePoint faces (all at least 32 px) `full640` and `full1280` have the same recall (1.000 and 0.979 against 1.000 and 0.978 for faces of 32 to 64 px and above 64 px) and `full640` is 3.6 times faster on the T4 (24.8 against 88.3 ms), while `full1280` is better only for faces under 32 px (WIDER recall 0.554 against 0.199 at 10 to 16 px). Four test runs of the helper: ChokePoint P1E_S1_C1 5th percentile 35 px (median 88) gives `full640`; P2L_S4_C3.2 24.9 px (median 33) gives `tiled`; YouTube Faces Aaron_Eckhart/0 95 px gives `full640`; Alexandre_Despatie/5 8.5 px (median 47, background faces included) gives `tiled`. The percentile counts every face the detector sees, background faces included, so the recommendation is conservative and should be checked on a few frames; the earlier statement that `full640` would have sufficed on ChokePoint holds for the annotated faces only. Phase 6 results used `full1280` and are not rerun.
6. **Backbone comparison. DONE (Abser: include).** Exploratory paired comparison in section 9: WebFace4M against MS1MV2 AdaFace R100. The choice of backbone for the final pipeline remains a decision with the supervisor.
7. **YouTube Faces usage terms. DEFERRED (Abser, 2026-11-19).** To be recorded in `DATASETS.md`.
8. **Appendix T2 (custom video). DEFERRED (Abser, 2026-11-19).** Still gated on written supervisor approval and signed consent.

## 13. Result tables

<!-- TABLES:BEGIN -->

#### Primary hypothesis (ChokePoint, pre-registered)

| condition | n_tracks | single_frame_rank1 | R1_aggregated_rank1 | difference_95CI | above_zero |
|---|---|---|---|---|---|
| AdaFace, GT tracks, SC, 12 px probes | 432 | 0.8347 | 0.9329 | +0.0982 (+0.0712 to +0.1251) | yes |

#### Rules at 12 px (SC): R1 against the single frame, plain and Bonferroni CIs

single-frame Rank-1 (track-weighted): 0.8347; tracks: 432

| rule | aggregated_rank1 | difference_vs_single | bonferroni_CI | dir1 | dir2 | minus_R1 | minus_R1_bonferroni_CI |
|---|---|---|---|---|---|---|---|
| R1 | 0.9329 | +0.0982 (+0.0712 to +0.1251) | +0.0614 to +0.1426 | 0.0904 | 0.1059 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R2 | 0.9329 | +0.0982 (+0.0712 to +0.1251) | +0.0614 to +0.1426 | 0.0904 | 0.1059 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R3_k1 | 0.8171 | -0.0176 (-0.0395 to +0.0059) | -0.0470 to +0.0144 | 0.0117 | -0.0469 | -0.1157 (-0.1499 to -0.0798) | -0.1684 to -0.0650 |
| R3_k3 | 0.8935 | +0.0588 (+0.0304 to +0.0878) | +0.0175 to +0.1026 | 0.0626 | 0.0550 | -0.0394 (-0.0698 to -0.0092) | -0.0838 to -0.0019 |
| R3_k5 | 0.9282 | +0.0935 (+0.0673 to +0.1220) | +0.0602 to +0.1370 | 0.0858 | 0.1013 | -0.0046 (-0.0268 to +0.0182) | -0.0346 to +0.0266 |
| R3_k10 | 0.9306 | +0.0958 (+0.0685 to +0.1247) | +0.0577 to +0.1404 | 0.0858 | 0.1059 | -0.0023 (-0.0110 to +0.0048) | -0.0140 to +0.0096 |
| R4 | 0.8935 | +0.0588 (+0.0312 to +0.0880) | +0.0208 to +0.1016 | 0.0673 | 0.0503 | -0.0394 (-0.0649 to -0.0135) | -0.0737 to -0.0027 |
| R4* | 0.9190 | +0.0843 (+0.0560 to +0.1130) | +0.0465 to +0.1277 | 0.0858 | 0.0828 | -0.0139 (-0.0349 to +0.0068) | -0.0440 to +0.0138 |
| R5 | 0.9329 | +0.0982 (+0.0712 to +0.1251) | +0.0614 to +0.1426 | 0.0904 | 0.1059 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R6 | 0.9306 | +0.0958 (+0.0699 to +0.1238) | +0.0612 to +0.1385 | 0.0950 | 0.0966 | -0.0023 (-0.0156 to +0.0120) | -0.0193 to +0.0203 |
| R4_lam10 | 0.9352 | +0.1005 (+0.0705 to +0.1279) | +0.0634 to +0.1428 | 0.0950 | 0.1059 | +0.0023 (-0.0120 to +0.0182) | -0.0182 to +0.0222 |
| R4_lam100 | 0.8681 | +0.0333 (+0.0061 to +0.0615) | -0.0040 to +0.0750 | 0.0395 | 0.0272 | -0.0648 (-0.0957 to -0.0355) | -0.1107 to -0.0276 |

#### Rules at 16 px (SC)

single-frame Rank-1 (track-weighted): 0.9937; tracks: 432

| rule | aggregated_rank1 | difference_vs_single | bonferroni_CI | dir1 | dir2 | minus_R1 | minus_R1_bonferroni_CI |
|---|---|---|---|---|---|---|---|
| R1 | 1.0000 | +0.0063 (+0.0029 to +0.0105) | +0.0021 to +0.0119 | 0.0040 | 0.0087 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R2 | 1.0000 | +0.0063 (+0.0029 to +0.0105) | +0.0021 to +0.0119 | 0.0040 | 0.0087 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R3_k1 | 0.9931 | -0.0006 (-0.0094 to +0.0066) | -0.0131 to +0.0082 | 0.0040 | -0.0052 | -0.0069 (-0.0182 to +0.0000) | -0.0243 to +0.0000 |
| R3_k3 | 1.0000 | +0.0063 (+0.0029 to +0.0105) | +0.0021 to +0.0119 | 0.0040 | 0.0087 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R3_k5 | 1.0000 | +0.0063 (+0.0029 to +0.0105) | +0.0021 to +0.0119 | 0.0040 | 0.0087 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R3_k10 | 1.0000 | +0.0063 (+0.0029 to +0.0105) | +0.0021 to +0.0119 | 0.0040 | 0.0087 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R4 | 1.0000 | +0.0063 (+0.0029 to +0.0105) | +0.0021 to +0.0119 | 0.0040 | 0.0087 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R4* | 1.0000 | +0.0063 (+0.0029 to +0.0105) | +0.0021 to +0.0119 | 0.0040 | 0.0087 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R5 | 1.0000 | +0.0063 (+0.0029 to +0.0105) | +0.0021 to +0.0119 | 0.0040 | 0.0087 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R6 | 1.0000 | +0.0063 (+0.0029 to +0.0105) | +0.0021 to +0.0119 | 0.0040 | 0.0087 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R4_lam10 | 1.0000 | +0.0063 (+0.0029 to +0.0105) | +0.0021 to +0.0119 | 0.0040 | 0.0087 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R4_lam100 | 0.9977 | +0.0040 (+0.0008 to +0.0078) | -0.0003 to +0.0094 | 0.0040 | 0.0041 | -0.0023 (-0.0071 to +0.0000) | -0.0103 to +0.0000 |

#### Rules at 12 px (MC)

single-frame Rank-1 (track-weighted): 0.8518; tracks: 1278

| rule | aggregated_rank1 | difference_vs_single | bonferroni_CI | dir1 | dir2 | minus_R1 | minus_R1_bonferroni_CI |
|---|---|---|---|---|---|---|---|
| R1 | 0.9390 | +0.0872 (+0.0639 to +0.1114) | +0.0575 to +0.1238 | 0.0874 | 0.0869 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R2 | 0.9390 | +0.0872 (+0.0644 to +0.1106) | +0.0569 to +0.1224 | 0.0890 | 0.0853 | -0.0000 (-0.0024 to +0.0022) | -0.0032 to +0.0031 |
| R3_k1 | 0.8451 | -0.0068 (-0.0225 to +0.0090) | -0.0307 to +0.0180 | -0.0036 | -0.0099 | -0.0939 (-0.1234 to -0.0652) | -0.1410 to -0.0553 |
| R3_k3 | 0.9084 | +0.0566 (+0.0399 to +0.0741) | +0.0335 to +0.0827 | 0.0597 | 0.0536 | -0.0306 (-0.0473 to -0.0152) | -0.0554 to -0.0088 |
| R3_k5 | 0.9258 | +0.0740 (+0.0529 to +0.0963) | +0.0478 to +0.1055 | 0.0643 | 0.0837 | -0.0132 (-0.0311 to +0.0018) | -0.0380 to +0.0073 |
| R3_k10 | 0.9359 | +0.0841 (+0.0616 to +0.1068) | +0.0567 to +0.1199 | 0.0828 | 0.0853 | -0.0031 (-0.0080 to +0.0008) | -0.0105 to +0.0016 |
| R4 | 0.8975 | +0.0457 (+0.0300 to +0.0608) | +0.0248 to +0.0672 | 0.0488 | 0.0425 | -0.0415 (-0.0621 to -0.0226) | -0.0707 to -0.0167 |
| R4* | 0.9187 | +0.0668 (+0.0471 to +0.0854) | +0.0417 to +0.0971 | 0.0658 | 0.0679 | -0.0203 (-0.0357 to -0.0076) | -0.0415 to -0.0039 |
| R5 | 0.9390 | +0.0872 (+0.0639 to +0.1114) | +0.0575 to +0.1238 | 0.0874 | 0.0869 | +0.0000 (+0.0000 to +0.0000) | +0.0000 to +0.0000 |
| R6 | 0.9351 | +0.0833 (+0.0617 to +0.1062) | +0.0558 to +0.1186 | 0.0797 | 0.0869 | -0.0039 (-0.0102 to +0.0015) | -0.0137 to +0.0031 |
| R4_lam10 | 0.9367 | +0.0849 (+0.0618 to +0.1118) | +0.0548 to +0.1272 | 0.0813 | 0.0885 | -0.0023 (-0.0132 to +0.0068) | -0.0196 to +0.0122 |
| R4_lam100 | 0.8795 | +0.0277 (+0.0140 to +0.0425) | +0.0083 to +0.0481 | 0.0303 | 0.0250 | -0.0595 (-0.0841 to -0.0370) | -0.0927 to -0.0290 |

#### R1 against the single frame, all conditions (GT tracks, AdaFace, mean of the two directions; difference = aggregated minus single frame, track-weighted)

| cond | probes | n_tracks | single | R1 | difference |
|---|---|---|---|---|---|
| SC | native | 432 | 0.9998 | 1.0000 | +0.0002 (+0.0000 to +0.0004) |
| SC | 16 px | 432 | 0.9937 | 1.0000 | +0.0063 (+0.0029 to +0.0105) |
| SC | 12 px | 432 | 0.8347 | 0.9329 | +0.0982 (+0.0712 to +0.1251) |
| MC | native | 1281 | 0.9996 | 1.0000 | +0.0004 (+0.0001 to +0.0008) |
| MC | 16 px | 1278 | 0.9950 | 1.0000 | +0.0050 (+0.0023 to +0.0079) |
| MC | 12 px | 1278 | 0.8518 | 0.9390 | +0.0872 (+0.0639 to +0.1114) |

#### Open set (TPIR20 at FPIR 0.3, 0.1, 0.01; CI for 0.01)

| cond | probes | rule | rank1 | tpir_fpir0_3 | tpir_fpir0_1 | tpir_fpir0_01 |
|---|---|---|---|---|---|---|
| SC | native | R0 single frame (all frames) | 0.9998 | 0.9998 | 0.9989 | 0.9883 (0.9829-0.9995) |
| SC | native | R1 (track template) | 1.0000 | 1.0000 | 1.0000 | 1.0000 (1.0000-1.0000) |
| SC | 16 px | R0 single frame (all frames) | 0.9936 | 0.9914 | 0.9797 | 0.8990 (0.8449-0.9715) |
| SC | 16 px | R1 (track template) | 1.0000 | 1.0000 | 1.0000 | 1.0000 (1.0000-1.0000) |
| SC | 12 px | R0 single frame (all frames) | 0.8387 | 0.7858 | 0.6329 | 0.3675 (0.1595-0.6986) |
| SC | 12 px | R1 (track template) | 0.9329 | 0.9167 | 0.8452 | 0.6518 (0.5000-0.9559) |
| MC | native | R0 single frame (all frames) | 0.9995 | 0.9992 | 0.9977 | 0.9863 (0.9800-0.9992) |
| MC | native | R1 (track template) | 1.0000 | 1.0000 | 1.0000 | 1.0000 (1.0000-1.0000) |
| MC | 16 px | R0 single frame (all frames) | 0.9946 | 0.9917 | 0.9805 | 0.9187 (0.8418-0.9688) |
| MC | 16 px | R1 (track template) | 1.0000 | 1.0000 | 1.0000 | 0.9940 (0.9888-1.0000) |
| MC | 12 px | R0 single frame (all frames) | 0.8539 | 0.8070 | 0.6514 | 0.3605 (0.2021-0.6641) |
| MC | 12 px | R1 (track template) | 0.9390 | 0.9354 | 0.8283 | 0.5155 (0.3989-0.9321) |

#### B k = 1 gallery (one enrolment frame, 20 draws) with aggregated probes

| cond | probes | n_tracks | single | R1 | difference |
|---|---|---|---|---|---|
| SC | native | 432 | 0.9956 | 0.9993 | +0.0037 (+0.0027 to +0.0047) |
| SC | 16 px | 432 | 0.9535 | 0.9934 | +0.0399 (+0.0318 to +0.0491) |
| SC | 12 px | 432 | 0.6304 | 0.7418 | +0.1114 (+0.0971 to +0.1255) |
| MC | native | 1281 | 0.9945 | 0.9989 | +0.0043 (+0.0030 to +0.0058) |
| MC | 16 px | 1278 | 0.9584 | 0.9950 | +0.0367 (+0.0295 to +0.0437) |
| MC | 12 px | 1278 | 0.6440 | 0.7496 | +0.1056 (+0.0943 to +0.1163) |

#### Predicted tracks (ByteTrack, BoT-SORT), R1 against the single frame

| tracker | cond | probes | n_tracks | single | R1 | difference |
|---|---|---|---|---|---|---|
| bytetrack | SC | native | 404 | 0.9933 | 1.0000 | +0.0067 (+0.0047 to +0.0087) |
| bytetrack | SC | 16 px | 404 | 0.9940 | 1.0000 | +0.0060 (+0.0028 to +0.0099) |
| bytetrack | SC | 12 px | 404 | 0.8399 | 0.9387 | +0.0988 (+0.0704 to +0.1263) |
| bytetrack | MC | native | 1025 | 0.9936 | 1.0000 | +0.0064 (+0.0047 to +0.0082) |
| bytetrack | MC | 16 px | 1025 | 0.9953 | 1.0000 | +0.0047 (+0.0023 to +0.0076) |
| bytetrack | MC | 12 px | 1025 | 0.8563 | 0.9494 | +0.0931 (+0.0681 to +0.1170) |
| botsort | SC | native | 398 | 0.9916 | 1.0000 | +0.0084 (+0.0063 to +0.0105) |
| botsort | SC | 16 px | 398 | 0.9940 | 1.0000 | +0.0060 (+0.0029 to +0.0099) |
| botsort | SC | 12 px | 398 | 0.8385 | 0.9378 | +0.0993 (+0.0705 to +0.1271) |
| botsort | MC | native | 1011 | 0.9921 | 1.0000 | +0.0079 (+0.0060 to +0.0097) |
| botsort | MC | 16 px | 1011 | 0.9954 | 1.0000 | +0.0046 (+0.0023 to +0.0074) |
| botsort | MC | 12 px | 1011 | 0.8557 | 0.9496 | +0.0939 (+0.0686 to +0.1185) |

#### Predicted tracks: purity and excluded tracks (native, per direction)

| tracker | cond | direction | tracks | scored | excluded | mean_purity_scored | share_of_detections_in_excluded |
|---|---|---|---|---|---|---|---|
| bytetrack | SC | G1->G2 | 328 | 188 | 140 | 1.000 | 0.279 |
| bytetrack | SC | G2->G1 | 285 | 216 | 69 | 1.000 | 0.084 |
| bytetrack | MC | G1->G2 | 974 | 467 | 507 | 1.000 | 0.419 |
| bytetrack | MC | G2->G1 | 824 | 558 | 266 | 1.000 | 0.190 |
| botsort | SC | G1->G2 | 342 | 182 | 160 | 1.000 | 0.310 |
| botsort | SC | G2->G1 | 301 | 216 | 85 | 1.000 | 0.087 |
| botsort | MC | G1->G2 | 1020 | 458 | 562 | 1.000 | 0.437 |
| botsort | MC | G2->G1 | 871 | 553 | 318 | 1.000 | 0.200 |

#### Tracker stress test (Addendum 2): mean over the 72 clips

| condition | IDF1_ByteTrack | IDF1_BoT_SORT | IDF1_difference | HOTA_difference | tracks_BT_vs_BoT | ID_switches_BT_vs_BoT |
|---|---|---|---|---|---|---|
| S0 original (reference) | 0.7058 | 0.6991 | +0.0067 | +0.0066 | 1798 / 1891 | 3 / 4 |
| S1 every 3rd frame | 0.7328 | 0.6326 | +0.1002 | +0.0402 | 1775 / 10273 | 133 / 2121 |
| S2 every 5th frame | 0.5583 | 0.4268 | +0.1316 | +0.0094 | 1574 / 13498 | 231 / 5129 |
| S3 every 10th frame | 0.2416 | 0.2470 | -0.0054 | -0.1417 | 750 / 10478 | 88 / 4702 |
| S4 10% detections dropped | 0.6820 | 0.6750 | +0.0070 | +0.0060 | 1835 / 2118 | 11 / 21 |
| S5 20% detections dropped | 0.6516 | 0.6441 | +0.0075 | +0.0053 | 1891 / 2412 | 21 / 68 |

Primary statistic (mean over clips of the per-clip IDF1 difference, ByteTrack minus BoT-SORT, averaged over S1 to S5): +0.0482 (95% CI +0.0351 to +0.0609); ByteTrack better in 58 of 72 clips, BoT-SORT in 14; CI excludes zero: True; provisional default: bytetrack.

Exploratory (not part of the decision): BoT-SORT with its association gate relaxed from IoU 0.5 to 0.2 (`proximity_thresh` 0.8):

| condition | IDF1_ByteTrack | IDF1_BoT_SORT_default | IDF1_BoT_SORT_relaxed | tracks_BT_default_relaxed | ID_switches_BT_default_relaxed |
|---|---|---|---|---|---|
| S0 original (reference) | 0.7058 | 0.6991 | 0.6991 | 1798 / 1891 / 1891 | 3 / 4 / 4 |
| S1 every 3rd frame | 0.7328 | 0.6326 | 0.6326 | 1775 / 10273 / 10273 | 133 / 2121 / 2121 |
| S2 every 5th frame | 0.5583 | 0.4268 | 0.4268 | 1574 / 13498 / 13498 | 231 / 5129 / 5129 |
| S3 every 10th frame | 0.2416 | 0.2470 | 0.2470 | 750 / 10478 / 10478 | 88 / 4702 / 4702 |
| S4 10% detections dropped | 0.6820 | 0.6750 | 0.6750 | 1835 / 2118 / 2118 | 11 / 21 / 21 |
| S5 20% detections dropped | 0.6516 | 0.6441 | 0.6441 | 1891 / 2412 / 2412 | 21 / 68 / 68 |

#### Curves: accuracy against the number of frames used (R1, SC, 20 seeded draws)

| frames per track | native | 16 px | 12 px |
|---|---|---|---|
| 1 | 0.9999 | 0.9927 | 0.8316 |
| 2 | 1.0 | 0.9994 | 0.887 |
| 4 | 1.0 | 1.0 | 0.9156 |
| 8 | 1.0 | 1.0 | 0.9295 |
| 16 | 1.0 | 1.0 | 0.9338 |
| all | 1.0 | 1.0 | 0.9329 |

By the track's mean MagFace norm (SC, 12 px, quartiles over tracks):

| quartile | tracks | mean_MagFace_norm | single_frame | R1 |
|---|---|---|---|---|
| Q1 (lowest norm) | 108 | 20.45 | 0.8356 | 0.9537 |
| Q2 | 108 | 21.68 | 0.8517 | 0.9259 |
| Q3 | 108 | 22.54 | 0.8616 | 0.9444 |
| Q4 (highest norm) | 108 | 24.13 | 0.7900 | 0.9074 |

#### Acceptance checks

- Checks 1 and 2 (synthetic tracks, all rules): all pass = True.
- Check 3 (single-frame reference equals Phase 6 section 0 and E to 5 decimals, 12 rows): True.
- check4_determinism_aggregation: {"first_run_dir": "outputs/video/agg7/run1", "second_run_dir": "outputs/video/agg7", "files": {"chokepoint": {"rows": [84, 84], "columns_compared": 30, "identical": true}, "predicted": {"rows": [120, 120], "columns_compared": 19, "identical": true}, "curves_frames": {"rows": [36, 36], "columns_compa
- check4_ytf_counts: {"official_counts": {"videos": 3425, "people": 1595, "pairs": 5000, "folds": 10}, "found": {"n_videos_in_meta": 3425, "n_people_in_meta": 1595, "n_videos_with_labels": 3425, "n_videos_without_labels": 0, "n_people_with_labels": 1595, "n_pairs": 5000, "n_folds": 10, "same_per_fold": [250, 250, 250, 2
- check4_ytf_video_rerun: {"video_idx": 1, "arrays": {"box": "identical", "emb_adaface": "identical", "frame_no": "identical", "iou": "identical", "kps": "identical", "lap_var": "identical", "lum": "identical", "mag_norm": "identical", "n_no_detection": "identical", "n_no_match": "identical", "n_selected": "identical", "n_un
- check5_caface_backbone_vs_standalone_adaface_webface4m: {"n_faces": 100, "cosine_min": 0.9999998807907104, "cosine_median": 1.0, "cosine_max": 1.000000238418579, "n_above_0_9999": 100, "note": "CAFace backbone (inside CAFace_AdaFaceWebFace4M.ckpt) against the standalone adaface_ir101_webface4m.ckpt through src/recognizer.py (flip off)"}

#### YouTube Faces

YouTube Faces verification (mean over folds, SE)

| method | accuracy | accuracy_se | auc | tar_far0_01 | tar_far0_1 | n_pairs | accuracy_min_draw | accuracy_max_draw |
|---|---|---|---|---|---|---|---|---|
| single frame (mean of 20 draws) | 0.9567 | 0.003 | 0.9785 | 0.9251 | 0.9537 | 4986 | 0.9535 | 0.9613 |
| R1 | 0.9657 | 0.0022 | 0.9827 | 0.9435 | 0.9607 | 4986 | nan | nan |
| R2 | 0.9643 | 0.0023 | 0.9827 | 0.9431 | 0.9615 | 4986 | nan | nan |
| R3_k1 | 0.9597 | 0.0029 | 0.9774 | 0.9303 | 0.9531 | 4986 | nan | nan |
| R3_k3 | 0.9633 | 0.0021 | 0.9807 | 0.9383 | 0.9571 | 4986 | nan | nan |
| R3_k5 | 0.9649 | 0.0028 | 0.9811 | 0.9391 | 0.9615 | 4986 | nan | nan |
| R3_k10 | 0.9665 | 0.0023 | 0.9815 | 0.9407 | 0.9611 | 4986 | nan | nan |
| R4 | 0.9669 | 0.0026 | 0.9816 | 0.9415 | 0.9599 | 4986 | nan | nan |
| R4* | 0.9669 | 0.0023 | 0.9822 | 0.9431 | 0.9619 | 4986 | nan | nan |
| R5 | 0.9633 | 0.0026 | 0.9825 | 0.9399 | 0.9623 | 4986 | nan | nan |
| R4_lam10 | 0.9653 | 0.0024 | 0.9826 | 0.9435 | 0.9619 | 4986 | nan | nan |
| R4_lam100 | 0.9643 | 0.0025 | 0.9802 | 0.9359 | 0.9583 | 4986 | nan | nan |

YouTube Faces: paired difference against the single-frame reference

| method | delta_accuracy | ci_lo | ci_hi | bonferroni_lo | bonferroni_hi | above_zero | minus_R1 | minus_R1_ci_lo | minus_R1_ci_hi | minus_R1_bonf_lo | minus_R1_bonf_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 0.009 | 0.0063 | 0.0117 | 0.0054 | 0.0127 | True | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| R2 | 0.0076 | 0.0052 | 0.0099 | 0.0043 | 0.0111 | True | -0.0014 | -0.0027 | -0.0004 | -0.0035 | 0.0 |
| R3_k1 | 0.003 | -0.0002 | 0.006 | -0.0011 | 0.0068 | False | -0.006 | -0.0096 | -0.0024 | -0.011 | -0.0009 |
| R3_k3 | 0.0066 | 0.0036 | 0.0092 | 0.0022 | 0.0103 | True | -0.0024 | -0.0056 | 0.0006 | -0.0065 | 0.0021 |
| R3_k5 | 0.0082 | 0.0061 | 0.0105 | 0.0053 | 0.0114 | True | -0.0008 | -0.0036 | 0.002 | -0.0047 | 0.0032 |
| R3_k10 | 0.0098 | 0.0074 | 0.0122 | 0.0068 | 0.0132 | True | 0.0008 | -0.0014 | 0.0032 | -0.002 | 0.0042 |
| R4 | 0.0102 | 0.0078 | 0.0126 | 0.0069 | 0.0135 | True | 0.0012 | -0.0012 | 0.004 | -0.0019 | 0.0049 |
| R4* | 0.0102 | 0.008 | 0.0126 | 0.007 | 0.0136 | True | 0.0012 | -0.0008 | 0.0033 | -0.0014 | 0.0042 |
| R5 | 0.0066 | 0.0042 | 0.0092 | 0.0032 | 0.01 | True | -0.0024 | -0.005 | 0.0 | -0.0058 | 0.0009 |
| R4_lam10 | 0.0086 | 0.0063 | 0.0109 | 0.0056 | 0.012 | True | -0.0004 | -0.0023 | 0.0013 | -0.0028 | 0.0024 |
| R4_lam100 | 0.0076 | 0.0051 | 0.0102 | 0.0044 | 0.0112 | True | -0.0014 | -0.0043 | 0.0018 | -0.0057 | 0.003 |

#### CAFace arm on ChokePoint GT tracks (CAFace backbone: AdaFace R100 WebFace4M; never pooled with the MS1MV2 rows)

| cond | probes | n_tracks | method | single_frame | aggregated | difference_vs_single | minus_R1 |
|---|---|---|---|---|---|---|---|
| SC | native | 432 | R1 plain mean (CAFace backbone) | 0.9999 | 1.0000 | +0.0001 (+0.0000 to +0.0001) | +0.0000 (+0.0000 to +0.0000) |
| SC | native | 432 | CAFace aggregator | 0.9999 | 1.0000 | +0.0001 (+0.0000 to +0.0001) | +0.0000 (+0.0000 to +0.0000) |
| SC | 16 px | 432 | R1 plain mean (CAFace backbone) | 0.9991 | 1.0000 | +0.0009 (+0.0002 to +0.0017) | +0.0000 (+0.0000 to +0.0000) |
| SC | 16 px | 432 | CAFace aggregator | 0.9991 | 1.0000 | +0.0009 (+0.0002 to +0.0017) | +0.0000 (+0.0000 to +0.0000) |
| SC | 12 px | 432 | R1 plain mean (CAFace backbone) | 0.9706 | 1.0000 | +0.0294 (+0.0185 to +0.0415) | +0.0000 (+0.0000 to +0.0000) |
| SC | 12 px | 432 | CAFace aggregator | 0.9706 | 0.9931 | +0.0225 (+0.0122 to +0.0347) | -0.0069 (-0.0160 to +0.0000) |
| MC | native | 1281 | R1 plain mean (CAFace backbone) | 0.9998 | 1.0000 | +0.0002 (+0.0000 to +0.0005) | +0.0000 (+0.0000 to +0.0000) |
| MC | native | 1281 | CAFace aggregator | 0.9998 | 1.0000 | +0.0002 (+0.0000 to +0.0005) | +0.0000 (+0.0000 to +0.0000) |
| MC | 16 px | 1278 | R1 plain mean (CAFace backbone) | 0.9992 | 1.0000 | +0.0008 (+0.0002 to +0.0015) | +0.0000 (+0.0000 to +0.0000) |
| MC | 16 px | 1278 | CAFace aggregator | 0.9992 | 1.0000 | +0.0008 (+0.0002 to +0.0015) | +0.0000 (+0.0000 to +0.0000) |
| MC | 12 px | 1278 | R1 plain mean (CAFace backbone) | 0.9775 | 1.0000 | +0.0225 (+0.0160 to +0.0297) | +0.0000 (+0.0000 to +0.0000) |
| MC | 12 px | 1278 | CAFace aggregator | 0.9775 | 0.9961 | +0.0186 (+0.0126 to +0.0256) | -0.0039 (-0.0093 to +0.0000) |

#### CAFace arm on YouTube Faces

CAFace backbone on YouTube Faces (mean over folds)

| method | accuracy | accuracy_se | auc | tar_far0_01 | tar_far0_1 | n_pairs | accuracy_min_draw | accuracy_max_draw |
|---|---|---|---|---|---|---|---|---|
| single frame (mean of 20 draws) | 0.9613 | 0.003 | 0.9811 | 0.9336 | 0.9583 | 4986 | 0.9575 | 0.9645 |
| R1 (plain mean of CAFace-backbone features) | 0.9695 | 0.0031 | 0.9846 | 0.9475 | 0.9672 | 4986 | nan | nan |
| CAFace aggregator | 0.9693 | 0.0034 | 0.9844 | 0.9483 | 0.9664 | 4986 | nan | nan |

CAFace backbone on YouTube Faces: paired differences

| method | delta_accuracy | ci_lo | ci_hi | bonferroni_lo | bonferroni_hi | above_zero | minus_R1 | minus_R1_ci_lo | minus_R1_ci_hi | minus_R1_bonf_lo | minus_R1_bonf_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 (plain mean of CAFace-backbone features) | 0.0082 | 0.006 | 0.0104 | 0.0051 | 0.0113 | True | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| CAFace aggregator | 0.008 | 0.0056 | 0.0107 | 0.0048 | 0.012 | True | -0.0002 | -0.0017 | 0.0012 | -0.0023 | 0.0018 |

#### Backbone comparison (exploratory, Addendum 1): A = AdaFace R100 MS1MV2, B = AdaFace R100 WebFace4M; paired differences B minus A

| data | cond | probes | units | single_A | single_B | R1_A | R1_B | B_minus_A_single | B_minus_A_R1 | B_minus_A_aggregation_gain |
|---|---|---|---|---|---|---|---|---|---|---|
| ChokePoint GT tracks | SC | native | 432 | 0.9998 | 0.9999 | 1.0000 | 1.0000 | +0.0001 (+0.0000 to +0.0003) | +0.0000 (+0.0000 to +0.0000) | -0.0001 (-0.0003 to +0.0000) |
| ChokePoint GT tracks | SC | 16 px | 432 | 0.9937 | 0.9991 | 1.0000 | 1.0000 | +0.0055 (+0.0024 to +0.0091) | +0.0000 (+0.0000 to +0.0000) | -0.0055 (-0.0091 to -0.0024) |
| ChokePoint GT tracks | SC | 12 px | 432 | 0.8347 | 0.9706 | 0.9329 | 1.0000 | +0.1359 (+0.0904 to +0.1843) | +0.0671 (+0.0297 to +0.1094) | -0.0688 (-0.0931 to -0.0445) |
| ChokePoint GT tracks | MC | native | 1266 | 0.9996 | 0.9998 | 1.0000 | 1.0000 | +0.0002 (+0.0000 to +0.0004) | +0.0000 (+0.0000 to +0.0000) | -0.0002 (-0.0004 to -0.0000) |
| ChokePoint GT tracks | MC | 16 px | 1260 | 0.9950 | 0.9992 | 1.0000 | 1.0000 | +0.0042 (+0.0020 to +0.0067) | +0.0000 (+0.0000 to +0.0000) | -0.0042 (-0.0067 to -0.0020) |
| ChokePoint GT tracks | MC | 12 px | 1260 | 0.8518 | 0.9775 | 0.9390 | 1.0000 | +0.1257 (+0.0828 to +0.1713) | +0.0610 (+0.0283 to +0.0977) | -0.0647 (-0.0848 to -0.0445) |
| YouTube Faces (pairs) |  | native | 4986 | 0.9571 | 0.9604 | 0.9657 | 0.9695 | +0.0032 (+0.0014 to +0.0050) | +0.0038 (+0.0006 to +0.0069) | +0.0006 (-0.0019 to +0.0032) |

<!-- TABLES:END -->
