# REPORT — Phase 7: per-track aggregation (ChokePoint + YouTube Faces, plus a CAFace arm)

**Status:** in progress. This file starts with the pre-registered protocol (section 0), committed on branch `phase_7` before any Phase 7 number was computed (see the git history of this file). Plan: `research/Sprint_Notes/Phase7_aggregation_plan.md` (approved 2026-11-17). Day-1 intake notes (YouTube Faces layout, norm pooling, CAFace interface) are in section 0.9.

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
- **Confirmations (Abser, 2026-11-19), given before any Phase 7 run:** (1) the YouTube Faces detector mode is `full640` (ChokePoint keeps `full1280`); (2) "evenly spaced over the frames" means up to 50 evenly spaced frames among the video's labelled frames (section 0.5); (3) R4 (lambda 50), R4* (lambda 5), the lambda sensitivity set and the Bonferroni correction over nine comparisons as written in section 0.2; (4) the primary hypotheses of sections 0.3 and 0.5 as written. The YouTube Faces usage terms are not yet recorded in `DATASETS.md` (internet bandwidth problem); they will be added later.
- **Extraction note.** The first extraction of `frame_images_DB.tar.gz` was cut off when a session ended (682 of 1,595 person folders). The last folder being written (`Gerard_Kleisterlee`) was deleted and the archive was extracted again with `--skip-old-files`; the final count of frames (621,126 expected) is checked by `video_ytf_prepare.py` before any YouTube Faces run.
