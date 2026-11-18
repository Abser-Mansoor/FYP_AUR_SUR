# Phase 7 plan: per-track aggregation, the headline result (ChokePoint + YouTube Faces, plus a CAFace arm)

**Status:** approved 2026-11-17. Executor: Claude Code on the lab PC with the team. Section 0 is pre-registered in `REPORT_phase7.md` and committed before any run.

## Context
- Phase 6 is merged into `main` (`3d5d6f6`) and approved at the supervisor checkpoint.
- Study C is done: 0.849 of the unmatched detections are live faces without an annotation, and 0.128 are reflections.
- Decisions so far:
  - keep both trackers;
  - YouTube Faces (YTF) is held, and its tests belong in Phase 7;
  - CAFace is included as an extra arm;
  - Claude Code on the lab PC executes, with the team.
- Question (brief experiments 1–2): does combining the embeddings of a track beat a single frame, and which rule is best? This is the thesis claim that tracking uses real information, where SR invented it.
- Headroom (from `REPORT_phase6.md` §6): single frames are at ceiling on native ChokePoint. Room exists only for probes at **16 px and 12 px** (protocol E: Rank-1 0.994 and 0.839, SC) and for a one-frame gallery (B, k = 1). The ChokePoint part must therefore be judged there.
- YTF adds a second dataset with the standard video-to-video verification protocol.
- Standing rules apply:
  - inference only;
  - every number from a script;
  - pre-register before computing;
  - one device per number: the lab PC (RTX 4000 Ada) for all of Phase 7;
  - failures logged.
- After approval, save this plan as `research/Sprint_Notes/Phase7_aggregation_plan.md`.

## 0. Pre-registration (committed before any Phase 7 number)
`REPORT_phase7.md` §0 holds the full protocol below. It is committed on branch `phase_7` before any run.

**Unit of analysis.**
- **GT track** = all matched detections of one (clip, person id).
- **Predicted track** = one (clip, track id) of ByteTrack or BoT-SORT. It uses all its detections at score ≥ 0.5 (the embedding floor), *including unmatched ones*, because a deployed system cannot tell which ones are annotated.
  - Its label is the majority person id of its matched detections.
  - Tracks with < 50% matched detections are counted and reported but not scored; study C shows that most of them are unannotated faces.
  - Track purity (the share of its majority id) is reported.
- **Paired comparison per track:**
  - *single-frame accuracy* = the mean, over the track's frames, of frame-level correctness (the expected accuracy of one random frame);
  - *aggregated accuracy* = the correctness of the track's one aggregated probe.
  - Paired delta per track; cluster bootstrap over identities (1,000 draws, seed 42).
  - Frame-level rows (Phase 6 E numbers) are shown as reference.

**Rules.** Per track, input: L2-normalised AdaFace embeddings `emb[N,512]` and quality `q[N]` (MagFace raw norm). Output: one normalised template.
- R0 single frame (reference)
- R1 plain mean — **primary rule**
- R2 MagFace-norm-weighted mean
- R3 top-k by quality, k ∈ {1, 3, 5, 10}, all reported, none selected
- R4 norm pooling (arXiv 2409.09832; equation to be checked against the paper on day 1). The AdaFace raw norm is not cached, so it uses the MagFace norm and is labelled as such.
- R5 score-level mean (the mean cosine of the frames to each gallery template)
- R6 majority vote of per-frame Rank-1 (ties go to the higher mean score; closed set only)

No hyperparameter is tuned: with 29 identities a tuning split is not credible, so every k is shown. Multiplicity: Bonferroni CIs over the R2–R6 comparisons with R1 are reported alongside.

**Primary hypothesis (ChokePoint).**
- Condition: AdaFace, GT tracks, SC, probes at 12 px.
- Test: R1 aggregated Rank-1 minus single-frame Rank-1, paired per track, mean of the two directions.
- "Aggregation helps" if the 95% CI is above zero.
- Secondary conditions: 16 px; native; MC; predicted tracks (both trackers); B k = 1 with aggregated probes; open-set TPIR20 at FPIR 0.3, 0.1 and 0.01.
  - Open-set unmated tracks come from the 6 left-out identities (rng 42), so their CIs will be wide; reported as they come.

**Curves (descriptive).**
- Accuracy against the number of frames used, n ∈ {1, 2, 4, 8, 16, all}: random frame subsets per track, 20 seeded draws.
- Accuracy against the track's mean MagFace norm (quartiles).

**YTF protocol** (official `splits.txt`: 5,000 pairs, 10 folds, unsupervised setting: no training).
- **Frames per video:** up to 50, evenly spaced over the frames where the subject is found. The subject is the SCRFD detection with IoU ≥ 0.5 to the YTF `labeled_faces` box. Frames without a match are skipped and counted.
- **Detector mode:** `full640`. Phase 6 Appendix T1 found that `full1280` misses large faces (5.0% vs 98.1%). This resolves Phase 6 decision 3 as "mode follows face size; YTF uses `full640`". **Abser confirms this in the pre-registration commit.**
- **Pair score:** cosine of the two videos' templates (R1–R5).
- **Single-frame reference:** one random frame per video, 20 seeded draws; mean and range reported.
- **Metrics:**
  - verification accuracy with the threshold chosen on the 9 other folds (mean ± SE over folds; the standard protocol);
  - AUC;
  - TAR@FAR 0.01 and 0.1 pooled;
  - paired delta against single-frame per pair, with a bootstrap over pairs grouped by identity.
- **Primary YTF hypothesis:** R1 minus single-frame accuracy, CI above zero.

**CAFace arm** (an extra arm, never pooled with the MS1MV2 AdaFace rows).
- Its own backbone: AdaFace R100 WebFace4M (downloaded in Phase 5).
- Its own single-frame and plain-mean rows from that backbone. CAFace is compared only against those rows.
- Run on ChokePoint GT tracks (native, 16 px, 12 px) and on YTF.

**Not in Phase 7 (recorded):**
- Study C as unmated probes: not run. It needs a separately pre-registered identity-level labelling, and no decision has been taken.
- Appendix T2 (custom video): still gated on consent.
- MagFace and CurricularFace rows: AdaFace only, to keep the run small. They can be added from cache for native ChokePoint if time remains.

## 1. Work items

**1a. YTF intake (day 1, lab PC).**
- Record source, licence, SHA-256 and verified counts in `DATASETS.md`: 3,425 videos, 1,595 people, 5,000 pairs. A mismatch is reported as found.
- Confirm the layout: `frame_images_DB/` full frames plus `*.labeled_faces.txt` boxes, `splits.txt`.
- New `scripts/video_ytf_prepare.py`:
  - parse the boxes and splits;
  - choose the frames by the pre-registered rule;
  - write `outputs/video/ytf/frames.csv`.

**1b. Code** (reusing existing code; tables injected between markers as in `scripts/phase6_report_tables.py`):
- `src/video/aggregate.py`: R1–R5 with one signature `(emb, q, **kw) -> emb[512]`; R6 vote takes per-frame scores. Pure NumPy.
- `scripts/video_aggregation.py --part chokepoint|curves|ytf`. It reuses:
  - `scripts/video_single_frame.py` (`get_matches`, groups, `adaface_emb`);
  - `scripts/video_phase7_bench.py` (`direction`, `tpir_at`, `boot`, and the E cache `outputs/video/bench7/emb/<clip>.npz`, which holds `emb_<s>` and `mag_<s>` for the every-5th matched-detection subsample);
  - `src/eval_protocols.py`, `src/metrics.py` (`bootstrap`, `paired_delta`).
- **Degraded tracks** use the E subsample, so the 12/16 px tracks are about every 5th frame. The single-frame row then equals E exactly, and no new GPU work is needed. For predicted tracks at degraded sizes, only the subsampled matched detections exist; this is stated as a limitation.
- **YTF embedding:** `scripts/run_video.py` with a new `ytf` dataset entry. `src/video/io.py` already reads image sequences, and `src/video/pipeline.py` writes `embed.npz` (AdaFace and MagFace norm). Detector mode is `full640` for YTF only, via the config.
- **CAFace:**
  - a new venv `envs/caface` at the pinned commit;
  - `scripts/video_caface.py` embeds the needed crops with the WebFace4M backbone and runs the CAFace aggregator;
  - outputs to `outputs/video/caface/`.
  - Day 1: verify the interface (does it need intermediate feature maps? is the center file training-only?). If CAFace cannot run inference-only from the official weights, **STOP** and drop the arm, with the reason recorded.
- `configs/video.yaml`: new `aggregation` and `ytf` blocks (rules, k list, n-frames list, frame cap 50, IoU 0.5, detector mode).
- `.gitignore`: track `outputs/video/agg7/*.csv`, `outputs/video/ytf/*.csv|json`, `outputs/video/caface/*.csv|json`. Never track embeddings or frames.

**1c. Runs (lab PC; one GPU job at a time; resumable; launch detached).**
1. ChokePoint aggregation, curves: CPU from cache, minutes.
2. YTF detect + embed: about 3,425 × ≤50 ≈ 171k faces, an estimated 4–5 h at the Phase 6 per-face cost. Run overnight.
3. CAFace backbone + aggregation: ChokePoint about 71k native plus E subsample degradations, then YTF. An estimated 2–3 h.
4. Determinism: rerun the aggregation scripts; outputs must be byte-identical. Then rerun one YTF video end to end.

## 2. Acceptance checks
1. A track of length 1 gives the single-frame result for every rule. Top-k with k = N equals the plain mean.
2. Shuffling frames inside a track does not change R1–R5 (top-k ties broken by frame index).
3. The ChokePoint single-frame reference reproduces Phase 6 E (12/16 px) and §0 (native) to the last digit.
4. YTF counts match the official ones (or the mismatch is reported). Every pair has both templates, or its failure is logged.
5. CAFace: the official example, if one ships, reproduces before our data is used.

## 3. Deliverable
`REPORT_phase7.md`:
- §0 pre-registration
- summary
- what was built
- checks
- ChokePoint primary and secondary results
- rules comparison
- curves
- predicted vs GT tracks (purity, excluded tracks)
- YTF results
- CAFace arm
- timings
- limitations: 29 identities; degraded = synthetic resolution loss; degraded tracks subsampled; YTF celebrities and not surveillance; a different backbone for CAFace
- decisions for Phase 8 (best rule for the gate experiment; tracker; study C labelling)
- STOP for the supervisor.

Update `CONTEXT.md` §3, §4 and §11, and mirror it to Drive.

## 4. Order of work (lab days Tue–Thu; Mon/Fri if available)

| When | Work |
|---|---|
| Day 1 | YTF intake + counts; CAFace interface check; norm-pooling equation check; write and commit §0 (Abser confirms the YTF detector mode and the primary hypotheses) |
| Day 2 | `aggregate.py` + checks 1–3; ChokePoint runs; launch YTF embedding overnight |
| Day 3 | YTF scoring; CAFace runs; determinism |
| Day 4–5 | tables, `REPORT_phase7.md`, `CONTEXT.md`, PR → STOP |

## Verification
- Acceptance checks 1–5 are written to `outputs/video/agg7/checks.json`.
- Aggregation and YTF scoring reruns are byte-identical.
- `git log` shows the §0 commit before any `outputs/video/agg7`, `ytf` or `caface` result file.
- The report's tables are injected by the table script, never hand-typed.

## Risks
- **CAFace not inference-runnable:** drop the arm and record why.
- **YTF layout differs** (e.g. no full frames): fall back to the aligned images without detection, and record the deviation before scoring.
- **Ceiling on YTF too:** report it as found; the 12/16 px ChokePoint result stays primary.
- **PC reboot:** all runs are resumable per clip or video.
