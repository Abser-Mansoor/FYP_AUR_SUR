# Phase 6 plan: core video pipeline and tracker validation

**Window:** W3–W4, 10–22 Nov 2026. Supervisor checkpoint at the end of W4. The Sprint 2 plan had W2–W3; Phase 5's re-run cost about one week.
**Who:**
- Claude on the lab PC: code, runs, report.
- Abser: owns the pipeline and the custom-video consent work.
- Umar: owns the embedding and index modules, and the YouTube Faces test.
**Parent:** `research/Sprint_Notes/Sprint_2_implementation_plan.md`, Phase 6.
**After approval:** save this plan as `research/Sprint_Notes/Phase6_core_pipeline_plan.md`.

## Context
- **Where we are.** Phases 4 and 5 are approved and merged into `main`, together with the SR-asymmetry extra study.
  - Restoration is closed.
  - The video pipeline is the main deliverable.
- **Phase 5 delivered:**
  - the detector choice: SCRFD `det_10g`, `full1280`, `preprocess="official"`, NMS 0.45 (`configs/video.yaml`);
  - ChokePoint eye ground truth and the fitted eye-to-box rule;
  - vendored ByteTrack and BoT-SORT;
  - `envs/video`.
- **Phase 6 builds the end-to-end path** on ChokePoint:
  - frames → detections → aligned faces → AdaFace embedding + MagFace norm → tracks → per-frame identity;
  - validation of both trackers against ground-truth face tracks.
- **Single-frame baseline.** Phase 6 also produces the single-frame identification baseline that Phase 7's aggregation must beat.
- **Testing appendix (Appendix T).**
  - The Phase 5 detector is tested on YouTube Faces (access obtained) and on a custom video of consenting volunteers.
  - This measures recall on footage that is not WIDER FACE, which Phase 5 listed as a limitation.
- **VLMs** are discussed as FYP II future work. They are not part of Phases 6–10.

**Standing rules (unchanged):**
- inference only;
- seed 42;
- every number comes from a script output;
- failures are logged as results;
- datasets are read-only;
- config-driven;
- one GPU job at a time on the lab PC, resumable per sequence;
- Windows-native;
- no ChokePoint, YTF or custom-video frames in git. The repo is public.

**Device rule:** every Phase 6 number comes from the lab PC (RTX 4000 Ada). No Colab is needed. The Phase 5 T4 runtimes are only cited.

## 1. Code: new modules in `src/video/`, reusing existing code
| Module | Content | Reuses |
|---|---|---|
| `io.py` | Frame source: an image sequence (ChokePoint JPEGs, YTF frame folders) or `cv2.VideoCapture`/PyAV (custom MP4). Yields `(frame_idx, timestamp, bgr)` | `src/sr/common.py:read_bgr` |
| `detect.py` | Unchanged from Phase 5 (`SCRFDDetector(preprocess="official")`, `detect_mode`) | — |
| `embed.py` | `norm_crop(bgr, kps)`, then AdaFace (and the other non-leaky recognizers, optional); MagFace raw norm; per-face luminance and `lap_var` | `src/align.py:norm_crop`, `src/recognizer.py:Recognizer`, `src/stratify/measures.py:crop_measures` |
| `src/recognizer.py` | Add `return_raw=False` to `get_embedding`, returning the unflipped raw norm next to the normalised embedding. The default behaviour stays byte-identical | the pattern in `scripts/extra_sr_asymmetry_h7.py:embed` and `src/stratify/measure_split.py` |
| `track.py` | One adapter over ByteTrack and BoT-SORT: `update(dets[N,5], frame) -> [(track_id, box)]`; 5-column input (the Phase 5 BoT-SORT fix) | `src/video/vendor/{bytetrack,botsort}`, `scripts/video_tracker_smoke.py` |
| `index.py` | Gallery enrolment and search: FAISS `IndexFlatIP` (exact), plus FAISS `IndexHNSWFlat` as the HNSW option (`faiss-cpu` is installed; no `hnswlib` needed) | — |
| `pipeline.py` | Orchestration per sequence. It writes `outputs/video/pipeline/<dataset>/<seq>/dets.parquet` (`frame, det_id, x1..y2, score, kps, track_id_bytetrack, track_id_botsort, mag_norm, lum, lap_var`) and `emb_<rec>.npy`. Atomic temp-then-rename writes; resumable per sequence | `src/sr/common.py:write_png` pattern |
| `trackeval/` (vendored) | The HOTA, CLEAR (MOTA) and Identity (IDF1) metric files from JonathonLuiten/TrackEval (MIT), with numpy-2 patches as for the trackers, recorded in `VENDOR.md`. The PyPI package is not installed and is old | `src/video/vendor/VENDOR.md` pattern |

**Scripts and config:**
- `scripts/run_video.py --stage detect|track|embed|all --dataset chokepoint|ytf|custom`.
- `scripts/video_tracker_eval.py`: GT tracks and TrackEval tables.
- `scripts/video_single_frame.py`: the baseline.
- `scripts/video_appendix_boxes.py`: Appendix T.
- New `pipeline:`, `tracking:` and `protocol:` blocks in `configs/video.yaml`.

## 2. ChokePoint runs
- **Detection.** SCRFD `full1280`, official preprocessing, on every frame of the 54 clips that have ground truth (176,916 listed frames). Detection runs at the 0.02 floor; the tracker gets detections above 0.5.
  - The crowded P2E_S5 and P2L_S5 clips are also run. They have no ground truth, so they get qualitative montages only.
  - First measure the frames per second on 1 clip and print the projected total before the full run.
- **Face-track ground truth.** Per clip and identity, the eye labels go through `outputs/video/gt/eye_box_rule.json` and become boxes. Contiguous frames form one track. The ground truth is written in MOTChallenge format.
  - The known chin cut-off on turned heads (Phase 5 §4) is a stated limitation.
- **Trackers.** ByteTrack and BoT-SORT, with re-ID off and default parameters from the vendored code, at 30 fps. Both run on the same predicted detections.
- **Metrics.** IDF1, HOTA, MOTA and ID switches per clip and pooled, from the vendored TrackEval.
- **Single-frame identification baseline.** The protocol is written into the report **before** it is run, the same way as the Part E pre-registration:
  - **Enrolment:** the G1 sequences of the official ChokePoint protocol (`DATASETS.md`). The gallery is the mean AdaFace embedding of each identity's ground-truth frames in G1. Each enrolment frame matches the ground-truth track whose box overlaps it with IoU ≥ 0.5.
  - **Probes:** every detected face in the G2 sequences that matches a ground-truth box (IoU ≥ 0.5). Then the roles swap (G2 enrols, G1 probes) and the results are averaged, as the official protocol alternates its groups.
  - **Metrics:** closed-set Rank-1 per frame, and open-set TPIR@FPIR=0.3 with 20% of the identities (seeded) left out of the gallery. Reuse `src/eval_protocols.py` logic and `src/metrics.py` for the bootstrap: 1,000 draws, resampled over identities.
  - **Recognizers:** AdaFace is primary; MagFace and CurricularFace are reported as rows.
  - The official set-level verification protocol (G1/G2 pairs) goes to Phase 7, where set aggregation belongs.
- **Stage timings:** median ms per frame for detect, align+embed and track at native 800×600, on the lab PC.

## 3. Acceptance checks
Any failure is a STOP.
1. **Identity check:** ground-truth boxes fed as detections give IDF1 ≥ 0.99 for both trackers.
2. **Embedding check:** the video path (`embed.py`) and the Phase 1 aligned path (`norm_crop` + `Recognizer.get_embedding`) give cosine > 0.9999 on the same faces with the same keypoints, for 100 seeded faces. The default `get_embedding` output is unchanged by `return_raw` (`np.array_equal`).
3. **TrackEval check:** the vendored metrics reproduce TrackEval's published values on its bundled MOT17 sample. If the sample cannot be fetched, use a hand-built 2-object toy case with known IDF1 and MOTA.
4. **Determinism:** rerun one clip end to end. The detections, tracks and metric tables are byte-identical.
5. **Coverage:** every listed frame has a row, or a logged error.

## 4. Deliverable
`REPORT_phase6.md`, in the style of `REPORT_phase5.md`, with tables injected between markers. It contains:
- the tracker table, with crowded clips shown separately (qualitatively);
- the single-frame baseline with CIs;
- stage timings;
- ID-switch montages. These stay local, because ChokePoint images are not committed; the report describes them;
- Appendix T;
- limitations;
- decisions for Phase 7.

It ends with a STOP for the checkpoint. `CONTEXT.md` gets §3 and §11 updates, mirrored to the Drive.

## Appendix T: testing the Phase 5 detector on YouTube Faces and a custom video
Goal: measure how the chosen detector (`scrfd__full1280`, official preprocessing) performs outside WIDER FACE. The `full640` setting runs as the reference row. This is a diagnostic appendix: its numbers do not change the Phase 5 decision unless they reveal a failure, which would then be raised as a decision for Abser.

**T1. YouTube Faces** (Umar runs it; Claude writes the code).
- **Data:**
  - Download `frame_images_DB` (frames plus per-frame face boxes) into `Dataset/YouTubeFaces/`.
  - Print the file sizes before downloading.
  - Record the source, licence terms and SHA-256 in `DATASETS.md`, and change YTF to "held".
  - No frames go into git.
  - Read the YTF README and confirm the label-file format (the box is expected to be centre x, y, width, height) **before** writing the parser. Check it on 10 overlays.
- **Sample:** seeded (42), 1 video for each of 500 identities × 10 evenly spaced frames = 5,000 frames, stratified by the labelled face height.
- **Metrics:**
  - recall at IoU 0.5 and 0.3 per face-height bin (the bins from `configs/video.yaml`);
  - the IoU distribution;
  - a centre-in-box hit rate as a sensitivity check. YTF boxes come from a Viola-Jones detector and have a different shape from SCRFD boxes, so IoU 0.5 understates matches;
  - extra detections per frame. These are not called false positives, because YTF labels only the subject;
  - median ms per frame on the lab PC.
- **Reuse:** `src/video/wider_eval.py:bin_recall` (the IoU matching and bins) and the `phase5_merge.py` table code.

**T2. Custom video of consenting volunteers** (Abser runs the consent and recording; Claude writes the code).
- **Gate:** no recording before **both** of these are in place:
  1. written supervisor approval. The supervisor discouraged campus clips earlier, so this needs explicit sign-off for the volunteer version;
  2. a signed consent form from every person on camera.
- **Consent form.** Update `docs/requests/campus_clip_consent.md` to cover the purpose (detector testing and the demo), storage on the lab PC only, no publication of frames, and a deletion date. Settle the ethics-approval question with the supervisor.
- **Data handling:**
  - Clips go to `Dataset/custom_video/` (gitignored) on the lab PC only. Not on the Drive or GitHub.
  - Faces are never used to enrol identities outside this test.
  - Numbers from these clips appear only in Appendix T as a detector diagnostic, never as a headline result. The earlier rule was "demo only", so state this change in the report.
- **Recording protocol:**
  - 5–10 volunteers;
  - 2 phone or IP cameras mounted at about 2.5 m, looking down at a doorway or corridor, like ChokePoint;
  - 1080p at 30 fps;
  - 6 clips of 1–2 min: walking toward the camera from far to near (faces from about 16 px to over 100 px), normal and dim lighting, and one group pass with occlusion.
- **Labels:**
  - A small OpenCV click-to-box tool (`scripts/video_label_boxes.py`) labels all visible faces on 100 seeded frames, spread over the clips.
  - Two team members label the same 20 frames, and the agreement is reported as the IoU between labelers.
- **Metrics:**
  - recall per face-height bin at IoU 0.5;
  - false positives per frame (all faces are labelled here);
  - median ms per frame at 1080p and downscaled to 720p.
  - An overlay video is made for the team's visual check. It stays local.

**T3. Appendix output:** one table per source (`full1280` against `full640`), 4 overlay descriptions, and a short reading of where the detector fails: size, blur, pose or lighting.

## Future work (FYP II): VLM integration
Written as a section of `REPORT_phase6.md` and carried into the Phase 10 thesis chapter. No implementation in FYP I.

- **Where a VLM could add value:**
  1. **Description-based search** when the face is unusable: "person in a red jacket near door 2". Text-to-person retrieval over track crops with CLIP/SigLIP-style image-text embeddings. This complements face identity on tiny faces, which is where the Phase 1–3 results show recognition failing.
  2. **Natural-language queries over the track database:** a VLM or LLM front end that turns a question into index filters (time, camera, attributes, identity).
  3. **Track summaries and alert explanations** for an operator. These are generated per track, not per frame, to keep latency down.
- **Hard constraint (from the SR lesson):**
  - Generative models hallucinate. That is why SR was closed.
  - So VLM output is never identity evidence. Identity stays with the face recognizer and its scores.
  - VLM text is descriptive, shown with its source crops, and checked against the detector and tracker output.
- **Candidates:**
  - open-weight models only, run inference-only on the lab PC (20 GB), quantised if needed;
  - for example small Qwen-VL or InternVL-family models, Florence-2 for captioning and grounding, and CLIP/SigLIP for retrieval;
  - the exact versions and licences are to be checked at FYP II start.
- **Evaluation:**
  - text-based person search benchmarks, such as CUHK-PEDES, ICFG-PEDES and RSTPReid, subject to access;
  - attribute accuracy on our own consented clips;
  - hallucination rate, judged against the tracker ground truth;
  - latency per track.
- **Risks:**
  - privacy (attribute search is surveillance-sensitive);
  - demographic bias in descriptions;
  - compute;
  - the same proof standard as Phases 3–5: pre-registered tests and every number from a script.

## Order of work
| When | Who | Work |
|---|---|---|
| 10–12 Nov | Claude | `return_raw`, `io.py`, `embed.py`, `track.py`, TrackEval vendoring; acceptance checks 2 and 3 |
| 10–13 Nov | Umar | YTF download, format check, T1 sample; review `index.py` |
| 10–14 Nov | Abser | Supervisor sign-off and consent forms for T2; recording once both are in place |
| 12–17 Nov | Claude | ChokePoint detection (one GPU job), GT tracks, tracker runs and eval; checks 1, 4, 5 |
| 15–18 Nov | Claude | Single-frame protocol committed, then run; T1 run |
| 17–20 Nov | Claude + Abser | T2 labels and run (if the gate is cleared); stage timings |
| 20–22 Nov | Claude | `REPORT_phase6.md`, `CONTEXT.md`; commits as Umar or Abser on branch `phase_6`; push; **STOP** |

## Verification
- **Acceptance checks 1–5** above, each written to `outputs/video/phase6_checks.json`.
- **Unit-level checks:**
  - `return_raw` leaves the default output identical;
  - the raw norm matches Phase 2 `magface_norm` on 100 TinyFace probes (rel. diff < 1e-4, as in the H7 check).
- **End to end:** `scripts/run_video.py --stage all --dataset chokepoint --seq P1E_S1_C1` runs from a clean checkout and writes the parquet, the embeddings and a tracker row.
- **Appendix:** T1 and T2 tables come from `outputs/video/appendix/*.csv`; the overlays are viewed by a team member.
- **Git:** `git add -n .` shows no frames, clips, overlays or parquet files with image data.

## Risks
- **TrackEval does not run under numpy 2:** vendor and patch it. If that fails, the IDF1 and MOTA code is small enough to port, checked against the toy case.
- **ChokePoint has 54 identities:** CIs will be wide. Report them as they come.
- **YTF boxes are loose:** use the IoU 0.3 and centre-hit sensitivity checks.
- **T2 consent not cleared by 17 Nov:** Appendix T ships with T1 only, and T2 moves to Phase 9 (the demo), with the reason stated.
- **The schedule slip** pushes Phase 7 to W5–W6. Raise it at the checkpoint.
