# Sprint 2 — Video Stream Recognition & Tracking: Literature, Techniques, Datasets

**Project:** Identity-Preserving Adaptive Surveillance Restoration & Recognition Pipeline (Approach B)
**Last updated:** 2026-10-22
**Status:** Brainstorm. Nothing run.
**Trigger:** The supervisor brief of 2026-10-20 (`supervisor_brief_2026-10-20.pdf`) closes SR as a negative result. It makes the video pipeline the main deliverable for the next 8 weeks.
**Rules carried over:** inference only, no training; fixed seeds; every number from a script output; one GPU type per reported number.

> Legend: **[V]** verified from a primary page (paper, repo or official dataset page) on 2026-10-22 · **[R]** reported by a secondary source; re-check before citing · **[?]** unverified.

---

## 1. Scope and constraints

- **Pipeline (brief §7):** SCRFD or RetinaFace detection, then 5-point alignment (existing code), then the AdaFace embedding, with the MagFace norm as a quality weight. Then ByteTrack or BoT-SORT tracking, per-track embedding aggregation, and a FAISS index match. Restoration is an optional, gated stage that is off by default.
- **Experiments (brief §7):**
  1. Single-frame vs track-aggregated identification.
  2. Aggregation rule.
  3. Restoration gate on vs off.
  4. FPS at 720p.
- **Time:** 8 weeks, 3 members (plan in brief §8).
- **Compute:**
  - RTX 4000 Ada, 20 GB, in the lab. It has rebooted repeatedly, so run one GPU job at a time, with resumable runs.
  - 3 Colab T4 accounts.
  - The Linux home PC is for editing only and runs nothing.
- **Search window:** papers from 2025–2026. Older works appear only as required infrastructure or baselines and are marked as such.

---

## 2. Papers (2025–2026)

| # | Paper | Contribution relevant to us | Code / weights | Use in our project |
|---|---|---|---|---|
| P1 | **FaceMoE: Mixture of Experts for Low-Resolution Face Recognition.** Narayan & Patel (JHU), arXiv 2606.32040, Jun 2026 [V] | Swin-style MoE (3 experts, top-2 routing). Experts specialise by facial region. Sparse activation keeps pretrained knowledge when adapting to LR. Claims SOTA on 11 sets incl. TinyFace, IJB-S, BRIAR [V] | github.com/Kartik-3004/FaceMoE, MIT [V]. Weights on Hugging Face `kartiknarayan/FaceMoE` [R]. Input **120 px**, not 112 [V] | Candidate 5th recognizer, inference only. It needs its own preprocessing, since its input is 120 px, not our 112 alignment. Run it first on the Phase 1 TinyFace split to check that the claims transfer. |
| P2 | **FANVID: A Benchmark for Face and License Plate Recognition in Low-Resolution Videos.** Viswanathan, Sethi et al. (IIT Bombay), arXiv 2506.07304, 2025, v2 Feb 2026 [V] | 1,203 clips downsampled to 320×180, so faces are unreadable in a single frame by design. One HR mugshot per identity, plus distractors. Baseline (VSR + detection + recognition) scores 0.58 on face matching [V] | Hosted on Hugging Face "upon publication", CC BY-NC [V]. No dataset URL in the paper [V]. Identity count inconsistent (48 vs 63) [V] | The closest public test of "track aggregation beats single frame". Also tests SR's role in video, which links back to the Phase 3 result. |
| P3 | **FaceQSORT (FaceSORT): Multi-Face Tracking Based on Biometric and Appearance Features.** arXiv 2501.11741, Jan 2025 [V] | StrongSORT-style tracker that mixes face-recognition features with generic appearance features. Face features need frontal faces, so appearance covers the gaps [V] | [?] | Design template for our association cost: motion + IoU + quality-gated face embedding. |
| P4 | **TrackTrack: Focusing on Tracks for Online Multi-Object Tracking.** Shim et al., CVPR 2025 [V] | Track-Perspective Association and Track-Aware Initialisation. Heuristic, no training. Reported SOTA on MOT17, MOT20 and DanceTrack [V] | [?] | Possible drop-in replacement for the BoT-SORT association step. |
| P5 | **MOTIP: Multiple Object Tracking as ID Prediction.** Gao, Qi, Wang, CVPR 2025 [V] | End-to-end tracker that decodes ID labels for detections from trajectory history [V] | github.com/MCG-NJU/MOTIP [V] | **Out:** needs training on face tracks. Cite as the end-to-end alternative. |
| P6 | **Boosting Tiny Face Detection in Videos with an Integral Score Framework.** Leyva, Shen, Bahadir, Sanchez, Guha, FG 2025 [V] | Adds a tiny-face detector that densely scans multi-scale support regions with integral scores. Fast enough for HD video. Evaluated on WIDER FACE and MEVA [V] | No public code found [V] | Cite for the tiny-face detection gap. Do not plan on running it. |
| P7 | **ViT-FIQA: Assessing Face Image Quality using Vision Transformers.** Atzori, Boutros, Damer (Fraunhofer IGD), ICCVW 2025, arXiv 2508.13957 [V] | A learnable quality token on a ViT regresses a quality score, trained with the CR-FIQA loss. Strong on pose- and quality-challenging sets [V] | [?] | Second quality source for weighted aggregation, alongside the MagFace norm. |
| P8 | **A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition.** ICCV 2025, arXiv 2508.00053 [V]. **FarSight**, arXiv 2505.04616 [V] | Quality-guided fusion of face, body and gait scores on BRIAR. FarSight reports TAR@0.1% FAR rising from 48.5% to 66.4% on BRIAR face-included [V] | Tied to BRIAR | Related work: quality weighting and a body cue when the face is too small. |
| P9 | **Idiap low-resolution FR study,** IJCB 2026, arXiv 2608.06580 [R] | SR pipeline (RRDB + domain translation) 57.53 mAP vs direct feed 65.12 on TinyFace, with EdgeFace-base [R] | n/a | Independent support for closing SR. **Figures are from a secondary summary. Read the paper before citing.** |
| P10 | **50 Years of Automated Face Recognition,** arXiv 2505.24247, 2025 [V]. **Tracking-by-detection in MOT: Survey and Experiments,** arXiv 2609.08265, 2026 [V] | Surveys of template aggregation and of tracking-by-detection | n/a | Literature review backbone for the new chapter. |
| P11 | **SFE-DETR,** Sensors 2025 (already in `research/papers/B`) | Transformer face detector for small faces | Unverified (brief §7) | Optional detector arm. Drop it if code is still missing in week 1. |

**Required pre-2025 infrastructure and baselines** (outside the search window, cited as established):
- ByteTrack (ECCV 2022) and BoT-SORT (2022).
- Norm Pooling and Sparse Pooling for templates (arXiv 2409.09832, 2024).
- **CAFace** (Kim et al., NeurIPS 2022): a pretrained set-to-feature aggregator built on AdaFace WebFace4M. Code is MIT and both checkpoints are on Google Drive [V]. It is the only learned aggregator we can use without training, and it matches our recognizer.
- NAN (CVPR 2017).

**Gap found:** no 2025–2026 *method* does tracklet-level face aggregation on low-resolution surveillance video. FANVID (P2) is a benchmark and FaceQSORT (P3) is a tracker. Neither measures how aggregation rules change identification. This gap carries the thesis framing from the brief, "tracking uses real information where SR invents it".

---

## 3. Techniques (all inference-only)

1. **Track aggregation rules** (experiment 2), on AdaFace embeddings:
   - Plain mean.
   - Mean weighted by MagFace norm.
   - Top-k frames by quality.
   - Norm pooling (2409.09832).
   - CAFace (pretrained).
   - Baseline: per-frame majority vote, which needs no aggregation.
2. **Quality source:**
   - The MagFace norm, which we already have.
   - ViT-FIQA or CR-FIQA, if its code runs.
   - Report whether the two quality sources agree, as Phase 2 did for its measures.
3. **Face-aware association:**
   - The BoT-SORT/ByteTrack cost is motion + IoU.
   - Add a face-embedding term only when quality is above a threshold (FaceQSORT idea).
   - Tiny, non-frontal faces then fall back to motion.
4. **Person-then-face tracking:**
   - Track bodies with a COCO person detector (YOLO or RT-DETR) plus ByteTrack.
   - Attach face detections that fall inside each body box.
   - Body tracks survive frames where the face is 10–20 px or turned away. This is likely the biggest robustness gain on CCTV.
5. **Tiny-face recall without training:**
   - Run the detector on sliced or tiled frames (SAHI-style) and on an upscaled frame.
   - Report detection rate by face-size bin. SCRFD found only 31% of TinyFace distractors (brief §7).
6. **Open-set track decision:**
   - Compare the track score with the gallery at a fixed FPIR, with a "not enrolled" outcome.
   - Report TPIR@FPIR, as in Phase 1 SurvFace open-set.
7. **Retrieval:**
   - FAISS `IndexFlatIP` is exact, and our galleries are small.
   - Compare it with HNSW only for the speed/recall table.
8. **Restoration gate** (experiment 3):
   - Use only the winner of the CPU low-light/deblur test (brief §6), applied per frame before embedding.
   - It stays off by default.
   - If the stop rule fires, report the gate as tested and closed.

Reusable repo code:
- `src/align.py` (5-point alignment).
- `src/recognizer.py` (four recognizers, weights SHA-checked).
- `src/metrics.py` and `src/eval_protocols.py` (Rank-1, TAR@FAR, TPIR, bootstrap CIs).
- `src/stratify/measures.py` (luminance and sharpness, for per-frame quality bins).

---

## 4. Datasets

### 4.1 Open / freely obtainable (primary)

| Dataset | What it is | Pros | Cons |
|---|---|---|---|
| **ChokePoint** (NICTA, arma.sourceforge.net/chokepoint, Zenodo 815657) [V] | 3 cameras above portals; 800×600, 30 fps; 48 sequences; 25 + 29 subjects, recorded a month apart; 64,204 face crops at 96×96 from manual eye labels; `groundtruth.tar.xz`; G1/G2 verification protocol; 2 crowded sequences | Real CCTV-style video with identity labels. Free for non-commercial research [V]. About 12 GB, fits Colab/Drive. Video-to-video protocol exists. Ground truth makes detection recall measurable. | From 2010. Only 54 identities, so CIs will be wide. Mostly one person per frame, so it tests tracking little. Faces larger than QMUL. |
| **YouTube Faces** (TAU, now via an Open University portal) [V] | 3,425 videos of 1,595 identities; frames, aligned faces, descriptors; 5,000-pair verification protocol | Standard video face benchmark, so results compare with published numbers. Large enough for tight CIs on single-frame vs aggregated. | Not surveillance; celebrities. Frames are pre-cropped around the face, so it does not test detection or tracking. Portal registration, and the portal may be down [?]. About 25 GB. |
| **FANVID** (2025) [V] | YouTube clips downsampled to 320×180; one HR mugshot per identity; distractor faces | Designed so that single frames fail and temporal aggregation is needed. HR-mugshot gallery matches the enrolment scenario. 2025 paper. | Not yet public: "released upon publication", CC BY-NC [V]. Small (48 or 63 identities). Degradation is synthetic downsampling. Celebrities. |
| **MEVA / MEVID** (Kitware, mevadata.org) [V] | 328 h (516 GB) of multi-camera ground video with consenting hired actors; EO + thermal; AWS S3; CC BY 4.0. MEVID adds person re-ID labels: 158 people, 598 outfits, 8,092 tracklets, 33 views | Real outdoor and indoor surveillance cameras, documented consent, open licence. Person tracklets with identities support person-then-face tracking. The FG 2025 tiny-face paper uses it. | No face annotations [V], so face recognition metrics need our own labelling. Very large: download only selected clips. Faces are often a few pixels. |
| **VoxCeleb2** [V] | 6,112 celebrities; YouTube URLs, face detections, face tracks, cropped face videos; CC BY 4.0 (one mirror says BY-SA) | Huge, already organised as face tracks, open licence. Good stress test of aggregation rules at scale. | Talking heads at high quality, nothing like CCTV. YouTube link rot; the cropped videos are the usable part. |
| **MOT17 / MOT20 / DanceTrack** | Pedestrian / dancer MOT benchmarks | Checks that the tracker integration is correct (HOTA, IDF1) before adding faces | No face identities |
| **WIDER FACE** | Face detection images with size variety | Detection rate by face size (experiment in brief week 1) | Images, not video |
| **Own campus clips** (written consent) | Demo footage | The real use case | Demo only, never reported metrics (brief §7) |

### 4.2 Locked / high fidelity (secondary)

| Dataset | Access | Pros | Cons |
|---|---|---|---|
| **IJB-S** (IARPA Janus Surveillance) | NIST stopped distributing IJB-A/B/C on 2023-03-14 [V]; IJB-S status unknown [?] | The gold standard: real surveillance video matched to HR enrolment. FaceMoE reports on it, so results would be comparable. | Probably unobtainable. Needs a licence signed by the department (brief §9 ask 3). Weeks of lead time even if available. |
| **BRIAR** (IARPA / ORNL) | US government programme; no academic request route found [V] | Long range, elevation, whole body; current SOTA venue (P1, P8) | Practically unavailable to us. Cite as a limitation. |
| **COX Face DB** (ICT-CAS, sponsored by OMRON) | Signed release agreement via vipl.ict.ac.cn [V] | Built for video-to-still and still-to-video: about 1,000 subjects, 3 camcorders, HR still gallery. Closest to our enrolment-plus-CCTV scenario. | Response time unknown. From 2015. Licence restrictions on publication. |
| **SCface** | Written request to Prof. Mislav Grgic, signed agreement, no fee, no redistribution even inside the institution [V] | Real surveillance cameras at 3 measured distances, visible + IR, HR mugshots. Good for detection and recognition by distance. | Still images, so no tracking. 130 subjects. Each member may need to be named on the agreement. |
| **QMUL-SurvFace / TinyFace** | Already held | Existing benchmark and splits. Reuse for the detector size study and to place FaceMoE. | Still crops: no frames, no tracks |

---

## 5. Feasibility ranking (8 weeks, RTX 4000 Ada + 3× T4, inference only)

| Rank | Item | Feasibility | Reason |
|---|---|---|---|
| 1 | **Core pipeline:** SCRFD + existing alignment + AdaFace + ByteTrack/BoT-SORT + FAISS Flat | High | Every part has public weights. Reuses `src/`. Fits a T4. Needed for any other result. |
| 2 | **Aggregation experiment** (mean / MagFace-weighted / top-k / norm pooling / majority vote) on **ChokePoint** + **YouTube Faces** | High | Training-free, open data, reuses Phase 1 metrics and bootstrap CIs. The headline new result. |
| 3 | **CAFace** as a learned aggregator arm | High | MIT, checkpoints public [V], built on AdaFace. One extra row in the rank-2 table. |
| 4 | **Detection rate by face size** (SCRFD vs RetinaFace, ± tiling) on TinyFace / SurvFace / WIDER FACE / ChokePoint | High | Planned for week 1 already. Cheap. Explains where the pipeline fails. |
| 5 | **Person-then-face tracking** on selected **MEVA** clips | Medium | Strongest answer to tiny faces. But there are no face labels, so metrics are limited to track continuity and qualitative results. Download clips selectively (about 516 GB total). |
| 6 | **FaceMoE** as a 5th recognizer | Medium | MIT code, weights on HF [R]. Its 120 px input needs separate preprocessing. Placing it on TinyFace takes about 1 day of Colab. |
| 7 | **COX Face DB / SCface** requests | Medium–low | Send in week 1. Use only if they arrive by week 3. Otherwise list them as future work. |
| 8 | **FANVID** | Low now | Data not yet public [V]. Email the authors in week 1. If released, it moves to rank 2. |
| 9 | **TrackTrack** association, **ViT-FIQA** quality | Low–medium | Optional ablations. Depends on code availability [?]. |
| 10 | **IJB-S, BRIAR** | Very low | Access blocked or unknown. State as a limitation. |
| — | **MOTIP, Integral-Score detector** | Out | Training needed (MOTIP) or no code (Integral Score). Cite only. |

### Suggested week-1 actions (fits brief §8)
- **Abser:**
  - Download ChokePoint and request YouTube Faces.
  - Email the FANVID authors.
  - Send the COX and SCface requests.
  - Ask the supervisor about IJB-S.
- **Umar:**
  - Run the detector size study, including tiling.
  - Check the FaceMoE and CAFace weights and record their SHA-256 in `MODEL_INVENTORY.md`.
- **Raghib:** run the CPU restoration test as planned.

---

## 6. Open questions for the supervisor
1. Is ChokePoint + YouTube Faces acceptable as the reported video benchmark if IJB-S, COX and FANVID do not arrive in time?
2. May we add FaceMoE (2026) as a fifth recognizer? It adds a 2026 baseline, but its preprocessing differs from the other four.
3. Is person-then-face tracking on MEVA (qualitative and track-continuity metrics only) acceptable as a robustness demonstration?

---

## Sources
- FaceMoE: https://arxiv.org/abs/2606.32040 · https://github.com/Kartik-3004/FaceMoE
- FANVID: https://arxiv.org/html/2506.07304v2
- FaceQSORT: https://arxiv.org/abs/2501.11741
- TrackTrack: https://mlanthology.org/cvpr/2025/shim2025cvpr-focusing
- MOTIP: https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html · https://github.com/MCG-NJU/MOTIP
- Integral Score tiny-face detection (FG 2025): https://eprints.gla.ac.uk/355100
- ViT-FIQA: https://arxiv.org/abs/2508.13957
- Quality-guided score fusion (ICCV 2025): https://arxiv.org/abs/2508.00053 · FarSight: https://arxiv.org/abs/2505.04616
- Idiap IJCB 2026 summary: https://www.insightface.ai/blog/august-2026-face-recognition-papers
- Surveys: https://arxiv.org/abs/2505.24247 · https://arxiv.org/abs/2609.08265
- Template pooling: https://arxiv.org/abs/2409.09832 · CAFace: https://github.com/mk-minchul/CAFace
- ChokePoint: https://arma.sourceforge.net/chokepoint/ · YouTube Faces: https://www.cs.tau.ac.il/~wolf/ytfaces/
- MEVA: https://mevadata.org/ · MEVID: https://github.com/Kitware/MEVID
- VoxCeleb2 licence: https://mm.kaist.ac.kr/datasets/voxceleb/files/license.txt
- NIST IJB distribution notice: https://www.nist.gov/node/761821
- COX Face DB record: https://data.4tu.nl/export/cff/datasets/12712559
- SCface: https://www.scface.org
