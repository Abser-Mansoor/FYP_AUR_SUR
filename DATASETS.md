# Datasets for the video phases (Phase 5 onwards)

Status values: **held** (downloaded and verified), **requested** (request drafted in `docs/requests/`, not yet sent unless stated), **blocked**, **not requested**. Dates are access dates. Datasets are read-only after the checksum.

| Dataset | Status | Source | Licence | Size / checksum | Notes |
|---|---|---|---|---|---|
| ChokePoint | held (downloaded and extracted 2026-10-22; `outputs/chokepoint_download.log`) | zenodo.org/records/815657, arma.sourceforge.net/chokepoint | CC BY-NC 4.0 (Zenodo record); the NICTA page gives non-commercial research terms and requires citing Wong et al., CVPRW 2011. Both forbid commercial use, so no ChokePoint image is committed to git | 25 files, 11.71 GB; per-file md5 from the Zenodo API, checked by `scripts/video_download_chokepoint.py` | 800x600, 30 fps, AXIS P1343. Frame archives hold nested per-camera tarballs. Eye ground truth in `groundtruth.tar.xz`. |
| WIDER FACE val | held (2026-10-22) | official Google Drive (`WIDER_val.zip`), shuoyang1213.me/WIDERFACE (`wider_face_split.zip`, `eval_tools.zip`) | WIDER FACE terms (non-commercial research) | WIDER_val.zip 362,752,168 B, sha256 f9efbd09f28c5d2d...; wider_face_split.zip sha256 c7561e4f5e7a118c...; eval_tools.zip sha256 1cf49c8243fa8a16... | 3,226 val images; official counts easy/medium/hard = 7,211 / 13,319 / 31,958 faces (reproduced by `src/video/wider_eval.py`). |
| TinyFace, QMUL-SurvFace | held (Phase 0) | QMUL | see dataset pages | `Dataset/` | Still crops, used for the crop detection rate. |
| YouTube Faces | held (downloaded by the team on home PCs and transferred, extracted 2026-11-19 to `Dataset/YouTubeFaces/`) | cs.tau.ac.il/~wolf/ytfaces (registered access; the host is unreachable from the lab network) | usage terms not yet recorded (internet bandwidth problem; to be added) | `frame_images_DB.tar.gz` sha256 09bac98f...ff828, `meta_data.tar.gz` sha256 10ba25da...29c8 (full list in `outputs/video/ytf/archives_sha256.txt`) | Counts verified against the official numbers: 3,425 videos, 1,595 people, 621,126 frames, 5,000 pairs in 10 folds of 500 (250 same-person pairs each; `meta_data/meta_and_splits.mat`, columns: 1-based video index 1, video index 2, same-person flag). Labelled boxes (`<person>.labeled_faces.txt`) are centre x, centre y, width, height (checked on 10 overlays). The Kaggle package in `Dataset/YTFaces` (2,194 videos, cropped frames) is a different, smaller copy used only for Phase 6 Appendix T1. |
| MEVID (test split, probe only) | held (probe only; downloaded and extracted 2026-11-23 to `Dataset/MEVID/`) | github.com/Kitware/MEVID; archives from `mevadata-public-01.s3.amazonaws.com/mevid-annotations/` | CC BY 4.0 (`MEVA-data-license.txt`) and consented actors (mevadata.org), as stated in the probe plan; the licence file itself was not re-read by us | mevid-v1-annotation-data.zip 12,211,153 B sha256 18a91691...82e8; mevid-v1-actor-checkin-photos.zip 628,139,082 B sha256 ad0e2560...3874; mevid-v1-bbox-test.tgz 13,974,475,339 B sha256 4d4ec267...221d (full values in `outputs/video/mevid_probe/archives_sha256.txt`) | Counts verified against the README (54 identities, 113 outfits, 1,754 test tracklets; 316 query and 1,438 gallery; 1,186,251 body chips, all present; 2,898 actor check-in photos, 54 of 54 test persons covered). The S3 host gave about 0.2 to 0.3 MB/s per connection from the lab network, so the archives were fetched with 32 parallel range requests (`scripts/mevid_download.py`) and checked by size, zip CRC and a full gzip read. Train package and videos not downloaded. No MEVID image is committed. |
| FANVID | request sent (reported by Abser 2026-11-03; draft `docs/requests/fanvid_authors.md`), awaiting reply | arXiv 2506.07304 | CC BY-NC, "upon publication" (per the paper) | - | Not public at the time of the notes. |
| COX Face DB | unavailable (reported by Abser 2026-11-03; reason not recorded; draft `docs/requests/coxface_request.md`) | vipl.ict.ac.cn | release agreement | - | Video-to-still protocol. |
| SCface | request sent (reported by Abser 2026-11-03; draft `docs/requests/scface_request.md`), awaiting reply | scface.org | signed agreement, no redistribution | - | Stills at 3 distances, no tracking. |
| IJB-S | unavailable (reported by Abser 2026-11-03; reason not recorded; draft `docs/requests/ijbs_supervisor_question.md`) | NIST / IARPA | licence | - | NIST stopped distributing IJB-A/B/C in March 2023; IJB-S status unconfirmed. |
| BRIAR | blocked | IARPA / ORNL | US government programme | - | No academic request route found in the Sprint 2 notes. |
| Campus demo clips | not pursued: the supervisor discourages it (reported by Abser 2026-11-03); consent form drafted (`docs/requests/campus_clip_consent.md`) | own recordings | consent form | - | Demo only, never used for reported metrics. Ethics approval question open. |

Access requests (reported by Abser, 2026-11-03): requests were sent to FANVID and SCface, and to one further dataset whose name he did not remember (to be identified; the status of the other rows is unchanged). Replies are awaited. COX Face DB and IJB-S are unavailable. The campus clip consent is discouraged by the supervisor and is not pursued.

## ChokePoint ground-truth parse (`scripts/video_chokepoint_gt.py`, run 2026-10-22)
Reported as found; the page figures were not adjusted to match.

| Quantity | Parsed from `groundtruth.tar.xz` | Dataset page (per the Sprint 2 notes) |
|---|---|---|
| Annotation files (clips) | 72 (24 P1 clips named Pn_Sk_Cj, plus 48 P2 clips split into recordings .1 and .2) | "48 sequences" |
| Frames listed | 176,916 | - |
| Labelled person-frame entries (both eyes) | 72,289 | 64,204 face crops |
| Identities | 29 distinct ids | 25 + 29 subjects, recorded a month apart |
| Entries missing an eye | 2 (P2E_S3_C1.1 frames 2119 and 2557), skipped | - |

The notes describe 64,204 as the number of 96x96 face crops made from the eye labels. Whether some entries were filtered out was not checked, so the difference is unexplained. The P2E_S5 and P2L_S5 frame archives have no ground-truth file, so they are not used for recall.

## ChokePoint protocol
Source: the ChokePoint page at http://arma.sourceforge.net/chokepoint/ ("Protocol" section), fetched 2026-11-03, and the Zenodo record 815657 (CC BY-NC 4.0, published 2017-06-22). The `README.txt` that came with the Zenodo download has no protocol section: it covers the recording setup, the sequence naming and the folder layout only. We define no split of our own; the text below is the page's, shortened.

**What it is.** A baseline **verification** protocol (`protocol_baseline`). The video sequences are divided into two groups, G1 and G2. Each group is the development set and the evaluation set in turn. In each group, all possible genuine and impostor pairs are generated. Parameters and the threshold are learned on the development set and applied to the evaluation set. The average verification rate is the reported result. It is a pair-verification protocol: it does not define a tracking or identification split.

| Group | Sequences |
|---|---|
| G1 | P1E_S1_C1, P1E_S2_C2, P2E_S2_C2, P2E_S1_C3, P1L_S1_C1, P1L_S2_C2, P2L_S2_C2, P2L_S1_C1 |
| G2 | P1E_S3_C3, P1E_S4_C1, P2E_S4_C2, P2E_S3_C1, P1L_S3_C3, P1L_S4_C1, P2L_S4_C2, P2L_S3_C3 |

The page also lists three case studies (indoor only and short interval; indoor and outdoor and short interval; indoor and outdoor and long interval, with set 1 matched against set 2) and two evaluation conditions: single camera (the camera with the most frontal view, as listed in the tables above) and multiple camera (all three cameras of the portal and sequence).

**Notes for Phase 6.**
- The two crowded sequences P2E_S5 and P2L_S5 are not in G1 or G2.
- The protocol uses only the camera and sequence names above. Our ground-truth parse found 72 annotation files and 72,289 entries, which do not match the page counts (see the table above); the page lists 48 sequences and 64,204 face images.
- Whether to use this protocol for the Phase 6 evaluation, or to define a tracking-specific one, is an open decision.
