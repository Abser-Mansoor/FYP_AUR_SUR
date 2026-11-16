# Study C: audit of unmatched detections (labelling protocol)

Pre-registration note for study C of `REPORT_phase6.md` section 0b. To be committed **before any label is made**. Written 2026-11-12 for Abser, Raghib and Umar, who share the labelling. Tool: `scripts/video_studyC_audit.py`.

## 1. Question
The Phase 6 ChokePoint run produced 56,571 detections (score 0.5 or higher, in the 72 ground-truth clips) that match no ground-truth box at IoU 0.5. Two frames we looked at showed a reflection in the glass door and the subject's own face in frames the eye annotation skips. We do not know the split. Study C measures it on a sample, by eye, without the recogniser's help:
1. What share of the unmatched detections is **not a live face** (reflection, image of a face, non-face)? This is the detector's real false-positive rate on this footage.
2. What share is a **live face of a person who is annotated in the same clip** (an annotation gap)?
3. What share is a **live face of someone else** (not annotated in the clip)?

Outcome 1 is primary. Outcomes 2 and 3 are secondary. No claim about the recogniser is made from study C (see section 8).

## 2. Sample (fixed by the script, seed 42)
- Population: every detection with score 0.5 or higher in the 72 ground-truth clips that matches no ground-truth box (the unmatched detections of `outputs/video/baseline/matches.csv`).
- Unit: a **track fragment** = the unmatched detections of one ByteTrack track in one clip. A detection that has no track is its own unit. One detection is drawn at random from each unit that is sampled, so one person walking past does not fill the sample.
- Draw: 350 units uniformly at random (`numpy.random.default_rng(42)`). Each sampled detection carries a weight equal to the number of unmatched detections in its unit, so that estimates are per detection, not per unit.
- Split among the three labellers (the script shuffles with the same seed):
  - **50 shared crops**, labelled by all three. They measure agreement.
  - **300 own crops**, 100 for each labeller, labelled by one person.
- The sample list (`sheet.csv`) holds clip, frame, detection id, box, score and unit; it holds no image and no label. Its SHA-256 is recorded in the report.

## 3. What the labeller sees
The tool shows, for each crop: (a) the face box region with a margin, enlarged; (b) the full frame with the box drawn; (c) up to two **reference crops** of annotated people from the same clip, from the nearest annotated frame within 30 frames. The labeller does **not** see the detector score, the track, any recogniser output, or anyone else's label.
Reference crops come from the annotation (eye labels turned into boxes), not from the recogniser, so they are not a hint about what the model thinks.

## 4. Labels (one per crop)
| Key | Label | Use it when |
|---|---|---|
| `a` | live face, **annotated person** | A real, live face that is clearly the same person as one of the reference crops (same clip, close in time) |
| `o` | live face, **other person** | A real, live face that is not any of the shown references, or there is no reference and the face is clearly real |
| `r` | reflection or image of a face | The face is a reflection (glass door, window, mirror, screen) or a picture of a face (poster, photo, screen) |
| `n` | not a face | The box is on something that is not a face (back of head, body, object, background) |
| `u` | unclear | You cannot decide among the above after one honest look (too small, too blurred, too occluded) |

Decision rules, in this order:
1. **Reflection of an annotated person is `r`, not `a`.** The reflection of a walking person in the glass door is a reflection, whoever it shows.
2. **Profile or partly hidden faces are faces** if a face is recognisably there (eye, nose or mouth region visible): `a` or `o`, not `n`.
3. **Back of a head** (hair only) is `n`.
4. **Two faces in one box:** label the larger face.
5. **Box covers only part of the face** (forehead or chin only) but the face is clearly there: `a` or `o`.
6. **Choosing between `a` and `o`:** `a` only if you would say, looking at the reference, that it is the same person. If you hesitate between `a` and `o`, use `o` (stricter claim goes with `a`). If you hesitate between a face and `u`, use `u`.
7. Do not zoom, enhance or search for more context than the tool shows. One look, one key.
8. Never talk about a crop with another labeller before **all three** have finished the shared 50.

## 5. Procedure
1. `envs\video\Scripts\python.exe scripts\video_studyC_audit.py sample` (once, on the lab PC; writes `outputs/video/studyC/sheet.csv`).
2. Each labeller runs `video_studyC_audit.py label --who abser|raghib|umar --pack <pack folder> --stage shared` (the 50 shared crops) and later `--stage own` (their 100), see Amendment 1 below. The tool saves after every label and resumes where it stopped. `b` goes back one crop, `q` saves and quits. The pack is made by `video_studyC_audit.py pack` after `sample`.
3. After all three have finished the shared 50: `... video_studyC_audit.py agreement`. If Fleiss' kappa on the shared 50 is **below 0.60**, the labellers meet, write down which rule was unclear, and the protocol is revised and **re-committed (version 2) before anyone labels the 300 own crops**. The shared 50 labels already made are kept and reported as they were (version 1); the labellers relabel them under version 2 if the rules changed, and both versions are reported.
4. After all labelling: `... video_studyC_audit.py analyze`.

**Amendment 1 (2026-11-12, before any label was made): labelling at home.** Abser decided that each of the three labels at home over the weekend instead of on the lab PC. What changes, and the conditions that go with it:
- The labellers receive a **pack** (`studyC_pack.zip`: 350 pre-rendered JPEGs of about 24 MB in total, a blind sheet, the labelling script, this protocol and a README), made on the lab PC by `video_studyC_audit.py pack`. The pack holds only the 350 sampled crops with their frames. It holds no detector score, clip name, track, ground-truth list or recogniser output, and no other ChokePoint frames. The full sample list (`sheet.csv`, with scores) stays on the lab PC; its SHA-256 is in the pack.
- ChokePoint is licensed for non-commercial research (CC BY-NC 4.0) and the crops show real people. The pack is used only by the three team members, only for this audit, is not copied to anyone else, not uploaded anywhere public and not committed to git. Each labeller **deletes the pack after sending the labels back**. Only the label files (sample id and letter) come back.
- The labelling is done in two **stages** so that the agreement check still comes before the own crops: Stage 1: everyone labels the 50 shared crops (`--stage shared`) and sends `labels_<who>.csv`. Claude runs `agreement` on the lab PC. If kappa is 0.60 or higher, Stage 2: everyone labels their own 100 (`--stage own`) and sends the file again. If kappa is below 0.60, the rules are revised and re-committed first (section 5, step 3), and Stage 1 is repeated.
- Labellers work **alone** and do not discuss any crop with each other until all three have sent Stage 1. A screen at least 1,200 pixels wide is needed (the window is 1,180 x 480); do not rescale it.
- Home labelling adds uncontrolled factors (screen, lighting, interruptions). They are not measured; the shared 50 and kappa are what shows whether they matter.

## 6. Final label of a crop
- Own crop: the label as made.
- Shared crop: the label chosen by at least 2 of the 3 labellers; if all three differ, `u`. The individual labels are kept in the output.

## 7. Analysis (fixed now)
- **Composition:** the weighted share of each label among all unmatched detections, with a 95% CI from a cluster bootstrap over clips (1,000 draws, seed 42). Estimated counts = share x total unmatched detections (from the sheet). Also by group (P1E, P1L, P2E, P2L) and by detector score band (0.5 to 0.7, 0.7 to 0.85, 0.85 and above).
- **Primary outcome:** the share of `r` plus `n` (not a live face), reported with its CI. `u` is reported separately and **not** counted in either direction; a sensitivity row counts `u` as not-a-face and another as live-face.
- **Agreement (shared 50):** percent of crops on which all three agree, pairwise percent agreement, and Fleiss' kappa over the five labels. Reported as found; no label is dropped.
- **Corrected counts for the tracker table:** with the primary outcome, the report states how many of the 56,571 unmatched detections are estimated to be real faces without an annotation (labels `a` and `o`). MOTA and DetA are **not** recomputed from a sample: the sample is too small to remove individual detections from the metric.

## 8. What study C can and cannot say
- It estimates the split of the unmatched detections. With 350 crops the CI on a share is about plus or minus 5 points at best (for a share of 0.5), wider for rare labels.
- `o` means "not any person shown as a reference", **not** "not enrolled": an enrolled identity can appear in a clip without being annotated in the nearby frames. So study C does **not** supply confirmed non-enrolled probes, and no open-set false-accept number is derived from it. If Phase 7 needs impostor probes from real faces, that needs a different labelling (identity-level), pre-registered separately.
- Labels are human judgements; the labellers know the project and may expect reflections. The shared 50 and kappa show how much that matters.

## 9. Output files (`outputs/video/studyC/`)
`sheet.csv`, `sheet_meta.json`, `labels_<who>.csv`, `agreement.json`, `final_labels.csv`, `composition.csv`. Small CSV and JSON files are tracked in git; the pack (`outputs/video/studyC/pack/`, with its images) is never committed.
