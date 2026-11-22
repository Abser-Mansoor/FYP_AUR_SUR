# MEVID feasibility probe (for Umar + Claude Code on the lab PC)

**Status:** approved 2026-11-23 (Abser). Executor: Claude Code on the lab PC with Umar. This file is the pre-registration: commit it before step 1.

## Context
- Phase 7's open-set result is inconclusive: there are only 6 non-enrolled ChokePoint identities.
- UCCS was removed from the web, and PaSC is blocked. VGGFace2 and the other celebrity web sets risk identity overlap with our recognizers' training data (MS1MV2, WebFace).
- Abser worries, rightly, that confirming choices on the data used to make them (R1, ByteTrack, detector mode) is circular. A held-out dataset is needed.
- **MEVID** (Kitware, WACV 2023; README read 2026-11-23) fits:
  - consented hired actors from MEVA, licence CC BY 4.0, so no celebrity overlap;
  - real multi-camera surveillance;
  - person tracklets averaging about 590 frames, with person and outfit IDs;
  - **actor check-in photos**, a possible high-resolution enrolment gallery.
- Unknowns the probe settles:
  1. how often faces are visible and how large they are inside the body crops;
  2. whether the check-in photos map to person IDs and contain detectable faces.
- The probe measures **data availability only. It computes no recognition score**, so a later MEVID protocol stays uncontaminated.
- Output: a go/no-go verdict by a rule fixed here, and `outputs/video/mevid_probe/`.
- After approval, save this plan as `research/Sprint_Notes/MEVID_feasibility_probe.md`, commit it to `main` before the probe runs (that is the pre-registration), and copy it to Drive `FYP_SR_SHARED/`.

## How Claude uses this file
- Explain each step to Umar in one or two sentences before running it. **[UMAR]** = stop and get his confirmation.
- **STOP** = do not work around the problem. Record the command, the error and the file state; Umar posts it to the team chat; wait.
- Rules:
  - inference only;
  - numbers only from script outputs;
  - seed 42;
  - `Dataset/` is read-only after checksums;
  - lab PC only;
  - one GPU job at a time;
  - no MEVID image is ever committed (the repo is public);
  - no recognition or identity scores in this probe.
- Lab-PC quirks: Windows paths; `envs\video\Scripts\python.exe`; long `sleep` is blocked; launch long jobs detached and check their logs.

## Step 0: setup
- `git checkout main && git pull && git checkout -b mevid_probe`.
- **[UMAR]** confirm the commit identity (`git -c user.name=... -c user.email=...` if it isn't his; never change the global config).
- Confirm D: has at least 20 GB free.

## Step 1: download (about 14 GB; S3 over HTTPS)
- Into `Dataset/MEVID/archives/`, from `https://mevadata-public-01.s3.amazonaws.com/mevid-annotations/`:
  - `mevid-v1-annotation-data.zip`
  - `mevid-v1-actor-checkin-photos.zip` (600 MB)
  - `mevid-v1-bbox-test.tgz` (13 GB)
- Use `curl -L -C - -o <file> <url>` so a broken download resumes.
- Record SHA-256 and bytes of each archive in `outputs/video/mevid_probe/archives_sha256.txt`.
- Extract to `Dataset/MEVID/`. Do not download the train package (30.5 GB) or the videos (127 GB) in the probe.
- If S3 is unreachable from the lab network (YTF's host was): **STOP**, and use the home-PC download → Drive → lab route as for YTF.
- Add a `DATASETS.md` row:
  - source github.com/Kitware/MEVID;
  - licence CC BY 4.0 (`MEVA-data-license.txt`), consented actors per mevadata.org;
  - sizes, SHA-256, access date;
  - status "held (probe only)".

## Step 2: inventory (no GPU) → `inventory.json`
- Parse `track_test_info.txt`, `test_name.txt` and `query_IDX.txt`.
- **Verify the README counts:** 54 identities, 113 outfits, 1,754 tracklets. Also count query vs gallery tracklets and chips per tracklet. A mismatch is reported as found, never fixed.
- Body-chip sizes: height and width percentiles on a seeded sample of 2,000 chips.
- **Check-in photos:**
  - list the files;
  - work out the person-ID mapping from file and folder names;
  - count photos per person; image size percentiles;
  - how many of the 54 test identities (and of all 158) have at least one photo.
- **[UMAR]** if no person-ID mapping can be found, show 5 file names and 5 photos (local viewer only) and record his reading.
  - If the mapping cannot be made: the high-resolution gallery route is **no-go**, and the probe continues for the tracklet-only route (step 5).

## Step 3: tracklet sample (fixed now)
- From the 1,754 test tracklets, draw **200**, stratified by person: each of the 54 persons gets ⌊200/54⌋ = 3, with the remainder drawn at random. `numpy.random.default_rng(42)`. If a person has fewer than 3 tracklets, take all of them and redistribute the shortfall at random.
- From each tracklet, **20 frames evenly spaced** over its chips (all chips if it has fewer than 20).
- Write `sample.csv` (tracklet index, person, outfit, camera, chip file). It is committed **before** detection runs.

## Step 4: face detection (GPU, about minutes)
- New script `scripts/mevid_probe.py --part inventory|sample|detect|summary`. It reuses `src/video/detect.py`:
  - `SCRFDDetector("models/detector/det_10g.onnx", preprocess="official")`;
  - `det.detect(bgr, size=640)` (the body chip is resized so its longest side is 640, i.e. the `full640` operating point);
  - SHA-256 of `det_10g.onnx` checked against `MODEL_INVENTORY.md`.
- **Face assignment per chip (fixed now):**
  - take the highest-scoring detection with score ≥ 0.5 whose box centre lies in the upper 50% of the chip (one person per chip; this avoids hands and background people);
  - face height = box height in **native chip pixels**;
  - none → "no face".
- **Usable face:** score ≥ 0.5 and native face height ≥ 16 px. The 16 px bound comes from Phase 7: single-frame Rank-1 is 0.994 at 16 px and falls to 0.84 at 12 px.
- **Check-in photos:** the same detector on every photo of the 54 test identities. A photo is a **usable gallery photo** if its largest face has score ≥ 0.5 and height ≥ 64 px.
- Write:
  - `detections.csv` (chip, score, box, face height, assignment);
  - `checkin_faces.csv`;
  - `overlays/` (20 seeded chips and 10 photos; **local only, gitignored**).
- **[UMAR]** open the overlays and confirm the boxes sit on faces. Record the answer.

## Step 5: summary and verdict → `summary.json`, `probe_tables.md`
**Reported:**
- share of sampled frames with a face and with a usable face;
- face-height percentiles (5/25/50/75/95);
- share of tracklets with at least 5 usable faces out of 20;
- the same by camera;
- number of persons with at least 1 usable tracklet;
- check-in coverage.
- CIs: cluster bootstrap over persons, 1,000 draws, seed 42.

**Go/no-go rule (fixed now; point estimates decide, CIs reported):**

| Criterion | Threshold |
|---|---|
| G1 tracklets usable (≥ 5 usable faces of 20) | ≥ 30% of the 200 |
| G2 persons with ≥ 1 usable tracklet | ≥ 20 of 54 |
| G3 test persons with ≥ 1 usable check-in photo | ≥ 27 of 54 |

- **GO (high-resolution gallery route):** G1, G2 and G3 all pass. The next step is a pre-registered MEVID open-set + aggregation protocol (check-in gallery; tracklet probes with R1 plain mean; held-out identities as non-enrolled probes). It may also use the train package (all 158 people), since we train nothing.
- **GO (tracklet-only route):** G1 and G2 pass, G3 fails. The gallery is built from tracklets on other dates and cameras. The protocol is written separately.
- **NO-GO:** G1 or G2 fails. Record the numbers and stop. MEVID then appears only as a limitation ("faces too rarely visible or too small"), and SCface remains the held-out candidate.
- The verdict is mechanical. Nobody changes a threshold after seeing the numbers. If Abser wants a different rule, it becomes a new, dated amendment, and both verdicts are reported.

## Step 6: finish
- `.gitignore`: track `outputs/video/mevid_probe/*.csv|json|md|txt`; ignore `overlays/` and all of `Dataset/MEVID/`.
- Check with `git status --ignored`: no image may appear.
- Rerun `summary` to confirm byte-identical outputs.
- Write a dated `CONTEXT.md` §11 entry: verdict, key numbers, any STOP. Mirror it to Drive.
- Commits:
  1. the plan (pre-registration, before step 1);
  2. inventory and sample;
  3. detections and summary.
- `git push -u origin mevid_probe`, then `gh pr create`. If `gh` is not logged in, push only.
- **[UMAR]** post a 5-line summary in the team chat: verdict, G1–G3 values, face-height median, check-in coverage, PR link.
- **Do not:**
  - compute any embedding or identity score;
  - download the train package or the videos;
  - change detector settings;
  - start the MEVID protocol.

## Verification
- Inventory counts versus the README are recorded (match or reported mismatch).
- `sample.csv` is committed before `detections.csv` exists (git history).
- Overlays were checked by Umar.
- The summary rerun is byte-identical.
- No image files are tracked.

## Expected time
About 1 lab day: download 1–2 h (network-dependent), inventory 30 min, detection under 30 min, summary and write-up 1–2 h.
