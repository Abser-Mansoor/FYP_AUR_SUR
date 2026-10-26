# Phase 5 continuation: merge the detector study, close Phase 5, then the extra SR-asymmetry study

**Written:** 2026-10-27 (by Abser's session).
**For:** Claude Code on the lab PC, working with **Umar**. Abser will not be present.
**When:** Tue 28 Oct (Parts A–D, about 3–4 h). Part E on Wed 29 or Thu 30 Oct. Lab days Tue–Thu are guaranteed; Mon and Fri are sometimes available.

## How to use this file (for Claude)
- Work through the parts in order.
- Steps marked **[UMAR]** are checkpoints. Stop, show Umar what is asked, and wait for his reply before you continue.
- A **STOP** means: do not try to fix it. Write down what happened (command, full error, file state). Tell Umar to post it in the team chat for Abser. Then wait.
- Umar is not expected to know the codebase. Explain each step to him in one or two plain sentences before running it.
- Read these first:
  - `CONTEXT.md`, especially sections 2 (standing rules) and 11 (open issues);
  - `research/Sprint_Notes/Phase5_video_foundations_plan.md` (the approved Phase 5 plan: verification list and report outline);
  - `configs/video.yaml`.

### Standing rules (short version; full text in `CONTEXT.md` §2)
1. Inference only. Nothing is trained or fine-tuned.
2. Every number in a report comes from a script output file. Never type in a number by hand, and never "fix" a number.
3. Failures are results. Log them; never drop them.
4. `Dataset/` is read-only.
5. Config-driven, seed 42.
6. One device per reported number. The detector-study numbers are from the Colab T4 only. Do not re-run the study on the lab PC GPU to "check" it.
7. Weights are SHA-256 checked against `MODEL_INVENTORY.md`.
8. Never delete anything on the shared Drive.

### Lab-PC quirks (learned in Phases 3–4)
- Windows: use `envs\video\Scripts\python.exe` for this phase, and Windows paths in commands.
- Run only one GPU job at a time. The PC has rebooted unexpectedly under heavy GPU load, so keep jobs short and resumable.
- Long `sleep` calls in a tool are blocked. Use short checks or `run_in_background`.
- In bash heredocs a literal `\n` becomes a real newline. Write patch scripts to files with the Write tool instead.
- Background jobs started from a tool call die if the session is interrupted. Check whether their outputs exist before assuming they finished.

---

## Part A: setup (about 10 min)

**A1. Git.**
```
git checkout main
git pull
git checkout -b phase_5
```
- `main` must include the Phase 4 merge (`9ceed1d`) and the 2026-10-27 commit that adds this file and the updated `CONTEXT.md`.
- If `git pull` reports local changes or conflicts: **STOP**.

**A2. [UMAR] Commit identity.**
- Show the output of `git config user.name` and `git config user.email`.
- Ask Umar whether commits should carry his name.
- If yes and the config shows someone else, commit with `git -c user.name="<name>" -c user.email="<email>" commit ...`. Do **not** change the global or repo git config.

**A3. Environment check.**
- Run `envs\video\Scripts\python.exe -c "import torch, onnxruntime, cv2, pandas, faiss; print(torch.__version__, onnxruntime.__version__)"`.
- It should print torch 2.14.x and onnxruntime 1.30.x (`MODEL_INVENTORY.md` §4). If the import fails: **STOP**.

---

## Part B: close 5b (no GPU; about 15 min)

**B1. Extra model hashes.**
- Read `outputs/video/model_downloads.json`. It is written by `scripts/phase5_fetch_models.py`, and it is gitignored, so it exists only on this PC.
- If it is missing, run `envs\video\Scripts\python.exe scripts\phase5_fetch_models.py`. The script is resumable and downloads only.
- For every file listed, recompute the SHA-256 (`certutil -hashfile <file> SHA256`, or Python `hashlib`). It must equal the json.
- Write the hashes into the CAFace and FaceMoE rows of `MODEL_INVENTORY.md` §4, replacing "filled in after download".
- On any mismatch: **STOP**.

**B2. Lab verification checks** (from the Phase 5 plan's Verification list).
- For each script: if its output json already exists, read it and report the result. Otherwise run the script. Each runs in seconds to minutes on the GPU; run them one at a time.

| Script | Output | Pass condition |
|---|---|---|
| `scripts\phase5_verify_scrfd.py` | `outputs/video/verify_scrfd.json` | Wrapper equals `detect_landmarks` (0 px); Phase 1 cache within 0.05 px |
| `scripts\phase5_verify_tiling.py` | `outputs/video/verify_tiling.json` | Exit code 0; exactly one detection per detector and mode |
| `scripts\video_tracker_smoke.py` | `outputs/video/tracker_smoke.json` | Exit code 0; no ID switches for ByteTrack or BoT-SORT |

- If a check fails: **STOP**. Do not change `src/video/detect.py` or the vendored trackers.

---

## Part C: merge the T4 detector study (about 30 min)

**C1. Copy the results from Drive.**
```
robocopy "G:\My Drive\FYP_SR_SHARED\phase5\results" outputs\video\detstudy /E
```
- Drive for Desktop may keep files online-only; robocopy downloads them. Robocopy exit codes 0–7 are success; 8 or higher is a failure (**STOP**).
- After the copy, `outputs\video\detstudy\` must contain `chunks\`, `runtime.json`, `device.json`, `xcheck.json` and `DONE`.

**C2. [UMAR] File counts.** Count the `.npz` files and show Umar the table:

| Folder | Expected |
|---|---|
| `chunks\wider\*\` | 6 folders × 17 = 102 |
| `chunks\chokepoint\*\` | 6 × 5 = 30 |
| `chunks\crops\*\` | 2 × 115 = 230 |

- These counts were read from the Drive on 2026-10-27. If the copy differs: **STOP**.

**C3. Merge.**
```
envs\video\Scripts\python.exe scripts\phase5_merge.py
```
- It reads the chunks plus the local ground truth: `Dataset/WIDER_FACE`, `outputs/video/chokepoint_sample/gt_boxes.csv` and `outputs/sr/subset/eval_images.csv`.
- It writes to `outputs/video/detstudy/results/`: `wider_ap.csv`, `wider_bins.csv`, `chokepoint_bins.csv`, `crops_rate.csv`, `runtime.csv`, `selection.json`, `problems.txt` and `detector_tables.md`.
- If it raises "chunk names do not match the expected items": **STOP**. Never edit, delete or re-order chunks.

**C4. Determinism.**
- Record the SHA-256 of every file in `results/`.
- Run the merge a second time. Every hash must be identical. If not: **STOP**.

**C5. Keep the run metadata with the results.**
- Copy `device.json` and `xcheck.json` from `outputs\video\detstudy\` into `outputs\video\detstudy\results\`, as Phase 4 did with `outputs/enhance/results/xcheck.json`.

**C6. Sanity gates.** If any gate fails: **STOP**, and do not write the report.
1. **WIDER AP:** SCRFD `full640` easy/medium/hard AP in `wider_ap.csv` is within 1 point of the published SCRFD-10GF values, 95.16 / 93.87 / 83.05. A larger gap means the evaluation port or the operating point differs, and the study cannot be used until Abser checks it.
2. **Recall by size:** in `wider_bins.csv`, recall rises (broadly) from the smallest to the largest face-height bin for every configuration.
3. **Errors:** `problems.txt` may list per-item errors. They are results: count them in the report and never drop them. If more than 1% of items in any configuration failed, **STOP**.

**C7. [UMAR] The selected detector.**
- Show Umar `selection.json`: the chosen configuration, its recall on the 16–32 px bins and its median runtime.
- The choice follows the rule fixed in `configs/video.yaml` before the T4 run. **It is final.** Do not re-select by another rule, change thresholds, or re-run anything on another GPU.
- Context: from the T4 `runtime.json`, RetinaFace `full1280` (107.7 ms) and `tiled` (124.4 ms) are over the 100 ms cap. They cannot be selected even if their recall is highest. The report must say so.

---

## Part D: ChokePoint protocol, gitignore, report, PR (about 2–3 h)

**D1. ChokePoint protocol.**
- Fetch the official evaluation protocol: G1/G2 groups, gallery and probe sequences. Sources:
  - the README on Zenodo (record 815657);
  - the arma.sourceforge.net/chokepoint page.
- Record it in `DATASETS.md` under a new heading "ChokePoint protocol". Quote the source and date. Do not define any split of our own.
- Update the ChokePoint row in `DATASETS.md`:
  - status "held (download date from `outputs/chokepoint_download.log`)";
  - licence note: the Zenodo record says CC BY-NC 4.0, and the NICTA page gives non-commercial research terms. Both forbid commercial use.
- If the protocol cannot be found: write that down, and list it as an open decision in the report.

**D2. [UMAR] Overlay spot check.**
- Ask Umar to open 5 images from `outputs\video\gt\overlays\` and say whether the boxes sit on the faces.
- Record his answer in the report (§4). He is checking only; nothing is changed.

**D3. `.gitignore`.** The current rules ignore all of `outputs/video/`. Add, after the `outputs/enhance` lines:
```
!/outputs/video/
/outputs/video/*
!/outputs/video/detstudy/
/outputs/video/detstudy/*
!/outputs/video/detstudy/results/
!/outputs/video/gt/
/outputs/video/gt/*
!/outputs/video/gt/eye_box_rule.json
!/outputs/video/gt/chokepoint_gt_summary.json
!/outputs/video/verify_scrfd.json
!/outputs/video/verify_tiling.json
!/outputs/video/tracker_smoke.json
!/outputs/video/model_downloads.json
```
- Then run `git status --ignored` and `git add -n .`.
- Only the files above may appear as new. Chunks (`.npz`), ChokePoint frames, overlays, `chokepoint_eyes.csv`, the sample folder and the bundle must **not** appear.
- The GitHub repo is public and ChokePoint is non-commercial, so its images never go into git.

**D4. `REPORT_phase5.md`.**
- Follow the 10-section outline in `Phase5_video_foundations_plan.md` ("Deliverable") and the style of `REPORT_phase4.md`: short sentences, one idea each, every number traceable to a file.
- Paste the tables from `outputs/video/detstudy/results/detector_tables.md`. Do not retype them.
- Sources per section:

| Section | Sources |
|---|---|
| 1 Summary and chosen detector | `selection.json` |
| 2 Data inventory, licences, access | `DATASETS.md`; ask Umar which requests were sent |
| 3 Environment, pins, vendored commits | `MODEL_INVENTORY.md` §4, `src/video/vendor/VENDOR.md` |
| 4 ChokePoint GT and eye-to-box rule | `outputs/video/gt/chokepoint_gt_summary.json`, `eye_box_rule.json`, Umar's overlay check |
| 5 Detector-by-size tables | `wider_ap.csv`, `wider_bins.csv`, `chokepoint_bins.csv`, `crops_rate.csv` |
| 6 Runtime | `runtime.csv` (T4, median ms at 1280×720) |
| 7 Cross-device check | `results/xcheck.json`, `results/device.json`, `verify_scrfd.json` |
| 8 SFE-DETR, CAFace, FaceMoE status | `MODEL_INVENTORY.md` §4, `model_downloads.json` |
| 9 Limitations | see the list below |
| 10 Decisions for Phase 6 | see the list below |

- §9 Limitations must state:
  - **Selection rule difference.** The Sprint 2 plan said "highest recall on 16–32 px at ≤ 0.1 false positives per frame". The Phase 5 plan and `configs/video.yaml` use "median 720p runtime ≤ 100 ms on the T4". The config rule was fixed before the run and is the one applied. Also report the false positives per frame of the chosen configuration.
  - **ChokePoint counts differ from the dataset page:** 72 annotation files vs "48 sequences"; 72,289 entries vs 64,204 crops; 29 ids vs 25 + 29. They are reported as found. The P2E_S5 and P2L_S5 clips have no ground truth.
  - The ChokePoint "ground truth" boxes come from a fitted eye-to-box rule, not from hand-drawn boxes.
  - The cross-device check covers SCRFD only. RetinaFace numbers are T4-only, so they are internally consistent.
  - The crop detection rate uses the Phase 1 padded-crop protocol and is not comparable to frame-level recall.
- §10 Decisions for Phase 6:
  - the selected detector and mode;
  - the ChokePoint protocol (or "still missing");
  - YouTube Faces: no registration yet. Fallback to VoxCeleb2 cropped face tracks if there is no access by 8 Nov;
  - CAFace needs its own AdaFace WebFace4M single-frame baseline;
  - FaceMoE needs 120 px preprocessing;
  - SFE-DETR dropped (no code).
- End the report with: **STOP: awaiting supervisor approval (together with Phase 4).**

**D5. `CONTEXT.md`.**
- Hand-edit §3: Phase 5 is "done, awaiting supervisor approval".
- Add a dated §11 entry: the chosen detector, the merge determinism result, and anything that caused a STOP.
- Leave the STATUS block alone: `scripts/update_context.py` only knows Phase 3.
- Copy the file to `G:\My Drive\FYP_SR_SHARED\CONTEXT.md`. This is the mirror rule; overwriting that one file is allowed.

**D6. Commit, push, PR.** Commit in logical steps:
1. `Phase 5: CAFace and FaceMoE hashes, verification results`
2. `Phase 5: detector study merged (T4), results and gitignore`
3. `Phase 5: ChokePoint protocol and DATASETS update`
4. `Phase 5: REPORT_phase5 and CONTEXT update`

Then:
```
git push -u origin phase_5
gh pr create --base main --head phase_5 --title "Phase 5: video foundations and detector study" --body "<summary from REPORT_phase5 §1>"
```
- If `gh` is not logged in, push only, and tell Umar to send Abser the branch name.
- Do **not** merge the PR yourself.

**D7. [UMAR] Team chat summary.** Draft a 5-line message for Umar to post:
1. the chosen detector;
2. WIDER AP sanity result;
3. merge determinism;
4. any STOPs or problems;
5. the PR link.

### Do not (Parts A–D)
- Start Phase 6 code.
- Change `configs/video.yaml`, `src/video/detect.py` or vendored code.
- Delete or move anything on the Drive, apart from overwriting `CONTEXT.md` (D5).
- Run any GPU job other than the B2 checks.

---

## Part E: extra research: why the SR penalty differs between SurvFace and TinyFace

**Start only after the Phase 5 PR is open.** This is a separate study on a separate branch. Expected effort: about 1 lab day (Wed 29 or Thu 30 Oct). CPU only.

### Why
- The supervisor asked us to explain why restoration changes recognition by different amounts on SurvFace and TinyFace.
- A clear explanation makes our claim that "restoration does not help recognition on these crops" more credible.
- It may also show the claim needs a narrower scope. Either outcome is reported as found.

### Hard constraints
- **No new restoration and no new inference.** Use only cached outputs from Phase 3:
  - embeddings: `outputs/sr/emb/eval/<variant>/eval__<recognizer>.npy`. Rows are aligned with `outputs/sr/subset/eval_images.csv`; check the alignment exactly as `src/sr/eval.py:get_embeddings` does;
  - per-crop measures (in git): `outputs/stratify/bins/*.csv`. Columns `h`, `w`, `min_side`, `luminance`, `lap_var`, `fft_hf`, `magface_norm`, `size_q`, `sharp_bin_sz`, `lum_bin10`;
  - pairs: `outputs/sr/subset/eval_pairs.csv`.
- If a variant's cache is missing on this PC, record it in the report and skip that variant. **Never** run a restorer to fill a gap.
- OSDFace embeddings were computed on Colab T4s and merged into the cache. Use them as they are, and state the device in the report.
- Do not change any Phase 3 file, table or number. All outputs are new files.
- Recognizers: AdaFace, MagFace, CurricularFace (non-leaky). ArcFace may appear in an appendix marked † for OSDFace.
- Variants: bicubic `pre512`, Real-ESRGAN `native`, InvSR `pre64_st1` and `pre128_st1`, TADSR `pre128_t250`, OSDFace `pre128`. Two modes each: both sides restored, and probes only.
- Code template: `scripts/sr_extra_analysis.py` (cache loading, probes-only mixing, paired bootstrap). Reuse `src/eval_protocols.py` and `src/metrics.py`; do not duplicate them.

### What is already known (the script must recompute every value; never copy these into the report)
- Phase 3 reports TinyFace changes as **paired** Rank-1 deltas with CIs. SurvFace changes appear only as **unpaired** AUC and TPIR comparisons (`REPORT_phase3.md` §3). The two datasets are not yet compared on a common scale.
- Eval-subset medians from `outputs/stratify/bins` joined with `eval_images.csv` (computed 2026-10-27):

| Split | n | Size h×w (px) | `lap_var` | Luminance | MagFace norm |
|---|---|---|---|---|---|
| TinyFace probe | 1,155 | 32×32 | 9.30 | 111.9 | 22.13 |
| TinyFace gallery match | 1,373 | 32×32 | 9.41 | 112.6 | (not measured) |
| TinyFace distractor | 4,000 | 17×14 | 2.79 | 93.0 | (not measured) |
| SurvFace gallery | 4,414 | 26×22 | 2.69 | 86.4 | 21.47 |
| SurvFace mated probe | 4,411 | 26×21 | 2.57 | 94.2 | 21.30 |
| SurvFace unmated probe | 3,943 | 27×22 | 2.98 | 92.0 | 21.58 |
| SurvFace verification | 3,408 | 25×19 | 3.60 | 86.1 | 21.96 |

  - In short: TinyFace's mated images are larger, sharper and brighter than SurvFace's, and its distractors are smaller and blurrier.
  - The n values are crops that have Phase 2 measures. Recompute them in the script.

### Step 0: pre-register (before any number is computed)
- Create branch `extra/sr_asymmetry` from `main`.
- Create `REPORT_extra_sr_asymmetry.md` containing only the hypotheses, tests and decision criteria below, copied verbatim.
- Commit it with the message `Extra: SR asymmetry, pre-registered analysis`. Only then write the script.
- **[UMAR]** confirm that the commit exists before you continue.

### Analyses (all bootstraps: 1,000 draws, seed 42, paired on the same items)

**E0. Common scale (do this first).** For every variant × recognizer × mode, compute the same metrics on both datasets:
- **Closed-set Rank-1:**
  - TinyFace: probes against gallery plus distractors (the Phase 3 protocol);
  - SurvFace: mated queries against the 200-identity gallery (new for SurvFace).
- **Verification AUC and d′:**
  - SurvFace: the existing 1,000 positive and 1,000 negative pairs;
  - TinyFace: 1,000 genuine and 1,000 impostor probe–gallery pairs, seeded. Save the pair list to `outputs/sr/results/asymmetry/tinyface_pairs.csv` before scoring.
- For each metric, report:
  - the paired change (restored − direct) with its 95% CI;
  - the relative change (Δ / direct).
- Output: `asym_common_scale.csv` and an asymmetry table in the report.
- **Decision:** if, on the matched metrics, the dataset difference in Δ falls inside the CIs for most variants, the asymmetry was a protocol artefact (**H1 supported**). Report that, run H4 for completeness, and stop. Otherwise continue with H2–H6.

**H2. Input information (crop size, sharpness, brightness).**
- Within each dataset, relate the per-crop change in genuine similarity (restored vs direct, mated pairs) to native `min_side`, `lap_var` and luminance. Use binned means with CIs; no fitted causal model.
- Then compare the datasets on **matched strata**: coarsened exact matching on `size_q` × `sharp_bin_sz`, using only strata present in both datasets.
- **Supported if** the dataset gap in Δ genuine similarity shrinks by at least 50% on matched strata.

**H3. Distractor composition (TinyFace only).**
- Recompute both-sides TinyFace Rank-1 with the 4,000 distractors kept at direct feed, and only probes and gallery matches restored.
- Split the impostor-score shift into gallery-match impostors and distractor impostors.
- **Supported if** keeping distractors at direct feed changes the TinyFace Rank-1 Δ by more than the half-width of its CI.

**H4. Genuine vs impostor decomposition.**
- Per dataset, variant and recognizer: Δ mean genuine score, Δ mean impostor score, Δ d′.
- This shows whether restoration makes faces more alike ("homogenises"): both score distributions rise, and separation falls.
- **Supported** (as the mechanism) if the arm with the larger penalty on a dataset shows a larger Δ impostor relative to Δ genuine there.

**H5. Camera domain (SurvFace).**
- SurvFace file names carry the camera (`<id>_cam<k>_<n>.jpg`).
- Split mated query–gallery comparisons into same-camera and cross-camera. Compare their Δ mated score.
- Also compute camera-ID predictability from the embeddings: leave-one-out kNN (k = 5) camera accuracy, direct vs restored, on the SurvFace gallery.
- **Supported if** restoration raises camera predictability, or hurts cross-camera pairs more than same-camera pairs (CI excludes zero).

**H6. Open-set threshold (SurvFace).**
- Split the TPIR@FPIR = 0.3 change into two parts:
  1. **Threshold shift:** the threshold moves because unmated scores change. Compute TPIR with the direct-feed threshold applied to restored mated scores.
  2. **Mated-score change:** the remainder.
- Report both parts. **Supported** (as the main driver) if the threshold-shift part is larger than the mated part.

**H7. Quality inflation (deferred; not run in this pass).**
- The question: do restored crops gain MagFace norm, so they look "higher quality", on the dataset with the larger penalty, without a matching gain in genuine similarity?
- The cached embeddings cannot answer it. They are L2-normalised (`src/recognizer.py`, `get_embedding`), so the raw norm is lost.
- Answering it would need new MagFace inference on the restored PNGs. That breaks this study's "no new inference" constraint. Also, OSDFace restorations are not stored at full size on the PC.
- Write H7 in the pre-registration as "deferred: needs re-embedding; decision for Abser". Do not run it.

**Multiplicity.** Show Bonferroni-adjusted CIs over the hypothesis tests next to the plain ones (as `REPORT_phase4.md` does). Verdicts use the plain CIs; the report mentions any verdict that changes under Bonferroni.

### Deliverables
- `scripts/extra_sr_asymmetry.py`: CLI `--config configs/sr.yaml`; CPU only; seed 42. Run it twice: the outputs must be byte-identical.
- `outputs/sr/results/asymmetry/*.csv`. They are tracked: `outputs/sr/results/` is already whitelisted in `.gitignore`.
- `REPORT_extra_sr_asymmetry.md`, with these sections:
  1. Summary (3–5 bullets).
  2. The pre-registered plan (unchanged from Step 0).
  3. Common-scale table (E0).
  4. One subsection per hypothesis, each with the verdict "supported", "not supported" or "inconclusive", and the numbers.
  5. What this means for the SR closure claim (one paragraph). Example form: "the larger penalty on SurvFace is explained by X, which the restorers cannot fix because Y".
  6. Limitations:
     - the subset sizes and the unequal identity counts (800 TinyFace vs 200 SurvFace);
     - strata are observational, not causal;
     - SurvFace closed-set Rank-1 is not an official SurvFace protocol;
     - OSDFace device (T4) vs the other arms (lab PC); paired deltas never mix devices.
- Commit, `git push -u origin extra/sr_asymmetry`, open a PR to `main`. **STOP** for supervisor review.
- **[UMAR]** post the summary bullets in the team chat.

---

## If something goes wrong
- **PC reboot mid-step:** rerun the step. The merge and the asymmetry script only read caches and rewrite their own outputs.
- **Drive not synced / robocopy fails:** check that Drive for Desktop is running and signed in. Wait, then rerun C1. Never copy partial folders by hand.
- **Unsure whether something breaks a standing rule:** treat it as a STOP.
