# Phase 6, Appendix T1: YouTube Faces detector test on a Colab T4 (runbook)

Written by Claude on the lab PC for whoever runs the Colab notebook (Umar or Abser), 2026-11-11. Parent plan: `Phase6_core_pipeline_plan.md`, Appendix T.

## What changed from the plan
The plan assumed the original YouTube Faces full-frame database (`frame_images_DB`). The copy held on the lab PC (`Dataset/YTFaces`, 17 GB) is the Kaggle package "YouTube Faces with Facial Keypoints" instead: 2,194 videos of 828 people, one `.npz` per video with **cropped** frames (about 120 to 300 px on a side), a face box per frame and 68 landmarks. So this test checks the detector on clearly visible faces in compressed, blurry video crops. It says nothing about small faces in a wide scene, which is what ChokePoint and the custom video are for. The box convention was checked on overlays (the boxes sit on the faces; they are tight, eyebrows to chin).

## Does it fit on the Drive?
- The full dataset (17 GB) is not uploaded and does not need to be. The test uses a seeded sample: 500 people, one video each, 10 evenly spaced frames per video = 5,000 PNG frames, **338 MB** as a zip (`ytf_sample.zip`), plus a 0.4 MB code zip.
- The drive mounted as H: reports 155.7 GB free (that figure is what Drive for desktop shows; it was not checked in the web interface). The bundle is far below it. `CONTEXT.md` §8 notes an older figure of 10 GB usable for the shared folder; 338 MB also fits under that.
- The detector weights are not copied again: the notebook uses `phase5/detectors.zip`, which is already on the Drive.

## Files on the Drive (new folder, nothing else was touched)
`FYP_SR_SHARED/phase6_ytf/`: `phase6_ytf_colab.ipynb`, `code_ytf.zip`, `ytf_sample.zip`, `MANIFEST.json`. Results will appear in `phase6_ytf/results/`.

## What you do
1. Wait for Drive to finish syncing the folder (the notebook checks sizes and SHA-256 and stops if a file is incomplete).
2. Open `phase6_ytf_colab.ipynb` from Drive in Colab. The account must see `FYP_SR_SHARED` in My Drive (owner, or *Add shortcut to My Drive*).
3. Runtime, Change runtime type, **T4 GPU**, then **Run all**. The whole job takes a few minutes.
4. When the last cell prints `exit code 0` it writes `results/DONE`. Tell Claude on the lab PC.
5. The notebook stops by itself, and you should not continue, if: the GPU is not a T4; a bundle file is missing or has a wrong checksum; ONNX Runtime is not using the GPU; or the detector weights differ from the lab PC's. Copy the printed message to the team chat. Do not try to fix it.

## What to expect (so a surprising result is not mistaken for a fault)
A CPU smoke test of 30 frames on the lab PC gave: `full640` found the face in all 30 frames; `full1280` found it in none. This is not a bug. `full1280` upscales each crop to 1280 px, so a 90 px face becomes about 550 px, and SCRFD does not detect faces that large (its maximum score was 0.07 to 0.27 at 1280 and about 0.8 at 640 on three test frames). It is the same effect as the lower recall of `full1280` on faces above 64 px in the Phase 5 WIDER FACE table. The notebook reports both modes as found.

## Outputs
`results/per_frame.csv` (one row per frame and mode), `results/summary.csv` (recall at IoU 0.5 and 0.3, centre-hit rate, IoU median, extra detections per frame and median ms per frame, by face-height bin, with cluster-bootstrap CIs over people), `results/device.json`.

## Device note
These numbers come from one Colab T4 and appear in Appendix T of `REPORT_phase6.md` marked as T4; the Phase 6 pipeline numbers come from the lab PC. The two are never pooled.
