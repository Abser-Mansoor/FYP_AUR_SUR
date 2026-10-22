## Objective
Re evaluate proposed architectures feasibility, frontier status and compatibility with problem statement. Start work on code and (maybe) begin training. 4 systems at hand, RTX 4000 Ada Generation (University provided) and 3 google colab accounts.

## Re evaluation Results
  1. DreamSR and FluxSR have Flux-scale backbone, not expected to fit on 20 GB or Colab without quantization. Additionally, they are not a good fit for very low size cropped images. Choose InvSR as primary and TADSR as optional second arm.
  2. InvSR is built on a Stable-Diffusion-class backbone (so is TADSR) and lets you pick 1 to 5 sampling steps at inference, this allows testing how many steps are needed before identity degrades and could become a supplementary report for QMUL dataset.
  3. https://github.com/zsyOAOA/InvSR, https://github.com/zty557/TADSR. Code and weights available as of 30/9/2026.
  4. Note: Defer this discussion for now. Test if geometric artifacts like those described in Shadows Don’t Lie and Lines Can’t Bend! by Sarkar et al. can be introduced by our restoration engine, thus eroding trust in results.
  5. New restoration arm found in OSDFace (paper uploaded). It provides Identity aware super resolution and deblurring in one model. Test this on QMUL crops. https://github.com/jkwang28/OSDFace.
  6. Caveat to remember, all SR models described expect larger images so we must test, and if possible, fine tune them to 20x16 QMUL crops.
  7. ID-CDM is unreliable and infeasible due to slow inference, lack of available weights or code and analysis on synthetic datasets which have evidence of being misleading when compared to real degradation.
  8. Blocked, Monitor for changes: Use FideDiff (ICLR 2026) as the diffusion deblurrer on the QMUL crops. Note that authors have not uploaded weights or code but it is planned for updation at a later date. As a replacement, use NAFNet or Restormer GoPro checkpoint.
  9. p-Diff does not have any github repository as of 30/9/2026 but there are similar diffusion based low light enhancement technologies that have published codes and weights like LightenDiffusion, QuadPrior (trained on COCO) and Diff-Retinex++. https://github.com/daooshee/QuadPrior/, https://github.com/JianghaiSCU/LightenDiffusion, https://github.com/XunpengYi/Diff-Retinex-Plus.
  10. While QMUL does provide low lighting images, it does not separate them from the rest so we must isolate this dataset by ourselves and test enhancement models.
  11. Compare SR alone against deblur then SR, with a gated variant. If SR alone matches deblur then SR on identity metrics, we will show that the deblurring stage is unnecessary at that resolution.
  12. To truly confirm that our restoration engine works on real CCTV degradation rather than a synthetic approximation, we must validate using datasets that contain paired, non-synthetic, multi-resolution identities. Instead of generating synthetic degradation, we must test our NAFNet -> LightenDiffusion -> InvSR pipeline on datasets like:
  13. IJB-S (IARPA Janus Surveillance): Features genuine low-resolution public surveillance video matched back to high-resolution, multi-ethnic enrollment mugshots.
  14. BRIAR (Biometric Recognition & Identification at Altitude and Range): Contains real-world long-range, atmospheric, and security camera footage matched against close-up ground truth. 

## Missing Items identified by Claude revision

  1. Tiny-crop protocol. SD-family VAEs downsample 8×, so a 20×16 crop can't be processed at native size. Pre-upsampling size is a hyperparameter to record. Landmark alignment will likely fail on such crops, so decide on the resize-only convention up front.
  2. Diffusion is stochastic. Run several seeds and report mean ± std.
  3. Identity metrics. Add embedding drift and impostor-score shift.
  4. Confounding in bins. Mean luminance correlates with skin tone, and sharpness correlates with texture. Prefer paired per-sample comparisons (before versus after restoration on the same crop) over absolute bin accuracy, and report bin sizes.
  5. Runtime and VRAM per model, since the proposal's throughput claims need measured numbers.
  6. Risk register. SFE-DETR's code availability is unverified, and it matters for the detection stage.
  7. Proposal deviations. You are dropping DreamSR, ID-CDM and π-Diff from the proposal. Rationale described above.

## Tasks:

  1. Load InvSR, TADSR and OSDFace and test all on QMUL crops. Create report comparing all three. No need to fine tune yet.
  2. Load NAFNet/Restormer and test on QMUL crops without SR to see if deblur works on small crops.
  3. Stratify real data by measured degradation. Compute mean luminance per QMUL crop, split into dark and normal bins, and compare identity metrics with and without enhancement per bin. Alternatively, Use DARK-FACE dataset.
  4. For blurred data subset, use Laplacian Variance.
  5. Test the following variations in restoration to see if any component produces a negligible or negative effect on quality. direct, SR only, deblur only, deblur→SR, SR→deblur, gated.

## Baselines

  1. (SR) -> direct feed (resize only), bicubic upsampling, Real-ESRGAN
  2. (Deblur) -> direct feed, unsharp mask, Real-ESRGAN (implicit deblurring). NAFNet and Restormer are the learned arms.
  3. (Low Light) -> direct feed, gamma, CLAHE on the L channel, Zero-DCE++

## Investigations into jury suggestions:

  1. If our pipeline shifts from local facial recognition to broader, full-body tracking across a multi-camera surveillance architecture, we can expand to the QMUL-GRID Dataset.
  2. SCface dataset tracks static visible and infrared imagery of subjects at explicitly measured distances and camera angles. It is ideal if our restoration pipeline needs to solve for distance-based blur and variable sensor types.
