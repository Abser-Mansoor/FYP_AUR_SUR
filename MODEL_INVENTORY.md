# MODEL_INVENTORY

Last updated 2026-10-15 (Phase 3 complete; see `REPORT_phase3.md`). One venv per model under `envs/`; weights under `models/weights/` (SR models) and `models/<repo>/preset/weights` (TADSR). All SHA-256 values are 16-character prefixes of the file hash.

## 1. Face recognizers (Phase 0/1; unchanged)
| Model | Backbone | Weights file | Training data | Pre-processing (Phase 1) | Notes |
|---|---|---|---|---|---|
| ArcFace | iresnet100 | `arcface_iresnet100.pth` (sha a566a62357f0) | **unknown** (MS1MV3 or Glint360K) | RGB, [-1,1] | confound; **leaky for OSDFace** (section 2.3) |
| AdaFace | IR-101 | `adaface_ir101_ms1mv2.ckpt` (4a26839460d8) | MS1MV2 | BGR, [-1,1] | |
| MagFace | iresnet100 | `magface_iresnet100_MS1MV2.pth` (cfeba792dada) | MS1MV2 | BGR, [0,1] | |
| CurricularFace | IR-101 | `CurricularFace_Backbone.pth` (031ea0f7ba5f) | not stated by repo | RGB, [-1,1] | |
| SCRFD det_10g | detector | `models/detector/det_10g.onnx` | - | - | face alignment (diagnostic arms only) |

## 2. SR / face-restoration arms (Phase 3)
### 2.1 Summary
Technical go / no-go means: official weights exist and the model runs natively. The recognition verdict is a judgement from `REPORT_phase3.md` (evaluation subset of 22,898 crops, all four recognizers).

| Arm | Technical go / no-go | Selected setting | Isolated speed, PC (s/crop median) | Peak VRAM | Recognition verdict |
|---|---|---|---|---|---|
| Direct feed (resize only) | baseline | - | - | - | reference |
| Bicubic | go | pre-512 | about 0.014 (mean, mostly file I/O) | 0 | equivalent to direct feed |
| Real-ESRGAN x4plus | go | native input | 0.028 | 0.1 GB | no benefit; harms ArcFace, MagFace and open-set |
| InvSR | go | pre-64, 1 step (rule-selected); pre-128 checked | 0.076 (pre-64) / 0.269 (pre-128) | 2.8 / 3.2 GB | harmful at both settings |
| TADSR | go | pre-128, timestep 250 | 0.363 | 4.1 GB | marginal: TinyFace gain for 2 of 4 recognizers only |
| OSDFace | go, ArcFace leaky | pre-128 | 0.515 | 4.0 GB (fp32) | best TinyFace accuracy only with identical restoration of probe and gallery; invents identities |
| CodeFormer / GFPGAN | not attempted (optional) | - | - | - | - |
Not attempted by instruction: DreamSR, FluxSR, ID-CDM, FideDiff. All models ran natively on Windows (no WSL2); OSDFace and InvSR pre-128 evaluation also ran on Linux Colab T4s.

Speeds on a Colab T4 (75-crop cross-device check): OSDFace 1.39 s/crop, InvSR pre-128 0.51 s/crop. Smoke test (75 seeded crops): 0 failures in every setting; montages in `outputs/sr/smoke/`.

### 2.1b Remote (Colab) execution
- Environment per job: Python 3.10 uv venv, torch 2.6.0+cu124 and the PC venv's pinned packages (`bundle/locks/*.txt`), repos at the commits in 2.2, weights from the same official sources with SHA-256 checks against the PC's files.
- Each Colab ran a cross-device check before its job: direct-feed embeddings matched the PC to about 1e-6 (max |delta cos|), restored smoke crops matched to pixel difference 0.04-0.05 of 255 and embedding cosine 0.9999; all passed.
- Jobs ran in 1,000-crop chunks, embeddings were computed on the Colab, and only embeddings, thumbnails and timings were returned (merge: `scripts/sr_colab_merge.py`, which rejects any gap, overlap or wrong order). Full-size outputs for those jobs were not copied.
- Devices per reported result: OSDFace (tuning, evaluation, seeds): Colab T4 only. InvSR pre-128 evaluation: Colab T4. All other arms: lab PC.

### 2.2 Repos, versions and weights
| Model | Repo / commit / date | Environment | Weights (source, size, sha256 prefix) |
|---|---|---|---|
| InvSR | https://github.com/zsyOAOA/InvSR @ `609788cd076f1990e38d3912092ca67286e1e0bc` (2026-02-08) | `envs/invsr`, Python 3.10, torch 2.6.0+cu124, xformers 0.0.29.post2 (repo pins torch 2.4.0 / xformers 0.0.27.post2, see 2.3) | noise predictor `noise_predictor_sd_turbo_v5.pth`, Hugging Face OAOA/InvSR, 129 MB, e1a03b0f447d6f26. Base: stabilityai/sd-turbo (Hugging Face), UNet 3.3 GB 438da6db739c8651 |
| TADSR | https://github.com/zty557/TADSR @ `b0eaa30d6658a8e61cae93811f2fa6e27ebc9ddd` (2026-03-02) | `envs/tadsr`, Python 3.10, torch 2.6.0+cu124 (repo pins torch 2.0.1 / xformers 0.0.20; xformers not installed, optional in the repo) | everything from the official Hugging Face repo zty557/TADSR into `models/TADSR/preset/weights`: `tadsr.pkl` 25 MB a04263729a7d276f, `DAPE.pth` a7028be2edcbe9ab, `ram_swin_large_14m.pth` 5.4 GB, UNet 3.3 GB 6dfae3e5f7d459b5, VAE, time-VAE, text encoder, BERT |
| OSDFace | https://github.com/jkwang28/OSDFace @ `065222742848c7dda06e5a6e99f47e4ceace18d6` (2026-07-23) | `envs/osdface`, Python 3.10, torch 2.6.0+cu124, xformers 0.0.29.post2, pytorch-lightning 1.0.8 (needed only to unpickle `associate_2.ckpt`) | **official Google Drive folder** (README link): `associate_2.ckpt` 1.79 GB d51ff7f0ddb0bd84, `embedding_change_weights.pth` 3b13ab83fed678ed, `pytorch_lora_weights.safetensors` bf7095b4d79f3e99. Base model: Stable Diffusion 2.1 base (see 2.4) |
| Real-ESRGAN x4plus | https://github.com/xinntao/Real-ESRGAN @ `a4abfb2979a7bbff3f69f58f58ae324608821e27` (2024-04-03) | `envs/realesrgan`, Python 3.10, torch 2.6.0+cu124 | official release v0.1.0 `RealESRGAN_x4plus.pth` 67 MB 4fa0d38905f75ac0; loads strictly into the vendored RRDBNet (all keys matched) |

### 2.3 Deviations, patches and workarounds (all logged, none edit the cloned repos)
| Item | Detail |
|---|---|
| torch 2.4.0 does not import on this machine | `fbgemm.dll` needs `libomp140.x86_64.dll`, which is not on the machine and not in Microsoft's VC++ 14.44 redistributable (checked by unpacking it). torch 2.6.0+cu124 imports fine, so InvSR and OSDFace use torch 2.6.0 and xformers 0.0.29.post2 instead of the pinned 2.4.0 / 0.0.27.post2. Result equivalence to the authors' environment is not verified. |
| TADSR pins | torch 2.0.1, xformers 0.0.20 and numpy 1.24.3 are not usable together on Windows; torch 2.6.0 is used, numpy/transformers/peft/huggingface_hub keep the repo pins, xformers is not installed. |
| TADSR `./weights/bert-base-uncased` | the RAM tagger loads BERT from a repo-relative path while the download puts it in `preset/weights`; the folder was copied (no symlink) to `models/TADSR/weights/bert-base-uncased`. |
| TADSR upstream bug | for outputs above about 768 px (pre-size 256, 1024-px output) the tiled-latent branch reads `self.timesteps`, which the repo never sets, and every crop fails with `AttributeError`. The runner sets `model.timesteps` before each call (`src/sr/run_tadsr.py`). Without it: 0 of 75 crops ran at pre-size 256. |
| InvSR reflect padding | the repo reflect-pads any input smaller than its chopping size; the runner sets `chopping_size` equal to the pre-size so each crop is one unpadded pass. |
| InvSR Linux cache path | the config's `cache_dir` is a Linux path; weights are pre-downloaded and passed with `--sd_path` / `--started_ckpt_path`. |
| OSDFace SD base | `stabilityai/stable-diffusion-2-1-base` on Hugging Face returns 401 (gated/removed); the authors' README points to ModelScope (section 2.4). Their `test.sh` uses an `hf-mirror.com` endpoint, which is not used here. |
| Real-ESRGAN `basicsr` | `basicsr` does not import on current torchvision; only the RRDBNet network definition and three helper functions were vendored (`src/sr/vendor/rrdbnet.py`, originals kept as `*_orig.py`). |
| Windows | no triton or xformers is needed at inference by any of the repos (guarded imports); DataLoader `num_workers` is not used; no symlinks. |
| Per-crop seeding | each runner re-seeds torch/numpy per crop, so a crop's output depends only on (seed, crop), not on processing order. |

### 2.4 OSDFace base-model provenance
- Official base: Stable Diffusion 2.1 base. Source used for the authoritative copy: **ModelScope `stabilityai/stable-diffusion-2-1-base`** (the authors' README recommendation) into `models/weights/osdface/sd21base`.
- Smoke tests ran from a Hugging Face community copy (`sd2-community/stable-diffusion-2-1-base`, `sd21base_hf`) because ModelScope was throttled. Hash comparison against the ModelScope copy: text encoder cce6febb0b6d876e **identical**, VAE a1d993488569e928 **identical**, UNet 6dfae3e5f7d459b5 **identical** (ModelScope download finished 2026-10-08). The two copies are bit-identical, so the smoke results stand and the evaluation will use the ModelScope copy (`--sd-path models/weights/osdface/sd21base`). The same UNet hash appears in TADSR's `unet/` folder, so TADSR is built on the same SD 2.1 base.

### 2.5 OSDFace identity leakage
- OSDFace's training uses a facial identity loss `L_ID = 1 - cos(F(I_H), F(I^H))` computed with **a pretrained ArcFace model** (OSDFace paper, arXiv 2411.17163, HTML version; the paper reports the same ArcFace as its "Deg." identity metric). Backbone and training set are not stated; the training code is not released (README TODO), so this cannot be checked further.
- **ArcFace is therefore the leaky recognizer for OSDFace and is flagged in every OSDFace table; conclusions rest on AdaFace, MagFace and CurricularFace.** Our ArcFace (iresnet100, unknown training set) may not be the same network, but it is flagged regardless because it is the same model family.
- Not checked for the other arms: I have not read the InvSR or TADSR papers or training code. Their READMEs mention no recognizer-derived loss (InvSR's README lists a fine-tuned LPIPS model and a discriminator for training), but that is not proof; treat other leakage as unverified.

## 3. Datasets and subsets (Phase 3)
- Subset config: `configs/sr.yaml`; builder `scripts/sr_build_subset.py`; outputs `outputs/sr/subset/{tuning,eval}_{images,pairs}.csv`. Identity-stratified (strata: tertiles of gallery images per identity), seed 42, tuning and eval share no image or identity.
- Tuning: 2,800 images (TinyFace 50 identities + 500 distractors; SurvFace 25 identities + 500 unmated queries + 150 positive and 150 negative pairs).
- Eval: 22,898 images (TinyFace 800 identities: 1,155 probes, 1,373 gallery images, + 4,000 distractors; SurvFace 200 identities: 4,486 gallery, 4,476 mated queries, + 4,000 unmated; 1,000 positive and 1,000 negative verification pairs, 3,408 images).
