## This folder contains papers that explore both counter examples (identified with CE prefix in filename) to Approach B or partial solutions that use diffusion based technologies to achieve any number of stages of our proposed pipeline.
## Gap identified
the field has plenty of (a) jointly-trained restoration-recognition networks<br>(b) restoration challenges scored by identity metrics<br>A handful of (c) CCTV-specific restoration→recognition pipelines.<br><br>What's genuinely underexplored is our specific combination: a modular, quality-gated, adaptive pipeline (skip/run restoration conditionally rather than always restoring) that runs a consistent baseline-vs-frontier ablation at every stage (SR, deblur, low-light, detection, recognition) and evaluates all of it on surveillance-native benchmarks (QMUL-SurvFace/TinyFace) rather than synthetic degradation sets.<br><br>*That combination — adaptive gating + full-pipeline ablation + surveillance-native eval — is the gap*

## Areas of Interest (Not the focus of our pipeline but good to integrate if feasible)
### Pipeline Contender 1
Joint Face Image Restoration and Frontalization for Recognition (Tu et al) is the closest to our pipeline in regards to its functionalities but the work is quite old (2021) and uses outdated techniques. Still it is a baseline for competition.

### Pipeline Contender 2
IDENTITY-PRESERVING FACE RESTORATION FOR LOWRESOLUTION CCTV IMAGES USING GAN AND STYLEGAN2 (Sugeng et al) used heavily augmented data to introduce plausible degradation and on test data, results were good at 93% accuracy for only blurry images but on images that underwent complete degradation, performance dropped to 58%. Hybrid GAN-based architecture

### Pipeline Contender 3
Degradation-Agnostic Statistical Facial Feature Transformation for Blind Face Restoration in Adverse Weather Conditions (Son et al) focuses on blind face image restoration on weather degraded images like rain streaks. Out of the scope of current pipeline but a good area to explore since even diffusion models struggle here.

### Face Frontalization
Explored in Joint Face Image Restoration and Frontalization for Recognition (Tu et al)

### Target Based Recognition
To select specific faces according to input target face. Explored in Robust Face Super-Resolution and Recognition Through Multi-Feature Aggregation in Diffusion Models

### Does face-recognition performance saturate once image quality is sufficient to preserve the identity-discriminative information needed by the recognizer?
Yes, but the saturation behavior is highly architecture-dependent. While the existence of a performance plateau is universal across all systems, where that plateau occurs, how sharply accuracy scales before it, and why the system stabilizes are determined entirely by the model's design.<br><br>Large Visual Transformers using AdaFace scale well with higher quality images whereas smaller older models like a CNN using ArcFace will plateau quickly with greater quality.<br><br>We should answer this question according to our architecture for optimization purposes such that we can stop enhancement at the identified threshold where recognition performance plateaus.
