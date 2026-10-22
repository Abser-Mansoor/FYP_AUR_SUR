## General Overview

1) we propose a high-order
degradation process to model practical degradations, and
utilize sinc filters to model common ringing and overshoot
artifacts.
2) We employ several essential modifications
(e.g., U-Net discriminator with spectral normalization) to
increase discriminator capability and stabilize the training
dynamics.
3) Real-ESRGAN trained with pure synthetic
data is able to restore most real-world images and achieve
better visual performance than previous works, making it
more practical in real-world applications.
4) Classical degradation model is widely adopted in blind SR methods.
Yet, real-world degradations are usually too complex to be explicitly modeled.
Thus, implicit modeling attempts to learn a degradation generation process within networks.
In this work, we propose a flexible high-order degradation model to synthesize more practical degradations.

## Synthetic Data Creation

1) ### Classical Degradation Model:
    The classical degradation model is usually
    adopted to synthesize the low-resolution input. Generally,
    the ground-truth image y is first convolved with blur kernel k. Then, a downsampling operation with scale factor r
    is performed. The low-resolution x is obtained by adding
    noise n. Finally, JPEG compression is also adopted, as it is
    widely-used in real-world images.
    <img width="195" height="26" alt="image" src="https://github.com/user-attachments/assets/1894f0f6-5826-44ce-8084-e26cd288e1e9" />
    where D denotes the degradation process.

2) ### High-Order Degradation Model:
    An n-order model
    involves n repeated degradation processes
    where each degradation process adopts the classical degradation model with the same procedure but
    different hyper-parameters. Note that the “high-order” here
    is different from that used in mathematical functions. It
    mainly refers to the implementation time of the same operation. The random shuffling strategy in may also include
    repeated degradation processes (e.g., double blur or JPEG).
    But we highlight that the high-order degradation process is
    the key, indicating that not all the shuffled degradations are
    necessary. In order to keep the image resolution in a reasonable range, the downsampling operation in Eq. 1 is replaced
    with a random resize operation. Empirically, we adopt a
    second-order degradation process, as it could resolve most
    real cases while keeping simplicity. <img width="252" height="32" alt="image" src="https://github.com/user-attachments/assets/e4a13828-2f87-45cf-8696-6fc1522ed97f" /> It is worth noting that the improved high-order degradation
    process is not perfect and could not cover the whole degradation space in the real world. Instead, it merely extends the
    solvable degradation boundary of previous blind SR methods through modifying the data synthesis process. Several
    typical limitation scenarios can be found in Fig. 11.

3) ### Ringing and Overshoot Artifacts:
    These artifacts are very common and usually produced by a sharping algorithm, JPEG compression, etc. We employ the sinc filter, an idealized filter that cuts
    off high frequencies, to synthesize ringing and overshoot
    artifacts for training pairs. The sinc filter kernel can be
    expressed as <img width="253" height="48" alt="image" src="https://github.com/user-attachments/assets/b5ea9b18-2bb2-4655-8d15-2a7f77a39c69" /> where (i, j) is the kernel coordinate; ωc is the cutoff frequency;
    and J1 is the first order Bessel function of the first
    kind. We adopt sinc filters in two places: the blurring process and the last step of the synthesis. The order of the last
    sinc filter and JPEG compression is randomly exchanged
    to cover a larger degradation space, as some images may be
    first over-sharpened (with overshoot artifacts) and then have
    JPEG compression; while some images may first do JPEG
    compression followed by sharpening operation.

## Training:

  ESRGAN generator. We adopt the same generator (SR
    network) as ESRGAN i.e., a deep network with several residual-in-residual dense blocks (RRDB). We also extend the original ×4 ESRGAN architecture to perform super-resolution with a scale factor of
    ×2 and ×1. As ESRGAN is a heavy network, we first
    employ the pixel-unshuffle (an inverse operation of pixelshuffle) to reduce the spatial size and enlarge the channel size before feeding inputs into the main ESRGAN architecture. Thus, the most calculation is performed in a smaller
    resolution space, which can reduce the GPU memory and
    computational resources consumption.
    U-Net discriminator with spectral normalization (SN).
    As Real-ESRGAN aims to address a much larger degradation space than ESRGAN, the original design of discriminator in ESRGAN is no longer suitable. Specifically, the
    discriminator in Real-ESRGAN requires a greater discriminative power for complex training outputs. Instead of discriminating global styles, it also needs to produce accurate
    gradient feedback for local textures. We also improve the VGG-style discriminator in ESRGAN
    to an U-Net design with skip connections (Fig. 6). The UNet outputs realness values for each pixel, and can provide
    detailed per-pixel feedback to the generator.
    In the meanwhile, the U-Net structure and complicate
    degradations also increase the training instability. We employ the spectral normalization regularization to stabilize the training dynamics. Moreover, we observe that spectral normalization is also beneficial to alleviate the over-
    sharp and annoying artifacts introduced by GAN training.
    With those adjustments, we are able to easily train the RealESRGAN and achieve a good balance of local detail enhancement and artifact suppression.
    The training process is divided into two stages. First, we
    train a PSNR-oriented model with the L1 loss. The obtained
    model is named by Real-ESRNet. We then use the trained
    PSNR-oriented model as an initialization of the generator,
    and train the Real-ESRGAN with a combination of L1 loss,
    perceptual loss and GAN loss.  

## Experiments:

  Training details. Similar to ESRGAN, we adopt
    DIV2K Flickr2K and OutdoorSceneTraining
    datasets for training. The training HR patch size is set
    to 256. We train our models with four NVIDIA V100
    GPUs with a total batch size of 48. We employ Adam
    optimizer. Real-ESRNet is finetuned from ESRGAN for faster convergence. We train Real-ESRNet for
    1000K iterations with learning rate 2 × 10−4 while training Real-ESRGAN for 400K iterations with learning rate
    1 × 10−4
    . We adopt exponential moving average (EMA)
    for more stable training and better performance. RealESRGAN is trained with a combination of L1 loss, perceptual loss and GAN loss, with weights {1, 1, 0.1}, respectively. We use the {conv1, ...conv5} feature maps
    (with weights {0.1, 0.1, 1, 1, 1}) before activation in the
    pre-trained VGG19 network as the perceptual loss. Our
    implementation is based on the BasicSR.
    Degradation details. We employ a second-order degradation model for a good balance of simplicity and effectiveness. Unless otherwise specified, the two degradation
    processes have the same settings. We adopt Gaussian kernels, generalized Gaussian kernels and plateau-shaped kernels, with a probability of {0.7, 0.15, 0.15}. The blur kernel size is randomly selected from {7, 9, ...21}.
    Blur standard deviation σ is sampled from [0.2, 3] ([0.2, 1.5] for the
    second degradation process). Shape parameter β is sampled from [0.5, 4] and [1, 2] for generalized Gaussian and
    plateau-shaped kernels, respectively. We also use sinc kernel with a probability of 0.1. We skip the second blur degradation with a probability of 0.2.
    We employ Gaussian noises and Poisson noises with a
    probability of {0.5, 0.5}. The noise sigma range and Poisson noise scale are set to [1, 30] and [0.05, 3], respectively
    ([1, 25] and [0.05, 2.5] for the second degradation process).
    The gray noise probability is set to 0.4. JPEG compression
    quality factor is set to [30, 95]. The final sinc filter is applied with a probability of 0.8. More details can be found in
    the released codes.
    Training pair pool. In order to improve the training efficiency, all degradation processes are implemented in PyTorch with CUDA acceleration, so that we are able to synthesize training pairs on the fly. However, batch processing
    limits the diversity of synthetic degradations in a batch. For
    example, samples in a batch could not have different resize
    scaling factors. Therefore, we employ a training pair pool
    to increase the degradation diversity in a batch. At each iteration, the training samples are randomly selected from the
    training pair poor to form a training batch. We set the pool
    size to 180 in our implementation.
    Sharpen ground-truth images during training. We further show a training trick to visually improve the sharpness,
    while not introducing visible artifacts. A typical way of
    sharpening images is to employ a post-process algorithm,
    such as unsharp masking (USM). However, this algorithm
    tends to introduce overshoot artifacts. We empirically find
    that sharpening ground-truth images during training could
    achieve a better balance of sharpness and overshoot artifact suppression. We denote the model trained with sharped
    ground-truth images as Real-ESRGAN+
    
