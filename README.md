## GAN vs VAE: Custom Dataset Image Generation
This repository contains a project where Generative Adversarial Networks (GAN) and Variational Autoencoders (VAE) are trained on a custom dataset to generate images. The results from the two models are then compared to evaluate their performance in terms of image generation quality.

## Overview
Objective:
- Train GAN and VAE models on a custom image dataset.
- Generate new images using both models.
- Compare the generated images to analyze the differences and strengths of each model.

Models Implemented:
- GAN (Generative Adversarial Network): A model composed of a Generator and a Discriminator, where the generator tries to create realistic images and the discriminator attempts to distinguish between real and generated images.
- VAE (Variational Autoencoder): A type of autoencoder that learns to generate images by mapping input data into a latent space and reconstructing from that space.

## Results
|Criteria|GAN|VAE|
|---|---|---|
|Image Sharpness|High sharpness, realistic details|Lower sharpness, often blurry|
|Training Stability|Unstable, requires careful tuning|Stable and predictable|
|Latent Space|Less structured, less interpretable|Well-organized, smooth interpolation|
|Image Diversity|Risk of mode collapse, limited diversity|High diversity, good variation|
|Training Time|Longer, due to unstable convergence|Faster and stable|
