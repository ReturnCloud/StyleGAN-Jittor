## Introduction
This project is StyleGAN implemented with Jittor. The code is based on  [style-based-gan-pytorch](https://github.com/rosinality/style-based-gan-pytorch).
Jittor is a high-performance deep learning framework based on Just-In-Time(JIT) compiling and meta-operators. It contains a wealth of high-performance model libraries, including: image recognition, detection, segmentation, generation, differentiable rendering, geometric learning, reinforcement learning, etc. . More information can be found in [Jittor](https://github.com/Jittor/jittor).
StyleGAN is a deep generative network model proposed by NVIDIA in 2018 for generating high-definition, style-controlled images. Details about the model architecture and experiments can be found in [StyleGAN](http://arxiv.org/abs/1812.04948).

## Usage

### Installation
This project can be directly copied by `git clone` using
`git clone https://github.com/ReturnCloud/StyleGAN-Jittor`
Experiments are conducted under python 3.7.11 and the environment can be created by
`conda env create -f environment.yaml`

### Dataset
Experiments in this project is based on [Unicode character dataset](https://github.com/rmunro/unicode_image_generator). The code for data preparation is included in `unicode`. Run `python unicode/create_unicode_stratified.py`
This will create the images in a `unicode/unicode_jpgs` directory, using the font file in the `unicode` directory. To change the image sizes, the font, the number of images per block, or other parameters, you can edit `create_unicode_stratified.py`

Set the minimum size(min_size) and the path to image files. Run `python resize.py` and images will be reduced to the size with exponential of 2 until min_size. These images generated will be used in the training process later.

### Training
`train.py` is a script for model training and `model.py` defines the architecture for generator and discriminator of the model. Details can be seen in the StyleGAN paper.
Most training parameters are listed below. You can set them according to your needs in the command line when running `python train.py`
`phase:`number of samples used in different phases during training
`lr:`learning rate
`sched:`whether to use learning rate sheduling
`init_size:`initial size of images
`max_size:`maximum size of images
`max_iter:`maximum number of iteration
`mixing:`whether to use mixing for regulization
`loss:`loss for training, choose between wgan-gp and r1
`batch_size:`batch size
`code_size:`size of code
`n_critic:`frequency of discriminator update for each update of generator
`n_mlp:`number of layers for mlp transforming z into w

### Viewing
Experiment results can be viewed by running `python generate.py`. This command will create `sample.png`, which gives the images sampled and `sample_mixing_i.png`, which shows the mixing results.

## Results
Using the default parameters in the code, we can get results below, which is generated by model trained after 10000, 20000, 30000 and 40000 epochs.

<img src=./img/1.png width="48%"> <img src=./img/2.png width="48%">
<img src=./img/3.png width="48%"> <img src=./img/4.png width="48%">
We can roughly see the process of style extraction and fusion from above results.

## References
1. https://github.com/rosinality/style-based-gan-pytorch
2. https://github.com/Jittor/jittor
3. https://github.com/rmunro/unicode_image_generator
4. http://arxiv.org/abs/1812.04948
5. https://towardsdatascience.com/creating-new-scripts-with-stylegan-c16473a50fd0

## TODO
1. Check for the code correctness on `wgan-gp` loss
2. Test on more datasets.
