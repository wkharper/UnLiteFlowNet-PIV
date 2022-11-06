# Unsupervised learning of Particle Image Velocimetry
This repository contains materials for ISC 2020 workshop paper [Unsupervised learning of Particle Image Velocimetry](https://arxiv.org/pdf/2007.14487.pdf).

## Introduction
Particle Image Velocimetry (PIV) is a classical flow estimation problem which is widely considered and utilised, especially as a diagnostic tool in experimental fluid dynamics and the remote sensing of environmental flows. We present here what we believe to be the first work which takes an unsupervised learning based approach to tackle PIV problems. The proposed approach is inspired by classic optical flow methods. Instead of using ground truth data, we make use of photometric loss between two consecutive image frames, consistency loss in bidirectional flow estimates and spatial smoothness loss to construct the total unsupervised loss function. The approach shows significant potential and advantages for fluid flow estimation. Results presented here demonstrate that is outputs competitive results compared with classical PIV methods as well as supervised learning based methods for a broad PIV dataset, and even outperforms these existing approaches in some difficult flow cases.

## Sample results
#### Syethetic data: samples from PIV dataset

- Backstep flow

<p align="center">
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/backstep_Re1000_00386.gif" width="24.5%" height="24.5%" />
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/backstep_385_un.png" width="50%" height="50%"/><br>
</p>

- Surface Quasi Geostrophic (SQG) flow

<p align="center">
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/SQG_01386.gif" width="24.5%" height="24.5%" />
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/SQG_1385_un.png" width="51%" height="51%"/><br>
</p>


#### Real experimental data: particle Images from [PIV challenge](http://www.pivchallenge.org/)

- Jet Flow

<p align="center">
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/results/jet_flow4_s2_arrow.gif" width="85%" height="85%" /><br>
  <em>From left to right: Particle images, UnLiteFlowNet-PIV(trained by full integrated loss) output, PIV-LiteNetFlow output</em>
</p>

## Unsupervised Loss

<p align="center">
  <img src="https://github.com/erizmr/UnLiteFlowNet-PIV/blob/master/images/network.png" width="80%" height="80%"><br>
</p>

## Dataset
The dataset used in this work is obtained from the work below:

- [PIV dataset](https://doi.org/10.1007/s00348-019-2717-2) (9GB)
```
Shengze Cai, Shichao Zhou, Chao Xu, Qi Gao. 
Dense motion estimation of particle images via a convolutional neural network, Exp Fluids, 2019
```
- [JHTDB](http://turbulence.pha.jhu.edu)
```
Y. Li, E. Perlman, M. Wan, Y. Yang, R. Burns, C. Meneveau, R. Burns, S. Chen, A. Szalay & G. Eyink. 
A public turbulence database cluster and applications to study Lagrangian evolution of velocity increments in turbulence. Journal of Turbulence 9, No. 31, 2008.
```
## Build and run docker container
Build the image: `docker build . -t flownet:test`

Allow docker to access XServer: `xhost +local:docker`

Start the container: `sudo docker run --privileged --cpus 8 --gpus all -e DISPLAY=$DISPLAY --net=host -v /tmp/.X11-unix:/tmp/.X11-unix -it --mount type=bind,source="$(pwd)"/,target=/opt/flownet flownet:test /bin/bash`

Ensure you have the most up to date `nvidia-docker2` package and `nvidia-driver-XXX` packages installed on your PC.

## Training
To train from scratch:

1. Download the PIV dataset, remove the current data in the folder ```sample_data``` and extract new data into it.

2. Run the scripts with ```--train``` argument:

    ```python main.py --train```

3. Trained model will be saved in the same folder. (A checkpoint is generated every 5 epochs in default during training)

## Trained model
The trained model ```UnsupervisedLiteFlowNet_pretrained.pt``` is available in the folder ```models```.

## Testing
If using docker, navigate to the working directory: `cd /opt/flownet`

The data samples for test use are in the folder ```sample_data```. To access full datasets see [here](), Untar the contents from the root of the project `tar -xzvhf piv_datasets.tar.gz`

Test and visualize the sample data results with the pretrained model using:

```python main.py --test --flow *name_of_flow* --fps *desired_fps_of_video* --arrow *desired_arrow_density*```

Where `name_of_flow` is the name of the flow folders in the `sample_data` directory.

Note that --arrow should be set between `1` and `256`. Setting lower values results in longer processing times, but greater fidelity of flow visualization.

The current implementation saves the output ground truth (if available) and UnLiteFlowNet-PIV output into the `output` directory. This directory contains an animated gif `movie.gif` that contains the flow field visualization.

It is recommended to clear your workspace every time you run the code by using `./clean.sh`.

## Output file formats
1. `uv_gt_XXXX.txt` consists of the ground truth velocity pixel gradients with each line representing a row of pairs of (u,v) values. Each pair of data is considered 1 column entry.

1. `uv_XXXX.txt` consists of the estimated velocity pixel gradients with each line representing a row of pairs of (u,v) values. Each pair of data is considered 1 column entry.

1. `stats.txt` consists of the output statistics from the dataset including mean, median, and standard error in that order.

1. `movie.gif` conists of the output animation.

1. `frame_XXXX.png` consists of the flow field estimates from each the `XXXX` sequence of images.

## Citation

In BibTeX format:
```
@article{zhang2020unsupervised,
  title={Unsupervised Learning of Particle Image Velocimetry},
  author={Mingrui Zhang and Matthew D. Piggott},
  journal={arXiv preprint arXiv:2007.14487},
  year={2020}
}
```

