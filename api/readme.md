# MIMAMO-Net api

This repository contain all scripts that needed for running MIMAMO Net for video files. MIMAMO Net is a model designed for temperoal emotion recognition, i.e., valence and arousal, where valence describes how positive or negative the person is, and arousal describes how active or calm the person is. Using this MIMAMO-api, you are able to get the valence and arousal predictions on each frame of an input video where the human faces are available.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

- python3
My recommendation is to use Anaconda to create an virtual environment. For example:
```
conda create --name myenv --python=3.6
```
And then activate the virtual environment by:
```
conda activate myenv
```
- matplotlib3.0.3
```
pip install matplotlib==3.0.3
```
- tqdm
```
pip install tqdm
```
- pandas
```
pip install pandas
```
- ffmpeg
```
sudo apt-get install ffmpeg 
```
- OpenFace

To install OpenFace in the root directory of this project:
```
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
```
Then download the needed models by:
```
bash download_models.sh
```
It's better to install some dependencies required by OpenCV before you run 'install.sh':
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config # Install developer tools used to compile OpenCV
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev #  Install libraries and packages used to read various image formats from disk
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev # Install a few libraries used to read video formats from disk
```
And then install OpenFace by:
```
bash install.sh
```
- pytorch-benchmarks
Install pytorch-benchmarks in the root directory of your project:
```
git clone https://github.com/albanie/pytorch-benchmarks.git
```
Create a directory in pytorch-benchmarks to store the resnet50 model and weights:
```
mkdir pytorch-benchmarks/ferplus/
```
Then download the resnet50 model and weights by
```
wget -P pytorch-benchmarks/ferplus/ http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/resnet50_ferplus_dag.py

wget -P pytorch-benchmarks/ferplus/ http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/resnet50_ferplus_dag.pth 
```
- torch, torchvision, numpy
My recommendation is to use Anaconda to install torch and torchvision with cudatoolkit:
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
The installed torch version should be 1.4.0 and the installed torchvision version should be 0.5.0. This line will also install numpy-1.18.1.

### Usage

A step by step tutorial on how to use this api:

First, you should download the MIMAMO net pretrained model weights:

```
cd models
bash download_models.sh
```

And run

```
python run_example.py
```
The print out results should be
```
Prediction takes 7.4361 seconds for 309 frames, average 0.0241 seconds for one frame.
utterance_1 predictions
      valence   arousal
0    0.573943  0.623364
1    0.563206  0.647939
2    0.539191  0.648681
3    0.524443  0.691737
4    0.380667  0.585094
..        ...       ...
304 -0.128373  0.573663
305 -0.200220  0.520263
306 -0.083073  0.392294
307 -0.211973  0.374694
308 -0.290508  0.416637
```


## Authors

* **Didan Deng** - *Initial work* 
## License

## Acknowledgments



