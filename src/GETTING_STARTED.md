<div align="center">   

# Getting Started
</div>

## Overview
[0. Data Download](#0-data-download)
</br>
[1. Installation](#1-installation)
</br>
[2. Model Training](#2-model-training)
</br>
[3. Model Evaluation](#3-model-evaluation)
</br>


## 0. Data Download

First of all, please download the [dataset](https://drive.google.com/file/d/1cfIdR63Plt546mOiHxGMlgg2cgy4a_HL/view?usp=drive_link). 

## 1. Installation

> Note: our code has been tested on Ubuntu 16.04/18.04 with Python 3.7, CUDA 11.1/11.0, PyTorch 1.7. It may work for other setups, but has not been tested.

Before you run our code, please follow the steps below to build up your environment. 

a. Clone the repository to local
   
```
git clone https://github.com/Toytiny/milliFlow
```
b. Set up a new environment (Python 3.7)  with Anaconda 
   
```
conda create -n $ENV_NAME$ python=3.7
source activate $ENV_NAME$
```
c. Install common dependices and pytorch

```
pip install -r requirements.txt
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```
d. Install [PointNet++](https://github.com/sshaoshuai/Pointnet2.PyTorch) library for basic point cloud operation
```
cd lib
python setup.py install
cd ..
```

## 2. Model Training
Make sure you have successfully completed all above steps before you start running code for model training. 

To train our model, please run:
```
python main.py --dataset_path $DATA_PATH$ --exp_name $EXP_NAME$  --model mmflow --dataset ClipDataset 
```

Here, `$DATA_PATH$` is the path where you save your preprocessed scene flow samples. `EXP_NAME` is the name of the current experiment defined by yourself. Training logs and results will be saved under `checkpoints/$EXP_NAME$/`. Besides, you can also modify training args, such as batch size, learning rate and number of epochs, by editing the configuration file `configs.yaml`.

## 3. Model Evaluation
We provide our trained models  in checkpoints/. You can evaluate our trained models or models trained by yourself.
To evaluate the trained models on the test set, please run:
```
python main.py --eval --dataset_path $DATA_PATH$ --exp_name $EXP_NAME$  --model mmflow --dataset ClipDataset 
```
