# ISE - Image Search Engine
<p align="center">
  <img src="readme_imgs/overview.gif" alt="Overview of how Image Search Engines Work" width="600"/>
</p>


## Introduction
This code implements a simple yet effective image-search engine that runs entirely on your local machine. Given a key image, the user can search for visually and semantically similair images from thousands of images on cheap consumer-grade hardware in little compute time. The code uses BLIP-2 to encode images and vectordb to organise the resulting embeddings. 

## Requirements
<ul>
  <li>Python 3.8+</li>
  <li><a href="https://developer.nvidia.com/cuda-gpus" target="_blank">CUDA Compatible GPUs</a></li> 
</ul>

## Install instructions

### Step 1 - Create Virtual Environment
It is recommended that you create a `conda` environment to avoid dependency issues
```bash
conda create -n ise python=3.8 -y
conda activate ise
conda install pip
```

### Step 2 - Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 - Install CUDA
<ul>
  <li><a href="https://pytorch.org/get-started/locally/" target="_blank">Install PyTorch w/ CUDA support</a></li>
  <li><a href="https://developer.nvidia.com/cuda-downloads" target="_blank">Install apprpriate CUDA toolkit</a></li>
  <li><a href="https://developer.nvidia.com/cuda-downloads](https://medium.com/@harunijaz/a-step-by-step-guide-to-installing-cuda-with-pytorch-in-conda-on-windows-verifying-via-console-9ba4cd5ccbef" target="_blank">CUDA Installation Tutorial</a></li>
</ul>

```bash
#Check Installation
python -c "import torch; print(torch.cuda.is_available())" #True
```


## Usage
⚠WIP⚠

