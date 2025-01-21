# ISE - Image Search Engine
<p align="center">
  <img src="readme_imgs/overview.gif" alt="Overview of how Image Search Engines Work" width="600"/>
</p>

## Introduction
This code implements a simple yet effective image-search engine that runs entirely on your local machine. Given a key image, the user can search for visually and semantically similair images from thousands of images on cheap consumer-grade hardware in little compute time. The code uses [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) to encode images and [```vectordb```](https://github.com/jina-ai/vectordb) to organise the resulting embeddings. The branch ```blip1``` contains the deprecated BLIP implementation.

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

### Step 0 - Image Directory
Place your images in a folder named ```photos```, inside of another top-level directory of your choosing (eg. ```cat_imgs/photos```).

### Step 1 - Encode Images
```bash
python encode_img.py path/to/cat_imgs
```

### Step 2 - Vectorize Image Database
```bash
python vectorize_encodings.py path/to/cat_imgs
```

### Step 3 - Image-Search 
```bash
#HNSW Search (Recommended)
python vector_search.py path/to/cat_imgs path/to/my_cat.jpg
```
Alternatively, replace ```vectorize_encodings.py``` with ```brute_search.py``` to perform brute force search.

### Step 4 - Retrieve Top N Similar Images
Only supply the name of the image without file extensions. (eg. <s>path/to/my_cat.jpg</s>, my_cat)
```bash
python img_retriever.py path/to/cat_imgs my_cat 100
```

After completing all steps, there should be a folder named ```img_top_N``` inside the top-level directory supplied in [Step 0](#step-0---image-directory)

## Disclaimer ⚠
This project is not actively maintained, however you are free to submit pull requests.
