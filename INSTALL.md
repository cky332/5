# VIP5 Environment Setup Guide

This guide walks you through setting up the development environment for VIP5 using Anaconda.

## Prerequisites

- Anaconda or Miniconda installed
- NVIDIA GPU with CUDA support (recommended)
- At least 16GB RAM

## Quick Setup (Recommended)

Use the provided `environment.yml` file:

```bash
# Navigate to the project directory
cd /path/to/VIP5

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate vip5
```

## Manual Setup (Step-by-Step)

If you prefer manual installation or encounter issues:

### Step 1: Create Conda Environment

```bash
# Create a new environment with Python 3.9
conda create -n vip5 python=3.9 -y

# Activate the environment
conda activate vip5
```

### Step 2: Install PyTorch

For CUDA 11.6 (recommended):
```bash
conda install pytorch=1.12.0 torchvision=0.13.0 torchaudio=0.12.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```

For CUDA 11.3:
```bash
conda install pytorch=1.12.0 torchvision=0.13.0 torchaudio=0.12.0 cudatoolkit=11.3 -c pytorch
```

For CPU only:
```bash
conda install pytorch=1.12.0 torchvision=0.13.0 torchaudio=0.12.0 cpuonly -c pytorch
```

### Step 3: Install Other Dependencies

```bash
# Core packages via conda
conda install numpy tqdm pyyaml -y

# Packages via pip
pip install transformers sentencepiece packaging

# Install CLIP (required for image features)
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
```

## Verify Installation

Run the following Python code to verify your installation:

```python
import torch
import transformers
import clip
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Transformers version: {transformers.__version__}")
print("All dependencies installed successfully!")
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

1. Check your NVIDIA driver version:
   ```bash
   nvidia-smi
   ```

2. Ensure your driver supports the CUDA version you're installing

3. Try a different CUDA version in the PyTorch installation command

### CLIP Installation Issues

If CLIP installation fails:

```bash
# Alternative installation method
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install -e .
cd ..
```

### Memory Issues

For limited GPU memory, reduce batch size in training scripts:
```bash
--batch_size 16  # or even smaller
```

## Project Setup

After environment setup, complete these additional steps:

1. **Download Data**: Get preprocessed data from the [Google Drive link](https://drive.google.com/drive/u/1/folders/1AjM8Gx4A3xo8seYFWwNUBHpM9uRbfydR) and unzip into `data/` and `features/` folders

2. **Create Output Directories**:
   ```bash
   mkdir -p snap log
   ```

3. **Test Training** (with 4 GPUs):
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_VIP5.sh 4 toys 13579 vitb32 2 8 20
   ```

## Environment Management

```bash
# Activate environment
conda activate vip5

# Deactivate environment
conda deactivate

# Export environment (for sharing)
conda env export > environment_exported.yml

# Remove environment
conda env remove -n vip5
```
