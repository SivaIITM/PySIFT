 ![PySIFT](pysift_banner.png)
# PySIFT

**GPU-Resident Deterministic SIFT for Deep Learning Vision Pipelines**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.x%20%7C%2012.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![arXiv](https://img.shields.io/badge/arXiv-2605.17869-b31b1b.svg)](https://arxiv.org/abs/2605.17869)
[![Kaggle](https://img.shields.io/badge/Kaggle-Tutorial-20BEFF.svg)](https://www.kaggle.com/code/sivakumarksce24d040/pysift-tutorial)
[![Kaggle](https://img.shields.io/badge/Kaggle-Determinism-20BEFF.svg)](https://www.kaggle.com/code/sivakumarksce24d040/pysift-determinism)

A pure-Python, GPU-resident SIFT implementation that matches OpenCV SIFT accuracy while running **26% faster end-to-end** with **4x matching speedup**. Zero-copy DLPack interop keeps tensors on the GPU across the full pipeline -- no PCIe round-trips,cross device deterministic

## Architecture

<p align="center">
  <img src="docs/images/fig_architecture.png" alt="PySIFT Architecture" width="800"/>
</p>

## Key Features

- **GPU-resident pipeline** -- Detection, description, matching, RANSAC, and blending all execute on the GPU via CuPy + Numba CUDA kernels
- **Zero-copy DLPack handoff** -- CuPy arrays pass to PyTorch tensors without memory copies, enabling seamless integration with deep learning pipelines
- **OpenCV-accurate** -- Numerically equivalent to OpenCV SIFT (Lowe 2004), verified across HPatches, Oxford 5K, IMC Phototourism, and MegaDepth-1500
- **Modular descriptor/matcher backends** -- Swap in HardNet, HyNet (learned descriptors) or LightGlue (learned matching) with a single config flag
- **Deterministic** -- Bitwise reproducible results via warp-shuffle reductions (no atomicAdd non-determinism)

## Qualitative Results

<p align="center">
  <img src="docs/images/fig_qualitative.png" alt="Stitching Results" width="800"/>
</p>

## Installation

### Prerequisites: CUDA dependencies

PySIFT requires an NVIDIA GPU with CUDA. Two dependencies must be installed manually because they are CUDA-version-specific:

```bash
# Check your CUDA version
nvcc --version

# 1. Install CuPy (pick ONE matching your CUDA version)
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda11x   # CUDA 11.x

# 2. Install PyTorch with CUDA (default pip installs CPU-only!)
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
```

> **Important:** Both CuPy and PyTorch-CUDA are required runtime dependencies but cannot be auto-installed by pip because the correct package varies by CUDA version. Install them before installing PySIFT.

### Install PySIFT

```bash
# From PyPI
pip install staysift

# Or from GitHub
pip install git+https://github.com/SivaIITM/PySIFT.git

# Or from source
git clone https://github.com/SivaIITM/PySIFT.git
cd PySIFT
pip install -e .
```

### Full install (all dependencies at once)

```bash
pip install cupy-cuda12x   # or cupy-cuda11x
pip install -r requirements.txt
pip install git+https://github.com/SivaIITM/PySIFT.git
```

### Recommended: depth-aware stitching

PySIFT uses MiDaS monocular depth estimation to split inliers into depth bands, giving each band its own homography. This significantly improves stitching quality for scenes with foreground/background parallax. Without `timm`, stitching falls back to a single global homography.

```bash
pip install timm>=0.9
```

### Optional dependencies

```bash
# Learned descriptors (HardNet, HyNet, OriNet)
pip install kornia

# YAML config file support
pip install pyyaml

# Or install all optional deps at once
pip install -e ".[all]"
```

## Quick Start

### Python API

```python
import cv2
from pysift import PySIFT, GPUPyStitch

# Feature extraction (input: grayscale numpy array)
gray = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
sift = PySIFT()
keypoints, descriptors = sift.detectAndCompute(gray)

# Panoramic stitching (input: BGR numpy arrays, not file paths)
img1 = cv2.imread("left.jpg")
img2 = cv2.imread("right.jpg")
stitcher = GPUPyStitch()
panorama = stitcher.stitch(img1, img2)
```

### CLI

```bash
# Basic stitching
pysift-stitch left.jpg right.jpg

# 3-image panorama with output directory
pysift-stitch left.jpg center.jpg right.jpg -o results/

# With config file
pysift-stitch left.jpg right.jpg --config config.yaml

# Learned pipeline
pysift-stitch left.jpg right.jpg --descriptor hardnet --matcher lightglue
```

## Configuration Presets

| Preset | Orientation | Descriptor | Matcher | Use Case |
|--------|-------------|------------|---------|----------|
| **Classic** | histogram | sift | ratio | Fastest. Full Lowe 2004 pipeline |
| **Modern** | histogram | sift | lightglue | Best accuracy with proven detection |
| **Learned** | orinet | hardnet | lightglue | Fully modern pipeline |
| **Mobile** | histogram | sift | ratio | Large phone images (auto-resize + denoise) |

See [`config.yaml`](config.yaml) for all parameters and presets.

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on RTX 3050 4GB and above)
- CUDA Toolkit 11.x or 12.x

### Software

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.9 | Runtime |
| PyTorch | >= 2.0 | Tensor ops, SVD, CUDA graphs |
| CuPy | >= 12.0 | GPU arrays, CUDA kernels |
| Numba | >= 0.57 | JIT-compiled CUDA kernels |
| NumPy | >= 1.22 | CPU array operations |
| OpenCV | >= 4.5 | Image I/O, CLAHE |
| kornia | >= 0.7 | *Optional:* HardNet, HyNet, OriNet |
| timm | >= 0.9 | *Optional:* MiDaS depth estimation |
| PyYAML | any | *Optional:* config file support |

## Citation

**Paper**: [arXiv:2605.17869](https://arxiv.org/abs/2605.17869)

```bibtex
@article{sivakumar2026pysift,
  title   = {PySIFT: GPU-Resident Deterministic SIFT for Deep Learning Vision Pipelines},
  author  = {Sivakumar, K.S.},
  journal = {arXiv preprint arXiv:2605.17869},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.
