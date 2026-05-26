Installation
============

PySIFT requires an NVIDIA GPU with CUDA 11.x or 12.x.

Step 1: GPU Dependencies
------------------------

CuPy and PyTorch-CUDA are CUDA-version-specific and must be installed manually:

.. code-block:: bash

   # Check your CUDA version
   nvcc --version

   # CuPy (pick ONE matching your CUDA version)
   pip install cupy-cuda12x   # CUDA 12.x
   pip install cupy-cuda11x   # CUDA 11.x

   # PyTorch with CUDA (default pip installs CPU-only!)
   pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
   pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8

Step 2: Install PySIFT
----------------------

.. code-block:: bash

   # From PyPI
   pip install staysift

   # Or from GitHub
   pip install git+https://github.com/SivaIITM/PySIFT.git

   # Or from source
   git clone https://github.com/SivaIITM/PySIFT.git
   cd PySIFT
   pip install -e .

Optional Dependencies
---------------------

.. code-block:: bash

   # Learned descriptors (HardNet, HyNet, OriNet)
   pip install kornia>=0.7

   # Depth-aware stitching (MiDaS)
   pip install timm>=0.9

   # YAML config file support
   pip install pyyaml

   # All optional deps at once
   pip install -e ".[all]"

Verification
------------

.. code-block:: python

   from pysift import PySIFT
   import cv2

   sift = PySIFT()
   gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
   kp, desc = sift.detectAndCompute(gray, None)
   print(f"Detected {len(kp)} keypoints, descriptor shape: {desc.shape}")

If this prints a keypoint count and shape ``(N, 128)``, PySIFT is working.

Hardware Tested
---------------

.. list-table::
   :header-rows: 1

   * - GPU
     - VRAM
     - CUDA
     - Status
   * - RTX 3050 Laptop
     - 4 GB
     - 12.x
     - Primary dev/test platform
   * - RTX 3050 A Laptop
     - 4 GB
     - 12.x
     - Cross-device determinism verified
   * - Tesla T4
     - 16 GB
     - 12.x
     - Kaggle verified
   * - RTX 4090
     - 24 GB
     - 12.x
     - Community reported

No GPU?
-------

If you don't have an NVIDIA GPU, use one of these free cloud options:

- **Kaggle**: Free T4 GPU. `PySIFT tutorial notebook <https://www.kaggle.com/code/sivakumarksce24d040/pysift-tutorial>`_
- **Google Colab**: Free T4 GPU. ``pip install cupy-cuda12x staysift`` in a cell.
