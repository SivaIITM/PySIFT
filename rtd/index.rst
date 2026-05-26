.. meta::
   :google-site-verification: KfdRtDSRxwba3xj-NR7yzKOAylRaYZrmAxdl7LHABNM

PySIFT -- GPU-Resident SIFT for Computer Vision
=================================================

**PySIFT** is a pure-Python, open-source GPU-resident implementation of the Scale-Invariant Feature Transform
(SIFT) built on CuPy and Numba CUDA kernels which is faster and yet more accurate. It runs the entire
detection-to-descriptor pipeline on your NVIDIA GPU with zero-copy DLPack interop to PyTorch so that
your downstream DL steps will be free from CPU PCIe bottlenecks.

PySIFT is the feature extraction foundation for downstream pipelines in
**medical imaging**, **drone/UAV stitching**, **SLAM**, **robotics**, **3D reconstruction** (NeRF, 3DGS, SfM),
and **satellite remote sensing**.

.. code-block:: bash

   pip install staysift

Key Results
-----------

Benchmarked against OpenCV SIFT on RTX 3050 Laptop GPU (4 GB VRAM):

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 15 15

   * - Benchmark
     - Metric
     - PySIFT
     - OpenCV
     - Delta
   * - HPatches
     - MMA@10
     - **0.703**
     - 0.681
     - +2.2pp
   * - IMC Phototourism
     - Inliers/pair
     - **303**
     - 205
     - +47%
   * - MegaDepth-1500
     - AUC@10
     - **0.503**
     - 0.447
     - +5.6pp
   * - ROxford5K
     - mAP (Medium)
     - **0.455**
     - 0.380
     - +7.5pp

Links
-----

- **Paper**: `arXiv:2605.17869 <https://arxiv.org/abs/2605.17869>`_
- **Code**: `github.com/SivaIITM/PySIFT <https://github.com/SivaIITM/PySIFT>`_
- **PyPI**: `pip install staysift <https://pypi.org/project/staysift/>`_
- **Tutorial**: `sivaiitm.github.io/PySIFT <https://sivaiitm.github.io/PySIFT/>`_
- **HF Space**: `huggingface.co/spaces/sivaIITM/PySIFT <https://huggingface.co/spaces/sivaIITM/PySIFT>`_
- **Kaggle**: `IMC 2026 Warm-Up Sprint <https://www.kaggle.com/competitions/imc-2026-warm-up-landmark-matching-sprint>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api
   benchmarks
   architecture
   use-cases
