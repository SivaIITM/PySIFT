Use Cases -- PySIFT in Downstream Pipelines
=============================================

Every vision pipeline that matches images starts with feature extraction.
PySIFT replaces the CPU bottleneck at the foundation with a GPU-resident alternative
that requires no retraining, no C++ compilation, and no domain-specific tuning.

Medical Image Registration
--------------------------

Register histopathology whole-slide images, align retinal fundus scans,
fuse multi-modal MRI/CT volumes.

SIFT's physics-based scale-space is domain-agnostic: it works on any imagery without
retraining. Learned detectors like SuperPoint were trained on MS-COCO street-level photos
and degrade on medical tissue textures, retinal vasculature, and radiological features.

.. code-block:: python

   from pysift import PySIFT
   import cv2

   sift = PySIFT(dsp=True)

   slide_a = cv2.imread("tissue_region_a.png", cv2.IMREAD_GRAYSCALE)
   slide_b = cv2.imread("tissue_region_b.png", cv2.IMREAD_GRAYSCALE)

   kp_a, desc_a = sift.detectAndCompute(slide_a, None)
   kp_b, desc_b = sift.detectAndCompute(slide_b, None)

   bf = cv2.BFMatcher(cv2.NORM_L2)
   matches = bf.knnMatch(desc_a, desc_b, k=2)
   good = [m for m, n in matches if m.distance < 0.85 * n.distance]

**Why PySIFT for medical imaging:**

- No training data from your target domain required
- Deterministic output for regulatory compliance (FDA, CE)
- Handles gigapixel whole-slide images (speed scales with resolution)
- DSP-SIFT pooling handles varying magnification levels

Drone / UAV Aerial Stitching
-----------------------------

Real-time aerial mosaic construction on edge GPUs.

.. code-block:: python

   from pysift import GPUPyStitch

   stitcher = GPUPyStitch()
   panorama = stitcher.stitch(frame_left, frame_right)

**Why PySIFT for drones:**

- Runs on 4 GB VRAM (Jetson Orin Nano compatible)
- DSP-SIFT handles altitude-varying scale changes
- Deterministic output for certifiable flight systems
- 3.2x faster than OpenCV at 4K -- real-time aerial frames

SLAM and Visual Odometry
-------------------------

GPU-resident features feed directly into visual odometry without PCIe stall.

.. code-block:: python

   from pysift import PySIFT
   import torch

   sift = PySIFT(dsp=True)

   kp, desc = sift.detectAndCompute(frame, None, gpu_output=True)
   # desc is CuPy (N, 128) in VRAM -- zero-copy to PyTorch
   torch_desc = torch.from_dlpack(desc)
   # Feed directly into pose estimator, loop closure, or map optimizer

**Why PySIFT for SLAM:**

- Zero-copy DLPack handoff eliminates PCIe stall per frame
- True scale invariance (DoG pyramid) vs. SuperPoint's 8px grid
- Deterministic features enable reproducible map building
- Modular: swap in LightGlue matcher with one flag

Robotics -- Object Localization
-------------------------------

Pick-and-place, bin picking, visual servoing.

**Why PySIFT for robotics:**

- Deterministic output = identical features on identical input = repeatable grasps
- No learned model weights to version or update across robot fleet
- Auditable perception stack (physics-based, inspectable code)
- Works on novel objects without fine-tuning

3D Reconstruction (NeRF / 3DGS / SfM)
--------------------------------------

Every NeRF and 3D Gaussian Splatting pipeline starts with COLMAP, which uses CPU SIFT.
PySIFT is a drop-in replacement that eliminates the preprocessing bottleneck.

.. code-block:: python

   from pysift import PySIFT

   sift = PySIFT(dsp=True)

   for img_path in scene_images:
       gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
       kp, desc = sift.detectAndCompute(gray, None, gpu_output=True)
       # GPU-resident features flow directly into matching + BA

**Why PySIFT for 3D reconstruction:**

- Eliminates COLMAP's CPU SIFT bottleneck
- GPU-resident descriptors feed directly into bundle adjustment
- +47% more inliers on IMC Phototourism landmarks
- DSP-SIFT improves wide-baseline matching (+5.6pp AUC@10 on MegaDepth)

Satellite and Remote Sensing
----------------------------

Gigapixel mosaics from satellite strips and multi-spectral band alignment.

**Why PySIFT for satellite/remote sensing:**

- Speed advantage grows with resolution (3.2x at 4K, expected 4-5x at 8K)
- Domain-agnostic: works on any spectral band without retraining
- Handles gigapixel images via VRAM-adaptive execution
- fp16 pyramid storage fits large images in 4 GB VRAM

Comparison: PySIFT vs Learned Detectors for Downstream Use
-----------------------------------------------------------

.. list-table::
   :header-rows: 1

   * - Property
     - PySIFT
     - SuperPoint / Learned
   * - Domain transfer
     - Works on any imagery
     - Trained on MS-COCO, degrades out-of-domain
   * - Determinism
     - Bitwise identical
     - GPU float non-determinism
   * - Scale invariance
     - True (DoG pyramid)
     - Single 8x grid
   * - Model weights
     - 0 MB
     - 614 MB (SuperPoint)
   * - Certifiability
     - Deterministic + inspectable
     - Black box
   * - 4 GB GPU
     - Yes (RTX 3050 tested)
     - Tight with large images
