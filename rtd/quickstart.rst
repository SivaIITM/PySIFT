Quick Start
===========

Feature Extraction
------------------

PySIFT is a drop-in replacement for ``cv2.SIFT_create()``:

.. code-block:: python

   from pysift import PySIFT
   import cv2

   sift = PySIFT()
   gray = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

   # Returns OpenCV-compatible KeyPoints + NumPy descriptors
   keypoints, descriptors = sift.detectAndCompute(gray, None)

GPU-Resident Output
-------------------

Keep descriptors in VRAM for zero-copy downstream consumption:

.. code-block:: python

   kp_gpu, desc_gpu = sift.detectAndCompute(gray, None, gpu_output=True)
   # kp_gpu: CuPy (N, 4) -- [x, y, size, angle]
   # desc_gpu: CuPy (N, 128) -- stays in VRAM

   # Zero-copy to PyTorch via DLPack
   import torch
   torch_desc = torch.from_dlpack(desc_gpu)

Feature Matching
----------------

.. code-block:: python

   from pysift import PySIFT
   import cv2
   import numpy as np

   sift = PySIFT(dsp=True)
   img_a = cv2.imread("left.jpg", cv2.IMREAD_GRAYSCALE)
   img_b = cv2.imread("right.jpg", cv2.IMREAD_GRAYSCALE)

   kp_a, desc_a = sift.detectAndCompute(img_a, None)
   kp_b, desc_b = sift.detectAndCompute(img_b, None)

   bf = cv2.BFMatcher(cv2.NORM_L2)
   matches = bf.knnMatch(desc_a, desc_b, k=2)
   good = [m for m, n in matches if m.distance < 0.85 * n.distance]

   # Estimate fundamental matrix
   pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
   pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
   F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.USAC_MAGSAC, 2.0, 0.9999, 10000)
   inliers = mask.ravel().sum()
   print(f"Matches: {len(good)}, Inliers: {inliers}")

Panoramic Stitching
-------------------

.. code-block:: python

   from pysift import GPUPyStitch
   import cv2

   img1 = cv2.imread("left.jpg")
   img2 = cv2.imread("right.jpg")

   stitcher = GPUPyStitch()
   panorama = stitcher.stitch(img1, img2)
   cv2.imwrite("panorama.jpg", panorama)

CLI Stitching
-------------

.. code-block:: bash

   # Basic
   pysift-stitch left.jpg right.jpg

   # 3-image panorama
   pysift-stitch left.jpg center.jpg right.jpg -o results/

   # Learned pipeline
   pysift-stitch left.jpg right.jpg --descriptor hardnet --matcher lightglue

Configuration Presets
---------------------

.. list-table::
   :header-rows: 1

   * - Preset
     - Orientation
     - Descriptor
     - Matcher
     - Use Case
   * - **Classic**
     - histogram
     - sift
     - ratio
     - Fastest. Full Lowe 2004 pipeline.
   * - **Modern**
     - histogram
     - sift
     - lightglue
     - Best accuracy with proven detection.
   * - **Learned**
     - orinet
     - hardnet
     - lightglue
     - Fully modern pipeline.
   * - **Mobile**
     - histogram
     - sift
     - ratio
     - Large phone images (auto-resize + denoise).
