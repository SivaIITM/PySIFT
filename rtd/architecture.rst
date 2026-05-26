Architecture -- The 5-Stage GPU Pipeline
=========================================

PySIFT implements Lowe's SIFT algorithm (IJCV 2004) entirely on the GPU using CuPy
RawKernels and Numba ``@cuda.jit``. The image enters VRAM once and never leaves until
the final output.

.. code-block:: text

   Image (CPU)
     -> cp.asarray()                         [1 PCIe H2D transfer]
     -> Gaussian Pyramid (fp16 storage, fp32 chain)
     -> DoG Pyramid (@cp.fuse subtraction)
     -> Extrema Detection (fused RawKernel, 26-neighbour)
     -> Taylor Refinement (Cramer's rule, 5 iterations)
     -> Orientation Assignment (warp-per-keypoint RawKernel)
     -> Descriptor Computation (cooperative warp-per-keypoint, 128-D)
     -> torch.from_dlpack()                  [zero-copy pointer swap]

Stage 1: CLAHE Preprocessing
-----------------------------

Adaptive histogram equalization normalizes local contrast. Yields 40-80% more usable
keypoints on night, haze, and low-contrast images (e.g., drone imagery at dawn,
medical scans with uneven illumination).

Stage 2: Gaussian Scale-Space Pyramid
--------------------------------------

Progressive blur + downsampling across octaves:

- **Shared-memory tiling**: Thread blocks load tiles into L1, reducing global memory reads.
- **fp16 storage with fp32 compute**: Gaussian levels stored in half-precision (2x VRAM savings). Octave 0 in fp32 to preserve weak gradients.
- **Gaussian truncation = 4**: Kernel radius ``int(4*sigma + 0.5)``, matching OpenCV.

Stage 3: DoG Extrema Detection
-------------------------------

``@cp.fuse`` kernel fusion subtracts adjacent Gaussian levels in a single pass.
Fused 26-neighbour comparison (3x3x3 max/min) with integrated contrast gating
replaces the naive three-pass approach.

Stage 4: Orientation Assignment
-------------------------------

Warp-per-keypoint architecture: each warp (32 threads) processes one keypoint.
36-bin weighted orientation histogram, smoothed with 2 iterations of ``[0.25, 0.5, 0.25]``.
80% peak threshold with parabolic sub-bin refinement.

Stage 5: Descriptor Computation
-------------------------------

Cooperative warp-per-keypoint RawKernel computes 128-D gradient-orientation histograms:

- 4x4 spatial bins, 8 orientation bins
- Rotated pixel-space Gaussian weighting
- All array indices use ``((x%n)+n)%n`` (C signed modulo safety)
- DSP-SIFT pooling: descriptors at 5 relative scales, averaged before normalization
- RootSIFT: L1-normalization + element-wise sqrt

Zero-Copy DLPack Handoff
-------------------------

The DLPack protocol exchanges a 64-byte metadata struct (shape, dtype, stride, device ID)
between CuPy and PyTorch. Both frameworks view the same VRAM allocation. No bytes are copied.

This is not merely an optimization -- it is an architectural contribution. Descriptors born
in VRAM (CuPy kernels) are consumed in VRAM (PyTorch matmul or LightGlue transformer)
without touching the PCIe bus.

Measured: DLPack stays sub-millisecond at all resolutions (480p to 8K).
PCIe alternative scales linearly with descriptor count (0.38 ms at 1K kp, 3.3 ms at 10K kp).

VRAM-Adaptive Execution
------------------------

PySIFT auto-detects GPU VRAM tier (``_HIGH_VRAM = vram_gb >= 12``):

.. list-table::
   :header-rows: 1

   * - Parameter
     - 4 GB GPU
     - 24 GB GPU
   * - ``n_octaves``
     - 4
     - 5
   * - ``fp16_pyramid``
     - True
     - False
   * - ``double_image`` cap
     - 4 MP
     - 16 MP

Same codebase, zero config changes. Scales from laptop to server.
