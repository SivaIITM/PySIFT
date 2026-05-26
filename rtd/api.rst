API Reference
=============

PySIFT
------

.. py:class:: PySIFT(n_octaves=4, n_scales=3, contrast_thresh=0.04, edge_thresh=10.0, double_image=True, rootsift=True, dsp=False, orientation='histogram', descriptor='sift', fp16_pyramid=True)

   GPU-resident SIFT feature detector and descriptor.

   :param int n_octaves: Number of octaves in the Gaussian pyramid (default: 4).
   :param int n_scales: Scale levels per octave (default: 3).
   :param float contrast_thresh: Contrast threshold for keypoint filtering (default: 0.04). Lower values detect more keypoints on low-contrast regions.
   :param float edge_thresh: Edge response threshold (default: 10.0). Higher values keep more edge-like keypoints.
   :param bool double_image: Upsample image 2x before building pyramid (default: True). Auto-suppressed for inputs >4 MP on 4 GB GPUs.
   :param bool rootsift: Apply RootSIFT normalization -- L1 + sqrt (default: True). Improves matching by converting L2 distance to Hellinger distance.
   :param bool dsp: Enable DSP-SIFT multi-scale descriptor pooling (default: False). Recommended for all matching tasks.
   :param str orientation: Orientation assignment method. ``'histogram'`` (default) or ``'orinet'`` (learned, requires kornia).
   :param str descriptor: Descriptor computation method. ``'sift'`` (default), ``'hardnet'``, or ``'hynet'`` (learned, require kornia).
   :param bool fp16_pyramid: Store Gaussian pyramid in half-precision (default: True). Halves VRAM usage with negligible quality loss.

   .. py:method:: detectAndCompute(image, mask=None, gpu_output=False)

      Detect keypoints and compute descriptors.

      :param numpy.ndarray image: Grayscale input image (uint8, HxW).
      :param numpy.ndarray mask: Optional binary mask (same size as image). Only keypoints inside mask are returned.
      :param bool gpu_output: If True, return CuPy arrays in VRAM instead of OpenCV KeyPoints + NumPy arrays.
      :returns: ``(keypoints, descriptors)``

         - **Default** (``gpu_output=False``): ``keypoints`` is a list of ``cv2.KeyPoint``, ``descriptors`` is ``numpy.ndarray (N, 128)``.
         - **GPU mode** (``gpu_output=True``): ``keypoints`` is ``cupy.ndarray (N, 4)`` [x, y, size, angle], ``descriptors`` is ``cupy.ndarray (N, 128)``.

      :rtype: tuple

GPUPyStitch
-----------

.. py:class:: GPUPyStitch(config=None, **kwargs)

   Full GPU-resident panoramic stitching pipeline: feature extraction, matching, RANSAC, warping, and blending.

   :param str config: Path to YAML config file (optional).
   :param kwargs: Override any config parameter (e.g., ``descriptor='hardnet'``, ``matcher='lightglue'``).

   .. py:method:: stitch(img1, img2, ...)

      Stitch two or more BGR images into a panorama.

      :param numpy.ndarray img1: First BGR image.
      :param numpy.ndarray img2: Second BGR image.
      :returns: Stitched panorama as BGR ``numpy.ndarray``.
      :rtype: numpy.ndarray

Zero-Copy DLPack Interop
-------------------------

PySIFT descriptors can be consumed by any DLPack-compatible framework without copying data:

.. code-block:: python

   # CuPy -> PyTorch (zero-copy)
   import torch
   kp, desc = sift.detectAndCompute(img, None, gpu_output=True)
   torch_desc = torch.from_dlpack(desc)

   # CuPy -> JAX (zero-copy)
   import jax.dlpack
   jax_desc = jax.dlpack.from_dlpack(desc.toDlpack())

The DLPack exchange transfers a 64-byte metadata struct (shape, dtype, stride, device ID).
No bytes of descriptor data are copied. This is O(1) regardless of keypoint count.
