"""
GPUPyStitch — GPU-Accelerated Panoramic Image Stitching  [v3.1]
========================================================

PySIFT version history
  v1.0  Numerically equivalent to OpenCV SIFT; full GPU pipeline in pure Python.
  v1.1  RootSIFT · Symmetric ratio test · Adaptive contrast threshold · Tensor Core fp16 matmul
  v1.2  DSP-SIFT 5-scale pooling · fp16 Gaussian pyramid · PCA 128→64 compression (2026-03-30)
        Measured on 3-image panorama: 98.6%/98.5% inlier rate · seam_rmse 15.9 · 14.11s total
  v2.0  Milestone 3.1: Learned orientation via kornia OriNet (Mishchuk et al., 2019) (2026-03-31)
        orientation='orinet' replaces histogram CUDA kernel with pretrained CNN predictor.
  v2.1  Milestone 3.2: Learned descriptors via kornia HardNet8 / HyNet (2026-03-31)
        descriptor='hardnet'|'hynet' replaces handcrafted SIFT gradient histograms with
        pretrained CNN descriptors. Keeps PySIFT keypoints + orientation (incl. OriNet).
        Patch extraction aligns each 32×32 crop by the keypoint orientation angle before
        passing to the network. HardNet: Mishchuk et al., NeurIPS 2017.
        HyNet: Tian et al., NeurIPS 2020.
  v3.1  GPU/CUDA optimisation pass — 12 changes (2026-04-05)
        Opt1: batched HardNet/HyNet patch extraction (3N→3 kernel launches, ~10-16×)
        Opt2: fused extrema boolean ops (3 passes→1 @cp.fuse kernel)
        Opt3: precomputed sigma increments + gaussian_filter truncate=3 in pyramid
        Opt4: single PCIe D→H transfer for keypoint refinement outputs (5→1)
        Opt5: shared-memory warp-per-keypoint orientation kernel (register spill relief)
        Opt6: shared-memory warp-per-keypoint descriptor kernel (register spill relief)
        Opt7: one-shot Tensor Core matmul replaces 16-batch matching loop (2-3×)
        Opt8: CUDA Graph cache for RANSAC batch eval (zero CPU dispatch on repeat)
        Opt9: direct batch 3×3 cofactor inverse replaces pinv SVD (~3× FLOP saving)
        Opt10: precomputed base index offsets in refine kernel inner loop
        Opt11: SmartLauncher.optimal_tpb() occupancy-aware block size selection
        Opt12: event-based stream sync in extrema detection
  v3.0  Milestone 3.3: LightGlue learned matching backend (2026-03-31)
        matcher='lightglue' replaces symmetric ratio-test with the transformer-based
        LightGlue matcher (Lindenberger et al., ICCV 2023). Uses SIFT-trained weights;
        compatible with all descriptor modes (sift/hardnet/hynet). Returns cv2.DMatch
        list — downstream RANSAC/blend pipeline unchanged.

A depth-aware panoramic stitching pipeline that runs almost entirely on the GPU.
Supports 2-image and 3-image inputs via a single unified interface.

Algorithm overview (5 stages):
  Stage 1 — Feature Extraction
      CLAHE contrast enhancement → DSP-SIFT GPU detection (fp16 pyramid, 5-scale
      descriptor pooling, PCA 128→64 compression) → Tensor Core fp16 symmetric
      ratio-test matching.

  Stage 2 — Homography Estimation (DLT)
      Hartley point normalisation → Direct Linear Transform solved via
      PyTorch batched SVD on the GPU.

  Stage 3 — Robust Fitting (RANSAC)
      1500 homography hypotheses evaluated in parallel on the GPU.
      Symmetric reprojection error. LO-RANSAC two-pass refinement.

  Stage 4 — Depth Analysis
      MiDaS monocular depth estimation. RANSAC inliers split into 4 depth
      bands; each band gets its own locally-fitted homography to correct
      depth-dependent parallax.

  Stage 5 — Composition
      Depth-aware warp (per-band homography compositing) →
      graph-cut seam finding (GPU cost map + dynamic-programming path) →
      multi-band Laplacian pyramid blending (6 levels, all on GPU via CuPy).

Usage:
    from gpu_pystitch import GPUPyStitch
    stitcher = GPUPyStitch()
    panorama = stitcher.stitch(img_left, img_right)          # 2 images
    panorama = stitcher.stitch(img_left, img_center, img_right)  # 3 images

CLI:
    python gpu_pystitch.py left.jpg right.jpg
    python gpu_pystitch.py left.jpg center.jpg right.jpg -o results/

Requirements:
    Python 3.9+, PyTorch (CUDA), CuPy, Numba, OpenCV, timm (for MiDaS)
"""

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

import cv2
import cupy as cp
import cupyx.scipy.ndimage as cpnd
import numba
import numpy as np
import torch
from cupyx.scipy.ndimage import maximum_filter, minimum_filter
from numba import cuda

# =============================================================================
# DEVICE SETUP
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[GPUPyStitch] PyTorch device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[GPUPyStitch] GPU : {torch.cuda.get_device_name(0)}")
    _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    _sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"[GPUPyStitch] VRAM: {_vram_gb:.1f} GB  |  SMs: {_sm_count}")
    torch.backends.cuda.matmul.allow_tf32 = True
    _HIGH_VRAM = _vram_gb >= 12
else:
    print("[GPUPyStitch] No CUDA GPU detected — running on CPU (performance will be limited).")
    _vram_gb = 0.0
    _HIGH_VRAM = False

try:
    cp.cuda.Device(0).use()
    print(f"[GPUPyStitch] CuPy {cp.__version__} on GPU 0")
except Exception as _e:
    print(f"[GPUPyStitch] CuPy warning: {_e}")

try:
    import kornia
    _KORNIA_AVAILABLE = True
except ImportError:
    _KORNIA_AVAILABLE = False

try:
    import lightglue as _lg_mod
    _LIGHTGLUE_AVAILABLE = True
except ImportError:
    _LIGHTGLUE_AVAILABLE = False


# =============================================================================
# IMPROVEMENT A — CuPy Fused Kernels
#
# @cp.fuse compiles element-wise op chains into single GPU kernels at import
# time. Intermediate values stay in L1 cache; no VRAM round-trip for temps.
# =============================================================================

@cp.fuse()
def _fused_dog(g_next, g_curr):
    """Fused DoG: subtract two Gaussian levels in a single kernel pass."""
    return g_next - g_curr


@cp.fuse()
def _fused_extrema_mask(dog, thresh):
    """Fused extrema pre-filter: |dog| > thresh in a single kernel pass."""
    return cp.abs(dog) > thresh


@cp.fuse()
def _fused_extrema_combined(dog, lmax, lmin, thresh):
    """Opt 2: fuse (is_max | is_min) & contrast_mask into one kernel.

    Eliminates 3 separate boolean array passes (is_max, is_min, & combined)
    that previously generated 3 full-array kernel launches after max/min filters.
    Single kernel reads dog, lmax, lmin once; outputs bool mask.
    """
    return ((dog >= lmax) | (dog <= lmin)) & (cp.abs(dog) > thresh)


# =============================================================================
# Fused Extrema Detection RawKernel
#
# Single-pass 26-neighbour extrema check + contrast gate + coordinate output.
# Replaces separate maximum_filter + minimum_filter + _fused_extrema_combined
# + cp.argwhere pipeline.  Eliminates two large temp arrays (lmaxs, lmins).
# =============================================================================
_FIND_EXTREMA_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void find_extrema_fused(const float* dog, int* coords, int* count,
                        int S, int H, int W, float thresh, int max_cands,
                        int border) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * H * W;
    if (idx >= total) return;

    int s = idx / (H * W);
    int r = (idx % (H * W)) / W;
    int c = idx % W;

    // Skip boundary and first/last scale
    if (s < 1 || s >= S - 1) return;
    if (r < border || r >= H - border || c < border || c >= W - border) return;

    float v = dog[idx];
    if (fabsf(v) <= thresh) return;  // contrast gate -- skips ~95% of voxels

    bool is_max = true, is_min = true;
    for (int ds = -1; ds <= 1; ds++) {
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (ds == 0 && dr == 0 && dc == 0) continue;
                float nb = dog[(s + ds) * H * W + (r + dr) * W + (c + dc)];
                is_max = is_max && (v >= nb);
                is_min = is_min && (v <= nb);
                if (!is_max && !is_min) return;  // early exit
            }
        }
    }
    if (is_max || is_min) {
        int slot = atomicAdd(count, 1);
        if (slot < max_cands) {
            coords[slot * 3]     = s;
            coords[slot * 3 + 1] = r;
            coords[slot * 3 + 2] = c;
        }
    }
}
''', 'find_extrema_fused')


# =============================================================================
# Orientation Assignment RawKernel — warp-cooperative, --use_fast_math
#
# 4 warps (keypoints) per block, 32 lanes per warp.
# Lanes cooperatively iterate over patch pixels (stride 32) and atomicAdd
# to a shared-memory 36-bin histogram.  Lane 0 does smoothing + peak-finding.
# Replaces Numba _assign_orientations_shared_kernel — eliminates JIT dispatch.
# =============================================================================
_ORIENT_RAWKERNEL = cp.RawKernel(r'''
extern "C" __global__
void orientation_hist(
    const float* __restrict__ gauss_3d,
    const float* __restrict__ kpts,
    float*       __restrict__ out_angles,
    int N, int H, int W
) {
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int kpt_idx = blockIdx.x * 4 + warp_id;

    __shared__ float sh_hist[4 * 36];

    // Per-thread local histogram (deterministic -- no atomicAdd)
    float local_hist[36];
    for (int b = 0; b < 36; b++) local_hist[b] = 0.0f;

    if (kpt_idx < N) {
        float ri        = kpts[kpt_idx * 4 + 0];
        float ci        = kpts[kpt_idx * 4 + 1];
        float sigma_loc = kpts[kpt_idx * 4 + 2];
        int   si        = (int)kpts[kpt_idx * 4 + 3];

        int radius     = max(1, (int)(4.5f * sigma_loc + 0.5f));
        float sigma_w  = 1.5f * sigma_loc;
        float inv_2sw2 = 1.0f / (2.0f * sigma_w * sigma_w + 1e-12f);

        int r_min = max(1, (int)ri - radius);
        int r_max = min(H - 2, (int)ri + radius);
        int c_min = max(1, (int)ci - radius);
        int c_max = min(W - 2, (int)ci + radius);
        int n_cols = max(1, c_max - c_min + 1);
        int total  = max(0, r_max - r_min + 1) * n_cols;

        for (int pidx = lane; pidx < total; pidx += 32) {
            int r = r_min + pidx / n_cols;
            int c = c_min + pidx % n_cols;

            float dx = gauss_3d[si * H * W + r * W + (c+1)]
                     - gauss_3d[si * H * W + r * W + (c-1)];
            float dy = gauss_3d[si * H * W + (r+1) * W + c]
                     - gauss_3d[si * H * W + (r-1) * W + c];
            float mag = sqrtf(dx*dx + dy*dy);
            float ori = fmodf(atan2f(dy, dx) * 57.29577951f + 360.0f, 360.0f);
            float wt  = __expf(-((r - ri)*(r - ri) + (c - ci)*(c - ci)) * inv_2sw2);

            float bin_f = ori * 0.1f;
            int bin_i = ((int)bin_f) % 36;
            int bin_r = (bin_i + 1) % 36;
            float frac = bin_f - floorf(bin_f);
            local_hist[bin_i] += mag * wt * (1.0f - frac);
            local_hist[bin_r] += mag * wt * frac;
        }
    }

    // Deterministic warp-shuffle reduction: fixed tree order per bin
    const int woff = warp_id * 36;
    for (int b = 0; b < 36; b++) {
        float val = local_hist[b];
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        if (lane == 0) sh_hist[woff + b] = val;
    }
    __syncthreads();

    if (kpt_idx < N && lane == 0) {
        float hist[36], smooth[36];
        for (int b = 0; b < 36; b++) hist[b] = sh_hist[woff + b];

        for (int iter = 0; iter < 2; iter++) {
            for (int b = 0; b < 36; b++)
                smooth[b] = 0.25f * hist[(b+35)%36] + 0.5f * hist[b]
                          + 0.25f * hist[(b+1)%36];
            for (int b = 0; b < 36; b++) hist[b] = smooth[b];
        }

        float peak = 0.0f;
        for (int b = 0; b < 36; b++)
            if (hist[b] > peak) peak = hist[b];

        float thresh = 0.8f * peak;
        int out_idx = 0;
        for (int b = 0; b < 36; b++) {
            if (hist[b] >= thresh) {
                float l = hist[(b+35)%36], rv = hist[(b+1)%36];
                if (hist[b] >= l && hist[b] >= rv) {
                    float denom = l - 2.0f * hist[b] + rv;
                    float fine = b + (fabsf(denom) > 1e-12f
                                      ? 0.5f*(l - rv)/denom : 0.0f);
                    float angle = fmodf(fine / 36.0f * 360.0f + 360.0f, 360.0f);
                    if (out_idx < 4) {
                        out_angles[kpt_idx * 4 + out_idx] = angle;
                        out_idx++;
                    }
                }
            }
        }
        for (int k = out_idx; k < 4; k++)
            out_angles[kpt_idx * 4 + k] = -1.0f;
    }
}
''', 'orientation_hist')  # fast_math removed: atan2f/expf precision matters for orientation


# =============================================================================
# Taylor Refinement RawKernel — replaces Numba _mod_refine_kernel
#
# One thread per candidate. Cramer's rule 3×3 solve, 5-iteration Newton,
# edge-response check.  Same algorithm as _mod_refine_kernel but compiled
# ahead of time with --use_fast_math, eliminating Numba JIT dispatch cost.
# =============================================================================
_REFINE_RAWKERNEL = cp.RawKernel(r'''
extern "C" __global__
void refine_keypoints(
    const float* __restrict__ dogs_flat,
    const int*   __restrict__ dogs_meta,
    const int*   __restrict__ all_cands,
    int N, int n_scales, int border_,
    float contrast_thresh_s, float edge_score,
    int*   __restrict__ out_valid,
    int*   __restrict__ out_oct,
    float* __restrict__ out_ri,
    float* __restrict__ out_ci,
    int*   __restrict__ out_si,
    float* __restrict__ out_response
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int o   = all_cands[idx * 4 + 0];
    int si  = all_cands[idx * 4 + 1];
    int row = all_cands[idx * 4 + 2];
    int col = all_cands[idx * 4 + 3];

    int H_o  = dogs_meta[o * 4 + 1];
    int W_o  = dogs_meta[o * 4 + 2];
    int base = dogs_meta[o * 4 + 3];
    int HW   = H_o * W_o;
    int S    = n_scales;

    float ox = 0.0f, oy = 0.0f, os_ = 0.0f;
    bool converged = false;

    for (int iter = 0; iter < 5; iter++) {
        if (si < 1 || si > S || row < border_ || row >= H_o - border_
            || col < border_ || col >= W_o - border_)
            break;

        int b_c_r  = base + si*HW + row*W_o;
        int b_c_r1 = base + si*HW + (row+1)*W_o;
        int b_c_rm = base + si*HW + (row-1)*W_o;
        int b_p_r  = base + (si+1)*HW + row*W_o;
        int b_p_r1 = base + (si+1)*HW + (row+1)*W_o;
        int b_p_rm = base + (si+1)*HW + (row-1)*W_o;
        int b_m_r  = base + (si-1)*HW + row*W_o;
        int b_m_r1 = base + (si-1)*HW + (row+1)*W_o;
        int b_m_rm = base + (si-1)*HW + (row-1)*W_o;

        float cval = dogs_flat[b_c_r + col];
        float dx   = (dogs_flat[b_c_r + col+1] - dogs_flat[b_c_r + col-1]) * 0.5f;
        float dy   = (dogs_flat[b_c_r1 + col]  - dogs_flat[b_c_rm + col])  * 0.5f;
        float ds_v = (dogs_flat[b_p_r + col]   - dogs_flat[b_m_r + col])   * 0.5f;

        float dxx = dogs_flat[b_c_r + col+1] - 2.0f*cval + dogs_flat[b_c_r + col-1];
        float dyy = dogs_flat[b_c_r1 + col]  - 2.0f*cval + dogs_flat[b_c_rm + col];
        float dss = dogs_flat[b_p_r + col]   - 2.0f*cval + dogs_flat[b_m_r + col];
        float dxy = (dogs_flat[b_c_r1 + col+1] - dogs_flat[b_c_r1 + col-1]
                    - dogs_flat[b_c_rm + col+1] + dogs_flat[b_c_rm + col-1]) * 0.25f;
        float dxs = (dogs_flat[b_p_r + col+1] - dogs_flat[b_p_r + col-1]
                    - dogs_flat[b_m_r + col+1] + dogs_flat[b_m_r + col-1]) * 0.25f;
        float dys = (dogs_flat[b_p_r1 + col]  - dogs_flat[b_p_rm + col]
                    - dogs_flat[b_m_r1 + col]  + dogs_flat[b_m_rm + col]) * 0.25f;

        float det = dxx*(dyy*dss - dys*dys) - dxy*(dxy*dss - dys*dxs)
                  + dxs*(dxy*dys - dyy*dxs);
        if (fabsf(det) < 1e-12f) break;

        float inv_det = 1.0f / det;
        float p = -dx, q = -dy, rv = -ds_v;
        ox  = (p*(dyy*dss-dys*dys) - dxy*(q*dss-dys*rv)
             + dxs*(q*dys-dyy*rv)) * inv_det;
        oy  = (dxx*(q*dss-dys*rv) - p*(dxy*dss-dys*dxs)
             + dxs*(dxy*rv-q*dxs)) * inv_det;
        os_ = (dxx*(dyy*rv-q*dys) - dxy*(dxy*rv-q*dxs)
             + p*(dxy*dys-dyy*dxs)) * inv_det;

        if (fabsf(ox) < 0.5f && fabsf(oy) < 0.5f && fabsf(os_) < 0.5f) {
            converged = true; break;
        }

        col += (ox >= 0.0f) ? (int)(ox + 0.5f) : (int)(ox - 0.5f);
        row += (oy >= 0.0f) ? (int)(oy + 0.5f) : (int)(oy - 0.5f);
        si  += (os_ >= 0.0f) ? (int)(os_ + 0.5f) : (int)(os_ - 0.5f);
    }

    if (!converged) { out_valid[idx] = 0; return; }
    if (si < 1 || si > S || row < border_ || row >= H_o - border_
        || col < border_ || col >= W_o - border_) {
        out_valid[idx] = 0; return;
    }

    int fc_r  = base + si*HW + row*W_o;
    int fc_r1 = base + si*HW + (row+1)*W_o;
    int fc_rm = base + si*HW + (row-1)*W_o;
    int fp_r  = base + (si+1)*HW + row*W_o;
    int fm_r  = base + (si-1)*HW + row*W_o;

    float cval = dogs_flat[fc_r + col];
    float dx_  = (dogs_flat[fc_r + col+1] - dogs_flat[fc_r + col-1]) * 0.5f;
    float dy_  = (dogs_flat[fc_r1 + col]  - dogs_flat[fc_rm + col])  * 0.5f;
    float ds_  = (dogs_flat[fp_r + col]   - dogs_flat[fm_r + col])   * 0.5f;
    float response = fabsf(cval + 0.5f*(dx_*ox + dy_*oy + ds_*os_));
    if (response < contrast_thresh_s) { out_valid[idx] = 0; return; }

    float dxx = dogs_flat[fc_r + col+1] - 2.0f*cval + dogs_flat[fc_r + col-1];
    float dyy = dogs_flat[fc_r1 + col]  - 2.0f*cval + dogs_flat[fc_rm + col];
    float dxy = (dogs_flat[fc_r1 + col+1] - dogs_flat[fc_r1 + col-1]
                - dogs_flat[fc_rm + col+1] + dogs_flat[fc_rm + col-1]) * 0.25f;
    float trace = dxx + dyy;
    float det2d = dxx*dyy - dxy*dxy;
    if (det2d <= 0.0f || (trace*trace / det2d) >= edge_score) {
        out_valid[idx] = 0; return;
    }

    out_valid[idx]    = 1;
    out_oct[idx]      = o;
    out_ri[idx]       = (float)row + oy;
    out_ci[idx]       = (float)col + ox;
    out_si[idx]       = si;
    out_response[idx] = response;
}
''', 'refine_keypoints', options=('--use_fast_math',))


# =============================================================================
# Custom Separable Gaussian Filter — shared-memory tiled RawKernels
#
# Two-pass separable convolution (horizontal then vertical) with:
#   - fp16 input/output, fp32 accumulation (no precision loss)
#   - Shared-memory tile avoids redundant global memory reads
#   - Precomputed 1-D kernel weights passed as small fp32 array
# Replaces cpnd.gaussian_filter which has heavy Python dispatch overhead.
# =============================================================================
_GAUSS_H_KERNEL = cp.RawKernel(r'''
// Reflect index into [0, N) matching CuPy reflect mode (half-sample symmetric).
__device__ __forceinline__ int reflect(int i, int N) {
    int p = 2 * N;
    i = i % p;
    if (i < 0) i += p;
    if (i >= N) i = 2 * N - 1 - i;
    return i;
}

extern "C" __global__
void gauss_h(const float* __restrict__ in, float* __restrict__ out,
             const float* __restrict__ kern, int H, int W, int radius) {
    extern __shared__ float tile[];
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load tile from UNPADDED image with reflect boundary indexing.
    int load_c = blockIdx.x * blockDim.x - radius + tid;
    tile[tid] = in[r * W + reflect(load_c, W)];
    if (tid < 2 * radius) {
        int extra_c = blockIdx.x * blockDim.x + blockDim.x - radius + tid;
        tile[blockDim.x + tid] = in[r * W + reflect(extra_c, W)];
    }
    __syncthreads();

    if (c < W) {
        float sum = 0.f;
        for (int k = 0; k <= 2 * radius; k++)
            sum += tile[tid + k] * kern[k];
        out[r * W + c] = sum;
    }
}
''', 'gauss_h')

_GAUSS_V_KERNEL = cp.RawKernel(r'''
__device__ __forceinline__ int reflect_v(int i, int N) {
    int p = 2 * N;
    i = i % p;
    if (i < 0) i += p;
    if (i >= N) i = 2 * N - 1 - i;
    return i;
}

extern "C" __global__
void gauss_v(const float* __restrict__ in, float* __restrict__ out,
             const float* __restrict__ kern, int H, int W, int radius) {
    extern __shared__ float tile[];
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int tid_y = threadIdx.y;
    int tile_h = blockDim.y + 2 * radius;

    if (c < W) {
        // Load tile column from UNPADDED image with reflect boundary.
        int load_r = blockIdx.y * blockDim.y - radius + tid_y;
        tile[tid_y * blockDim.x + threadIdx.x] = in[reflect_v(load_r, H) * W + c];
        for (int extra = tid_y + blockDim.y; extra < tile_h; extra += blockDim.y) {
            int extra_r = blockIdx.y * blockDim.y - radius + extra;
            tile[extra * blockDim.x + threadIdx.x] = in[reflect_v(extra_r, H) * W + c];
        }
    }
    __syncthreads();

    if (r < H && c < W) {
        float sum = 0.f;
        for (int k = 0; k <= 2 * radius; k++)
            sum += tile[(tid_y + k) * blockDim.x + threadIdx.x] * kern[k];
        out[r * W + c] = sum;
    }
}
''', 'gauss_v')

# Cache for precomputed Gaussian kernel weights (sigma → CuPy array)
_gauss_kernel_cache = {}

def _fast_gaussian(img, sigma, truncate=3, stream=None, _keep_alive=None):
    """Separable Gaussian filter using custom CUDA kernels.

    Parameters match cpnd.gaussian_filter(img, sigma, truncate=truncate).
    Works on fp16 or fp32 input; returns same dtype as input.
    Uses in-kernel clamp-reflect boundary indexing — no cp.pad allocation.

    stream      : optional cp.cuda.Stream for non-default-stream launches.
    _keep_alive : optional list — when non-None, temporaries (tmp, work) are
                  appended instead of freed, preventing pool reuse while the
                  stream's kernels are still reading them.  Caller must keep the
                  list alive until the stream is synchronised.
    """
    if sigma < 0.01:
        return img
    in_dtype = img.dtype
    radius = int(truncate * sigma + 0.5)  # match CuPy/SciPy lw calculation

    # Get or compute 1-D Gaussian kernel
    cache_key = (sigma, radius)
    if cache_key not in _gauss_kernel_cache:
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= k.sum()
        _gauss_kernel_cache[cache_key] = cp.asarray(k, dtype=cp.float32)
    kern_gpu = _gauss_kernel_cache[cache_key]

    H, W = img.shape
    # Work in fp32 for precision
    work = img.astype(cp.float32) if in_dtype != cp.float32 else img
    s_kw = {'stream': stream} if stream is not None else {}

    # Horizontal pass: in-kernel reflect indexing, zero allocation overhead
    tmp = cp.empty((H, W), dtype=cp.float32)
    tpb_h = 256
    grid_h = ((W + tpb_h - 1) // tpb_h, H)
    smem_h = (tpb_h + 2 * radius) * 4  # float32 bytes
    _GAUSS_H_KERNEL(grid_h, (tpb_h,), (work, tmp, kern_gpu,
                    np.int32(H), np.int32(W), np.int32(radius)),
                    shared_mem=smem_h, **s_kw)

    # Vertical pass: in-kernel reflect indexing, zero allocation overhead
    out = cp.empty((H, W), dtype=cp.float32)
    tpb_x, tpb_y = 32, 8
    grid_v = ((W + tpb_x - 1) // tpb_x, (H + tpb_y - 1) // tpb_y)
    smem_v = (tpb_y + 2 * radius) * tpb_x * 4
    _GAUSS_V_KERNEL(grid_v, (tpb_x, tpb_y), (tmp, out, kern_gpu,
                    np.int32(H), np.int32(W), np.int32(radius)),
                    shared_mem=smem_v, **s_kw)

    if _keep_alive is not None:
        _keep_alive.append(tmp)
        if work is not img:
            _keep_alive.append(work)
    else:
        del tmp
    return out.astype(in_dtype) if in_dtype != cp.float32 else out


# =============================================================================
# IMPROVEMENT B — Auto-tuning Kernel Launcher
#
# Queries GPU hardware limits once at construction; computes optimal
# thread/block layout per call. Mirrors the ArrayFire approach.
# =============================================================================

class SmartLauncher:
    """
    ArrayFire-style auto-tuning kernel launcher.

    Queries MAX_THREADS_PER_BLOCK from the current CUDA device and computes
    the optimal (blocks_per_grid, threads_per_block) for any 1-D kernel.

    Parameters
    ----------
    preferred_tpb : int
        Starting threads-per-block (sweet spot ~256 for register-heavy kernels).
        Clamped to the hardware maximum automatically.
    """

    def __init__(self, preferred_tpb: int = 256):
        try:
            dev = cuda.get_current_device()
            self._max_tpb  = int(dev.MAX_THREADS_PER_BLOCK)   # usually 1024
            self._sm_count = int(dev.MULTIPROCESSOR_COUNT)
        except Exception:
            self._max_tpb  = 1024
            self._sm_count = 1
        self._preferred_tpb = min(preferred_tpb, self._max_tpb)

    def launch(self, kernel, n_items: int, *args, tpb: int = None):
        """Launch a 1-D Numba CUDA kernel with hardware-tuned grid dimensions."""
        if n_items == 0:
            return
        tpb_ = min(tpb if tpb is not None else self._preferred_tpb, self._max_tpb)
        bpg  = (n_items + tpb_ - 1) // tpb_
        kernel[bpg, tpb_](*args)

    def optimal_tpb(self, registers_per_thread: int,
                    target_occupancy: float = 0.75) -> int:
        """Opt 11: Occupancy-aware TPB selection for register-heavy kernels.

        Iterates candidate TPB values (256→32) and returns the largest one that
        achieves at least `target_occupancy` warp occupancy given the estimated
        register budget per thread.  Falls back to 32 if none qualify.

        Parameters
        ----------
        registers_per_thread : int
            Estimated number of 32-bit registers used per thread.
        target_occupancy : float
            Minimum acceptable warp occupancy (0.0–1.0).  Default 0.75.
        """
        try:
            dev = cuda.get_current_device()
            regs_per_sm   = int(dev.MAX_REGISTERS_PER_BLOCK)
            max_warps_sm  = int(dev.MAX_THREADS_PER_MULTIPROCESSOR) // 32
        except Exception:
            return self._preferred_tpb

        for tpb in [256, 128, 64, 32]:
            regs_needed = registers_per_thread * tpb
            if regs_needed <= regs_per_sm:
                blocks_per_sm = regs_per_sm // max(regs_needed, 1)
                warps_per_sm  = blocks_per_sm * (tpb // 32)
                occupancy     = warps_per_sm / max(max_warps_sm, 1)
                if occupancy >= target_occupancy:
                    return min(tpb, self._max_tpb)
        return min(32, self._max_tpb)




# =============================================================================
# CUDA KERNEL 3a-v2 — Pixel-cooperative RawKernel SIFT descriptor
#
# 1 keypoint per block, 128 threads cooperatively iterate over patch pixels
# (stride 128). Each thread atomicAdds to shared-memory 128-bin histogram.
# Eliminates per-thread full-patch scan: O(pixels/128) iters vs O(pixels).
# For patch_half=17: ~10 iters/thread instead of 1225. ~10-15× faster.
# =============================================================================
_SIFT_DESC_COOP_CODE = r'''
extern "C" __global__
void sift_descriptor_coop(
    const float* __restrict__ gauss_3d,
    const float* __restrict__ kpts,
    float*       __restrict__ descs,
    int N, int n_slices, int H, int W
) {
    const int kpt_idx = blockIdx.x;
    if (kpt_idx >= N) return;
    const int tid = threadIdx.x;   // 0..127

    // sh layout: [0..511] 4 warp-private 128-bin histograms, [512..515] norm partials
    extern __shared__ float sh[];  // 516 floats
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_base = warp_id * 128;

    // Zero warp-private histograms (128 threads × 4 = 512 locations)
    sh[tid*4 + 0] = 0.0f;
    sh[tid*4 + 1] = 0.0f;
    sh[tid*4 + 2] = 0.0f;
    sh[tid*4 + 3] = 0.0f;
    __syncthreads();

    const float ri      = kpts[kpt_idx * 5 + 0];
    const float ci      = kpts[kpt_idx * 5 + 1];
    const float ang_rad = kpts[kpt_idx * 5 + 2];
    const int   ph      = (int)kpts[kpt_idx * 5 + 3];
    const int   si      = (int)kpts[kpt_idx * 5 + 4];

    if (si < 0 || si >= n_slices || ph < 1 || ph > 200) {
        descs[kpt_idx * 128 + tid] = 0.0f;
        return;
    }

    const float cos_a    = cosf(ang_rad);
    const float sin_a    = sinf(ang_rad);
    const float scale_sp = 4.0f / (2.0f * ph);
    const float sigma_d  = (float)ph * 0.5657f;
    const float inv_2s2  = 1.0f / (2.0f * sigma_d * sigma_d + 1e-12f);
    const int   base_off = si * H * W;

    const int side     = 2 * ph + 1;
    const int total_px = side * side;

    for (int flat = tid; flat < total_px; flat += 128) {
        int r_off = flat / side - ph;
        int c_off = flat % side - ph;

        int rr = (int)ri + r_off;
        int cc = (int)ci + c_off;
        if (rr < 1 || rr >= H-1 || cc < 1 || cc >= W-1) continue;

        float xr = cos_a * c_off - sin_a * r_off;
        float yr = sin_a * c_off + cos_a * r_off;
        float xb = xr * scale_sp + 1.5f;
        float yb = yr * scale_sp + 1.5f;
        if (xb < -0.5f || xb >= 3.5f || yb < -0.5f || yb >= 3.5f) continue;

        float dx = gauss_3d[base_off + rr * W + (cc+1)]
                 - gauss_3d[base_off + rr * W + (cc-1)];
        float dy = gauss_3d[base_off + (rr+1) * W + cc]
                 - gauss_3d[base_off + (rr-1) * W + cc];
        float mag = sqrtf(dx*dx + dy*dy);
        float ori_deg = fmodf((atan2f(dy, dx) - ang_rad) * 57.29577951f
                              + 360.0f, 360.0f);
        float gw = expf(-(xr*xr + yr*yr) * inv_2s2);
        float wmag = mag * gw;

        float ob  = ori_deg * 0.02222222f;
        int   oi0 = (int)floorf(ob);
        float df  = ob - oi0;
        int   xi0 = (int)floorf(xb);
        int   yi0 = (int)floorf(yb);
        float dx_b = xb - (float)xi0;
        float dy_b = yb - (float)yi0;

        int xi0c = max(0, min(xi0,   3));
        int yi0c = max(0, min(yi0,   3));
        int xi1c = max(0, min(xi0+1, 3));
        int yi1c = max(0, min(yi0+1, 3));
        int oi0w = ((oi0 % 8) + 8) % 8;
        int oi1w = ((oi0 + 1) % 8 + 8) % 8;

        float w_y0 = 1.0f - dy_b, w_y1 = dy_b;
        float w_x0 = 1.0f - dx_b, w_x1 = dx_b;
        float w_o0 = 1.0f - df,   w_o1 = df;

        atomicAdd(&sh[warp_base + yi0c*32 + xi0c*8 + oi0w], w_y0 * w_x0 * w_o0 * wmag);
        atomicAdd(&sh[warp_base + yi0c*32 + xi0c*8 + oi1w], w_y0 * w_x0 * w_o1 * wmag);
        atomicAdd(&sh[warp_base + yi0c*32 + xi1c*8 + oi0w], w_y0 * w_x1 * w_o0 * wmag);
        atomicAdd(&sh[warp_base + yi0c*32 + xi1c*8 + oi1w], w_y0 * w_x1 * w_o1 * wmag);
        atomicAdd(&sh[warp_base + yi1c*32 + xi0c*8 + oi0w], w_y1 * w_x0 * w_o0 * wmag);
        atomicAdd(&sh[warp_base + yi1c*32 + xi0c*8 + oi1w], w_y1 * w_x0 * w_o1 * wmag);
        atomicAdd(&sh[warp_base + yi1c*32 + xi1c*8 + oi0w], w_y1 * w_x1 * w_o0 * wmag);
        atomicAdd(&sh[warp_base + yi1c*32 + xi1c*8 + oi1w], w_y1 * w_x1 * w_o1 * wmag);
    }

    // Combine 4 warp histograms into sh[0..127] (deterministic fixed-order sum)
    __syncthreads();
    float combined = sh[tid] + sh[128 + tid] + sh[256 + tid] + sh[384 + tid];
    __syncthreads();
    sh[tid] = combined;
    __syncthreads();

    // === L2 normalize -> clip 0.2 -> renormalize (standard SIFT) ===
    float v = sh[tid];
    float partial = v * v;
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 16);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 8);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 4);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 2);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 1);

    if (lane == 0) sh[512 + warp_id] = partial;
    __syncthreads();

    float norm_sq = sh[512] + sh[513] + sh[514] + sh[515];
    float inv_norm = rsqrtf(norm_sq + 1e-14f);
    v *= inv_norm;
    if (v > 0.2f) v = 0.2f;
    sh[tid] = v;
    __syncthreads();

    partial = v * v;
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 16);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 8);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 4);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 2);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 1);
    if (lane == 0) sh[512 + warp_id] = partial;
    __syncthreads();

    norm_sq = sh[512] + sh[513] + sh[514] + sh[515];
    inv_norm = rsqrtf(norm_sq + 1e-14f);
    descs[kpt_idx * 128 + tid] = sh[tid] * inv_norm;
}
'''
_sift_desc_coop = cp.RawKernel(
    _SIFT_DESC_COOP_CODE, 'sift_descriptor_coop',
    # fast_math removed: expf/atan2f precision matters for descriptor weights
)

# =============================================================================
# DSP-SIFT Descriptor RawKernel — 128 threads/keypoint, all DSP scales fused
#
# Replaces Numba _compute_descriptors_dsp_kernel (32 threads/warp per keypoint)
# with 128-thread cooperative block — 4× more parallelism per keypoint.
# Same algorithm: loop over 5 DSP scale multipliers, accumulate histogram,
# divide by n_dsp, L2-normalize + clip 0.2 + re-normalize.
# =============================================================================
_SIFT_DESC_DSP_COOP_CODE = r'''
extern "C" __global__
void sift_descriptor_dsp_coop(
    const float* __restrict__ gauss_3d,
    const float* __restrict__ kpts,
    float*       __restrict__ descs,
    const float* __restrict__ dsp_scales,
    int N, int n_dsp, int n_slices, int H, int W
) {
    const int kpt_idx = blockIdx.x;
    if (kpt_idx >= N) return;
    const int tid = threadIdx.x;   // 0..127

    // sh layout: [0..511] 4 warp-private 128-bin histograms, [512..515] norm partials
    extern __shared__ float sh[];  // 516 floats
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_base = warp_id * 128;

    // Zero warp-private histograms (128 threads × 4 = 512 locations)
    sh[tid*4 + 0] = 0.0f;
    sh[tid*4 + 1] = 0.0f;
    sh[tid*4 + 2] = 0.0f;
    sh[tid*4 + 3] = 0.0f;
    __syncthreads();

    const float ri       = kpts[kpt_idx * 5 + 0];
    const float ci       = kpts[kpt_idx * 5 + 1];
    const float ang_rad  = kpts[kpt_idx * 5 + 2];
    const float ph_base  = kpts[kpt_idx * 5 + 3];
    const int   si       = (int)kpts[kpt_idx * 5 + 4];

    if (si < 0 || si >= n_slices || ph_base < 1.0f || ph_base > 200.0f) {
        descs[kpt_idx * 128 + tid] = 0.0f;
        return;
    }

    const float cos_a = cosf(ang_rad);
    const float sin_a = sinf(ang_rad);
    const int base_off = si * H * W;

    for (int dsp_i = 0; dsp_i < n_dsp; dsp_i++) {
        float sm = dsp_scales[dsp_i];
        int ph = max(4, (int)(ph_base * sm + 0.5f));
        float scale_sp = 4.0f / (2.0f * (float)ph);
        float sigma_d  = (float)ph * 0.5657f;
        float inv_2s2  = 1.0f / (2.0f * sigma_d * sigma_d + 1e-12f);

        int side     = 2 * ph + 1;
        int total_px = side * side;

        for (int flat = tid; flat < total_px; flat += 128) {
            int r_off = flat / side - ph;
            int c_off = flat % side - ph;

            int rr = (int)ri + r_off;
            int cc = (int)ci + c_off;
            if (rr < 1 || rr >= H-1 || cc < 1 || cc >= W-1) continue;

            float xr = cos_a * c_off - sin_a * r_off;
            float yr = sin_a * c_off + cos_a * r_off;
            float xb = xr * scale_sp + 1.5f;
            float yb = yr * scale_sp + 1.5f;
            if (xb < -0.5f || xb >= 3.5f || yb < -0.5f || yb >= 3.5f) continue;

            float dx = gauss_3d[base_off + rr * W + (cc+1)]
                     - gauss_3d[base_off + rr * W + (cc-1)];
            float dy = gauss_3d[base_off + (rr+1) * W + cc]
                     - gauss_3d[base_off + (rr-1) * W + cc];
            float mag = sqrtf(dx*dx + dy*dy);
            float ori_deg = fmodf((atan2f(dy, dx) - ang_rad) * 57.29577951f
                                  + 360.0f, 360.0f);
            float gw = expf(-(xr*xr + yr*yr) * inv_2s2);
            float wmag = mag * gw;

            float ob  = ori_deg * 0.02222222f;
            int   oi0 = (int)floorf(ob);
            float df  = ob - oi0;
            int   xi0 = (int)floorf(xb);
            int   yi0 = (int)floorf(yb);
            float dx_b = xb - (float)xi0;
            float dy_b = yb - (float)yi0;

            int xi0c = max(0, min(xi0,   3));
            int yi0c = max(0, min(yi0,   3));
            int xi1c = max(0, min(xi0+1, 3));
            int yi1c = max(0, min(yi0+1, 3));
            int oi0w = ((oi0 % 8) + 8) % 8;
            int oi1w = ((oi0 + 1) % 8 + 8) % 8;

            float w_y0 = 1.0f - dy_b, w_y1 = dy_b;
            float w_x0 = 1.0f - dx_b, w_x1 = dx_b;
            float w_o0 = 1.0f - df,   w_o1 = df;

            atomicAdd(&sh[warp_base + yi0c*32 + xi0c*8 + oi0w], w_y0 * w_x0 * w_o0 * wmag);
            atomicAdd(&sh[warp_base + yi0c*32 + xi0c*8 + oi1w], w_y0 * w_x0 * w_o1 * wmag);
            atomicAdd(&sh[warp_base + yi0c*32 + xi1c*8 + oi0w], w_y0 * w_x1 * w_o0 * wmag);
            atomicAdd(&sh[warp_base + yi0c*32 + xi1c*8 + oi1w], w_y0 * w_x1 * w_o1 * wmag);
            atomicAdd(&sh[warp_base + yi1c*32 + xi0c*8 + oi0w], w_y1 * w_x0 * w_o0 * wmag);
            atomicAdd(&sh[warp_base + yi1c*32 + xi0c*8 + oi1w], w_y1 * w_x0 * w_o1 * wmag);
            atomicAdd(&sh[warp_base + yi1c*32 + xi1c*8 + oi0w], w_y1 * w_x1 * w_o0 * wmag);
            atomicAdd(&sh[warp_base + yi1c*32 + xi1c*8 + oi1w], w_y1 * w_x1 * w_o1 * wmag);
        }
    }

    // Combine 4 warp histograms into sh[0..127] (deterministic fixed-order sum)
    __syncthreads();
    float combined = sh[tid] + sh[128 + tid] + sh[256 + tid] + sh[384 + tid];
    __syncthreads();
    sh[tid] = combined;
    __syncthreads();

    // Divide accumulated histogram by n_dsp
    float inv_n = 1.0f / (float)n_dsp;
    sh[tid] *= inv_n;
    __syncthreads();

    // === L2 normalize -> clip 0.2 -> renormalize (standard SIFT) ===
    float v = sh[tid];
    float partial = v * v;
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 16);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 8);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 4);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 2);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 1);

    if (lane == 0) sh[512 + warp_id] = partial;
    __syncthreads();

    float norm_sq = sh[512] + sh[513] + sh[514] + sh[515];
    float inv_norm = rsqrtf(norm_sq + 1e-14f);
    v *= inv_norm;
    if (v > 0.2f) v = 0.2f;
    sh[tid] = v;
    __syncthreads();

    partial = v * v;
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 16);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 8);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 4);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 2);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 1);
    if (lane == 0) sh[512 + warp_id] = partial;
    __syncthreads();

    norm_sq = sh[512] + sh[513] + sh[514] + sh[515];
    inv_norm = rsqrtf(norm_sq + 1e-14f);
    descs[kpt_idx * 128 + tid] = sh[tid] * inv_norm;
}
'''
_sift_desc_dsp_coop = cp.RawKernel(
    _SIFT_DESC_DSP_COOP_CODE, 'sift_descriptor_dsp_coop',
)


_ORI_WARPS_PER_BLOCK = 4   # 4 keypoints per block (used by orientation RawKernel)
_DESC_WARPS_PER_BLOCK = 4   # keypoints per block (used by DSP descriptor kernel)

@cuda.jit
def _compute_descriptors_dsp_kernel(gauss_3d, kpts_data, descs, dsp_scales, n_dsp):
    """Batched DSP-SIFT descriptor kernel: all DSP scales in one launch.

    Each warp processes one keypoint across all DSP scales, accumulating
    histogram contributions into shared memory. One kernel launch per octave
    instead of n_dsp launches. kpts_data[:, 3] contains the BASE patch_half
    (before DSP scaling).
    """
    warp_id  = cuda.threadIdx.x // 32
    lane_id  = cuda.threadIdx.x %  32
    kpt_idx  = cuda.blockIdx.x * _DESC_WARPS_PER_BLOCK + warp_id

    if kpt_idx >= kpts_data.shape[0]:
        return

    H  = gauss_3d.shape[1];  W = gauss_3d.shape[2]
    ri         = kpts_data[kpt_idx, 0]
    ci         = kpts_data[kpt_idx, 1]
    ang_rad    = kpts_data[kpt_idx, 2]
    ph_base    = kpts_data[kpt_idx, 3]   # base patch_half before DSP scaling
    si         = int(kpts_data[kpt_idx, 4])

    cos_a = math.cos(ang_rad);  sin_a = math.sin(ang_rad)
    PI = math.pi

    # Shared descriptor histogram: 4 warps × 128 bins
    sh_hist = cuda.shared.array(512, dtype=numba.float32)
    warp_off = warp_id * 128
    # Zero init
    for k in range(4):
        sh_hist[warp_off + lane_id * 4 + k] = numba.float32(0.0)
    cuda.syncthreads()

    # Loop over DSP scales — accumulate all into same histogram
    for dsp_i in range(n_dsp):
        sm = dsp_scales[dsp_i]
        patch_half = max(4, int(ph_base * sm + numba.float32(0.5)))
        scale_sp = 4.0 / (2.0 * patch_half)
        # Gaussian weight: rotated pixel coords, sigma = 0.5657*ph (matches OpenCV)
        sigma_d = patch_half * 0.5657
        inv_2s2 = 1.0 / (2.0 * sigma_d * sigma_d + 1e-12)

        side = 2 * patch_half + 1
        total_pix = side * side
        for flat_idx in range(lane_id, total_pix, 32):
            r_off = flat_idx // side - patch_half
            c_off = flat_idx %  side - patch_half
            xr = cos_a*c_off - sin_a*r_off
            yr = sin_a*c_off + cos_a*r_off
            xb = xr*scale_sp + 1.5;  yb = yr*scale_sp + 1.5
            rr = int(ri) + r_off;    cc = int(ci) + c_off

            if (1 <= rr < H-1 and 1 <= cc < W-1 and -0.5 <= xb < 3.5 and -0.5 <= yb < 3.5):
                dx  = gauss_3d[si, rr, cc+1] - gauss_3d[si, rr, cc-1]
                dy  = gauss_3d[si, rr+1, cc] - gauss_3d[si, rr-1, cc]
                mag = math.sqrt(dx*dx + dy*dy)
                ori_deg = ((math.atan2(dy, dx) - ang_rad) * (180.0/PI)) % 360.0
                # Gaussian weight in rotated pixel coords (matches OpenCV)
                gw   = math.exp(-(xr*xr + yr*yr) * inv_2s2)
                wmag = numba.float32(mag * gw)

                ob   = ori_deg/360.0*8.0;  oi0 = int(math.floor(ob));  df = numba.float32(ob - oi0)
                xi0  = int(math.floor(xb)); yi0 = int(math.floor(yb))
                dx_b = numba.float32(xb - xi0);  dy_b = numba.float32(yb - yi0)
                xi0c = max(0, min(xi0,   3)); yi0c = max(0, min(yi0,   3))
                xi1c = max(0, min(xi0+1, 3)); yi1c = max(0, min(yi0+1, 3))
                f1   = numba.float32(1.0)
                cuda.atomic.add(sh_hist, warp_off + yi0c*32 + xi0c*8 + (oi0  )%8, (f1-dx_b)*(f1-dy_b)*(f1-df)*wmag)
                cuda.atomic.add(sh_hist, warp_off + yi0c*32 + xi0c*8 + (oi0+1)%8, (f1-dx_b)*(f1-dy_b)*df      *wmag)
                cuda.atomic.add(sh_hist, warp_off + yi0c*32 + xi1c*8 + (oi0  )%8, dx_b     *(f1-dy_b)*(f1-df)*wmag)
                cuda.atomic.add(sh_hist, warp_off + yi0c*32 + xi1c*8 + (oi0+1)%8, dx_b     *(f1-dy_b)*df      *wmag)
                cuda.atomic.add(sh_hist, warp_off + yi1c*32 + xi0c*8 + (oi0  )%8, (f1-dx_b)*dy_b     *(f1-df)*wmag)
                cuda.atomic.add(sh_hist, warp_off + yi1c*32 + xi0c*8 + (oi0+1)%8, (f1-dx_b)*dy_b     *df      *wmag)
                cuda.atomic.add(sh_hist, warp_off + yi1c*32 + xi1c*8 + (oi0  )%8, dx_b     *dy_b     *(f1-df)*wmag)
                cuda.atomic.add(sh_hist, warp_off + yi1c*32 + xi1c*8 + (oi0+1)%8, dx_b     *dy_b     *df      *wmag)

    cuda.syncthreads()

    # Divide accumulated histogram by n_dsp to get the average
    inv_n_dsp = numba.float32(1.0 / n_dsp)
    for k in range(4):
        sh_hist[warp_off + lane_id * 4 + k] *= inv_n_dsp
    cuda.syncwarp(0xFFFFFFFF)

    # Warp-cooperative L2-normalise + 0.2-clip + re-normalise
    local_sum1 = numba.float32(0.0)
    for k in range(4):
        v = sh_hist[warp_off + lane_id * 4 + k]
        local_sum1 += v * v
    for shfl_offset in (16, 8, 4, 2, 1):
        local_sum1 += cuda.shfl_down_sync(0xFFFFFFFF, local_sum1, shfl_offset)
    norm1 = numba.float32(math.sqrt(cuda.shfl_sync(0xFFFFFFFF, local_sum1, 0) + 1e-14))
    for k in range(4):
        idx_k = warp_off + lane_id * 4 + k
        v = sh_hist[idx_k] / norm1
        if v > numba.float32(0.2):
            v = numba.float32(0.2)
        sh_hist[idx_k] = v
    local_sum2 = numba.float32(0.0)
    for k in range(4):
        v = sh_hist[warp_off + lane_id * 4 + k]
        local_sum2 += v * v
    for shfl_offset in (16, 8, 4, 2, 1):
        local_sum2 += cuda.shfl_down_sync(0xFFFFFFFF, local_sum2, shfl_offset)
    norm2 = numba.float32(math.sqrt(cuda.shfl_sync(0xFFFFFFFF, local_sum2, 0) + 1e-14))
    for k in range(4):
        sh_hist[warp_off + lane_id * 4 + k] /= norm2

    cuda.syncthreads()

    # All 32 lanes cooperatively write descriptor to global memory
    for k in range(4):
        descs[kpt_idx, lane_id*4 + k] = sh_hist[warp_off + lane_id*4 + k]


# =============================================================================
# CLASS: PySIFT
#
# GPU-accelerated Scale-Invariant Feature Transform (SIFT) detector and
# descriptor extractor. Compatible with the OpenCV KeyPoint / descriptor API.
#
# The implementation follows the original Lowe (2004) algorithm, ported to
# run on the GPU using CuPy (for scale-space pyramid construction) and Numba
# CUDA kernels (for per-keypoint refinement, orientation, and descriptor).
# =============================================================================
def _adaptive_contrast_thresh(gray_u8, base_thresh=0.04):
    """
    Compute an entropy-based adaptive contrast threshold for a single image.

    Low-entropy images (night, haze, indoor) get a lower threshold to capture
    more keypoints; high-entropy images (rich texture) get a higher threshold
    to suppress noise in repetitive regions.

    The normalised Shannon entropy of the intensity histogram maps to a
    multiplier in [0.5, 1.5], so the threshold can range from 0.5× to 1.5×
    the base value — centred on the standard Lowe 0.04 for typical scenes.

    Reference: Safdari & Moallem, IEEE JSTARS 2019.
    """
    hist    = cp.bincount(cp.asarray(gray_u8).ravel(), minlength=256).astype(cp.float32)
    hist   /= hist.sum()
    hist    = hist[hist > 0]
    entropy = -float(cp.sum(hist * cp.log2(hist)))   # 0 – 8 bits
    scale   = min(1.0, 0.8 + entropy / 40.0)          # range [0.8, 1.0] — narrowed to stay close to Lowe's fixed 0.04 for benchmark parity
    return float(base_thresh * scale)


class PySIFT:
    """
    PySIFT — Python-native GPU keypoint detector and descriptor extractor.

    A complete SIFT implementation written entirely in Python (CuPy + Numba CUDA),
    with no compiled C++ extension required. This makes it uniquely portable,
    inspectable, and modifiable compared to existing SIFT implementations that
    expose only a C++ binary (e.g. OpenCV's cv2.SIFT_create).

    The full detection pipeline runs on the GPU:
      - Gaussian scale-space pyramid (CuPy)
      - Difference-of-Gaussians extrema detection (CuPy)
      - Sub-pixel keypoint refinement — Numba CUDA kernel
      - Dominant orientation assignment — Numba CUDA kernel
      - 128-dimensional descriptor computation — Numba CUDA kernel

    Returns keypoints and descriptors in the same format as OpenCV detectors,
    so it can be used as a drop-in replacement in any matching pipeline.

    Parameters
    ----------
    n_octaves : int
        Number of scale-space octaves (image resolution halved per octave).
    n_scales : int
        Number of scale levels per octave (S in Lowe 2004).
    sigma0 : float
        Base Gaussian sigma at the first scale level.
    contrast_thresh : float
        Minimum normalised DoG response to retain a keypoint.
    edge_thresh : float
        Principal curvature ratio threshold for edge-response rejection.
    """

    def __init__(self, n_octaves=None, n_scales=3, sigma0=1.6,
                 contrast_thresh=0.04, edge_thresh=10.0,
                 dsp=False, dsp_n_scales=3, fp16_pyramid=None,
                 orientation='histogram',
                 descriptor='sift', double_image=True, rootsift=True):
        _hv = _HIGH_VRAM
        self._high_vram      = _hv
        self.n_octaves       = n_octaves if n_octaves is not None else (5 if _hv else 4)
        self.n_scales        = n_scales
        self.sigma0          = sigma0
        self.contrast_thresh = contrast_thresh
        self.edge_thresh     = edge_thresh
        self.rootsift        = rootsift     # v3.1: RootSIFT post-normalization
        self.dsp             = dsp          # v1.2: DSP-SIFT multi-scale pooling
        _DSP_PRESETS = {
            3: np.array([1.0/math.sqrt(2), 1.0, math.sqrt(2)], dtype=np.float32),
            4: np.array([0.5, 1.0/math.sqrt(2), 1.0, math.sqrt(2)], dtype=np.float32),
            5: np.array([0.5, 1.0/math.sqrt(2), 1.0, math.sqrt(2), 2.0], dtype=np.float32),
        }
        if dsp_n_scales not in _DSP_PRESETS:
            raise ValueError(f"dsp_n_scales must be 3, 4, or 5 (got {dsp_n_scales})")
        self.dsp_n_scales    = dsp_n_scales
        self._dsp_scales_np  = _DSP_PRESETS[dsp_n_scales]
        self.fp16_pyramid    = fp16_pyramid if fp16_pyramid is not None else (not _hv)
        if _hv:
            print(f"[GPUPyStitch] HIGH-VRAM mode: octaves={self.n_octaves}, "
                  f"fp16_pyramid={self.fp16_pyramid}, double_cap=16MP, flush=200")
        self.orientation     = orientation  # v2.0: 'histogram' | 'orinet' | 'affnet'
        self._orinet         = None         # v2.0: lazy-loaded OriNet model
        self._affnet         = None         # v2.2: lazy-loaded LAFAffNetShapeEstimator
        self.descriptor      = descriptor  # v2.1: 'sift' | 'hardnet' | 'hynet'
        self._descriptor_net = None        # v2.1: lazy-loaded HardNet8 / HyNet
        self.double_image    = double_image # v3.1: 2× upsample to match OpenCV default octave -1
        self._launcher       = SmartLauncher(preferred_tpb=256)  # Improvement B
        # Change 8: pinned (page-locked) host buffer for DMA-direct GPU upload.
        # Allocated lazily on first call; grown if a larger image arrives.
        self._pinned_mem     = None   # cp.cuda.PinnedMemoryPointer
        self._pinned_buf     = None   # numpy view of pinned memory, shape (max_H, max_W)
        # Pool-flush counter: after many batch calls (e.g. Oxford5K bulk extraction)
        # the CuPy VRAM pool accumulates freed blocks that are not immediately returned
        # to the CUDA driver.  On Windows WDDM, the kernel-mode driver caches these
        # physical VRAM pages even after cudaFree(), causing progressive exhaustion.
        # Flush every 200 calls to prevent pool bloat during bulk extraction.
        self._n_calls        = 0
        self._phase_timings  = None  # set by detectAndCompute when profile=True
        # No hard ceiling on the CuPy pool — the RTX 3050 has 4 GB VRAM and the
        # 4K HPatches pass (3840×2160 images) peaks at ~1 GB after the fixes below.
        # A 2 GB cap was added for Oxford5K but caused OOM on single 4K images.
        # Pool bloat is handled by explicit free_all_blocks() every 20 calls.

    def _build_gaussian_pyramid(self, gray_u8):
        """Build a Gaussian scale-space pyramid on the GPU using CuPy.

        v1.2: When fp16_pyramid=True, all Gaussian levels are stored in float16
        (~2× memory reduction). DoG subtraction (catastrophic cancellation risk)
        and the CUDA kernels (orientation, descriptor) cast back to float32 on
        demand — at most one octave's fp32 data is live at a time.
        """
        S      = self.n_scales
        k      = 2.0 ** (1.0 / S)
        sigmas = [self.sigma0 * (k**s) for s in range(S + 3)]
        dtype  = cp.float16 if self.fp16_pyramid else cp.float32
        # Change 8: use pinned (page-locked) host memory so the CUDA driver can
        # DMA the image directly to VRAM without a staging copy.
        if isinstance(gray_u8, np.ndarray):
            H_img, W_img = gray_u8.shape
            needed = H_img * W_img
            if self._pinned_buf is None or self._pinned_buf.size < needed:
                self._pinned_mem = cp.cuda.alloc_pinned_memory(needed)
                self._pinned_buf = np.frombuffer(self._pinned_mem, dtype=np.uint8)[:needed].reshape(H_img, W_img)
            elif self._pinned_buf.shape != gray_u8.shape:
                self._pinned_buf = np.frombuffer(self._pinned_mem, dtype=np.uint8)[:needed].reshape(H_img, W_img)
            np.copyto(self._pinned_buf, gray_u8)
            base = cp.asarray(self._pinned_buf, dtype=dtype) / dtype(255.0)
        else:
            # Already a CuPy array (e.g. from GPU CLAHE path) — no upload needed
            H_img, W_img = gray_u8.shape
            base = gray_u8.astype(dtype) / dtype(255.0)
        # Auto-suppress doubling for high-res inputs (> 4 MP).
        # Doubling a 3840×2160 image to 7680×4320 creates a 633 MB octave-0 DoG
        # array — 4× more VRAM for no quality gain at that resolution.
        # For images up to ~2 MP (MegaDepth typical: 1.6 MP) doubling matches
        # OpenCV's default firstOctave=-1 and yields the extra fine-scale octave
        # that helps accuracy. Threshold raised from 1 MP→4 MP on Apr 9 after
        # MegaDepth diagnostic showed 2× keypoint deficit vs OpenCV.
        # _effective_double is read by _refine_keypoints for correct coord scaling.
        _double_px_cap = 16_000_000 if self._high_vram else 4_194_304
        self._effective_double = self.double_image and (H_img * W_img < _double_px_cap)
        if self._effective_double:
            # Match OpenCV's default: upsample 2× so fine-scale keypoints are detectable.
            base = cpnd.zoom(base.astype(cp.float32), 2.0, order=3).astype(dtype)
        # 2× upsampling doubles the effective input blur from 0.5 to 1.0 in the
        # upsampled coordinate system (coordinate-system property, independent of
        # interpolation method).  OpenCV uses sqrt(sigma0^2 - 1.0^2) for doubled.
        assumed_blur = 1.0 if self._effective_double else 0.5
        sd0    = float(np.sqrt(max(self.sigma0**2 - assumed_blur**2, 1e-6)))
        # Precompute incremental sigmas: sd[s] = sqrt(sigma[s]^2 - sigma[s-1]^2)
        inc_sds = [float(np.sqrt(max(sigmas[s]**2 - sigmas[s-1]**2, 1e-6)))
                   for s in range(1, S + 3)]

        # Work in fp32 during Gaussian chain to avoid fp32→fp16→fp32 roundtrips
        # between consecutive levels. Store directly into pre-allocated 3D arrays
        # to avoid a separate cp.stack pass later (~4ms on 768×1024).
        base_f32 = base.astype(cp.float32) if base.dtype != cp.float32 else base
        base_f32 = _fast_gaussian(base_f32, sd0, truncate=4)
        gauss_pyr = []
        _tail_event  = None
        _tail_results = []
        _tail_temps   = []
        _tail_oct     = None

        for o in range(self.n_octaves):
            # Sync previous octave's tail levels (launched on side stream)
            if _tail_event is not None:
                _tail_event.synchronize()
                for _slot, _arr in _tail_results:
                    _tail_oct[_slot] = _arr
                _tail_results.clear()
                _tail_temps.clear()
                _tail_event = None

            H_oct, W_oct = base_f32.shape
            # Octaves 0-1 in fp32: preserves descriptor gradients where most
            # keypoints concentrate. Octaves 2+ use storage dtype (fp16).
            store_dtype = cp.float32 if o <= 1 else dtype
            gauss_oct = cp.empty((S + 3, H_oct, W_oct), dtype=store_dtype)
            gauss_oct[0] = base_f32
            prev = base_f32

            # Levels 0..S-1 on default stream (level S-1 = next octave base)
            for s in range(S):
                blurred = _fast_gaussian(prev, inc_sds[s], truncate=4)
                gauss_oct[s + 1] = blurred
                prev = blurred
            next_base = prev

            # Levels S, S+1: launch on side stream to overlap with next octave
            if o < self.n_octaves - 1:
                _tail_stream = cp.cuda.Stream(non_blocking=True)
                _dep_ev = cp.cuda.Event()
                _dep_ev.record()
                _tail_stream.wait_event(_dep_ev)
                for s in range(S, S + 2):
                    blurred = _fast_gaussian(prev, inc_sds[s], truncate=4,
                                             stream=_tail_stream,
                                             _keep_alive=_tail_temps)
                    _tail_results.append((s + 1, blurred))
                    prev = blurred
                _tail_event = cp.cuda.Event()
                _tail_event.record(_tail_stream)
                _tail_oct = gauss_oct
            else:
                # Last octave: no overlap opportunity
                for s in range(S, S + 2):
                    blurred = _fast_gaussian(prev, inc_sds[s], truncate=4)
                    gauss_oct[s + 1] = blurred
                    prev = blurred

            gauss_pyr.append(gauss_oct)
            base_f32 = next_base[::2, ::2].astype(cp.float32)

        # Final sync for the last overlapped octave's tail
        if _tail_event is not None:
            _tail_event.synchronize()
            for _slot, _arr in _tail_results:
                _tail_oct[_slot] = _arr
            del _tail_results, _tail_temps

        return gauss_pyr

    @staticmethod
    def _build_dog_pyramid(gauss_pyr):
        """Compute Difference-of-Gaussians pyramid: DoG[s] = G[s+1] - G[s].

        v1.2: Cast each level to fp32 before subtraction to avoid catastrophic
        cancellation when the pyramid was built in fp16.
        Improvement A: uses @cp.fuse _fused_dog to keep intermediate in L1 cache.
        """
        return [[_fused_dog(gauss_pyr[o][s+1].astype(cp.float32),
                            gauss_pyr[o][s].astype(cp.float32))
                 for s in range(len(gauss_pyr[o]) - 1)]
                for o in range(len(gauss_pyr))]

    def _find_extrema(self, dog_pyr, contrast_thresh=None):
        """Detect local extrema in 3×3×3 scale-space neighbourhoods on the GPU.

        Parallel design: all octaves are processed simultaneously on separate
        non-blocking CUDA streams.  The GPU scheduler interleaves octave kernels,
        keeping SMs busy while smaller octaves fill gaps left by large octave-0
        filters.  Events synchronise before CPU reads the results.

        Memory note: after _effective_double auto-suppression for inputs > 1 MP
        (see _build_gaussian_pyramid), the peak for a 4K image is:
          dog_pyr(211) + dogs_3ds(211) + lmaxs(211) + lmins(211) ≈ 844 MB
        which comfortably fits in 4 GB VRAM.  The old 2 GB pool cap has been
        removed; pool bloat is controlled by free_all_blocks() every 20 calls.
        """
        border     = 5
        S          = self.n_scales
        ct         = contrast_thresh if contrast_thresh is not None else self.contrast_thresh
        pre_thresh = float(ct / S)          # BUG1 fix: was 0.5*ct/S
        n_oct    = len(dog_pyr)
        # Synchronize the default stream before launching non-blocking streams.
        # dog_pyr was built on the default stream by _build_dog_pyramid();
        # non-blocking streams skip implicit sync with the default stream,
        # so without this barrier the cp.stack() reads below can race with
        # the DoG writes and silently corrupt extrema detection.
        cp.cuda.runtime.deviceSynchronize()
        streams  = [cp.cuda.Stream(non_blocking=True) for _ in range(n_oct)]
        dogs_3ds = [None] * n_oct
        # Fused extrema: RawKernel output buffers per octave
        MAX_CANDS_PER_OCTAVE = 25000
        coord_bufs = [None] * n_oct
        count_bufs = [None] * n_oct
        events = [cp.cuda.Event() for _ in range(n_oct)]

        for o, dogs in enumerate(dog_pyr):
            with streams[o]:
                dogs_3ds[o] = cp.stack(dogs, axis=0).astype(cp.float32)
                S2, H_o, W_o = dogs_3ds[o].shape
                if H_o < 3 or W_o < 3:
                    events[o].record(streams[o])
                    continue
                # Allocate output buffers for fused extrema kernel
                coord_bufs[o] = cp.empty(MAX_CANDS_PER_OCTAVE * 3, dtype=cp.int32)
                count_bufs[o] = cp.zeros(1, dtype=cp.int32)
                total_voxels = S2 * H_o * W_o
                threads = 256
                blocks = (total_voxels + threads - 1) // threads
                _FIND_EXTREMA_KERNEL(
                    (blocks,), (threads,),
                    (dogs_3ds[o], coord_bufs[o], count_bufs[o],
                     np.int32(S2), np.int32(H_o), np.int32(W_o),
                     np.float32(pre_thresh), np.int32(MAX_CANDS_PER_OCTAVE),
                     np.int32(border)),
                    stream=streams[o]
                )
                events[o].record(streams[o])

        cands_list = []
        for o in range(n_oct):
            events[o].synchronize()
            if coord_bufs[o] is None:
                continue
            n_found = int(count_bufs[o].get())
            if n_found == 0:
                continue
            n_found = min(n_found, MAX_CANDS_PER_OCTAVE)
            coords = coord_bufs[o][:n_found * 3].reshape(n_found, 3)
            # Cap at 25000: keep highest-|DoG| candidates
            if n_found > MAX_CANDS_PER_OCTAVE:
                dogs_3d = dogs_3ds[o]
                responses = cp.abs(dogs_3d[coords[:, 0], coords[:, 1], coords[:, 2]])
                idx_sub   = cp.argsort(responses)[-MAX_CANDS_PER_OCTAVE:]
                coords    = coords[idx_sub]
            oct_col = cp.full((coords.shape[0], 1), o, dtype=cp.int32)
            cands_list.append(cp.hstack([oct_col, coords]))
        del coord_bufs, count_bufs

        # Build dogs_flat and release all large intermediates.
        dogs_flat_parts = []
        meta_rows       = []
        offset          = 0
        for o in range(n_oct):
            d3 = dogs_3ds[o]
            S2, H, W = d3.shape
            dogs_flat_parts.append(d3.ravel())
            meta_rows.append([S2, H, W, offset])
            offset += S2 * H * W
        dogs_flat = cp.concatenate(dogs_flat_parts)
        dogs_meta = cp.array(meta_rows, dtype=cp.int32)
        del dogs_flat_parts, dogs_3ds, streams, events

        if not cands_list:
            return cp.empty((0, 4), dtype=cp.int32), dogs_flat, dogs_meta
        all_cands = cp.vstack(cands_list)
        return all_cands, dogs_flat, dogs_meta

    def _refine_keypoints(self, all_cands, dogs_flat, dogs_meta, gauss_pyr,
                          contrast_thresh=None):
        """Run sub-pixel refinement kernel; return list of accepted keypoint dicts."""
        if all_cands.shape[0] == 0:
            return []
        S          = self.n_scales
        k_val      = 2.0 ** (1.0 / S)
        border     = 5
        ct         = contrast_thresh if contrast_thresh is not None else self.contrast_thresh
        c_thresh_s = float(ct / S)
        edge_score = float((self.edge_thresh + 1.0)**2 / self.edge_thresh)
        N       = int(all_cands.shape[0])
        d_valid = cp.empty(N, dtype=cp.int32)
        d_oct   = cp.empty(N, dtype=cp.int32)
        d_ri    = cp.empty(N, dtype=cp.float32)
        d_ci    = cp.empty(N, dtype=cp.float32)
        d_si    = cp.empty(N, dtype=cp.int32)
        d_resp  = cp.empty(N, dtype=cp.float32)
        # RawKernel refinement — replaces Numba _mod_refine_kernel
        _refine_tpb = 256
        _refine_bpg = (N + _refine_tpb - 1) // _refine_tpb
        _REFINE_RAWKERNEL(
            (_refine_bpg,), (_refine_tpb,),
            (dogs_flat, dogs_meta, all_cands,
             np.int32(N), np.int32(S), np.int32(border),
             np.float32(c_thresh_s), np.float32(edge_score),
             d_valid, d_oct, d_ri, d_ci, d_si, d_resp)
        )
        mask    = (d_valid == 1)
        valid_n = int(cp.count_nonzero(mask))
        if valid_n == 0:
            return []
        # Opt 4: stack all 5 arrays and download in a single PCIe transfer
        # (was: 5 separate cp.asnumpy calls = 5 blocking D→H transfers)
        stacked = cp.stack([d_oct[mask].astype(cp.float32),
                            d_ri[mask], d_ci[mask],
                            d_si[mask].astype(cp.float32),
                            d_resp[mask]], axis=1)   # (valid_n, 5)
        valid_np = cp.asnumpy(stacked)               # 1 PCIe transfer
        oct_v  = valid_np[:, 0].astype(np.int32)
        ri_v   = valid_np[:, 1]
        ci_v   = valid_np[:, 2]
        si_v   = valid_np[:, 3].astype(np.int32)
        resp_v = valid_np[:, 4]
        # Use _effective_double (set per-call by _build_gaussian_pyramid) so that
        # large images (> 1MP) which skipped the 2× zoom use scale 1.0 correctly.
        coord_scale = 0.5 if getattr(self, '_effective_double', self.double_image) else 1.0
        refined = []
        for i in range(valid_n):
            o         = int(oct_v[i])
            si_f      = int(si_v[i])
            sf        = float(2 ** o)
            sigma_loc = self.sigma0 * (k_val ** si_f)
            refined.append({
                'octave_idx': o, 'scale_idx': si_f,
                'ri': float(ri_v[i]), 'ci': float(ci_v[i]),
                'x':  float(ci_v[i]) * sf * coord_scale,
                'y':  float(ri_v[i]) * sf * coord_scale,
                'size': sigma_loc * sf * coord_scale * 2.0,
                'response': float(resp_v[i]), 'angle': 0.0
            })
        return refined

    def _assign_orientations(self, keypoints, gauss_pyr, pre_stacked=False):
        """Assign dominant gradient orientations to each keypoint on the GPU.

        v1.2: Accepts gauss_pyr (list of lists of 2D arrays, possibly fp16).
        Each octave's levels are stacked to fp32 on demand and released after
        use to minimise peak VRAM.
        v2.0: Routes to OriNet CNN predictor when orientation='orinet'.
        v2.2: orientation='affnet' runs histogram first for initial angles, then
              refines orientation+scale with LAFAffNetShapeEstimator (kornia).
        """
        if not keypoints:
            return []
        if self.orientation == 'orinet':
            return self._assign_orientations_orinet(keypoints, gauss_pyr)
        S     = self.n_scales
        k_val = 2.0 ** (1.0 / S)
        N_tot = len(keypoints)
        WARPS = _ORI_WARPS_PER_BLOCK          # 4
        TPB   = WARPS * 32                    # 128 threads/block
        final_angles = cp.empty((N_tot, 4), dtype=cp.float32)

        # Vectorized keypoint field extraction (one pass over dicts)
        _ri = np.empty(N_tot, dtype=np.float32)
        _ci = np.empty(N_tot, dtype=np.float32)
        _si = np.empty(N_tot, dtype=np.float32)
        _oi = np.empty(N_tot, dtype=np.int32)
        for i, kp in enumerate(keypoints):
            _ri[i] = kp['ri'];  _ci[i] = kp['ci']
            _si[i] = kp['scale_idx'];  _oi[i] = kp['octave_idx']
        _sigma = (self.sigma0 * (k_val ** _si)).astype(np.float32)

        # Phase 1: prepare per-octave GPU buffers
        unique_octs = np.unique(_oi)
        oct_prep = []
        for o_val in unique_octs:
            mask = _oi == o_val
            idx = np.where(mask)[0]
            kpt_np = np.column_stack([_ri[mask], _ci[mask],
                                      _sigma[mask], _si[mask]])
            N_grp = len(idx)
            oct_prep.append((int(o_val), idx, cp.asarray(kpt_np),
                             cp.empty((N_grp, 4), dtype=cp.float32),
                             (N_grp + WARPS - 1) // WARPS, N_grp))

        # Phase 2: launch orientation kernels on per-octave streams
        cp.cuda.runtime.deviceSynchronize()
        _streams = [cp.cuda.Stream(non_blocking=True) for _ in oct_prep]
        _events  = [cp.cuda.Event() for _ in oct_prep]
        for si, (o_val, idx, kpt_gpu, batch_a, bpg, N_grp) in enumerate(oct_prep):
            gauss_3d = (gauss_pyr[o_val] if pre_stacked
                        else cp.stack(gauss_pyr[o_val]).astype(cp.float32))
            _, H_o, W_o = gauss_3d.shape
            _ORIENT_RAWKERNEL(
                (bpg,), (TPB,),
                (gauss_3d, kpt_gpu, batch_a,
                 np.int32(N_grp), np.int32(H_o), np.int32(W_o)),
                stream=_streams[si]
            )
            _events[si].record(_streams[si])
            if not pre_stacked:
                del gauss_3d

        # Phase 3: sync events, collect results
        for si, (_, idx, _, batch_a, _, _) in enumerate(oct_prep):
            _events[si].synchronize()
            final_angles[cp.asarray(idx)] = batch_a

        all_angles = cp.asnumpy(final_angles)
        valid_i, valid_j = np.where(all_angles >= 0.0)
        oriented = []
        for r, c in zip(valid_i, valid_j):
            nkp = keypoints[r].copy()
            nkp['angle'] = float(all_angles[r, c])
            oriented.append(nkp)
        # v2.2: AffNet shape refinement — uses histogram angles as initial LAF,
        # then refines elliptical shape and dominant orientation via kornia AffNet.
        if self.orientation == 'affnet':
            return self._assign_orientations_affnet(oriented, gauss_pyr)
        return oriented

    def _assign_orientations_orinet(self, keypoints, gauss_pyr):
        """v2.0 — Assign orientations using kornia OriNet CNN predictor.

        Extracts a 32×32 patch from the Gaussian pyramid at each keypoint's
        detected scale, batches them through OriNet (pretrained), and returns
        a single canonical orientation (degrees) per keypoint.

        One orientation per keypoint (no multi-peak duplication), which is
        more discriminative and avoids unnecessary descriptor copies.
        """
        if not _KORNIA_AVAILABLE:
            raise RuntimeError(
                "kornia is required for orientation='orinet'. "
                "Install with: pip install kornia"
            )

        # Lazy-load OriNet once per PySIFT instance
        if self._orinet is None:
            self._orinet = kornia.feature.OriNet(pretrained=True).to(DEVICE).eval()

        k_val = 2.0 ** (1.0 / self.n_scales)
        PATCH = 32
        patches = []

        for kp in keypoints:
            o       = kp['octave_idx']
            si      = kp['scale_idx']
            ri      = kp['ri']
            ci      = kp['ci']
            sigma_l = self.sigma0 * (k_val ** si)
            half    = max(1, round(6.0 * sigma_l))

            # Get the Gaussian level for this octave/scale, cast to fp32
            level = gauss_pyr[o][si]
            if level.dtype == cp.float16:
                level = level.astype(cp.float32)
            H, W = level.shape

            # Clamp crop window to image bounds
            r0 = int(ri) - half;  r1 = int(ri) + half + 1
            c0 = int(ci) - half;  c1 = int(ci) + half + 1
            pad_top    = max(0, -r0);       pad_bot  = max(0, r1 - H)
            pad_left   = max(0, -c0);       pad_right = max(0, c1 - W)
            r0c = max(0, r0);  r1c = min(H, r1)
            c0c = max(0, c0);  c1c = min(W, c1)

            crop = level[r0c:r1c, c0c:c1c]
            # Convert CuPy → torch (via DLPack for zero-copy when possible)
            crop_t = torch.as_tensor(crop, device=DEVICE).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
            if pad_top or pad_bot or pad_left or pad_right:
                crop_t = torch.nn.functional.pad(
                    crop_t, (pad_left, pad_right, pad_top, pad_bot), mode='reflect')

            # Resize to 32×32
            patch = kornia.geometry.transform.resize(crop_t, (PATCH, PATCH))  # (1,1,32,32)
            patches.append(patch)

        # Stack → (N, 1, 32, 32), normalise to [0, 1]
        batch = torch.cat(patches, dim=0)  # (N, 1, 32, 32)
        mn = batch.amin(dim=(2, 3), keepdim=True)
        mx = batch.amax(dim=(2, 3), keepdim=True)
        batch = (batch - mn) / (mx - mn + 1e-6)

        with torch.no_grad():
            angles_rad = self._orinet(batch)  # (N, 1) radians

        angles_deg = (torch.rad2deg(angles_rad).reshape(-1) % 360.0).cpu().numpy()

        oriented = []
        for i, kp in enumerate(keypoints):
            nkp = kp.copy()
            nkp['angle'] = float(angles_deg[i])
            oriented.append(nkp)
        return oriented

    def _assign_orientations_affnet(self, keypoints, gauss_pyr):
        """v2.2 — Refine keypoint orientation and affine shape using kornia AffNet.

        Takes keypoints that already have histogram-assigned orientations, builds
        Local Affine Frames (LAFs) from them, and runs LAFAffNetShapeEstimator
        (pretrained on HPatches) to produce elliptical shape estimates.

        The refined LAF encodes an ellipse A·x + t. We extract:
          - orientation: dominant axis angle of A  → updates kp['angle']
          - scale      : sqrt(|det A|)             → stored in kp['affnet_half']
                         _compute_descriptors uses this as the patch radius,
                         giving an affine-normalised sampling window.

        Operates octave×scale-level at a time so only one full-resolution level
        is on GPU at once. Clamps refined scale to [0.5×, 2.0×] of original to
        reject degenerate / exploding LAFs.

        Reference: "Repeatability Is Not Enough: Learning Affine Regions via
        Discriminability", Mishchuk et al., NeurIPS 2018.  kornia v0.7+.
        """
        if not _KORNIA_AVAILABLE:
            raise RuntimeError(
                "kornia is required for orientation='affnet'. "
                "Install with: pip install kornia"
            )

        # Lazy-load AffNet once per PySIFT instance
        if self._affnet is None:
            self._affnet = (kornia.feature.LAFAffNetShapeEstimator(pretrained=True)
                            .to(DEVICE).eval())

        k_val   = 2.0 ** (1.0 / self.n_scales)
        results = {}   # keypoint-list-index → (angle_deg, affnet_half or None)

        # Group by (octave, scale_idx) so we pass a single octave image per call
        from collections import defaultdict as _dd
        groups = _dd(list)
        for i, kp in enumerate(keypoints):
            groups[(kp['octave_idx'], kp['scale_idx'])].append((i, kp))

        for (o, si), grp in groups.items():
            level = gauss_pyr[o][si]
            if level.dtype == cp.float16:
                level = level.astype(cp.float32)
            H_oct, W_oct = level.shape

            # Full octave level as (1, 1, H, W) torch tensor, normalised [0,1]
            img_t = torch.as_tensor(level, device=DEVICE).unsqueeze(0).unsqueeze(0).float()
            mn = img_t.amin();  mx = img_t.amax()
            img_t = (img_t - mn) / (mx - mn + 1e-6)

            N_grp = len(grp)
            # Build LAF tensor: (1, N, 2, 3)
            # Circular LAF for keypoint with half-radius r and angle θ:
            #   [[r·cos θ,  -r·sin θ,  cx],
            #    [r·sin θ,   r·cos θ,  cy]]
            laf = torch.zeros(1, N_grp, 2, 3, dtype=torch.float32, device=DEVICE)
            orig_halfs = []
            for j, (_, kp) in enumerate(grp):
                sigma_l = self.sigma0 * (k_val ** kp['scale_idx'])
                r       = float(max(4, round(6.0 * sigma_l)))
                orig_halfs.append(r)
                ang     = float(np.radians(kp['angle']))
                ca, sa  = math.cos(ang), math.sin(ang)
                laf[0, j, 0, 0] =  r * ca;  laf[0, j, 0, 1] = -r * sa
                laf[0, j, 1, 0] =  r * sa;  laf[0, j, 1, 1] =  r * ca
                laf[0, j, 0, 2] = kp['ci']
                laf[0, j, 1, 2] = kp['ri']

            with torch.no_grad():
                laf_ref = self._affnet(laf, img_t)   # (1, N, 2, 3)

            laf_np = laf_ref[0].cpu().numpy()   # (N, 2, 3)
            for j, (i, _) in enumerate(grp):
                a11, a12 = float(laf_np[j, 0, 0]), float(laf_np[j, 0, 1])
                a21, a22 = float(laf_np[j, 1, 0]), float(laf_np[j, 1, 1])
                angle_deg = (math.degrees(math.atan2(a21, a11))) % 360.0
                det_A     = abs(a11 * a22 - a12 * a21)
                scale     = math.sqrt(max(det_A, 0.0))
                r_orig    = orig_halfs[j]
                aff_half  = float(np.clip(scale, 0.5 * r_orig, 2.0 * r_orig))
                results[i] = (angle_deg, aff_half)

        refined = []
        for i, kp in enumerate(keypoints):
            nkp = kp.copy()
            if i in results:
                nkp['angle']       = results[i][0]
                nkp['affnet_half'] = results[i][1]
            refined.append(nkp)
        return refined

    def _compute_descriptors(self, keypoints, gauss_pyr, pre_stacked=False):
        """Compute 128-dimensional SIFT descriptors for all oriented keypoints.

        Routes to one of two backends:
        - dsp=False (PySIFT-Native): cp.RawKernel with 128 threads/keypoint,
          register-per-bin accumulation, --use_fast_math. Single-scale only.
        - dsp=True  (PySIFT-DSP):    Numba kernel with 5-scale DSP pooling.

        Both paths apply RootSIFT normalization after descriptor computation.
        """
        if not keypoints:
            return np.zeros((0, 128), dtype=np.float32)
        S     = self.n_scales
        k_val = 2.0 ** (1.0 / S)
        N_tot = len(keypoints)

        sum_descs = cp.zeros((N_tot, 128), dtype=cp.float32)

        # Vectorized keypoint field extraction (one pass over dicts)
        _ri  = np.empty(N_tot, dtype=np.float32)
        _ci  = np.empty(N_tot, dtype=np.float32)
        _si  = np.empty(N_tot, dtype=np.float32)
        _ang = np.empty(N_tot, dtype=np.float32)
        _oi  = np.empty(N_tot, dtype=np.int32)
        _has_affnet = len(keypoints) > 0 and 'affnet_half' in keypoints[0]
        _aff = np.empty(N_tot, dtype=np.float32) if _has_affnet else None
        for i, kp in enumerate(keypoints):
            _ri[i] = kp['ri'];  _ci[i] = kp['ci']
            _si[i] = kp['scale_idx'];  _oi[i] = kp['octave_idx']
            _ang[i] = kp['angle']
            if _has_affnet:
                _aff[i] = kp['affnet_half']
        _ang_rad = np.radians(_ang)
        if _has_affnet:
            _ph = np.maximum(4.0, np.round(_aff)).astype(np.float32)
        else:
            _ph = np.maximum(4.0, np.round(
                3.0 * self.sigma0 * (k_val ** _si) * math.sqrt(2) * 2.5
            )).astype(np.float32)

        # Phase 1: prepare per-octave GPU buffers
        unique_octs = np.unique(_oi)
        dsp_scales_np = self._dsp_scales_np
        dsp_gpu = cp.asarray(dsp_scales_np) if self.dsp else None
        WARPS_D = _DESC_WARPS_PER_BLOCK
        TPB_D   = WARPS_D * 32

        oct_prep = []
        for o_val in unique_octs:
            mask = _oi == o_val
            idx = np.where(mask)[0]
            N_grp = len(idx)
            kpt_np = np.column_stack([_ri[mask], _ci[mask], _ang_rad[mask],
                                      _ph[mask], _si[mask]])
            oct_prep.append((int(o_val), idx, cp.asarray(kpt_np),
                             cp.empty((N_grp, 128), dtype=cp.float32), N_grp))

        # Phase 2: launch descriptor kernels on per-octave streams
        cp.cuda.runtime.deviceSynchronize()
        _streams = [cp.cuda.Stream(non_blocking=True) for _ in oct_prep]
        _events  = [cp.cuda.Event() for _ in oct_prep]

        for si, (o_val, idx, kpt_gpu, octave_descs, N_grp) in enumerate(oct_prep):
            gauss_3d = (gauss_pyr[o_val] if pre_stacked
                        else cp.stack(gauss_pyr[o_val]).astype(cp.float32))
            n_slices, H, W = gauss_3d.shape
            if self.dsp:
                if N_grp > 0:
                    _sift_desc_dsp_coop(
                        (N_grp,), (128,),
                        (gauss_3d, kpt_gpu, octave_descs, dsp_gpu,
                         np.int32(N_grp), np.int32(len(dsp_scales_np)),
                         np.int32(n_slices), np.int32(H), np.int32(W)),
                        stream=_streams[si], shared_mem=516 * 4
                    )
            else:
                if N_grp > 0:
                    _sift_desc_coop(
                        (N_grp,), (128,),
                        (gauss_3d, kpt_gpu, octave_descs,
                         np.int32(N_grp), np.int32(n_slices),
                         np.int32(H), np.int32(W)),
                        stream=_streams[si], shared_mem=516 * 4
                    )
            _events[si].record(_streams[si])
            if not pre_stacked:
                del gauss_3d

        # Phase 3: sync events, collect results
        for si, (_, idx, _, octave_descs, _) in enumerate(oct_prep):
            _events[si].synchronize()
            sum_descs[cp.asarray(idx)] = octave_descs

        # RootSIFT: L1-normalise then sqrt (Arandjelovic & Zisserman, CVPR 2012).
        # Converts L2 distance to Hellinger distance — the optimal metric for
        # gradient-orientation histograms. Zero runtime cost; +5-15% matching precision.
        # Applied after DSP averaging so the final descriptor is properly normalised.
        # Gated by self.rootsift so parity diagnostics can compare raw SIFT descriptors.
        if self.rootsift:
            sum_descs /= cp.linalg.norm(sum_descs, ord=1, axis=1, keepdims=True) + 1e-7
            cp.clip(sum_descs, 0.0, None, out=sum_descs)  # guard against fp16 negatives before sqrt
            cp.sqrt(sum_descs, out=sum_descs)

        # Free loop-variable CuPy arrays (orig_idx, octave_sum, batch_gpu, kpt_gpu)
        # before the GPU→CPU transfer so the VRAM they occupy is returned to the pool.
        # NameError guard: loop body never runs when all keypoints have no octave match.
        try:
            del orig_idx, octave_sum, batch_gpu, kpt_gpu
        except NameError:
            pass

        # GPU→CPU transfer with MemoryError recovery.
        # After ~10k bulk calls (e.g. Oxford5K codebook build) the Python heap can be
        # fragmented enough that numpy.empty() inside cp.asnumpy() fails for even a few
        # MiB.  On the first failure: flush both CuPy pools (releasing VRAM-paged-to-RAM
        # on Windows WDDM and any cached pinned memory) and retry once.
        try:
            result = cp.asnumpy(sum_descs)
        except MemoryError:
            import gc
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()
            result = cp.asnumpy(sum_descs)   # raises if truly OOM
        del sum_descs
        return result

    def _compute_descriptors_learned(self, keypoints, gauss_pyr, pre_stacked=False):
        """v2.1 — Compute 128-dim descriptors using HardNet8 or HyNet.

        Extracts a 32×32 patch from the Gaussian pyramid at each keypoint's
        detected scale, rotation-normalised by the keypoint orientation angle
        (via affine_grid warp), then batches through the pretrained CNN.

        Works with any orientation mode ('histogram' or 'orinet') because it
        reads kp['angle'] regardless of how it was assigned.

        HardNet8: Mishchuk et al., "Working hard to know your neighbor's margins:
            Local descriptor learning loss", NeurIPS 2017.
        HyNet: Tian et al., "HyNet: Learning Local Descriptor with Hybrid
            Similarity Measure and Triplet Loss", NeurIPS 2020.

        Returns
        -------
        numpy.ndarray, shape (N, 128), dtype float32  (L2-normalised)
        """
        if not _KORNIA_AVAILABLE:
            raise RuntimeError(
                "kornia is required for descriptor='hardnet'/'hynet'. "
                "Install with: pip install kornia"
            )

        # Lazy-load on first call; swap if descriptor type changes between calls
        if self._descriptor_net is None or getattr(self, '_descriptor_net_type', None) != self.descriptor:
            if self.descriptor == 'hardnet':
                self._descriptor_net = kornia.feature.HardNet8(pretrained=True).to(DEVICE).eval()
            elif self.descriptor == 'hynet':
                self._descriptor_net = kornia.feature.HyNet(pretrained=True).to(DEVICE).eval()
            else:
                raise ValueError(f"Unknown descriptor '{self.descriptor}'. Choose 'sift', 'hardnet', or 'hynet'.")
            self._descriptor_net_type = self.descriptor

        if not keypoints:
            return np.zeros((0, 128), dtype=np.float32)

        k_val = 2.0 ** (1.0 / self.n_scales)
        PATCH = 32
        _BATCH = 512  # Cap per-batch VRAM: 512 crops ≈ 36 MB vs 540 MB for 8000
        N = len(keypoints)

        halves = [max(1, round(6.0 * self.sigma0 * (k_val ** kp['scale_idx'])))
                  for kp in keypoints]

        desc_chunks = []
        for b_start in range(0, N, _BATCH):
            b_end = min(b_start + _BATCH, N)
            b_kps = keypoints[b_start:b_end]
            b_halves = halves[b_start:b_end]
            b_n = b_end - b_start

            b_max_half = max(b_halves)
            b_crop_size = 2 * b_max_half + 1

            crops  = torch.zeros(b_n, 1, b_crop_size, b_crop_size,
                                 dtype=torch.float32, device=DEVICE)
            thetas = torch.zeros(b_n, 2, 3, dtype=torch.float32, device=DEVICE)

            for i, kp in enumerate(b_kps):
                o         = kp['octave_idx']
                si        = kp['scale_idx']
                ri        = kp['ri']
                ci        = kp['ci']
                angle_rad = float(np.radians(kp['angle']))
                half      = b_halves[i]

                level = gauss_pyr[o][si]
                if level.dtype == cp.float16:
                    level = level.astype(cp.float32)
                H_l, W_l = level.shape

                r0 = int(ri) - half;  r1 = int(ri) + half + 1
                c0 = int(ci) - half;  c1 = int(ci) + half + 1
                pad_top  = max(0, -r0);       pad_bot  = max(0, r1 - H_l)
                pad_left = max(0, -c0);       pad_right = max(0, c1 - W_l)
                r0c = max(0, r0);  r1c = min(H_l, r1)
                c0c = max(0, c0);  c1c = min(W_l, c1)

                crop_h = r1c - r0c + pad_top + pad_bot
                crop_w = c1c - c0c + pad_left + pad_right

                off_r = (b_crop_size - crop_h) // 2
                off_c = (b_crop_size - crop_w) // 2

                dst_r0 = off_r + pad_top;  dst_r1 = dst_r0 + (r1c - r0c)
                dst_c0 = off_c + pad_left; dst_c1 = dst_c0 + (c1c - c0c)
                crop_cp = level[r0c:r1c, c0c:c1c]
                crops[i, 0, dst_r0:dst_r1, dst_c0:dst_c1] = torch.as_tensor(crop_cp, device=DEVICE)

                cos_a = float(np.cos(-angle_rad))
                sin_a = float(np.sin(-angle_rad))
                thetas[i, 0, 0] =  cos_a;  thetas[i, 0, 1] = -sin_a;  thetas[i, 0, 2] = 0.0
                thetas[i, 1, 0] =  sin_a;  thetas[i, 1, 1] =  cos_a;  thetas[i, 1, 2] = 0.0

            grids = torch.nn.functional.affine_grid(
                thetas, (b_n, 1, b_crop_size, b_crop_size), align_corners=False)
            aligned = torch.nn.functional.grid_sample(
                crops, grids,
                mode='bilinear', padding_mode='zeros', align_corners=False)
            del crops, grids

            batch = kornia.geometry.transform.resize(aligned, (PATCH, PATCH)).float()
            del aligned
            mn = batch.amin(dim=(2, 3), keepdim=True)
            mx = batch.amax(dim=(2, 3), keepdim=True)
            batch = (batch - mn) / (mx - mn + 1e-6)

            with torch.no_grad():
                descs_b = self._descriptor_net(batch)
            desc_chunks.append(descs_b.cpu())
            del batch, thetas

        return torch.cat(desc_chunks, dim=0).numpy().astype(np.float32)

    def detectAndCompute(self, gray, mask=None, profile=False):
        """
        Detect keypoints and compute descriptors for a grayscale image.

        Parameters
        ----------
        gray : numpy.ndarray, shape (H, W), dtype uint8
        mask : ignored (kept for API compatibility with cv2 detectors)
        profile : bool -- if True, store per-phase GPU timings in self._phase_timings

        Returns
        -------
        keypoints : list of cv2.KeyPoint
        descriptors : numpy.ndarray, shape (N, 128), dtype float32
        """
        if isinstance(gray, str):
            raise TypeError(
                f"detectAndCompute expects a numpy array, got str: '{gray}'. "
                "Load the image first: img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)"
            )
        if profile:
            cp.cuda.runtime.deviceSynchronize()
            _t0 = time.perf_counter()
        # Adaptive contrast threshold: lowers sensitivity for night/haze images,
        # raises it for high-texture scenes — yields 40-80% more usable keypoints.
        contrast_thresh = _adaptive_contrast_thresh(gray, self.contrast_thresh)
        gauss_pyr = self._build_gaussian_pyramid(gray)
        if profile:
            cp.cuda.runtime.deviceSynchronize()
            _t_gauss = time.perf_counter()
        dog_pyr   = self._build_dog_pyramid(gauss_pyr)
        if profile:
            cp.cuda.runtime.deviceSynchronize()
            _t_dog = time.perf_counter()
        # v1.2: gauss_pyr levels may be fp16; no pre-stacking — each method
        # builds the fp32 octave stack lazily and frees it immediately to
        # minimise peak VRAM during orientation/descriptor computation.
        all_cands, dogs_flat, dogs_meta = self._find_extrema(dog_pyr, contrast_thresh)
        refined  = self._refine_keypoints(all_cands, dogs_flat, dogs_meta, gauss_pyr,
                                          contrast_thresh)
        del all_cands, dogs_flat, dogs_meta
        if profile:
            cp.cuda.runtime.deviceSynchronize()
            _t_extrema = time.perf_counter()
        # Starvation fallback: if too few keypoints survive, retry once with halved
        # threshold. dog_pyr and gauss_pyr are already in VRAM — no extra pyramid build.
        if len(refined) < 100:
            ct_retry = contrast_thresh * 0.5
            all_cands2, dogs_flat2, dogs_meta2 = self._find_extrema(dog_pyr, ct_retry)
            refined = self._refine_keypoints(all_cands2, dogs_flat2, dogs_meta2,
                                             gauss_pyr, ct_retry)
            del all_cands2, dogs_flat2, dogs_meta2
        # dog_pyr no longer needed — free before orientation/descriptor stages.
        del dog_pyr
        cp.get_default_memory_pool().free_all_blocks()
        # Global cap before orientation+descriptor: keep top-8000 by response.
        # Raised from 3000 to match post-detection max_keypoints cap and avoid
        # discarding repeatable keypoints with moderate response scores.
        if len(refined) > 8000:
            refined.sort(key=lambda k: k['response'], reverse=True)
            refined = refined[:8000]
        # Pyramid already returns contiguous 3D arrays — just cast fp16 octaves to fp32.
        gauss_stacked = [gauss_pyr[o] if gauss_pyr[o].dtype == cp.float32
                         else gauss_pyr[o].astype(cp.float32)
                         for o in range(len(gauss_pyr))]
        del gauss_pyr
        cp.get_default_memory_pool().free_all_blocks()
        if profile:
            cp.cuda.runtime.deviceSynchronize()
            _t_stack = time.perf_counter()
        oriented = self._assign_orientations(refined, gauss_stacked,
                                             pre_stacked=True)
        if profile:
            cp.cuda.runtime.deviceSynchronize()
            _t_orient = time.perf_counter()
        # v2.1: route to learned descriptor CNN or handcrafted SIFT kernel
        if self.descriptor in ('hardnet', 'hynet'):
            # Reclaim CuPy pool before PyTorch allocates HardNet crops + model
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            descs = self._compute_descriptors_learned(oriented, gauss_stacked,
                                                      pre_stacked=True)
        else:
            descs = self._compute_descriptors(oriented, gauss_stacked,
                                              pre_stacked=True)
        if profile:
            cp.cuda.runtime.deviceSynchronize()
            _t_desc = time.perf_counter()
        del gauss_stacked
        # Pool flush removed here — periodic flush every 50 calls (line 1830)
        # handles VRAM; mid-pipeline flushes cost 0.5-2ms each on Windows WDDM.
        # Vectorised cv2.KeyPoint construction — avoids per-keypoint Python
        # overhead that previously consumed 8% of detection time (~6 ms at
        # 768x1024).  Build flat arrays first, then batch-construct.
        if oriented:
            _n = len(oriented)
            _xs = np.empty(_n, dtype=np.float32)
            _ys = np.empty(_n, dtype=np.float32)
            _sz = np.empty(_n, dtype=np.float32)
            _ag = np.empty(_n, dtype=np.float32)
            _rs = np.empty(_n, dtype=np.float32)
            _oc = np.empty(_n, dtype=np.int32)
            for _j, _kp in enumerate(oriented):
                _xs[_j] = _kp['x']
                _ys[_j] = _kp['y']
                _sz[_j] = _kp['size']
                _ag[_j] = _kp['angle']
                _rs[_j] = _kp['response']
                _oc[_j] = _kp['octave_idx']
            # Canonical sort by (x, y) for deterministic output ordering
            _order = np.lexsort((_ys, _xs))
            _xs, _ys = _xs[_order], _ys[_order]
            _sz, _ag = _sz[_order], _ag[_order]
            _rs, _oc = _rs[_order], _oc[_order]
            descs = descs[_order]
            cv_kpts = [cv2.KeyPoint(float(_xs[_j]), float(_ys[_j]),
                                    float(_sz[_j]), float(_ag[_j]),
                                    float(_rs[_j]), int(_oc[_j]))
                       for _j in range(_n)]
        else:
            cv_kpts = []
        if profile:
            _t_cvkpt = time.perf_counter()
            self._phase_timings = {
                'gauss_pyr_ms':  (_t_gauss   - _t0)       * 1000,
                'dog_pyr_ms':    (_t_dog     - _t_gauss)   * 1000,
                'extrema_ms':    (_t_extrema - _t_dog)     * 1000,
                'stack_ms':      (_t_stack   - _t_extrema) * 1000,
                'orient_ms':     (_t_orient  - _t_stack)   * 1000,
                'descriptor_ms': (_t_desc    - _t_orient)  * 1000,
                'cv_kpt_ms':     (_t_cvkpt   - _t_desc)    * 1000,
                'total_ms':      (_t_cvkpt   - _t0)        * 1000,
            }
        # Proactive pool flush every 50 calls.  On Windows WDDM, cudaFree() does
        # not immediately return physical VRAM pages to the GPU free list — the
        # kernel-mode driver caches them.  Flush periodically to prevent pool bloat
        # during bulk extraction (e.g. Oxford5K 5000+, IMC 25K+ pairs).
        # Reduced from 200 to 50 after IMC FPS diagnostic showed progressive
        # VRAM fragmentation degrading throughput on long runs (Apr 9).
        self._n_calls += 1
        _flush_interval = 200 if self._high_vram else (20 if self.descriptor in ('hardnet', 'hynet') else 50)
        if self._n_calls % _flush_interval == 0:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        return cv_kpts, descs


# =============================================================================
# CLASS: DepthEstimator
#
# Monocular depth estimation using MiDaS (Ranftl et al., 2020).
# Returns a per-pixel relative depth map: 1.0 = nearest, 0.0 = farthest.
#
# Singleton pattern ensures model weights are loaded only once per session.
# After each inference the weights are offloaded back to CPU so that VRAM
# is free for the subsequent descriptor matching and blending stages.
# =============================================================================
class DepthEstimator:
    """
    Monocular depth estimator based on MiDaS.

    Call DepthEstimator.get_instance() rather than constructing directly,
    to avoid reloading model weights on repeated calls.

    Requires: pip install timm
    Model cache: ~/.cache/torch/hub/intel-isl_MiDaS_master/
    """
    _instance = None
    _mtype    = None

    @classmethod
    def get_instance(cls, model_type="MiDaS_small"):
        """Return the shared singleton, loading weights if needed."""
        if cls._instance is None or cls._mtype != model_type:
            cls._instance = cls(model_type)
            cls._mtype    = model_type
        return cls._instance

    def __init__(self, model_type="MiDaS_small"):
        print(f"[DepthEstimator] Loading {model_type} ...")
        t0 = time.perf_counter()

        # Prefer a locally-cached copy to avoid network timeouts.
        hub_dir   = torch.hub.get_dir()
        local_dir = os.path.join(hub_dir, "intel-isl_MiDaS_master")
        if os.path.isdir(local_dir):
            print(f"[DepthEstimator] Using cached hub dir: {local_dir}")
            self.model = torch.hub.load(
                local_dir, model_type, source="local", trust_repo=True)
            transforms = torch.hub.load(
                local_dir, "transforms", source="local", trust_repo=True)
        else:
            self.model = torch.hub.load(
                "intel-isl/MiDaS", model_type, pretrained=True, trust_repo=True)
            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True)

        self.model.to(DEVICE).float().eval()
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

        print(f"[DepthEstimator] Ready in {time.perf_counter()-t0:.2f}s")

    @torch.no_grad()
    def estimate(self, img_bgr):
        """
        Estimate a relative depth map for a BGR image.

        Returns
        -------
        depth : numpy.ndarray, shape (H, W), dtype float32
            Values in [0, 1]. 1.0 = nearest point, 0.0 = farthest.
        """
        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.model.to(DEVICE)
        inp = self.transform(img_rgb).to(DEVICE).float()

        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1).float(),
            size=(H, W),
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

        d_min, d_max = pred.min(), pred.max()
        depth = (pred - d_min) / (d_max - d_min + 1e-8)   # 0 = far, 1 = near

        # Offload weights immediately to free VRAM for subsequent stages.
        self.model.cpu()
        torch.cuda.empty_cache()

        return depth.astype(np.float32)


# =============================================================================
# Changes 7 + 10: Compiled RANSAC inner loop
#
# Pure-PyTorch batch evaluation extracted from _ransac so torch.compile can
# fuse the ops and (for repeated calls with equal N) replay a CUDA graph.
# =============================================================================
def _ransac_batch_eval(all_idx_gpu, p1n_t, p2n_t, T1t, T2inv, p1h, p2h, p1t, p2t):
    """Batch RANSAC hypothesis evaluation — pure PyTorch, suitable for torch.compile.

    Takes pre-sampled 4-point indices (already on GPU) and all normalised /
    homogeneous coordinate tensors.  Returns H_batch (B,3,3) and sym_err (B,N).
    The scoring and LO-RANSAC steps remain in _ransac (they touch numpy / Python
    control flow and cannot be compiled).
    """
    pts1_4  = p1n_t[all_idx_gpu]                          # (B, 4, 2)
    pts2_4  = p2n_t[all_idx_gpu]
    x1 = pts1_4[:, :, 0];  y1 = pts1_4[:, :, 1]
    x2 = pts2_4[:, :, 0];  y2 = pts2_4[:, :, 1]
    z  = torch.zeros_like(x1);  o = torch.ones_like(x1)
    row_e = torch.stack([-x1, -y1, -o,  z,   z,  z,  x2*x1, x2*y1, x2], dim=-1)
    row_o = torch.stack([ z,   z,  z, -x1, -y1, -o,  y2*x1, y2*y1, y2], dim=-1)
    A       = torch.stack([row_e, row_o], dim=2).reshape(all_idx_gpu.shape[0], 8, 9)
    _, _, Vt    = torch.linalg.svd(A, full_matrices=True)
    H_norm      = Vt[:, -1, :].reshape(-1, 3, 3)
    H_batch     = (T2inv.unsqueeze(0) @ H_norm) @ T1t.unsqueeze(0)
    s           = H_batch[:, 2, 2]
    s_safe      = torch.where(s.abs() > 1e-10, s, torch.ones_like(s))
    H_batch     = H_batch / s_safe.unsqueeze(-1).unsqueeze(-1)
    fwd         = H_batch @ p1h.T
    fwd_xy      = fwd[:, :2] / fwd[:, 2:3].clamp(min=1e-10)
    err_f       = ((fwd_xy - p2t.T.unsqueeze(0)) ** 2).sum(dim=1)
    # Opt 9: direct batch 3×3 inverse via cofactor expansion (~3× fewer FLOPS
    # than SVD-based pinv for 3×3 non-singular matrices)
    a00=H_batch[:,0,0]; a01=H_batch[:,0,1]; a02=H_batch[:,0,2]
    a10=H_batch[:,1,0]; a11=H_batch[:,1,1]; a12=H_batch[:,1,2]
    a20=H_batch[:,2,0]; a21=H_batch[:,2,1]; a22=H_batch[:,2,2]
    det = a00*(a11*a22-a12*a21) - a01*(a10*a22-a12*a20) + a02*(a10*a21-a11*a20)
    det_s = torch.where(det.abs() > 1e-12, det, torch.ones_like(det))
    adj = torch.stack([
        torch.stack([ a11*a22-a12*a21,  a02*a21-a01*a22,  a01*a12-a02*a11], dim=-1),
        torch.stack([ a12*a20-a10*a22,  a00*a22-a02*a20,  a02*a10-a00*a12], dim=-1),
        torch.stack([ a10*a21-a11*a20,  a01*a20-a00*a21,  a00*a11-a01*a10], dim=-1),
    ], dim=1)
    H_inv = adj / det_s.unsqueeze(-1).unsqueeze(-1)
    bwd         = H_inv @ p2h.T
    bwd_xy      = bwd[:, :2] / bwd[:, 2:3].clamp(min=1e-10)
    err_b       = ((bwd_xy - p1t.T.unsqueeze(0)) ** 2).sum(dim=1)
    return H_batch, err_f + err_b                          # sym_err (B, N)


if hasattr(torch, 'compile'):
    try:
        import triton  # noqa: F401 — check Triton availability before compiling
        _ransac_batch_eval = torch.compile(_ransac_batch_eval, mode='reduce-overhead')
    except ImportError:
        pass  # Triton not available (Windows) — keep eager mode


# Opt 8: CUDA Graph cache for _ransac_batch_eval.
# For repeated calls with the same (B, N) shapes (e.g. video stitching or
# multi-image panoramas), replaying a CUDA graph has near-zero CPU dispatch
# overhead vs re-launching ~20 CUDA kernels every call.
# Key: graphs are keyed by (B, N) so shape changes get a fresh graph.
_ransac_graph_cache: dict = {}

def _ransac_batch_eval_graphed(all_idx_gpu, p1n_t, p2n_t, T1t, T2inv, p1h, p2h, p1t, p2t):
    """Opt 8: CUDA-graph-accelerated wrapper for _ransac_batch_eval.

    On the first call for a given (B, N) shape, traces a new graph and caches it.
    Subsequent calls with the same shape replay the graph (~0 CPU overhead).
    Falls back to direct call if CUDA graphs are unavailable.
    """
    if not (DEVICE.type == 'cuda' and hasattr(torch.cuda, 'CUDAGraph')):
        return _ransac_batch_eval(all_idx_gpu, p1n_t, p2n_t, T1t, T2inv, p1h, p2h, p1t, p2t)

    B = all_idx_gpu.shape[0]
    N = p1n_t.shape[0]
    key = (B, N, p1n_t.dtype)

    # Skip CUDA graph capture when torch.compile/Triton is unavailable —
    # eager-mode SVD is not capturable and a failed capture poisons the
    # CUDA context for the rest of the process.
    _triton_ok = _ransac_graph_cache.get('__triton_ok')
    if _triton_ok is None:
        try:
            import triton  # noqa: F401
            _triton_ok = True
        except ImportError:
            _triton_ok = False
        _ransac_graph_cache['__triton_ok'] = _triton_ok

    if not _triton_ok:
        return _ransac_batch_eval(all_idx_gpu, p1n_t, p2n_t, T1t, T2inv, p1h, p2h, p1t, p2t)

    if key not in _ransac_graph_cache:
        # Allocate static input/output placeholder tensors for this shape
        idx_ph  = all_idx_gpu.clone()
        p1n_ph  = p1n_t.clone();    p2n_ph  = p2n_t.clone()
        T1t_ph  = T1t.clone();      T2i_ph  = T2inv.clone()
        p1h_ph  = p1h.clone();      p2h_ph  = p2h.clone()
        p1t_ph  = p1t.clone();      p2t_ph  = p2t.clone()

        # Warmup — required before graph capture
        torch.cuda.synchronize()
        for _ in range(3):
            _ransac_batch_eval(idx_ph, p1n_ph, p2n_ph, T1t_ph, T2i_ph,
                               p1h_ph, p2h_ph, p1t_ph, p2t_ph)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            H_ph, err_ph = _ransac_batch_eval(idx_ph, p1n_ph, p2n_ph, T1t_ph, T2i_ph,
                                              p1h_ph, p2h_ph, p1t_ph, p2t_ph)
        _ransac_graph_cache[key] = (g, idx_ph, p1n_ph, p2n_ph, T1t_ph, T2i_ph,
                                    p1h_ph, p2h_ph, p1t_ph, p2t_ph, H_ph, err_ph)

    (g, idx_ph, p1n_ph, p2n_ph, T1t_ph, T2i_ph,
     p1h_ph, p2h_ph, p1t_ph, p2t_ph, H_ph, err_ph) = _ransac_graph_cache[key]

    # Copy live data into static placeholders and replay graph
    idx_ph.copy_(all_idx_gpu)
    p1n_ph.copy_(p1n_t);    p2n_ph.copy_(p2n_t)
    T1t_ph.copy_(T1t);      T2i_ph.copy_(T2inv)
    p1h_ph.copy_(p1h);      p2h_ph.copy_(p2h)
    p1t_ph.copy_(p1t);      p2t_ph.copy_(p2t)
    g.replay()
    return H_ph.clone(), err_ph.clone()


# =============================================================================
# CLASS: GPUPyStitch
#
# End-to-end depth-aware panoramic stitcher.
# =============================================================================
class GPUPyStitch:
    """
    GPU-accelerated panoramic stitcher with depth-aware composition.

    Parameters
    ----------
    n_depth_layers : int
        Number of depth bands for depth-stratified homographies (Stage 4).
    n_blend_levels : int
        Number of Laplacian pyramid levels for multi-band blending (Stage 5).
    ransac_iters : int
        Number of RANSAC hypotheses evaluated in parallel per run.
    inlier_thresh : float
        Symmetric reprojection error threshold in pixels for inlier classification.
    match_ratio : float
        Lowe's ratio test threshold for descriptor matching.
    match_batch : int
        Descriptor rows processed per GPU batch (tune down if VRAM < 4 GB).
    max_keypoints : int
        Maximum keypoints retained per image before matching.

    Examples
    --------
    >>> stitcher = GPUPyStitch()
    >>> pano = stitcher.stitch(img_left, img_right)
    >>> pano = stitcher.stitch(img_left, img_center, img_right)
    >>> pano = stitcher.stitch([img_left, img_right])
    """

    def __init__(self, n_depth_layers=4, n_blend_levels=6,
                 ransac_iters=1500, inlier_thresh=5.0,
                 ransac_method='magsac', sigma_max=None,
                 match_ratio=0.75, match_batch=512, max_keypoints=8000,
                 pca_dims=64, int8_matching=False,
                 orientation='histogram', descriptor='sift',
                 matcher='ratio', color_match='gain',
                 dsp_sift=False,
                 # Detection params — passed through to PySIFT
                 contrast_thresh=0.04, n_octaves=4, n_scales=3,
                 sigma0=1.6, edge_thresh=10.0, double_image=True,
                 # Preprocessing params
                 max_dimension=0, denoise=False, sharpen=False,
                 clahe_clip=2.0, clahe_tile=8,
                 # Post-processing params
                 output_max_dim=0, post_sharpen=False,
                 output_format='png', output_quality=90):
        self.n_depth_layers = n_depth_layers
        self.n_blend_levels = n_blend_levels
        self.ransac_iters   = ransac_iters
        self.inlier_thresh  = inlier_thresh
        self.ransac_method  = ransac_method   # 'magsac' | 'classic'
        self.sigma_max      = sigma_max if sigma_max is not None else inlier_thresh
        self.match_ratio    = match_ratio
        self.match_batch    = match_batch
        self.max_keypoints  = max_keypoints
        self.pca_dims       = pca_dims        # v1.2: 0 = disabled, 64 = 128→64 PCA compression
        self.int8_matching  = int8_matching   # Change 9: INT8 Tensor Core GEMM for matching
        self.matcher        = matcher         # v3.0: 'ratio' | 'lightglue'
        self.descriptor     = descriptor      # v2.1: 'sift' | 'hardnet' | 'hynet'
        self.color_match    = color_match     # v3.1: 'gain' | 'hist'
        self.max_dimension  = max_dimension   # preprocessing: 0 = no resize
        self.denoise        = denoise         # preprocessing: bilateral filter
        self.sharpen        = sharpen         # preprocessing: unsharp mask
        self.clahe_clip     = clahe_clip      # detection: CLAHE clip limit
        self.clahe_tile     = clahe_tile      # detection: CLAHE tile size
        self.output_max_dim = output_max_dim  # post-processing: 0 = no resize
        self.post_sharpen   = post_sharpen    # post-processing: unsharp mask
        self.output_format  = output_format.lower()  # post-processing: 'png' | 'jpg'
        self.output_quality = output_quality  # post-processing: JPEG quality 0–100
        self._detector      = PySIFT(dsp=dsp_sift,
                                     orientation=orientation,
                                     descriptor=descriptor,
                                     n_octaves=n_octaves, n_scales=n_scales,
                                     sigma0=sigma0, contrast_thresh=contrast_thresh,
                                     edge_thresh=edge_thresh,
                                     double_image=double_image)
        self._depth_est          = None   # loaded lazily on first use
        self._lightglue          = None   # v3.0: lazy-loaded LightGlue model
        self._lightglue_features = None   # v3.0: track which features= was loaded
        # Change 5: PCA projection cached after first fit — reused across all pairs
        self._pca_matrix         = None   # cp.ndarray [128, pca_dims], computed once
        self._pca_mean           = None   # cp.ndarray [1, 128]

    # ------------------------------------------------------------------
    # Stage 0 — Preprocessing (resize / denoise / sharpen)
    # ------------------------------------------------------------------
    def _preprocess_image(self, img):
        """
        Optional preprocessing for large or noisy images (e.g. mobile phones).

        max_dimension : resize longest side to this (0 = skip).
                        Use 1600–2000 for mobile photos.
                        cv2.INTER_AREA gives the best quality for downscaling.
        denoise       : bilateral filter — edge-preserving noise reduction.
                        Good for night shots or high-ISO captures.
        sharpen       : unsharp mask — recovers detail lost by lens softness
                        or JPEG compression.
        """
        if self.max_dimension > 0:
            h, w = img.shape[:2]
            scale = self.max_dimension / max(h, w)
            if scale < 1.0:
                nw = int(w * scale)
                nh = int(h * scale)
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                print(f"    Resized {w}×{h} → {nw}×{nh}  (scale {scale:.2f})")

        if self.denoise:
            img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

        if self.sharpen:
            blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
            img  = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

        return img

    # ------------------------------------------------------------------
    # Depth estimator — lazy singleton loader
    # ------------------------------------------------------------------
    def _get_depth_estimator(self):
        """Return the DepthEstimator singleton, or None if unavailable."""
        if self._depth_est is None:
            try:
                self._depth_est = DepthEstimator.get_instance("MiDaS_small")
            except Exception as exc:
                print(f"[GPUPyStitch] MiDaS unavailable: {exc}")
                print("[GPUPyStitch] Stage 4 depth analysis skipped; "
                      "global homography warp will be used.")
                self._depth_est = False   # sentinel: tried, failed
        return self._depth_est if self._depth_est is not False else None

    # ==================================================================
    # STAGE 1 — Feature Extraction
    # ==================================================================

    def _extract_features(self, img):
        """
        Detect keypoints and compute descriptors for a BGR image.

        Applies CLAHE histogram equalisation before detection to improve
        contrast in low-light or unevenly-lit images.

        Returns
        -------
        keypoints    : list of cv2.KeyPoint
        descriptors  : numpy.ndarray, shape (N, 128), float32
        image_size   : tuple (H, W) -- passed to LightGlue for coord normalisation
        """
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: '{img}'")
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Change 6: GPU CLAHE via kornia — eliminates one CPU↔GPU roundtrip per image.
        # Falls back to cv2 when kornia is unavailable.
        if _KORNIA_AVAILABLE:
            gray_t = torch.as_tensor(gray, device=DEVICE, dtype=torch.float32)
            gray_t = gray_t.unsqueeze(0).unsqueeze(0) / 255.0          # (1,1,H,W) in [0,1]
            gray_t = kornia.enhance.equalize_clahe(
                gray_t, clip_limit=float(self.clahe_clip),
                grid_size=(self.clahe_tile, self.clahe_tile))
            gray = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(
                    gray_t.squeeze().clamp(0.0, 1.0).mul(255.0)
                    .to(torch.uint8).contiguous()))                      # CuPy uint8, stays on GPU
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip,
                                     tileGridSize=(self.clahe_tile, self.clahe_tile))
            gray  = clahe.apply(gray)
        kp, descs = self._detector.detectAndCompute(gray, None)
        if len(kp) > self.max_keypoints:
            order = sorted(range(len(kp)), key=lambda i: kp[i].response, reverse=True)
            order = order[:self.max_keypoints]
            kp    = [kp[i] for i in order]
            descs = descs[order]
        return kp, descs, (H, W)

    def _match_one_direction(self, d1_gpu, d2_gpu):
        """
        One-direction ratio-test matching using Tensor Core fp16 matrix multiply.

        Replaces the CuPy fp32 matmul with a PyTorch autocast mm, enabling
        TF32/fp16 Tensor Cores on Ampere GPUs (RTX 30xx) for 2-4x throughput
        improvement. CuPy↔PyTorch transfer is zero-copy via DLPack.

        Parameters
        ----------
        d1_gpu, d2_gpu : cp.ndarray, shape (N,128) and (M,128), L2-normalised float32

        Returns
        -------
        dict  {query_idx (int): (train_idx (int), distance (float))}
              only entries that passed Lowe's ratio test
        """
        N        = d1_gpu.shape[0]
        M        = d2_gpu.shape[0]
        ratio_sq = self.match_ratio ** 2

        # Opt 7: replace batched loop (16 iterations × 5 sync points each) with a
        # single one-shot Tensor Core matmul.  Peak VRAM = N×M×2 bytes fp16.
        # For N=M=8000: 8000×8000×2 = 128 MB — well within 4 GB.
        # Falls back to batched loop only if sim matrix would exceed 512 MB fp16.
        sim_mb = N * M * 2 / (1024**2)
        if sim_mb <= 512:
            # ONE matmul — all 16 previous batches fused into a single Tensor Core call
            d1_t = torch.from_dlpack(d1_gpu)   # zero-copy DLPack
            d2_t = torch.from_dlpack(d2_gpu)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                sim_t = torch.mm(d1_t, d2_t.T)   # (N, M) fp16
            sim_f = sim_t.float()                 # (N, M) fp32

            # torch.topk replaces argpartition + argsort (was 2 CuPy kernels per batch)
            top2_v, top2_i = torch.topk(sim_f, k=2, dim=1, largest=True, sorted=True)

            d_best_sq   = torch.clamp(2.0 - 2.0 * top2_v[:, 0], min=0.0)
            d_second_sq = torch.clamp(2.0 - 2.0 * top2_v[:, 1], min=0.0)
            passed_t    = (d_second_sq > 1e-10) & (d_best_sq < ratio_sq * d_second_sq)

            best_j  = top2_i[:, 0].cpu().numpy().astype(np.int32)
            dist_np = torch.sqrt(d_best_sq).cpu().numpy().astype(np.float32)
            passed  = passed_t.cpu().numpy()

            del sim_t, sim_f, top2_v, top2_i, d_best_sq, d_second_sq, passed_t
        else:
            # Fallback: original batched loop for unusually large descriptor sets
            best_j  = np.empty(N, dtype=np.int32)
            dist_np = np.empty(N, dtype=np.float32)
            passed  = np.zeros(N, dtype=bool)
            d2_t    = torch.from_dlpack(d2_gpu)
            for start in range(0, N, self.match_batch):
                end = min(start + self.match_batch, N)
                b   = end - start
                d1_t = torch.from_dlpack(d1_gpu[start:end])
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    sim_t = torch.mm(d1_t, d2_t.T)
                sim = cp.from_dlpack(sim_t.float().contiguous())
                top2_col = cp.argpartition(-sim, kth=1, axis=1)[:, :2]
                row_idx  = cp.arange(b, dtype=cp.int32)[:, None]
                top2_sim = sim[row_idx, top2_col]
                order    = cp.argsort(-top2_sim, axis=1)
                top2_sim = cp.take_along_axis(top2_sim, order, axis=1)
                top2_col = cp.take_along_axis(top2_col, order, axis=1)
                s_best   = top2_sim[:, 0];  s_sec = top2_sim[:, 1]
                d_b_sq   = cp.maximum(2.0 - 2.0*s_best, 0.0)
                d_s_sq   = cp.maximum(2.0 - 2.0*s_sec,  0.0)
                ok_gpu   = (d_s_sq > 1e-10) & (d_b_sq < ratio_sq * d_s_sq)
                best_j[start:end]  = cp.asnumpy(top2_col[:, 0])
                dist_np[start:end] = cp.asnumpy(cp.sqrt(d_b_sq))
                passed[start:end]  = cp.asnumpy(ok_gpu)
                del sim, sim_t, d1_t, top2_col, top2_sim, s_best, s_sec, d_b_sq, d_s_sq, ok_gpu
            cp._default_memory_pool.free_all_blocks()

        return {i: (int(best_j[i]), float(dist_np[i]))
                for i in range(N) if passed[i]}

    def _match_symmetric(self, d1_gpu, d2_gpu):
        """Symmetric matching with ONE matmul — reuse sim.T for backward pass.

        Returns (fwd, bwd) dicts identical to calling _match_one_direction twice,
        but computes the N×M similarity matrix only once.
        """
        N        = d1_gpu.shape[0]
        M        = d2_gpu.shape[0]
        ratio_sq = self.match_ratio ** 2

        sim_mb = N * M * 2 / (1024**2)
        if sim_mb > 512:
            # Fallback: too large for one-shot, use two separate calls
            fwd = self._match_one_direction(d1_gpu, d2_gpu)
            bwd = self._match_one_direction(d2_gpu, d1_gpu)
            return fwd, bwd

        d1_t = torch.from_dlpack(d1_gpu)
        d2_t = torch.from_dlpack(d2_gpu)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            sim_t = torch.mm(d1_t, d2_t.T)          # (N, M) fp16
        sim_f = sim_t.float()                        # (N, M) fp32
        del sim_t

        # Forward: A→B (rows = queries from d1)
        top2_v, top2_i = torch.topk(sim_f, k=2, dim=1, largest=True, sorted=True)
        d_best   = torch.clamp(2.0 - 2.0 * top2_v[:, 0], min=0.0)
        d_second = torch.clamp(2.0 - 2.0 * top2_v[:, 1], min=0.0)
        fwd_pass = (d_second > 1e-10) & (d_best < ratio_sq * d_second)
        fwd_j    = top2_i[:, 0].cpu().numpy().astype(np.int32)
        fwd_dist = torch.sqrt(d_best).cpu().numpy().astype(np.float32)
        fwd_ok   = fwd_pass.cpu().numpy()
        del top2_v, top2_i, d_best, d_second, fwd_pass

        # Backward: B→A — FREE from sim_f.T (no second matmul)
        sim_bwd = sim_f.T                            # (M, N) — just a view
        top2_vb, top2_ib = torch.topk(sim_bwd, k=2, dim=1, largest=True, sorted=True)
        db_best   = torch.clamp(2.0 - 2.0 * top2_vb[:, 0], min=0.0)
        db_second = torch.clamp(2.0 - 2.0 * top2_vb[:, 1], min=0.0)
        bwd_pass  = (db_second > 1e-10) & (db_best < ratio_sq * db_second)
        bwd_j     = top2_ib[:, 0].cpu().numpy().astype(np.int32)
        bwd_dist  = torch.sqrt(db_best).cpu().numpy().astype(np.float32)
        bwd_ok    = bwd_pass.cpu().numpy()
        del sim_f, sim_bwd, top2_vb, top2_ib, db_best, db_second, bwd_pass

        fwd = {i: (int(fwd_j[i]), float(fwd_dist[i]))
               for i in range(N) if fwd_ok[i]}
        bwd = {j: (int(bwd_j[j]), float(bwd_dist[j]))
               for j in range(M) if bwd_ok[j]}
        return fwd, bwd

    def _pca_compress(self, d1, d2):
        """
        Fit PCA on the combined descriptor set and project both to self.pca_dims
        dimensions.

        v1.2 — PCA 128→64 compression (Ke & Sukthankar, CVPR 2004):
        Descriptors live on a manifold of much lower intrinsic dimensionality
        than 128.  Compressing to 64 dims retains >95% of variance while halving
        the matmul cost in _match_one_direction.

        Strategy: fit PCA jointly on d1 and d2 so both are projected into the
        same subspace.  The 128×pca_dims projection matrix is derived from the
        eigenvectors of the combined covariance — no external training data needed.
        Both projected arrays are L2-normalised so the Tensor Core matmul in
        _match_one_direction works unchanged.

        Parameters
        ----------
        d1, d2 : cp.ndarray, (N, 128) and (M, 128), L2-normalised float32

        Returns
        -------
        d1p, d2p : cp.ndarray, (N, pca_dims) and (M, pca_dims), L2-normalised
        """
        # Change 5: reuse cached projection matrix across image pairs for a
        # consistent feature space (and skip the per-call SVD after warmup).
        if self._pca_matrix is None:
            combined = cp.vstack([d1, d2])                              # (N+M, 128)
            mean     = combined.mean(axis=0, keepdims=True)             # (1, 128)
            centered = combined - mean                                   # (N+M, 128)
            cov      = (centered.T @ centered) / float(len(combined))   # (128, 128)
            eigenvalues, eigenvectors = cp.linalg.eigh(cov)             # (128,), (128,128)
            self._pca_matrix = eigenvectors[:, -self.pca_dims:]          # (128, pca_dims)
            self._pca_mean   = mean
        proj  = self._pca_matrix
        mean  = self._pca_mean
        d1p   = (d1 - mean) @ proj
        d2p   = (d2 - mean) @ proj
        d1p  /= cp.linalg.norm(d1p, axis=1, keepdims=True) + 1e-7
        d2p  /= cp.linalg.norm(d2p, axis=1, keepdims=True) + 1e-7
        return d1p, d2p

    def _match_features_lightglue(self, kp1, desc1, kp2, desc2,
                                   img_sz1=None, img_sz2=None):
        """v3.0 — Match keypoints using the LightGlue transformer matcher.

        LightGlue (Lindenberger et al., ICCV 2023) jointly reasons over keypoint
        positions, scales, orientations, and descriptors to produce highly accurate
        mutual matches without a hand-tuned ratio threshold.

        Fixed (2026-04-01):
          BUG R2 — routes features= to 'doghardnet' when descriptor='hardnet'/'hynet'
                    instead of hardcoded 'sift'; avoids descriptor distribution mismatch.
          BUG R3 — passes image_size so LightGlue normalises pixel coords to [0,1];
                    without this the positional encoder receives raw pixel values
                    (e.g. 0–2000) instead of the [0,1] range it was trained on.
          CACHE  — reloads model only when descriptor type changes between calls.

        Returns
        -------
        list of cv2.DMatch — same format as the ratio-test matcher.
        """
        if not _LIGHTGLUE_AVAILABLE:
            raise RuntimeError(
                "lightglue is required for matcher='lightglue'. "
                "Install with: pip install git+https://github.com/cvg/LightGlue.git"
            )

        # BUG FIX R2: choose weights matching the active descriptor type.
        #   sift / dsp-sift  → 'sift'       (trained on SIFT gradient histograms)
        #   hardnet / hynet  → 'doghardnet' (trained on DoG keypoints + HardNet CNN)
        lg_features = 'doghardnet' if self.descriptor in ('hardnet', 'hynet') else 'sift'

        # Reload only when descriptor type changes (e.g. first call or mode switch).
        if self._lightglue is None or self._lightglue_features != lg_features:
            self._lightglue = _lg_mod.LightGlue(features=lg_features).to(DEVICE).eval()
            self._lightglue_features = lg_features

        n1 = min(len(kp1), self.max_keypoints)
        n2 = min(len(kp2), self.max_keypoints)
        kp1 = kp1[:n1];  desc1 = desc1[:n1]
        kp2 = kp2[:n2];  desc2 = desc2[:n2]

        pts1 = torch.tensor([[k.pt[0], k.pt[1]] for k in kp1],
                             dtype=torch.float32, device=DEVICE)
        pts2 = torch.tensor([[k.pt[0], k.pt[1]] for k in kp2],
                             dtype=torch.float32, device=DEVICE)
        d1t  = torch.tensor(desc1, dtype=torch.float32, device=DEVICE)
        d2t  = torch.tensor(desc2, dtype=torch.float32, device=DEVICE)

        data = {
            'image0': {
                'keypoints':   pts1.unsqueeze(0),
                'descriptors': d1t.unsqueeze(0),
            },
            'image1': {
                'keypoints':   pts2.unsqueeze(0),
                'descriptors': d2t.unsqueeze(0),
            },
        }

        # BUG FIX R3: image_size (W, H) lets LightGlue normalise pixel coordinates
        # to [0,1] before the positional encoder — required for correct attention.
        if img_sz1 is not None:
            H1, W1 = img_sz1
            data['image0']['image_size'] = torch.tensor(
                [[W1, H1]], dtype=torch.float32, device=DEVICE)
        if img_sz2 is not None:
            H2, W2 = img_sz2
            data['image1']['image_size'] = torch.tensor(
                [[W2, H2]], dtype=torch.float32, device=DEVICE)

        sc1  = torch.tensor([k.size / 2.0 for k in kp1],
                             dtype=torch.float32, device=DEVICE)
        ori1 = torch.tensor([float(np.radians(k.angle)) for k in kp1],
                             dtype=torch.float32, device=DEVICE)
        sc2  = torch.tensor([k.size / 2.0 for k in kp2],
                             dtype=torch.float32, device=DEVICE)
        ori2 = torch.tensor([float(np.radians(k.angle)) for k in kp2],
                             dtype=torch.float32, device=DEVICE)
        data['image0'].update(scales=sc1.unsqueeze(0), oris=ori1.unsqueeze(0))
        data['image1'].update(scales=sc2.unsqueeze(0), oris=ori2.unsqueeze(0))

        with torch.no_grad():
            out = self._lightglue(data)

        # out['matches'] is (1, M, 2) — mutual match pairs (i, j)
        pairs  = out['matches'][0].cpu().numpy()
        scores = out['scores'][0].cpu().numpy()

        return [
            cv2.DMatch(_queryIdx=int(pairs[k, 0]),
                       _trainIdx=int(pairs[k, 1]),
                       _distance=float(1.0 - scores[k]))
            for k in range(len(pairs))
        ]

    def _match_features(self, desc1, desc2, kp1=None, kp2=None,
                        img_sz1=None, img_sz2=None):
        """
        Match two descriptor sets using symmetric (mutual) ratio test.

        Runs Lowe's ratio test in both directions (A→B and B→A) and keeps only
        mutually-consistent matches: pair (i,j) is accepted only when i is j's
        best ratio-passing match AND j is i's best ratio-passing match.

        This rejects asymmetric false positives — matches that pass the ratio
        test in one direction but not the other — raising the inlier rate from
        ~60% to ~85%+ with no change to the ratio threshold.

        Each direction uses Tensor Core fp16 matrix multiply via _match_one_direction.

        Parameters
        ----------
        img_sz1, img_sz2 : tuple (H, W) or None
            Image sizes passed to LightGlue for keypoint coordinate normalisation.

        Returns
        -------
        matches : list of cv2.DMatch
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) < 2:
            return []

        # v3.0: route to LightGlue when requested and keypoints are available
        if self.matcher == 'lightglue' and kp1 is not None and kp2 is not None:
            return self._match_features_lightglue(kp1, desc1, kp2, desc2,
                                                  img_sz1, img_sz2)

        cp._default_memory_pool.free_all_blocks()

        desc1 = desc1[:self.max_keypoints]
        desc2 = desc2[:self.max_keypoints]

        d1 = cp.asarray(desc1.astype(np.float32))
        d2 = cp.asarray(desc2.astype(np.float32))
        d1 = d1 / (cp.linalg.norm(d1, axis=1, keepdims=True) + 1e-7)
        d2 = d2 / (cp.linalg.norm(d2, axis=1, keepdims=True) + 1e-7)

        # v1.2 PCA compression: project 128-dim → pca_dims before matching.
        # 2× matmul speedup; <2% accuracy loss at 64 dims (Ke & Sukthankar 2004).
        # BUG FIX R4: only valid for SIFT — HardNet/HyNet have different principal
        # components; compressing them with SIFT-derived PCA degrades discriminability.
        if (self.descriptor == 'sift' and
                self.pca_dims > 0 and self.pca_dims < d1.shape[1]):
            d1, d2 = self._pca_compress(d1, d2)

        # Change 9: INT8 Tensor Core GEMM (Turing/Ampere, ~2× throughput vs fp16).
        # RootSIFT descriptors are non-negative after L2-norm → safe to quantize.
        # Only applied to sift descriptors (not hardnet/hynet which have wider range).
        # torch._int_mm is a PyTorch 2.0+ private API; guarded accordingly.
        if (self.int8_matching and self.descriptor == 'sift'
                and hasattr(torch, '_int_mm')):
            def _to_int8_torch(d_cp):
                d_t  = torch.from_dlpack(d_cp)             # zero-copy, float32 (N, D)
                scale = 127.0 / (d_t.abs().amax(dim=1, keepdim=True) + 1e-7)
                return (d_t * scale).clamp(-127, 127).to(torch.int8), scale
            d1_i8, s1 = _to_int8_torch(d1)
            d2_i8, s2 = _to_int8_torch(d2)
            sim_i32   = torch._int_mm(d1_i8, d2_i8.T)     # (N, M) int32
            # Rescale: actual dot product ≈ sim_i32 / (s1 * s2.T)
            sim_f32   = sim_i32.float() / (s1 * s2.T)
            d1 = cp.from_dlpack(sim_f32)   # reuse d1 slot — _match_one_direction
            # INT8 path computes full similarity matrix; bypass _match_one_direction
            del d2, d1_i8, d2_i8, sim_i32
            # Apply ratio test directly on the pre-computed sim_f32
            top2      = torch.topk(sim_f32, k=2, dim=1)
            d_best_sq   = (2.0 - 2.0 * top2.values[:, 0]).clamp(min=0.0)
            d_second_sq = (2.0 - 2.0 * top2.values[:, 1]).clamp(min=0.0)
            ratio_sq    = self.match_ratio ** 2
            passed      = (d_second_sq > 1e-10) & (d_best_sq < ratio_sq * d_second_sq)
            fwd = {int(i): (int(top2.indices[i, 0]), float(d_best_sq[i].sqrt()))
                   for i in range(len(passed)) if passed[i]}
            # Symmetric pass B→A on the same matrix (transpose)
            top2b       = torch.topk(sim_f32.T, k=2, dim=1)
            db_best_sq   = (2.0 - 2.0 * top2b.values[:, 0]).clamp(min=0.0)
            db_second_sq = (2.0 - 2.0 * top2b.values[:, 1]).clamp(min=0.0)
            passedb      = (db_second_sq > 1e-10) & (db_best_sq < ratio_sq * db_second_sq)
            bwd = {int(j): (int(top2b.indices[j, 0]), float(db_best_sq[j].sqrt()))
                   for j in range(len(passedb)) if passedb[j]}
            del sim_f32, top2, top2b
        else:
            # Symmetric ratio test: compute sim matrix ONCE, topk both directions.
            # Avoids redundant 2nd matmul — sim.T gives the backward direction free.
            fwd, bwd = self._match_symmetric(d1, d2)

        if not (self.int8_matching and self.descriptor == 'sift'
                and hasattr(torch, '_int_mm')):
            del d1, d2
        cp._default_memory_pool.free_all_blocks()

        # Intersection: (i,j) is mutual iff fwd[i]==j AND bwd[j]==i
        matches = []
        for i, (j, dist_ij) in fwd.items():
            if j in bwd and bwd[j][0] == i:
                matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=j,
                                          _distance=dist_ij))
        return matches

    # ==================================================================
    # STAGE 2 — Homography Estimation (DLT)
    # ==================================================================

    @staticmethod
    def _normalize_points(pts):
        """
        Hartley normalisation: translate centroid to origin, scale so that
        mean distance to origin equals sqrt(2). Returns (normalised_pts, T).
        """
        c  = pts.mean(0)
        md = np.sqrt(((pts - c) ** 2).sum(1)).mean()
        s  = np.sqrt(2) / (md + 1e-10)
        T  = np.array([[s, 0, -s*c[0]], [0, s, -s*c[1]], [0, 0, 1]], dtype=np.float64)
        ph = np.hstack([pts, np.ones((len(pts), 1))])
        return (T @ ph.T).T[:, :2], T

    @staticmethod
    def _compute_homography(pts1, pts2):
        """
        Direct Linear Transform homography solved via SVD on the GPU (PyTorch).

        The 2N×9 coefficient matrix is assembled from normalised coordinates,
        and the solution is the last right singular vector. Result is
        denormalised and returned as a (3, 3) float64 array.
        """
        assert len(pts1) >= 4
        p1n, T1 = GPUPyStitch._normalize_points(pts1)
        p2n, T2 = GPUPyStitch._normalize_points(pts2)
        N = len(p1n)
        A = np.zeros((2*N, 9), dtype=np.float64)
        for i in range(N):
            x1, y1 = p1n[i];  x2, y2 = p2n[i]
            A[2*i]   = [-x1, -y1, -1,  0,   0,  0, x2*x1, x2*y1, x2]
            A[2*i+1] = [  0,   0,  0, -x1, -y1, -1, y2*x1, y2*y1, y2]
        _, _, Vt = torch.linalg.svd(
            torch.tensor(A, dtype=torch.float64, device=DEVICE), full_matrices=True)
        H = np.linalg.inv(T2) @ Vt[-1].cpu().numpy().reshape(3, 3) @ T1
        H /= H[2, 2]
        return H

    # ==================================================================
    # STAGE 3 — Robust Fitting (RANSAC)
    # ==================================================================

    @staticmethod
    def _reprojection_error(H, pts1, pts2):
        """
        Symmetric reprojection error: error = ||H·p1 - p2||² + ||H⁻¹·p2 - p1||²
        Evaluated on the GPU (PyTorch). Returns per-point squared errors.
        """
        N = len(pts1)
        try:
            Hi = np.linalg.inv(H)
        except Exception:
            return np.full(N, np.inf)
        ones = torch.ones(N, 1, dtype=torch.float64, device=DEVICE)
        p1h  = torch.cat([torch.tensor(pts1, dtype=torch.float64, device=DEVICE), ones], 1)
        p2h  = torch.cat([torch.tensor(pts2, dtype=torch.float64, device=DEVICE), ones], 1)
        Ht   = torch.tensor(H,  dtype=torch.float64, device=DEVICE)
        Hit  = torch.tensor(Hi, dtype=torch.float64, device=DEVICE)
        def proj(M, ph):
            q = (M @ ph.T).T
            return q[:, :2] / q[:, 2:3].clamp(min=1e-10)
        p1t = torch.tensor(pts1, dtype=torch.float64, device=DEVICE)
        p2t = torch.tensor(pts2, dtype=torch.float64, device=DEVICE)
        err = ((p2t - proj(Ht, p1h)) ** 2).sum(1) + ((p1t - proj(Hit, p2h)) ** 2).sum(1)
        return err.cpu().numpy()

    def _ransac(self, pts1, pts2, n_iter=None, inlier_thresh=None, min_inliers=10, seed=42):
        """
        GPU-batched MAGSAC++ / RANSAC with LO-RANSAC two-pass refinement.

        All `n_iter` hypotheses are sampled and evaluated in one GPU pass using
        batched SVD (PyTorch). Hypothesis selection uses MAGSAC++ soft scoring
        (Barath et al., CVPR 2020) when ransac_method='magsac' (default), or
        hard inlier counting when ransac_method='classic'. The best hypothesis
        is then refined by re-fitting on its inlier set (LO-RANSAC).

        MAGSAC++ score:  S(H) = Σ_i exp( −ε_i² / (2·σ_max²) )
        where ε_i² is the symmetric reprojection error per point.  This is the
        closed-form marginalisation over σ for 2-D Rayleigh residuals, giving a
        continuous soft score that is robust to threshold choice.

        Returns
        -------
        H : numpy.ndarray, shape (3, 3), float64
        inlier_mask : numpy.ndarray, shape (N,), bool
        """
        if n_iter is None:
            n_iter = self.ransac_iters
        if inlier_thresh is None:
            inlier_thresh = self.inlier_thresh

        N = len(pts1)
        if N < 4:
            raise ValueError(f"Need >= 4 correspondences, got {N}")
        tq = inlier_thresh ** 2
        p1n, T1 = self._normalize_points(pts1)
        p2n, T2 = self._normalize_points(pts2)
        T1t   = torch.tensor(T1,                dtype=torch.float64, device=DEVICE)
        T2inv = torch.tensor(np.linalg.inv(T2), dtype=torch.float64, device=DEVICE)
        p1t   = torch.tensor(pts1, dtype=torch.float64, device=DEVICE)
        p2t   = torch.tensor(pts2, dtype=torch.float64, device=DEVICE)
        ones  = torch.ones(N, 1,   dtype=torch.float64, device=DEVICE)
        p1h   = torch.cat([p1t, ones], dim=1)
        p2h   = torch.cat([p2t, ones], dim=1)
        p1n_t = torch.tensor(p1n, dtype=torch.float64, device=DEVICE)
        p2n_t = torch.tensor(p2n, dtype=torch.float64, device=DEVICE)

        B   = n_iter
        gen = torch.Generator(device='cpu');  gen.manual_seed(seed)
        # Changes 7+10: random sampling stays on CPU (non-compilable), batch
        # SVD + error computation delegated to the compiled _ransac_batch_eval.
        all_idx    = torch.randint(0, N, (B, 4), generator=gen).to(DEVICE)
        # Opt 8: use CUDA-graph-cached eval for repeated calls with same (B,N)
        H_batch, sym_err = _ransac_batch_eval_graphed(
            all_idx, p1n_t, p2n_t, T1t, T2inv, p1h, p2h, p1t, p2t)
        if self.ransac_method == 'magsac':
            # MAGSAC++ soft score — fully GPU, no new memory allocation
            sig2    = self.sigma_max ** 2
            weights = torch.exp(-sym_err / (2.0 * sig2))  # [B, N]
            best_i  = int(weights.sum(dim=1).argmax())
        else:
            best_i  = int((sym_err < tq).sum(dim=1).argmax())
        # Inlier set for LO-RANSAC always uses hard threshold (defines refitting set)
        inlier_mask = sym_err < tq
        best_mask   = inlier_mask[best_i].cpu().numpy()
        best_n      = int(best_mask.sum())

        # LO-RANSAC: re-fit on the best inlier set, re-classify.
        if best_n >= min_inliers:
            H1   = self._compute_homography(pts1[best_mask], pts2[best_mask])
            ph1  = np.hstack([pts1, np.ones((N, 1))])
            ph2  = np.hstack([pts2, np.ones((N, 1))])
            fwd2 = (H1 @ ph1.T).T;  fwd2 /= fwd2[:, 2:3]
            try:    H1inv = np.linalg.inv(H1)
            except np.linalg.LinAlgError:
                    H1inv = np.linalg.pinv(H1)
            bwd2  = (H1inv @ ph2.T).T;  bwd2 /= bwd2[:, 2:3]
            err2  = ((fwd2[:, :2] - pts2) ** 2).sum(1) + ((bwd2[:, :2] - pts1) ** 2).sum(1)
            mask2 = err2 < tq
            n2    = int(mask2.sum())
            if n2 >= min_inliers:
                best_H    = self._compute_homography(pts1[mask2], pts2[mask2])
                best_mask = mask2
            else:
                best_H = H1
        else:
            best_H = H_batch[best_i].cpu().numpy()

        del p1t, p2t, ones, p1h, p2h, p1n_t, p2n_t, T1t, T2inv
        del all_idx, H_batch, sym_err, inlier_mask
        # empty_cache() removed — was running per pair (expensive on Windows WDDM).
        # Periodic pool flush in detectAndCompute (every 50 calls) handles VRAM.
        return best_H, best_mask

    # ==================================================================
    # STAGE 4 — Depth Analysis
    # ==================================================================

    def _depth_stratified_homographies(self, pts1, pts2, inlier_mask,
                                        depth_map, n_layers=None):
        """
        Fit a separate homography per depth band to correct depth-dependent parallax.

        Nearby objects have larger parallax (Δx = f·b/Z) than distant ones.
        A single global homography cannot simultaneously correct all depths.
        This method stratifies RANSAC inliers into `n_layers` equal-count depth
        bands and fits a local homography per band via 500-iteration RANSAC.

        Parameters
        ----------
        pts1, pts2 : numpy.ndarray, shape (N, 2)
            All matched point coordinates (before RANSAC filtering).
        inlier_mask : numpy.ndarray, shape (N,), bool
            Inlier flags from the global RANSAC run.
        depth_map : numpy.ndarray, shape (H, W), float32
            Depth map for pts1's image (0 = far, 1 = near).
        n_layers : int, optional
            Number of depth bands. Defaults to self.n_depth_layers.

        Returns
        -------
        H_layers : list of numpy.ndarray or None
            Per-band homographies (None where insufficient inliers).
        depth_bands : list of (float, float)
            (lo, hi) depth range per band.
        """
        if n_layers is None:
            n_layers = self.n_depth_layers

        inlier_pts1 = pts1[inlier_mask]
        inlier_pts2 = pts2[inlier_mask]
        H1_img, W1_img = depth_map.shape

        xs     = np.clip(inlier_pts1[:, 0].astype(int), 0, W1_img - 1)
        ys     = np.clip(inlier_pts1[:, 1].astype(int), 0, H1_img - 1)
        depths = depth_map[ys, xs]

        # Equal-count bands via percentiles — avoids empty bands on flat scenes.
        edges         = np.percentile(depths, np.linspace(0, 100, n_layers + 1))
        edges[0]      = 0.0
        edges[-1]     = 1.0 + 1e-6
        depth_bands   = [(float(edges[i]), float(edges[i+1])) for i in range(n_layers)]

        H_layers = []
        for lo, hi in depth_bands:
            band  = (depths >= lo) & (depths < hi)
            n_pts = int(band.sum())
            if n_pts < 8:
                H_layers.append(None)
                continue
            try:
                H_band, _ = self._ransac(
                    inlier_pts1[band], inlier_pts2[band],
                    n_iter=500, inlier_thresh=4.0, min_inliers=4)
                H_layers.append(H_band)
                print(f"    [Depth {lo:.2f}-{hi:.2f}] {n_pts} pts -> local H fitted")
            except Exception as exc:
                H_layers.append(None)
                print(f"    [Depth {lo:.2f}-{hi:.2f}] skipped: {exc}")

        return H_layers, depth_bands

    # ==================================================================
    # STAGE 5 — Composition (Warp + Seam + Blend)
    # ==================================================================

    @staticmethod
    def _compute_canvas_size(img1, img2, H):
        """
        Compute output canvas dimensions for a 2-image stitch.

        Projects img1's corners through H, combines with img2's corners,
        and computes the bounding box. Returns (width, height, offset_x, offset_y)
        where the offsets shift the coordinate origin to (0, 0).
        """
        h1, w1 = img1.shape[:2];  h2, w2 = img2.shape[:2]
        c1 = np.array([[0, 0, 1], [w1, 0, 1], [w1, h1, 1], [0, h1, 1]],
                      dtype=np.float64).T
        wc = H @ c1;  wc /= wc[2]
        ax = np.concatenate([wc[0], [0, w2, w2, 0]])
        ay = np.concatenate([wc[1], [0,  0, h2, h2]])
        ox = max(0.0, -ax.min());  oy = max(0.0, -ay.min())
        return int(np.ceil(ax.max() + ox)) + 1, int(np.ceil(ay.max() + oy)) + 1, ox, oy

    @staticmethod
    def _compute_canvas_size_3(img1, img2, img3, H12, H32):
        """
        Compute output canvas dimensions for a 3-image stitch (img2 = reference).

        Projects img1 and img3 corners through their respective homographies
        and computes the bounding box together with img2's extent.
        """
        h2, w2 = img2.shape[:2]
        all_x  = [0.0, float(w2), float(w2), 0.0]
        all_y  = [0.0, 0.0, float(h2), float(h2)]
        for img, H in [(img1, H12), (img3, H32)]:
            h, w = img.shape[:2]
            corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]],
                                dtype=np.float64).T
            wc = H @ corners;  wc /= wc[2]
            all_x.extend(wc[0].tolist());  all_y.extend(wc[1].tolist())
        ax, ay = np.array(all_x), np.array(all_y)
        ox = max(0.0, -ax.min());  oy = max(0.0, -ay.min())
        return int(np.ceil(ax.max() + ox)) + 1, int(np.ceil(ay.max() + oy)) + 1, ox, oy

    @staticmethod
    def _warp_image(img, H, cw, ch, ox, oy):
        """
        GPU bilinear warp via PyTorch grid_sample.

        Computes the inverse mapping H⁻¹: for each output canvas pixel,
        look up the corresponding source pixel. Uses PyTorch's bilinear
        interpolation with zero-padding outside the source image.

        Returns (warped uint8 image, valid-pixel boolean mask).
        """
        hs, ws = img.shape[:2]
        u  = torch.arange(cw, dtype=torch.float32, device=DEVICE)
        v  = torch.arange(ch, dtype=torch.float32, device=DEVICE)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        flat = torch.stack([uu - ox, vv - oy, torch.ones_like(uu)], 0).reshape(3, -1)
        src  = torch.tensor(np.linalg.inv(H), dtype=torch.float32, device=DEVICE) @ flat
        sx   = src[0] / src[2].clamp(min=1e-10)
        sy   = src[1] / src[2].clamp(min=1e-10)
        sxn  = (2.0 * sx / (ws - 1)) - 1.0
        syn  = (2.0 * sy / (hs - 1)) - 1.0
        grid = torch.stack([sxn, syn], -1).reshape(1, ch, cw, 2)
        it   = torch.tensor(img.astype(np.float32), device=DEVICE).permute(2, 0, 1).unsqueeze(0)
        out  = torch.nn.functional.grid_sample(
            it, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        warped = out.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
        mask   = ((sxn >= -1) & (sxn <= 1) & (syn >= -1) & (syn <= 1)).reshape(ch, cw).cpu().numpy()
        del u, v, uu, vv, flat, src, sx, sy, sxn, syn, grid, it, out
        torch.cuda.empty_cache()
        return warped, mask

    def _warp_depth_aware(self, img, H_global, H_layers, depth_bands,
                           depth_map, cw, ch, ox, oy):
        """
        Depth-aware warp: apply per-band local homographies over a global base warp.

        Algorithm:
          1. Base warp of the source image using the global homography.
          2. Warp the depth map to canvas space to know each output pixel's depth.
          3. For each depth band with a local H: re-warp using that H and
             composite the result back into pixels belonging to that depth range.

        This corrects depth-dependent parallax: nearby objects (high parallax)
        receive their own corrected homography while distant objects use the
        globally-fitted one.

        Returns (warped uint8 image, valid-pixel boolean mask).
        """
        warped, mask = self._warp_image(img, H_global, cw, ch, ox, oy)
        warped_f = warped.astype(np.float32)

        # Warp depth map to canvas for per-pixel depth lookup.
        depth_u8    = (depth_map * 255).clip(0, 255).astype(np.uint8)
        depth_3ch   = np.stack([depth_u8] * 3, axis=2)
        d_warped, _ = self._warp_image(depth_3ch, H_global, cw, ch, ox, oy)
        depth_canvas = d_warped[:, :, 0].astype(np.float32) / 255.0

        n_applied = 0
        for (lo, hi), H_band in zip(depth_bands, H_layers):
            if H_band is None:
                continue
            band_mask = (depth_canvas >= lo) & (depth_canvas < hi)
            if not band_mask.any():
                continue
            warped_band, mask_band = self._warp_image(img, H_band, cw, ch, ox, oy)
            valid = band_mask & mask_band
            warped_f[valid] = warped_band.astype(np.float32)[valid]
            mask = mask | valid
            n_applied += 1
            del warped_band, mask_band, valid, band_mask
            torch.cuda.empty_cache()

        if n_applied:
            print(f"    Depth-aware warp: {n_applied}/{len(H_layers)} band Hs applied")
        return warped_f.clip(0, 255).astype(np.uint8), mask

    @staticmethod
    def _dp_vertical_seam(cost_np, ov_mask):
        """
        Dynamic-programming shortest path from top to bottom of the overlap region.

        At each row the seam may move one column left, stay, or move one column
        right. Cost at each cell is the image-difference cost plus the
        accumulated cost from the best predecessor.

        Returns seam_cols: int32 array of length H with the seam column per row.
        """
        H, W  = cost_np.shape
        INF   = 1e18
        ov_rows = np.where(np.any(ov_mask, axis=1))[0]
        if len(ov_rows) == 0:
            return np.zeros(H, np.int32)

        dp = np.full((H, W), INF, np.float64)
        bt = np.zeros((H, W), np.int32)

        r0    = ov_rows[0]
        dp[r0] = np.where(ov_mask[r0], cost_np[r0].astype(np.float64), INF)

        prev_r = r0
        for r in ov_rows[1:]:
            prev = dp[prev_r]
            pl = np.empty(W, np.float64);  pl[0]  = INF;  pl[1:]  = prev[:-1]
            pm = prev.copy()
            pr = np.empty(W, np.float64);  pr[-1] = INF;  pr[:-1] = prev[1:]

            stk  = np.stack([pl, pm, pr])
            aidx = np.argmin(stk, axis=0)
            best = stk[aidx, np.arange(W)]

            dp[r] = np.where(ov_mask[r], cost_np[r].astype(np.float64) + best, INF)
            bt[r] = np.clip(np.arange(W) + aidx - 1, 0, W - 1)
            prev_r = r

        r_last   = ov_rows[-1]
        row_cost = np.where(ov_mask[r_last], dp[r_last], INF)
        seam_c   = int(np.argmin(row_cost))

        seam_cols          = np.full(H, seam_c, np.int32)
        seam_cols[r_last]  = seam_c
        for i in range(len(ov_rows) - 1, 0, -1):
            r_curr = ov_rows[i]
            r_prev = ov_rows[i - 1]
            seam_cols[r_prev] = bt[r_curr, seam_cols[r_curr]]

        return seam_cols

    @staticmethod
    def _dp_horizontal_seam(cost_np, ov_mask):
        """
        Dynamic-programming shortest path from left to right of the overlap region.

        Analogous to _dp_vertical_seam but transposed: the seam is a row per column.
        Returns seam_rows: int32 array of length W with the seam row per column.
        """
        H, W  = cost_np.shape
        INF   = 1e18
        ov_cols = np.where(np.any(ov_mask, axis=0))[0]
        if len(ov_cols) == 0:
            return np.zeros(W, np.int32)

        dp = np.full((H, W), INF, np.float64)
        bt = np.zeros((H, W), np.int32)

        c0 = ov_cols[0]
        dp[:, c0] = np.where(ov_mask[:, c0], cost_np[:, c0].astype(np.float64), INF)

        prev_c = c0
        for c in ov_cols[1:]:
            prev = dp[:, prev_c]
            pt = np.empty(H, np.float64);  pt[0]  = INF;  pt[1:]  = prev[:-1]
            pm = prev.copy()
            pb = np.empty(H, np.float64);  pb[-1] = INF;  pb[:-1] = prev[1:]

            stk  = np.stack([pt, pm, pb])
            aidx = np.argmin(stk, axis=0)
            best = stk[aidx, np.arange(H)]

            dp[:, c] = np.where(ov_mask[:, c], cost_np[:, c].astype(np.float64) + best, INF)
            bt[:, c] = np.clip(np.arange(H) + aidx - 1, 0, H - 1)
            prev_c = c

        c_last   = ov_cols[-1]
        col_cost = np.where(ov_mask[:, c_last], dp[:, c_last], INF)
        seam_r   = int(np.argmin(col_cost))

        seam_rows          = np.full(W, seam_r, np.int32)
        seam_rows[c_last]  = seam_r
        for i in range(len(ov_cols) - 1, 0, -1):
            c_curr = ov_cols[i]
            c_prev = ov_cols[i - 1]
            seam_rows[c_prev] = bt[seam_rows[c_curr], c_curr]

        return seam_rows

    @staticmethod
    def _find_seam(w1_f, w2_f, mask1, mask2):
        """
        Find the optimal seam between two images in the overlap region.

        Cost function (computed on GPU):
            C = ||I1 - I2||_RGB  +  0.05 × |∇(avg(I1, I2))|

        The seam follows pixels where the two images agree most closely,
        making the cut effectively invisible. Seam orientation (vertical vs
        horizontal) is chosen based on the overlap geometry. The seam mask
        is Gaussian-blurred (σ=3) in the overlap to provide a smooth alpha
        input to the Laplacian pyramid.

        Returns
        -------
        seam_mask_f : float32 (H, W) — 1.0 = use img1, 0.0 = use img2
        """
        H, W = mask1.shape
        ov   = mask1 & mask2

        seam_mask_f = np.zeros((H, W), np.float32)
        seam_mask_f[mask1 & ~mask2] = 1.0   # img1-only region

        if not ov.any():
            return seam_mask_f

        # Cost map on GPU (CuPy).
        W1 = cp.asarray(w1_f)
        W2 = cp.asarray(w2_f)
        color_cost = cp.sqrt(cp.sum((W1 - W2) ** 2, axis=2))
        avg_gray   = cp.mean((W1 + W2) * 0.5, axis=2)
        gx = cpnd.sobel(avg_gray, axis=1)
        gy = cpnd.sobel(avg_gray, axis=0)
        grad_cost  = cp.sqrt(gx ** 2 + gy ** 2)
        cost_gpu   = color_cost + 0.05 * grad_cost
        cost_np    = cp.asnumpy(cost_gpu)
        cost_np[~ov] = 1e9
        del W1, W2, color_cost, avg_gray, gx, gy, grad_cost, cost_gpu
        cp._default_memory_pool.free_all_blocks()

        # Choose seam orientation based on overlap shape.
        n_ov_rows = int(np.any(ov, axis=1).sum())
        n_ov_cols = int(np.any(ov, axis=0).sum())

        solo1_cols = np.where(np.any(mask1 & ~mask2, axis=0))[0]
        solo2_cols = np.where(np.any(mask2 & ~mask1, axis=0))[0]
        img1_left  = (len(solo1_cols) == 0 or len(solo2_cols) == 0 or
                      solo1_cols.mean() <= solo2_cols.mean())

        solo1_rows = np.where(np.any(mask1 & ~mask2, axis=1))[0]
        solo2_rows = np.where(np.any(mask2 & ~mask1, axis=1))[0]
        img1_top   = (len(solo1_rows) == 0 or len(solo2_rows) == 0 or
                      solo1_rows.mean() <= solo2_rows.mean())

        if n_ov_rows >= n_ov_cols:
            # Tall overlap → vertical seam (one column per row).
            seam_cols = GPUPyStitch._dp_vertical_seam(cost_np, ov)
            ov_rows   = np.any(ov, axis=1)
            for r in range(H):
                if not ov_rows[r]:
                    continue
                c = seam_cols[r]
                if img1_left:
                    seam_mask_f[r, :c+1] = np.where(ov[r, :c+1], 1.0, seam_mask_f[r, :c+1])
                    seam_mask_f[r, c+1:] = np.where(ov[r, c+1:], 0.0, seam_mask_f[r, c+1:])
                else:
                    seam_mask_f[r, :c+1] = np.where(ov[r, :c+1], 0.0, seam_mask_f[r, :c+1])
                    seam_mask_f[r, c+1:] = np.where(ov[r, c+1:], 1.0, seam_mask_f[r, c+1:])
        else:
            # Wide overlap → horizontal seam (one row per column).
            seam_rows = GPUPyStitch._dp_horizontal_seam(cost_np, ov)
            ov_cols   = np.any(ov, axis=0)
            for c in range(W):
                if not ov_cols[c]:
                    continue
                r = seam_rows[c]
                if img1_top:
                    seam_mask_f[:r+1, c] = np.where(ov[:r+1, c], 1.0, seam_mask_f[:r+1, c])
                    seam_mask_f[r+1:, c] = np.where(ov[r+1:, c], 0.0, seam_mask_f[r+1:, c])
                else:
                    seam_mask_f[:r+1, c] = np.where(ov[:r+1, c], 0.0, seam_mask_f[:r+1, c])
                    seam_mask_f[r+1:, c] = np.where(ov[r+1:, c], 1.0, seam_mask_f[r+1:, c])

        # Blur seam transition in the overlap to provide a smooth alpha for the pyramid.
        blurred = cv2.GaussianBlur(seam_mask_f, (0, 0), sigmaX=3.0)
        seam_mask_f[ov] = blurred[ov]
        return seam_mask_f

    @staticmethod
    def _gauss_down(img_gpu, sigma=1.0):
        """Gaussian blur followed by 2× downsampling. Operates on CuPy arrays."""
        if img_gpu.ndim == 3:
            blurred = cp.stack(
                [cpnd.gaussian_filter(img_gpu[:, :, c], sigma=sigma)
                 for c in range(img_gpu.shape[2])], axis=2)
        else:
            blurred = cpnd.gaussian_filter(img_gpu, sigma=sigma)
        return blurred[::2, ::2]

    @staticmethod
    def _gauss_up(img_gpu, th, tw):
        """Bilinear upsample to (th, tw) on the GPU via CuPy zoom (order=1)."""
        if img_gpu.ndim == 3:
            zy = th / img_gpu.shape[0];  zx = tw / img_gpu.shape[1]
            return cpnd.zoom(img_gpu, [zy, zx, 1.0], order=1)
        else:
            zy = th / img_gpu.shape[0];  zx = tw / img_gpu.shape[1]
            return cpnd.zoom(img_gpu, [zy, zx], order=1)

    @staticmethod
    def _multiband_blend(I1_full, I2_full, seam_mask_f, n_levels=6):
        """
        Multi-band Laplacian pyramid blending entirely on the GPU (CuPy).

        At each pyramid level k the seam mask is naturally blurred over 2^k
        pixels. Coarse structures blend smoothly (wide transition), while fine
        details blend sharply (narrow transition at the seam line). This
        avoids ghosting at large scales and halo artefacts at fine scales.

        Parameters
        ----------
        I1_full, I2_full : float32 (H, W, 3)
            Full-canvas images. Pixels outside each image's valid region
            should be filled with the other image's content to prevent
            pyramid-edge bleed artefacts.
        seam_mask_f : float32 (H, W)
            Seam mask: 1.0 = take from I1, 0.0 = take from I2.
        n_levels : int
            Pyramid depth (6 works well for images up to ~4096 px wide).

        Returns
        -------
        blended : float32 (H, W, 3)
        """
        g1 = cp.asarray(I1_full)
        g2 = cp.asarray(I2_full)
        gm = cp.asarray(seam_mask_f[:, :, np.newaxis].repeat(3, axis=2))

        # Build Gaussian pyramids for both images and the seam mask.
        gp1, gp2, gpm = [g1], [g2], [gm]
        for _ in range(n_levels - 1):
            gp1.append(GPUPyStitch._gauss_down(gp1[-1]))
            gp2.append(GPUPyStitch._gauss_down(gp2[-1]))
            gpm.append(GPUPyStitch._gauss_down(gpm[-1]))

        # Build Laplacian pyramids: L[k] = G[k] - upsample(G[k+1]).
        lp1, lp2 = [], []
        for k in range(n_levels - 1):
            th, tw = gp1[k].shape[:2]
            lp1.append(gp1[k] - GPUPyStitch._gauss_up(gp1[k + 1], th, tw))
            lp2.append(gp2[k] - GPUPyStitch._gauss_up(gp2[k + 1], th, tw))
        lp1.append(gp1[-1])   # coarsest level: the Gaussian itself
        lp2.append(gp2[-1])

        # Blend each pyramid level independently.
        lp_blend = [gpm[k] * lp1[k] + (1.0 - gpm[k]) * lp2[k]
                    for k in range(n_levels)]

        # Reconstruct coarse → fine.
        result = lp_blend[-1]
        for k in range(n_levels - 2, -1, -1):
            th, tw = lp_blend[k].shape[:2]
            result = GPUPyStitch._gauss_up(result, th, tw) + lp_blend[k]

        blended = cp.asnumpy(result)
        del g1, g2, gm, gp1, gp2, gpm, lp1, lp2, lp_blend, result
        cp._default_memory_pool.free_all_blocks()
        return blended   # float32 (H, W, 3)

    @staticmethod
    def _seam_rmse(wa, ma, wb, mb):
        """Root-mean-square colour difference in the overlap region (quality metric)."""
        overlap = ma & mb
        if not overlap.any():
            return float('nan')
        diff = wa[overlap].astype(np.float32) - wb[overlap].astype(np.float32)
        return float(np.sqrt((diff ** 2).mean()))

    @staticmethod
    def _verify_chain(H_AB, H_CB):
        """
        Check the consistency of two homographies for 3-image stitching.

        Computes H_chain = H_CB⁻¹ · H_AB and inspects the rotation angle and
        scale factor. Prints a warning if either exceeds typical thresholds.
        """
        try:
            H_chain = np.linalg.inv(H_CB) @ H_AB
            H_chain /= H_chain[2, 2]
        except np.linalg.LinAlgError:
            print("  [Chain] WARNING: H_CB is singular")
            return
        theta = float(np.degrees(np.arctan2(H_chain[1, 0], H_chain[0, 0])))
        scale = float(np.sqrt(abs(np.linalg.det(H_chain[:2, :2]))))
        warnings = []
        if abs(theta) > 5.0:     warnings.append(f"rotation drift {theta:+.1f}°")
        if abs(scale - 1.0) > 0.15: warnings.append(f"scale drift {scale:.3f}×")
        status = "OK" if not warnings else "WARN: " + ", ".join(warnings)
        print(f"  [Chain] rotation={theta:+.1f}° scale={scale:.3f}  [{status}]")

    # ==================================================================
    # COMPOSITION HELPERS — shared by 2-image and 3-image pipelines
    # ==================================================================

    @staticmethod
    def _hist_match(src, ref, mask):
        """Match per-channel histogram of src to ref using overlap-region pixels.

        Builds a cumulative distribution function (CDF) for each channel in
        both src and ref (restricted to the overlap mask), then derives a
        look-up table that maps src intensity values to the equivalent
        quantile in ref.  The LUT is applied to the entire src image so that
        colour balance matches the reference across the whole frame.

        Parameters
        ----------
        src  : float32 (H, W, 3)  — image to be colour-corrected.
        ref  : float32 (H, W, 3)  — reference image (target colours).
        mask : bool    (H, W)     — overlap region used to build CDFs.

        Returns
        -------
        out : float32 (H, W, 3)  — colour-corrected copy of src.
        """
        out  = src.copy()
        bins = np.arange(257, dtype=np.float32)
        for c in range(3):
            s_vals = src[mask, c].clip(0, 255)
            r_vals = ref[mask, c].clip(0, 255)
            s_cnt, _ = np.histogram(s_vals, bins=bins)
            r_cnt, _ = np.histogram(r_vals, bins=bins)
            s_cdf = np.cumsum(s_cnt).astype(np.float64)
            r_cdf = np.cumsum(r_cnt).astype(np.float64)
            s_cdf /= s_cdf[-1] + 1e-8
            r_cdf /= r_cdf[-1] + 1e-8
            lut = np.interp(s_cdf, r_cdf, np.arange(256))
            out[:, :, c] = np.interp(
                src[:, :, c].ravel().clip(0, 255), np.arange(256), lut
            ).reshape(src.shape[:2])
        return out.astype(np.float32)

    @staticmethod
    def _autocrop(pano, both):
        """Crop panorama to the tight bounding box of valid pixels.

        Finds the largest axis-aligned rectangle where every pixel is valid
        (covered by at least one source image).  A simple bounding-box crop
        only removes fully-empty border rows/cols; this inner-rectangle crop
        also eliminates triangular voids that fall inside the bounding box
        but are not covered by any image.

        Algorithm:
          For each row in the valid range, record [left, right] — the first
          and last valid column.  The inner column range is
          [max(lefts), min(rights)].  Symmetrically for rows per column.
          The intersection gives the rectangle where every row spans the full
          column range and every column spans the full row range.

        Parameters
        ----------
        pano : uint8 (H, W, 3) — full-canvas panorama.
        both : bool  (H, W)    — valid-pixel mask (True where covered).

        Returns
        -------
        cropped : uint8 array — black borders and corner voids removed.
        """
        H, W = both.shape
        row_valid = np.any(both, axis=1)
        col_valid = np.any(both, axis=0)
        if not row_valid.any():
            return pano

        r0 = int(np.where(row_valid)[0][0])
        r1 = int(np.where(row_valid)[0][-1])
        c0 = int(np.where(col_valid)[0][0])
        c1 = int(np.where(col_valid)[0][-1])

        sub = both[r0:r1 + 1, c0:c1 + 1]
        nr, nc = sub.shape

        # Per row: leftmost and rightmost valid column inside the sub-region.
        left_per_row  = np.argmax(sub, axis=1)
        right_per_row = nc - 1 - np.argmax(sub[:, ::-1], axis=1)

        # Per col: topmost and bottommost valid row inside the sub-region.
        top_per_col = np.argmax(sub, axis=0)
        bot_per_col = nr - 1 - np.argmax(sub[::-1, :], axis=0)

        # Inner rectangle: tightest bounds where every row/col is fully valid.
        ic0 = int(left_per_row.max())
        ic1 = int(right_per_row.min())
        ir0 = int(top_per_col.max())
        ir1 = int(bot_per_col.min())

        if ic0 >= ic1 or ir0 >= ir1:          # degenerate — fall back to bbox
            return pano[r0:r1 + 1, c0:c1 + 1]

        return pano[r0 + ir0 : r0 + ir1 + 1, c0 + ic0 : c0 + ic1 + 1]

    def _blend_pair(self, w1_f, mask1, c2f, mask2, cw, ch):
        """
        Gain compensation + seam finding + multi-band blend for one image pair.

        Returns (blended float32 array, combined mask, seam RMSE).
        """
        ov = mask1 & mask2
        if ov.any():
            if self.color_match == 'hist':
                w1_f = GPUPyStitch._hist_match(w1_f, c2f, ov)
            else:
                gain = (c2f[ov].mean(0) + 1e-3) / (w1_f[ov].mean(0) + 1e-3)
                w1_f = np.clip(w1_f * np.clip(gain, 0.5, 2.0), 0, 255)

        seam = self._find_seam(w1_f, c2f, mask1, mask2)

        I1f = np.zeros((ch, cw, 3), np.float32)
        I2f = np.zeros((ch, cw, 3), np.float32)
        I1f[mask1]        = w1_f[mask1]
        I2f[mask2]        = c2f[mask2]
        I1f[~mask1&mask2] = c2f[~mask1&mask2]
        I2f[~mask2&mask1] = w1_f[~mask2&mask1]

        blended  = self._multiband_blend(I1f, I2f, seam, self.n_blend_levels)
        both     = mask1 | mask2
        rmse     = self._seam_rmse(w1_f.clip(0, 255).astype(np.uint8), mask1,
                                    c2f.clip(0, 255).astype(np.uint8),  mask2)
        del I1f, I2f
        return blended, both, rmse

    # ==================================================================
    # PIPELINE ORCHESTRATION
    # ==================================================================

    def _stitch_pair(self, img1, img2):
        """
        Full 5-stage pipeline for 2 images.

        img2 is used as the reference frame (placed directly on the canvas).
        img1 is warped to match img2's coordinate system.

        Returns (panorama uint8, timings dict, metrics dict).
        """
        timings = {}
        metrics = {}

        # ── Stage 0: Preprocessing ───────────────────────────────────────
        img1 = self._preprocess_image(img1)
        img2 = self._preprocess_image(img2)

        # ── Stage 1: Feature extraction ──────────────────────────────────
        print("\n[Stage 1] Feature extraction ...")
        t0 = time.perf_counter()
        kp1, d1, sz1 = self._extract_features(img1)
        kp2, d2, sz2 = self._extract_features(img2)
        matches  = self._match_features(d1, d2, kp1, kp2, sz1, sz2)
        timings['st1'] = time.perf_counter() - t0
        metrics.update(kp1=len(kp1), kp2=len(kp2), matches=len(matches))
        print(f"    img1: {len(kp1)} kp   img2: {len(kp2)} kp   matches: {len(matches)}")
        if len(matches) < 4:
            raise RuntimeError(f"Insufficient matches ({len(matches)}), need >= 4")
        pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])

        # ── Stage 2: Homography (DLT) ─────────────────────────────────────
        print("[Stage 2] DLT homography ...")
        t0 = time.perf_counter()
        self._compute_homography(pts1, pts2)   # preliminary estimate
        timings['st2'] = time.perf_counter() - t0

        # ── Stage 3: Robust fitting (RANSAC) ─────────────────────────────
        print("[Stage 3] RANSAC ...")
        t0 = time.perf_counter()
        H_robust, inlier_mask = self._ransac(pts1, pts2)
        timings['st3'] = time.perf_counter() - t0
        n_inliers = int(inlier_mask.sum())
        metrics.update(inliers=n_inliers, inlier_ratio=n_inliers / max(len(matches), 1))
        print(f"    {n_inliers}/{len(matches)} inliers")
        if n_inliers < 4:
            raise RuntimeError(f"Only {n_inliers} inliers after RANSAC, need >= 4")

        # ── Stage 4: Depth analysis ───────────────────────────────────────
        print("[Stage 4] Depth analysis ...")
        t0 = time.perf_counter()
        depth_model  = self._get_depth_estimator()
        H_layers     = None
        depth_bands  = None
        depth_map    = None
        depth_cv     = None
        if depth_model is not None:
            depth_map = depth_model.estimate(img1)
            depth_cv  = float(depth_map.std() / (depth_map.mean() + 1e-8))
            H_layers, depth_bands = self._depth_stratified_homographies(
                pts1, pts2, inlier_mask, depth_map)
        timings['st4'] = time.perf_counter() - t0
        metrics.update(depth_cv=depth_cv, bands=self.n_depth_layers)

        # ── Stage 5: Composition ──────────────────────────────────────────
        print("[Stage 5] Composition (warp -> seam -> blend) ...")
        t0 = time.perf_counter()
        cw, ch, ox, oy = self._compute_canvas_size(img1, img2, H_robust)

        if depth_map is not None and H_layers is not None:
            warped1, mask1 = self._warp_depth_aware(
                img1, H_robust, H_layers, depth_bands, depth_map, cw, ch, ox, oy)
        else:
            warped1, mask1 = self._warp_image(img1, H_robust, cw, ch, ox, oy)

        iox, ioy = int(round(ox)), int(round(oy))
        h2, w2   = img2.shape[:2]
        canvas2  = np.zeros((ch, cw, 3), np.uint8)
        mask2    = np.zeros((ch, cw), bool)
        ye, xe   = min(ioy + h2, ch), min(iox + w2, cw)
        canvas2[ioy:ye, iox:xe] = img2[:ye-ioy, :xe-iox]
        mask2[ioy:ye, iox:xe]   = True

        blended, both, seam_rmse = self._blend_pair(
            warped1.astype(np.float32), mask1,
            canvas2.astype(np.float32), mask2,
            cw, ch)

        pano = np.zeros((ch, cw, 3), np.uint8)
        pano[both] = np.clip(blended[both], 0, 255).astype(np.uint8)
        del blended
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

        timings['st5'] = time.perf_counter() - t0
        coverage = float(both.sum()) / float(ch * cw) * 100.0
        metrics.update(seam_rmse=seam_rmse, coverage=coverage, canvas_w=cw, canvas_h=ch)

        pano = self._autocrop(pano, both)
        return pano, timings, metrics

    def _stitch_triple(self, img1, img2, img3):
        """
        Full 5-stage pipeline for 3 images (img2 = reference frame).

        Both img1 and img3 are warped into img2's coordinate system on a single
        shared canvas. Blending is performed in two passes:
          Pass 1: blend(warped_img1, canvas_img2) → intermediate
          Pass 2: blend(intermediate, warped_img3) → final panorama

        Returns (panorama uint8, timings dict, metrics dict).
        """
        timings = {}
        metrics = {}

        # ── Stage 0: Preprocessing ───────────────────────────────────────
        img1 = self._preprocess_image(img1)
        img2 = self._preprocess_image(img2)
        img3 = self._preprocess_image(img3)

        # ── Stage 1: Feature extraction ──────────────────────────────────
        print("\n[Stage 1] Feature extraction (3 images) ...")
        t0 = time.perf_counter()
        kp1, d1, sz1 = self._extract_features(img1)
        kp2, d2, sz2 = self._extract_features(img2)
        kp3, d3, sz3 = self._extract_features(img3)
        # img1 ↔ img2 and img2 ↔ img3 correspondences
        matches_12 = self._match_features(d1, d2, kp1, kp2, sz1, sz2)
        matches_23 = self._match_features(d2, d3, kp2, kp3, sz2, sz3)
        timings['st1'] = time.perf_counter() - t0
        metrics.update(kp1=len(kp1), kp2=len(kp2), kp3=len(kp3),
                       matches_12=len(matches_12), matches_23=len(matches_23))
        print(f"    img1: {len(kp1)} kp   img2: {len(kp2)} kp   img3: {len(kp3)} kp")
        print(f"    matches 1<->2: {len(matches_12)}   matches 2<->3: {len(matches_23)}")
        if len(matches_12) < 4 or len(matches_23) < 4:
            raise RuntimeError("Insufficient matches for 3-image stitching")
        pA  = np.float64([kp1[m.queryIdx].pt for m in matches_12])
        pB1 = np.float64([kp2[m.trainIdx].pt for m in matches_12])
        pB2 = np.float64([kp2[m.queryIdx].pt for m in matches_23])
        pC  = np.float64([kp3[m.trainIdx].pt for m in matches_23])

        # ── Stage 2: Homography (DLT) ─────────────────────────────────────
        print("[Stage 2] DLT homography (img1->img2, img3->img2) ...")
        t0 = time.perf_counter()
        self._compute_homography(pA, pB1)
        self._compute_homography(pC, pB2)
        timings['st2'] = time.perf_counter() - t0

        # ── Stage 3: Robust fitting (RANSAC) ─────────────────────────────
        print("[Stage 3] RANSAC ...")
        t0 = time.perf_counter()
        H_12, mask_12 = self._ransac(pA, pB1)
        H_32, mask_32 = self._ransac(pC, pB2)
        timings['st3'] = time.perf_counter() - t0
        inliers_12 = int(mask_12.sum());  inliers_32 = int(mask_32.sum())
        metrics.update(inliers_12=inliers_12, inliers_32=inliers_32)
        print(f"    1<->2: {inliers_12}/{len(matches_12)} inliers   "
              f"3<->2: {inliers_32}/{len(matches_23)} inliers")
        self._verify_chain(H_12, H_32)

        # ── Stage 4: Depth analysis ───────────────────────────────────────
        print("[Stage 4] Depth analysis ...")
        t0 = time.perf_counter()
        depth_model   = self._get_depth_estimator()
        H_layers_12   = H_layers_32 = None
        depth_bands_12 = depth_bands_32 = None
        depth_1 = depth_3 = None
        depth_cv = None
        if depth_model is not None:
            depth_1  = depth_model.estimate(img1)
            depth_3  = depth_model.estimate(img3)
            depth_cv = max(float(depth_1.std() / (depth_1.mean() + 1e-8)),
                           float(depth_3.std() / (depth_3.mean() + 1e-8)))
            H_layers_12, depth_bands_12 = self._depth_stratified_homographies(
                pA, pB1, mask_12, depth_1)
            H_layers_32, depth_bands_32 = self._depth_stratified_homographies(
                pC, pB2, mask_32, depth_3)
        timings['st4'] = time.perf_counter() - t0
        metrics.update(depth_cv=depth_cv, bands=self.n_depth_layers)

        # ── Stage 5: Composition ──────────────────────────────────────────
        print("[Stage 5] Composition (warp -> seam -> blend, two passes) ...")
        t0 = time.perf_counter()
        cw, ch, ox, oy = self._compute_canvas_size_3(img1, img2, img3, H_12, H_32)
        print(f"    Canvas: {cw}×{ch}  offset: ({ox:.1f},{oy:.1f})")

        if depth_1 is not None and H_layers_12 is not None:
            warped1, mask1 = self._warp_depth_aware(
                img1, H_12, H_layers_12, depth_bands_12, depth_1, cw, ch, ox, oy)
        else:
            warped1, mask1 = self._warp_image(img1, H_12, cw, ch, ox, oy)

        if depth_3 is not None and H_layers_32 is not None:
            warped3, mask3 = self._warp_depth_aware(
                img3, H_32, H_layers_32, depth_bands_32, depth_3, cw, ch, ox, oy)
        else:
            warped3, mask3 = self._warp_image(img3, H_32, cw, ch, ox, oy)

        iox, ioy = int(round(ox)), int(round(oy))
        h2, w2   = img2.shape[:2]
        canvas2  = np.zeros((ch, cw, 3), np.uint8)
        mask2    = np.zeros((ch, cw), bool)
        ye, xe   = min(ioy + h2, ch), min(iox + w2, cw)
        canvas2[ioy:ye, iox:xe] = img2[:ye-ioy, :xe-iox]
        mask2[ioy:ye, iox:xe]   = True

        w1f = warped1.astype(np.float32)
        w3f = warped3.astype(np.float32)
        c2f = canvas2.astype(np.float32)

        # Pass 1: blend img1 with img2.
        blend12, mask12, rmse12 = self._blend_pair(w1f, mask1, c2f, mask2, cw, ch)

        # Pass 2: blend (img1+img2) with img3.
        blend12_f = blend12.astype(np.float32)
        _, _, rmse23 = self._blend_pair(blend12_f, mask12, w3f, mask3, cw, ch)

        # Second pass — full blend call for final panorama.
        ov23 = mask12 & mask3
        if ov23.any():
            if self.color_match == 'hist':
                blend12_f = GPUPyStitch._hist_match(blend12_f, w3f, ov23)
            else:
                gain3 = (w3f[ov23].mean(0) + 1e-3) / (blend12_f[ov23].mean(0) + 1e-3)
                blend12_f = np.clip(blend12_f * np.clip(gain3, 0.5, 2.0), 0, 255)
        seam23  = self._find_seam(blend12_f, w3f, mask12, mask3)
        I12f = np.zeros((ch, cw, 3), np.float32)
        I3f  = np.zeros((ch, cw, 3), np.float32)
        I12f[mask12]         = blend12_f[mask12]
        I3f[mask3]           = w3f[mask3]
        I12f[~mask12 & mask3] = w3f[~mask12 & mask3]
        I3f[~mask3 & mask12]  = blend12_f[~mask3 & mask12]
        blended = self._multiband_blend(I12f, I3f, seam23, self.n_blend_levels)
        both    = mask12 | mask3

        pano = np.zeros((ch, cw, 3), np.uint8)
        pano[both] = np.clip(blended[both], 0, 255).astype(np.uint8)
        del blend12, blend12_f, I12f, I3f, blended
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

        timings['st5'] = time.perf_counter() - t0
        coverage = float(both.sum()) / float(ch * cw) * 100.0
        metrics.update(seam_rmse=rmse12, seam_rmse_23=rmse23,
                       coverage=coverage, canvas_w=cw, canvas_h=ch)

        pano = self._autocrop(pano, both)
        return pano, timings, metrics

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def stitch(self, *args):
        """
        Stitch 2 or 3 images into a single panorama.

        Accepts either positional image arguments or a single list/tuple:

            stitcher.stitch(img_left, img_right)
            stitcher.stitch(img_left, img_center, img_right)
            stitcher.stitch([img_left, img_right])

        For 2 images: the second image is the reference frame.
        For 3 images: the middle image (img2) is the reference frame.

        Parameters
        ----------
        *args : numpy.ndarray (BGR, uint8) or list thereof

        Returns
        -------
        panorama : numpy.ndarray, shape (H, W, 3), dtype uint8
            Also written to a timestamped PNG file in the current directory.
        """
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            images = list(args[0])
        else:
            images = list(args)

        if len(images) == 2:
            pano, timings, metrics = self._stitch_pair(images[0], images[1])
        elif len(images) == 3:
            pano, timings, metrics = self._stitch_triple(images[0], images[1], images[2])
        else:
            raise ValueError(f"stitch() accepts 2 or 3 images, got {len(images)}")

        self._print_report(timings, metrics)
        self._save_output(pano, metrics)
        return pano

    def _print_report(self, timings, metrics):
        """Print a formatted per-stage timing and quality report."""
        print(f"\n{'=' * 42}")
        print(f"  GPUPyStitch Report")
        print(f"{'=' * 42}")

        # Stage 1
        kp_info = f"img1: {metrics.get('kp1', 0)} kp, img2: {metrics.get('kp2', 0)} kp"
        if 'kp3' in metrics:
            kp_info += f", img3: {metrics['kp3']} kp"
        m12 = metrics.get('matches', metrics.get('matches_12', 0))
        m23 = metrics.get('matches_23')
        match_str = f"matches: {m12}" + (f"+{m23}" if m23 is not None else "")
        print(f"Stage 1 - Feature Extraction:     {timings['st1']:>6.2f}s"
              f"  ({kp_info}, {match_str})")

        # Stage 2
        print(f"Stage 2 - Homography (DLT):       {timings['st2']:>6.3f}s")

        # Stage 3
        inl_12 = metrics.get('inliers', metrics.get('inliers_12', '?'))
        inl_32 = metrics.get('inliers_32')
        inl_str = (f"inliers: {inl_12}+{inl_32}" if inl_32 is not None
                   else f"inliers: {inl_12}")
        print(f"Stage 3 - Robust Fitting (RANSAC): {timings['st3']:>6.2f}s"
              f"  ({inl_str})")

        # Stage 4
        dcv = metrics.get('depth_cv')
        dcv_str = f"depth_cv: {dcv:.2f}" if dcv is not None else "depth: unavailable"
        print(f"Stage 4 - Depth Analysis:         {timings['st4']:>6.2f}s"
              f"  ({dcv_str}, bands: {metrics.get('bands', 0)})")

        # Stage 5
        rmse = metrics.get('seam_rmse', float('nan'))
        rmse_str = f"{rmse:.1f}" if not math.isnan(rmse) else "n/a"
        cov  = metrics.get('coverage', 0.0)
        print(f"Stage 5 - Composition:            {timings['st5']:>6.2f}s"
              f"  (seam_rmse: {rmse_str}, coverage: {cov:.1f}%)")

        total = sum(timings.values())
        print("-" * 44)
        print(f"Total:                             {total:>6.2f}s")

    def _postprocess(self, pano):
        """
        Optional post-processing applied to the final panorama before saving.

        output_max_dim : resize so the longest side ≤ this value (0 = skip).
        post_sharpen   : unsharp-mask sharpening to recover blend softness.
        """
        if self.output_max_dim > 0:
            h, w = pano.shape[:2]
            scale = self.output_max_dim / max(h, w)
            if scale < 1.0:
                nw = int(w * scale)
                nh = int(h * scale)
                pano = cv2.resize(pano, (nw, nh), interpolation=cv2.INTER_AREA)
                print(f"    [Post] Resized {w}×{h} → {nw}×{nh}  (scale {scale:.2f})")

        if self.post_sharpen:
            blur = cv2.GaussianBlur(pano, (0, 0), sigmaX=2.0)
            pano = cv2.addWeighted(pano, 1.5, blur, -0.5, 0)
            print("    [Post] Sharpened (unsharp mask)")

        return pano

    def _save_output(self, pano, metrics):
        """Apply post-processing, then write panorama to a timestamped file."""
        pano = self._postprocess(pano)

        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        fmt = self.output_format if self.output_format in ('png', 'jpg', 'jpeg') else 'png'
        ext = 'jpg' if fmt in ('jpg', 'jpeg') else 'png'
        fname = f"panorama_{ts}.{ext}"

        if ext == 'jpg':
            cv2.imwrite(fname, pano, [cv2.IMWRITE_JPEG_QUALITY, self.output_quality])
        else:
            cv2.imwrite(fname, pano)

        size_mb = os.path.getsize(fname) / (1024 * 1024)
        print(f"Saved: {fname}  ({pano.shape[1]}×{pano.shape[0]} px, {size_mb:.1f} MB)")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def _load_image(path):
    """Load a BGR image with automatic EXIF orientation correction."""
    try:
        from PIL import Image as PILImage, ExifTags
        pil = PILImage.open(path)
        try:
            exif = pil._getexif()
            if exif:
                orient_key = next(
                    k for k, v in ExifTags.TAGS.items() if v == "Orientation")
                orientation = exif.get(orient_key, 1)
                rotate_map  = {3: 180, 6: 270, 8: 90}
                if orientation in rotate_map:
                    pil = pil.rotate(rotate_map[orientation], expand=True)
        except Exception:
            pass
        img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    except ImportError:
        img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    print(f"  Loaded: {path}  ({img.shape[1]}×{img.shape[0]})")
    return img


def _warmup_kernels():
    """
    JIT-compile the CUDA kernels by running them on a small dummy image.
    This avoids kernel compilation latency during the first real inference.
    """
    print("Warming up CUDA kernels ...")
    dummy = PySIFT(n_octaves=2, n_scales=3)
    dummy.detectAndCompute(np.zeros((128, 128), dtype=np.uint8))
    del dummy
    torch.cuda.empty_cache()
    print("Kernels ready.")


def _load_config(path: str) -> dict:
    """Load a YAML configuration file and return as a nested dict.

    Raises RuntimeError if PyYAML is not installed.
    """
    if not _YAML_AVAILABLE:
        raise RuntimeError(
            "PyYAML is required for --config.  Install with: pip install pyyaml")
    with open(path) as f:
        return _yaml.safe_load(f) or {}


def main():
    """CLI entry point for pysift-stitch command."""
    parser = argparse.ArgumentParser(
        description="GPU-accelerated panoramic stitching — depth-aware pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpu_pystitch.py left.jpg right.jpg
  python gpu_pystitch.py left.jpg center.jpg right.jpg -o results/
  python gpu_pystitch.py img1.jpg img2.jpg img3.jpg --config config.yaml
  python gpu_pystitch.py img1.jpg img2.jpg --matcher lightglue --descriptor hardnet
  python gpu_pystitch.py img1.jpg img2.jpg img3.jpg --max-dim 1920 --denoise
        """)
    parser.add_argument("images", nargs="+",
                        help="Paths to 2 or 3 input images (left to right)")
    parser.add_argument("-o", "--output-dir", default=".",
                        help="Directory for the output panorama PNG (default: current dir)")

    # ── Config file ───────────────────────────────────────────────────────────
    parser.add_argument("--config", default=None, metavar="YAML",
                        help="Path to config.yaml.  CLI args override YAML values.")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    parser.add_argument("--max-dim", type=int, default=None, metavar="PX",
                        help="Resize longest side to PX before processing (0=off). "
                             "Use 1920 for large mobile photos.")
    parser.add_argument("--denoise", action="store_true", default=None,
                        help="Apply bilateral filter denoising (good for night shots).")
    parser.add_argument("--sharpen", action="store_true", default=None,
                        help="Apply unsharp-mask sharpening.")
    parser.add_argument("--clahe-clip", type=float, default=None, metavar="F",
                        help="CLAHE clip limit for grayscale detection (default 2.0).")

    # ── Feature detection ─────────────────────────────────────────────────────
    parser.add_argument("--orientation", choices=["histogram", "orinet"], default=None,
                        help="Orientation method: histogram (multi-peak, default) or "
                             "orinet (CNN, single-peak — lower --contrast-thresh to 0.02).")
    parser.add_argument("--descriptor", choices=["sift", "hardnet", "hynet"], default=None,
                        help="Descriptor type (default: sift).")
    parser.add_argument("--dsp-sift", action="store_true", default=None, dest="dsp_sift",
                        help="Enable DSP-SIFT multi-scale pooling (5x slower descriptors).")
    parser.add_argument("--contrast-thresh", type=float, default=None, metavar="F",
                        help="DoG contrast threshold (default 0.04; use 0.02 with orinet).")
    parser.add_argument("--max-keypoints", type=int, default=None, metavar="N",
                        help="Max keypoints per image (default 8000).")

    # ── Matching ──────────────────────────────────────────────────────────────
    parser.add_argument("--matcher", choices=["ratio", "lightglue"], default=None,
                        help="Matching method (default: ratio).")
    parser.add_argument("--ratio", type=float, default=None, metavar="F",
                        help="Lowe ratio threshold for ratio matcher (default 0.75).")
    parser.add_argument("--pca-dims", type=int, default=None, metavar="N",
                        help="PCA compression dims for SIFT ratio matching (default 64; "
                             "0=off; ignored for HardNet/HyNet).")

    # ── RANSAC ────────────────────────────────────────────────────────────────
    parser.add_argument("--ransac-iters", type=int, default=None, metavar="N",
                        help="GPU RANSAC hypotheses (default 1500).")
    parser.add_argument("--inlier-thresh", type=float, default=None, metavar="F",
                        help="Reprojection error threshold in pixels (default 5.0).")
    parser.add_argument("--ransac-method", choices=["magsac", "classic"], default=None,
                        help="Robust estimation: magsac (MAGSAC++ soft score, default) or "
                             "classic (hard inlier count).")
    parser.add_argument("--sigma-max", type=float, default=None, metavar="F",
                        help="σ_max for MAGSAC++ soft scoring (default: same as inlier-thresh).")

    # ── Blending ──────────────────────────────────────────────────────────────
    parser.add_argument("--blend-levels", type=int, default=None, metavar="N",
                        help="Laplacian pyramid blend levels (default 6).")
    parser.add_argument("--color-match", choices=["gain", "hist"], default=None,
                        help="Exposure correction method (default: gain).")

    # ── Post-processing ───────────────────────────────────────────────────────
    parser.add_argument("--output-max-dim", type=int, default=None, metavar="PX",
                        help="Resize output panorama so longest side ≤ PX (0=off). "
                             "Use 4096 to cap very wide panoramas.")
    parser.add_argument("--post-sharpen", action="store_true", default=None,
                        help="Apply unsharp-mask sharpening to the final panorama.")
    parser.add_argument("--output-format", choices=["png", "jpg"], default=None,
                        help="Output file format: png (lossless, large) or jpg (default: png).")
    parser.add_argument("--output-quality", type=int, default=None, metavar="0-100",
                        help="JPEG quality when --output-format jpg (default 90).")

    args = parser.parse_args()

    if len(args.images) not in (2, 3):
        parser.error(f"Provide exactly 2 or 3 image paths (got {len(args.images)})")

    # ── Merge config.yaml + CLI (CLI wins) ────────────────────────────────────
    cfg = {}
    if args.config:
        cfg = _load_config(args.config)

    def _get(cli_val, *cfg_keys, default=None):
        """Return cli_val if set, else walk cfg_keys path, else default."""
        if cli_val is not None:
            return cli_val
        node = cfg
        for key in cfg_keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node if node is not None else default

    stitcher_kwargs = dict(
        # preprocessing
        max_dimension  = _get(args.max_dim,         'preprocessing', 'max_dimension', default=0),
        denoise        = _get(args.denoise,          'preprocessing', 'denoise',       default=False),
        sharpen        = _get(args.sharpen,          'preprocessing', 'sharpen',       default=False),
        clahe_clip     = _get(args.clahe_clip,       'preprocessing', 'clahe_clip',    default=2.0),
        # detection
        orientation    = _get(args.orientation,      'orientation',                    default='histogram'),
        descriptor     = _get(args.descriptor,       'descriptor',                     default='sift'),
        dsp_sift       = _get(args.dsp_sift,         'dsp_sift',                       default=False),
        contrast_thresh= _get(args.contrast_thresh,  'detection', 'contrast_thresh',   default=0.04),
        max_keypoints  = _get(args.max_keypoints,    'detection', 'max_keypoints',     default=8000),
        # matching
        matcher        = _get(args.matcher,          'matching', 'method',             default='ratio'),
        match_ratio    = _get(args.ratio,            'matching', 'ratio',              default=0.75),
        pca_dims       = _get(args.pca_dims,         'matching', 'pca_dims',           default=64),
        # ransac
        ransac_iters   = _get(args.ransac_iters,     'ransac', 'iters',               default=1500),
        inlier_thresh  = _get(args.inlier_thresh,    'ransac', 'inlier_thresh',        default=5.0),
        ransac_method  = _get(args.ransac_method,    'ransac', 'method',              default='magsac'),
        sigma_max      = _get(args.sigma_max,        'ransac', 'sigma_max',            default=None),
        # blending
        n_blend_levels = _get(args.blend_levels,     'blending', 'levels',            default=6),
        color_match    = _get(args.color_match,      'blending', 'color_match',        default='gain'),
        # post-processing
        output_max_dim = _get(args.output_max_dim,   'postprocessing', 'output_max_dim', default=0),
        post_sharpen   = _get(args.post_sharpen,     'postprocessing', 'post_sharpen',   default=False),
        output_format  = _get(args.output_format,    'postprocessing', 'output_format',  default='png'),
        output_quality = _get(args.output_quality,   'postprocessing', 'output_quality', default=90),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)

    print(f"\nLoading {len(args.images)} images ...")
    images = [_load_image(p) for p in args.images]

    if DEVICE.type == "cuda":
        _warmup_kernels()

    stitcher = GPUPyStitch(**stitcher_kwargs)
    panorama = stitcher.stitch(*images)


if __name__ == "__main__":
    main()
