# Benchmark Results of PySIFT

Raw benchmark data backing **Table 3** (Ablation) and **Table 2** (Cross-dataset) in the paper.

All runs on NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), May 6-7 2026.

## Table 3: Ablation Across Hybrid Configurations

| # | Orient. | Desc. | Match | MMA@10^H | mAA@10deg^I | AUC@10deg^M | FPS^I |
|---|---------|-------|-------|----------|-------------|-------------|-------|
| 1 | Hist. | CV-SIFT | Ratio/CPU | 0.897 | 0.506 | 0.232 | 5.54 |
| 2 | Hist. | **PySIFT*** | Ratio/TC | **0.919** | **0.517** | **0.288** | **10.74** |
| 3 | OriNet | PySIFT* | Ratio/TC | 0.897 | 0.464 | 0.253 | 3.42 |
| 4 | Hist. | HardNet | Ratio/TC | 0.892 | 0.387 | 0.189 | 6.11 |
| 5 | Hist. | PySIFT* | LightGlue | **0.921** | **0.517** | 0.286 | 11.09 |
| 6 | OriNet | HardNet | Ratio/TC | 0.913 | 0.377 | 0.171 | 2.86 |
| 7 | OriNet | HardNet | LightGlue | 0.571 | 0.378 | 0.172 | 2.87 |
| 8 | SuperPoint | -- | LightGlue | 0.975 | 0.485 | 0.216 | 20.16 |

**Legend:**
- *PySIFT* = DSP-SIFT multi-scale pooling + RootSIFT Hellinger norm
- ^H = HPatches (116 sequences, native resolution)
- ^I = IMC Phototourism (25,534 pairs, 9 scenes)
- ^M = MegaDepth-1500 (804 pairs)
- TC = Tensor Core fp16 matmul; CPU = OpenCV brute-force kNN on host

## Table 2: Cross-Dataset Summary (Config 1 vs Config 2)

| Dataset | Metric | OpenCV | PySIFT | Delta |
|---------|--------|--------|--------|-------|
| IMC Phototourism | Avg inliers | 205.4 | **303.0** | +47.5% |
| IMC Phototourism | Pose mAA@10deg | 0.506 | **0.517** | +1.1 pp |
| IMC Phototourism | Pipeline FPS | 5.54 | **10.74** | +93.9% |
| IMC Phototourism | Wall clock (s) | 4,604 | **2,377** | -48.4% |
| MegaDepth | Avg inliers | 127.2 | **172.4** | +35.6% |
| MegaDepth | AUC@10deg | 0.232 | **0.288** | +5.6 pp |
| MegaDepth | Per-pair (ms) | 655 | **272** | -383 ms |
| ROxford5K | mAP (Medium) | 0.449 | **0.524** | +7.5 pp |

## File Descriptions

| File | Config | Description |
|------|--------|-------------|
| `config_1_opencv_baseline.json` | OpenCV + CPU BF | Reference baseline (Lowe 2004 via OpenCV) |
| `config_2_pysift_classical.json` | PySIFT Classical | DSP-SIFT + histogram orientation + GPU Tensor Core matching |
| `config_3_pysift_orinet.json` | +OriNet | Learned orientation (kornia OriNet) |
| `config_4_pysift_hardnet.json` | +HardNet | Learned descriptor (kornia HardNet) |
| `config_5_pysift_lightglue.json` | +LightGlue | Attention-based matcher |
| `config_6_pysift_orinet_hardnet.json` | +OriNet+HardNet | Both learned components |
| `config_7_pysift_full_hybrid.json` | Full Hybrid | OriNet + HardNet + LightGlue |
| `config_8_superpoint.json` | SuperPoint | External CNN baseline (DeTone et al. 2018) |

## JSON Schema

Each file contains:
```json
{
  "meta": { "extractor", "orientation", "descriptor", "matcher", "cuda_device", ... },
  "hpatches": { "native": { "MMA@1px"..."MMA@10px", "Rep@3px", "mAA@10px", "AvgCornerErr", "t_detect_ms", "t_match_ms" } },
  "imc": { "avg_inliers", "n_pairs", "pipeline_fps", "pose_maa_10deg", "total_wall_clock_s" },
  "megadepth": { "avg_inliers", "n_pairs", "pipeline_fps", "auc_10deg", "peak_vram_4k_mb" },
  "oxford5k": { "mAP", "query_time_ms", ... },
  "elapsed_s": <total_runtime_seconds>
}
```

## Reproducing

```bash
pip install staysift
python -c "from pysift import PySIFT; print(PySIFT)"
```

The benchmark script (`Benchmark_Pysift_HOIM.py`) runs all configurations end-to-end. Dataset paths must be configured for your system. See the paper for dataset download links.
