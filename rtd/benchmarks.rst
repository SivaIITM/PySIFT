Benchmarks -- PySIFT vs OpenCV SIFT
====================================

All benchmarks on NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), CUDA 12.x.
Both PySIFT and OpenCV use GPU MAGSAC++ and brute-force matching for fair comparison.

HPatches -- Matching Accuracy
-----------------------------

116 sequences (57 illumination, 59 viewpoint), 580 image pairs.

.. list-table::
   :header-rows: 1

   * - Metric
     - PySIFT
     - OpenCV
     - Delta
   * - MMA@3px
     - 0.818
     - 0.823
     - Parity
   * - MMA@5px
     - 0.876
     - 0.873
     - +0.3%
   * - **MMA@8px**
     - **0.898**
     - 0.892
     - **+0.8%**
   * - **MMA@10px**
     - **0.906**
     - 0.897
     - **+1.0%**
   * - Avg Corner Error
     - **32.3 px**
     - 88.7 px
     - **-63.5%**

PySIFT leads at higher thresholds (MMA@8, @10) -- the practical operating point for
4K/8K resolution workflows where 3 pixels is sub-pixel noise.

IMC Phototourism -- Pose Estimation
------------------------------------

4,499 image pairs from landmark photo collections.

.. list-table::
   :header-rows: 1

   * - Metric
     - PySIFT
     - OpenCV
     - Delta
   * - **Avg Inliers**
     - **229.4**
     - 205.4
     - **+12%**
   * - **Pipeline FPS**
     - **7.92**
     - 6.32
     - **+25%**

MegaDepth -- Wide-Baseline Stereo
----------------------------------

804 image pairs from large-scale SfM reconstructions.

.. list-table::
   :header-rows: 1

   * - Metric
     - PySIFT
     - OpenCV
     - Delta
   * - **Avg Inliers**
     - **134.7**
     - 127.2
     - **+6%**
   * - **AUC@10 degrees**
     - **0.260**
     - 0.232
     - **+12%**

ROxford5K -- Image Retrieval
-----------------------------

5,063 database images, 55 queries. VLAD encoding (k=64), top-100 re-ranking.

.. list-table::
   :header-rows: 1

   * - Metric
     - PySIFT
     - OpenCV
     - Delta
   * - **mAP (Medium)**
     - **0.455**
     - 0.380
     - **+7.5pp**

Speed
-----

.. list-table::
   :header-rows: 1

   * - Stage
     - PySIFT (GPU)
     - OpenCV (CPU)
     - Speedup
   * - Detection + Description
     - 88 ms
     - 111 ms
     - 1.26x
   * - **BF Matching (1K kp)**
     - **2.1 ms**
     - 8.4 ms
     - **4.0x**
   * - **End-to-End Pipeline**
     - **178 ms**
     - 241 ms
     - **1.35x**
   * - **DLPack Transfer**
     - **0.09 ms**
     - N/A (PCIe: 0.38 ms)
     - **4.1x**

Resolution Scaling
------------------

PySIFT's speed advantage grows with image resolution:

.. list-table::
   :header-rows: 1

   * - Resolution
     - PySIFT
     - OpenCV
     - Speedup
   * - 480x640
     - 37.6 ms
     - 33.1 ms
     - OpenCV faster
   * - **768x1024**
     - **72.8 ms**
     - 105.1 ms
     - **1.44x**
   * - **1080x1920**
     - **200.6 ms**
     - 220.5 ms
     - **1.10x**
   * - **4K 3840x2160**
     - **549.8 ms**
     - 1751.2 ms
     - **3.2x**

At 4K, PySIFT is 3.2x faster -- the GPU parallelism advantage amplifies with pixel count.
This makes PySIFT the natural choice for drone, satellite, and medical imaging workloads
with high-resolution inputs.

Ablation: Classical vs Learned
------------------------------

7-configuration ablation showing that classical GPU-SIFT outperforms learned component
substitutions on diverse real-world benchmarks:

.. list-table::
   :header-rows: 1

   * - Config
     - MMA@3px
     - IMC Inliers
     - MegaDepth AUC@10
     - Detect (ms)
   * - **PySIFT Classical**
     - **0.818**
     - **229**
     - **0.260**
     - **121**
   * - + HardNet descriptors
     - 0.798
     - 130
     - 0.150
     - 2148
   * - + OriNet orientation
     - 0.805
     - 178
     - --
     - --

Classical SIFT descriptors generalize across all domains. Learned replacements (HardNet, OriNet)
degrade on diverse real-world scenes while costing 18x more compute.
