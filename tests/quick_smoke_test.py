"""
quick_smoke_test.py -- Verify PySIFT installs and runs on this GPU.

No datasets required -- uses the WImage files bundled in the repo.

Usage:
    python tests/quick_smoke_test.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from pysift import PySIFT, GPUPyStitch


def check_gpu():
    if not torch.cuda.is_available():
        print("FAIL: No CUDA GPU detected. Install PyTorch with CUDA support.")
        sys.exit(1)
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu}  VRAM: {vram:.1f} GB")
    return gpu


def test_feature_extraction(img_path):
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"  SKIP: Cannot read {img_path}")
        return None

    sift = PySIFT(dsp=True, fp16_pyramid=True)

    # Warmup
    warm = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    sift.detectAndCompute(warm)

    # Timed extraction
    t0 = time.perf_counter()
    kps, descs = sift.detectAndCompute(gray)
    t1 = time.perf_counter()

    print(f"  {img_path.name}: {len(kps)} keypoints, desc shape={descs.shape}, "
          f"time={((t1-t0)*1000):.1f}ms")

    # Determinism check: run twice, compare
    kps2, descs2 = sift.detectAndCompute(gray)
    pts1 = np.array([[k.pt[0], k.pt[1]] for k in kps], dtype=np.float32)
    pts2 = np.array([[k.pt[0], k.pt[1]] for k in kps2], dtype=np.float32)
    if len(kps) == len(kps2) and np.array_equal(pts1, pts2) and np.array_equal(descs, descs2):
        print(f"    Determinism: PASS (bitwise identical on re-run)")
    else:
        print(f"    Determinism: WARN (run1={len(kps)} kps, run2={len(kps2)} kps)")

    return {"n_keypoints": len(kps), "desc_shape": descs.shape, "time_ms": (t1-t0)*1000}


def test_stitching(img_paths):
    images = [cv2.imread(str(p)) for p in img_paths]
    for i, img in enumerate(images):
        if img is None:
            print(f"  SKIP: Cannot read {img_paths[i]}")
            return None

    stitcher = GPUPyStitch()
    t0 = time.perf_counter()
    panorama = stitcher.stitch(*images)
    t1 = time.perf_counter()
    print(f"  Stitching {len(images)} images: output={panorama.shape[1]}x{panorama.shape[0]}, "
          f"time={((t1-t0)*1000):.0f}ms")

    out_path = Path("smoke_test_panorama.png")
    cv2.imwrite(str(out_path), panorama)
    print(f"  Panorama saved: {out_path}")
    return {"shape": panorama.shape, "time_ms": (t1-t0)*1000}


def main():
    print("=" * 60)
    print("PySIFT Smoke Test")
    print("=" * 60)

    gpu = check_gpu()
    print()

    repo_root = Path(__file__).resolve().parent.parent
    wimages = [repo_root / f"WImage{i}.jpg" for i in range(1, 4)]
    missing = [w for w in wimages if not w.exists()]
    if missing:
        print(f"ERROR: Missing test images: {[str(m) for m in missing]}")
        print(f"Expected in: {repo_root}")
        sys.exit(1)

    print("--- Feature Extraction ---")
    results = {}
    for w in wimages:
        r = test_feature_extraction(w)
        if r:
            results[w.name] = r

    print()
    print("--- Stitching ---")
    stitch_result = test_stitching(wimages)

    print()
    print("=" * 60)
    print("SMOKE TEST SUMMARY")
    print(f"  GPU: {gpu}")
    print(f"  Images tested: {len(results)}")
    for name, r in results.items():
        print(f"    {name}: {r['n_keypoints']} kps, {r['time_ms']:.1f}ms")
    if stitch_result:
        print(f"  Stitching: OK ({stitch_result['time_ms']:.0f}ms)")
    else:
        print(f"  Stitching: SKIPPED")
    print("=" * 60)

    all_pass = len(results) == 3 and stitch_result is not None
    print(f"\nOverall: {'PASS' if all_pass else 'PARTIAL'}")
    if all_pass:
        print("PySIFT is working on this GPU. Proceed to cross_determinism_test.py")


if __name__ == "__main__":
    main()
