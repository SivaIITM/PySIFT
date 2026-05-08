"""
quick_test.py -- Fast single-image sanity check for PySIFT descriptor modes.

Usage:
  python quick_test.py <image_path> [--mode native|dsp|hardnet|hynet] [--runs N]

Examples:
  python quick_test.py photo.jpg
  python quick_test.py photo.jpg --mode hardnet --runs 10
  python quick_test.py photo.jpg --mode dsp --profile
"""
import argparse
import time

import cv2
import numpy as np

from pysift import PySIFT


MODE_CONFIG = {
    "native":  dict(dsp=False, descriptor="sift"),
    "dsp":     dict(dsp=True,  descriptor="sift"),
    "hardnet": dict(dsp=False, descriptor="hardnet"),
    "hynet":   dict(dsp=False, descriptor="hynet"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Quick single-image test for PySIFT descriptor modes")
    parser.add_argument("image", help="Path to grayscale or colour image")
    parser.add_argument("--mode", choices=list(MODE_CONFIG.keys()),
                        default="native", help="Descriptor mode (default: native)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of timed runs after warmup (default: 5)")
    parser.add_argument("--profile", action="store_true",
                        help="Print per-phase timing breakdown")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    cfg = MODE_CONFIG[args.mode]
    sift = PySIFT(**cfg)
    print(f"Mode: PySIFT-{args.mode.upper()}  dsp={sift.dsp}  descriptor={sift.descriptor}")
    print(f"Image: {args.image}  size={img.shape[0]}x{img.shape[1]}")

    # JIT warmup
    warm = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    sift.detectAndCompute(warm)
    sift.detectAndCompute(warm)

    # Timed runs
    times = []
    kps, descs = None, None
    for i in range(args.runs):
        do_profile = args.profile and (i == args.runs - 1)
        t0 = time.perf_counter()
        kps, descs = sift.detectAndCompute(img, profile=do_profile)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    med = np.median(times)
    print(f"\nKeypoints: {len(kps)}")
    print(f"Descriptors: {descs.shape}  dtype={descs.dtype}")
    print(f"Timing ({args.runs} runs): median={med:.1f}ms")
    print(f"  runs: {[f'{t:.1f}' for t in times]}")

    # OpenCV comparison
    ocv = cv2.SIFT_create()
    ocv_times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        ocv_kps, _ = ocv.detectAndCompute(img, None)
        t1 = time.perf_counter()
        ocv_times.append((t1 - t0) * 1000)
    ocv_med = np.median(ocv_times)
    print(f"\nOpenCV SIFT: {len(ocv_kps)} kps, median={ocv_med:.1f}ms")
    speedup = ocv_med / med
    print(f"PySIFT speedup: {speedup:.2f}x {'FASTER' if speedup > 1 else 'slower'}")


if __name__ == "__main__":
    main()
