"""
stitch_demo.py -- Minimal panoramic stitching with PySIFT.

Usage:
  python stitch_demo.py left.jpg right.jpg
  python stitch_demo.py left.jpg center.jpg right.jpg -o output/
"""
import argparse
import os

import cv2

from pysift import GPUPyStitch


def main():
    parser = argparse.ArgumentParser(description="Stitch 2-3 images into a panorama")
    parser.add_argument("images", nargs="+", help="2 or 3 input images (left to right)")
    parser.add_argument("-o", "--output-dir", default=".", help="Output directory")
    args = parser.parse_args()

    if len(args.images) not in (2, 3):
        parser.error("Provide exactly 2 or 3 image paths")

    images = [cv2.imread(p) for p in args.images]
    for i, img in enumerate(images):
        if img is None:
            raise FileNotFoundError(f"Cannot read: {args.images[i]}")

    stitcher = GPUPyStitch()
    panorama = stitcher.stitch(*images)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "panorama.png")
    cv2.imwrite(out_path, panorama)
    print(f"Panorama saved: {out_path}  ({panorama.shape[1]}x{panorama.shape[0]})")


if __name__ == "__main__":
    main()
