import cv2
from pysift import PySIFT, GPUPyStitch

# Feature extraction
sift = PySIFT()
img = cv2.imread("WImage1.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("WImage1.jpg not found in current directory")
keypoints, descriptors = sift.detectAndCompute(img)
print(f"Keypoints: {len(keypoints)}, Descriptors: {descriptors.shape}")

# Panoramic stitching (2 or 3 images) — accepts file paths directly
stitcher = GPUPyStitch()
panorama = stitcher.stitch("WImage1.jpg", "WImage2.jpg", "WImage3.jpg")