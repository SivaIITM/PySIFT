"""
cross_determinism_test.py -- Compare PySIFT outputs across two GPU devices.

Loads .npz reference files (generated on Device A = RTX 3050), re-extracts
on Device B, and reports statistical equivalence metrics.

Usage:
    python tests/cross_determinism_test.py \
        --reference-dir determinism_reference \
        --output-dir determinism_results \
        --hpatches-dir test_images/hpatches-sequences-release \
        --imc-dir test_images/imc_phototourism
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pysift import PySIFT


def find_image(manifest_entry, hpatches_dir, imc_dir):
    """Locate the test image on this machine using benchmark+name metadata."""
    bench = manifest_entry["benchmark"]
    name = manifest_entry["name"]

    if bench == "hpatches":
        img = Path(hpatches_dir) / name / "1.ppm"
        if img.exists():
            return str(img)
    elif bench == "imc":
        scene_dir = Path(imc_dir) / name / "set_100" / "images"
        if scene_dir.exists():
            ref_filename = Path(manifest_entry.get("source_path", "")).name
            if ref_filename:
                exact = scene_dir / ref_filename
                if exact.exists():
                    return str(exact)
            imgs = sorted([f for f in scene_dir.glob("*.jpg")
                           if not f.name.startswith("._")])
            if imgs:
                return str(imgs[0])
    return None


def compare_extractions(ref_data, new_data, tolerance_px=0.5):
    """Compare two sets of keypoints+descriptors statistically."""
    ref_pts = ref_data["points"]
    new_pts = new_data["points"]
    ref_desc = ref_data["descriptors"]
    new_desc = new_data["descriptors"]

    results = {
        "ref_n_keypoints": len(ref_pts),
        "new_n_keypoints": len(new_pts),
        "kp_count_diff": abs(len(ref_pts) - len(new_pts)),
        "kp_count_ratio": len(new_pts) / max(len(ref_pts), 1),
    }

    if len(ref_pts) == 0 or len(new_pts) == 0:
        results["matched_pct"] = 0.0
        results["verdict"] = "FAIL_EMPTY"
        return results

    from scipy.spatial import cKDTree
    tree = cKDTree(new_pts)
    dists, indices = tree.query(ref_pts, k=1)

    within_tol = dists <= tolerance_px
    matched_pct = np.mean(within_tol) * 100
    results["matched_within_0.5px_pct"] = round(matched_pct, 2)

    matched_mask = within_tol
    if np.any(matched_mask):
        ref_matched = ref_desc[matched_mask]
        new_matched = new_desc[indices[matched_mask]]

        desc_dists = np.linalg.norm(ref_matched - new_matched, axis=1)
        results["desc_l2_mean"] = round(float(np.mean(desc_dists)), 4)
        results["desc_l2_median"] = round(float(np.median(desc_dists)), 4)
        results["desc_l2_max"] = round(float(np.max(desc_dists)), 4)
        results["desc_l2_p95"] = round(float(np.percentile(desc_dists, 95)), 4)

        norms_r = np.linalg.norm(ref_matched, axis=1, keepdims=True) + 1e-8
        norms_n = np.linalg.norm(new_matched, axis=1, keepdims=True) + 1e-8
        cos_sim = np.sum((ref_matched / norms_r) * (new_matched / norms_n), axis=1)
        results["desc_cosine_mean"] = round(float(np.mean(cos_sim)), 4)
        results["desc_cosine_min"] = round(float(np.min(cos_sim)), 4)

        ref_angles = ref_data["angles"][matched_mask]
        new_angles = new_data["angles"][indices[matched_mask]]
        angle_diff = np.abs(ref_angles - new_angles)
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)
        results["angle_diff_mean_deg"] = round(float(np.mean(angle_diff)), 2)
        results["angle_diff_max_deg"] = round(float(np.max(angle_diff)), 2)

    img_h, img_w = ref_data["image_shape"][:2]
    nbins = 8
    ref_hist, _, _ = np.histogram2d(ref_pts[:, 0], ref_pts[:, 1],
                                    bins=nbins, range=[[0, img_w], [0, img_h]])
    new_hist, _, _ = np.histogram2d(new_pts[:, 0], new_pts[:, 1],
                                    bins=nbins, range=[[0, img_w], [0, img_h]])
    ref_hist = ref_hist / max(ref_hist.sum(), 1)
    new_hist = new_hist / max(new_hist.sum(), 1)
    spatial_corr = np.corrcoef(ref_hist.ravel(), new_hist.ravel())[0, 1]
    results["spatial_distribution_corr"] = round(float(spatial_corr), 4)

    if matched_pct >= 95 and results.get("desc_cosine_mean", 0) >= 0.99:
        results["verdict"] = "EQUIVALENT"
    elif matched_pct >= 85 and results.get("desc_cosine_mean", 0) >= 0.95:
        results["verdict"] = "CLOSE"
    elif matched_pct >= 70:
        results["verdict"] = "DIVERGENT_BUT_USABLE"
    else:
        results["verdict"] = "DIVERGENT"

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-device determinism test for PySIFT")
    parser.add_argument("--reference-dir", required=True,
                        help="Path to determinism_reference/ folder with .npz files")
    parser.add_argument("--output-dir", default="determinism_results",
                        help="Output directory for report JSON")
    parser.add_argument("--hpatches-dir", required=True,
                        help="Path to hpatches-sequences-release/")
    parser.add_argument("--imc-dir", required=True,
                        help="Path to imc_phototourism/")
    args = parser.parse_args()

    ref_dir = Path(args.reference_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ref_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest.json not found in {ref_dir}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Reference device: {manifest['device']}")
    print(f"Reference timestamp: {manifest['timestamp']}")
    print(f"Images to compare: {len(manifest['images'])}")
    print()

    import torch
    this_device = (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu")
    print(f"This device: {this_device}")
    print()

    params = manifest["pysift_params"]
    sift = PySIFT(
        n_octaves=params["n_octaves"],
        n_scales=params["n_scales"],
        sigma0=params["sigma0"],
        contrast_thresh=params["contrast_thresh"],
        edge_thresh=params["edge_thresh"],
        dsp=params["dsp"],
        dsp_n_scales=params["dsp_n_scales"],
        fp16_pyramid=params["fp16_pyramid"],
        orientation=params["orientation"],
        descriptor=params["descriptor"],
    )

    report = {
        "ref_device": manifest["device"],
        "this_device": this_device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [],
    }

    for entry in manifest["images"]:
        img_path = find_image(entry, args.hpatches_dir, args.imc_dir)
        if img_path is None:
            print(f"  SKIP [{entry['benchmark']}] {entry['name']} -- image not found")
            continue

        print(f"  [{entry['benchmark']}] {entry['name']}:")

        ref = dict(np.load(ref_dir / entry["file"]))

        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        kps, descs = sift.detectAndCompute(gray)
        new_data = {
            "points": np.array([[k.pt[0], k.pt[1]] for k in kps], dtype=np.float32),
            "sizes": np.array([k.size for k in kps], dtype=np.float32),
            "angles": np.array([k.angle for k in kps], dtype=np.float32),
            "responses": np.array([k.response for k in kps], dtype=np.float32),
            "octaves": np.array([k.octave for k in kps], dtype=np.int32),
            "descriptors": descs,
            "image_shape": np.array(gray.shape, dtype=np.int32),
        }

        comp = compare_extractions(ref, new_data)
        comp["benchmark"] = entry["benchmark"]
        comp["name"] = entry["name"]
        report["results"].append(comp)

        verdict = comp["verdict"]
        matched = comp.get("matched_within_0.5px_pct", 0)
        cos = comp.get("desc_cosine_mean", 0)
        spatial = comp.get("spatial_distribution_corr", 0)
        print(f"    KPs: {comp['ref_n_keypoints']} ref / {comp['new_n_keypoints']} new")
        print(f"    Matched <0.5px: {matched:.1f}%  |  Desc cosine: {cos:.4f}"
              f"  |  Spatial corr: {spatial:.4f}")
        print(f"    Verdict: {verdict}")
        print()

    verdicts = [r["verdict"] for r in report["results"]]
    print("=" * 60)
    print("SUMMARY")
    print(f"  Device A (reference): {manifest['device']}")
    print(f"  Device B (this):      {this_device}")
    print(f"  Images tested: {len(report['results'])}")
    for v in ["EQUIVALENT", "CLOSE", "DIVERGENT_BUT_USABLE", "DIVERGENT"]:
        count = verdicts.count(v)
        if count > 0:
            print(f"  {v}: {count}")
    print("=" * 60)

    out_file = out_dir / "determinism_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved: {out_file}")
    print("\nPlease send determinism_report.json back to Siva.")


if __name__ == "__main__":
    main()
