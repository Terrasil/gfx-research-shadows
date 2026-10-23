#!/usr/bin/env python3
"""Check paired native captures for the crossing-shadow regression."""
import argparse
import json
import math
import sys
from pathlib import Path

from analyze_results import analyze, read_csv, read_pfm


def check(before, after):
    manifests, rows, images, references, masks = [], [], [], [], []
    for root in (before, after):
        summary, manifest = analyze(root)
        if not summary["reference_converged"]:
            raise ValueError("Reference convergence is required")
        selection = [r for r in read_csv(root / "raw.csv") if r["method"] == "coherent-ndc-pcss"]
        if len(selection) != 1:
            raise ValueError("Each run must contain exactly one coherent-PCSS configuration")
        row = selection[0]
        config = row["case_id"].removesuffix("-coherent-ndc-pcss")
        manifests.append(manifest)
        rows.append(row)
        images.append(read_pfm(root / "captures" / (row["case_id"] + ".pfm"))[2])
        references.append(read_pfm(root / "captures" / (config + "-reference.pfm"))[2])
        masks.append(read_pfm(root / "captures" / (config + "-receiver.pfm"))[2])
    keys = ("gpu", "map_backend", "reference_samples", "filter_samples", "blocker_samples",
            "local_plane_capacity", "warp", "normal_bias", "depth_bias", "max_radius_world",
            "camera", "rotated_samples", "custom_camera", "camera_yaw", "camera_pitch",
            "camera_distance", "camera_target")
    if any(manifests[0][key] != manifests[1][key] for key in keys):
        raise ValueError("Runs do not share the same sampling, camera and representation settings")
    if any(rows[0][key] != rows[1][key] for key in
           ("scene", "width", "height", "map_resolution", "angle", "map_backend")):
        raise ValueError("Runs use different scene configurations")
    if manifests[0].get("contour_samples") != 0 or manifests[1].get("contour_samples", 0) <= 0:
        raise ValueError("Compare continuation disabled against continuation enabled")
    if len(images[0]) != len(images[1]) or masks[0] != masks[1] or any(
            abs(a-b) > 1e-6 for a, b in zip(*references)):
        raise ValueError("Runs do not share the same receiver mask and reference")
    umbra = [i for i, (r, m) in enumerate(zip(references[0], masks[0])) if m > 0.5 and r <= 0.01]
    if not umbra:
        raise ValueError("No full-shadow receiver pixels")
    leaks = [math.sqrt(sum(max(image[i] - references[0][i], 0.0)**2 for i in umbra) / len(umbra))
             for image in images]
    old, new = (float(row["rmse"]) for row in rows)
    old_pen, new_pen = (float(row["penumbra_rmse"]) for row in rows)
    result = {"passed": leaks[0] > 0 and leaks[1] <= leaks[0]*0.6 and new <= old*0.8 and
                         new_pen <= old_pen*1.05,
              "full_shadow_pixels": len(umbra), "before_rmse": old, "after_rmse": new,
              "before_positive_umbra_rmse": leaks[0], "after_positive_umbra_rmse": leaks[1],
              "before_penumbra_rmse": old_pen, "after_penumbra_rmse": new_pen,
              "scope": "Paired crossing regression; quality gates do not imply equal rendering cost"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    try:
        result = check(args.before, args.after)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    except (ValueError, KeyError, OSError, ZeroDivisionError) as error:
        print(f"Overlap regression failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
