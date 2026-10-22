#!/usr/bin/env python3
"""Validate a complete run and summarize measurements without inventing results."""
import argparse
import array
import csv
import html
import json
import math
from pathlib import Path
import statistics
import struct
import sys


def read_pfm(path):
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"Pf":
            raise ValueError(f"Expected grayscale PFM: {path}")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        if scale == 0 or width < 1 or height < 1:
            raise ValueError(f"Invalid PFM header: {path}")
        values = array.array("f")
        values.frombytes(stream.read())
        if (scale < 0) != (sys.byteorder == "little"):
            values.byteswap()
        if len(values) != width * height or any(not math.isfinite(v) for v in values):
            raise ValueError(f"Invalid PFM payload: {path}")
        return width, height, values


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def analyze(root, require_hardware=False, publication=False, validate_captures=True):
    root = Path(root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("completed") is not True:
        raise ValueError("Run is incomplete")
    if (require_hardware or publication) and manifest.get("software_renderer"):
        raise ValueError("Software rasterizer timings cannot be published as hardware GPU results")
    if publication and manifest.get("profile") != "publication":
        raise ValueError("Publication validation requires the full publication profile")
    rows = read_csv(root / "raw.csv")
    if not rows or len(rows) != manifest["expected_cases"]:
        raise ValueError("Case matrix is incomplete")
    by_id = {row["case_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate case IDs")
    numeric = ("map_gpu_ms", "shadow_gpu_ms", "total_gpu_ms", "rmse", "penumbra_rmse",
               "gradient_rmse", "max_abs_error", "monotonic_max_brightening", "monotonic_violation_fraction")
    for row in rows:
        for key in numeric:
            value = float(row[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key} in {row['case_id']}")
        if int(row["invalid_pixels"]) or int(row["receiver_pixels"]) <= 0:
            raise ValueError(f"Invalid image metrics in {row['case_id']}")
        if row["method"] == "reference" and float(row["rmse"]) > 1e-6:
            raise ValueError("Reference is not identical to its saved comparison image")
    timings = read_csv(root / "timings.csv")
    grouped = {key: [] for key in by_id}
    for timing in timings:
        case = timing["case_id"]
        if case not in grouped:
            raise ValueError(f"Unexpected timing case {case}")
        for key in ("map_gpu_ms", "shadow_gpu_ms", "total_gpu_ms"):
            if not math.isfinite(float(timing[key])) or float(timing[key]) < 0:
                raise ValueError("Invalid timestamp result")
        if abs(float(timing["map_gpu_ms"]) + float(timing["shadow_gpu_ms"]) - float(timing["total_gpu_ms"])) > 1e-5:
            raise ValueError("Timing scopes do not add up")
        grouped[case].append(timing)
    for case, values in grouped.items():
        count = int(by_id[case]["timing_samples"])
        if len(values) != count or sorted(int(v["sample"]) for v in values) != list(range(count)):
            raise ValueError(f"Missing or duplicate timing samples in {case}")
        for key in ("map_gpu_ms", "shadow_gpu_ms", "total_gpu_ms"):
            actual = statistics.median(float(v[key]) for v in values)
            if abs(actual - float(by_id[case][key])) > max(1e-5, actual * 1e-6):
                raise ValueError(f"Timing median mismatch in {case}")
    convergence = read_csv(root / "reference_convergence.csv")
    if not convergence:
        raise ValueError("Missing reference convergence checks")
    for row in convergence:
        for key in ("rmse", "max_abs_error", "cpu_gpu_max_abs_error"):
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError("Invalid reference validation metric")
        if float(row["cpu_gpu_max_abs_error"]) > 0.025:
            raise ValueError("GPU reference fails the independent CPU oracle check")
    converged = all(float(row["rmse"]) <= 0.005 and float(row["max_abs_error"]) <= 0.05 for row in convergence)
    if publication and not converged:
        raise ValueError("Reference is not sufficiently converged; increase --reference-samples")
    if validate_captures:
        artifacts = read_csv(root / "capture_manifest.csv")
        seen = set()
        for artifact in artifacts:
            path = (root / artifact["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("Missing capture or invalid capture path")
            if artifact["path"] in seen:
                raise ValueError("Duplicate capture path")
            seen.add(artifact["path"])
            width, height = int(artifact["width"]), int(artifact["height"])
            if path.suffix == ".pfm":
                actual_w, actual_h, values = read_pfm(path)
                if (actual_w, actual_h) != (width, height):
                    raise ValueError("Wrong PFM dimensions")
                if any(v < -1e-5 or v > 1.00001 for v in values):
                    raise ValueError("PFM visibility or mask is outside [0,1]")
            else:
                with path.open("rb") as stream:
                    header = stream.read(24)
                if header[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", header[16:24]) != (width, height):
                    raise ValueError("Invalid PNG capture or dimensions")
        for row in rows:
            case = row["case_id"]
            config = case.removesuffix("-" + row["method"])
            required = [f"captures/{case}.pfm", f"captures/{case}-shaded.png",
                        f"captures/{case}-visibility.png", f"captures/{case}-difference.png"]
            if any(path not in seen for path in required):
                raise ValueError(f"Missing artifacts for {case}")
            _, _, image = read_pfm(root / required[0])
            _, _, reference = read_pfm(root / "captures" / f"{config}-reference.pfm")
            _, _, mask = read_pfm(root / "captures" / f"{config}-receiver.pfm")
            count = sum(v > 0.5 for v in mask)
            rmse = math.sqrt(sum((a-b)**2 for a, b, m in zip(image, reference, mask) if m > 0.5) / count)
            if abs(rmse - float(row["rmse"])) > 1e-5:
                raise ValueError(f"Saved captures disagree with reported RMSE in {case}")
    summary = {"cases": len(rows), "hardware_timings": not manifest["software_renderer"],
               "reference_converged": converged, "methods": {}}
    for method in sorted({row["method"] for row in rows}):
        selection = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {
            "cases": len(selection), "median_total_gpu_ms": statistics.median(float(r["total_gpu_ms"]) for r in selection),
            "mean_rmse": statistics.mean(float(r["rmse"]) for r in selection),
            "mean_penumbra_rmse": statistics.mean(float(r["penumbra_rmse"]) for r in selection),
            "max_brightening": max(float(r["monotonic_max_brightening"]) for r in selection)}
    return summary, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-hardware", action="store_true")
    parser.add_argument("--publication", action="store_true")
    args = parser.parse_args()
    try:
        summary, manifest = analyze(args.directory, args.require_hardware, args.publication)
        (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        notice = "Hardware GPU measurements" if summary["hardware_timings"] else "Software renderer verification only"
        table = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in
                       [name, data["cases"], f"{data['median_total_gpu_ms']:.4f}", f"{data['mean_rmse']:.6f}",
                        f"{data['mean_penumbra_rmse']:.6f}", f"{data['max_brightening']:.6f}"]) + "</tr>"
                        for name, data in summary["methods"].items())
        report = ("<!doctype html><meta charset='utf-8'><title>gfx-research-shadows</title>"
                  "<style>body{font:16px system-ui;margin:40px;max-width:1100px}td,th{padding:9px;border-bottom:1px solid #ccc;text-align:left}table{border-collapse:collapse}</style>"
                  f"<h1>gfx-research-shadows</h1><p>{notice}: {html.escape(manifest['gpu'])}</p>"
                  f"<p>Cases: {summary['cases']}. Reference convergence criterion met: {summary['reference_converged']}.</p>"
                  "<p>Aggregates mix experimental configurations. Use raw.csv for paired comparisons. Low-sample and split methods are baselines, not established improvements.</p>"
                  "<table><tr><th>Method</th><th>Cases</th><th>GPU ms</th><th>RMSE</th><th>Penumbra RMSE</th><th>Max brightening</th></tr>" + table + "</table>")
        (args.directory / "report.html").write_text(report, encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    except (ValueError, KeyError, OSError, ZeroDivisionError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
