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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_filters import verify as verify_filters
from verify_sampling import verify as verify_sampling
METHODS = ("hard", "pcf", "pcss", "arr", "geometric", "dpcf", "chs")


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
    if manifest.get("version") not in ("0.11.0", "0.12.0") or manifest.get("method_count") != 7:
        raise ValueError("Unsupported method catalog/version")
    if publication and manifest.get("profile") != "publication":
        raise ValueError("Publication validation requires the full publication profile")
    filters = verify_filters(root / "filter_gpu.csv")
    if manifest["version"] == "0.12.0" and "smooth_mean_gap" not in filters["max_abs_error"]:
        raise ValueError("Missing arithmetic blocker mean validation")
    if manifest.get("filter_validation_cases") != filters["cases"]:
        raise ValueError("Missing filter validation")
    sampling=verify_sampling(root/"sampling_gpu.csv")
    if manifest.get("sampling_validation_cases")!=sampling["cases"]:
        raise ValueError("Missing sampling validation")
    rows = read_csv(root / "raw.csv")
    if not rows or len(rows) != manifest["expected_cases"]:
        raise ValueError("Case matrix is incomplete")
    by_id = {row["case_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate case IDs")
    for row in rows:
        if row["method"] not in METHODS:
            raise ValueError("Unexpected method")
        mode = int(row["smoothing"] if "smoothing" in row else row["geometric_smoothing"])
        if mode not in range(8 if manifest["version"] == "0.12.0" else 6):
            raise ValueError("Invalid smoothing mode")
        if manifest["version"] == "0.12.0":
            configured = row['configured_sample_pattern']
            if configured not in ('vogel', 'random', 'stratified'):
                raise ValueError("Invalid configured sample pattern")
            kernel = 1 if mode >= 6 else mode
            if row['method'] == 'pcf' and kernel == 2:
                kernel = 1
            if row['method'] not in ('pcf', 'pcss', 'geometric'):
                kernel = 0
            pattern = ('random' if mode == 6 else 'stratified') if mode >= 6 else configured
            if row['method'] not in ('pcss', 'geometric'):
                pattern = 'vogel'
            if int(row['smoothing_kernel']) != kernel or row['sample_pattern'] != pattern:
                raise ValueError("Effective smoothing/sampling configuration disagrees")
        if row["sample_pattern"] not in ('vogel','random','stratified') or int(row['sample_seed'])<0 or int(row['rotate_samples']) not in (0,1):
            raise ValueError("Invalid PCSS sampling configuration")
        for key in ("map_gpu_ms", "shadow_gpu_ms", "total_gpu_ms", "rmse", "penumbra_rmse", "gradient_rmse",
                    "max_abs_error", "monotonic_max_brightening", "monotonic_violation_fraction"):
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError("Invalid image/timing metric")
        if int(row["invalid_pixels"]) or int(row["receiver_pixels"]) <= 0:
            raise ValueError("Invalid image metrics")
        is_arr = row["method"] == "arr"
        expected = 0 if is_arr else 1
        if (int(row["map_count"]) != expected or int(row["depth_layers"]) != expected or
                int(row["map_generation_passes"]) != expected or int(row["auxiliary_map_bytes"]) != 0 or
                int(row["map_bytes"]) != expected * int(row["map_resolution"])**2 * 4):
            raise ValueError("Single-depth resource contract violated")
        if not is_arr and int(row["shadow_passes"]) != 1:
            raise ValueError("Single-evaluation contract violated")
        if is_arr and float(row["rmse"]) > 1e-6:
            raise ValueError("ARR comparison image differs")
    grouped = {key: [] for key in by_id}
    for row in read_csv(root / "timings.csv"):
        if row["case_id"] not in grouped:
            raise ValueError("Unexpected timing case")
        for key in ("map_gpu_ms", "shadow_gpu_ms", "total_gpu_ms"):
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError("Invalid timestamp result")
        if abs(float(row["map_gpu_ms"])+float(row["shadow_gpu_ms"])-float(row["total_gpu_ms"])) > 1e-5:
            raise ValueError("Timing scopes disagree")
        grouped[row["case_id"]].append(row)
    for case, values in grouped.items():
        if sorted(int(v["sample"]) for v in values) != list(range(int(by_id[case]["timing_samples"]))):
            raise ValueError("Missing or duplicate timing samples")
        for key in ("map_gpu_ms", "shadow_gpu_ms", "total_gpu_ms"):
            median = statistics.median(float(v[key]) for v in values)
            if abs(median-float(by_id[case][key])) > max(1e-5, median*1e-6):
                raise ValueError("Timing median mismatch")
    convergence = read_csv(root / "reference_convergence.csv")
    if not convergence:
        raise ValueError("Missing reference convergence")
    for row in convergence:
        for key in ("rmse", "max_abs_error", "cpu_gpu_max_abs_error"):
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError("Invalid reference metric")
        if float(row["cpu_gpu_max_abs_error"]) > 0.025:
            raise ValueError("ARR fails independent CPU oracle")
    converged = all(float(row["rmse"]) <= .005 and float(row["max_abs_error"]) <= .05 for row in convergence)
    if publication and not converged:
        raise ValueError("Reference is not sufficiently converged")
    if validate_captures:
        artifacts = read_csv(root / "capture_manifest.csv")
        seen = set()
        for artifact in artifacts:
            path = (root / artifact["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or artifact["path"] in seen:
                raise ValueError("Missing/duplicate capture or invalid path")
            seen.add(artifact["path"])
            shape = (int(artifact["width"]), int(artifact["height"]))
            if path.suffix == ".pfm":
                w,h,values = read_pfm(path)
                if (w,h) != shape or any(v < -1e-5 or v > 1.00001 for v in values):
                    raise ValueError("Invalid PFM dimensions or range")
            else:
                with path.open("rb") as stream:
                    header = stream.read(24)
                if header[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II",header[16:24]) != shape:
                    raise ValueError("Invalid PNG capture")
        for row in rows:
            case = row["case_id"]
            config = case.removesuffix("-"+row["method"])
            required = [f"captures/{case}.pfm"] + [f"captures/{case}-{view}.png" for view in ("shaded","visibility","difference")]
            required += [f"captures/{config}-{view}.pfm" for view in ("reference","receiver")]
            if any(p not in seen for p in required):
                raise ValueError("Missing comparison capture")
            _,_,image = read_pfm(root / required[0])
            _,_,reference = read_pfm(root / required[-2])
            _,_,mask = read_pfm(root / required[-1])
            count = sum(v>.5 for v in mask)
            rmse = math.sqrt(sum((a-b)**2 for a,b,m in zip(image,reference,mask) if m>.5)/count)
            if abs(rmse-float(row["rmse"])) > 1e-5:
                raise ValueError("Capture disagrees with reported RMSE")
    summary = {"cases": len(rows), "hardware_timings": not manifest["software_renderer"],
               "reference_converged": converged, "filter_oracles": filters, "sampling_oracles":sampling,"methods": {}}
    for method in METHODS:
        selection = [r for r in rows if r["method"] == method]
        if selection:
            summary["methods"][method] = {"cases": len(selection),
                "median_total_gpu_ms": statistics.median(float(r["total_gpu_ms"]) for r in selection),
                "mean_rmse": statistics.mean(float(r["rmse"]) for r in selection),
                "mean_penumbra_rmse": statistics.mean(float(r["penumbra_rmse"]) for r in selection),
                "max_abs_error": max(float(r["max_abs_error"]) for r in selection),
                "max_brightening": max(float(r["monotonic_max_brightening"]) for r in selection)}
    return summary, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-hardware", action="store_true")
    parser.add_argument("--publication", action="store_true")
    args = parser.parse_args()
    try:
        summary,manifest = analyze(args.directory,args.require_hardware,args.publication)
        (args.directory/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        title = "Hardware measurements" if summary["hardware_timings"] else "Software renderer verification only"
        table = "".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in
                (name,data["mean_rmse"],data["mean_penumbra_rmse"],data["max_abs_error"],data["median_total_gpu_ms"]))+"</tr>"
                for name,data in summary["methods"].items())
        report = ("<!doctype html><meta charset='utf-8'><title>gfx-research-shadows</title>"
                  "<style>body{font:16px system-ui;margin:40px}td,th{padding:8px;text-align:left}</style>"
                  f"<h1>gfx-research-shadows {html.escape(manifest['version'])}</h1><p>{title}: {html.escape(manifest['gpu'])}</p>"
                  "<p>Use paired rows in raw.csv. Smoothing reconstructs PCF/PCSS comparison results; modes 2/3 also weight PCSS blocker statistics. It does not recover hidden occluders.</p>"
                  "<table><tr><th>Method</th><th>RMSE</th><th>Penumbra RMSE</th><th>Max error</th><th>GPU ms</th></tr>"+table+"</table>")
        (args.directory/"report.html").write_text(report,encoding="utf-8")
        print(json.dumps(summary,indent=2));return 0
    except (ValueError,KeyError,OSError,ZeroDivisionError) as error:
        print(f"Validation failed: {error}",file=sys.stderr);return 1


if __name__ == "__main__":
    raise SystemExit(main())
