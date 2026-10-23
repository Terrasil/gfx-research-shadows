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
from verify_prefilter import verify as verify_prefilter
from verify_intervals import verify as verify_intervals
from verify_angular import verify as verify_angular, verify_filter as verify_angular_filter
METHODS = ("hard", "pcf", "pcss", "arr", "geometric", "dpcf", "chs", "geometric-layered", "vsm", "csm", "esm", "arr-triangulation")


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


def analyze_performance(root, manifest, validate_captures=True):
    if manifest.get("version") not in ("0.16.0", "0.16.1") or manifest.get("quality_metrics") is not False or manifest.get("quality_reference") is not None:
        raise ValueError("Unsupported performance manifest or fabricated quality metrics")
    counts = manifest["object_counts"]
    if not counts or len(set(counts)) != len(counts) or any(type(n) is not int or not 0 <= n <= 10_000_000 for n in counts):
        raise ValueError("Invalid stress population matrix")
    selected = [m for m in METHODS if manifest["version"] != "0.16.0" or m not in ("arr", "arr-triangulation")] if manifest["configured_method"] == "all" else [manifest["configured_method"]]
    if any(m not in METHODS for m in selected):
        raise ValueError("Unknown stress method")
    expected = {(n, m) for n in counts for m in selected}
    rows = read_csv(root / "raw.csv")
    by_id = {row["case_id"]: row for row in rows}
    actual = {(int(row["objects"]), row["method"]) for row in rows}
    if len(rows) != manifest["expected_cases"] or len(rows) != len(expected) or len(by_id) != len(rows) or actual != expected:
        raise ValueError("Incomplete stress case matrix")
    grouped = {key: [] for key in by_id}
    timing_keys = ("gbuffer_gpu_ms", "map_gpu_ms", "shadow_gpu_ms", "total_gpu_ms", "wall_ms")
    for row in rows:
        n, method = int(row["objects"]), row["method"]
        angular = method in ("arr", "arr-triangulation")
        terms = int(row["csm_terms"])
        auxiliary_bpp = 16 if method == "vsm" else 8 if method == "esm" else 16 * terms if method == "csm" else 0
        expected_bytes = (0 if angular else 4 + auxiliary_bpp) * int(row["map_resolution"]) ** 2
        if (row["scene"] != "dense-stress" or int(row["caster_base_triangles"]) != n * 12 or
                int(row["gbuffer_cube_draws"]) != int(n > 0) or int(row["map_cube_draws"]) != int(n > 0 and not angular) or
                int(row["instance_storage_bytes"]) != 0 or int(row["map_bytes"]) != expected_bytes or
                int(row["prefilter_passes"]) != (3 if auxiliary_bpp else 0) or int(row["invalid_pixels"]) or
                int(row["timing_samples"]) != manifest["measure"] or (manifest["version"] == "0.16.0" and angular and n > 32)):
            raise ValueError("Invalid stress resource/instance accounting")
        for key in timing_keys:
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError("Invalid stress timing")
        if manifest["version"] == "0.16.1":
            queries = int(manifest["arr_samples"]) if method == "arr" else int(manifest["tri_samples"]) if method == "arr-triangulation" else 0
            smoothing = int(manifest["arr_smoothing"]) if angular and float(manifest["arr_smoothing_width"]) > 0 else 0
            filter_bytes = int(row["width"]) * int(row["height"]) * 8 if smoothing else 0
            if (int(row["angular_queries"]) != queries or int(row["angular_smoothing"]) != smoothing or
                    int(row["shadow_passes"]) != 1 + int(smoothing != 0) or int(row["screen_filter_bytes"]) != filter_bytes or
                    int(row["angular_direction_bytes"]) != (16384 * 16 if method == "arr" else 0) or
                    int(row["tri_result_masks"]) != int(method == "arr-triangulation" and manifest["tri_result_masks"] == 1)):
                raise ValueError("Incorrect stress angular configuration/resources")
    for row in read_csv(root / "timings.csv"):
        if row["case_id"] not in grouped:
            raise ValueError("Unexpected stress timing case")
        for key in timing_keys:
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError("Invalid stress sample")
        if abs(sum(float(row[k]) for k in timing_keys[:3]) - float(row["total_gpu_ms"])) > 1e-5:
            raise ValueError("Stress timing scopes disagree")
        grouped[row["case_id"]].append(row)
    for case_id, samples in grouped.items():
        if sorted(int(s["sample"]) for s in samples) != list(range(manifest["measure"])):
            raise ValueError("Incomplete stress timing samples")
        for key in timing_keys:
            if abs(statistics.median(float(s[key]) for s in samples) - float(by_id[case_id][key])) > 1e-5:
                raise ValueError("Stress medians disagree")
    if manifest["version"] == "0.16.1":
        checks = read_csv(root / "stress_gpu.csv")
        if (manifest["stress_validation_cases"] != 12 or len(checks) != 12 or
                {(int(c["objects"]), int(c["mode"])) for c in checks} != {(n, m) for n in (33, 65, 1024, 1_000_000) for m in range(3)}):
            raise ValueError("Missing full-population procedural ARR validation")
        for check in checks:
            error = float(check["bounds_max_abs_error"])
            if (int(check["instance"]) != int(check["objects"]) - 1 or
                    int(check["expected_blocked"]) != int(int(check["mode"]) == 0) or
                    float(check["gpu_blocked"]) != int(check["expected_blocked"]) or not math.isfinite(error) or error > 2e-6):
                raise ValueError("Procedural ARR failed the late-instance CPU/GPU oracle")
    if validate_captures:
        captures = read_csv(root / "capture_manifest.csv")
        if len(captures) != len(rows) * 2 or {(r["case_id"], r["kind"]) for r in captures} != {(c, k) for c in by_id for k in ("shaded", "visibility")}:
            raise ValueError("Incomplete stress captures")
        for capture in captures:
            p = (root / capture["path"]).resolve()
            if not p.is_relative_to(root.resolve()):
                raise ValueError("Stress capture outside run directory")
            data = p.read_bytes()
            expected_size = (int(by_id[capture["case_id"]]["width"]), int(by_id[capture["case_id"]]["height"]))
            if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", data[16:24]) != expected_size:
                raise ValueError("Invalid stress PNG")
    return {"experiment_kind": "performance", "cases": len(rows), "hardware_timings": not manifest["software_renderer"],
            "quality_metrics": False, "populations": rows,
            "methods": {m: {"median_total_gpu_ms": statistics.median(float(r["total_gpu_ms"]) for r in rows if r["method"] == m)} for m in selected}}, manifest


def analyze(root, require_hardware=False, publication=False, validate_captures=True):
    root = Path(root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("completed") is not True:
        raise ValueError("Run is incomplete")
    if (require_hardware or publication) and manifest.get("software_renderer"):
        raise ValueError("Software rasterizer timings cannot be published as hardware GPU results")
    if manifest.get("experiment_kind") == "performance":
        if publication:
            raise ValueError("Stress performance data has no quality reference for publication validation")
        return analyze_performance(root, manifest, validate_captures)
    if manifest.get("version") not in ("0.11.0", "0.12.0", "0.13.0", "0.14.0", "0.14.1", "0.15.0", "0.15.1", "0.15.2", "0.15.3", "0.16.0", "0.16.1") or manifest.get("method_count") != (12 if manifest.get("version") in ("0.15.0", "0.15.1", "0.15.2", "0.15.3", "0.16.0", "0.16.1") else 11 if manifest.get("version") in ("0.14.0","0.14.1") else 7):
        raise ValueError("Unsupported method catalog/version")
    if publication and manifest.get("profile") != "publication":
        raise ValueError("Publication validation requires the full publication profile")
    filters = verify_filters(root / "filter_gpu.csv")
    if manifest["version"] != "0.11.0" and "smooth_mean_gap" not in filters["max_abs_error"]:
        raise ValueError("Missing arithmetic blocker mean validation")
    if manifest.get("filter_validation_cases") != filters["cases"]:
        raise ValueError("Missing filter validation")
    sampling=verify_sampling(root/"sampling_gpu.csv")
    if manifest.get("sampling_validation_cases")!=sampling["cases"]:
        raise ValueError("Missing sampling validation")
    prefilter = None
    if manifest['version'] in ('0.14.0','0.14.1','0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1'):
        prefilter = verify_prefilter(root/'prefilter_gpu.csv')
        if manifest.get('prefilter_validation_cases') != prefilter['cases']:
            raise ValueError('Missing prefilter validation')
    intervals = None
    if manifest['version'] in ('0.14.1','0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1'):
        intervals=verify_intervals(root/'interval_gpu.csv')
        if manifest.get('interval_validation_cases')!=intervals['cases']:
            raise ValueError('Missing interval validation')
    angular = angular_filter = None
    if manifest['version'] in ('0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1'):
        angular=verify_angular(root/'angular_gpu.csv')
        angular_filter=verify_angular_filter(root/'angular_filter_gpu.csv')
        if manifest.get('angular_validation_cases')!=angular['cases'] or manifest.get('angular_filter_validation_cases')!=angular_filter['cases']:
            raise ValueError('Missing angular verification')
        if manifest.get('oracle_smoothing')!=0:
            raise ValueError('Oracle must remain unfiltered')
    if manifest["version"] in ("0.15.2", "0.15.3", "0.16.0", "0.16.1") and manifest.get("tri_result_masks") not in (0,1):
        raise ValueError("Missing triangulation result mode")
    if manifest["version"] in ("0.15.3", "0.16.0", "0.16.1"):
        minimum=float(manifest["layer_min_filter_texels"])
        if not math.isfinite(minimum) or not 0<=minimum<=4 or type(manifest.get("shared_sample_pattern")) is not bool:
            raise ValueError("Invalid soft packed configuration")
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
        if mode not in range(8 if manifest["version"] != "0.11.0" else 6):
            raise ValueError("Invalid smoothing mode")
        if manifest["version"] != "0.11.0":
            configured = row['configured_sample_pattern']
            if configured not in ('vogel', 'random', 'stratified'):
                raise ValueError("Invalid configured sample pattern")
            kernel = 1 if mode >= 6 else mode
            if row['method'] == 'pcf' and kernel == 2:
                kernel = 1
            if row['method'] not in ('pcf', 'pcss', 'geometric', 'geometric-layered'):
                kernel = 0
            pattern = ('random' if mode == 6 else 'stratified') if mode >= 6 else configured
            if row['method'] not in ('pcss', 'geometric', 'geometric-layered'):
                pattern = 'vogel'
            if int(row['smoothing_kernel']) != kernel or row['sample_pattern'] != pattern:
                raise ValueError("Effective smoothing/sampling configuration disagrees")
        if manifest["version"] in ("0.13.0", "0.14.0", "0.14.1", "0.15.0", "0.15.1", "0.15.2", "0.15.3", "0.16.0", "0.16.1"):
            configured_penumbra = manifest['configured_penumbra']
            if configured_penumbra not in ('inner', 'full'):
                raise ValueError("Invalid penumbra setting")
            want = configured_penumbra if row['method'] in ('pcss', 'geometric', 'geometric-layered') else 'native'
            if row['penumbra_mode'] != want:
                raise ValueError("Incorrect effective penumbra mode")
        if row["sample_pattern"] not in ('vogel','random','stratified') or int(row['sample_seed'])<0 or int(row['rotate_samples']) not in (0,1):
            raise ValueError("Invalid PCSS sampling configuration")
        for key in ("map_gpu_ms", "shadow_gpu_ms", "total_gpu_ms", "rmse", "penumbra_rmse", "gradient_rmse",
                    "max_abs_error", "monotonic_max_brightening", "monotonic_violation_fraction"):
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError("Invalid image/timing metric")
        if int(row["invalid_pixels"]) or int(row["receiver_pixels"]) <= 0:
            raise ValueError("Invalid image metrics")
        is_arr = row["method"] in ("arr","arr-triangulation")
        expected = 0 if is_arr else 1
        prefiltered = row['method'] in ('vsm','csm','esm')
        terms = int(row.get('csm_terms',8))
        if prefiltered and (terms<4 or terms>16 or terms%2):
            raise ValueError('Invalid Fourier order')
        bytes_per_pixel = 16 if row['method']=='vsm' else 8 if row['method']=='esm' else 16*terms if row['method']=='csm' else 0
        auxiliary = bytes_per_pixel*int(row['map_resolution'])**2
        if (int(row['map_count']) != expected or int(row['depth_layers']) != expected or
                int(row['map_generation_passes']) != expected or int(row['auxiliary_map_bytes']) != auxiliary or
                int(row['map_bytes']) != expected*int(row['map_resolution'])**2*4+auxiliary):
            raise ValueError('Shadow resource accounting violated')
        if manifest['version'] in ('0.14.0','0.14.1','0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1') and int(row['prefilter_passes']) != (3 if prefiltered else 0):
            raise ValueError('Incorrect prefilter pass count')
        if manifest['version'] in ('0.14.1','0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1'):
            packed=row['method']=='geometric-layered' and manifest['configured_layer_depth']=='interval'
            storage='none' if is_arr else 'r32ui-unorm16-minmax' if packed else 'd32f'
            components=0 if is_arr else 2 if packed else 1
            if row['map_storage']!=storage or int(row['stored_depth_components'])!=components:
                raise ValueError('Incorrect depth representation accounting')
        if manifest['version'] in ('0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1'):
            configured=int(manifest['arr_smoothing'])
            width=float(manifest['arr_smoothing_radius'])
            smoothing=configured if is_arr and width>0 else 0
            if smoothing not in (0,1,3,4,5) or int(row['angular_smoothing'])!=smoothing:
                raise ValueError('Incorrect angular smoothing')
            if int(row['shadow_passes'])!=1+int(smoothing!=0):
                raise ValueError('Incorrect angular pass count')
            expected_queries=(int(manifest['arr_samples']) or int(manifest['reference_samples'])) if row['method']=='arr' else int(manifest['tri_samples']) if row['method']=='arr-triangulation' else 0
            expected_bytes=8*int(row['width'])*int(row['height']) if smoothing else 0
            if int(row['angular_queries'])!=expected_queries or int(row['screen_filter_bytes'])!=expected_bytes:
                raise ValueError('Incorrect angular query/resource accounting')
            if int(row['angular_direction_bytes'])!=(16384*16 if row['method']=='arr' else 0):
                raise ValueError('Incorrect direction buffer accounting')
        if manifest["version"] in ("0.15.2", "0.15.3", "0.16.0", "0.16.1"):
            expected_masks=int(row["method"]=="arr-triangulation" and manifest["tri_result_masks"]==1)
            if int(row["tri_result_masks"])!=expected_masks:
                raise ValueError("Incorrect triangulation result mode")
        if manifest["version"] in ("0.15.3", "0.16.0", "0.16.1"):
            expected_min=float(manifest["layer_min_filter_texels"]) if row["method"]=="geometric-layered" else 0
            expected_shared=int(row["sample_pattern"]!="vogel" and manifest["shared_sample_pattern"])
            if abs(float(row["layer_min_filter_texels"])-expected_min)>1e-5 or int(row["shared_sample_pattern"])!=expected_shared:
                raise ValueError("Incorrect effective soft packed configuration")
        if not is_arr and int(row["shadow_passes"]) != 1:
            raise ValueError("Single-evaluation contract violated")
        oracle_candidate=(row["method"]=="arr" and (manifest['version'] not in ('0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1') or
            (int(row['angular_queries'])==int(manifest['reference_samples']) and int(row['angular_smoothing'])==0)))
        if oracle_candidate and float(row["rmse"]) > 1e-6:
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
    outline_cases, outline_pixels = 0, 0
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
            if row.get('penumbra_mode') == 'inner':
                hard_path = f"captures/{config}-hard.pfm"
                if hard_path not in seen:
                    raise ValueError("Missing hard-outline comparison capture")
                hw, hh, hard = read_pfm(root / hard_path)
                if (hw, hh) != (int(row['width']), int(row['height'])):
                    raise ValueError("Hard-outline capture dimensions disagree")
                # Test every pixel, including visible caster surfaces. The
                # receiver-only RMSE mask would miss halos on the boxes.
                for value, baseline in zip(image, hard):
                    if baseline > .999999:
                        outline_pixels += 1
                        if value < .999999:
                            raise ValueError("Shadow extends outside the hard outline")
                outline_cases += 1
    summary = {"cases": len(rows), "hardware_timings": not manifest["software_renderer"],
               "reference_converged": converged, "prefilter_oracles": prefilter, "angular_oracles": angular, "angular_filter_oracles": angular_filter, "interval_oracles": intervals, "filter_oracles": filters, "sampling_oracles":sampling,
               "outline_checks": {"cases": outline_cases, "outside_pixels": outline_pixels, "violations": 0},
               "methods": {}}
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
        if summary.get("experiment_kind") == "performance":
            table = "".join("<tr>"+"".join(f"<td>{html.escape(str(row[k]))}</td>" for k in
                           ("objects", "method", "gbuffer_gpu_ms", "map_gpu_ms", "shadow_gpu_ms", "total_gpu_ms"))+"</tr>"
                           for row in summary["populations"])
            report = ("<!doctype html><meta charset='utf-8'><title>Shadow stress benchmark</title>"
                      "<style>body{font:16px system-ui;margin:40px}td,th{padding:8px;text-align:left}</style>"
                      f"<h1>dense-stress</h1><p>{title}: {html.escape(manifest['gpu'])}</p>"
                      "<p>Performance only. No ARR quality reference. Counts are submitted cube instances; 12 base triangles each, before map tessellation.</p>"
                      "<table><tr><th>Objects</th><th>Method</th><th>G-buffer ms</th><th>Map ms</th><th>Shadow ms</th><th>GPU sum ms</th></tr>"+table+"</table>")
            (args.directory/"report.html").write_text(report,encoding="utf-8")
            print(json.dumps(summary,indent=2));return 0
        table = "".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in
                (name,data["mean_rmse"],data["mean_penumbra_rmse"],data["max_abs_error"],data["median_total_gpu_ms"]))+"</tr>"
                for name,data in summary["methods"].items())
        report = ("<!doctype html><meta charset='utf-8'><title>gfx-research-shadows</title>"
                  "<style>body{font:16px system-ui;margin:40px}td,th{padding:8px;text-align:left}</style>"
                  f"<h1>gfx-research-shadows {html.escape(manifest['version'])}</h1><p>{title}: {html.escape(manifest['gpu'])}</p>"
                  "<p>Use paired rows in raw.csv. Smoothing reconstructs PCF/PCSS comparison results; modes 2/3 also weight PCSS blocker statistics. Inner mode clips to the central hard outline and is an artistic constraint; full mode retains the original PCSS. Neither recovers hidden occluders.</p>"
                  "<table><tr><th>Method</th><th>RMSE</th><th>Penumbra RMSE</th><th>Max error</th><th>GPU ms</th></tr>"+table+"</table>")
        (args.directory/"report.html").write_text(report,encoding="utf-8")
        print(json.dumps(summary,indent=2));return 0
    except (ValueError,KeyError,OSError,ZeroDivisionError) as error:
        print(f"Validation failed: {error}",file=sys.stderr);return 1


if __name__ == "__main__":
    raise SystemExit(main())
