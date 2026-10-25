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
from verify_multiview_storage import verify as verify_multiview_storage
from verify_rgba_moments import verify as verify_rgba_moments
from verify_rgba_shared import verify as verify_rgba_shared
from verify_rgba_pcf import verify as verify_rgba_pcf
from verify_rgba_pcss import verify as verify_rgba_pcss
from verify_rgba_depth_scale import verify as verify_rgba_depth_scale
from verify_rgba_blend import verify as verify_rgba_blend
from verify_rgba_edge import verify as verify_rgba_edge
from verify_rgba import verify as verify_rgba
from verify_filters import verify as verify_filters
from verify_sampling import verify as verify_sampling
from verify_blue_noise import verify as verify_blue_noise
from verify_contact_blend import verify as verify_contact_blend
from verify_prefilter import verify as verify_prefilter
from verify_intervals import verify as verify_intervals
from verify_angular import verify as verify_angular, verify_filter as verify_angular_filter
METHODS = ("hard", "pcf", "pcss", "arr", "geometric", "dpcf", "chs", "geometric-layered", "vsm", "csm", "esm", "arr-triangulation", "angular-rgba", "gsm")


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


def verify_result_options(row, manifest):
    supported=row['method'] in ('pcss','geometric','geometric-layered')
    fixed_radius=manifest.get('rgba_pcf_world_radius_m',0) if manifest.get('version') in ('0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") and manifest.get('rgba_pcf_world_units') else manifest.get('rgba_pcf_radius',0)
    angular=(row['method']=='angular-rgba' and manifest.get('version') in ('0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") and
             manifest['rgba_pcf_mode']!=0 and manifest['rgba_pcf_strength']>0 and
             ((manifest['rgba_shared_radius_m']>0 and (not manifest['rgba_shared_adaptive'] or manifest['rgba_pcf_max_radius']>0)) if manifest['rgba_pcf_mode']==3 else (fixed_radius if manifest['rgba_pcf_mode']==1 else manifest['rgba_pcf_max_radius'])>0))
    blend=float(manifest['result_blend']) if supported or angular else 0.0
    guard=float(manifest['contact_guard']) if supported else 0.0
    gap=float(manifest['contact_max_gap']) if guard>0 else 0.0
    radius=int(manifest['result_blend_radius']) if blend>0 else 0
    sigma=float(manifest['result_blend_sigma']) if blend>0 else 0.0
    expected={'result_blend':blend,'contact_guard':guard,'contact_max_gap':gap,'result_blend_sigma':sigma}
    if (not 0<=blend<=1 or not 0<=guard<=1 or (guard>0 and not .05<=gap<=3) or
            (blend>0 and (radius not in (1,2) or not .05<=sigma<=.5))):
        raise ValueError('Invalid contact/result-blend configuration')
    for key,value in expected.items():
        actual=float(row[key])
        if not math.isfinite(actual) or abs(actual-value)>1e-5:
            raise ValueError('Effective contact/result blend differs from configuration')
    if (int(row['result_blend_radius'])!=radius or
            int(row['result_blend_bytes'])!=(int(row['width'])*int(row['height'])*4 if blend>0 else 0)):
        raise ValueError('Incorrect result-blend resource accounting')
    return int(blend>0)


def multiview_variant(manifest):
    storage = manifest['multiview_storage']
    return storage + ('-cascaded' if manifest['rgba_cascades'] else '')


def multiview_storage(row, manifest):
    if manifest.get('version') in ('0.30.0','0.31.0'):
        expected = multiview_variant(manifest)
        if manifest.get('multiview_variant') != expected or row.get('multiview_variant') != (expected if row['method']=='angular-rgba' else 'none'):
            raise ValueError('Incorrect explicit multiple light view variant')
    configured = manifest.get('multiview_storage', 'rgba')
    if manifest.get('version') in ('0.29.0','0.30.0',"0.31.0") and configured not in ('rgba', 'array', 'separate'):
        raise ValueError('Invalid multi-view storage configuration')
    active = row['method'] == 'angular-rgba'
    if manifest.get('version') in ('0.29.0','0.30.0',"0.31.0"):
        if row.get('multiview_storage') != (configured if active else 'none'):
            raise ValueError('Effective multi-view storage differs from configuration')
        textures = 0 if row['method'] in ('arr', 'arr-triangulation') else 4 if active and configured == 'separate' else 1
        if int(row['map_texture_count']) != textures:
            raise ValueError('Incorrect multi-view texture count')
    return {'rgba':'rgba32f-angular', 'array':'d32f-array-angular', 'separate':'d32f-separate-angular'}[configured] if active else None


def verify_gsm_options(row, manifest):
    if manifest.get('version') != '0.31.0':
        return
    power = float(manifest['gsm_sharpness'])
    radius = manifest['gsm_prefilter_radius']
    if not math.isfinite(power) or not 1 <= power <= 128 or type(radius) is not int or not 0 <= radius <= 16:
        raise ValueError('Invalid GSM map parameters')
    active = row['method'] == 'gsm'
    expected_power, expected_radius = (power, radius) if active else (0.0, 0)
    if (not math.isfinite(float(row['gsm_sharpness'])) or abs(float(row['gsm_sharpness']) - expected_power) > 1e-5 or
            int(row['gsm_prefilter_radius']) != expected_radius):
        raise ValueError('Incorrect effective GSM map parameters')


def verify_rgba_options(manifest):
    if manifest.get('version') in ('0.30.0','0.31.0') and manifest.get('multiview_variant') != multiview_variant(manifest):
        raise ValueError('Incorrect explicit multiple light view variant')
    if manifest.get('version') in ('0.29.0','0.30.0',"0.31.0") and manifest.get('multiview_storage') not in ('rgba', 'array', 'separate'):
        raise ValueError('Invalid multi-view storage configuration')
    if manifest.get('version') in ('0.28.0','0.29.0',"0.30.0","0.31.0") and (manifest.get('rgba_pcss_bounds') not in (0,1) or manifest.get('rgba_stable_focus') not in (0,1)):
        raise ValueError('Invalid PCSS cascade correction settings')
    if manifest.get('rgba_cascades') not in (0,1) or manifest.get('rgba_mix_mode') not in (range(5) if manifest.get('version') in ('0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") else range(3)):
        raise ValueError('Invalid RGBA cascade or mixing mode')
    if manifest.get('version') in ('0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        if manifest.get('rgba_reconstruct') not in (0,1) or not 2<=int(manifest['rgba_edge_search'])<=128:
            raise ValueError('Invalid angular reconstruction configuration')
        for key,hi in [('rgba_reconstruct_strength',1),('rgba_edge_contact',3)]:
            value=float(manifest[key])
            if not math.isfinite(value) or not 0<=value<=hi:raise ValueError('Invalid edge reconstruction setting')
    if manifest.get('version') in ('0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        allowed=range(5) if manifest.get('version') in ('0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") else (0,1,2,3) if manifest.get('version') in ('0.24.0','0.25.0','0.26.0') else (0,1,2)
        if manifest.get('rgba_pcf_sharp') not in (0,1) or manifest.get('rgba_pcf_mode') not in allowed or manifest.get('rgba_pcf_samples') not in (8,16,32,64):
            raise ValueError('Invalid channel PCF configuration')
        for key,hi in [('rgba_pcf_strength',1),('rgba_pcf_radius',32),('rgba_pcf_max_radius',64)]:
            value=float(manifest[key])
            if not math.isfinite(value) or not 0<=value<=hi:raise ValueError('Invalid channel PCF argument')
    if manifest.get('version') in ('0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        if manifest.get('rgba_shared_adaptive') not in (0,1) or not math.isfinite(float(manifest['rgba_shared_radius_m'])) or not 0<=float(manifest['rgba_shared_radius_m'])<=3:
            raise ValueError('Invalid shared world PCF configuration')
    if manifest.get('version') in ('0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        if manifest.get('rgba_moments') not in (0,1) or not math.isfinite(float(manifest['rgba_moment_strength'])) or not 0<=float(manifest['rgba_moment_strength'])<=1:
            raise ValueError('Invalid PCF moment configuration')
    if manifest.get('version') in ('0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        if manifest.get('rgba_pcf_world_units') not in (0,1):raise ValueError('Invalid fixed PCF units')
        for key,hi in [('rgba_pcf_world_radius_m',3),('rgba_geometric_blend',1)]:
            value=float(manifest[key])
            if not math.isfinite(value) or not 0<=value<=hi:raise ValueError('Invalid fixed PCF or geometric mixing argument')
    if manifest.get('version') in ('0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        if type(manifest.get('rgba_pcss_blocker_samples')) is not int or not 1<=manifest['rgba_pcss_blocker_samples']<=128 or manifest.get('rgba_pcss_mean') not in (0,1):
            raise ValueError('Invalid per-channel PCSS blocker settings')
        value=float(manifest['rgba_pcss_search_radius_m'])
        if not math.isfinite(value) or not 0<=value<=20:raise ValueError('Invalid per-channel PCSS search limit')
    bounds={'rgba_extent':(.5,128),'rgba_cascade_scale':(1,4),'rgba_cascade_blend':(0,.5),'rgba_angular_mix':(0,1)}
    for key,(lo,hi) in bounds.items():
        value=float(manifest[key])
        if not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError('Invalid RGBA cascade setting: '+key)


def verify_rgba_pcf_options(row,manifest):
    mode=int(manifest['rgba_pcf_mode']) if row['method']=='angular-rgba' else 0
    for key in ['rgba_pcf_mode','rgba_pcf_samples','rgba_pcf_strength','rgba_pcf_radius','rgba_pcf_max_radius']:
        expected=float(manifest[key]) if mode else 0.0
        actual=float(row[key])
        if not math.isfinite(actual) or abs(actual-expected)>1e-5:
            raise ValueError('Incorrect effective channel PCF settings')
    if manifest.get('version') in ('0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        for key in ('rgba_shared_adaptive','rgba_shared_radius_m'):
            expected=float(manifest[key]) if mode==3 else 0.0
            if not math.isfinite(float(row[key])) or abs(float(row[key])-expected)>1e-5:
                raise ValueError('Incorrect effective shared PCF settings')
    if manifest.get('version') in ('0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        fixed=(manifest.get('version') in ('0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") and mode==1 and
               (manifest['rgba_pcf_world_radius_m'] if manifest['rgba_pcf_world_units'] else manifest['rgba_pcf_radius'])>0)
        adaptive=mode==2 or (mode==4 and manifest.get('version') in ('0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"))
        active=(row['method']=='angular-rgba' and manifest['rgba_moments'] and manifest['rgba_moment_strength']>0 and
                (fixed or (adaptive and manifest['rgba_pcf_max_radius']>0)) and manifest['rgba_pcf_strength']>0 and
                manifest['rgba_offset'] and manifest['rgba_radius']>=.1 and manifest['rgba_mix_mode']==0 and
                (not manifest['rgba_cascades'] or manifest['rgba_angular_mix']>0))
        if int(row['rgba_moments'])!=int(bool(active)) or abs(float(row['rgba_moment_strength'])-(float(manifest['rgba_moment_strength']) if active else 0.0))>1e-5 or not math.isfinite(float(row['rgba_moment_strength'])):
            raise ValueError('Incorrect effective PCF moment configuration')
    if manifest.get('version') in ('0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        world=mode==1 and manifest['rgba_pcf_world_units']
        mixing=row['method']=='angular-rgba' and (not manifest['rgba_cascades'] or manifest['rgba_angular_mix']>0)
        expected={'rgba_pcf_world_units':int(bool(world)),
                  'rgba_pcf_world_radius_m':manifest['rgba_pcf_world_radius_m'] if world else 0,
                  'rgba_mix_mode':manifest['rgba_mix_mode'] if mixing else 0,
                  'rgba_geometric_blend':manifest['rgba_geometric_blend'] if mixing and manifest['rgba_mix_mode']==4 else 0}
        for key,value in expected.items():
            actual=float(row[key])
            if not math.isfinite(actual) or abs(actual-value)>1e-5:raise ValueError('Incorrect effective world PCF or geometric mixing settings')
    if manifest.get('version') in ('0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        for key in ('rgba_pcss_blocker_samples','rgba_pcss_mean','rgba_pcss_search_radius_m'):
            expected=float(manifest[key]) if mode==4 else 0.0
            actual=float(row[key])
            if not math.isfinite(actual) or abs(actual-expected)>1e-5:raise ValueError('Incorrect effective per-channel PCSS settings')
    if manifest.get('version') in ('0.28.0','0.29.0',"0.30.0","0.31.0"):
        bounds=int(bool(mode==4 and manifest['rgba_pcss_bounds']))
        focus=int(bool(row['method']=='angular-rgba' and manifest['rgba_stable_focus']))
        if int(row['rgba_pcss_bounds'])!=bounds or int(row['rgba_stable_focus'])!=focus:
            raise ValueError('Incorrect effective PCSS cascade corrections')


def analyze_performance(root, manifest, validate_captures=True):
    if manifest.get("version") not in ("0.16.0", "0.16.1", "0.17.0", "0.18.0", "0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0") or manifest.get("quality_metrics") is not False or manifest.get("quality_reference") is not None:
        raise ValueError("Unsupported performance manifest or fabricated quality metrics")
    if manifest['version'] in ('0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") and manifest.get('method_count')!=(14 if manifest['version']=='0.31.0' else 13):
        raise ValueError('Invalid performance method catalog')
    counts = manifest["object_counts"]
    if not counts or len(set(counts)) != len(counts) or any(type(n) is not int or not 0 <= n <= 10_000_000 for n in counts):
        raise ValueError("Invalid stress population matrix")
    selected = [m for m in METHODS if (m!="gsm" or manifest["version"]=="0.31.0") and (m!="angular-rgba" or manifest["version"] in ("0.19.0","0.20.0","0.21.0","0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0")) and (manifest["version"] != "0.16.0" or m not in ("arr", "arr-triangulation"))] if manifest["configured_method"] == "all" else [manifest["configured_method"]]
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
        if manifest['version'] in ('0.29.0','0.30.0',"0.31.0"):multiview_storage(row,manifest)
        if manifest["version"] in ("0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0"):verify_rgba_pcf_options(row,manifest)
        verify_gsm_options(row,manifest)
        n, method = int(row["objects"]), row["method"]
        angular = method in ("arr", "arr-triangulation")
        terms = int(row["csm_terms"])
        auxiliary_bpp = 16 if method == "vsm" else 8 if method in ("esm","gsm") else 16 * terms if method == "csm" else 0
        expected_bytes = (0 if angular else (16 if method=="angular-rgba" else 4) + auxiliary_bpp) * int(row["map_resolution"]) ** 2
        if (row["scene"] != "dense-stress" or int(row["caster_base_triangles"]) != n * 12 or
                int(row["gbuffer_cube_draws"]) != int(n > 0) or int(row["map_cube_draws"]) != (int(n > 0 and not angular)*(4 if method=="angular-rgba" else 1)) or
                int(row["instance_storage_bytes"]) != 0 or int(row["map_bytes"]) != expected_bytes or
                int(row["prefilter_passes"]) != (3 if auxiliary_bpp else 0) or int(row["invalid_pixels"]) or
                int(row["timing_samples"]) != manifest["measure"] or (manifest["version"] == "0.16.0" and angular and n > 32)):
            raise ValueError("Invalid stress resource/instance accounting")
        if manifest['version'] in ('0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
            verify_result_options(row,manifest)
            packed=method in ('pcss','geometric','geometric-layered') and manifest['configured_layer_depth']=='interval'
            expected_storage=multiview_storage(row,manifest) if method=='angular-rgba' else 'none' if angular else 'r32ui-unorm16-minmax' if packed else 'd32f'
            if row['map_storage']!=expected_storage:
                raise ValueError('Incorrect stress depth representation')
        if manifest['version'] in ('0.17.0','0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
            pattern=row['sample_pattern']
            configured=row['configured_sample_pattern']
            mode=int(row['smoothing'])
            expected_pattern=('random' if mode==6 else 'stratified') if mode>=6 else configured
            if method not in ('pcss','geometric','geometric-layered'):
                expected_pattern='vogel'
            blue=pattern=='blue-noise'
            tile=float(row['blue_noise_tile_m'])
            if (configured not in ('vogel','random','stratified','blue-noise') or pattern!=expected_pattern or
                    not math.isfinite(tile) or abs(tile-(float(manifest['blue_noise_tile_m']) if blue else 0))>1e-5 or
                    int(row['sampling_texture_bytes'])!=(2048**2 if blue else 0)):
                raise ValueError('Incorrect stress blue-noise configuration/resources')
        for key in timing_keys:
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError("Invalid stress timing")
        if manifest["version"] in ("0.16.1", "0.17.0", "0.18.0", "0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0"):
            queries = int(manifest["arr_samples"]) if method == "arr" else int(manifest["tri_samples"]) if method == "arr-triangulation" else 0
            smoothing = int(manifest["arr_smoothing"]) if angular and float(manifest["arr_smoothing_width"]) > 0 else 0
            filter_bytes = int(row["width"]) * int(row["height"]) * 8 if smoothing else 0
            if (int(row["angular_queries"]) != queries or int(row["angular_smoothing"]) != smoothing or
                    int(row["shadow_passes"]) != 1 + int(smoothing != 0) + (verify_result_options(row,manifest) if manifest["version"] in ("0.18.0","0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0") else 0) or int(row["screen_filter_bytes"]) != filter_bytes or
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
    if manifest["version"] in ("0.16.1", "0.17.0", "0.18.0", "0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0"):
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
    if manifest.get("version") in ("0.20.0","0.21.0","0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0"):verify_rgba_options(manifest)
    if manifest.get("experiment_kind") == "performance":
        if publication:
            raise ValueError("Stress performance data has no quality reference for publication validation")
        return analyze_performance(root, manifest, validate_captures)
    if manifest.get("version") not in ("0.11.0", "0.12.0", "0.13.0", "0.14.0", "0.14.1", "0.15.0", "0.15.1", "0.15.2", "0.15.3", "0.16.0", "0.16.1", "0.17.0", "0.18.0", "0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0") or manifest.get("method_count") != (14 if manifest["version"]=="0.31.0" else 13 if manifest.get("version") in ("0.19.0","0.20.0","0.21.0","0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0") else 12 if manifest.get("version") in ("0.15.0", "0.15.1", "0.15.2", "0.15.3", "0.16.0", "0.16.1", "0.17.0", "0.18.0", "0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0") else 11 if manifest.get("version") in ("0.14.0","0.14.1") else 7):
        raise ValueError("Unsupported method catalog/version")
    multiview=verify_multiview_storage(root/'multiview_storage_gpu.csv') if manifest.get('version') in ('0.29.0','0.30.0',"0.31.0") else None
    rgba_shared=verify_rgba_shared(root/"rgba_shared_gpu.csv") if manifest.get("version") in ("0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0") else None
    rgba_moments=verify_rgba_moments(root/"rgba_moments_gpu.csv") if manifest.get("version") in ("0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0") else None
    rgba_pcss=verify_rgba_pcss(root/'rgba_pcss_gpu.csv') if manifest.get('version') in ('0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") else None
    rgba_depth_scale=verify_rgba_depth_scale(root/'rgba_depth_scale_gpu.csv') if manifest.get('version') in ('0.28.0','0.29.0',"0.30.0","0.31.0") else None
    rgba_pcf=None
    rgba_blend=verify_rgba_blend(root/"rgba_blend_gpu.csv") if manifest.get("version") in ("0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0") else None
    if manifest.get("version") in ("0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0"):rgba_pcf=verify_rgba_pcf(root/"rgba_pcf_gpu.csv")
    rgba_edge=None
    if manifest.get("version") in ("0.21.0","0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0"):rgba_edge=verify_rgba_edge(root/"rgba_edge_gpu.csv")
    rgba_mix=None
    if manifest['version'] in ('0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        rgba=read_csv(root/'rgba_depth_gpu.csv')
        count=16384 if manifest['version'] in ('0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") else 8192
        if manifest['version'] in ('0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):rgba_mix=verify_rgba(root/'rgba_mix_gpu.csv')
        if len(rgba)!=count or len({(r['offset'],r.get('cascades','0'),r['channel'],r['x'],r['y']) for r in rgba})!=count:
            raise ValueError('Incomplete RGBA depth validation')
        for r in rgba:
            actual,expected=float(r['gpu']),float(r['cpu'])
            if not math.isfinite(actual) or not 0<=actual<=1 or abs(actual-expected)>2e-5:
                raise ValueError('RGBA depth differs from CPU oracle')
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
    blue_noise = None
    if manifest['version'] in ('0.17.0','0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        blue_noise=verify_blue_noise(root/'blue_noise_gpu.csv')
        if manifest.get('blue_noise_validation_cases')!=blue_noise['cases']:
            raise ValueError('Missing blue-noise validation')
        tile=float(manifest['blue_noise_tile_m'])
        if not math.isfinite(tile) or not .01<=tile<=64:
            raise ValueError('Invalid blue-noise tile size')
    contact_blend = None
    if manifest['version'] in ('0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        contact_blend=verify_contact_blend(root/'contact_blend_gpu.csv')
        if manifest.get('contact_blend_validation_cases')!=contact_blend['cases'] or manifest.get('contact_guard_samples')!=12:
            raise ValueError('Missing contact/blend verification')
    prefilter = None
    if manifest['version'] in ('0.14.0','0.14.1','0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1','0.17.0','0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        prefilter = verify_prefilter(root/'prefilter_gpu.csv')
        if manifest.get('prefilter_validation_cases') != prefilter['cases']:
            raise ValueError('Missing prefilter validation')
    intervals = None
    if manifest['version'] in ('0.14.1','0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1','0.17.0','0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        intervals=verify_intervals(root/'interval_gpu.csv')
        if manifest.get('interval_validation_cases')!=intervals['cases']:
            raise ValueError('Missing interval validation')
    angular = angular_filter = None
    if manifest['version'] in ('0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1','0.17.0','0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
        angular=verify_angular(root/'angular_gpu.csv')
        angular_filter=verify_angular_filter(root/'angular_filter_gpu.csv')
        if manifest.get('angular_validation_cases')!=angular['cases'] or manifest.get('angular_filter_validation_cases')!=angular_filter['cases']:
            raise ValueError('Missing angular verification')
        if manifest.get('oracle_smoothing')!=0:
            raise ValueError('Oracle must remain unfiltered')
    if manifest["version"] in ("0.15.2", "0.15.3", "0.16.0", "0.16.1", "0.17.0", "0.18.0", "0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0") and manifest.get("tri_result_masks") not in (0,1):
        raise ValueError("Missing triangulation result mode")
    if manifest["version"] in ("0.15.3", "0.16.0", "0.16.1", "0.17.0", "0.18.0", "0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0"):
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
        if manifest["version"] in ("0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0"):verify_rgba_pcf_options(row,manifest)
        verify_gsm_options(row,manifest)
        if row["method"] not in METHODS or (row["method"]=="gsm" and manifest["version"]!="0.31.0"):
            raise ValueError("Unexpected method")
        mode = int(row["smoothing"] if "smoothing" in row else row["geometric_smoothing"])
        if mode not in range(8 if manifest["version"] != "0.11.0" else 6):
            raise ValueError("Invalid smoothing mode")
        if manifest["version"] != "0.11.0":
            configured = row['configured_sample_pattern']
            if configured not in (('vogel', 'random', 'stratified', 'blue-noise') if manifest['version'] in ('0.17.0','0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") else ('vogel', 'random', 'stratified')):
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
        if manifest["version"] in ("0.13.0", "0.14.0", "0.14.1", "0.15.0", "0.15.1", "0.15.2", "0.15.3", "0.16.0", "0.16.1", "0.17.0", "0.18.0", "0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0"):
            configured_penumbra = manifest['configured_penumbra']
            if configured_penumbra not in ('inner', 'full'):
                raise ValueError("Invalid penumbra setting")
            want = configured_penumbra if row['method'] in ('pcss', 'geometric', 'geometric-layered') else 'native'
            if row['penumbra_mode'] != want:
                raise ValueError("Incorrect effective penumbra mode")
        if row["sample_pattern"] not in (('vogel','random','stratified','blue-noise') if manifest['version'] in ('0.17.0','0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") else ('vogel','random','stratified')) or int(row['sample_seed'])<0 or int(row['rotate_samples']) not in (0,1):
            raise ValueError("Invalid PCSS sampling configuration")
        if manifest['version'] in ('0.17.0','0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
            blue=row['sample_pattern']=='blue-noise'
            expected_tile=float(manifest['blue_noise_tile_m']) if blue else 0
            tile=float(row['blue_noise_tile_m'])
            if (int(row['sampling_texture_bytes'])!=(2048**2 if blue else 0) or
                    not math.isfinite(tile) or abs(tile-expected_tile)>1e-5):
                raise ValueError('Incorrect blue-noise resource/tile accounting')
        for key in ("map_gpu_ms", "shadow_gpu_ms", "total_gpu_ms", "rmse", "penumbra_rmse", "gradient_rmse",
                    "max_abs_error", "monotonic_max_brightening", "monotonic_violation_fraction"):
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError("Invalid image/timing metric")
        if int(row["invalid_pixels"]) or int(row["receiver_pixels"]) <= 0:
            raise ValueError("Invalid image metrics")
        is_arr = row["method"] in ("arr","arr-triangulation")
        storage_layout=multiview_storage(row,manifest) if manifest['version'] in ('0.29.0','0.30.0',"0.31.0") else ('rgba32f-angular' if row['method']=='angular-rgba' else None)
        expected = 0 if is_arr else 4 if storage_layout in ('d32f-array-angular','d32f-separate-angular') else 1
        prefiltered = row['method'] in ('vsm','csm','esm','gsm')
        terms = int(row.get('csm_terms',8))
        if prefiltered and (terms<4 or terms>16 or terms%2):
            raise ValueError('Invalid Fourier order')
        bytes_per_pixel = 16 if row['method']=='vsm' else 8 if row['method'] in ('esm','gsm') else 16*terms if row['method']=='csm' else 0
        auxiliary = bytes_per_pixel*int(row['map_resolution'])**2
        if (int(row['map_count']) != expected or int(row['depth_layers']) != expected or
                int(row['map_generation_passes']) != (4 if row['method']=='angular-rgba' else expected) or int(row['auxiliary_map_bytes']) != auxiliary or
                int(row['map_bytes']) != int(row['map_resolution'])**2*(16 if row['method']=='angular-rgba' else expected*4)+auxiliary):
            raise ValueError('Shadow resource accounting violated')
        if manifest['version'] in ('0.14.0','0.14.1','0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1','0.17.0','0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") and int(row['prefilter_passes']) != (3 if prefiltered else 0):
            raise ValueError('Incorrect prefilter pass count')
        if manifest['version'] in ('0.14.1','0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1','0.17.0','0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
            packed=row['method'] in (('pcss','geometric','geometric-layered') if manifest['version'] in ('0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") else ('geometric-layered',)) and manifest['configured_layer_depth']=='interval'
            storage=storage_layout if row['method']=='angular-rgba' else 'none' if is_arr else 'r32ui-unorm16-minmax' if packed else 'd32f'
            components=4 if row["method"]=="angular-rgba" else 0 if is_arr else 2 if packed else 1
            if row['map_storage']!=storage or int(row['stored_depth_components'])!=components:
                raise ValueError('Incorrect depth representation accounting')
        if manifest['version'] in ('0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1','0.17.0','0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0"):
            configured=int(manifest['arr_smoothing'])
            width=float(manifest['arr_smoothing_radius'])
            smoothing=configured if is_arr and width>0 else 0
            if smoothing not in (0,1,3,4,5) or int(row['angular_smoothing'])!=smoothing:
                raise ValueError('Incorrect angular smoothing')
            if int(row['shadow_passes'])!=1+int(smoothing!=0)+(verify_result_options(row,manifest) if manifest['version'] in ('0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") else 0):
                raise ValueError('Incorrect angular pass count')
            expected_queries=(int(manifest['arr_samples']) or int(manifest['reference_samples'])) if row['method']=='arr' else int(manifest['tri_samples']) if row['method']=='arr-triangulation' else 0
            expected_bytes=8*int(row['width'])*int(row['height']) if smoothing else 0
            if int(row['angular_queries'])!=expected_queries or int(row['screen_filter_bytes'])!=expected_bytes:
                raise ValueError('Incorrect angular query/resource accounting')
            if int(row['angular_direction_bytes'])!=(16384*16 if row['method']=='arr' else 0):
                raise ValueError('Incorrect direction buffer accounting')
        if manifest["version"] in ("0.15.2", "0.15.3", "0.16.0", "0.16.1", "0.17.0", "0.18.0", "0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0"):
            expected_masks=int(row["method"]=="arr-triangulation" and manifest["tri_result_masks"]==1)
            if int(row["tri_result_masks"])!=expected_masks:
                raise ValueError("Incorrect triangulation result mode")
        if manifest["version"] in ("0.15.3", "0.16.0", "0.16.1", "0.17.0", "0.18.0", "0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0"):
            expected_min=float(manifest["layer_min_filter_texels"]) if row["method"]=="geometric-layered" else 0
            expected_shared=int(row["sample_pattern"] not in (("vogel","blue-noise") if manifest["version"] in ("0.17.0","0.18.0", "0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0") else ("vogel",)) and manifest["shared_sample_pattern"])
            if abs(float(row["layer_min_filter_texels"])-expected_min)>1e-5 or int(row["shared_sample_pattern"])!=expected_shared:
                raise ValueError("Incorrect effective soft packed configuration")
        if not is_arr and int(row["shadow_passes"]) != 1+(verify_result_options(row,manifest) if manifest["version"] in ("0.18.0","0.19.0", "0.20.0", "0.21.0", "0.22.0","0.23.0","0.24.0","0.25.0","0.26.0","0.27.0","0.28.0","0.29.0","0.30.0","0.31.0") else 0):
            raise ValueError("Incorrect PCSS evaluation/result-blend pass count")
        oracle_candidate=(row["method"]=="arr" and (manifest['version'] not in ('0.15.0','0.15.1','0.15.2','0.15.3','0.16.0','0.16.1','0.17.0','0.18.0','0.19.0','0.20.0','0.21.0','0.22.0','0.23.0','0.24.0','0.25.0','0.26.0','0.27.0','0.28.0',"0.29.0","0.30.0","0.31.0") or
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
    summary = {"cases": len(rows), "hardware_timings": not manifest["software_renderer"],"multiview_storage_oracles":multiview,"rgba_depth_scale_oracles":rgba_depth_scale,"rgba_pcss_oracles":rgba_pcss,"rgba_blend_oracles":rgba_blend,"rgba_shared_oracles":rgba_shared,"rgba_moment_oracles":rgba_moments,
               "reference_converged": converged, "prefilter_oracles": prefilter, "angular_oracles": angular, "angular_filter_oracles": angular_filter, "interval_oracles": intervals, "filter_oracles": filters, "sampling_oracles":sampling, "blue_noise_oracles":blue_noise, "contact_blend_oracles":contact_blend,"rgba_mix_oracles":rgba_mix,"rgba_edge_oracles":rgba_edge,"rgba_pcf_oracles":rgba_pcf,
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
