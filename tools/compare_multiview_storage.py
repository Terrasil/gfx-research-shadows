#!/usr/bin/env python3
"""Run the same multi-view configuration with three storage backends."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

from analyze_results import read_csv, read_pfm

ROOT = Path(__file__).resolve().parents[1]


def main(four_variants=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=ROOT / 'cmake-build-release' /
                        ('gfx-research-shadows.exe' if sys.platform == 'win32' else 'gfx-research-shadows'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / ('multiview-variants' if four_variants else 'multiview-storage'))
    parser.add_argument('--profile', choices=('quality', 'stress'), default='quality' if four_variants else 'stress')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--require-hardware', action='store_true')
    args, extra = parser.parse_known_args()
    # These options would invalidate the matched configuration or output tree.
    forbidden = ('--output', '--method', '--multiview-variant', '--multiview-storage', '--rgba-storage', '--benchmark',
                 '--stress-benchmark', '--smoke', '--publication-quick', '--publication-suite', '--headless', '--help', '--version')
    if any(value.split('=', 1)[0] in forbidden for value in extra):
        parser.error('Pass method/storage/profile/output through this comparison tool only')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Comparison output must be empty; existing measurements are not overwritten')
    output.mkdir(parents=True, exist_ok=True)
    rows, baseline, baseline_manifest = [], {}, None
    software = False
    entries = [('rgba', 'rgba'), ('separate', 'separate'), ('rgba-cascaded', 'rgba'), ('separate-cascaded', 'separate')] if four_variants else [(s, s) for s in ('rgba', 'array', 'separate')]
    for variant, storage in entries:
        destination = output / variant
        command = [sys.executable, str(ROOT / 'tools' / 'run_experiment.py'),
                   '--profile=' + ('manual' if args.profile == 'quality' else 'stress'),
                   '--exe=' + str(args.exe.resolve()), '--output=' + str(destination)]
        if args.headless:
            command.append('--headless')
        choice = '--multiview-variant=' + variant if four_variants else '--multiview-storage=' + storage
        command += extra + ['--method=multiple-light-view', choice]
        subprocess.run(command, cwd=ROOT, check=True)
        manifest = json.loads((destination / 'run_manifest.json').read_text(encoding='utf-8'))
        software |= manifest['software_renderer']
        if args.require_hardware and software:
            raise ValueError('Software renderer: this run is not a hardware performance comparison')
        ignored = {'multiview_storage', 'multiview_variant'} | ({'rgba_cascades'} if four_variants else set())
        comparable = {k: v for k, v in manifest.items() if k not in ignored}
        if four_variants and manifest['multiview_variant'] != variant:
            raise ValueError('Native renderer did not activate the requested comparison variant')
        if baseline_manifest is None:
            baseline_manifest = comparable
        elif comparable != baseline_manifest:
            raise ValueError('Comparison changes settings other than map storage')
        captures = read_csv(destination / 'capture_manifest.csv')
        for row in read_csv(destination / 'raw.csv'):
            if row['method'] != 'angular-rgba':
                continue
            case = row['case_id']
            pair = (case, bool(manifest['rgba_cascades']))
            difference_rmse, difference_max = '', ''
            if args.profile == 'quality':
                visibility = next(c for c in captures if c['case_id'] == case and c['kind'] == 'visibility-pfm')
                width, height, values = read_pfm(destination / visibility['path'])
                if storage == 'rgba':
                    baseline[pair] = (width, height, values)
                bw, bh, previous = baseline[pair]
                if (width, height) != (bw, bh):
                    raise ValueError('Comparison changes output resolution')
                differences = [abs(a - b) for a, b in zip(values, previous)]
                difference_rmse = math.sqrt(sum(d*d for d in differences)/len(differences))
                difference_max = max(differences)
            result = dict(case_id=case, scene=row['scene'], objects=row.get('objects', ''), variant=manifest['multiview_variant'], cascades=int(manifest['rgba_cascades']), storage=storage,
                          map_bytes=row['map_bytes'], textures=row['map_texture_count'],
                          map_gpu_ms=row['map_gpu_ms'], shadow_gpu_ms=row['shadow_gpu_ms'],
                          total_gpu_ms=row['total_gpu_ms'], hardware_timings=int(not manifest['software_renderer']),
                          visibility_rmse_vs_rgba=difference_rmse,
                          visibility_max_error_vs_rgba=difference_max,
                          rmse_vs_geometry=row.get('rmse', ''), penumbra_rmse_vs_geometry=row.get('penumbra_rmse', ''))
            rows.append(result)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output / 'comparison.json').write_text(json.dumps(dict(profile=args.profile, cases=len(rows),
        hardware_timings=not software, backend_order=[e[1] for e in entries], variant_order=[e[0] for e in entries],
        visibility_comparison='RGBA with the same cascade setting',
        timing_scope=baseline_manifest['timing_scope'],
        note='Array versus separate uses the same D32F raster path. RGBA uses color MIN instead of depth testing.',
        completed=True), indent=2) + '\n', encoding='utf-8')
    print('Saved', output / 'comparison.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
