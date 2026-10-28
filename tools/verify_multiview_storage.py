#!/usr/bin/env python3
"""Check native raster/filter equivalence and backend-switch fixtures."""
import csv
import math
from collections import defaultdict
from pathlib import Path


def verify(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[int(row['case_id'])].append(row)
    if len(rows) != 240 or set(groups) != set(range(60)):
        raise ValueError('Incomplete multi-view storage fixture matrix')
    maxima = dict(depth=0.0, visibility_rmse=0.0, visibility=0.0, guide=0.0, array_separate=0.0)
    configurations = set()
    for group in groups.values():
        if [r['storage'] for r in group] != ['rgba', 'array', 'separate', 'rgba']:
            raise ValueError('Incomplete multi-view backend switch sequence')
        first = group[0]
        configuration = tuple(first[k] for k in ('scene', 'map_backend', 'cascades', 'filter'))
        if configuration in configurations:
            raise ValueError('Duplicate multi-view configuration')
        configurations.add(configuration)
        for row in group:
            if any(row[k] != first[k] for k in ('scene', 'map_backend', 'cascades', 'warp', 'filter', 'mix', 'smoothing')):
                raise ValueError('Multi-view comparison changes filter or camera settings')
            expected_layers = 1 if row['storage'] == 'rgba' else 4
            expected_textures = 4 if row['storage'] == 'separate' else 1
            if (int(row['map_bytes']) != 64 * 64 * 16 or int(row['layers']) != expected_layers or
                    int(row['textures']) != expected_textures or int(row['gl_error']) != 0):
                raise ValueError('Invalid multi-view resources or GL state')
            for key, column, limit in [('depth', 'depth_max_error', 2e-5), ('visibility_rmse', 'visibility_rmse', 1e-4),
                                       ('visibility', 'visibility_max_error', .002), ('guide', 'guide_max_error', float('inf')),
                                       ('array_separate', 'array_separate_max_error', 1e-5)]:
                value = float(row[column])
                if not math.isfinite(value) or value < 0 or value > limit:
                    raise ValueError('Multi-view storage equivalence failed: ' + column)
                maxima[key] = max(maxima[key], value)
    expected = {(scene, backend, str(cascades), str(mode))
                for scene in ('mixed-overlap', 'crossing-four', 'sloped')
                for backend in ('raster', 'analytic') for cascades in (0, 1) for mode in range(5)}
    if configurations != expected:
        raise ValueError('Missing multi-view scene/filter coverage')
    return dict(cases=len(rows), configurations=len(groups), max_error=maxima)
