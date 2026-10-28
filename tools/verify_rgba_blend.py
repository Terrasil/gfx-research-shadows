#!/usr/bin/env python3
"""Check invariants and error reduction on native joint reconstruction fixtures."""
import csv
import math
from pathlib import Path


def verify(path):
    rows = list(csv.DictReader(Path(path).open()))
    keys = {(int(r['scene']), int(r['mode']), int(r['x']), int(r['y'])) for r in rows}
    if len(rows) != 6144 or len(keys) != 6144 or keys != {
            (s, m, x, y) for s in range(8) for m in range(3) for y in range(16) for x in range(16)}:
        raise ValueError('Incomplete angular blend fixture matrix')
    exact_error = 0.0
    noise_errors = [[], [], []]
    for row in rows:
        scene, mode, x, y = (int(row[k]) for k in ('scene', 'mode', 'x', 'y'))
        original, ideal, value = (float(row[k]) for k in ('input', 'ideal', 'gpu'))
        if not all(math.isfinite(v) for v in (original, ideal, value)) or not -1e-6 <= value <= 1.000001:
            raise ValueError('Invalid angular blend value')
        exact = mode == 0 or scene in (0, 3, 4, 5, 6, 7) or (scene == 1 and 2 <= x <= 13 and 2 <= y <= 13)
        if exact:
            exact_error = max(exact_error, abs(value-original))
        if scene == 2 and 2 <= x <= 13 and 2 <= y <= 13:
            noise_errors[mode].append((value-ideal)**2)
    if exact_error > 2e-6:
        raise ValueError('Angular blend altered a protected edge, constant, ramp, or disabled result')
    rms = [math.sqrt(sum(v)/len(v)) for v in noise_errors]
    if not rms[1] < rms[0] * .7 or not rms[2] < rms[0] * .7:
        raise ValueError('Angular reconstruction did not reduce fixture noise')
    return {'cases':len(rows), 'preservation_max_abs_error':exact_error,
            'noisy_ramp_rmse':rms, 'protected_edge_violations':0}


if __name__ == '__main__':
    import json
    import sys
    print(json.dumps(verify(sys.argv[1]), indent=2))
