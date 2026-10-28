#!/usr/bin/env python3
"""Independent CPU filtering oracle for near-only support and the result blend."""
import csv
import math
from pathlib import Path
from verify_sampling import f32


def fixture_depths(x, y, packed):
    if y == 15:
        return (1.0, 1.0)
    a, b = (.495 if x < 4 else .4), (.495 if x < 8 else .41)
    if y == 0:
        a = b = .51
    if packed:
        return tuple(math.floor(f32(f32(v) * 65535) + .5) / 65535 for v in (a, b))
    return (f32(a), f32(a))


def contact(case, packed):
    uv = ((case % 16 + f32(.37)) / 16, (case // 16 + f32(.63)) / 4)
    origin = tuple(v * 2 - 1 for v in uv)
    depth = .5 + (case % 3) * f32(.001)
    gradient = ((case % 3 - 1) * f32(.002), (case // 3 % 3 - 1) * f32(.001))

    def stats(p):
        st = tuple((v * .5 + .5) * 16 - .5 for v in p)
        base = tuple(math.floor(v) for v in st)
        fraction = tuple(v - b for v, b in zip(st, base))
        logs = coverage = 0.0
        for y in range(2):
            for x in range(2):
                tx, ty = base[0] + x, base[1] + y
                if not 0 <= tx < 16 or not 0 <= ty < 16:
                    continue
                center = ((tx + .5) / 8 - 1, (ty + .5) / 8 - 1)
                receiver = depth + sum(g * (c - o) for g, c, o in zip(gradient, center, origin))
                valid = [receiver - d for d in fixture_depths(tx, ty, packed)
                         if d < 1 and receiver - d > f32(.00015) and (receiver - d) * 80 < 1]
                if not valid:
                    continue
                gap = min(valid)
                t = max(0.0, min(1.0, (gap * 80 - .75) / .25))
                weight = (fraction[0] if x else 1 - fraction[0]) * (fraction[1] if y else 1 - fraction[1]) * (1 - t * t * (3 - 2 * t))
                coverage += weight
                logs += weight * math.log(max(gap, 1e-8))
        return logs, coverage

    logs, center_coverage = stats(origin)
    if center_coverage <= 1e-6:
        return (0.0,) * 4
    gap = math.exp(logs / center_coverage)
    radius = min(gap * 80 * f32(.026) / f32(.2), 3 / f32(.2))
    coverages = [center_coverage]
    if radius > .03125:
        coverages = []
        for i in range(12):
            r = math.sqrt((i + .5) / 12) * radius
            a = i * 2.39996322972865
            coverages.append(stats((origin[0] + math.cos(a) * r, origin[1] + math.sin(a) * r))[1])
    return min(center_coverage, sum(coverages) / len(coverages)), radius * f32(.2), center_coverage, gap * 80


def blend_fixture(x, y):
    visibility = 1.0 if x == 0 else 0.0 if x == 3 else f32(((x * 13 + y * 7 + x * y * 3) % 97) / 96)
    depth = 1.0 if y == 15 else .5 if x < 8 else f32(.7)
    normal = (0, 1, 0) if x == 12 else (0, 0, 1)
    guide = (f32(.01 if x < 8 else .1), 1.0 if x == 3 else 0.0)
    return visibility, depth, normal, guide


def blended(mode, x, y):
    value, depth, normal, guide = blend_fixture(x, y)
    strength = 0.0 if mode == 0 else 1.0 if mode == 2 else f32(.8)
    if depth >= 1 or strength == 0:
        return value
    radius, sigma = (1 if mode < 2 else 2), f32(.1 if mode == 3 else .2)
    total = weighted = 0.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            qx, qy = x + dx, y + dy
            if not 0 <= qx < 16 or not 0 <= qy < 16:
                continue
            qv, qdepth, qnormal, qguide = blend_fixture(qx, qy)
            if qdepth >= 1 or sum(a * b for a, b in zip(normal, qnormal)) < .95:
                continue
            delta = (dx / 8, dy / 8, 2 * (qdepth - depth))
            if max(abs(sum(a * b for a, b in zip(delta, n))) for n in (normal, qnormal)) > .03125:
                continue
            spatial = ((2 if dx == 0 else 1) * (2 if dy == 0 else 1)) if radius == 1 else (6, 4, 1)[abs(dx)] * (6, 4, 1)[abs(dy)]
            dv = (qv - value) / sigma
            dr = abs(math.log((qguide[0] + .001) / (guide[0] + .001)))
            dc = (qguide[1] - guide[1]) / .12 if mode >= 2 else 0.0
            weight = spatial * math.exp(-.5 * dv * dv - 2 * dr - .5 * dc * dc)
            total += weight
            weighted += weight * qv
    result = value * (1 - strength) + (weighted / total if total else value) * strength
    if mode >= 2:
        result = min(result, 1 - guide[1])
    if mode == 3 and value >= .999999:
        result = value
    return max(0, min(1, result))


def verify(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_keys = {("contact", mode, case, 0) for mode in range(2) for case in range(64)}
    expected_keys |= {("blend", mode, x, y) for mode in range(4) for y in range(16) for x in range(16)}
    keys = {(r['kind'], int(r['mode']), int(r['x']), int(r['y'])) for r in rows}
    if len(rows) != 1152 or keys != expected_keys:
        raise ValueError('Incomplete contact/blend validation')
    maximum = {'contact': 0.0, 'blend': 0.0}
    hidden_supported = 0
    for row in rows:
        kind, mode, x, y = row['kind'], int(row['mode']), int(row['x']), int(row['y'])
        expected = contact(x, mode == 1) if kind == 'contact' else (blended(mode, x, y), 0, 0, 0)
        actual = tuple(float(row[k]) for k in ('value', 'radius', 'coverage', 'gap'))
        error = max(abs(a - b) for a, b in zip(actual, expected))
        if any(not math.isfinite(v) for v in actual) or error > (0.0002 if kind == 'contact' else 0.00002):
            raise ValueError(f'{kind} CPU/GPU mismatch: {mode}/{x}/{y}: {actual} vs {expected}')
        maximum[kind] = max(maximum[kind], error)
        if not 0 <= actual[0] <= 1.000001:
            raise ValueError('Contact/blend range failure')
        if kind == 'blend' and mode >= 2 and x == 3 and y < 15 and actual[0] != 0:
            raise ValueError('Blend lightened a protected core')
        if kind == 'contact' and mode == 1 and 5 <= x % 16 <= 6 and actual[0] > .9:
            hidden_supported += 1
    if hidden_supported == 0:
        raise ValueError('Packed endpoints failed to preserve hidden near support')
    return {'cases': len(rows), 'max_abs_error': maximum, 'protected_core_violations': 0,
            'packed_hidden_support_cases': hidden_supported, 'geometry_step_guard': True}
