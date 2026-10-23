#!/usr/bin/env python3
"""CPU oracles for complete filter kernels. CHS uses upstream contactFixedPCF, not gather4."""
import csv
import math
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def original_tables(size):
    source = (ROOT / f'upstream/shadowfx/AMD_SHADOWFX_FILTER_SIZE_{size}_FIXED.inc').read_text()
    tables = []
    for name in ('low', 'medium', 'high', 'ultra'):
        match = re.search(r'g_' + name + r'GaussianWeight\[[^;]+?=\s*\{(.*?)\};', source, re.S)
        values = list(map(float, re.findall(r'[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', match.group(1))))
        if len(values) != size * size:
            raise ValueError('Invalid upstream CHS table')
        tables.append(values)
    return tables


def original_poisson():
    source = (ROOT / 'upstream/filament/shadowing.fs').read_text()
    source = source.split('poissonDisk[64]', 1)[1].split('// tap count', 1)[0]
    return [tuple(map(float, p)) for p in re.findall(r'vec2\(([-.\d]+),\s*([-.\d]+)\)', source)]


def fixture_depth(x, y):
    if not (0 <= x < 32 and 0 <= y < 32):
        return 1.0
    return f32(((x * 13 + y * 7 + x * y * 3) % 89 + 5) / 100)


def bilinear_compare(u, v, z):
    x, y = u * 32 - 0.5, v * 32 - 0.5
    ix, iy = math.floor(x), math.floor(y)
    fx, fy = x - ix, y - iy
    return sum(float(z <= fixture_depth(ix + dx, iy + dy)) *
               (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
               for dy in (0, 1) for dx in (0, 1))


def chs_oracle(u, v, z, sun_area, size, tables):
    """Independent, straightforward AMD contactFixedPCF backend (FS^2 bilinear tests)."""
    biased = f32(z - f32(0.00015))
    cx, cy = math.floor(u * 32 + 0.5), math.floor(v * 32 + 0.5)
    fr = size // 2
    blockers = []
    for row in range(-fr, fr + 1, 2):
        for col in range(-fr, fr + 1, 2):
            for dy in (-1, 0):
                for dx in (-1, 0):
                    depth = fixture_depth(cx + col + dx, cy + row + dy)
                    if depth < biased:
                        blockers.append(depth)
    count = len(blockers)
    if count in (0, (size + 1) ** 2):
        return 1.0 - count / (size + 1) ** 2
    blocker = sum(blockers) / count
    t = math.sqrt(max(0, min(1, (z - blocker) * sun_area / max(blocker, 1e-8))))
    coefficients = ((1-t)**3, 3*(1-t)**2*t, 3*t*t*(1-t), t**3)
    weights = [sum(c * table[i] for c, table in zip(coefficients, tables)) for i in range(size * size)]
    total = sum(bilinear_compare(u + col/32, v + row/32, biased) * weights[(row+fr)*size+col+fr]
                for row in range(-fr, fr+1) for col in range(-fr, fr+1))
    return total / sum(weights)


def dpcf_oracle(u, v, z, slope, taps, ratio_scale, angle, poisson):
    blockers = []
    c, s = math.cos(angle), math.sin(angle)
    for x, y in poisson[:taps]:
        dx, dy = (c*x-s*y)*2.75/32, (s*x+c*y)*2.75/32
        depth = fixture_depth(math.floor((u+dx)*32), math.floor((v+dy)*32))
        if z + slope[0]*dx + slope[1]*dy >= depth:
            blockers.append(depth)
    if not blockers:
        return 1.0
    average = sum(blockers)/len(blockers)
    ratio = max(0, min(1, (z-average)/max(average, 1e-8)*ratio_scale))
    p = len(blockers)/taps
    # Closed piecewise polynomial, independent of Filament's sign-based shader expression.
    hardened = 4*p**3 if p <= 0.5 else 1-4*(1-p)**3
    return 1 - ((1-ratio)*hardened + ratio*p)


def smooth_oracle(u, v, z, gradient, hermite):
    # Explicit four-corner texel fetches, not GPU gather ordering.
    def unwarp(p):
        return p / (9 - 8 * abs(p))
    origin = (unwarp(u*2-1), unwarp(v*2-1))
    x, y = u*32-.5, v*32-.5
    ix, iy = math.floor(x), math.floor(y)
    fx, fy = x-ix, y-iy
    if hermite:
        fx, fy = fx*fx*(3-2*fx), fy*fy*(3-2*fy)
    occluded, log_sum, gap_sum, support = 0.0, 0.0, 0.0, 0.0
    for dy in (0, 1):
        for dx in (0, 1):
            depth = fixture_depth(ix+dx, iy+dy)
            center = (unwarp((ix+dx+.5)/32*2-1), unwarp((iy+dy+.5)/32*2-1))
            gap = z + sum(g*(p-o) for g,p,o in zip(gradient, center, origin)) - depth
            weight = (fx if dx else 1-fx)*(fy if dy else 1-fy)
            if depth < 1 and gap > .00015:
                occluded += weight
                log_sum += weight*math.log(gap)
                gap_sum += weight*gap
                support += weight
    return (1-occluded, math.exp(log_sum/support) if support else 0.0,
            gap_sum/support if support else 0.0)


def cubic_oracle(u, v, z, gradient, catmull=False):
    # Scalar cardinal B-spline definition, independent of four gathered quads.
    def basis(distance):
        a = abs(distance)
        if catmull:
            return (1.5*a**3-2.5*a*a+1 if a<1 else -.5*a**3+2.5*a*a-4*a+2 if a<2 else 0.0)
        return ((4-6*a*a+3*a**3)/6 if a < 1 else (2-a)**3/6 if a < 2 else 0.0)
    def unwarp(p):
        return p/(9-8*abs(p))
    origin = (unwarp(u*2-1), unwarp(v*2-1))
    x,y = u*32-.5,v*32-.5
    ix,iy = math.floor(x),math.floor(y)
    occluded = 0.0
    for row in range(iy-1,iy+3):
        for col in range(ix-1,ix+3):
            depth = fixture_depth(col,row)
            if depth >= 1:
                continue
            center = (unwarp((col+.5)/32*2-1),unwarp((row+.5)/32*2-1))
            gap = z+sum(g*(p-o) for g,p,o in zip(gradient,center,origin))-depth
            if gap > .00015:
                occluded += basis(x-col)*basis(y-row)
    return max(0,min(1,1-occluded))


def verify(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 320 or sorted(int(r['case_id']) for r in rows) != list(range(320)):
        raise ValueError('Incomplete GPU filter validation')
    tables = {size: original_tables(size) for size in (7,9,11,13,15)}
    poisson = original_poisson()
    if len(poisson) != 64:
        raise ValueError('Invalid original DPCF disk')
    errors = dict.fromkeys(('dpcf', 'chs', 'smooth_visibility', 'smooth_gap'), 0.0)
    limits = {'dpcf': 0.00003, 'chs': 0.00005, 'smooth_visibility': 0.000004, 'smooth_gap': 0.000004}
    if 'smooth_mean_gap' in rows[0]:
        errors['smooth_mean_gap'] = 0.0
        limits['smooth_mean_gap'] = 0.000004
    for row in rows:
        case = int(row['case_id'])
        i = case%80
        reconstruction = case//80+2
        hermite = reconstruction==3
        if int(row['reconstruction']) != reconstruction:
            raise ValueError('Invalid interpolation variant')
        u, v = f32(f32((i*7)%29 + f32(.37))/32), f32(f32((i*11)%29+f32(.61))/32)
        z = f32(f32((i%8)*f32(.1477))+f32(.053))
        slope = (f32((i%3-1)*f32(.017)), f32(((i//3)%3-1)*f32(.017)))
        size = 7+2*(i%5)
        visible, gap, mean = smooth_oracle(u,v,z,slope,hermite)
        if reconstruction in (4,5):
            visible = cubic_oracle(u,v,z,slope,reconstruction==5)
        expected = {'dpcf': dpcf_oracle(u,v,z,slope,1+(i*11)%64,1+(i%4)*1.5,.21*(i%7),poisson),
                    'chs': chs_oracle(u,v,z,.25+(i%9)*3,size,tables[size]),
                    'smooth_visibility': visible, 'smooth_gap': gap}
        if 'smooth_mean_gap' in row:
            expected['smooth_mean_gap'] = mean
        for name, want in expected.items():
            actual = float(row[name])
            error = abs(actual-want)
            if not math.isfinite(actual) or error > limits[name]:
                raise ValueError(f'{name} GPU/CPU mismatch in fixture {i}: {actual} vs {want} (error {error})')
            errors[name] = max(errors[name], error)
    return {'cases': len(rows), 'max_abs_error': errors}


if __name__ == '__main__':
    import json
    import sys
    print(json.dumps(verify(sys.argv[1]), indent=2))
