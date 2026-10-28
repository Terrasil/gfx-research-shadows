#!/usr/bin/env python3
"""CPU oracle for world wrapping, scalar phases and rotated Vogel quadrature."""
import csv
import math
from pathlib import Path
from verify_sampling import pcg


def verify(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 128 or sorted(int(r["case_id"]) for r in rows) != list(range(128)):
        raise ValueError("Incomplete blue-noise validation")
    max_phase_error = max_point_error = 0.0
    for row in rows:
        case = int(row["case_id"])
        tile = (.5, 1, 4, 16)[case // 32]
        world = ((case % 16 - 8) * .125 + .03125, (case // 16 - 4) * .25 + .0625)
        seed = (0, 19, 35, 2147483647)[case % 4]
        shift = (pcg(seed) % 8, pcg(seed ^ 0x68bc21eb) % 8)
        p = tuple((math.floor((v / tile % 1) * 8) + s) % 8 for v, s in zip(world, shift))
        phases = []
        for stream in range(2):
            x, y = ((v + stream * 4) % 8 for v in p)
            code = (x * 17 + y * 29 + 13) % 256
            phases.append((code + .5) * 2 * math.pi / 256)
        actual = [float(row[k]) for k in ("blocker_angle", "filter_angle", "periodic_blocker_angle", "periodic_filter_angle")]
        error = max(abs(a - b) for a, b in zip(actual, phases + phases))
        if any(not math.isfinite(v) for v in actual) or error > 1e-6 or actual[:2] != actual[2:]:
            raise ValueError(f"Blue-noise world phase/period mismatch in case {case}")
        max_phase_error = max(max_phase_error, error)
        count = (1, 3, 4, 32, 48, 96, 256)[case % 7]
        index = case % count
        radius = math.sqrt((index + .5) / count)
        for stream, fields in enumerate((("blocker_x", "blocker_y"), ("filter_x", "filter_y"))):
            angle = index * 2.39996322972865 + phases[stream]
            expected = (radius * math.cos(angle), radius * math.sin(angle))
            point = tuple(float(row[k]) for k in fields)
            error = max(abs(a - b) for a, b in zip(point, expected))
            if any(not math.isfinite(v) for v in point) or error > 1e-4 or sum(v * v for v in point) > 1.000002:
                raise ValueError(f"Blue-noise disk mismatch in case {case}")
            max_point_error = max(max_point_error, error)
    return {"cases": len(rows), "max_phase_error": max_phase_error, "max_point_error": max_point_error,
            "periodic_phase_mismatches": 0}
