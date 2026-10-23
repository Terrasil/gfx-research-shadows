#!/usr/bin/env python3
"""Compare fixed and distance-adaptive exponents with identical native settings."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Other native options, including camera and scene, are forwarded to every run.")
    parser.add_argument("--exe", type=Path, default=ROOT / "cmake-build-release" / ("gfx-research-shadows.exe" if sys.platform == "win32" else "gfx-research-shadows"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/exponent-sweep")
    parser.add_argument("--exponents", type=float, nargs="+", default=[0, 0.25, 0.5, 1, 2])
    parser.add_argument("--headless", action="store_true")
    args, extra = parser.parse_known_args()
    if any(not math.isfinite(y) or not 0 <= y <= 4 for y in args.exponents):
        parser.error("Exponents must be finite values in [0,4]")
    if len(set(args.exponents)) != len(args.exponents):
        parser.error("Duplicate exponents would overwrite a run")
    if any(value.split("=", 1)[0] in ("--method", "--exponent", "--profile", "--benchmark", "--smoke", "--publication-suite", "--publication-quick") for value in extra):
        parser.error("This tool owns method, exponent and profile selection")
    variants = [("ndc-pcss", None), ("geometric-ndc-pcss", None)]
    variants += [(method, y) for method in ("power-ndc-pcss", "inverse-ndc-pcss") for y in args.exponents]
    variants += [("distance-power-ndc-pcss", None), ("distance-inverse-ndc-pcss", None)]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"completed": False, "variants": [], "shared_arguments": extra}
    path = args.output / "sweep_manifest.json"
    rows, reference_hash = [], None
    for method, y in variants:
        label = method + ("-y-" + format(y, ".9g").replace(".", "p") if y is not None else "")
        output = args.output / label
        command = [sys.executable, str(ROOT / "tools/run_experiment.py"), "--profile", "manual", "--exe", str(args.exe.resolve()), "--output", str(output.resolve())]
        if args.headless:
            command.append("--headless")
        command += extra + ["--method=" + method]
        if y is not None:
            command.append("--exponent=" + str(y))
        manifest["variants"].append({"label": label, "command": command})
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
        digest = hashlib.sha256((output / "captures/c0-reference.pfm").read_bytes()).hexdigest()
        if reference_hash is not None and digest != reference_hash:
            raise RuntimeError("Sweep variants do not have the same saved reference")
        reference_hash = digest
        with (output / "raw.csv").open(newline="", encoding="utf-8") as stream:
            rows += [{"variant": label, **row} for row in csv.DictReader(stream)]
    with (args.output / "sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest.update(completed=True, cases=len(rows), reference_sha256=reference_hash)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(variants)} variants / {len(rows)} cases: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
