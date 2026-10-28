#!/usr/bin/env python3
"""Equal-budget geometric PCSS comparisons, retaining the raw ARR oracle."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "blue-noise")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--blocker-samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args, extra = parser.parse_known_args()
    locked = {"--sample-pattern", "--blue-noise-tile", "--sample-seed", "--shared-pattern", "--rotate-samples",
              "--clean-blue-noise", "--method", "--smoothing", "--geometric-smoothing", "--soft-packed", "--output", "--stress-benchmark",
              "--benchmark", "--smoke", "--publication-quick", "--publication-suite", "--profile"}
    if any(flag.split("=", 1)[0] in locked for flag in extra):
        parser.error("Method, sampler and smoothing are fixed by this comparison")
    variants = (
        ("vogel", "vogel", []),
        ("vogel-screen-rotation", "vogel", ["--rotate-samples"]),
        ("random", "random", []),
        ("stratified", "stratified", []),
        ("blue-noise-1m", "blue-noise", ["--blue-noise-tile=1"]),
        ("blue-noise-16m", "blue-noise", ["--blue-noise-tile=16"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, pattern, flags in variants:
        output = args.output / label
        command = [sys.executable, str(ROOT / "tools/run_experiment.py"), "--profile=manual",
                   f"--exe={args.exe.resolve()}", f"--output={output.resolve()}", "--method=geometric",
                   f"--samples={args.samples}", f"--blocker-samples={args.blocker_samples}", "--smoothing=1",
                   "--shared-pattern=0", f"--sample-seed={args.seed}", f"--sample-pattern={pattern}"]
        if args.headless:
            command.append("--headless")
        subprocess.run(command + flags + extra, cwd=ROOT, check=True)
        with (output / "raw.csv").open(newline="") as stream:
            result = next(row for row in csv.DictReader(stream) if row["method"] == "geometric")
        rows.append({"variant": label, **result})
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison: {args.output / 'comparison.csv'}")


if __name__ == "__main__":
    main()
