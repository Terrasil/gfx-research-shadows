#!/usr/bin/env python3
"""Run a native suite and retain source hashes alongside the measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["manual", "smoke", "quick", "full", "stress"], default="quick")
    parser.add_argument("--exe", type=Path, default=ROOT / "cmake-build-release" / ("gfx-research-shadows.exe" if sys.platform == "win32" else "gfx-research-shadows"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--analytic", action="store_true")
    args, extra = parser.parse_known_args()
    output = args.output or ROOT / "results" / args.profile
    flag = {"manual": "--benchmark", "smoke": "--smoke", "quick": "--publication-quick", "full": "--publication-suite", "stress": "--stress-benchmark"}[args.profile]
    command = [str(args.exe.resolve()), flag, f"--output={output.resolve()}"]
    if args.headless:
        command.append("--headless")
    if args.analytic:
        command.append("--map-backend=analytic")
    command += extra
    source_hashes = {}
    for directory in ("src", "include", "shaders", "tools", "tests", "upstream"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                source_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_hashes["CMakeLists.txt"] = hashlib.sha256((ROOT / "CMakeLists.txt").read_bytes()).hexdigest()
    executable_hash = hashlib.sha256(args.exe.resolve().read_bytes()).hexdigest()
    version = subprocess.run([str(args.exe.resolve()), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    dependencies = {}
    thirdparty = ROOT / "thirdparty"
    if thirdparty.exists():
        for dependency in thirdparty.iterdir():
            if dependency.is_dir():
                revision = subprocess.run(["git", "-C", str(dependency), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
                if revision.returncode == 0:
                    dependencies[dependency.name] = revision.stdout.strip()
    metadata = {"command": command, "platform": platform.platform(), "python": sys.version,
                "executable_sha256": executable_hash, "executable_version": version, "dependencies": dependencies,
                "sha256": source_hashes, "completed": False}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "source_manifest.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    result = subprocess.run(command, cwd=ROOT, check=False)
    metadata["completed"] = result.returncode == 0
    metadata["exit_code"] = result.returncode
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if result.returncode:
        return result.returncode
    return subprocess.run([sys.executable, str(ROOT / "tools/analyze_results.py"), str(output)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
