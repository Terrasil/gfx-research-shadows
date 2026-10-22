param(
    [ValidateSet("smoke", "quick", "full")][string]$Profile = "quick",
    [string]$Out = "results/quick",
    [string]$Exe = "cmake-build-release/gfx-research-shadows.exe",
    [switch]$Analytic
)
$ErrorActionPreference = "Stop"
$arguments = @("tools/run_experiment.py", "--profile", $Profile, "--exe", $Exe, "--output", $Out)
if ($Analytic) { $arguments += "--analytic" }
python @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
