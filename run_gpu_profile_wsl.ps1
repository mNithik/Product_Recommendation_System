$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$wslProjectPath = $projectRoot -replace '^([A-Za-z]):', '/mnt/$1'
$wslProjectPath = $wslProjectPath -replace '\\', '/'
$wslProjectPath = $wslProjectPath.ToLower()

Write-Host "Running WSL GPU profile experiment..."
Write-Host "WSL path: $wslProjectPath"

$cmd = @"
set -euo pipefail
cd '$wslProjectPath'
source .venv-wsl-gpu/bin/activate
python scripts/verify_implicit_gpu.py
python main.py --config configs/gpu_profile.yaml --experiment wsl_gpu_run
"@

wsl bash -lc $cmd
