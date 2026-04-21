$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$wslProjectPath = $projectRoot -replace '^([A-Za-z]):', '/mnt/$1'
$wslProjectPath = $wslProjectPath -replace '\\', '/'
$wslProjectPath = $wslProjectPath.ToLower()

Write-Host "Setting up implicit GPU in WSL..."
Write-Host "WSL path: $wslProjectPath"

$cmd = @"
set -euo pipefail
cd '$wslProjectPath'
chmod +x scripts/setup_implicit_gpu_wsl.sh
bash scripts/setup_implicit_gpu_wsl.sh
"@

wsl bash -lc $cmd
