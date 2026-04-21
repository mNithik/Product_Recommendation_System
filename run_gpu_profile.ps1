$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Running GPU profile experiment..."
Write-Host "Config: configs/gpu_profile.yaml"

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"

python main.py --config configs/gpu_profile.yaml --experiment gpu_profile_run
