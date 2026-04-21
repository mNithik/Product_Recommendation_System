$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Write-Host "=== implicit GPU setup (Windows, Python 3.11) ==="

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$py311 = $null
try {
    $py311 = (py -3.11 -c "import sys; print(sys.executable)").Trim()
} catch {
    Write-Host ""
    Write-Host "Python 3.11 was not found."
    Write-Host "Install it first, then rerun this script."
    Write-Host "Example (winget):"
    Write-Host "  winget install --id Python.Python.3.11 -e"
    exit 1
}

Write-Host "Python 3.11 found: $py311"

$venvPath = Join-Path $projectRoot ".venv311-gpu"
if (-not (Test-Path $venvPath)) {
    & py -3.11 -m venv $venvPath
}

$venvPy = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Error "Virtual environment python not found at $venvPy"
}

Write-Host "Using venv: $venvPath"

# CUDA path setup for this session.
$defaultCuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if (-not $env:CUDA_PATH -and (Test-Path $defaultCuda)) {
    $env:CUDA_PATH = $defaultCuda
}
if (-not $env:CUDAHOME -and $env:CUDA_PATH) {
    $env:CUDAHOME = $env:CUDA_PATH
}

Write-Host "CUDA_PATH: $($env:CUDA_PATH)"
Write-Host "CUDAHOME : $($env:CUDAHOME)"

$hasCl = $null -ne (Get-Command cl.exe -ErrorAction SilentlyContinue)
if (-not $hasCl) {
    Write-Host ""
    Write-Host "MSVC compiler (cl.exe) is not on PATH."
    Write-Host "Open 'x64 Native Tools Command Prompt for VS 2022' or 'Developer PowerShell for VS 2022',"
    Write-Host "then rerun:"
    Write-Host "  .\scripts\setup_implicit_gpu_windows.ps1"
    exit 1
}

& $venvPy -m pip install --upgrade pip setuptools wheel
& $venvPy -m pip install --upgrade numpy scipy cython
& $venvPy -m pip uninstall -y implicit

Write-Host ""
Write-Host "Building implicit from source (GPU attempt)..."
& $venvPy -m pip install --no-binary implicit implicit

Write-Host ""
Write-Host "Running GPU verification..."
& $venvPy scripts\verify_implicit_gpu.py

Write-Host ""
Write-Host "Done. If verification says CUDA extension missing, this Windows toolchain still didn't build GPU kernels."
