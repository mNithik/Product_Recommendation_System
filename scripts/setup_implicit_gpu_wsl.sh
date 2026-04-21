#!/usr/bin/env bash
set -euo pipefail

echo "=== implicit GPU setup (WSL/Linux) ==="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python in WSL and rerun."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found in WSL. Install WSL NVIDIA driver support first."
  exit 1
fi

PYTHON_BIN="python3"
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
fi

echo "Using interpreter: ${PYTHON_BIN}"

"${PYTHON_BIN}" -m venv .venv-wsl-gpu
source .venv-wsl-gpu/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade numpy scipy cython
python -m pip install -r requirements.txt

# Force a prebuilt Linux wheel for implicit (GPU-capable path on supported Python versions).
# If no wheel exists (e.g., unsupported Python version), this fails fast.
python -m pip uninstall -y implicit || true
python -m pip install --only-binary=implicit --force-reinstall "implicit==0.7.2"

echo
echo "Running GPU verification..."
python scripts/verify_implicit_gpu.py

echo
echo "Setup complete."
echo "To run experiment:"
echo "  source .venv-wsl-gpu/bin/activate"
echo "  python main.py --config configs/gpu_profile.yaml --experiment wsl_gpu_run"
