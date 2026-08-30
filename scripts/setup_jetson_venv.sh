#!/usr/bin/env bash
# Idempotent post-sync setup that exposes system-installed JetPack packages
# (TensorRT, pycuda) to the uv-managed venv. Re-run after every
# `uv sync --reinstall` or after the venv is wiped.
#
# Usage:
#     bash scripts/setup_jetson_venv.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PTH="$PROJECT_ROOT/.venv/lib/python3.10/site-packages/_jetpack_system.pth"

if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
    echo "error: $PROJECT_ROOT/.venv does not exist. Run 'uv sync --extra dev' first." >&2
    exit 1
fi

cat > "$PTH" <<'EOF'
/usr/lib/python3.10/dist-packages
/usr/local/lib/python3.10/dist-packages
EOF

echo "wrote $PTH"

# Smoke test.
uv run python - <<'PY'
import tensorrt as trt
print("tensorrt", trt.__version__)
import pycuda.driver
pycuda.driver.init()
print("pycuda device 0:", pycuda.driver.Device(0).name())
PY
