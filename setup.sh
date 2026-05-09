#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="cs348k"
PYTHON_VERSION="3.11"

echo "=== CS 348K — Kinematic-to-Dynamic Gap ==="

# Create conda env (skip if already exists)
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists, skipping creation."
else
    echo "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# Activate and install deps
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "Installing Python dependencies..."
# Install torch with CUDA (edit the cu121 suffix to match your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Activate the env with:  conda activate ${ENV_NAME}"
echo ""
echo "To generate MotionBricks clips (one-time):"
echo "  1. Clone and install MotionBricks:"
echo "       git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git"
echo "       cd GR00T-WholeBodyControl/motionbricks && git lfs pull && pip install -e . && cd -"
echo "  2. python generate_motions.py"
echo ""
echo "To run evaluation:"
echo "  python run_eval.py --data_dir data/synthetic    # quick sanity check"
echo "  python run_eval.py --data_dir data/motionbricks # after generation"
