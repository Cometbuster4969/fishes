#!/usr/bin/env bash
# Phase 1: one-command environment setup for GeoFish-AI.
#
# Notes on this sandbox (recorded in outputs/DECISIONS.md):
#   * CPU only (2 vCPU, ~3 GB RAM), no GPU.
#   * PyPI is the only index reachable; download.pytorch.org is blocked, so the
#     PyPI `torch` wheel (which is the CUDA build) is installed, together with
#     its nvidia-* runtime dependencies. Training/inference run on CPU.
#   * The venv lives at ../.venv (excluded from git and from workspace snapshots),
#     so this script is idempotent and safe to re-run in a fresh sandbox.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "[setup] creating venv at $VENV"
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" -q install --upgrade pip wheel
echo "[setup] installing requirements.txt"
"$VENV/bin/pip" install -r "$REPO_ROOT/requirements.txt"
echo "[setup] done: $("$VENV/bin/python" -c 'import torch,sys;print("torch",torch.__version__,"cuda",torch.cuda.is_available(),"py",sys.version.split()[0])')"
