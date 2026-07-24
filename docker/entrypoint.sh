#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace /data /outputs \
  "${TORCH_HOME:-$HOME/.cache/torch}" \
  "${HF_HOME:-$HOME/.cache/huggingface}" \
  "${MPLCONFIGDIR:-$HOME/.cache/matplotlib}"

if [[ -f /workspace/pyproject.toml || -f /workspace/setup.py ]]; then
  # Editable installation is intentionally opt-in through AUTO_INSTALL_PROJECT=1
  # to avoid reinstalling the package every time the container starts.
  if [[ "${AUTO_INSTALL_PROJECT:-0}" == "1" ]]; then
    python -m pip install -e /workspace
  fi
fi

exec "$@"
