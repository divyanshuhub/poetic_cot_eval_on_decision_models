#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

PYTHON_BIN="${ARC_DECISION_PYTHON:-$WORKSPACE_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
    printf 'Set ARC_DECISION_PYTHON to this project\x27s Python executable.\n' >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LAYA_DEVICE="${LAYA_DEVICE:-cuda}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" "$@"