#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Preserve caller-supplied environment variables so .env does not clobber them
ORIG_KEV_MODEL_ID="${KEV_MODEL_ID:-}"
ORIG_KEV_PORT="${KEV_PORT:-}"
ORIG_KEV_REPO_DIR="${KEV_REPO_DIR:-}"
ORIG_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

if [[ -n "$ORIG_KEV_MODEL_ID" ]]; then
    KEV_MODEL_ID="$ORIG_KEV_MODEL_ID"
fi
if [[ -n "$ORIG_KEV_PORT" ]]; then
    KEV_PORT="$ORIG_KEV_PORT"
fi
if [[ -n "$ORIG_KEV_REPO_DIR" ]]; then
    KEV_REPO_DIR="$ORIG_KEV_REPO_DIR"
fi
if [[ -n "$ORIG_CUDA_VISIBLE_DEVICES" ]]; then
    CUDA_VISIBLE_DEVICES="$ORIG_CUDA_VISIBLE_DEVICES"
fi

# Allow passing checkpoint directly as first argument: bash ./run_kev_server.sh [checkpoint_path]
if [[ $# -gt 0 && "$1" != -* ]]; then
    KEV_MODEL_ID="$1"
    shift
fi

# Resolve relative checkpoint path against PWD or SCRIPT_DIR before cd-ing into KEV_REPO_DIR
if [[ -n "${KEV_MODEL_ID:-}" && "$KEV_MODEL_ID" != /* ]]; then
    if [[ -e "$PWD/$KEV_MODEL_ID" ]]; then
        KEV_MODEL_ID="$(cd -- "$PWD" && pwd)/$KEV_MODEL_ID"
    elif [[ -e "$SCRIPT_DIR/$KEV_MODEL_ID" ]]; then
        KEV_MODEL_ID="$SCRIPT_DIR/$KEV_MODEL_ID"
    fi
fi

if [[ -n "${KEV_MODEL_ID:-}" && "$KEV_MODEL_ID" == /* ]]; then
    if [[ ! -d "$KEV_MODEL_ID" || ! -f "$KEV_MODEL_ID/head.pt" ]]; then
        printf 'Error: Kev checkpoint not found or missing head.pt: %s\n' "$KEV_MODEL_ID" >&2
        exit 1
    fi
fi

if [[ -z "${KEV_REPO_DIR:-}" ]]; then
    KEV_REPO_DIR="$SCRIPT_DIR/kev"
    for candidate in "$SCRIPT_DIR/kev/kev" "$SCRIPT_DIR/kev"; do
        if [[ -f "$candidate/pyproject.toml" && -f "$candidate/kev/serve.py" ]]; then
            KEV_REPO_DIR="$candidate"
            break
        fi
    done
fi
if [[ "$KEV_REPO_DIR" != /* ]]; then
    KEV_REPO_DIR="$SCRIPT_DIR/$KEV_REPO_DIR"
fi
if [[ ! -d "$KEV_REPO_DIR" ]]; then
    printf 'Kev repository not found: %s\n' "$KEV_REPO_DIR" >&2
    printf 'Clone jaredpalmer/kev there or set KEV_REPO_DIR in .env.\n' >&2
    exit 1
fi
if [[ ! -f "$KEV_REPO_DIR/pyproject.toml" || ! -f "$KEV_REPO_DIR/kev/serve.py" ]]; then
    printf 'Kev server module not found under project root: %s\n' "$KEV_REPO_DIR" >&2
    printf 'Set KEV_REPO_DIR to the checkout containing pyproject.toml and kev/serve.py.\n' >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export KEV_DTYPE="${KEV_DTYPE:-bf16}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

unset VIRTUAL_ENV
cd "$KEV_REPO_DIR"
exec uv run --extra serve python -m kev.serve \
    --run "${KEV_MODEL_ID:-jaredpalmer/kev-4b}" \
    --port "${KEV_PORT:-8008}" \
    "$@"