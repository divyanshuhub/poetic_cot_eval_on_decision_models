#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

if ! command -v docker >/dev/null 2>&1; then
    printf 'Docker is required to launch the local Clef SGLang server.\n' >&2
    exit 1
fi

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_ID="${CLEF_MODEL_ID:-Cloudflare/clef-flash}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

exec docker run --rm \
    --gpus "device=$GPU_ID" \
    --shm-size 32g \
    --ipc=host \
    -p 30000:30000 \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -e CUDA_VISIBLE_DEVICES=0 \
    lmsysorg/sglang:dev-clef \
    sglang serve \
        --model-path "$MODEL_ID" \
        --host 0.0.0.0 \
        --port 30000