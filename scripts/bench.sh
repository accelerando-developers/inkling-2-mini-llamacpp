#!/bin/bash
# llama-bench wrapper — ⚠️ WARNING (2026-11-21): llama-bench (and any loader variant with an
# explicit `-ts`) tries a MONOLITHIC allocation of the entire model buffer on CUDA0 (~15.9 GiB)
# for `inkling2_loop` on multi-GPU → OOM on 16 GB GPUs.
# Only the llama-server with AUTO-FIT (do NOT set TENSOR_SPLIT!) distributes the layer-split
# correctly across both GPUs. → use scripts/depthscan.sh instead of llama-bench.
# This wrapper stays useful for single-GPU runs or after #32704 merges and the loader is fixed.
# usage: bash scripts/bench.sh <name> <llama-bench args...>
# Env: GPUS="GPU-<uuid>,GPU-<uuid>" to pin specific GPUs.
set -u
NAME="$1"; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RES_DIR="$ROOT/results"
mkdir -p "$RES_DIR"
GPUS_ENV=(); if [ -n "${GPUS:-}" ]; then GPUS_ENV=(-e CUDA_VISIBLE_DEVICES="$GPUS"); fi

# VRAM sampler (1 Hz, all GPUs)
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv -l 1 > "$RES_DIR/vram-$NAME.csv" 2>/dev/null &
SAMP_PID=$!

docker run --rm --gpus all --ipc=host \
  "${GPUS_ENV[@]}" \
  -v "$ROOT/models:/models:ro" \
  --entrypoint /app/llama-bench \
  llamacpp-inkling2:server-cuda "$@" \
  > "$RES_DIR/bench-$NAME.md" 2> "$RES_DIR/bench-$NAME.err"
RC=$?

kill $SAMP_PID 2>/dev/null
echo "=== $NAME (rc=$RC) ==="
grep -v '^$' "$RES_DIR/bench-$NAME.md" | tail -n +2
if [ -s "$RES_DIR/bench-$NAME.err" ]; then
  echo "--- stderr (tail) ---"
  tail -n 15 "$RES_DIR/bench-$NAME.err"
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | awk '{print "  VRAM "$0}'
