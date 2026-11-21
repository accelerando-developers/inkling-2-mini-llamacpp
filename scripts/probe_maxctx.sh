#!/bin/bash
# Max-context scan for Inkling 2 Mini: throwaway server ladder per config.
# Per config: start → hard health wait (dead container / timeout → FAIL) →
#   ubatch fallback 512→256→128 (the MoE compute buffer scales with ctx!) →
#   one /completion probe (512-token prompt + 256 gen, temp 0) → pp/tg + VRAM + CPU-spill check.
# usage: bash scripts/probe_maxctx.sh             # full ladder (see below)
#        bash scripts/probe_maxctx.sh 131072 q8_0  # single config
# Env: GPUS="GPU-<uuid>,GPU-<uuid>" to pin specific GPUs (else all visible GPUs).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${TMPDIR:-/tmp}/inkling-maxctx"
PORT=18082
CORPUS="${TMPDIR:-/tmp}/inkling-corpus.txt"
mkdir -p "$WORK"
test -s "$CORPUS" || { echo "corpus missing — run scripts/depthscan.sh first (it generates $CORPUS)"; exit 1; }
head -c 2048 "$CORPUS" > "$WORK/probe-prompt.txt"

probe_one() { # probe_one <ctx> <kv> <label>
  local ctx=$1 kv=$2 label=$3
  local CN="llmtest-probe-$label"
  local RS="$WORK/$label.res"
  docker rm -f "$CN" >/dev/null 2>&1

  nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv -l 1 > "$WORK/vram-$label.csv" 2>/dev/null &
  local SPID=$!
  local SECS=0 ok=0 UBAT=512
  local GPUS_ENV=(); if [ -n "${GPUS:-}" ]; then GPUS_ENV=(-e CUDA_VISIBLE_DEVICES="$GPUS"); fi

  start_one() { # start_one <ubatch>
    docker run -d --name "$CN" --gpus all --ipc=host \
      "${GPUS_ENV[@]}" \
      -v "$ROOT/models:/models:ro" -p $PORT:$PORT \
      -e LLAMA_ARG_MODEL=/models/Inkling-2-Mini-UD-Q4_K_XL.gguf \
      -e LLAMA_ARG_CTX_SIZE=$ctx -e LLAMA_ARG_N_PARALLEL=1 \
      -e LLAMA_ARG_FLASH_ATTN=on \
      -e LLAMA_ARG_CACHE_TYPE_K=$kv -e LLAMA_ARG_CACHE_TYPE_V=$kv \
      -e LLAMA_ARG_UBATCH=$1 -e LLAMA_ARG_SPLIT_MODE=layer \
      -e LLAMA_ARG_THREADS="${THREADS:-12}" -e LLAMA_ARG_THREADS_BATCH="${THREADS:-12}" \
      -e LLAMA_ARG_HOST=0.0.0.0 -e LLAMA_ARG_PORT=$PORT \
      llamacpp-inkling2:server-cuda >/dev/null 2>&1
    SECS=0; ok=0
    until curl -sf http://localhost:$PORT/health >/dev/null 2>&1; do
      sleep 2; SECS=$((SECS+2))
      [ "$(docker inspect -f '{{.State.Running}}' "$CN" 2>/dev/null)" != "true" ] && break
      [ $SECS -ge 240 ] && break
    done
    curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && ok=1
  }

  start_one "$UBAT"
  if [ $ok -eq 0 ]; then
    for ubf in 256 128; do
      docker rm -f "$CN" >/dev/null 2>&1; UBAT=$ubf; start_one "$UBAT"
      [ $ok -eq 1 ] && break
    done
  fi

  local status="❌ FAIL(start/health)" pp="-" tg="-" spill="0"
  if [ $ok -eq 1 ]; then
    jq -n --rawfile p "$WORK/probe-prompt.txt" --argjson np 256 \
      '{prompt: $p, n_predict: $np, temperature: 0, cache_prompt: false}' > "$WORK/$label.req"
    local RC=0
    curl -s --max-time 900 http://localhost:$PORT/completion -H "Content-Type: application/json" -d @"$WORK/$label.req" > "$RS" || RC=$?
    if [ $RC -eq 0 ] && [ -s "$RS" ]; then
      read -r pp tg < <(python3 -c "import json,sys; t=json.load(open(sys.argv[1]))['timings']; print('%.0f %.1f' % (t['prompt_per_second'], t['predicted_per_second']))" "$RS")
    else
      status="❌ FAIL(completion rc=$RC)"; pp="-"; tg="-"
    fi
    spill=$(docker logs "$CN" 2>&1 | grep -c "assigned to CPU")
    local v0 v2
    v0=$(awk -F, '{gsub(/[^0-9]/,"",$3); if ($2==0 && $3+0>m) m=$3+0} END {print m+0}' "$WORK/vram-$label.csv")
    v2=$(awk -F, '{gsub(/[^0-9]/,"",$3); if ($2==2 && $3+0>m) m=$3+0} END {print m+0}' "$WORK/vram-$label.csv")
    status="✅ OK"
    if [ "$spill" -gt 0 ]; then status="⚠️ CPU-SPILL($spill)"; fi
    printf "ctx=%-7s kv=%-5s ub=%-3s %-18s pp=%-6s tg=%-6s VRAM=%s/%s MiB start=%ss\n" \
      "$ctx" "$kv" "$UBAT" "$status" "$pp" "$tg" "$v0" "$v2" "$SECS" | tee -a "$WORK/PROBE-RESULTS.txt"
  else
    printf "ctx=%-7s kv=%-5s ub<=128 %-18s start=%ss\n" "$ctx" "$kv" "$status" "$SECS" | tee -a "$WORK/PROBE-RESULTS.txt"
    docker logs --tail 8 "$CN" > "$WORK/$label.fail" 2>&1
  fi
  kill $SPID 2>/dev/null
  docker rm -f "$CN" >/dev/null 2>&1
}

if [ $# -ge 2 ]; then
  probe_one "$1" "$2" "x$1$2"
  echo "PROBE-DONE"
  exit 0
fi

echo "Note: all VRAM must be free — stop any running 8091 stack first."
echo "Results are also appended to $WORK/PROBE-RESULTS.txt"
: > "$WORK/PROBE-RESULTS.txt"

# --- q8_0 KV ladder (until the first hard FAIL) ---
for c in 65536 98304 131072 163840 196608; do
  before=$(grep -c 'FAIL(start' "$WORK/PROBE-RESULTS.txt" 2>/dev/null); before=${before:-0}
  probe_one $c q8_0 "d$c-q8"
  after=$(grep -c 'FAIL(start' "$WORK/PROBE-RESULTS.txt" 2>/dev/null); after=${after:-0}
  [ "$after" -gt "$before" ] && { echo "-> q8_0 ladder stops after FAIL at $c"; break; }
done

# --- q4_0 KV ladder for max context ---
for c in 262144 393216 524288 786432 1048576; do
  before=$(grep -c 'FAIL(start' "$WORK/PROBE-RESULTS.txt" 2>/dev/null); before=${before:-0}
  probe_one $c q4_0 "d$c-q4"
  after=$(grep -c 'FAIL(start' "$WORK/PROBE-RESULTS.txt" 2>/dev/null); after=${after:-0}
  [ "$after" -gt "$before" ] && { echo "-> q4_0 ladder stops after FAIL at $c"; break; }
done

echo; echo "===== Ladder result ====="
cat "$WORK/PROBE-RESULTS.txt"
echo "PROBE-DONE"
