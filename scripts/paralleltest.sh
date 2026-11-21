#!/bin/bash
# Parallel test up to 64k combined context — real concurrent curls (pattern from the qwen38 project).
# Scenarios:
#   (1) 4× ~2.6k tok + 256 gen each  → 4-way decode fairness / aggregate
#   (2) 4× ~15.6k tok = 64k combined (PP) + 64 gen each
#   (3) 2× ~31k   tok = 64k combined (PP) + 64 gen each
#   (4) /slots check
# Note: np>1 = per-slot KV → each slot gets ctx/np (131072/4 = 32k per slot).
# usage: [NP=4] [UB=256] bash scripts/paralleltest.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT=18081
WORK="${TMPDIR:-/tmp}/inkling-p2"
NP="${NP:-4}"
UB="${UB:-256}"
CORPUS="${TMPDIR:-/tmp}/inkling-corpus.txt"
mkdir -p "$WORK"
test -s "$CORPUS" || { echo "corpus missing — run scripts/depthscan.sh first (it generates $CORPUS)"; exit 1; }

GPUS_ENV=(); if [ -n "${GPUS:-}" ]; then GPUS_ENV=(-e CUDA_VISIBLE_DEVICES="$GPUS"); fi

docker rm -f llmtest-parallel >/dev/null 2>&1
docker run -d --name llmtest-parallel --gpus all --ipc=host \
  "${GPUS_ENV[@]}" \
  -v "$ROOT/models:/models:ro" -p $PORT:$PORT \
  -e LLAMA_ARG_MODEL=/models/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  -e LLAMA_ARG_CTX_SIZE=131072 \
  -e LLAMA_ARG_N_PARALLEL=$NP \
  -e LLAMA_ARG_FLASH_ATTN=on \
  -e LLAMA_ARG_CACHE_TYPE_K=q8_0 -e LLAMA_ARG_CACHE_TYPE_V=q8_0 \
  -e LLAMA_ARG_UBATCH=$UB \
  -e LLAMA_ARG_SPLIT_MODE=layer \
  -e LLAMA_ARG_THREADS="${THREADS:-12}" -e LLAMA_ARG_THREADS_BATCH="${THREADS:-12}" \
  -e LLAMA_ARG_HOST=0.0.0.0 -e LLAMA_ARG_PORT=$PORT \
  llamacpp-inkling2:server-cuda >"$WORK/container.err" 2>&1 || { echo DOCKER-RUN-FAILED; cat "$WORK/container.err"; exit 1; }

SECS=0; until curl -sf http://localhost:$PORT/health >/dev/null 2>&1; do sleep 2; SECS=$((SECS+2)); [ $SECS -ge 900 ] && { echo HEALTH-TIMEOUT; docker logs --tail 40 llmtest-parallel; docker rm -f llmtest-parallel; exit 1; }; done
echo "server up after ${SECS}s"

# ---------- request files (~4 chars ≈ 1 token) ----------
for i in 0 1 2 3; do
  jq -n --rawfile p <(head -c 10500 "$CORPUS") --arg s "Question: What topics are discussed in the text above? Respond with a concise two-sentence answer." --argjson np 256 \
     '{prompt: ($p + $s), n_predict: $np, temperature: 0, cache_prompt: false}' > "$WORK/g$i.req"
  jq -n --rawfile p <(head -c 62500 "$CORPUS") --arg s "Question: What topics are discussed in the text above? Respond with a concise two-sentence answer." --argjson np 64 \
     '{prompt: ($p + $s), n_predict: $np, temperature: 0, cache_prompt: false}' > "$WORK/p$i.req"
  jq -n --rawfile p <(head -c 125000 "$CORPUS") --arg s "Question: What topics are discussed in the text above? Respond with a concise two-sentence answer." --argjson np 64 \
     '{prompt: ($p + $s), n_predict: $np, temperature: 0, cache_prompt: false}' > "$WORK/h$i.req"
done

send_batch() { # send_batch <suffix> <n>
  local suf=$1 n=$2 start end
  start=$(date +%s%N)
  for ((i=0;i<n;i++)); do
    curl -s --max-time 2400 http://localhost:$PORT/completion -H "Content-Type: application/json" \
      -d @"$WORK/$suf$i.req" > "$WORK/$suf$i.res" &
  done
  wait
  end=$(date +%s%N)
  echo $(( (end-start)/1000000 ))
}

pp_tg_report() { # pp_tg_report <suffix> <n>
  python3 - "$WORK" "$1" "$2" <<'PYEOF'
import json, sys, os
work, suf, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
toks, gens, wpp, tgs = 0, 0, 0.0, []
for i in range(n):
    try:
        t = json.load(open(f"{work}/{suf}{i}.res"))["timings"]
    except Exception as e:
        print(f"  req{i}: ERROR ({e})")
        continue
    print(f"  req{i}: prompt_n={t['prompt_n']:7} | pp={t['prompt_per_second']:7.0f} t/s | tg={t['predicted_per_second']:6.1f} t/s ({t['predicted_n']} tok)")
    toks += t["prompt_n"]; gens += t["predicted_n"]; wpp += t["prompt_per_second"] * t["prompt_n"]; tgs.append(t["predicted_per_second"])
mean_tg = sum(tgs)/max(len(tgs),1)
print(f"  Σ prompt={toks} tok | pp-weighted={0 if toks==0 else wpp/toks:7.0f} t/s | mean tg={mean_tg:6.1f} t/s | Σ gen={gens} tok")
PYEOF
}

echo "=== (1) 4 parallel generation requests (~2.6k tok + up to 256 gen each) ==="
W=$(send_batch g 4); pp_tg_report g 4
python3 - "$WORK" "$W" <<'PYEOF'
import json, sys
work, W = sys.argv[1], int(sys.argv[2])/1000
tot = sum(json.load(open(f"{work}/g{i}.res"))["timings"]["predicted_n"] for i in range(4))
print(f"  wall: {W:.1f}s | aggregate: {tot/W:.1f} tok/s | {tot} tokens total")
PYEOF

echo; echo "=== (2) 4 parallel PP (~14.5k tok each = 64k combined) + 64 gen ==="
W=$(send_batch p 4); pp_tg_report p 4
python3 - "$WORK" "$W" <<'PYEOF'
import json, sys
work, W = sys.argv[1], int(sys.argv[2])/1000
pt = sum(json.load(open(f"{work}/p{i}.res"))["timings"]["prompt_n"] for i in range(4))
print(f"  wall: {W:.1f}s | aggregate PP: {pt/W:.0f} tok/s ({pt} tokens)")
PYEOF

echo; echo "=== (3) 2 parallel PP (~28.9k tok each = 64k combined) + 64 gen ==="
W=$(send_batch h 2); pp_tg_report h 2
python3 - "$WORK" "$W" <<'PYEOF'
import json, sys
work, W = sys.argv[1], int(sys.argv[2])/1000
pt = sum(json.load(open(f"{work}/h{i}.res"))["timings"]["prompt_n"] for i in range(2))
print(f"  wall: {W:.1f}s | aggregate PP: {pt/W:.0f} tok/s ({pt} tokens total)")
PYEOF

echo; echo "=== (4) /slots ==="
curl -s http://localhost:$PORT/slots | python3 -c '
import json,sys
for s in json.load(sys.stdin)[:8]:
    print("  slot {}: prompt_tokens={}".format(s["id"], len(s.get("prompt",{}).get("tokens") or [])))'

docker rm -f llmtest-parallel >/dev/null 2>&1
echo "PARALLEL-DONE"
