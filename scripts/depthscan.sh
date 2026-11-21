#!/bin/bash
# Depth scan: pp/tg per prompt depth — server-based, because llama-bench cannot run
# `inkling2_loop` on multi-GPU (it attempts a monolithic CUDA0 allocation → OOM).
# Each depth = one fresh request (cache_prompt: false), 128 generated tokens, temp 0.
# usage: [UB=256] [CTX=131072] bash scripts/depthscan.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT=18080
UB="${UB:-256}"
CTX="${CTX:-131072}"
RES_DIR="$ROOT/results"
WORK="${TMPDIR:-/tmp}/inkling-depth"
mkdir -p "$WORK" "$RES_DIR"

CORPUS="${TMPDIR:-/tmp}/inkling-corpus.txt"
GPUS_ENV=()
if [ -n "${GPUS:-}" ]; then GPUS_ENV=(-e CUDA_VISIBLE_DEVICES="$GPUS"); fi

# ---------- corpus (~720 KB ≈ 181k tokens), deterministic and varied ----------
if [ ! -s "$CORPUS" ]; then
  python3 - "$CORPUS" <<'PYEOF'
import sys
subjects=["the replication of digital artifacts","communist building-material economists","an unusual cartography of wind tunnels","quantized compote theories","the migration of cranes","vertical gardens in Kiel","the fault tolerance of ravioli recipes","astronomical bookbinderies","the tactics of water-polo managers","holographic stamps","decentralized bee corporations","the ethics of robotic lawnmowers","cold fusion of homemade jelly","the statistics of unplayed banjos","cryptic frescoes in the subway tunnel"]
verbs=["documents","undermines","anticipates","refactors","reflects","orchestrates","paraphrases","calibrates","synthesizes","domesticates"]
objects=["the longevity of gummy bears","a precise thermodynamics of quitting time","the cascades of paper boats","the rhythm of delayed trains","a library of singing bowls","the geometry of chessboard rooftops","the market for used fog machines","the legend of the nineteen forklifts","a chronology of forgotten holidays","the echo of old kettles"]
out=[]; text=""; i=0
while len(text) < 720000:
    i+=1
    s=subjects[i%len(subjects)]; v=verbs[(i//3)%len(verbs)]; o=objects[(i//7)%len(objects)]
    chunk=(f"Section {i} {v} {s} and, in passing, {o}. The protocols show that "
           f"sample {i} was accompanied over two winters without losing the "
           f"original state of the measuring station. ")
    out.append(chunk); text = text + chunk
open(sys.argv[1],"w").write(text)
PYEOF
fi

docker rm -f llmtest-depthscan >/dev/null 2>&1
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv -l 1 > "$RES_DIR/vram-depthscan-ub$UB.csv" 2>/dev/null &
SAMP_PID=$!

docker run -d --name llmtest-depthscan --gpus all --ipc=host \
  "${GPUS_ENV[@]}" \
  -v "$ROOT/models:/models:ro" -p $PORT:$PORT \
  -e LLAMA_ARG_MODEL=/models/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  -e LLAMA_ARG_CTX_SIZE=$CTX -e LLAMA_ARG_N_PARALLEL=1 \
  -e LLAMA_ARG_FLASH_ATTN=on \
  -e LLAMA_ARG_CACHE_TYPE_K=q8_0 -e LLAMA_ARG_CACHE_TYPE_V=q8_0 \
  -e LLAMA_ARG_UBATCH=$UB \
  -e LLAMA_ARG_SPLIT_MODE=layer \
  -e LLAMA_ARG_THREADS="${THREADS:-12}" -e LLAMA_ARG_THREADS_BATCH="${THREADS:-12}" \
  -e LLAMA_ARG_HOST=0.0.0.0 -e LLAMA_ARG_PORT=$PORT \
  llamacpp-inkling2:server-cuda >/dev/null

SECS=0; until curl -sf http://localhost:$PORT/health >/dev/null 2>&1; do sleep 2; SECS=$((SECS+2)); [ $SECS -ge 900 ] && { echo HEALTH-TIMEOUT; docker logs --tail 40 llmtest-depthscan; exit 1; }; done
echo "server up after ${SECS}s (ctx $CTX, ub $UB)"

CSV="$RES_DIR/depthscan-ub$UB.csv"
echo "label,bytes,prompt_n,pp_tps,tg_tps,predicted_n" > "$CSV"

gen() { # gen <bytes> <label>
  local bytes=$1 label=$2
  jq -n --rawfile p <(head -c $bytes "$CORPUS") --arg s "Question: What topics are discussed in the text above? Respond with a concise two-sentence answer." \
    '{prompt: ($p + $s), n_predict: 128, temperature: 0, cache_prompt: false}' > "$WORK/req-$label.json"
  curl -s --max-time 900 http://localhost:$PORT/completion -H "Content-Type: application/json" -d @"$WORK/req-$label.json" \
    > "$WORK/res-$label.json"
  python3 - "$WORK/res-$label.json" "$bytes" "$label" "$CSV" <<'PYEOF'
import json, sys, csv
r = json.load(open(sys.argv[1])); t = r["timings"]
bytes_, label, csvp = int(sys.argv[2]), sys.argv[3], sys.argv[4]
print(f"  {label:>6} ({bytes_:>6} B): prompt_n={t['prompt_n']:>6} pp={t['prompt_per_second']:>7.0f} t/s | tg={t['predicted_per_second']:>6.1f} t/s (n={t['predicted_n']})")
with open(csvp, "a", newline="") as f:
    csv.writer(f).writerow([label, bytes_, t["prompt_n"], f"{t['prompt_per_second']:.0f}", f"{t['predicted_per_second']:.1f}", t["predicted_n"]])
PYEOF
}

echo "=== Depth scan (ub $UB, KV q8_0, fresh) ==="
echo "--- ~2.6k tok (10.5KB)";  gen 10500 d2p6k
echo "--- ~8k tok (33KB)";       gen 33000 d8k
echo "--- ~16k tok (66KB)";      gen 66000 d16k
echo "--- ~32k tok (133KB)";     gen 133000 d32k
echo "--- ~64k tok (266KB)";     gen 266000 d64k
echo "--- ~96k tok (407KB)";     gen 407000 d96k
echo "--- ~128k tok (537KB)";    gen 537000 d128k

kill $SAMP_PID 2>/dev/null
docker rm -f llmtest-depthscan >/dev/null 2>&1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | awk '{print "  VRAM "$0}'
echo DEPTHSCAN-DONE
