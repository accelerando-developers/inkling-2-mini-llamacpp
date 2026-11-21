#!/bin/bash
# Download Inkling-2-Mini UD-Q4_K_XL (17.63 GB) + mmproj-F16 vision projector (639 MB).
# Resume + retry, logs to <repo>/dl.log and dl.err.
#
# IMPORTANT: the files under Mainline_Support/ use the `inkling2_loop` layout that MAINLINE
# llama.cpp reads (https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/Mainline_Support/README.md).
# The GGUFs in the repo ROOT use unsloth's own looped layout and require unsloth's prebuilt
# llama.cpp (b12588-mix-c0c3de8) — do NOT mix them up.
# The `.gguf_file` extension is intentional; the files are renamed to `.gguf` after download.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models" || exit 1
cd "$ROOT/models" || exit 1
BASE="https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main"
LOG="$ROOT/dl.log"
ERR="$ROOT/dl.err"

declare -A FILES=(
  ["Inkling-2-Mini-UD-Q4_K_XL.gguf"]="$BASE/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file"
  ["mmproj-F16.gguf"]="$BASE/mmproj-F16.gguf"
)
# Optional extras (see docs/BENCHMARKS.md):
#   dspark-Inkling-2-Mini-Q8_0.gguf (0.4 GB speculative-decoding drafter — needs the unsloth build!)
#   Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file (reference quant, 22.87 GB)

for f in Inkling-2-Mini-UD-Q4_K_XL.gguf mmproj-F16.gguf; do
  echo "[$(date +%T)] START $f" >> "$LOG"
  curl -L --fail --retry 8 --retry-delay 5 -C - -o "$f" "${FILES[$f]}" 2>> "$ERR"
  rc=$?
  echo "[$(date +%T)] DONE rc=$rc $f size=$(stat -c%s "$f" 2>/dev/null)" >> "$LOG"
  if [ $rc -ne 0 ]; then
    echo "[$(date +%T)] RETRY-ONCE $f" >> "$LOG"
    curl -L --fail --retry 8 --retry-delay 10 -C - -o "$f" "${FILES[$f]}" 2>> "$ERR"
    echo "[$(date +%T)] DONE rc=$? $f size=$(stat -c%s "$f" 2>/dev/null)" >> "$LOG"
  fi
done
echo "[$(date +%T)] ALL-DOWNLOADS-FINISHED" >> "$LOG"
ls -lh "$ROOT/models" >> "$LOG" 2>&1
