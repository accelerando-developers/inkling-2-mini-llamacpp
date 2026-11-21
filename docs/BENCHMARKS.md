# Benchmark Report: Inkling 2 Mini (UD-Q4_K_XL) on llama.cpp — 2× RTX 5060 Ti

## TL;DR — the ideal configuration

| Goal | Config |
|---|---|
| **Standard (compose default)** | ctx **131072**, KV q8_0/q8_0, `-fa on`, **ub 256**, layer-split + auto-fit, np 1 → tg ~22–25 t/s, pp up to 475 |
| **Max context at full speed** | ctx **262144** + KV q4_0/q4_0 + **ub 128** → tg 24.7 @ 256k (!), VRAM 13.3/14.6 GiB |
| Absolute VRAM maximum | ctx 393216 @ q4_0/ub128 → tg 17.6 (tight split); 524288 = OOM |
| **4 parallel users** | np 4 → 4 slots × 32.7k; **aggregate PP 485 tok/s** (64k combined), tg fairly split 4× ~9 t/s |
| **Key insight** | PP *gains* from parallelism (+15–23 %); aggregate tg stays flat (~22) |

Don't forget: **ub 512 only works up to ~64k ctx** (the MoE compute buffer scales with ctx) and
**setting `N_GPU_LAYERS`/`TENSOR_SPLIT` = auto-fit abort = OOM** (see Pitfalls).

| Standard-context headline numbers | Value |
|---|---|
| pp, 2.6k → 128k depth | 475 → 190 t/s (−60 %) |
| tg, 2.6k → 128k depth | 22.5 → 17.2 t/s |
| VRAM @131k (comfortable split) | 13.3 / 14.6 GiB |

All raw data: [`results/`](../results) (`depthscan-*.csv`, `vram-*.csv`, `PROBE-RESULTS.txt`).

---

**Test date:** Nov 21, 2026 · **Host:** AMD Ryzen 5 7500F (6C/12T) · **GPU:** 2× NVIDIA RTX 5060 Ti 16 GB (sm_120, layer-split + auto-fit), desktop on an RTX 3060 12 GB (excluded)
**Model:** [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf` — 17.63 GB (Apache-2.0; loop-MoE 26.2B/5.4B active, 48 layer applications, hybrid local+global attention, 1M native, multimodal)
**Image:** `llamacpp-inkling2:server-cuda` (locally built) = llama.cpp @ `2a28df0f9bd5` (inkling2_loop #32694 + fix #32704) · CUDA 12.8.1, sm_120
**Background:** [unsloth run guide](https://unsloth.ai/docs/models/inkling-2-mini) · [PR #32704](https://github.com/ggml-org/llama.cpp/pull/32704) · [Open LLM Radar: Inkling 2 Mini](https://open-llm-radar.com/models/inkling-2-mini)

## The three commissioned tests (this report)

1. **Maximum context size** — server ladder until OOM (`probe_maxctx.sh`): ctx × KV type (q8_0 / q4_0), VRAM, tg
2. **pp / tg** — server-based depth ladder d2.6k…d128k (`depthscan.sh`)
3. **Parallelism up to 64k** — 4×2.6k generation fairness, 4×15.6k & 2×31k (= 64k combined), 62k single reference (`paralleltest.sh`)

## Base ladder (fresh server requests, layer-split + auto-fit, fa on, KV q8_0/q8_0, t12, ub 256, ctx 131072)

llama-bench cannot run this architecture on multi-GPU (monolithic CUDA0 allocation — see `scripts/bench.sh`),
so the ladder is **server-based** (128 generated tokens, temp 0, `cache_prompt: false`; the 128k row is a
single 125,439-token prompt request).

| Prompt depth | pp (t/s) | tg (t/s) | Note |
|---:|---:|---:|---|
| 2.6k | 475 | 22.5 | short prompt |
| 8k | 459 | 23.0 | |
| 16k | 419 | 22.0 | |
| 32k | 360 | 23.0 | |
| 64k | 280 | 20.2 | |
| 96k | 225 | 19.7 | |
| 128k | 190 | 17.2 | 125,439 tokens in one request, full KV |

- **pp curve:** −15 … −18 % per doubling of depth (190 t/s at ~2.5× the d0 figure — loop ×3 + NVFP4 expert
  kernels make 26B / 48 layer-applications expensive; compare Qwen3.8-27B dense ~1800 pp, Maxwell 35B/3B-A ~3400 pp)
- **tg curve:** flat ≈ 22–23 t/s up to 64k, −14 % @ 96k (19.7), −26 % @ 128k (17.2)
- **VRAM @131k:** 13.3 / 14.6 GiB (auto-fit distributes unevenly, ~55/45)
- **ub 512 does NOT fit at ctx 131072**: the MoE compute buffer scales with ctx (init OOM:
  `allocating 20157 MiB … compute pp buffers` at ub512); ub256/ub128 work. From ~64k ctx, ub512 is fine again.
- A cache demo after the d128k run was evicted (the slot had last served 125k) → no clean cache number;
  prompt caching is functional (see the qwen38 project's §6 for a clean measurement).

## Max context (server ladder, `probe_maxctx.sh`, ubatch fallback 512→256→128)

**Watch the auto-fit split: a "comfortable" distribution (~13.3/14.6 GiB, one GPU carries fewer weights)
→ tg ~25 t/s full speed; a "tight" distribution (14.7–14.9+ GiB on both, VMM borderline)
→ tg ~17 (VMM spill without any visible CPU warning).** Probe = 512-token prompt + 256 generated, temp 0, single request.

| ctx | KV | ub | VRAM G0/G2 (GiB) | pp | tg | Status |
|---:|---|---:|---|---:|---:|---|
| 65536 | q8_0 | 512 | 13.3 / 14.9 | 558 | 26.0 | ✅ comfortable |
| 98304 | q8_0 | 512 | 14.8 / 14.7 | 569 | 17.3 | ⚠️ tight |
| **131072** | q8_0 | 256 | 13.3 / 14.6 | 453 | **24.7** | ✅ **comfortable = compose default** |
| 163840 | q8_0 | 256 | 14.8 / 14.6 | 562 | 18.5 | ⚠️ tight |
| 196608 | q8_0 | 256 | 14.9 / 14.9 | 424 | 16.7 | ⚠️ at the edge |
| **262144** | q4_0 | 128 | 13.3 / 14.6 | 314 | **24.7** | ✅ **256k at full tg speed!** |
| 393216 | q4_0 | 128 | 14.8 / 14.8 | 306 | 17.6 | ✅ runs, tight |
| 524288 | q4_0 | ≤128 | — | — | — | ❌ OOM (no buffer fits) |

**Max-context conclusions:**
- **131072 @ q8_0/ub256** = standard with full ~25 t/s (comfortable split).
- **Max at full speed: 262144 @ q4_0 KV + ub128** — exactly the comfortable basis of 131k/q8_0 (the same
  ~27.9 GiB total): 256k is the sweet spot for long contexts (natively the model allows up to 1M; VRAM
  ends here).
- **Absolute VRAM maximum: 393216 @ q4_0/ub128** (tg 17.6, tight split — realistically ~17 t/s; 524288 does not run).
- The pp values in this table are 512-token probes (not comparable to the depth ladder).
- Not a single config produced the explicit `assigned to CPU` warning — the tg degradation on tight
  splits comes from VMM overflow (shared-memory offload), visible in the logs only through the VRAM counters.

## Parallel test up to 64k (`paralleltest.sh`, ctx 131072, **np 4**, ub 256, layer-split)

Important preliminary finding: **explicit `N_PARALLEL=4` disables the unified KV pool** → 4 slots ×
**32768** each (ctx/4, deterministic). Prompts >32.7k per slot → `exceed_context_size_error`.
The 64k single reference therefore comes from the depth ladder (np=1, full slot).

| Scenario | Per request | Aggregate | Note |
|---|---|---|---|
| (1) 4× ~2.6k tok + 256 gen | pp 177–325 · **tg 8.2–9.2 t/s** | **22.5 tok/s** (1024 tok / 45.5 s) | 4-way decode, fair split |
| (2) 4× ~14.5k tok = 64k combined + 64 gen | pp 334–479 · tg 0.8–9.7 t/s* | **485 tok/s PP** (57,960 tok / 119.4 s) | sequential equivalent 147 s → **parallel +23 %** |
| (3) 2× ~28.9k tok = 64k combined + 64 gen | pp 394–406 · tg 13.7 / 1.0 t/s* | **425 tok/s PP** (57,728 tok / 135.7 s) | vs single-stream d32k (360) → **+18 %** |
| (4) 1× ~62k tok (single reference, np1) | **pp 280 · tg 20.2** (depth ladder d64k + 128 gen) | — | the np4 slot limit forbids 62k in 4-slot mode |
| (5) `/slots` | 4 slots × 32768, idle after requests (prompt_tokens=0) | — | per-slot KV confirmed |

\* The per-request tg numbers in (2)/(3) are contaminated: the decode-phase timing runs while other
slots are still prefilling (time-slicing). The clean parallel-decode number is scenario (1).

**Parallel conclusions (up to 64k combined):**
- **PP gains from parallelism** on this model: aggregate 425–485 tok/s vs. single-stream 280–420 t/s
  depending on depth (+15–23 %). Explanation: batched prefill across slots + MoE activation — more
  tokens per active-weight pass; unlike the 27B dense baseline (no PP gain from parallelism there).
- **Aggregate tg stays flat**: 4 streams × ~9 t/s ≈ 22.5 tok/s ≈ single stream (22–23). Decode here is
  KV/loop-bound — multi-user gets a fair time-slice, no throughput boost, no collapse.
- **Slot limitation**: np4 → each user max ~32.7k context; "64k combined" is the real 4-user case.
  Single user with >64k: np1 (full 131k/256k slot, see max-context section).
- No slow-slot phenomenon in 4-way MoE decode (unlike the Maxwell 2×55k case).

## Pitfalls (learned the hard way)

1. **`-sm tensor` fails in the loader**: `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture
   'inkling2_loop'`. Layer-split is the only multi-GPU path (unlike Maxwell/Qwen-NVFP4).
2. **Auto-fit vs. TENSOR_SPLIT**: with `LLAMA_ARG_TENSOR_SPLIT` set, `common_fit_params` aborts
   (`tensor_split already set by user`) and the loader tries to allocate the 16 GB model **monolithically
   on CUDA0** → OOM. Without tensor_split, auto-fit distributes correctly (~55/45 by free VRAM).
3. **llama-bench has the same problem**: monolithic CUDA0 allocation even with `-ts 1,1` → unusable for
   this architecture on multi-GPU (`scripts/bench.sh` carries the warning); all numbers here were
   measured server-side.
4. **The MoE compute buffer scales with ctx** (not just ubatch): ub512 + ctx 131072 → init OOM
   (`allocating 20157 MiB … compute pp buffers`); ub256 fits at 131k, ub128 is required for 262k q4_0.
   From ~64k ctx, ub512 works.
5. **The auto-fit distribution decides your tg**: comfortable split (~13.3/14.6 GiB) → full tg ~25;
   tight split (both ≥14.8) → tg ~17 via VMM overflow (no CPU warning in the log! the VRAM counters are
   the indicator). **Additional auto-fit abort trigger besides TENSOR_SPLIT: `LLAMA_ARG_N_GPU_LAYERS`
   ("all" = -2)** → deliberately NOT set in the compose file.
6. **np>1 = per-slot KV**: ctx/np per slot (131072/np4 → 4×32768); a 62k request is rejected at np4 —
   use np1 for large contexts.
7. Corpus generator bug (`len(list)` instead of chars → 187 MB) — fixed; corpus is 720 KB ≈ 181k tokens.

## Reproduction

```bash
bash scripts/download-models.sh              # UD-Q4_K_XL (Mainline_Support!) + mmproj  (~18.3 GB)
docker build -f docker/Dockerfile -t llamacpp-inkling2:server-cuda docker/  # llama.cpp @ 2a28df0f
UB=256 bash scripts/depthscan.sh             # pp/tg ladder (server-based, ub256)
bash scripts/probe_maxctx.sh                 # max-context ladder q8_0 & q4_0 (ub fallback 512→256→128)
bash scripts/paralleltest.sh                 # parallel up to 64k combined (np4, ctx 131072)
docker compose -f docker/docker-compose.yml up -d   # stack, port 8091 (default: ctx 131k, ub256, np1)
```

End-to-end verified (Nov 21, compose stack): `/health` healthy after 8 s · `/v1/models` → `inkling-2-mini` ·
chat at temp 1.0/top-p 1.0 with embedded reasoning (reasoning_content + a correct content answer) ·
tg 22.4 t/s in chat mode · `/metrics` Prometheus live · VRAM 13.3/14.5 GiB.

Raw data: `results/depthscan-ub256.csv`, `results/vram-depthscan-*.csv`, `results/PROBE-RESULTS.txt`,
`results/llama-bench-multi-gpu-bug.*` (the failed llama-bench attempts kept as bug evidence).
