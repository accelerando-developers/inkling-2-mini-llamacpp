# Inkling 2 Mini on llama.cpp — Docker Stack, Benchmarks & Context Scaling on 2× RTX 5060 Ti

Standalone, reproducible setup for running [Thinking Machines Lab](https://thinkingmachines.ai)'s **Inkling 2 Mini**
(unsloth `UD-Q4_K_XL` GGUF) on **mainline llama.cpp** in Docker on a dual-RTX-5060-Ti rig — including
max-context ladder, pp/tg depth scans, parallel-request tests up to 64k combined context, and an
**interactive animated data-flow visualization** comparing three architectures.

**🌐 Live demo:** <https://accelerando-developers.github.io/inkling-2-mini-llamacpp/>

> Tested 2026-11-21 · llama.cpp @ `2a28df0f` (inkling2_loop layout **#32694** + scale-reshare fix **#32704**)

## The three models (all tracked on [Open LLM Radar](https://open-llm-radar.com))

| Model | Arch | Active / Token | This repo |
|---|---|---|---|
| [**Inkling 2 Mini**](https://open-llm-radar.com/models/inkling-2-mini) · [HF: thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [GGUF: unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) | Loop-MoE 26.2B, 24 layers, middle 12 run **3×** → 48 layer applications | 5.4B effective (3 loop passes) | main subject — full stack & benchmarks |
| [**Qwen3.8-27B**](https://open-llm-radar.com/models/qwen38-27b) · [HF: Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) · [GGUF: unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | dense 27B, 64 layers | 27B (everything read) | comparison baseline (numbers from sibling project `qwen38/`) |
| [**Maxwell-V1-Flash**](https://open-llm-radar.com/models/maxwell-v10-flash) · [HF: maxwell-model/Maxwell-V1-Flash](https://huggingface.co/maxwell-model/Maxwell-V1-Flash) · [GGUF: AtomicChat/Maxwell-V1-Flash-GGUF](https://huggingface.co/AtomicChat/Maxwell-V1-Flash-GGUF) | MoE 35B / 3B active, 45 applications, 12+1 of 256 experts awake | 3B | comparison baseline (sibling project `maxwell/`) |

## TL;DR (Inkling 2 Mini, UD-Q4_K_XL, 2× RTX 5060 Ti, layer-split, KV q8_0, ub 256)

| Metric | Result |
|---|---|
| pp (prompt processing) | **475 → 190 t/s** (2.6k → 128k depth) |
| tg (token generation) | **22.5 → 17.2 t/s** (flat ~22–23 up to 64k) |
| Max context @ full speed | **262,144** (q4_0 KV + ub 128) — tg 24.7 t/s |
| Absolute VRAM max | **393,216** (q4_0, ub 128, tight split — tg ~17.6); 524,288 = OOM |
| 4 parallel users | 4 slots × 32k; **aggregate PP 485 tok/s** (+23 % vs sequential); tg fairly split 4× ~9 t/s |
| Compose default | ctx 131072 · KV q8_0 · fa on · **ub 256** · layer-split + auto-fit · np 1 |

**Why is a "small" 26B MoE as slow as a 27B dense model?** — measured, not guessed:

1. **The loop is not free.** The middle 12 layers run **three times** with shared weights → 48 layer
   applications per token. VRAM is saved (weights shared), but compute happens 3× — and passes are
   strictly sequential (pass 2 needs pass 1's output).
2. **NVFP4 expert weights = young kernel path.** Qwen/Maxwell use Q4_K/Q5_K with years of MMQ
   optimization. Inkling's experts are NVFP4 with per-expert scales (exactly why fix #32704 exists) —
   the mainline CUDA kernels are new and visibly slower. pp 190–475 t/s vs. 1096–3400 for the baselines.
3. **Routing overhead × 48.** Top-8-of-128 expert selection + scatter/gather + 8 small GEMVs per
   application, 48× per token at decode batch=1.
4. **The tg curve proves it.** tg stays flat (22–23) until 64k and only then drops — a KV-bound model
   (like the dense baseline) would degrade linearly with depth. Inkling is compute/overhead-bound,
   not bandwidth-bound.

**What would fix it:** DSpark speculative decoding (0.4 GB drafter, [PR #32218](https://github.com/ggml-org/llama.cpp/pull/32218) —
still open in mainline; works today with unsloth's prebuilt `b12588-mix-c0c3de8` + root GGUFs) and
maturing NVFP4 kernels after #32704 merges.

## Interactive visualization

[`docs/inference-visualization.html`](docs/inference-visualization.html) — open it in any browser (self-contained, no dependencies).
Watch one decode tick flow through all three architectures side by side: input → ① attention (KV cache
on the left) → ② router (MoE only) → ③ weight GEMM (weights on the right) → output, with live packets
sized by bytes read, an explicit **loop-back edge** for Inkling's 3 passes, expert banks that sleep
(Maxwell) vs. fully-lit weight tiles (Qwen), a KV-depth slider, per-token phase bars, and all measured
ladder data.

## Critical findings for `inkling2_loop` on llama.cpp (as of 2026-11-21)

1. **Stock `ghcr.io/ggml-org/llama.cpp:server-cuda` produces wrong output** for the
   [Mainline_Support](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/Mainline_Support/README.md)
   GGUFs: layout support (#32694) is merged, but the scale-reshare fix
   [#32704](https://github.com/ggml-org/llama.cpp/pull/32704) is **not** — without it, loop passes 2 and 3
   lose their per-expert NVFP4 scales (perplexity ~130 instead of 7.65). This repo's
   `docker/Dockerfile` builds llama.cpp at the PR head (`2a28df0f9bd5`).
2. **`-sm tensor` is not implemented** for `inkling2_loop` → **layer-split is mandatory**.
3. **Never set `LLAMA_ARG_TENSOR_SPLIT` or `LLAMA_ARG_N_GPU_LAYERS`**: either one aborts the auto-fit
   path (`common_fit_params`), and the loader then tries to allocate the whole 16 GB model buffer on
   CUDA0 monotonically → OOM on 16 GB GPUs. `llama-bench` has the same bug → all benchmarks here are
   server-based (`scripts/depthscan.sh`, `scripts/probe_maxctx.sh`).
4. **The MoE compute buffer scales with context, not just ubatch**: ub 512 works only up to ~64k ctx;
   131k needs ub 256, 262k needs ub 128.
5. **`N_PARALLEL > 1` = per-slot KV** (ctx/np per slot): a 62k request is rejected at np=4. For long
   single-user contexts use np 1.
6. **The auto-fit split decides your tg**: comfortable split (~13.3/14.6 GiB) → full speed ~25 t/s;
   tight split (both GPUs ≥14.8 GiB) → ~17 t/s via VMM overflow, with *no* CPU warning in the log —
   the VRAM counters are the only indicator.

## Quick start

```bash
# 1) Download model (Mainline_Support variant! + mmproj, ~18.3 GB, resumable)
bash scripts/download-models.sh                     # writes into ./models/

# 2) Build the patched image (~10–25 min, sm_120; adjust CUDA arch for your GPU)
docker build -f docker/Dockerfile -t llamacpp-inkling2:server-cuda docker/

# 3) Run the stack (port 8091)
docker compose -f docker/docker-compose.yml up -d   # adjust ./models path in the compose file first

# 4) Benchmarks
bash scripts/depthscan.sh        # pp/tg ladder 2.6k→128k (server-based)
bash scripts/probe_maxctx.sh     # max-context ladder (q8_0 + q4_0, ubatch fallback)
bash scripts/paralleltest.sh     # parallel requests up to 64k combined
```

## Repo layout

```
├── README.md                        # this file
├── docs/
│   ├── BENCHMARKS.md                # full benchmark report (translated from RESULTS.md)
│   └── inference-visualization.html # animated 3-architecture data-flow comparison
├── docker/
│   ├── Dockerfile                   # llama.cpp @ 2a28df0f (#32694 + fix #32704), CUDA sm_120
│   ├── docker-compose.yml           # production stack, port 8091
│   └── .dockerignore
├── scripts/
│   ├── download-models.sh           # Mainline_Support UD-Q4_K_XL + mmproj-F16
│   ├── depthscan.sh                 # server-based pp/tg depth ladder
│   ├── probe_maxctx.sh              # max-context ladder with ubatch fallback
│   ├── paralleltest.sh              # 4×2.6k gen, 4×15.6k & 2×31k PP (=64k), single 62k
│   └── bench.sh                     # llama-bench wrapper (kept for single-GPU / post-merge use)
└── results/                         # raw data: depth CSVs, VRAM samples, max-ctx ladder
```

## Hardware & environment

- 2× NVIDIA RTX 5060 Ti 16 GB (Blackwell sm_120), layer-split + auto-fit
- AMD Ryzen 5 7500F (6C/12T) · 30 GB RAM · Docker 29.10 · CUDA 12.8.1
- Model: `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf` (17.63 GB; unsloth Dynamic 3.0, imatrix;
  mean KLD 0.0107 / 98.0 % same-top-token vs UD-Q8_K_XL) · `mmproj-F16.gguf` (639 MB, vision, optional)

## Methodology notes

- All t/s curves are **measured** (server-based fresh requests, `cache_prompt: false`, temp 0);
  raw data in [`results/`](results) and full tables in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
- The phase split in the visualization (weights / KV+attention / routing overhead per token) is a
  **calibrated estimate**: overhead = ms/token@depth0 − weights, calibrated at 378 GB/s effective
  bandwidth from the dense baseline; the KV share is the remainder of the measured curve. Curves are
  real, the decomposition is modeled — stated transparently in the page itself.
- Comparison numbers for Qwen3.8-27B and Maxwell-V1-Flash come from the sibling benchmark projects
  measured on the **same machine** with their documented stacks.

## Links

- [Inkling 2 Mini — Open LLM Radar](https://open-llm-radar.com/models/inkling-2-mini) · [unsloth run guide](https://unsloth.ai/docs/models/inkling-2-mini) · [model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [Mainline_Support README](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/Mainline_Support/README.md)
- [Qwen3.8-27B — Open LLM Radar](https://open-llm-radar.com/models/qwen38-27b) · [GGUF repo](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- [Maxwell-V1.0-Flash — Open LLM Radar](https://open-llm-radar.com/models/maxwell-v10-flash) · [GGUF repo](https://huggingface.co/AtomicChat/Maxwell-V1-Flash-GGUF)
- llama.cpp PRs: [#32694 inkling2_loop layout](https://github.com/ggml-org/llama.cpp/pull/32694) · [#32704 scale reshare fix](https://github.com/ggml-org/llama.cpp/pull/32704) · [#32218 DSpark support](https://github.com/ggml-org/llama.cpp/pull/32218)

## License

- Code, configs and scripts in this repo: **MIT** (see [LICENSE](LICENSE))
- Model weights: Apache 2.0 (Thinking Machines Lab) — see the [model acceptable use policy](https://thinkingmachines.ai/model-acceptable-use-policy/)
