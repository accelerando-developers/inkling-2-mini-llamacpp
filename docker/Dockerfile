# llama.cpp with the inkling2_loop scale-reshare fix for Inkling 2 Mini (Thinking Machines Lab / unsloth GGUF)
# Source: https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/Mainline_Support/README.md
#
# IMPORTANT: the stock image ghcr.io/ggml-org/llama.cpp:server-cuda contains the inkling2_loop
# layout (PR #32694, merged), but NOT the fix PR #32704 (status 2026-11-21: OPEN):
#   "model : inkling2_loop re-share the loop slots after the scale pass"
# Without the fix, the 2nd and 3rd passes through the looped layers lose their per-expert
# NVFP4 scales → output is WRONG (perplexity ~130 instead of 7.65). The fix is mandatory.
#
# Base: llama.cpp @ PR head 2a28df0f9bd55c4975b6bffb35229a6020720f13
#       (contains #32694 inkling2_loop layout + #32704 fix; ref: inkling2-loop-scale-reshare)
# Once #32704 merges into master, any build after that works — and you can simply use the
# stock image instead of building this.
#
# UI: the WebUI dist is built in a Node stage (same as the official .devops/cuda.Dockerfile) —
# the server target depends on llama-ui-assets, the build fails without the dist.
#
# Build (build context = repo root):
#   docker build -f docker/Dockerfile -t llamacpp-inkling2:server-cuda .
#   (~10–25 min; adjust CMAKE_CUDA_ARCHITECTURES / GGML_CUDA_ARCHITECTURES for your GPU,
#    e.g. 86 for RTX 30xx, 89 for RTX 40xx; native builds can drop both and add -DGGML_NATIVE=OFF)

# ---------- Stage 1: WebUI (vite/svelte) ----------
FROM node:24-bookworm AS web
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init . \
 && git remote add origin https://github.com/ggml-org/llama.cpp.git \
 && git fetch --depth 1 origin 2a28df0f9bd55c4975b6bffb35229a6020720f13 \
 && git checkout --detach FETCH_HEAD
WORKDIR /src/ui/apps/chat
RUN npm ci
RUN LLAMA_BUILD_NUMBER="0" npm run build

# ---------- Stage 2: llama.cpp + inkling2_loop fix ----------
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init . \
 && git remote add origin https://github.com/ggml-org/llama.cpp.git \
 && git fetch --depth 1 origin 2a28df0f9bd55c4975b6bffb35229a6020720f13 \
 && git checkout --detach FETCH_HEAD \
 && echo "llama.cpp @ 2a28df0f9bd55c4975b6bffb35229a6020720f13 (inkling2_loop + fix #32704)" > /SRC-SHA
COPY --from=web /src/ui/apps/chat/dist /src/ui/apps/chat/dist
RUN cmake -B /build -S /src \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DGGML_CUDA_ARCHITECTURES=120 \
      -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      # Don't link the driver API (cuMem*) at build time — official: .devops/cuda.Dockerfile
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
 && cmake --build /build -j"$(nproc)" --target llama-server llama-cli llama-bench

# ---------- Stage 3: Runtime ----------
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 tzdata curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/bin/ /app/
COPY --from=build /SRC-SHA /app/SRC-SHA
ENV LD_LIBRARY_PATH=/app
WORKDIR /app
ENTRYPOINT ["/app/llama-server"]
