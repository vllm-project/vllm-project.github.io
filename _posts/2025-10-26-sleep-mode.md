---
layout: post
title: "Zero-Reload Model Switching with vLLM Sleep Mode"
author: "Embedded LLM"
image: /assets/figures/2025-vllm-sleep-mode/sleepmode.png
thumbnail-img: /assets/figures/2025-vllm-sleep-mode/sleepmode.png
share-img: /assets/figures/2025-vllm-sleep-mode/sleepmode.png
---

## Introduction

**The multi-model serving problem:** You have two LLMs that each fit on your GPU, but not both at once. Traditional solutions force a bad tradeoff:

1. **Keep both models loaded** → Requires 2x the GPU memory (expensive, often impossible)
2. **Reload models on-demand** → 30-100+ seconds per switch (slow, wasteful)

![vLLM Sleep Mode](/assets/figures/2025-vllm-sleep-mode/sleepmode.png)

**vLLM Sleep Mode offers a third way:** Models hibernate in seconds and wake up fast—delivering the efficiency of on-demand loading with the speed of persistent serving.

### Two Sleep Levels for Different Needs

- **Level 1:** Offloads weights to CPU RAM (fast wake time)
- **Level 2:** Discards weights entirely (nearly as fast wake time, minimal RAM usage)

Both levels are **18-200x faster** than full reload and work seamlessly with Tensor Parallelism (TP), Pipeline Parallelism (PP), and Expert Parallelism (EP).

### Why Sleep Mode Beats Fast Weight Loaders

Even with instant weight loading, every cold start pays hidden costs that Sleep Mode avoids:

| Cost | Description | Fast Weight Loaders | Sleep Mode |
|------|-------------|---------------------|------------|
| 1. VRAM load time | Copying weights to GPU | ✅ Optimized | ✅ Preserved |
| 2. Memory allocator setup | CUDA allocator initialization | ❌ Every time | ✅ Preserved |
| 3. CUDA graph capture | Record execution graphs | ❌ Every time | ✅ Preserved |
| 4. GPU kernel JIT compilation | DeepGEMM, FlashInfer, TorchInductor | ❌ Every time | ✅ Preserved (after initial warmup) |
| 5. Cache warm-up | First-request overhead | ❌ Every time | ⚡ Quick re-warm |

By keeping the process alive, Sleep Mode preserves infrastructure (#2-4) and avoids expensive reinitialization. This is why benchmarks show **Sleep Mode inference is 61-88% faster** than cold starts.

**This post covers:**
- Comprehensive benchmarks across model sizes (0.6B to 235B) and GPUs (A4000 to A100)
- Technical deep-dives explaining the performance gains
- Ablation studies on warm-up impact and FP8 quantization
- Decision guide for choosing the right sleep level

## Quick Start: Using Sleep Mode

### Online Serving API

Start two vLLM servers with Sleep Mode enabled:

```bash
# Terminal 1: Start Phi-3-vision
export VLLM_SERVER_DEV_MODE=1
vllm serve microsoft/Phi-3-vision-128k-instruct --enable-sleep-mode --port 8001

# Terminal 2: Start Qwen3-0.6B
export VLLM_SERVER_DEV_MODE=1
vllm serve Qwen/Qwen3-0.6B --enable-sleep-mode --port 8002
```

### Sleep and Wake Models

```bash
# Put Phi-3-vision to sleep (Level 2 - minimal RAM usage)
curl -X POST 'localhost:8001/sleep?level=2'

# Put Qwen3-0.6B to sleep (Level 2)
curl -X POST 'localhost:8002/sleep?level=2'

# Wake up Phi-3-vision for inference
curl -X POST 'localhost:8001/wake_up'
curl -X POST 'localhost:8001/collective_rpc' \
  -H 'Content-Type: application/json' \
  -d '{"method":"reload_weights"}'

# IMPORTANT: Reset prefix cache after waking (Level 2 only)
curl -X POST 'localhost:8001/reset_prefix_cache'

# Now run inference on Phi-3-vision...
# (your inference requests here)

# Put back to sleep when done
curl -X POST 'localhost:8001/sleep?level=2'

# Wake up Qwen3-0.6B
curl -X POST 'localhost:8002/wake_up'
# (Level 1 doesn't need reload_weights or reset_prefix_cache)

# Run inference on Qwen3-0.6B...
```

> [!NOTE]
> For Level 2 sleep, you must call `reload_weights` and `reset_prefix_cache` after waking. Level 1 sleep doesn't require these extra steps.

> [!WARNING]
> **Security:** The `/sleep`, `/wake_up`, `/collective_rpc`, and `/reset_prefix_cache` endpoints require `VLLM_SERVER_DEV_MODE=1` and should only be exposed in trusted networks. These administrative endpoints can disrupt service and are intended for closed environments like training clusters or backend applications.

## Performance Overview

Let's see how Sleep Mode performs compared to traditional model reloading.

### Sleep Mode L1 vs No Sleep Mode Performance

The interactive chart below shows the **total time to perform 5 model switches**: running inference on Model A, switching to Model B, running inference on Model B, then repeating this pattern (A→B→A→B→A→B).

**With Sleep Mode:** Models sleep/wake between switches, preserving infrastructure.
**Without Sleep Mode:** Each switch requires a full vLLM restart and reload.

<div style="margin: 2rem 0;">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<div id="plotly-sleep-mode" style="width: 100%; height: 250px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  <strong>Model A:</strong> Qwen3-235B-A22B-Instruct-2507-FP8 (TP=4) | <strong>Model B:</strong> Qwen3-Coder-30B-A3B-Instruct (TP=1)<br>
  GPU: A100 | vLLM 0.11.0 | Sleep Level: 1 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code><br>
  
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-sleep-mode.js"></script>
</div>

## Inference Performance Boost

Beyond faster model switching, Sleep Mode also delivers **faster inference times**. Because models are already warmed up when woken from sleep, they skip the cold start overhead that affects freshly loaded models.

<div style="margin: 2rem 0;">
<div id="plotly-inference-comparison" style="width: 100%; height: 300px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  Inference time comparison showing wake mode (already warmed up) vs cold start (just loaded).<br>
  <strong>Inference time = prefill + decode (first request after wake/load).</strong> Each request uses a different question to avoid caching, limited to 100 tokens output.<br>
  Error bars show min/max variation across multiple runs. Values displayed on bars.<br>
  GPU: A100 | vLLM 0.11.0 | Sleep Level: 1 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code>
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-inference-comparison.js"></script>
</div>

#### Why Sleep Mode Improves Inference Speed

The 61-88% inference speedup isn't from faster weight loading—it's from **preserving expensive infrastructure** that cold starts must rebuild from scratch.

**What Sleep Mode Preserves:**

| Component | Preserved? | Cold Start Must Pay |
|-----------|-----------|---------------------|
| Memory allocator (CuMemAllocator) | ✅ Yes | ❌ Reinitialize every time |
| CUDA graphs | ✅ Yes | ❌ Re-capture every time |
| Process state (Python, CUDA context) | ✅ Yes | ❌ Restart every time |
| GPU kernel JIT cache | ✅ Yes (after initial warmup) | ❌ Recompile every time |

**The Critical Difference:**

- **Without Sleep Mode:** Process dies on unload → **You CANNOT benefit from pre-warm-up**
  - Must restart Python process and CUDA context
  - Must reinitialize memory allocator
  - Must re-capture CUDA graphs
  - Must re-JIT compile kernels (DeepGEMM, FlashInfer, TorchInductor)
  - **Result:** First inference is **4-7x slower** (see benchmarks: 0.92s wake vs 3.72s cold start)

- **With Sleep Mode:** Process stays alive → **Pre-warm-up pays off**
  - ✅ Allocator, graphs, process state, and JIT kernels all preserved after initial warmup
  - **Result:** First inference stays fast (~1s), avoiding the 3-4s cold start penalty

> [!NOTE]
> Timing varies significantly by model size, GPU generation, and configuration. See the [Impact of Warm-Up](#impact-of-warm-up-on-sleep-mode) section for detailed measurements showing 5-7x slowdown without warm-up.

## Model Switching Performance

The most dramatic benefit of Sleep Mode is in model switching time. Waking a sleeping model is **18-20x faster** than loading a fresh vLLM instance.

<div style="margin: 2rem 0;">
<div id="plotly-switching-comparison" style="width: 100%; height: 300px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  Model switching time: Wake from sleep vs cold start (fresh load).<br>
  Error bars show min/max variation across multiple runs. Values displayed on bars.<br>
  GPU: A100 | vLLM 0.11.0 | Sleep Level: 1 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code>
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-switching-comparison.js"></script>
</div>

## Hardware Scalability: A4000 GPU Results

Sleep Mode benefits aren't limited to high-end GPUs. Here's the same workload on an **A4000 GPU** with smaller models, demonstrating that the performance gains scale across different hardware tiers and model sizes.

<div style="margin: 2rem 0;">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<div id="plotly-sleep-mode-a4000" style="width: 100%; height: 300px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  <strong>Model A:</strong> Qwen3-0.6B | <strong>Model B:</strong> Phi-3-vision-128k-instruct<br>
  GPU: A4000 (TP=1) | vLLM 0.11.0 | Sleep Level: 1 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code><br>
  
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-sleep-mode-a4000.js"></script>
</div>

### A4000: Inference Performance

<div style="margin: 2rem 0;">
<div id="plotly-inference-a4000" style="width: 100%; height: 300px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  Inference time comparison on A4000: wake mode (already warmed up) vs cold start (just loaded).<br>
  <strong>Inference time = prefill + decode (first request after wake/load).</strong> Each request uses a different question to avoid caching, limited to 100 tokens output.<br>
  Error bars show min/max variation across multiple runs. Values displayed on bars.<br>
  GPU: A4000 (TP=1) | vLLM 0.11.0 | Sleep Level: 1 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code>
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-inference-a4000.js"></script>
</div>

### A4000: Model Switching Performance

<div style="margin: 2rem 0;">
<div id="plotly-switching-a4000" style="width: 100%; height: 300px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  Model switching time on A4000: Wake from sleep vs cold start (fresh load).<br>
  Error bars show min/max variation across multiple runs. Values displayed on bars.<br>
  GPU: A4000 (TP=1) | vLLM 0.11.0 | Sleep Level: 1 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code>
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-switching-a4000.js"></script>
</div>

**Key Observations on A4000:**
- **Inference Performance:** Wake mode delivers 83% faster inference for Qwen3-0.6B and 81% faster for Phi-3-vision
- **Model Switching:** Wake times are incredibly fast (~0.1-0.8s), achieving **58-203x speedup** vs cold starts
- **Total time savings: 62%** (85s vs 226s for 5 model switches)
- **Near-instant switching** for small models (0.1s wake time), making multi-model serving feel seamless
- Demonstrates that Sleep Mode is effective across different GPU classes and model sizes

## Sleep Levels: Choosing the Right Mode

vLLM Sleep Mode offers two levels with different tradeoffs:

**Level 1 (Default):** Offloads model weights to CPU memory, discards KV cache
- **Fastest wake times** (~0.1-0.8s for small models, ~3-6s for large models)
- **Requires sufficient CPU RAM** to store model weights
- **Best for:** Systems with adequate CPU memory, frequent model switching

**Level 2:** Discards model weights and KV cache, keeps only buffers (rope scaling tensors, etc.) in CPU
- **Slower wake times** (~0.8-2.6s for small models) due to weight reload from disk
- **Minimal CPU RAM usage** - only small buffers retained
- **Best for:** Systems with limited CPU RAM or when managing many models that won't all fit in memory

### Performance Comparison: Level 1 vs Level 2 vs No Sleep

<div style="margin: 2rem 0;">
<div id="plotly-sleep-levels-comparison" style="width: 100%; height: 300px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  <strong>Model A:</strong> Qwen3-0.6B | <strong>Model B:</strong> Phi-3-vision-128k-instruct<br>
  GPU: A100 (TP=1) | vLLM 0.11.0 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code><br>
  <span style="font-size:0.9rem; margin-top:0.25rem; display:inline-block;">Comparing all three modes: Level 1 (fastest), Level 2 (minimal RAM), No Sleep. Hover for exact timing.</span>
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-sleep-levels-comparison.js"></script>
</div>

**Performance Summary:**

| Mode | Total Time | Wake Time (A/B) | CPU RAM | Best For |
|------|------------|-----------------|---------|----------|
| **No Sleep** | 357.1s | N/A (full reload) | Minimal | Single model, no switching |
| **Level 1** | 112.6s | 0.26s / 0.82s | High (~GB per model) | Frequent switching, ample RAM |
| **Level 2** | 124.6s | 0.85s / 2.58s | Minimal (~MB per model) | Limited RAM, cost optimization |

**Key Insights:**
- **Level 1 is fastest** (68% faster than no sleep) but needs significant CPU RAM
- **Level 2 is nearly as fast** (65% faster than no sleep) with minimal RAM requirements
- **Level 2 wake is ~3x slower than Level 1** (0.85s vs 0.26s for Qwen3-0.6B) due to weight reload
- Both sleep modes deliver **massive improvements** over no sleep mode

#### Why Level 2 is Still Faster Than No Sleep Mode

At first glance, this seems counterintuitive: **Level 2 reloads weights from SSD** (just like "No Sleep Mode"), so why is it **23-45x faster overall?**

**The Answer: Weight loading is only ONE of FIVE costs**

When you reload a model without Sleep Mode, you pay all these costs:

| Cost | Level 2 | No Sleep Mode |
|------|---------|---------------|
| 1. Weight load (SSD → VRAM) | ❌ Must pay | ❌ Must pay |
| 2. Process initialization | ✅ **Skipped** | ❌ Must pay |
| 3. Memory allocator setup | ✅ **Skipped** | ❌ Must pay |
| 4. CUDA graph capture | ✅ **Skipped** | ❌ Must pay |
| 5. GPU kernel JIT compilation | ✅ **Preserved (already compiled)** | ❌ Full compilation + warm-up |

**Level 2 Strategy:**
- Weight reload from SSD (same as No Sleep)
- **Everything else preserved:** Process state, allocator instance, CUDA graphs, and compiled JIT kernels all intact
- **No recompilation needed:** Kernels were compiled during initial warmup and remain cached
- **Average per switch: ~2.6s** (see benchmark data above)

**No Sleep Mode Reality:**
- Weight reload from SSD (same as Level 2)
- **Everything else rebuilt:** Process restart + allocator init + graph re-capture
- **JIT kernels:** Full compilation + explicit warm-up routine (`kernel_warmup()` + dummy runs)
- **Average per switch: ~48s** (see benchmark data above)

**The benchmark data proves it:** For 5 model switches:
- **Level 2:** 124.6s total (average ~2.6s per switch)
- **No Sleep:** 357.1s total (average ~48s per switch)

Even though both reload weights from SSD, Level 2 is **2.9x faster overall** because it preserves the expensive infrastructure (process state, allocator, CUDA graphs) that No Sleep Mode must rebuild from scratch every single time.

### Level 2: Inference Performance

<div style="margin: 2rem 0;">
<div id="plotly-level2-inference" style="width: 100%; height: 300px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  Inference time comparison with Sleep Level 2: wake mode vs cold start.<br>
  <strong>Inference time = prefill + decode (first request after wake/load).</strong> Each request uses a different question to avoid caching, limited to 100 tokens output.<br>
  Error bars show min/max variation across multiple runs. Values displayed on bars.<br>
  GPU: A100 (TP=1) | vLLM 0.11.0 | Sleep Level: 2 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code>
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-level2-inference.js"></script>
</div>

### Level 2: Model Switching Performance

<div style="margin: 2rem 0;">
<div id="plotly-level2-switching" style="width: 100%; height: 300px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  Model switching time with Sleep Level 2: wake from sleep vs cold start.<br>
  Error bars show min/max variation across multiple runs. Values displayed on bars.<br>
  GPU: A100 (TP=1) | vLLM 0.11.0 | Sleep Level: 2 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code>
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-level2-switching.js"></script>
</div>

**Key Observations:**

| Metric | No Sleep | Level 2 | Improvement |
|--------|----------|---------|-------------|
| **Total Time (5 switches)** | 357.1s | 124.6s | **65% faster** |
| **Qwen3-0.6B Switch Time** | 37.6s avg | 0.85s avg | **45x faster** |
| **Phi-3-vision Switch Time** | 58.1s avg | 2.58s avg | **23x faster** |
| **Qwen3-0.6B Inference** | 3.67s avg | 0.53s avg | **86% faster** |
| **Phi-3-vision Inference** | 6.30s avg | 0.76s avg | **88% faster** |
| **Wake Time vs Level 1** | - | 3-10x slower | Trade CPU RAM for speed |

**When to Use Level 2:**
- **Limited CPU RAM:** System cannot hold all model weights in CPU memory
- **Cost Optimization:** Cheaper cloud instances with less CPU RAM
- **Many Models:** Switching between many models where CPU memory is a constraint
- **Still Significant Gains:** Even with weight reload, Level 2 is 23-45x faster than no sleep mode

**Level 1 vs Level 2 Comparison:**
- Level 1: ~0.1-0.8s wake time, needs ~10-100GB+ CPU RAM per model
- Level 2: ~0.8-2.6s wake time, needs only ~MB CPU RAM per model
- Both dramatically faster than full reload (~20-100s)

## Ablation Studies

### Impact of Warm-Up on Sleep Mode

Does skipping the warm-up phase affect performance? Warm-up pre-compiles CUDA graphs during initial load, which can take several seconds. Let's compare with and without warm-up.

<div style="margin: 2rem 0;">
<div id="plotly-ablation-warmup" style="width: 100%; height: 300px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  <strong>Model A:</strong> Qwen3-0.6B | <strong>Model B:</strong> Phi-3-vision-128k-instruct<br>
  GPU: A100 (TP=1) | vLLM 0.11.0 | Sleep Level: 1 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code><br>
  <span style="font-size:0.9rem; margin-top:0.25rem; display:inline-block;">Comparing with warm-up (pre-compiled) vs without warm-up (lazy compilation). Hover for exact timing.</span>
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-ablation-warmup.js"></script>
</div>

**Key Findings:**

| Metric | With Warm-Up | Without Warm-Up | Difference |
|--------|--------------|-----------------|------------|
| **Initial Load Time** | 108.7s (includes 8.4s warm-up) | 101.1s (no warm-up) | 7.6s saved initially |
| **First Inference (A)** | 0.45s | 2.59s | **5.8x slower** without warm-up |
| **First Inference (B)** | 0.93s | 6.61s | **7.1x slower** without warm-up |
| **Subsequent Inferences** | 0.43s avg | 0.41s avg | No difference |
| **Total Time (5 switches)** | 119.5s | 119.0s | Nearly identical |

**Insights:**
- **Warm-Up Compiles Kernels Once, Benefits All Wake Cycles:** With initial warmup, JIT compilation and CUDA graph capture happen once during load and are preserved across all subsequent sleep/wake cycles
- **Without Warm-Up, Every Wake-Up Pays Compilation Cost:** The 5-7x slowdown happens on the first inference after **every single wake-up**, not just once
- **Compiled Kernels Are Preserved Across Sleep/Wake:** After warmup during initial load (8.4s), all subsequent wake-ups have fast first inference (0.45s, 0.93s) proving kernels stay cached
- **Minimal Warmup Sufficient:** A single 1-token inference is enough to trigger full JIT compilation and CUDA graph capture, making warmup very cheap
- **Trade Initial Load Time for Consistent Performance:** The 8.4s warmup cost is paid once and amortized across all model switches
- **Recommendation: Always Use Warm-Up** for production workloads where consistent, fast inference is expected

### Impact of Quantization on Sleep Mode

Does quantization (FP8) affect Sleep Mode performance? We tested the same workload with and without FP8 quantization on A100 GPU.

<div style="margin: 2rem 0;">
<div id="plotly-ablation-quant" style="width: 100%; height: 300px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  <strong>Model A:</strong> Qwen3-0.6B | <strong>Model B:</strong> Phi-3-vision-128k-instruct<br>
  GPU: A100 (TP=1) | vLLM 0.11.0 | Sleep Level: 1 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code><br>
  <span style="font-size:0.9rem; margin-top:0.25rem; display:inline-block;">Comparing BF16 (baseline) vs FP8 quantization. Hover for exact timing.</span>
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-ablation-quant.js"></script>
</div>

### Ablation: Inference Performance (BF16 vs FP8)

<div style="margin: 2rem 0;">
<div id="plotly-ablation-inference" style="width: 100%; height: 300px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  Inference time comparison: BF16 vs FP8 quantization with Sleep Mode.<br>
  <strong>Inference time = prefill + decode (first request after wake/load).</strong> Each request uses a different question to avoid caching, limited to 100 tokens output.<br>
  Error bars show min/max variation across multiple runs. Values displayed on bars.<br>
  GPU: A100 (TP=1) | vLLM 0.11.0 | Sleep Level: 1 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code>
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-ablation-inference.js"></script>
</div>

### Ablation: Model Switching (BF16 vs FP8)

<div style="margin: 2rem 0;">
<div id="plotly-ablation-switching" style="width: 100%; height: 300px;"></div>
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:0.75rem;">
  Model switching time: BF16 vs FP8 quantization with Sleep Mode.<br>
  Error bars show min/max variation across multiple runs. Values displayed on bars.<br>
  GPU: A100 (TP=1) | vLLM 0.11.0 | Sleep Level: 1 | Compilation: <code style="font-size:0.8rem;">cudagraph_mode: FULL_AND_PIECEWISE</code>
</div>
<script src="/assets/figures/2025-vllm-sleep-mode/plotly-ablation-switching.js"></script>
</div>

**Key Findings:**

| Metric | BF16 | FP8 | Improvement |
|--------|------|-----|-------------|
| **Total Time (5 switches)** | 108.2s | 113.6s | -5% (slightly slower) |
| **Qwen3-0.6B Wake Time** | 0.27s avg | 0.18s avg | **33% faster** |
| **Phi-3-vision Wake Time** | 0.90s avg | 0.78s avg | **13% faster** |
| **Qwen3-0.6B Inference** | 0.41s avg | 0.44s avg | -7% (slightly slower) |
| **Phi-3-vision Inference** | 0.81s avg | 0.57s avg | **30% faster** |
| **Initial Load Time** | 90.5s | 96.9s | -7% (longer warmup) |

**Insights:**
- **FP8 has faster wake operations** (13-33% faster) due to less memory movement
- **FP8 improves inference for larger models** (30% faster for Phi-3-vision) but shows minimal difference for tiny models
- **Initial load takes longer with FP8** due to quantization overhead during warmup
- **After initial load, FP8 provides smoother switching** with faster wake cycles
- For workloads with frequent switching, FP8's faster wake times can offset the longer initial load

## Decision Guide: Which Sleep Level to Use?

### Use Sleep Level 1 When:
- You have sufficient CPU RAM to hold all model weights
- You need the fastest possible wake times (0.1-6s)
- You're switching models very frequently (every few seconds/minutes)
- Inference latency consistency is critical

### Use Sleep Level 2 When:
- CPU RAM is limited (can't hold all model weights)
- You're optimizing cloud costs (cheaper instances with less RAM)
- You have many models to manage (10+)

### Skip Sleep Mode When:
- You're only using a single model (no switching needed)
- Model switches are extremely rare (once per day/week)
- Both models fit simultaneously in GPU memory

## Conclusion

vLLM Sleep Mode transforms multi-model GPU serving from a 30-100 second reload penalty into sub-second switches. The benchmarks speak for themselves:

- **18-200x faster model switching** depending on model size and hardware
- **61-88% faster inference** for warmed models vs cold starts
- **65-68% total time savings** across complete workloads
- **Works at every scale:** 0.6B to 235B parameters, small and large GPUs

The future of LLM serving is multi-model. Sleep Mode makes it practical today.

## Acknowledgements

Special thanks to **Vensen Mu**, **Jeff Aw**, **Jun Kang Chow**, **Tun Jian Tan**, **Pin Siang Tan**, **Amir Balwel**, **Ye Hur Cheong**, **Zhiyao Cen** and **Kaichao You** for developing the Sleep Mode feature and this blog post.
