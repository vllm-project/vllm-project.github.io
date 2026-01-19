---
layout: post
title: "Beyond Porting: How vLLM Orchestrates High-Performance Inference on AMD ROCm"
author: "AMD and Embedded LLM"
image: /assets/figures/2025-12-16-rocm-attention-backend/vLLM-Attention.png
math: true
---

## Introduction

For a long time, enabling AMD support meant "porting": just making code run. **That era is over.**

With AMD CDNA 3 architecture hardwares (MI300X, MI325X, MI355X) and complex model structures like DeepSeek's MLA, "just running" isn't enough. These workloads demand _architectural co-design_, where software orchestration and hardware primitives work together.

vLLM now provides 7 attention backends on AMD ROCm. This post explains each one: why they exist, their trade-offs, and when to use them. We provide transparent benchmarks comparing all backends, and show how `ROCM_AITER_FA` for MHA and the AITER MLA backends deliver **1.2-6.3x performance gains** through AMD's AITER primitives and vLLM's kernel orchestration.

---

## The Challenge: Mixed Workloads in Every Batch

In production LLM serving, each inference step processes a mixed batch of tokens from different request types. The industry has recognized this challenge—various solutions exist, from unified kernels with sophisticated internal scheduling to multi-path routing with specialized kernels. AMD's `ROCM_AITER_FA` takes the explicit routing approach, making workload-aware optimization a first-class design principle rather than an internal kernel detail.

- **Prefill**: New prompts arriving at the server. These contain thousands of input tokens that need attention computation all at once. The GPU is doing heavy matrix multiplication here, making prefill **compute-bound**.

- **Decode**: Generating output tokens one at a time. Each decode step loads the entire KV cache from memory to produce a single token. The bottleneck is memory bandwidth, making decode **memory-bound**.

- **Extend**: Continuing conversations where part of the context is already cached (prefix cache hit). This is a hybrid scenario where new tokens must attend to both cached context and fresh input.

These request types arrive randomly and are batched together for efficiency.

![Contiguous-Batching Example](/assets/figures/2025-12-16-rocm-attention-backend/contiguous-batching.png)
_Figure 1: Online serving with 5 concurrent requests. Step 4 shows prefill, extend, and decode tokens batched together._

The optimization challenge: prefill wants large tile sizes and maximum ALU utilization, while decode wants coalesced memory access and minimal cache fetches. **A kernel tuned for one workload leaves performance on the table for the other.**

This mixed-workload scenario is exactly what `ROCM_AITER_FA`'s 3-path routing addresses: instead of forcing all request types through one kernel, it routes each type to a specialized kernel optimized for that workload's characteristics.

---

## Other MHA Backends

Before diving into `ROCM_AITER_FA`, let's understand the other MHA backends available:

### Unified Attention Backends

These backends process all tokens (prefill/extend/decode) through a single kernel path:

| Backend                                                                                                                         | Kernel Source                                                                                                            | Use Case                 |
| ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------ |
| [TRITON_ATTN](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py)                         | [vLLM Triton kernel](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py#L57)  | Default fallback         |
| [ROCM_AITER_UNIFIED_ATTN](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_aiter_unified_attn.py) | [AITER Triton kernel](https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/attention/unified_attention.py#L54) | Single-kernel AITER path |

```python
def forward():
    # Stage 1: Save Key/Value into KV-Cache
    reshape_and_cache_flush(new_key, new_value, ...)
    # Stage 2: Single kernel for all attention
    unified_attention_kernel(new_query, KV-Cache, ...)
```

### ROCM_ATTN: Legacy 2-Path Backend

[ROCM_ATTN](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py) uses 2-path routing with different kernels per phase:

- **Prefill**: Triton kernel
- **Decode**: HIP paged attention kernel (when supported)

This backend has two important characteristics:

1. **Legacy 2-path architecture**: Uses separate kernels for prefill (Triton) and decode (HIP paged attention). Note that the HIP paged attention kernel only supports certain KV head sizes—for unsupported configurations (like Qwen3-235B), it falls back to Triton decode kernels, resulting in significantly slower performance.

2. **Radeon GPU support**: Along with `TRITON_ATTN`, this backend supports **Radeon GPUs**—useful for consumer hardware deployments where AITER primitives aren't available.

---

## The ROCM_AITER_FA Backend: Kernel Orchestration for AMD

`ROCM_AITER_FA` isn't just a kernel wrapper—it's a sophisticated orchestration layer that routes requests to specialized kernels, combining vLLM's high-level management with AMD's AITER primitives.

![AITER FA Backend](/assets/figures/2025-12-16-rocm-attention-backend/ROCm-Attention.png)
_Figure 2: (a) Unified attention processes all tokens through one kernel. (b) ROCM_AITER_FA routes tokens to three specialized paths._

### Key Innovations

**1. Three-Path Routing**: Requests are dynamically categorized into Decode, Prefill, and Extend paths—each with optimized kernels:

![Three-Path Architecture](/assets/figures/2025-12-16-rocm-attention-backend/three_path_architecture.png)
_Each path uses specialized kernels optimized for its workload characteristics._

- **Decode**: Single token generation uses AITER highly optimized kernel for memory bandwidth
- **Prefill**: New sequences use `flash_attn_varlen_func`—leveraging CDNA matrix cores for compute-heavy work
- **Extend**: Continuing sequences use chunked attention with LSE merging—handling 100K+ contexts efficiently

**2. Batch Reordering**: Requests are reordered to `[decode:extend:prefill]` for contiguous memory access, eliminating redundant KV cache fetches.

![Batch Reordering](/assets/figures/2025-12-16-rocm-attention-backend/batch_reordering.png)
_Batch reordering ensures each kernel path operates on contiguous tokens, eliminating redundant KV cache fetches._

**3. Chunked Context Processing**: Long sequences are processed in chunks sized by a fixed per-iteration token budget (~32K tokens total), split across extend requests; LSE-based merging ensures numerical stability.

![Chunked Context Flow](/assets/figures/2025-12-16-rocm-attention-backend/chunked_context_flow.png)
_100K+ token contexts are processed in 32K chunks with LSE-based merging for numerical stability._

**4. Hardware-Optimized KV Cache Layout**: Uses a preshuffled KV cache layout designed by AMD's AITER kernel team:

```python
k_cache: [num_blocks, num_heads, head_dim // x, block_size, x]
v_cache: [num_blocks, num_heads, block_size // x, head_dim, x]
```

This layout aligns memory access patterns with AMD's CDNA architecture, enabling the decode path to call AITER's `paged_attention` kernel with **zero layout conversion overhead**—delivering **15-20% decode throughput improvement** compared to standard KV cache layouts.

### Why Explicit 3-Path Routing?

`ROCM_AITER_FA` makes a deliberate architectural choice: route workloads at the software layer rather than relying on a single kernel to handle everything.

This explicit approach offers:

- **Debuggability**: Each path can be profiled, tuned, and optimized independently
- **Portability**: The same routing logic works across MI300X → MI325X → MI355X without hardware-specific changes
- **Extensibility**: New workload types or kernel variants can be added without redesigning the core architecture
- **Predictability**: Execution paths are deterministic, making performance analysis straightforward

The extend path is particularly important: prefix caching and multi-turn conversations are now standard in production deployments. Having a dedicated path with chunked context attention (rather than treating it as a special case of prefill) ensures these workloads get first-class optimization.

### Three-Path Processing in Detail

**Decode Path**: Leverages the shuffled KV cache layout directly. A custom `reshape_and_cache_flush` operator ensures the cache is always in the shuffled layout, allowing the attention backend to call AITER's performant `paged_attention` kernel with zero layout conversion overhead.

**Prefill Path**: Query/Key/Value are in the standard `[num_tokens, num_heads, head_dim]` layout to align with the highly optimized AITER MHA kernel and avoid any extra memory copy operations.

**Extend Path**: This is the most challenging path. New tokens must compute attention with context tokens stored in the shuffled KV cache layout. Since the shuffled layout is incompatible with AITER's MHA kernel for long-context computation, we insert an extra KV Cache fetching operator (`cp_mha_gather_cache`) to fetch and convert context Key/Value to standard layout. Long contexts are chunked into segments to manage memory:

```python
def extend_forward():
    # Stage 1: Attention for new tokens
    flash_attn_varlen_func()  # calling AITER MHA

    # Stage 2: Context Chunk Loop Processing
    for chunk in context_chunks:
        cp_mha_gather_cache()      # Triton gather kernel
        flash_attn_varlen_func()   # calling AITER MHA
        merge_attn_states()        # LSE-based merge

    # Stage 3: Get the final result
    merge_attn_states()
```

Each chunk produces an output and LSE (log-sum-exp). The LSE captures the softmax denominator, enabling numerically stable merging—chunks with higher attention scores naturally dominate the final result.

---

## The ROCM_AITER_TRITON_MLA Backend: Hybrid Strategy for DeepSeek

DeepSeek and Kimi's MLA architecture compresses the KV cache to **576 dimensions** (vs ~8K for standard MHA)—a 14x memory reduction. This compression changes the performance characteristics of attention, requiring a different optimization strategy than standard MHA.

### The Hybrid Approach

vLLM provides two AITER-based MLA backends with different prefill implementations:

| Backend                 | Prefill Kernel | Decode Kernel  |
| ----------------------- | -------------- | -------------- |
| `TRITON_MLA`            | vLLM Triton    | vLLM Triton    |
| `ROCM_AITER_MLA`        | AITER MHA (CK on gfx942, ASM on gfx950) | AITER Assembly |
| `ROCM_AITER_TRITON_MLA` | AITER Triton MHA | AITER Assembly |

The base `TRITON_MLA` backend uses vLLM's default Triton kernels for both phases. The AITER backends replace the decode kernel with hand-tuned assembly (`mla_decode_fwd`), which is where most of the performance gain comes from. The only difference between the two AITER backends is the prefill path: `ROCM_AITER_MLA` calls `aiter.flash_attn_varlen_func` (AITER MHA), while `ROCM_AITER_TRITON_MLA` calls `aiter.ops.triton.mha.flash_attn_varlen_func` (AITER Triton MHA). On gfx942 (MI300X/MI325X), AITER MHA resolves to CK kernels; on gfx950 (MI355X), it resolves to the newer assembly MHA kernels.

**Why the prefill winner flips by architecture?** Two reasons:

1. **gfx942 (MI300X/MI325X): XCD-aware scheduling**: The AITER Triton kernel explicitly remaps head distribution across XCDs via `remap_xcd()`, ensuring balanced load. The CK path doesn't expose this logic at the Python level.

2. **gfx942 (MI300X/MI325X): Runtime config flexibility**: Triton selects architecture-specific configs at runtime (e.g., `BLOCK_M=128, BLOCK_N=64` for MI300X), while CK uses a fixed set of pre-generated kernels.

On **gfx950 (MI355X)**, AITER MHA switches to the assembly kernel, which is hand-tuned for MI355X and tends to beat the Triton MHA path. That is why `ROCM_AITER_MLA` often pulls ahead on TTFT even when TPOT stays close.

### Absorbed vs Non-Absorbed Recipe

All MLA backends use the same fundamental processing strategy:

- **Prefill/Extend (Non-Absorbed)**: Compute attention with standard MHA kernels on the uncompressed representation
- **Decode (Absorbed)**: Use specialized MLA kernels operating directly on the compressed 576-dim latent space

```python
def _forward_prefill():
    # Stage 1: Attention for new tokens (non-absorbed)
    _run_prefill_new_tokens()

    # Stage 2: For extend path, context chunk loop
    for chunk in context_chunks:
        gather_and_maybe_dequant_cache()
        _run_prefill_context_chunk()
        merge_attn_states()

    # Stage 3: Final merge
    merge_attn_states()
```

During **decode**, the model generates one token at a time. The compressed KV cache means less data to load, but you're still bottlenecked by memory bandwidth—making decode **memory-bound**. The AITER assembly kernel (`mla_decode_fwd`) maximizes every byte of HBM3 bandwidth, significantly outperforming generic Triton decode kernels.

### Why the Assembly Decode Kernel Matters

`ROCM_AITER_TRITON_MLA` compared to `TRITON_MLA` (vLLM's pure-Triton baseline):

| Phase       | AITER Kernel              | vLLM TRITON_MLA Kernel        |
| ----------- | ------------------------- | ----------------------------- |
| **Prefill** | Triton flash attention    | Triton flash attention        |
| **Decode**  | Assembly `mla_decode_fwd` | Triton `decode_attention_fwd` |

The **1.27-1.39x speedup** primarily comes from the assembly decode kernel. TPOT is decode-heavy (1K iterations), so optimizing decode yields the largest throughput gains.

Beyond raw kernel performance, this backend inherits the full feature set of FlashMLABackend, including FULL_AND_PIECEWISE CUDA graph support and MTP support. Another advantage is its near-identical performance across virtually any KV cache block size—you can treat every token as prefix cache without worrying about performance penalties typically associated with fine-grained caching.

---

## Performance Benchmarks

**Benchmark Methodology**: All benchmarks were run using `rocm/vllm-dev:nightly_main_20260107`. We warm up kernels with initial requests first; reported results exclude the first run to eliminate JIT compilation overhead.

### MHA Benchmark Results

**Model**: [Qwen3-235B-A22B-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8), TP8+EP8 | **Workload**: ISL=10K, OSL=1K, 64 & 128 concurrent

![MHA TPOT Comparison](/assets/figures/2025-12-16-rocm-attention-backend/mha_tpot_comparison.png)
_ROCM_AITER_FA delivers 4.0-6.3x faster TPOT compared to legacy ROCM_ATTN across MI300X/MI325X/MI355X._

![MHA TTFT Comparison](/assets/figures/2025-12-16-rocm-attention-backend/mha_ttft_comparison.png)
_TTFT (Time To First Token) comparison shows ROCM_AITER_FA and ROCM_AITER_UNIFIED lead in prefill performance at 64 and 128 concurrency levels._

![MHA TPS Comparison](/assets/figures/2025-12-16-rocm-attention-backend/mha_tps_comparison.png)
_Output throughput (TPS) mirrors TPOT results—ROCM_AITER_FA achieves 2.7-4.3x higher throughput than legacy ROCM_ATTN._

**Relative to ROCM_AITER_FA (higher = slower, 64 concurrent):**

| Hardware | ROCM_AITER_FA | ROCM_AITER_UNIFIED_ATTN | TRITON_ATTN | ROCM_ATTN |
| -------- | ------------- | ----------------------- | ----------- | --------- |
| MI300X   | 1.00x         | 1.06x                   | 1.30x       | 4.09x     |
| MI325X   | 1.00x         | 1.02x                   | 1.17x       | 4.65x     |
| MI355X   | 1.00x         | 1.01x                   | 1.13x       | 4.00x     |

**Relative to ROCM_AITER_FA (higher = slower, 128 concurrent):**

| Hardware | ROCM_AITER_FA | ROCM_AITER_UNIFIED_ATTN | TRITON_ATTN | ROCM_ATTN |
| -------- | ------------- | ----------------------- | ----------- | --------- |
| MI300X   | 1.00x         | 1.06x                   | 1.36x       | 2.76x     |
| MI325X   | 1.00x         | 1.01x                   | 1.28x       | 3.32x     |
| MI355X   | 1.00x         | 1.02x                   | 1.23x       | 6.27x     |

The relative performance is consistent across GPU generations. Note that `ROCM_AITER_UNIFIED_ATTN` (single-kernel path) is only ~1-6% slower than `ROCM_AITER_FA` (3-path routing) in this uniform workload scenario.

_Note: ROCM_ATTN shows 4-6x slower performance because Qwen3-235B has unsupported KV head sizes for HIP paged attention, forcing it to fall back to Triton decode kernels. Models with supported head sizes would see smaller gaps._

### MLA Benchmark Results

**Model**: [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528), TP8, block_size=16 | **Workload**: ISL=10K, OSL=1K, 64 & 128 concurrent

![MLA TPOT Comparison](/assets/figures/2025-12-16-rocm-attention-backend/mla_tpot_comparison.png)
_ROCM_AITER_TRITON_MLA delivers 1.27-1.54x faster TPOT compared to TRITON_MLA across MI300X/MI325X/MI355X._

![MLA TTFT Comparison](/assets/figures/2025-12-16-rocm-attention-backend/mla_ttft_comparison.png)
_TTFT comparison shows ROCM_AITER_MLA achieves the best TTFT on MI355X at 128 concurrency._

![MLA TPS Comparison](/assets/figures/2025-12-16-rocm-attention-backend/mla_tps_comparison.png)
_Output throughput (TPS) shows ROCM_AITER_TRITON_MLA achieving up to 1.5x higher throughput than TRITON_MLA._

**TPOT (ms) - lower is better (64 concurrent):**

| Hardware | ROCM_AITER_TRITON_MLA | ROCM_AITER_MLA | TRITON_MLA |
| -------- | --------------------- | -------------- | ---------- |
| MI300X   | **70.09**             | 71.95          | 98.06      |
| MI325X   | **66.67**             | 67.65          | 100.44     |
| MI355X   | **52.07**             | 52.68          | 80.12      |

**Relative to ROCM_AITER_TRITON_MLA (higher = slower, 64 concurrent):**

| Hardware | ROCM_AITER_TRITON_MLA | ROCM_AITER_MLA | TRITON_MLA |
| -------- | --------------------- | -------------- | ---------- |
| MI300X   | 1.00x                 | 1.03x          | 1.40x      |
| MI325X   | 1.00x                 | 1.01x          | 1.51x      |
| MI355X   | 1.00x                 | 1.01x          | 1.54x      |

**TPOT (ms) - lower is better (128 concurrent):**

| Hardware | ROCM_AITER_TRITON_MLA | ROCM_AITER_MLA | TRITON_MLA |
| -------- | --------------------- | -------------- | ---------- |
| MI300X   | **125.98**            | 129.88         | 160.06     |
| MI325X   | **111.30**            | 115.83         | 148.52     |
| MI355X   | **81.88**             | 83.23          | 113.96     |

**Relative to ROCM_AITER_TRITON_MLA (higher = slower, 128 concurrent):**

| Hardware | ROCM_AITER_TRITON_MLA | ROCM_AITER_MLA | TRITON_MLA |
| -------- | --------------------- | -------------- | ---------- |
| MI300X   | 1.00x                 | 1.03x          | 1.27x      |
| MI325X   | 1.00x                 | 1.04x          | 1.33x      |
| MI355X   | 1.00x                 | 1.02x          | 1.39x      |

`ROCM_AITER_TRITON_MLA` (Triton prefill + ASM decode) shows marginal TPOT improvement (1-4%) over `ROCM_AITER_MLA`. On gfx942, this is mostly Triton beating CK; on gfx950, the gap narrows because `ROCM_AITER_MLA` uses the AITER assembly MHA prefill. `ROCM_AITER_MLA` achieves the best TTFT on MI355X at 128 concurrency. Both AITER MLA backends deliver similar overall performance—the auto-selected `ROCM_AITER_MLA` is recommended for most workloads. Users who want to squeeze out maximum TPOT can try `ROCM_AITER_TRITON_MLA`.

_Note: These benchmarks use uniform request sizes. Production workloads with prefix caching, mixed context lengths, and varied request patterns would exercise the 3-path routing architecture more fully._

---

## The Collaboration: vLLM + AITER

The performance gains don't come from a single optimization—they emerge from how vLLM's orchestration layer and AMD's AITER primitives work together. Understanding this collaboration explains why "just porting" falls short.

<img src="/assets/figures/2025-12-16-rocm-attention-backend/system_stack.png" alt="System Stack" style="width: 80%;">

_The complete system stack: from user request through vLLM orchestration to AITER primitives on AMD hardware._

### Innovation Attribution

Where does the performance come from? Both layers working together:

![Innovation Attribution](/assets/figures/2025-12-16-rocm-attention-backend/innovation_attribution.png)
_vLLM orchestration handles routing and chunking; AITER provides hardware-optimized primitives._

**The key insight**: AITER provides highly optimized attention primitives purpose-built for CDNA. vLLM's orchestration layer adds workload-aware routing and chunked processing that unlock the final performance tier. Neither alone achieves optimal results.

---

## Get Started

### Quick Start

```bash
# Recommended: Let vLLM auto-select optimized backends
export VLLM_ROCM_USE_AITER=1
vllm serve <your-model> --tensor-parallel-size <tp>
```

With `VLLM_ROCM_USE_AITER=1`, vLLM automatically selects:

- `ROCM_AITER_FA` for MHA models (Llama, Qwen, Mistral)
- `ROCM_AITER_MLA` for MLA models (DeepSeek, Kimi)

### Selecting a Backend Explicitly

For advanced users who want to experiment, backends can be specified via `--attention-backend`:

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528 \
    --tensor-parallel-size 8 \
    --attention-backend ROCM_AITER_TRITON_MLA
```

Our benchmarks show `ROCM_AITER_TRITON_MLA` provides marginal improvement (1-4%) over `ROCM_AITER_MLA` in specific workloads. For most users, the auto-selected defaults work well.

### Hardware Support

| GPU    | Memory      | Status        |
| ------ | ----------- | ------------- |
| MI300X | 192GB HBM3  | ✅ Production |
| MI325X | 256GB HBM3e | ✅ Production |
| MI355X | 288GB HBM3e | ✅ Production |

### Complete Backend Reference

vLLM provides 7 attention backends on AMD ROCm, each optimized for different scenarios:

| Category | Backend                 | How to enable                                 | Notes                                     |
| :------- | :---------------------- | :-------------------------------------------- | :---------------------------------------- |
| MHA      | TRITON_ATTN             | `--attention-backend TRITON_ATTN`             | Baseline, Radeon support                  |
| MHA      | ROCM_AITER_UNIFIED_ATTN | `--attention-backend ROCM_AITER_UNIFIED_ATTN` | AITER unified kernel                      |
| MHA      | ROCM_ATTN               | `--attention-backend ROCM_ATTN`               | Legacy 2-path, Radeon support             |
| MHA      | **ROCM_AITER_FA**       | `--attention-backend ROCM_AITER_FA`           | **Recommended**, auto-selected with AITER |
| MLA      | TRITON_MLA              | `--attention-backend TRITON_MLA`              | Baseline, Radeon support                  |
| MLA      | **ROCM_AITER_MLA**      | `--attention-backend ROCM_AITER_MLA`          | **Recommended**, auto-selected with AITER |
| MLA      | ROCM_AITER_TRITON_MLA   | `--attention-backend ROCM_AITER_TRITON_MLA`   | 1-4% faster in some workloads             |

---

## Conclusion

The era of "just porting" is over. This post covered all 7 attention backends available on AMD ROCm in vLLM, with transparent benchmarks showing their trade-offs.

**Key Results (ISL=10K, OSL=1K benchmark):**

- `ROCM_AITER_FA`: **4.0-6.3x** faster than legacy ROCM_ATTN on MHA models
- `ROCM_AITER_MLA`: **1.2-1.5x** faster than TRITON_MLA on DeepSeek MLA via assembly decode kernel
- Performance scales across MI300X → MI325X → MI355X

**Our recommendation**: Simply use `export VLLM_ROCM_USE_AITER=1` and let vLLM auto-select the optimal backends. The defaults (`ROCM_AITER_FA` for MHA, `ROCM_AITER_MLA` for MLA) deliver excellent performance across all tested workloads.

This is what native AMD optimization looks like: not ported, purpose-built. The 3-path routing architecture reflects a deliberate design choice—explicit workload separation at the software layer, with each path calling hardware-optimized AITER primitives. The result is a system that's debuggable, portable across GPU generations, and ready for the mixed workloads of production LLM serving.

---

## Acknowledgements

We would like to thank the many talented people who have contributed to this collaborations:

**AMD**: Peng Sun, Hattie Wu, Yi Gan, Zejun Chen, Carlus Huang, Lingpeng Jin, and the AITER team.

**Embedded LLM**: Pin Siang Tan, Tun Jian Tan, and the Embedded LLM team.

## Resources

- [AITER Library (AMD)](https://github.com/ROCm/aiter)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3-235B Model](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8)
- [DeepSeek-R1 Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)

---
