---
layout: post
title: "MiniMax M3 in vLLM: Day-0 Serving for 1M-Token Multimodal Reasoning"
author: "vLLM Team"
summary: "How vLLM serves MiniMax M3 with MiniMax Sparse Attention, multimodal and reasoning parsers, MXFP8 weights, and long-context deployment recipes."
tags:
  - minimax
  - day-0-support
  - moe
  - long-context
---

<!--
Publication checklist:
- Confirm public model-card links for MiniMaxAI/MiniMax-M3 and MiniMaxAI/MiniMax-M3-MXFP8.
- Replace release-candidate benchmark snapshot with the approved public benchmark figure.
- Confirm whether NeMo-RL day-0 support should be mentioned. Current Slack evidence is a question, not a claim.
- Finalize public contributor and partner acknowledgments.
- Replace the hero descriptor with approved art.
- ~~Replace diagram placeholders with approved assets.~~ Done — 7 SVGs in assets/figures/minimax-m3/.
-->

We are excited to announce day-0 vLLM support for the MiniMax M3 family, including the BF16 and MXFP8 checkpoints at [`MiniMaxAI/MiniMax-M3`](https://huggingface.co/MiniMaxAI/MiniMax-M3) and [`MiniMaxAI/MiniMax-M3-MXFP8`](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8).

MiniMax M3 is built for the workloads that are becoming normal in production: million-token context, native multimodal reasoning, coding and agentic workflows, tool use, and controllable thinking behavior. The hard part is not only loading the model. It is making the new MiniMax Sparse Attention path, multimodal preprocessing, MXFP8 MoE execution, EAGLE3 speculative decoding, prefix caching, and deployment recipes work together in a serving engine that users can actually run.

This post walks through the model features, the vLLM implementation, the kernel and cache work behind the release, and the next optimizations we are landing after day 0.

![Figure 1: MiniMax M3 day-0 support brings long-context, multimodal, sparse-attention serving to vLLM.](/assets/figures/minimax-m3/hero-minimax-m3-vllm.svg)

## TL;DR

vLLM ships initial day-0 support for MiniMax M3:

- **Model family:** BF16 and MXFP8 MiniMax M3 checkpoints, with 1M-token context support subject to hardware capacity and deployment configuration.
- **Core architecture:** MiniMax Sparse Attention (MSA), a hybrid dense/sparse attention design that scores 128-token KV blocks, selects top blocks per query and KV group, and runs GQA attention over the selected blocks.
- **Serving stack:** `minimax_m3` tool and reasoning parsers, thinking-mode control, text-only and multimodal paths, TP/EP deployment, prefix caching, chunked prefill, EAGLE3 speculative decoding, and a Docker image available to use.
- **Speculative decoding:** Day-0 EAGLE3 support with the draft model released at [`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3).
- **Performance work:** MSA prefill and decode kernels, indexer-score and top-k kernels, fused QKNorm + RoPE + KV insert, GemmaNorm and quantization-path optimizations, and MXFP8 MoE backend integration.
- **Roadmap:** FP8 indexer/KV-cache work, TRTLLM-Gen MoE, broader disaggregated serving recipes, context-parallel long-prefill work, and further multimodal gateway optimization.

## MiniMax M3 Support Matrix

| Capability | What MiniMax M3 Adds | vLLM Support |
| --- | --- | --- |
| 1M-token context | Long-context text, code, agent traces, and document workloads | `--max-model-len` configuration, block-size 128 recipes, prefix caching, chunked prefill, MSA kernels |
| MiniMax Sparse Attention | Block-sparse GQA over selected 128-token KV blocks | Hybrid attention backend, indexer-score kernels, top-k block selection, sparse GQA prefill/decode |
| MXFP8 model weights | Efficient MoE serving for large-scale deployments | DeepGEMM MXFP8 MoE backend on Blackwell-class systems and Marlin MXFP8 on Hopper-class systems |
| Native multimodality | Image and video inputs alongside text | Model-specific multimodal preprocessing path and vLLM serving integration |
| Tool and reasoning outputs | Agentic workflows and controllable thinking | `minimax_m3` tool parser, `minimax_m3` reasoning parser, `thinking_mode` chat-template control |
| EAGLE3 speculative decoding | Draft-model acceleration for generation | Day-0 EAGLE3 recipe with [`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3) |

## Quickstart: Run MiniMax M3 with vLLM

For the MXFP8 checkpoint on a Blackwell-class node, the starting point is:

```bash
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --block-size 128 \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3
```

For BF16:

```bash
vllm serve MiniMaxAI/MiniMax-M3 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --block-size 128 \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3
```

On AMD ROCm (MI350X/MI355X, MI300X/MI325X, ROCm 7.2+), MSA runs on the Triton attention backend, so add `--attention-backend TRITON_ATTN`. On CDNA4 (MI350X/MI355X, gfx950) the MXFP8 variant fits at `--tensor-parallel-size 4`:

```bash
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --block-size 128 \
  --attention-backend TRITON_ATTN \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3
```

Use `--tensor-parallel-size 8` for lower latency or longer context, or run the BF16 checkpoint at TP=8.

The exact recipe depends on the target accelerator, model dtype, context length, traffic shape, and whether the deployment prioritizes throughput, latency, or maximum context capacity. Verification has been done on NVIDIA H200, GB200, and B300, and on AMD MI300X and MI355X. For the full set of NVIDIA and AMD launch recipes, deployment strategies, and tuning knobs, see the [vLLM recipe for MiniMax M3](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3).

### Deployment Knobs That Matter

MiniMax M3 has a few knobs that matter more than usual. `--block-size 128` aligns vLLM cache blocks with MSA's sparse block granularity. `--max-model-len` controls the advertised context length and KV capacity planning. `--tensor-parallel-size` and `--enable-expert-parallel` determine how attention, projections, and MoE experts are split across GPUs. The `minimax_m3` tool and reasoning parsers should be enabled for agent workloads, and long-context recipes should state whether prefix caching, chunked prefill, EAGLE3 speculative decoding, and multimodal preprocessing are enabled for that target.

### EAGLE3 Speculative Decoding

MiniMax M3 also has day-0 EAGLE3 speculative decoding support in vLLM. The draft model is released at [`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3), enabling deployments to use a draft-model path for lower generation latency when the workload and acceptance behavior fit the target traffic.

To enable EAGLE3, add a speculative decoding configuration to the serving command:

```bash
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --block-size 128 \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --speculative-config '{"method":"eagle3","model":"Inferact/MiniMax-M3-EAGLE3","num_speculative_tokens":3,"attention_backend":"FLASH_ATTN"}'
```

The example uses `num_speculative_tokens=3`, which is a conservative starting point for validation. Production recipes should tune this value against acceptance rate, TPOT, throughput, and target latency for the deployment's traffic mix.

### Thinking Mode

MiniMax M3 exposes controllable thinking behavior. In vLLM, pass the mode through `chat_template_kwargs`:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
model = client.models.list().data[0].id

messages = [{"role": "user", "content": "Explain MiniMax Sparse Attention."}]

for mode in ["enabled", "disabled", "adaptive"]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={
            "chat_template_kwargs": {
                "thinking_mode": mode,
            },
        },
    )
    print(mode, response.choices[0].message.content)
```

## Model Key Features and New Capabilities

MiniMax M3 matters for inference systems in three directions.

### 1M-Token Context with MiniMax Sparse Attention

The central architectural change is MiniMax Sparse Attention (MSA). Instead of letting every query attend densely over the full KV cache, MSA uses an index path to score KV blocks and select the most relevant blocks for the real attention computation. The default granularity is a 128-token KV block, and the selected blocks are shared across a GQA group.

In practical terms, every query token follows three steps:

1. Score candidate KV blocks with a small index head.
2. Select the top blocks, while applying the configured block rules.
3. Run online-softmax attention over only those selected KV blocks.

This preserves the long-context behavior users expect while bounding the amount of attention work per generated token. Practically, MiniMax Sparse Attention is the mechanism that makes MiniMax M3's 1M-token context practical for vLLM serving.

![Figure 2: MiniMax Sparse Attention keeps local and global context available while selecting sparse 128-token KV blocks from a 1M-token history.](/assets/figures/minimax-m3/msa-1m-context.svg)

### MSA Mechanics in More Detail

MSA separates two questions: which past blocks are worth reading, and how to run attention over those blocks. The index path answers the first question by scoring fixed 128-token KV blocks. The sparse GQA path answers the second by running attention over the selected blocks.

The selected set is not only learned top-k. The M3 config exposes `init_blocks` / `sparse_init_block` and `local_blocks` / `sparse_local_block`, but the current recipe uses `init_blocks=0` and `local_blocks=1`. In practice, the deterministic rule is the local-window block near the query token, while the remaining selected blocks come from indexer-scored top-k selection. Correctness depends on small details: partial final blocks must be masked, causal boundaries inside a block must be respected, local blocks that also rank in the top-k must not be counted twice, and batched requests can have different valid block ranges.

### Native Multimodality

MiniMax M3 is a multimodal model, not a text-only checkpoint with a separate sidecar. The serving path has to handle image and video inputs, preprocess them into patch tensors, preserve grid metadata, and hand the result to the model without stealing GPU time from generation.

For vLLM deployment, the release work includes model-specific multimodal preprocessing and parser support so users can run text-only, tool-use, reasoning, and multimodal workloads through the same serving surface.

### MXFP8 MoE Weights

The MXFP8 checkpoint is designed for efficient large-scale serving. Validation has used the DeepGEMM MXFP8 MoE backend for Blackwell-class systems, and Marlin MXFP8 for Hopper-class systems.

## vLLM Implementation

MiniMax M3 is a hybrid model: some layers route to dense attention, while sparse layers route to the MiniMax MSA backend. vLLM keeps that distinction behind the model and attention backend, so scheduler, cache allocation, batching, prefix caching, and serving continue to look familiar from the outside. For readers new to those internals, [Anatomy of vLLM](/blog/2025-09-05-anatomy-of-vllm) is a good companion to this section.

### MiniMax Sparse Attention Backend

The MSA backend has two distinct responsibilities.

First, it computes the sparse metadata. The indexer scores KV blocks, applies the configured block-selection rules, and emits top-k block IDs. For M3, selection is block-based: the unit of sparsity is the same page-like 128-token block that the cache manager already understands.

Second, it computes attention over those blocks. Prefill and decode have different shapes, so vLLM uses specialized kernels:

- **Prefill indexer-score:** Triton kernels compute block scores and top-k block selections.
- **Prefill sparse GQA:** Triton and the [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) CuTe/SM100 path support block-sparse GQA attention. The CuTe path inverts the query-to-block mapping into a K-major CSR form so KV blocks can be reused efficiently.
- **Decode indexer-score:** Split-style decode kernels scan candidate blocks, score them, and merge top-k results.
- **Decode sparse GQA:** GQA decode kernels consume the selected block pages and merge partial attention outputs.

### Prefill Execution

Prefill processes the prompt and creates the KV cache. For M3, prompt length and sparse metadata both matter. The path has four conceptual stages:

1. **Build query, key, value, and index projections.** Dense projections produce the representations needed by the indexer and attention kernels.
2. **Score blocks.** The index path computes a score for each candidate KV block. The scoring reduction can use block-level rules such as max or log-sum-exp, depending on the model configuration.
3. **Select blocks.** Top-k selection combines learned block scores with configured block rules, then emits block IDs for each query and KV group.
4. **Run sparse GQA.** The attention kernel reads only selected KV blocks and computes the same online-softmax attention result as a dense attention pass restricted to that selected set.

There are two useful schedules for the final sparse GQA work. A query-major schedule is straightforward: each query walks through its selected KV blocks. A KV-block-major schedule is better for long prompts when many queries select the same block. In that schedule, vLLM builds a K-to-Q mapping so one KV block can be loaded and reused across many queries before the output merge.

### Decode Execution

Decode has a different shape. Each step usually processes one new token per active sequence, but the batch can contain many sequences with different context lengths. The runtime updates cache state, scores candidate blocks, applies local-window handling, selects top blocks, runs sparse GQA decode, and merges partial outputs if the kernel uses split work. Because this happens every generated token, indexer-score and top-k kernels are part of TPOT, not just setup overhead.

M3's sparse-attention config controls block size, top-k count, optional init blocks, local-window blocks, index dimension, sparse layer IDs, score type, and layers where index attention is used for selection only. The key implementation rule is that every selected block ID must map back to the same logical request state that vLLM's scheduler and cache manager know about.

![Figure 3: vLLM routes dense layers through standard attention and sparse layers through the MiniMax MSA backend.](/assets/figures/minimax-m3/msa-backend-dispatch.svg?v=2)

### KV Cache Layout: Standard Storage, Sparse Computation

MiniMax M3 can store KV as ordinary paged KV and apply sparsity in the computation path. That lets vLLM keep the cache manager simple while adding the flexibility the kernels need:

- The main attention KV cache and indexer K cache are tracked explicitly.
- Prefix caching and chunked prefill can keep using stable cache blocks once the recipe's cache-state interactions are validated.
- Related disaggregated-serving and NIXL-style transfer paths can treat the cache as paged state while the attention backend handles sparse selection.

### Prefix Caching and Chunked Prefill

Prefix caching matters because M3 workloads often reuse long prompts: codebases, documents, multi-turn agent traces, and multimodal context. Chunked prefill matters because a 1M-token request should not monopolize the engine as one giant prefill. Together they are release-readiness stress tests: index cache state, main attention KV state, dense attention state, prefix hits, preemption, batching, and long-context chunk boundaries all need to agree on the same block tables before a recipe should be treated as production-ready.

### Multimodal and Parser Integration

MiniMax M3 includes model-specific parsing behavior for tools, reasoning, and multimodal input. vLLM support includes:

- `--tool-call-parser minimax_m3` for tool-call formatting.
- `--reasoning-parser minimax_m3` for reasoning output extraction.
- Chat template support for `thinking_mode`.
- Multimodal preprocessing integration for image and video inputs.

For production deployments, preprocessing is best handled before GPU execution whenever possible. The target architecture is a gateway that downloads media, decodes frames, samples video, resizes and normalizes images, creates patch tensors, and passes ready-to-run tensors to the worker.

This matters because multimodal requests can look small at the API boundary but large after preprocessing. One video can require frame sampling, per-frame resizing, patch generation, and metadata packing. Keeping CPU-heavy media work upstream makes GPU scheduling easier to reason about.

The parser side is equally important for agent traffic. Tool-call and reasoning parsers turn model-specific text conventions into structured API responses. Without the right parser, the model can generate useful text that is hard for an application to consume.

![Figure 4: For MiniMax M3, CPU-side image and video preprocessing should hand ready tensors to the vLLM worker so GPU time is reserved for inference.](/assets/figures/minimax-m3/multimodal-request-path.svg?v=2)

## Performance Optimizations

MiniMax M3 shifts the bottlenecks. MSA reduces dense attention work, but introduces indexer-score work, block selection, sparse metadata construction, and additional small kernels. The vLLM day-0 implementation focuses on keeping those new pieces cheap.

The guiding principle is simple: do not spend more time deciding which blocks to read than you save by not reading all blocks. That principle shows up in three places: block-major prefill, lean decode indexer-score kernels, and fusing small elementwise or cache-write kernels around the attention path.

### KV-Block-Major Prefill

During prefill, many query tokens can select the same KV block. A naive query-major sparse attention kernel would repeatedly move the same KV block from HBM to on-chip memory. The block-sparse structure gives us a better schedule: organize the work around KV blocks, then process all queries that need each block.

The [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) CuTe/SM100 path does this by building a K-to-Q CSR mapping, running a block-major sparse attention kernel, and using a log-sum-exp reduction to combine partial outputs. This improves arithmetic intensity for long prompts and agentic traffic where long cached contexts are common.

![Figure 5: KV-block-major prefill reuses selected KV blocks across queries, reducing redundant memory movement before the final LSE reduction.](/assets/figures/minimax-m3/kv-block-major-prefill.svg)

### Decode Indexer-Score Kernels

In decode, the indexer is on the critical path for every generated token. The engine must compare query-side index vectors against candidate key-side index vectors, reduce each 128-token block into a score, apply local-window handling, and keep only the top blocks for sparse GQA.

The optimized decode path uses specialized indexer-score kernels instead of treating the problem as a padded dense GEMM. This avoids adding extra work around ragged per-request block ranges and keeps the top-k boundary close to the score computation.

The decode path also has to be careful about memory traffic. Selected KV blocks are sparse in logical sequence space but still page-like in memory, so the kernel should avoid turning sparse pages into large temporary dense tensors unless reuse justifies it.

### Speculative Decoding in the Decode Kernels

EAGLE3 support also requires the MiniMax M3 decode kernels to handle speculative verification efficiently. In speculative decoding, one request can verify multiple draft tokens at once, so the MSA decode kernels cannot assume exactly one query token per request.

One fallback is to use prefill kernels for speculative verification, but that comes at a high cost: prefill kernels are usually tuned for much larger token counts, so they perform poorly on small draft-token batches. They are also usually not compatible with full CUDA graph mode, which is an important optimization for low-latency decoding.

The day-0 implementation updates the MSA decode indexer, top-k selection, and sparse GQA decode kernels to support a uniform `decode_query_len`. The kernels flatten speculative verification tokens in request-major order, then map each query token back to the correct request metadata, sequence length, block table, and causal position. This lets EAGLE3 verification use the decode-specialized split-K path instead of falling back to a less targeted prefill-style path, while keeping the speculative path close to the existing decode implementation.

The same path supports full CUDA graph coverage for uniform speculative decode batches. Kernel launch grids stay shape-stable, selected arguments avoid unnecessary Triton specialization, and padded request rows are handled explicitly so captured graphs can be replayed safely. These details matter because speculative decoding only improves TPOT when draft-token acceptance is not offset by extra kernel launches, recompiles, or cache-state overhead. We expect to keep optimizing this path across different draft lengths, concurrency levels, and traffic mixes.

### Kernel Fusions

Several smaller kernels were fused or routed through custom ops to reduce launch overhead and HBM round trips:

- **QKNorm + RoPE + KV insert:** combines normalization, position encoding, and cache write for the MSA path.
- **GemmaNorm and AllReduce + Norm work:** reduces overhead around normalization in tensor-parallel execution.
- **Quantization-path cleanup:** improves `silu_mul_quant_fp8` and related MXFP8/MoE input paths.
- **Router and MoE kernels:** reduce overhead in the sparse expert path and prepare for deeper TRTLLM-Gen integration.

The release path is intentionally conservative: correctness and stable cache behavior win over enabling every possible graph or fusion knob on day 0. More aggressive fusions can land as the public recipes mature.

### Quantization and KV Cache Dtype

The MXFP8 checkpoint primarily changes weight and MoE execution, not the conceptual structure of the KV cache. Public recipes should state model dtype, MoE backend, and KV-cache policy separately: "MXFP8 model" does not automatically mean every cache and intermediate tensor is MXFP8. The roadmap includes FP8 indexer and KV-cache paths because KV capacity directly controls how much long-context and batched traffic a deployment can serve.

### CUDA Graphs and Compile Behavior

CUDA graphs are valuable for decode because M3 introduces several small operations around each token step. But graph capture only helps when the captured path is stable across batch shapes, cache states, and sparse metadata. The release-candidate path uses conservative graph settings where needed, then expands coverage as validation catches up.

## Release-Candidate Validation

Before the public release, the vLLM team ran daily release-candidate validation across accuracy, throughput, speculative decoding, and container usability.

The validation loop had three goals:

1. **Functional correctness:** the model loads, serves requests, parses tool and reasoning outputs, and handles text-only plus multimodal inputs.
2. **Accuracy parity:** benchmark results stay aligned with expected model behavior after kernel, cache, parser, and recipe changes.
3. **Serving readiness:** container images run with the intended TP/EP/speculative-decoding settings on target accelerators.

The most useful tests combine short correctness tasks with long-output and long-context workloads. Short tasks catch parser, formatting, and obvious numerical issues quickly. Long-context tasks catch MSA metadata, prefix caching, chunked prefill, and KV-cache layout problems. Speculative decoding tests catch acceptance regressions that may not show up in ordinary accuracy runs.

A June 11 release-candidate snapshot on B300:

| Dimension | Result |
| --- | ---: |
| GSM8K strict / flexible accuracy | 91.51% / 91.66% |
| ShareGPT @256 throughput | 8,530 tok/s |
| ShareGPT @256 TPOT | 56.0 ms |
| Speculative Sonnet TPOT, concurrency 1 / 16 / 64 | 4.51 / 9.04 / 14.36 ms |
| Speculative acceptance on Sonnet | ~67%, mean accept length ~3.0 |

These numbers are useful for engineering validation, not a definitive benchmark ranking. Final public measurements should be rerun on locked images, public weights, public recipes, and clearly described hardware.

![Figure 6: Release-candidate validation checks accuracy, throughput, and speculative decoding before public MiniMax M3 recipes are published.](/assets/figures/minimax-m3/validation-dashboard.svg)

## Roadmap: The Path Ahead

The day-0 implementation is the starting line. The next pieces of work are already in flight:

- **FP8 indexer and KV-cache paths:** reduce KV-cache memory pressure and increase batch capacity while preserving sparse-attention accuracy.
- **TRTLLM-Gen MoE:** improve Blackwell performance for MXFP8 expert execution.
- **Context parallelism:** improve very-long-context prefill scaling when one node is not enough.
- **Disaggregated serving:** expand NIXL and prefill/decode disaggregation recipes for M3 traffic, building on the directions in [Large-Scale Serving with vLLM](/blog/2025-12-17-large-scale-serving).
- **Kernel fusion:** reduce the many small indexer, top-k, quantization, and normalization kernels that MSA introduces.
- **Multimodal gateway path:** keep image and video preprocessing out of the critical GPU generation loop.

## MiniMax M3 vLLM FAQ

### Does vLLM support MiniMax M3?

Yes. This draft covers day-0 vLLM support for the MiniMax M3 BF16 and MXFP8 checkpoints, including MSA attention, model-specific parsers, EAGLE3 speculative decoding, multimodal preprocessing, TP/EP serving recipes, and a Docker image available to use.

### What is MiniMax Sparse Attention?

MiniMax Sparse Attention scores fixed 128-token KV blocks, selects the most relevant blocks for each query and GQA group, applies the configured local-window rule, and runs sparse GQA over that selected set. In the current M3 recipe, that corresponds to `init_blocks=0` and `local_blocks=1`.

### Does MXFP8 mean the KV cache is MXFP8?

No. MXFP8 describes the model weight and MoE execution path. KV-cache dtype is a separate serving decision; the current sparse-attention validation treats native KV storage and quantized KV-cache support as separate roadmap work.

### What settings matter most for 1M-token context?

The important starting points are `--block-size 128`, enough GPU memory for the chosen batch and context shape, and a recipe that states whether prefix caching, chunked prefill, and EAGLE3 speculative decoding are enabled. By default vLLM reads the context length from the model config, so you do not need to set `--max-model-len`. If you have limited GPU memory or do not need the full 1M-token window, you can pass `--max-model-len` to cap it lower and reduce KV-cache pressure.

## Acknowledgments

Thank you to the teams at MiniMax, NVIDIA, AMD, Inferact and the vLLM community for helping bring up day-0 MiniMax M3 support!

## Related vLLM Reading

MiniMax M3 builds on several areas of vLLM:

- [Anatomy of vLLM](/blog/2025-09-05-anatomy-of-vllm) for scheduler, KV cache, prefix caching, and distributed execution background.
- [Speculative Decoding in vLLM](/blog/2024-10-17-spec-decode) and [P-EAGLE](/blog/2026-03-13-p-eagle) for the draft-model path.
- [Large-Scale Serving with vLLM](/blog/2025-12-17-large-scale-serving), [KV Offloading Connector](/blog/2026-01-08-kv-offloading-connector), and [Moriio KV Connector](/blog/2026-04-07-moriio-kv-connector) for prefix reuse, KV movement, and disaggregated serving.
