---
layout: post
title: "Efficient Decode Context Parallelism with vLLM for Long Context Workloads"
date: 2026-07-27
author: "Seonghee Lee, Sungsoo Ha, Omri Almog (NVIDIA)"
summary: "Decode Context Parallelism (DCP) in vLLM shards KV cache across GPUs by sequence dimension, enabling 3× higher throughput on long-context agentic workloads compared to standard tensor parallelism."
image: /assets/figures/2026-07-27-decode-context-parallelism/figure-1.png
social_image: /assets/figures/2026-07-27-decode-context-parallelism/figure-1.png
tags:
  - performance
  - attention
  - parallelism
---

## 1. Introduction

Long-context inference is becoming essential for agentic AI, where assistants may need to reason over large code repositories and long chat histories. Agent-trace benchmarks now run from 64K all the way to 1M tokens and their KV caches are correspondingly large. Under a baseline tensor-parallel (TP) setup, once TP exceeds the number of KV heads, this KV cache is duplicated across GPUs and eats into GPU memory, leaving very little room to serve additional requests. This caps the number of concurrent requests the system can handle, driving down throughput and pushing up cost per token.

Decode Context Parallelism addresses this by splitting KV cache across the GPUs so each GPU stores and reads only part of the KV cache. This frees up GPU memory, allowing each GPU to take on more requests and thus run at a larger batch size. On systems with high-bandwidth GPU-to-GPU interconnects, this helps preserve interactive responsiveness while serving many long-context agents at once.

## 2. Performance Results

To quantify the benefit of Decode Context Parallelism, we compared a baseline tensor-parallel deployment against DCP on an identical set of GPUs, holding the model, hardware, and workload fixed and varying only how the KV cache is sharded during decode.

<p align="center">
<img src="/assets/figures/2026-07-27-decode-context-parallelism/figure-1.png" alt="Throughput comparison figure 1" width="100%">
</p>

<p align="center">
<img src="/assets/figures/2026-07-27-decode-context-parallelism/figure-2.png" alt="Throughput comparison figure 2" width="100%">
</p>

### 2.1 Dataset

The dataset is an [agentic long-context trace from the Dynamo Kimi K2.6 performance recipes](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl). It's an agentic multi-turn workload of long inputs paired with short generations, chosen to reflect realistic long-horizon agent behavior. Inputs are centered around a median of ~67k tokens and paired with short ~400-token outputs, but the input distribution is bimodal rather than uniformly huge: roughly half the requests sit at 64k+ (≈53%, with a heavy tail reaching ~1M tokens) and half are short-to-mid (≈47% under 64k, ~18% under 8k). About 8% of requests exceed 128k and ~3–4% exceed 256k.

### 2.2 Benefits of Decode Context Parallelism

We ran an experiment on a single 8×B200 node serving Kimi K2.6 in NVFP4 with vLLM, sweeping request concurrency from 16 to 512 (see table below). DCP sustains far higher concurrency and delivers markedly higher throughput per GPU across the entire throughput–interactivity Pareto frontier.

<p align="center">
<img src="/assets/figures/2026-07-27-decode-context-parallelism/figure-3.png" alt="DCP vs TP throughput benchmark" width="100%">
</p>

The difference comes down to where the KV cache lives. Baseline TP replicates the KV cache on every GPU, so peak memory fills quickly. It reaches 100% at a concurrency of 64 and hits a wall, and throughput plateaus near 1,863 tok/s/GPU because no additional requests can fit. On the other hand, DCP shards the KV cache along the sequence dimension, so each GPU stores only 1/N of every request's KV. This allows space on the GPU to support more incoming requests. As a result, even at high concurrencies DCP keeps scaling where TP hits a wall. DCP reaches 6,091 tok/s/GPU at c512 while still sitting at just 82% KV usage. **The core value of DCP is that it sustains far higher concurrency, even on long-context runs, precisely the regime where replicated-KV TP runs out of memory first.**

### 2.3 Comparison by Sequence Length

<p align="center">
<img src="/assets/figures/2026-07-27-decode-context-parallelism/figure-4.png" alt="DCP Pareto frontier across full sequence-length bands" width="100%">
</p>

We also plotted performance against full sequence length (input + output). The figure shows a single throughput–interactivity Pareto frontier with requests grouped into five length bands (&lt;32k, 32–64k, 64–128k, 128–200k, and 200k+) so we can see how performance shifts with context length. **DCP keeps a high, stable frontier even in the 200k+ range**, with the curves for short and long buckets nearly overlapping: throughput scales with concurrency while per-user speed stays usable at the long context lengths where the replicated-KV baseline runs out of memory and cannot scale.

## 3. Challenges of Serving Long Contexts

Under tensor parallelism, the KV cache is partitioned **by the attention head**. Each KV head owns its own separate K and V tensors, and the head is the smallest unit you can hand to a GPU. A standard TP has no mechanism to slice a single head's KV cache. So if you have K KV heads, you can give each GPU a distinct subset of those heads, but only down to the point where every GPU holds one head. Once TP goes beyond K, there aren't enough distinct heads to go around, so two or more GPUs end up holding a copy of the same head's KV cache instead of a unique slice.

## 4. What is DCP?

Unlike pure TP methods, DCP is able to split KV cache across GPUs by sequence (context) dimension. Each GPU is made responsible for the KV cache of a chunk of *token positions* from the same sequence. For a single 200K-token request, GPU 0 might hold the cache for tokens 0–50K, GPU 1 for tokens 50K–100K, GPU 2 for 100K–150K, and GPU 3 for 150K–200K. By sharding KV cache, the KV cache footprint per GPU keeps shrinking as you add GPUs, freeing the memory that lets you raise the batch size and serve higher concurrencies.

<p align="center">
<img src="/assets/figures/2026-07-27-decode-context-parallelism/dcp_vs_tp_diagram.svg" alt="DCP sequence sharding diagram" width="100%">
</p>

### 4.1 Decode Context Parallelism Process

Standard Decode Context Parallelism keeps the communication pattern simple, following the rhythm **AllGather Q → Compute → AllGather + ReduceScatter**.

- **AllGather Q:** Each GPU has computed only a fragment of the query, but attention requires the full query vector to score against any key. An all-gather across the DCP group assembles a complete copy of the query on every GPU. This is cheap during decode because the query is a single token.

- **Compute:** Each GPU runs attention between the gathered query and its *local* slice of the KV cache. In vLLM this is `k_up` for MLA or `tensor_broadcast` for GQA.

- **AllGather + ReduceScatter (`cp_lse_ag_out_rs`):** The partial results are combined into the true output. AllGather shares each GPU's partial output and LSE; the LSE values reweight and merge the partials (the online-softmax trick), and ReduceScatter sums them while handing each GPU back only its own head-slice.

## 5. vLLM Usage

DCP is enabled with a single extra argument, `decode_context_parallel_size`, alongside your existing tensor-parallel setting.

### 5.1 Offline

```python
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="deepseek-ai/DeepSeek-V2-Lite",
    tensor_parallel_size=2,
    decode_context_parallel_size=2,
)
outputs = llm.generate(prompts, sampling_params)
```

### 5.2 Online

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite \
    --tensor-parallel-size 2 \
    --decode-context-parallel-size 2
```

### 5.3 MLA Backend

**Models:** DeepSeek-V2 / V3 / R1, Kimi K2.6 models using Multi-head Latent Attention.

**Why it's different.** MLA compresses the Key/Value into a single low-rank *latent* vector that is shared across all query heads — effectively one KV "head." Under pure tensor parallelism there's nothing to split by head, so that latent KV cache is replicated in full on *every* TP rank. TP does nothing to shrink it, which makes MLA the ideal candidate for DCP: the whole cache is redundant, so the whole cache can be sequence-split.

**What they do.** DCP splits the latent KV cache along the sequence dimension, so each rank stores only its chunk of the latent; at attention time each rank up-projects its latent slice (the `k_up` step) to reconstruct the Keys/Values it needs. Because the effective KV-head count is 1, the sequence can be split up to the full TP degree — hence the constraints:

- `tensor_parallel_size >= decode_context_parallel_size`
- `tensor_parallel_size % decode_context_parallel_size == 0`

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 8
```

### 5.4 GQA Backend

**Example models:** Qwen3-235B, and other Grouped-Query-Attention models (Llama-family, etc.).

**Why it's different.** GQA stores `num_key_value_heads` KV heads, and TP splits the KV cache by those heads first. That works cleanly only up to `num_key_value_heads`; once `tensor_parallel_size` exceeds it, the KV cache begins duplicating, with `tp // num_key_value_heads` identical copies across ranks.

**What they do.** DCP takes those would-be-duplicate copies and fills them with *different* sequence chunks instead, while the shared KV heads are broadcast across their query heads (the "tensor broadcast for GQA" step). So the sequence-split degree is capped by the duplication factor `tp // num_key_value_heads`:

- `(tensor_parallel_size // num_key_value_heads) >= decode_context_parallel_size`
- `(tensor_parallel_size // num_key_value_heads) % decode_context_parallel_size == 0`

```python
# Qwen3-235B has num_key_value_heads = 4; tp=8 gives 8//4 = 2 redundant copies,
# so dcp can be up to 2.
vllm serve Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 2
```

## 6. Future Work

Looking ahead, we plan to extend DCP along three main directions. We will add support for finer-grained parallelism sizes for both TP and DCP, giving users more precise control over their parallelism layout and reclaiming efficiency lost to over-provisioned sharding. We are also developing better DCP all-to-all (A2A) communication kernels for both multinode and single-node settings, reducing exposed communication and improving overlap with compute as context length and device count grow. Finally, we aim to broaden DCP's reach by extending support to a wider variety of backends and integrating it with speculative decoding, hybrid models, and Dynamic Chunked Pipeline Parallelism, so a much wider range of workloads can benefit from context-parallel efficiency gains.

## 7. Conclusion

Decode Context Parallelism represents a fundamental rethinking of how GPUs are organized for long-context inference. Rather than forcing GPUs to duplicate KV cache or sit underutilized, DCP puts every GPU to work: sharding the sequence during attention, then immediately reconfiguring those same GPUs to amortize FFN weight loading across the full pool. The result is a system that scales gracefully with context length rather than degrading under it.

With native support in vLLM, Decode Context Parallelism is ready to power the next generation of long-context agentic applications — from document reasoning to multi-session agentic pipelines — at the throughput and latency that production demands. It joins a broader industry move toward Decode Context Parallelism, a direction [NVIDIA has also pursued with Helix Parallelism](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog22_Helix_Parallelism_Scaling_Multi_Million_Token_Decoding_with_KV_Cache_Sharding.md) in TensorRT-LLM.

## About Us

This work was completed by engineers at [NVIDIA](https://www.nvidia.com/). We build and validate high-throughput LLM inference on NVIDIA GPUs — from long-context agentic serving to parallelism strategies like Decode Context Parallelism — and work closely with the vLLM community so that these capabilities land upstream and are usable in production. The DCP results in this post were measured on NVIDIA B200 GPUs with Kimi K2.6 in NVFP4, and the recipes can be reproduced with current vLLM releases that support `--decode-context-parallel-size`.

Special thanks to Anahita Bhiwandiwalla, Xin Li, Pavani Majety, Nidhi Bhatia, Roman Ageev, Pen Chung Li, and Chris Hoge for their reviews, benchmarking support, and engineering input throughout this study.

We also thank the vLLM community, whose open-source engine and continued collaboration made this benchmarking effort possible.
