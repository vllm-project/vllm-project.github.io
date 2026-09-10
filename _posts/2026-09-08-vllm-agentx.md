---
layout: post
title: "vLLM x AgentX: Optimizing for Real-World Agentic Serving"
author: "vLLM Team and Inferact"
date: 2026-09-08 23:00:00 +0000
summary: "How vLLM optimizes KV cache management, parallelism, scheduling, and P/D disaggregation for agentic workloads, validated on SemiAnalysis AgentX with up to 130K tokens per GPU-second and a 14.6x-106x serving-cost advantage over Opus 5."
image: /assets/figures/2026-09-08-vllm-agentx/hero-vllm-agentx.png
social_image: /assets/figures/2026-09-08-vllm-agentx/hero-vllm-agentx.png
tags:
  - agentic
  - kv_cache
  - parallelism
  - large-scale-serving
  - disaggregation
  - performance
---

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/hero-vllm-agentx.png" alt="vLLM x AgentX: Optimizing vLLM for Real-World Agentic Serving" width="100%">
</p>

**TL;DR:** Agentic workloads are becoming a major source of vLLM traffic. Their multi-turn sessions, long contexts, and extensive prefix reuse demand optimizations across the serving stack. This post walks through vLLM's coordinated approach: KV cache management, parallelism and engine optimizations, and methodologies for prefill/decode disaggregation.

Measured on [AgentX](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat), SemiAnalysis's public agentic benchmark, vLLM achieves up to 130K total tokens per GPU-second on DeepSeek V4 Pro, and an interactivity of up to 376 tokens per second on MiniMax M3. Across DeepSeek V4 Pro, MiniMax M3, and Kimi K3, vLLM delivers a 14.6×–106× serving-cost advantage over Opus 5 API pricing (see [Performance](#performance-agentic-first-and-openly-verifiable)).

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/agentx-pareto-summary.png" alt="vLLM on SemiAnalysis AgentX: cost efficiency vs. P90 interactivity for DeepSeek V4 Pro, MiniMax M3, and Kimi K3" width="100%">
<br>
<em>Figure 1: vLLM on SemiAnalysis AgentX. Total tokens per &#36;1 of TCO against P90 interactivity for the best vLLM configuration of DeepSeek V4 Pro, MiniMax M3, and Kimi K3, with DeepSeek V4 Pro on GB300 NVL72 as a case study. Data source: <a href="https://inferencex.semianalysis.com/inference">SemiAnalysis AgentX</a>.</em>
</p>

<iframe class="vllm-embed" src="/assets/interactive_pages/vllm-agentx-pareto.html" title="vLLM on AgentX: cost efficiency vs. interactivity (interactive)" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 640px; border: 0; border-radius: 12px; overflow: hidden;"></iframe>

<p align="center">
<em>Interactive version of Figure 1. Hover over a point to see its configuration, or <a href="/assets/interactive_pages/vllm-agentx-pareto.html">open it full-screen</a>.</em>
</p>

## Characterizing agentic workloads: a second look

Since our first post on [serving agentic workloads](https://vllm.ai/blog/2026-05-06-mooncake-store) in May, the share of agentic traffic has continued to grow. As of June 2026, [OpenAI reported](https://openai.com/signals/enterprise-data/) that Codex generated 64% of combined Codex and ChatGPT output tokens among enterprise customers.

This growing token consumption stresses serving infrastructure along two axes: cost and latency. Cost efficiency determines how many concurrent agents fit in a fixed hardware budget; latency determines how quickly each agent progresses through its reasoning and tool-use cycles. Optimizing agentic serving therefore means improving the latency-cost frontier as a whole.

To evaluate that frontier under representative traffic, SemiAnalysis recently released [AgentX](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat), a public benchmark built from real-world agentic coding traces. These traces provide a concrete view of the workload characteristics that serving systems must accommodate:

- **Long-running, multi-turn sessions.** Median 43 turns per session.
- **Long contexts with short outputs.** Median input 142K tokens, median output 444 tokens.
- **Extensive prefix reuse.** Prefix-cache hit rate above 96%.
- **Subagent-heavy traffic.** 44% of sessions contain at least one subagent, with a median of four subagent rollouts among those sessions.

These statistics follow from how an agentic session is built. Each turn appends the latest tool result to the accumulated context and sends the whole thing back to the model, so the input keeps growing while each turn adds only a short new prefill, and almost all of the request is a prefix the engine has already seen. Subagents either fork from that context or start fresh, and their results are joined back into the parent before the final answer. Figure 2 walks through one such session: use the slider to step from the first turn to the final answer and see how much of each request is reused prefix versus new prefill.

<iframe class="vllm-embed" src="/assets/interactive_pages/agentic-workload-explorer.html" title="Agentic workload explorer: sessions grow through reuse and branching" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 1000px; border: 0; border-radius: 12px; overflow: hidden;"></iframe>

<p align="center">
<em>Figure 2: Agentic sessions accumulate context across turns and branch into subagents. Each request carries earlier context forward, while subagents may inherit the parent context or start fresh. Step through the trace with the slider, or <a href="/assets/interactive_pages/agentic-workload-explorer.html">open the explorer full-screen</a>.</em>
</p>

## Challenges in serving agentic workloads

These workload characteristics create three challenges for efficient serving.

1. **Prefix cache pressure**. Every turn of a multi-turn session replays the full conversation so far. To keep many sessions running at once, the engine has to offload KV caches between turns. This becomes harder at scale, where KV cache management, prefix caching, and offloading must work efficiently across GPUs, prefill/decode disaggregated instances, and replicas.
2. **Execution efficiency**. Agentic workloads feature long contexts and tight latency requirements, so the engine has to process more tokens and do more work per token in less time. This requires adapting parallelism, kernels, scheduling, speculative decoding, and other engine optimizations to the new request shape.
3. **Finding the right P/D ratio**. Context lengths and cache hit rates vary wildly across sessions and subagents, and routing must efficiently balance cache affinity and load across ranks. These factors make it difficult to find the throughput-optimal P/D ratio, which also shifts with concurrency.

## The vLLM approach: optimizations across the stack

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/full-stack-overview.png" alt="Full-stack optimization for agentic serving: control plane, execution plane, and data plane" width="100%">
<br>
<em>Figure 3: Full-stack optimization for agentic serving. The data plane manages a distributed shared KV cache, the execution plane maps each model to appropriate parallelism and kernels, and the control plane coordinates proper P/D ratio and request scheduling.</em>
</p>

Figure 3 summarizes the three planes. The rest of this section walks through them, starting from the data plane.

### Data plane: keep KV caches warm and close to compute

#### Hybrid KV cache management: a foundation that keeps evolving

KV cache management has been central to vLLM since PagedAttention, and agentic workloads with long contexts put heavier pressure on KV cache capacity. Modern hybrid models complicate allocation further by combining sliding-window and linear attention with full attention, whose cached blocks differ in size and lifetime.

vLLM's hybrid KV cache manager tackles this complexity with a simple core idea: a uniform memory page as the basic allocation unit, managed through one shared block pool (Figure 4).

A shared pool lets vLLM reallocate memory dynamically on demand instead of statically partitioning capacity by attention type. This matters because full-attention KV grows with sequence length, while sliding-window and recurrent state follow different lifetimes and scaling rules. The best partition therefore changes with concurrency, context length, and prefix-reuse patterns.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/hybrid-kv-cache-manager.png" alt="vLLM's hybrid KV cache manager: one page size for all attention types, one shared block pool" width="100%">
<br>
<em>Figure 4: vLLM's hybrid KV cache manager. One page size serves all attention types, and a single block pool is shared between them.</em>
</p>

The abstraction continues to evolve as new architectures expose fragmentation and transfer inefficiencies. For example, [DeepSeek V4](https://vllm.ai/blog/2026-04-24-deepseek-v4)'s initial KV cache layout fragmented the different cache types into three size buckets and allocated 92 separate tensors. As Figure 5 shows, this fragmentation wastes memory on padding and is inefficient for P/D transfer and KV cache offloading.

The new [packed KV cache layout](https://github.com/vllm-project/vllm/pull/44577) instead stores all cache groups and layers in one contiguous backing allocation per block rather than 92 fragmented ones. This reduces descriptor and P/D transfer overhead, and also permits a smaller allocation unit when the FP4 indexer is enabled, saving [roughly 10% of KV cache memory](https://github.com/vllm-project/vllm/pull/48993).

<iframe class="vllm-embed" src="/assets/interactive_pages/dsv4-kv-cache-layout.html" title="DeepSeek V4 Pro hybrid KV cache: size-bucketed tensors vs. packed layout" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 700px; border: 0; border-radius: 12px; overflow: hidden;"></iframe>

<p align="center">
<em>Figure 5: Hybrid KV cache groups and the packed KV cache layout for DeepSeek V4 Pro. Toggle between the MXFP4 and FP8 indexer configurations, or <a href="/assets/interactive_pages/dsv4-kv-cache-layout.html">open the layout full-screen</a>.</em>
</p>

#### Hierarchical KV cache offloading: distributed KV cache pool with smart retention policies

To preserve prefix caches beyond GPU memory capacity and across each engine, vLLM has integrated [Mooncake Store](https://github.com/kvcache-ai/Mooncake) as a distributed KV cache pool, with the design covered in [our previous blog](https://vllm.ai/blog/2026-05-06-mooncake-store). Adoption has grown steadily since, and we keep shipping new features and performance improvements for capacity, efficiency, and retention on agentic workloads.

**Model architecture parity.** KV cache offloading remains a first-class citizen in vLLM, with full support for new model architectures including sparse attention, compressed attention, and linear attention. This is done while keeping other engine features fully functional and performant, including asynchronous scheduling, P/D disaggregation, speculative decoding, and parallelism.

**Hierarchical KV cache offloading.** vLLM supports hierarchical tiers for the distributed KV cache pool to further extend capacity with disks and extra CPU-only nodes. This is achieved with vLLM's [Mooncake Store](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/#configure-mooncake) `standalone-store` mode, which makes an external Mooncake client own the CPU pool and disk tier, and turns vLLM workers into pure requesters. By launching a standalone Mooncake client on each node, we can freely expand the KV cache pool with CPU memory and disks. We have also integrated the distributed shared KV cache pool with routers such as [Dynamo](https://github.com/ai-dynamo/dynamo) and [llm-d](https://github.com/llm-d/llm-d), which simplifies the routing policy because requests can get cache hits on any instance.

**Performance optimizations.** Hybrid models must construct keys and perform lookups separately for each attention type, which multiplies CPU overhead. We reduced this cost through more efficient data structures, asynchronous lookups, work moved off the scheduler's critical path, and parallel send and receive operations. Implementation details are in [PR#46188](https://github.com/vllm-project/vllm/pull/46188/changes), [PR#45444](https://github.com/vllm-project/vllm/pull/45444/changes), [PR#45659](https://github.com/vllm-project/vllm/pull/45659/changes), and [PR#47317](https://github.com/vllm-project/vllm/pull/47317/changes).

**Session-aware prefix-cache retention.** For hybrid models with linear or sliding-window layers alongside full attention, prefix reuse requires preserving the linear state or sliding-window cache at the reuse boundary. Keeping these snapshots at every token is expensive, so we combine two complementary policies:

1. [**Interval-based retention**](https://github.com/vllm-project/vllm/pull/43447) automatically preserves prompt-end caches/linear states at each turn. Subsequent turns and forked subagents, which typically replay and extend an earlier turn's context, can then reuse the cached context.

   However, shared prefixes typically end within a turn, so interval-based retention may not preserve a checkpoint. To capture this reuse, we introduce a second policy:

2. [**Marconi-style selective retention**](https://github.com/vllm-project/vllm/pull/47782) retains a checkpoint when a prefix is observed a second time. When a request encounters a previously observed prefix without a retained checkpoint, vLLM recomputes the missing state and saves a checkpoint at that boundary. Subsequent requests sharing the prefix can then reuse it.

Together, these policies preserve a high cache hit rate without excessive storage overhead on large-scale agentic workloads. Our [vLLM Kimi K3 blog](https://vllm.ai/blog/2026-07-27-k3) explains the technical details in depth.

### Execution plane: generate tokens fast

#### Model-specific parallelism

Modern inference systems expose several axes of parallelism, such as tensor parallelism (TP), data parallelism (DP), expert parallelism (EP), pipeline parallelism (PP), and context parallelism (CP).

The optimal parallelism, however, depends on the model architecture, hardware topology, workload patterns, and latency SLOs. In this section, we examine two representative models on NVIDIA GB-series and B-series GPUs and their AMD counterparts, and discuss our optimizations and findings.

**Kimi K3**

Kimi K3 features multi-head latent attention (MLA) and Kimi Delta Attention (KDA). Since MLA compresses KV into a single latent space with one head, plain tensor parallelism (TP), which replicates that latent cache across ranks, is not very efficient.

As an alternative to TP, we have found strong performance gains from [decode context parallelism (DCP)](https://vllm.ai/blog/2026-08-07-decode-context-parallelism), which shards the cache along the sequence dimension, leaving each rank with 1/N of the KV state. Specifically, DCP offers two benefits for agentic workloads (Figure 6):

* **Lower decode latency**. MLA attention is memory-bound, and its cost grows with context length. As agentic prefixes grow, attention becomes a larger share of each decode step, and sharding it across ranks shortens that step.
* **Higher throughput and KV capacity**. Avoiding KV cache replication lets the engine keep more sequences in flight without stalling on KV admission, and hence achieve higher throughput.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/k3-tp8-vs-dcp8.png" alt="Kimi K3 decode: TP8 vs. DCP8, P50 TPOT vs. sequences per rank" width="90%">
<br>
<em>Figure 6: For Kimi K3, DCP8 achieves lower decode latency than TP8 and scales to higher concurrency.</em>
</p>

DCP's tradeoff is extra communication: the KV cache is sharded by sequence, so every MLA decode layer needs a query gather before attention and a partial-output reduction after it.

We carefully optimized the DCP compute path to bypass NCCL operations and avoid these overheads. We use symmetric-memory buffers that peer GPUs can load from and store to directly. Queries are multicast straight into the buffers consumed by the attention kernels. Each GPU then writes its partial attention outputs and log-sum-exp (LSE) statistics directly into its peers' receive slots, where each rank locally merges the results with online softmax. These GPU-to-GPU writes are fused with the computation into the same kernels (Figure 7), cutting latency by about 13% per layer compared with the default DCP8 implementation.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/k3-dcp-symmem.gif" alt="Animation: MLA decode path under DCP4 using symmetric memory" width="80%">
<br>
<em>Figure 7: MLA decode path under DCP4 using symmetric memory. Each step is fused into a single kernel, replacing the NCCL all-gather, staging copy, all-to-all, and unpack steps.</em>
</p>

A larger scale-up domain can change the best strategy. On an NVL72-class system, for example, wide EP with data parallelism (DEP) can scale better than DCP and deliver higher throughput at the same decode latency SLO (Figure 8). At larger, multi-node DCP sizes, the communication cost of sharded attention outweighs the compute it saves. DEP assigns requests and their KV caches to different data-parallel ranks, avoiding DCP's attention collectives while sharding the MoE experts across ranks.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/k3-dcp8-vs-dep16.png" alt="Kimi K3 decode: DCP8 vs. DEP16, P50 TPOT vs. sequences per rank" width="90%">
<br>
<em>Figure 8: For Kimi K3, wide EP (DEP16) scales better than DCP8 once the per-rank batch size exceeds 3.</em>
</p>

**DeepSeek V4**

DeepSeek V4 also has MLA-style KV caches that replicate under TP, leading to inefficient memory use. In addition, its compressed sparse attention makes TP head sharding compute-inefficient for three reasons:

* The compressor paths produce only one shared KV representation per compressed position rather than independent per-head states. TP therefore cannot shard compute along the KV-head dimension, and every rank repeats the compressor work.
* The indexer, while having 64 heads, produces only one global top-k selection per token. The current TP path therefore replicates the full indexer on every rank, avoiding a dense score reduction before top-k but duplicating the work.
* Sparse MLA is dominated by scanning and gathering top-k KV cache entries, not by attention arithmetic. TP repeats much of this memory-bound work on every rank while dividing only the cheaper head-wise computation.

In practice, prefill context parallelism (PCP) performs best for long prefills, while data and expert parallelism (DEP) works well across a broader range of serving conditions.

PCP shards the prompt sequence (the query tensor), distributing compressor and indexer work across ranks, while giving sparse MLA a wider, more efficient head-local shape. For a 32K prompt, PCP8 achieves a 2.65× prefill speedup over TP8, substantially reducing TTFT. However, it still replicates decode-side state across ranks, so it is best suited to dedicated prefill workers.

DCP is less effective for DeepSeek V4 than for Kimi K3 because of V4's more complex model architecture (see [the bitter lessons](#decode-context-parallelism-dcp-does-not-transfer-cleanly-to-deepseek-v4)).

DEP instead distributes requests and decoded tokens across data-parallel ranks and keeps the attention path completely local. This makes DEP our default for most DeepSeek V4 configurations.

#### Scheduling mixed agentic traffic at two levels

Agentic serving mixes frequent append-only requests, which reuse long prefixes and need only short prefills, with occasional long fresh prefills spanning tens of thousands of tokens. This creates two scheduling problems: within an instance, a long prefill can block short interactive turns; across DEP ranks, uneven prefill placement creates load imbalance. We address them with two complementary scheduling controls.

##### Breaking head-of-line blocking

By default, vLLM's chunked-prefill scheduler runs in first-in, first-out order. One long prefill can claim the entire token budget step after step, and the short turns queued on the same rank cannot be scheduled at all until the long prefill finishes. This is known as head-of-line blocking; Figure 9 illustrates it in the session view of one rank's queue.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/hol-blocking.gif" alt="Animation: head-of-line blocking in the prefill queue, with and without a chunk cap" width="100%">
<br>
<em>Figure 9: Head-of-line blocking in the prefill queue, session view of one rank. Left: without a chunk cap, a long prefill claims the whole budget and the short cached turns wait. Right: with a 512-token cap, short turns join every step and begin decoding sooner.</em>
</p>

We tackle this issue with a simple scheduling policy: we use `--long-prefill-token-threshold` to cap how many tokens one request may schedule per step. With a 512-token threshold, a long prefill leaves room for short turns to join the same batch and begin decoding sooner. With DeepSeek V4 Pro on B300s, this increases total tokens per GPU-second (TPGS) by up to 93% and improves P90 interactivity by roughly 2.3×. The trade-off is a higher TTFT for the long request itself, so TTFT-sensitive deployments should use a larger threshold.

##### Align DEP prefill schedule cadence

DEP introduces a second inefficiency: MoE all-to-all communication forces ranks to advance in lockstep, so a rank processing prefill work slows the entire group. When prefills arrive on different steps on different ranks, this penalty is paid repeatedly.

To alleviate this imbalance, we set `--prefill-schedule-interval` to admit prefill work only every Nth engine step, using a counter aligned across data-parallel ranks. This concentrates prefill work onto the same steps across ranks and increases the fraction of the remaining steps devoted entirely to decode. Figure 10 illustrates this cadence across a DEP8 group.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/prefill-schedule-interval.gif" alt="Animation: prefill-schedule-interval aligns prefill cadence across a DEP8 group" width="100%">
<br>
<em>Figure 10: Prefill schedule cadence across a DEP8 group. Left: prefills arrive on different steps and stall the lockstep group repeatedly. Right: with an interval of 4, prefills coalesce onto the cadence steps and the steps in between are decode-only.</em>
</p>

### Scaling with optimal P/D disaggregation configurations

Optimizing a single engine is not enough to find the best latency-cost point for a distributed deployment, and more GPUs or disaggregation will not automatically improve the frontier. The prefill and decode stages must be rate-matched.

We use a standardized two-phase rate-matching methodology that can be automated by an agentic workflow:

**Phase 1: Saturation profiling.** Benchmark prefill-only and decode-only deployments separately, sweeping parallelism strategies (e.g., TP vs. wide EP) and deployment sizes (8, 16, or 32 GPUs) with increasing concurrency until throughput saturates. The output is a saturation table: max prefill/decode req/s for each (parallelism, size) configuration.

**Phase 2: P/D sweep.** Derive the P/D ratio from each configuration's Phase 1 saturation points, then sweep concurrency on the combined disaggregated deployment to collect metrics across the operating range.

### Closing the loop: model-specific kernels and community contributions

Agentic workloads also shift kernel bottlenecks toward long-context attention, speculative decoding, and communication. Here we highlight a few changes with measured end-to-end impact. All of our kernels are fully open source, and some have already been adopted by other open-source engines.

For MiniMax M3, a [CuteDSL long-context indexer](https://github.com/vllm-project/vllm/pull/48582) improves reported GB300 indexer latency by roughly 3% to 31%, depending on shape. The upstreamed MSA top-k path improves worst-case kernel performance by up to 4× and AgentX end-to-end throughput by roughly 7%; the speculative-verification path improves medium-batch decode performance by about 20% in reported tests.

For Kimi K3, [GEMM and reduce-scatter fusion](https://github.com/vllm-project/vllm/pull/52079) improves sequence-parallel communication, while [latent-tail MoE fusion](https://github.com/vllm-project/vllm/pull/53152) reduces end-to-end latency by roughly 5%.

For DeepSeek V4, community contributions improved MXFP4 MoE and HCA compression ([#43584](https://github.com/vllm-project/vllm/pull/43584) and [#44230](https://github.com/vllm-project/vllm/pull/44230)), added [multi-stream C4A](https://github.com/vllm-project/vllm/pull/42925), and improved [cluster-based top-k](https://github.com/vllm-project/vllm/pull/43008).

## Performance: agentic-first and openly verifiable

We demonstrate that vLLM is agentic-first through independent validation on [SemiAnalysis AgentX](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat), an open dataset built from &#36;3M of real-world agentic coding traces with 1M context, run on a public benchmark infrastructure of more than 1,000 chips and roughly 2 MW of compute.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/k3-agentx-dashboard.png" alt="Kimi K3 AgentX dashboard: total tokens per &#36;1 TCO vs. P90 interactivity across hardware" width="100%">
<br>
<em>Figure 11: Total tokens per &#36;1 under varying P90 interactivities with Kimi K3 running on various hardware. Source: <a href="https://inferencex.semianalysis.com/inference/kimi-k3?i_seq=agentic-traces&i_xmode=interactivity&g_model=Kimi-K3&i_best=0&i_active=b200_dynamo-vllm%2Cb300_vllm%2Cgb200_dynamo-vllm%2Cgb300_dynamo-vllm%2Cmi355x_vllm">Kimi K3 SemiAnalysis AgentX Dashboard</a>.</em>
</p>

Figure 11 shows the Kimi K3 dashboard as an example; the benchmark and all of its results are publicly accessible on the [AgentX Dashboard](https://inferencex.semianalysis.com/inference?i_seq=agentic-traces&i_xmode=interactivity&g_runid=33418433573&i_best=0&i_active=b200_vllm%2Cb300_vllm%2Cgb200_dynamo-vllm%2Cgb300_dynamo-vllm&i_hc=1&i_advlabel=0&i_label=0). We strongly recommend exploring the Pareto results for the other models and configurations.

In this post, we focus on the results of three open frontier models: DeepSeek V4 Pro, MiniMax M3, and Kimi K3. For each model, we report the highest-throughput vLLM configuration that maintains P90 interactivity above 50 tokens per second per user, a common and demanding latency SLO. The table below summarizes the key results.

| Model | GPUs / concurrency | Total tokens per GPU-second (TPGS)<sup><a href="#note-tpgs">1</a></sup> @ P90 > 50 tok/s | P90 interactivity |
| :---- | ----: | ----: | ----: |
| [DeepSeek V4 Pro 1.6T](https://inferencex.semianalysis.com/inference/agentic/439873) | 12 GB300s / 256 | **83K TPGS** | 58.3 tok/s |
| [MiniMax M3 428B](https://inferencex.semianalysis.com/inference/agentic/439907) | 2 B300s / 24 | 70K TPGS | **74.2 tok/s** |
| [Kimi K3 2.8T](https://inferencex.semianalysis.com/inference/agentic/441066) | 16 GB300s / 48 | 11.8K TPGS | 62.7 tok/s |

<p id="note-tpgs"><small><sup>1</sup> Total tokens per GPU-second (TPGS) counts input, output, and cached tokens. A detailed breakdown is available via each model's link.</small></p>

DeepSeek V4 Pro represents the high-throughput, cost-efficient case. A 12-chip GB300 P/D deployment serves 256 concurrent agent sessions while sustaining 58.3 tokens/s/user at P90. At this operating point, it processes 83K total tokens per GPU-second.

MiniMax M3 pushes interactivity further. With only 2 B300s, it sustains 74.2 tokens/s/user at P90 and delivers 70K total TPGS.

Kimi K3, one of the largest open frontier models, makes the case for frontier intelligence. At 2.8 trillion parameters, it is too large for a conventional single-server deployment, yet 16 GB300s sustain 62.7 tokens/s/user at P90 while processing 11.8K total TPGS.

Beyond performance, cost is the metric most relevant to users' daily use and to tokenomics. The table below compares the serving cost of all three open models against Opus 5.

| Model | GPU TCO/hour | Equivalent Opus 5 cost/hour<sup><a href="#note-opus">2</a></sup> | Cost advantage |
| :---- | ----: | ----: | ----: |
| [DeepSeek V4 Pro 1.6T](https://inferencex.semianalysis.com/inference/agentic/439873) | &#36;27.72 | &#36;2,926 | **106×** |
| [MiniMax M3 428B](https://inferencex.semianalysis.com/inference/agentic/439907) | &#36;4.52 | &#36;384 | **85×** |
| [Kimi K3 2.8T](https://inferencex.semianalysis.com/inference/agentic/441066) | &#36;36.96 | &#36;538 | **14.6×** |

<p id="note-opus"><small><sup>2</sup> The Opus 5 calculation uses cached input × &#36;0.50/M + uncached input × &#36;5/M + output × &#36;25/M. It assumes a perfect theoretical cache hit rate and excludes cache-write charges and long-context pricing premiums, which is conservative and favorable to Opus. The comparison is about serving cost, not model quality.</small></p>

The cost advantage comes from the defining property of agentic traffic: with a theoretical cache hit rate of more than 96%, vLLM reuses prefixes effectively and turns that reuse into serving efficiency across all three models, under the same settings as the table above.

For DeepSeek V4 Pro, serving the measured workload costs approximately &#36;28 per hour in GB300 infrastructure TCO. Processing the same token volume with Opus 5 would cost approximately &#36;2,926, even after applying the cache-read price to every theoretically reusable token. MiniMax M3 on B300s shows an 85× cost advantage, while Kimi K3 on GB300s remains 14.6× cheaper despite its substantially larger model size.

These are the numbers as of today; the dashboard is live and accessible to everyone. The AgentX harness is public at [SemiAnalysisAI/agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness), and every result above links to its run on the InferenceX dashboard for easy reproduction.

## The bitter lessons: where we failed and what we learned

Every failed idea narrows the search space. We observed several cases where plausible intuitions did not survive end-to-end measurement. We are still improving these features, but we want to share what we have learned so far.

#### Pipeline parallelism (PP) does not fit warm agentic turns

PP, including [chunked pipeline parallelism (CPP)](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.html), performs well on long, fresh prompts. Large prefills provide enough work to keep pipeline stages occupied, and throughput can scale nearly linearly with little communication cost.

Most agentic turns, however, already have system prompts and previous turns cached, and each new request may add only a few hundred or a few thousand tokens. There is not enough fresh computation to fill the pipeline efficiently, and pipeline bubbles consume much of the potential gain.

The lesson is not that PP is ineffective. It is effective for cold, compute-heavy prefills, but it should not be the default for warm, prefix-heavy turns that dominate agentic sessions.

#### Decode context parallelism (DCP) does not transfer cleanly to DeepSeek V4

DCP works well for pure MLA models (e.g., DeepSeek R1, Kimi K2.5, and K2.7) and hybrid MLA models (e.g., Kimi K3), as shown earlier. However, realizing a similar benefit for DeepSeek V4 is considerably harder because of its more complex attention stack. The compressed sparse attention and highly compressed attention include an indexer, an additional compressor, and the main attention operation. Context parallelism must partition and coordinate all of these sublayers, introducing substantial communication and implementation complexity.

We invested heavily in overlapping communication with computation and in optimizing the corresponding kernels. Even after those improvements, DCP only matched DEP rather than surpassing it. The result reinforces a broader point from the execution-plane section: parallelism must follow model architecture. A strategy that succeeds for one latent-attention model may not generalize to another.

#### Load balance does not guarantee better performance

In aggregated DEP deployments, we observed substantial imbalance in KV cache usage across ranks. The natural response was to balance requests according to queue depth, running tokens, or current KV utilization.

However, in our experiments with AgentX, all of these policies underperformed simple session-aware sticky routing. The reason is cache locality: many agentic sessions have short inter-turn delays, so the next turn frequently arrives while its prefix is still resident on the previous GPU. Moving the session to a less-loaded rank forces the system to retrieve the KV cache, even though the prefix is preserved in the distributed KV cache pool. The transfer is asynchronous and overlaps with computation, but it is not free. Prefetched blocks temporarily occupy GPU KV cache capacity, reducing the number of sequences the destination rank can admit. The system can therefore achieve a more balanced queue while processing fewer concurrent requests overall.

For workloads with short inter-turn delays, preserving session locality is more valuable than perfectly balancing instantaneous load. Routing decisions must account for the state already resident on each worker, not only the amount of queued work.

## The path ahead: planned optimizations and future work

The next step is to make agentic structure explicit throughout the serving stack. Here are a few examples for each layer.

In the control plane, we can make routing more explicit for first-turn requests, which tend to need long fresh prefills to fill up the prefix cache, and turn 2+ requests, which get high cache reuse and relatively short append prefill. This separation avoids head-of-line blocking and lets us configure engine setups and parallelism differently on each side, for example with PCP and CPP, to maximize efficiency on both.

In the execution plane and data plane, we are working with the community to support:

- **Agent hints**. Agentic frameworks or harnesses could carry hints along with the requests, such as session structure, potential branching points and cache positions, tool-call latencies, or session lifecycle. Our first step is to consume these hints through standardized APIs, and then use them to guide the engine on scheduling, cache eviction policies, and other optimizations.
- **Programmable KV cache**. Different workloads require different placement, retention, replication, and eviction policies. A programmable interface would let users control prefetching, eviction, or soft-pinning of KV caches to match their workload patterns.
- **Session-based KV cache management**. Inter-turn gaps create an opportunity to move the retained KV state toward the worker likely to serve the next turn. Prefetching during this idle interval can hide transfer latency and reduce cold resumptions.

## Acknowledgments

This effort was led by [Inferact](https://inferact.ai/) with extensive support from the vLLM community. We thank SemiAnalysis for developing and operating the open AgentX benchmark and for making its methodology and results reproducible. We also thank NVIDIA and AMD for their close collaboration and support throughout this work.

<script>
(function () {
  // Resize embedded interactive figures to their content height (they post it when framed).
  window.addEventListener('message', function (event) {
    var d = event.data;
    if (!d || d.type !== 'vllm-embed-resize' || typeof d.height !== 'number') return;
    var frames = document.querySelectorAll('iframe.vllm-embed');
    for (var i = 0; i < frames.length; i++) {
      if (frames[i].contentWindow === event.source) {
        frames[i].style.height = Math.ceil(d.height) + 'px';
      }
    }
  });
})();
</script>
