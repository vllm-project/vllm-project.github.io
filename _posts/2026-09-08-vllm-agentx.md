---
layout: post
title: "vLLM x AgentX: Optimizing for Real-World Agentic Serving"
author: "vLLM Team and Inferact"
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

Measured on [AgentX](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat), SemiAnalysis's public agentic benchmark, vLLM achieves up to 130K total tokens per GPU-second on DeepSeek V4 Pro, and an interactivity of up to 398 tokens per second on MiniMax M3. Across DeepSeek V4 Pro, Minimax M3, and Kimi K3, vLLM delivers a 14.6×–106× serving-cost advantage over Opus 5 API pricing (see [Performance](#performance-agentic-first-and-openly-verifiable)).

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/agentx-pareto-summary.png" alt="vLLM on SemiAnalysis AgentX: cost efficiency vs. P90 interactivity for DeepSeek V4 Pro, MiniMax M3, and Kimi K3" width="100%">
<br>
<em>Explore the <a href="/assets/interactive_pages/vllm-agentx-pareto.html">interactive curve</a>. Data source: <a href="https://inferencex.semianalysis.com/inference">SemiAnalysis AgentX</a>.</em>
</p>

## Characterizing agentic workloads: a second look

Since our first post on [serving agentic workloads](https://vllm.ai/blog/2026-05-06-mooncake-store) in May, agentic traffic share has continued to grow. As of June 2026, [OpenAI reported](https://openai.com/signals/enterprise-data/) that Codex generated 64% of combined Codex and ChatGPT output tokens among enterprise customers.

This growing token consumption stresses serving infrastructure along two axes: cost and latency. Cost efficiency determines how many concurrent agents fit in a fixed hardware budget; latency affects how quickly each agent progresses through reasoning and tool-use cycles. Optimizing agentic serving requires improving the latency-cost frontier as a whole.

To evaluate that frontier under representative traffic, SemiAnalysis recently released [AgentX](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat), a public benchmark built from real-world agentic coding traces. These traces provide a concrete view of the workload characteristics that serving systems must accommodate, as illustrated in the Figure:

- **Long-running, multi-turn sessions.** Median 43 turns per session.
- **Long contexts with short outputs.** Median input 142K tokens, median output 444 tokens.
- **Extensive prefix reuse.** Prefix-cache hit rate above 95%.
- **Subagent-heavy traffic.** 44% of sessions contain at least one subagent, with a median of four subagent rollouts among those sessions.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/agentic-session-structure.png" alt="Agentic sessions grow through reuse and branching" width="90%">
<br>
<em>Figure: Agentic sessions accumulate context across turns and branch into subagents. Each request carries earlier context forward, while subagents may inherit the parent context or start fresh. Explore the <a href="https://agentic-workload-explorer.inferact-inc-1374.chatgpt.site">interactive session explorer</a>.</em>
</p>

## Challenges in serving agentic workloads

These workload characteristics create three challenges for efficient serving.

- **Prefix cache pressure.** Every turn of a multi-turn session replays the full conversation so far. To keep many sessions running at once, the engine has to offload KV caches between turns. This is even more challenging when deployed at scale, as KV cache management, prefix caching, and offloading must take the strain and work efficiently across GPUs, prefill/decode disaggregated instances, and replicas.
- **Execution efficiency.** Agentic workloads feature long contexts and tight latency requirements, so the engine has to process more tokens and do more work per token in less time. This requires adapting parallelism, kernels, scheduling, speculative decoding, and other engine optimizations properly to the new request shape.
- **Finding the right P/D ratio.** Context lengths and cache hit rates vary wildly across sessions and subagents, and routing must efficiently balance cache affinity and load across ranks. These factors make it difficult to find the optimal P/D ratio under different concurrency for maximized throughput.

## The vLLM approach: optimizations across the stack

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/full-stack-overview.png" alt="Full-stack optimization for agentic serving: control plane, execution plane, and data plane" width="100%">
<br>
<em>Figure: Full-stack optimization for agentic serving. The data plane manages a distributed shared KV cache, the execution plane maps each model to appropriate parallelism and kernels, and the control plane coordinates proper P/D ratio and request scheduling.</em>
</p>

### Data plane: keep KV caches warm and close to compute

#### Hybrid KV cache management: a foundation that keeps evolving

KV cache management has been central to vLLM since PagedAttention, and agentic workloads with long contexts put heavier pressure on KV cache capacity. Modern hybrid models complicate allocation further by combining sliding-window and linear attention with full attention, whose cached blocks differ in size and lifetime.

vLLM's hybrid KV cache manager tackles the complexity with a simple core idea: having a uniform memory page as the basic allocation unit, then managing those units through one shared block pool.

A shared pool lets vLLM reallocate memory dynamically on demand instead of statically partitioning capacity by attention type. This matters because full-attention KV grows with sequence length, while sliding-window and recurrent state follow different lifetimes and scaling rules. The best partition therefore changes with concurrency, context length, and prefix-reuse patterns.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/hybrid-kv-cache-manager.png" alt="vLLM's hybrid KV cache manager: one page size for all attention types, one shared block pool" width="100%">
<br>
<em>Figure: vLLM's hybrid KV cache manager.</em>
</p>

The abstraction continues to evolve as new architectures expose fragmentation and transfer inefficiencies. For example, [DeepSeek V4](https://vllm.ai/blog/2026-04-24-deepseek-v4)'s initial KV cache layout fragments different cache types into three size buckets and allocated 92 separate tensors. As shown in the figure below, the fragmentation causes some extra padding waste, and remains inefficient for P/D transfer and KV cache offloading.

The new [packed KV cache layout](https://github.com/vllm-project/vllm/pull/44577) instead, stores cache groups and layers in one contiguous backing allocation per block rather than 92 fragmented ones. This reduces the descriptor and PD transfer overhead, and also permits a smaller allocation unit when the FP4 indexer is enabled, saving [roughly 10% of KV cache memory](https://github.com/vllm-project/vllm/pull/48993).

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/dsv4-packed-kv-layout.png" alt="DeepSeek V4 Pro hybrid KV cache: size-bucketed tensors before vs. packed by cache group after" width="100%">
<br>
<em>Figure: Hybrid KV cache manager and packed KV cache layout for DeepSeek V4 Pro.</em>
</p>

#### Hierarchical KV cache offloading: distributed KV cache pool with smart retention policies

To preserve prefix caches beyond GPU memory capacity and across each engine, vLLM has integrated [MooncakeStore](https://github.com/kvcache-ai/Mooncake) to provide a distributed KV cache pool, with details covered in [our previous blog](https://vllm.ai/blog/2026-05-06-mooncake-store). Since the integration, we have seen its increasing adoption, and we keep shipping new features and performance improvements for capacity, efficiency, and retention on agentic workloads.

**Model architecture parity.** KV cache offloading integration remains a first-class citizen in vLLM with full support for new model architectures including: sparse attention, compressed attention, linear attention, etc. This is done while making sure that other engine features remain fully functional and performant, such as asynchronous scheduling, P/D disaggregation, speculative decoding, parallelism, etc.

**Hierarchical KV cache offloading.** vLLM supports hierarchical tiers for the distributed KV cache pool to further extend capacity with disks and extra CPU-only nodes. This is achieved with vLLM's [MooncakeStore](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/#configure-mooncake) `standalone-store` mode, which makes an external Mooncake client own the CPU pool and disk tier, and turns vLLM workers into pure requesters. By launching a standalone Mooncake client on each node, we freely expand the KV cache pool with CPU memory and disks. We have also integrated the distributed shared KV cache pool with routers, such as [Dynamo](https://github.com/ai-dynamo/dynamo) and [llm-d](https://github.com/llm-d/llm-d), to simplify the routing policy so that requests can get cache hits on any instance.

**Performance optimizations.** Hybrid models must construct keys and perform lookups separately for each attention type, which multiplies CPU overhead. We reduced this cost through more efficient data structures, asynchronous lookup, work moved off the scheduler's critical path, and parallel send and receive operations. Implementation details are available in [PR#46188](https://github.com/vllm-project/vllm/pull/46188), [PR#45444](https://github.com/vllm-project/vllm/pull/45444), [PR#45659](https://github.com/vllm-project/vllm/pull/45659), and [PR#47317](https://github.com/vllm-project/vllm/pull/47317).

**Session-aware prefix-cache retention.** For hybrid models with linear or sliding-window layers alongside full attention, prefix reuse requires preserving linear state or sliding-window caches. Keeping these snapshots at every token is expensive, so we combine two complementary policies:

- **[Interval-based retention](https://github.com/vllm-project/vllm/pull/43447)** automatically preserves prompt-end caches/linear states at each turn. Subsequent turns and forked subagents, which typically replay and extend earlier turn's context, can then reuse the cached context.

  However, shared prefixes typically end within a turn, so interval-based retention may not preserve a checkpoint. To capture this reuse, we introduce a second policy:

- **[Marconi-style selective retention](https://github.com/vllm-project/vllm/pull/47782)** retains a checkpoint when a prefix is observed a second time. When a request encounters a previously observed prefix without a retained checkpoint, vLLM recomputes the missing state and saves a checkpoint at that boundary. Subsequent requests sharing the prefix can then reuse it.

Together, these policies preserve high cache hit rate and avoid excessive storage overhead for large-scale agentic workloads. Our [vLLM Kimi K3 blog](https://vllm.ai/blog/2026-07-27-k3) explains the technical details in depth.

### Execution plane: generate tokens fast

#### Model-specific parallelism

Modern inference systems expose several axes of parallelism, such as tensor parallelism (TP), data parallelism (DP), expert parallelism (EP), pipeline parallelism (PP), and context parallelism (CP).

Determining the optimal parallelism, however, depends on the model architecture, hardware topology, workload patterns, and latency SLOs. In this section, we examine two representative models on NVIDIA GB/B-series GPUs and AMD counterparts and discuss our optimizations and findings.

**Kimi K3**

Kimi K3 features multi-head latent attention (MLA) and Kimi linear attention (KDA). Since MLA compresses KV into a single latent space with one head, plain tensor parallelism (TP), which replicates that latent cache across ranks, is not very efficient.

As an alternative to TP, we have found strong performance gains in [decode context parallelism (DCP)](https://vllm.ai/blog/2026-08-07-decode-context-parallelism), which shards the cache along the sequence dimension, leaving each rank with 1/N of the KV state. Specifically, DCP offers two benefits for agentic workloads:

- **Lower decode latency.** MLA attention is memory-bound, and its cost grows with context length. As agentic prefixes grow, attention becomes a larger share of each decode step.
- **Higher throughput and KV capacity.** Avoiding KV cache replication enables the engine to keep more sequences in flight without stalling on KV admission, and hence achieve higher throughput.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/k3-tp8-vs-dcp8.png" alt="Kimi K3 decode: TP8 vs. DCP8, P50 TPOT vs. sequences per rank" width="90%">
<br>
<em>Figure: For Kimi K3, DCP achieves lower decode latency and scales to higher concurrency.</em>
</p>

DCP's tradeoff is extra communication: the KV cache is sharded by sequence, so every MLA decode layer needs a query gather before attention and a partial-output reduction after it.

We carefully optimized the DCP compute path to bypass NCCL operations and avoid such overheads. We leverage symmetric-memory buffers that peer GPUs can directly load/store. Queries are multicast directly into the buffers consumed by attention kernels. Each GPU then writes its partial attention outputs and log-sum-exp (LSE) statistics directly into its peers' receive slots, where each rank locally merges the results with online softmax. These GPU-to-GPU writes are fused with the computation into the same kernels, cutting the latency by about 13% per layer against the default DEP8 implementation.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/k3-dcp-symmem.gif" alt="Animation: MLA decode path under DCP4 using symmetric memory" width="80%">
<br>
<em>Figure: MLA decode path under DCP4 using symmetric memory. Each step is fused into a single kernel.</em>
</p>

A larger scale-up domain can change the best strategy. For example, on an NVL72-class system, wide EP with data parallelism (DEP) can scale better than DCP and deliver higher throughput at the same decode latency SLO. This is because at larger, multi-node DCP sizes, communication cost in sharded attention outweighs the compute it saves. DEP assigns requests and their KV caches to different data-parallel ranks, avoiding DCP's attention collectives while sharding the MoE experts across ranks.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/k3-dcp8-vs-dep16.png" alt="Kimi K3 decode: DCP8 vs. DEP16, P50 TPOT vs. sequences per rank" width="90%">
<br>
<em>Figure: WideEP (DEP16) scales better than DCP8 when per-rank batch size > 3.</em>
</p>

**DeepSeek V4**

DeepSeek V4 also has MLA-style KV caches that cause memory inefficiency due to replication under TP. In addition, its unique compressed sparse attention makes TP head sharding compute inefficient for three reasons:

- The compressor paths produce only one shared KV representation per compressed position rather than independent per-head states. TP therefore cannot shard compute along the KV-head dimension, and every rank repeats the compressor work.
- The indexer, while having 64 heads, produces only one global top-k selection per token. The current TP path therefore replicates the full indexer on every rank, avoiding a dense score reduction before top-k but duplicating the work.
- Sparse MLA is dominated by scanning and gathering top-k KV cache entries, not by attention arithmetic. TP repeats much of this memory-bound work on every rank while dividing only the cheaper head-wise computation.

In our practice, prefill context parallelism (PCP) performs best for long prefills, while data and expert parallelism (DEP) works across a broader range of serving conditions.

PCP shards the prompt sequence (the query tensor), distributing compressor and indexer work across ranks, while giving sparse MLA a wider, more efficient head-local shape. For a 32K prompt, PCP8 achieves an 8.9× prefill speedup over TP8, substantially reducing TTFT. However, it still replicates decode-side state across ranks, and hence is most suitable for dedicated prefill workers.

DCP is less effective for DeepSeek V4 than for Kimi K3 due to its more complex model architecture. We examine this trade-off [shortly](#decode-context-parallelism-dcp-does-not-transfer-cleanly-to-deepseek-v4). DEP instead distributes requests and decoded tokens across data-parallel ranks and keeps the attention path completely local. This makes DEP our default for most DeepSeek V4 configurations.

#### Scheduling mixed agentic traffic at two levels

Agentic serving mixes frequent, append-only requests with long prefix reuse and short prefill, with occasional long fresh prefills spanning tens of thousands of tokens. This creates two scheduling problems: within an instance, a long prefill can block short interactive turns; across DEP ranks, uneven prefill placement creates load imbalance. We address them with two complementary scheduling controls.

##### Breaking head-of-line blocking

By default, vLLM's chunked-prefill scheduler follows the first-in, first-out order. One long prefill can claim the entire budget step after step, and the short turns queued on the same rank cannot be scheduled at all until the long prefill finishes. This is known as head-of-line blocking and is illustrated below in the session view of one rank's queue.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/hol-blocking.gif" alt="Animation: head-of-line blocking in the prefill queue, with and without a chunk cap" width="100%">
</p>

We tackle this issue with a simple scheduling policy: using `--long-prefill-token-threshold` to cap how many tokens one request may schedule per step. With a 512-token threshold, a long prefill leaves room for short turns to join the same batch and begin decoding sooner. With DeepSeek V4 Pro on B300s, this increases tokens per GPU-second (TPGS) by up to 93% and improves p90 interactivity by roughly 2.3x. The trade-off is higher TTFT for the long request itself, so TTFT-sensitive deployments should use a larger threshold.

##### Align DEP prefill schedule cadence

DEP introduces a second form of interference. MoE all-to-all communication forces ranks to advance in lockstep, so a rank processing prefill work slows the entire group. When prefills arrive on different steps across ranks, this penalty is repeatedly exposed.

To alleviate this imbalance, we set `--prefill-schedule-interval` to admit prefill work only every Nth engine step, using a counter aligned across data-parallel ranks. This concentrates prefill work onto the same steps across ranks and increases the fraction of intervening steps devoted entirely to decode. The figure below illustrates this cadence across a DEP8 group.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/prefill-schedule-interval.gif" alt="Animation: prefill-schedule-interval aligns prefill cadence across a DEP8 group" width="100%">
</p>

### Scaling with optimal P/D disaggregation configurations

Optimizing a single engine is not enough to find the best latency-cost point for a distributed deployment. More GPUs or more P/D disaggregation do not automatically improve the frontier; the prefill and decode stages must be rate-matched.

We use a two-phase rate-matching workflow:

**Phase 1: Saturation profiling.** Benchmark prefill-only and decode-only deployments separately, sweeping parallelism strategies (e.g., TP vs. wide-EP) and deployment sizes (8/16/32 GPUs) with increasing concurrency until throughput saturates. The output is a saturation table: max prefill/decode req/s for each (parallelism, size) configuration.

**Phase 2: P:D sweep.** Derive the P/D ratio from each configuration's Phase 1 saturation points, then sweep concurrency on the combined disaggregated deployment to collect throughput/latency data points across the operating range.

**Automation.** Agents run the workflow end to end and stream results to a dashboard that plots TPGS against latency for every configuration. The resulting frontier identifies the lowest-cost P/D composition that satisfies a chosen latency objective.

### Closing the loop: model-specific kernels and community contributions

Agentic workloads also shift kernel bottlenecks toward long-context attention, speculative decoding, and communication. Here we highlight a few changes with measured end-to-end impact. All our kernels are fully open-sourced, and some of them have already been adopted by other OSS engines.

For MiniMax M3, a [CuteDSL long-context indexer](https://github.com/vllm-project/vllm/pull/48582) improves reported GB300 indexer latency by roughly 3% to 31%, depending on shape. The upstreamed MSA top-k path improves worst-case kernel performance by up to 4x and AgentX end-to-end throughput by roughly 7%; the speculative-verification path improves medium-batch decode performance by about 20% in reported tests.

For Kimi K3, [GEMM and reduce-scatter fusion](https://github.com/vllm-project/vllm/pull/52079) improves sequence-parallel communication, while [latent-tail MoE fusion](https://github.com/vllm-project/vllm/pull/53152) reduces end-to-end latency by roughly 5%.

For DeepSeek V4, community contributions improved MXFP4 MoE and HCA compression ([#43584](https://github.com/vllm-project/vllm/pull/43584) and [#44230](https://github.com/vllm-project/vllm/pull/44230)), added [multi-stream C4A](https://github.com/vllm-project/vllm/pull/42925), and improved [cluster-based top-k](https://github.com/vllm-project/vllm/pull/43008).

## Performance: agentic-first and openly verifiable

We showcase that vLLM is agentic-first with independent validation on [SemiAnalysis AgentX](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat), an open dataset built from $3M real-world agentic coding traces with a 1M context window, and a public benchmark infrastructure running on >1000 chips and ~2 MW compute.

<p align="center">
<img src="/assets/figures/2026-09-08-vllm-agentx/k3-agentx-dashboard.png" alt="Kimi K3 AgentX dashboard: total tokens per $1 TCO vs. P90 interactivity across hardware" width="100%">
<br>
<em>Figure: Total tokens per $1 under varying P90 interactivities with Kimi K3 running on various hardware. Source: <a href="https://inferencex.semianalysis.com/inference/kimi-k3?i_seq=agentic-traces&i_xmode=interactivity&g_model=Kimi-K3&i_best=0&i_active=b200_dynamo-vllm%2Cb300_vllm%2Cgb200_dynamo-vllm%2Cgb300_dynamo-vllm%2Cmi355x_vllm">Kimi K3 AgentX Dashboard</a>.</em>
</p>

The figure shows the Kimi K3 dashboard as an example, but both the benchmark and its results are publicly accessible online at the [AgentX Dashboard](https://inferencex.semianalysis.com/inference?i_seq=agentic-traces&i_xmode=interactivity&g_runid=33418433573&i_best=0&i_active=b200_vllm%2Cb300_vllm%2Cgb200_dynamo-vllm%2Cgb300_dynamo-vllm&i_hc=1&i_advlabel=0&i_label=0). We encourage interested readers to check out and compare the results for the other models.

In this post, we focus on the results of three open frontier models: DeepSeek V4 Pro, Minimax M3, and Kimi K3. For each model, we display the highest-throughput vLLM configuration that maintains p90 interactivity above 50 tokens per second per user, which is a commonly demanding latency SLO. The table below summarizes the key results.

| Model | GPUs / concurrency | Total tokens per GPU second (TPGS) @ >50TPS | P90 interactivity |
|---|---|---|---|
| [DeepSeek V4 Pro 1.6T](https://inferencex.semianalysis.com/inference/agentic/439873) | 12 GB300s / 256 | **83K TPGS** | 58.3 tok/s |
| [MiniMax M3 428B](https://inferencex.semianalysis.com/inference/agentic/439907) | 2 B300s / 24 | 70K TPGS | **74.2 tok/s** |
| [Kimi K3 2.8T](https://inferencex.semianalysis.com/inference/agentic/441066) | 16 GB300s / 48 | 11.8K TPGS | 62.7 tok/s |

*Footnote: Total tokens per GPU second (TPGS) is measured with input, output, and cached tokens. Detailed breakdown is available following each model's link.*

DeepSeek V4 Pro represents a high-throughput and cost-efficient use case. A 12-chip GB300 PD deployment serves 256 concurrent agent sessions while sustaining 58.3 tokens/s/user at p90. At this operating point, it processes 83K total tokens per GPU second, reaching 1.83 million aggregate tokens per second.

MiniMax M3 pushes interactivity further and features high response speed. With only 2 B300s, it sustains 74.2 tokens/s/user at P90 and delivers 70K total TPGS.

Kimi K3, as one of the largest open frontier models, demonstrates the case for frontier intelligence. At 2.8 trillion parameters, it is too large for a conventional single-server deployment. Nevertheless, 16 GB300s sustain 62.7 tokens/s/user at P90 while processing approximately 11.8K total TPGS.

The interactive chart below plots the full cost-efficiency vs. interactivity curve for each of the three models.

<embed
  src="/assets/interactive_pages/vllm-agentx-pareto.html"
  type="text/html"
  title="vLLM on AgentX: cost efficiency vs. interactivity"
  width="100%"
  height="640"
  style="display: block; width: 100%; max-width: 100%; overflow: hidden; border: 0; border-radius: 12px;"
>

[Open the interactive curve full-screen](/assets/interactive_pages/vllm-agentx-pareto.html)

Besides performance, we found cost to be a very interesting and useful metric, which is perhaps more relevant to users' daily use and inference providers' economy. The table below compares the cost of all three open models against Opus 5 to illustrate the idea.

| Model | GPU TCO/hour | Equivalent Opus 5 cost/hour | Cost advantage |
|---|---|---|---|
| [DeepSeek V4 Pro 1.6T](https://inferencex.semianalysis.com/inference/agentic/439873) | $27.72 | $2,926 | **106×** |
| [MiniMax M3 428B](https://inferencex.semianalysis.com/inference/agentic/439907) | $4.52 | $384 | **85×** |
| [Kimi K3 2.8T](https://inferencex.semianalysis.com/inference/agentic/441066) | $36.96 | $538 | **14.6×** |

*Footnote: The Opus 5 calculation uses: cached input x $0.50/M + uncached input x $5/M + output x $25/M. It assumes a perfect theoretical cache hit rate, excludes cache-write charges and long-context pricing premiums, making the comparison conservative for Opus. This is a serving-cost comparison at the measured token mix and cache-reuse rate, not a claim of equivalent model quality.*

The cost advantage comes from the defining property of agentic traffic: with a theoretical cache hit rate of more than 96%, vLLM effectively reuses prefixes and unleashes serving efficiency across all three models under the same setting as the table above.

For DeepSeek V4 Pro, serving the measured workload costs approximately $28 per hour in GB300 infrastructure TCO. Processing the same token volume with Opus 5 would cost approximately $2,926, even after applying the cache-read price to every theoretically reusable token. MiniMax M3 on B300s shows an 85x cost advantage, while Kimi K3 on GB300s remains 14.6x cheaper despite its substantially larger model size.

We report numbers as of today, but the dashboard is live and interactively accessible to everyone. The AgentX harness is public at [SemiAnalysisAI/agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness), and every result above links to its run on the InferenceX dashboard for easy reproduction.

## The bitter lessons: what failed and what we learned

Every failed idea narrows the search space. We did observe a few cases where plausible intuitions did not survive end-to-end measurement. While we are still improving these features, we'd also like to share what we have learned so far.

#### Pipeline parallelism (PP) does not fit warm agentic turns

PP, including [chunked pipeline parallelism (CPP)](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.html), performs well on long, fresh prompts. Large prefills provide enough work to keep pipeline stages occupied, and throughput can scale nearly linearly with little communication cost.

For most agentic turns, however, they already have system prompts and previous turns cached, and each new request may add only a few hundred or a few thousand tokens. There is not enough fresh computation to fill the pipeline efficiently, and pipeline bubbles consume much of the potential gain.

The lesson is not that PP is ineffective. It is effective for cold, compute-heavy prefills, but it should not be the default for warm, prefix-heavy turns that dominate agentic sessions.

#### Decode context parallelism (DCP) does not transfer cleanly to DeepSeek V4

DCP works well for pure MLA models (e.g., DeepSeek R1, Kimi K2.5, and K2.7) and hybrid MLA models (e.g., Kimi K3) as elaborated earlier. However, realizing a similar benefit for DeepSeek V4 is way more challenging due to its more complicated attention stack. The compressed sparse attention and highly compressed attention include an indexer, an additional compressor, and the main attention operation. Context parallelism must partition and coordinate all of these sublayers, introducing substantial communication and implementation complexity.

We invested heavily in overlapping communication with computation and optimizing the corresponding kernels. Even after those improvements, DCP only matched DEP rather than surpassing it. The result reinforces a broader point from the execution-plane section: parallelism must follow model architecture. A strategy that succeeds for one latent-attention model may not generalize to another.

#### Load balance does not guarantee better performance

In aggregated DEP deployments, we observed substantial imbalance in KV cache usage across ranks. The natural response was to balance requests according to queue depth, running tokens, or current KV utilization. However, in our experiments with AgentX, all of these policies underperformed simple session-aware sticky routing.

The reason is cache locality. Many agentic sessions have short inter-turn delays, so the next turn frequently arrives while its prefix remains resident on the previous GPU. Moving the session to a less-loaded rank, although its prefix caches are preserved in the distributed KV cache pool, forces the system to retrieve KV caches. The transfer itself is asynchronous and overlaps with computation, yet still not free. Prefetched blocks temporarily occupy GPU KV cache capacity, reducing the number of sequences the destination rank can admit. The system can therefore achieve a more balanced queue while processing fewer concurrent requests overall.

For workloads with short inter-turn delays, preserving session locality is more valuable than perfectly balancing instantaneous load. Routing decisions must account for the state already resident on each worker, not only the amount of queued work.

## The path ahead: planned optimizations and future work

The next step is to make agentic structure explicit throughout the serving stack, and here we exemplify a few in each layer.

In the control plane, we can make routing more explicit for first-turn requests, which tend to need long fresh prefills to fill up the prefix cache, and turn 2+ requests, which get high cache reuse and relatively short append prefill. This separation avoids head-of-line blocking and allows us to configure engine setups and parallelism differently, for example, PCP and CPP, for maximized efficiency on both sides.

In the execution plane and data plane, we are working with the community to support:

- **Agent hints.** Agentic frameworks or harnesses could carry hints along with the requests, such as session structure, potential branching points and cache positions, tool call latencies, or the lifecycle of sessions, etc. Our first step is to consume these hints with standardized APIs, and then use them to guide the engine on scheduling, cache eviction policies, and other optimizations.
- **Programmable KV cache.** Different workloads require different placement, retention, replication, and eviction policies. A programmable interface would allow users to control prefetching, eviction, or soft pinning KV caches given the workload patterns.
- **Session-based KV cache management.** Inter-turn gaps create an opportunity to move the retained KV state toward the worker likely to serve the next turn. Prefetching during this idle interval can hide transfer latency and reduce cold resumptions.

## Acknowledgments

This effort was led by Inferact with extensive support from the vLLM community. We thank SemiAnalysis for developing and operating the open AgentX benchmark and for making its methodology and results reproducible.

<!-- TODO (from doc comment): decide whether to also acknowledge NVIDIA / AMD here. -->
