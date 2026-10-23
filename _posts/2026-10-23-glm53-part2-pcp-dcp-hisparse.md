---
layout: post
title: "GLM-5.3 Optimizations, Part 2: PCP, DCP, and Hybrid HiSparse on 16 H200s"
math: true
author: "vLLM Team"
summary: "Splitting each prompt across GPUs by token (PCP) instead of by head (TP) removes GLM-5.3's replicated indexer work and runs sparse MLA at ~7× the arithmetic intensity; sharding the KV by token (DCP) lets one prefill engine hold a 1M-token request. Paired with a Hybrid HiSparse decode node, it gives 1.7–3.2× the per-user speed of data-parallel prefill at the same throughput on an agentic workload."
image: /assets/figures/2026-10-23-glm53-part2-pcp-dcp-hisparse/agentx-pareto.svg
social_image: /assets/figures/2026-10-23-glm53-part2-pcp-dcp-hisparse/agentx-pareto.png
tags:
  - glm
  - parallelism
  - kv-cache
  - disaggregation
  - performance
---

**TL;DR:** Long agentic contexts push GLM-5.3 against two limits on H200s: KV cache memory and prefill speed. We address both on 16 H200s, split into one prefill node and one decode node. **Prefill Context Parallelism (PCP)** splits each prompt across GPUs by token instead of by head, which suits GLM-5.3's sparse attention far better than tensor parallelism. **Decode Context Parallelism (DCP)** shards the prefill engine's KV cache by token, and **Hybrid HiSparse** lets decode keep KV in host memory, so a 1M-token request fits. The hard part is making them compose: PCP, DCP, Hybrid HiSparse, expert parallelism, CPU KV offloading, MTP speculative decoding, and P/D disaggregation over NIXL all have to work together in one deployment. vLLM's [Model Runner V2](/blog/2026-03-24-mrv2) core makes that much easier. On AgentX, our full setup (PCP8 $\times$ DCP8 prefill with Hybrid HiSparse decode) gives $1.7–3.2\times$ the per-user speed of the same deployment with data-parallel prefill at the same throughput, and it is the only configuration we tested that serves 1M-token contexts.

[Part 1](/blog/2026-09-08-glm53-part1-hybrid-sparse-offloading) showed how Hybrid HiSparse keeps GLM-5.3 requests decoding after their KV outgrows the GPU, on a single $8\times$ H200 node that is tight on memory for a model this size. This post scales up to a prefill/decode deployment on two nodes. On the prefill engine, PCP makes long prompts prefill fast and DCP makes their KV fit; on the decode node, Hybrid HiSparse keeps them decoding.

## Why not TP or DP?

Agentic contexts are long and grow every turn, so prefill matters: without an efficient prefill deployment, the whole system bottlenecks on the prefill engines. Two standard ways to run GLM-5.3 prefill on an $8\times$ H200 node are DP8 and TP8:

* **DP8** assigns each prompt to a single GPU, so its attention cannot use the other seven. This limits time to first token (TTFT) for long prompts, and the prompt's KV cache must fit in that GPU's memory. <!--: about 300K tokens.--> <!--RLC: where is the 300K number comming from? our glm5.3+h200+vllm setup specifically, considering gpu_memory_utilization etc? in other words, is this the number we empirically see in the vLLM startup log? -->
* **TP8** partitions the attention heads across GPUs. Because MLA's latent KV is shared by all 64 heads, every GPU stores the full cache and reads it to serve only 8 heads. The indexer, which scores every earlier token to select the 2,048 that sparse MLA attends to, is not partitioned at all, so its quadratic cost is repeated on every GPU.

Neither scheme spreads one long prompt's attention work across the eight GPUs without also keeping its whole KV on each GPU that serves it. The next two sections fix each half: PCP splits a prompt's queries across GPUs, and DCP splits its KV.

## PCP: split the tokens, not the heads

In our $8\times$ H200 prefill engine, PCP splits each prompt across the eight GPUs by token: every GPU computes the queries for one eighth of the prompt, with all 64 heads. A naive split would be unbalanced. Under causal attention a token attends to every token before it, so the GPU holding the end of the prompt would do far more work than the one holding the start. PCP instead cuts the prompt into 16 chunks and pairs them up: GPU $r$ takes chunk $r$ from the front and chunk $15 − r$ from the back, so every GPU gets the same mix of cheap and expensive tokens (Figure 3 shows this as a V-shaped pattern across the eight GPUs). This is the load-balanced sharding from [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783). Finally, each GPU's new KV is all-gathered, so every query can still attend to the full context.

<p align="center">
<img src="/assets/figures/2026-10-23-glm53-part2-pcp-dcp-hisparse/tp-vs-pcp-prefill.svg" alt="Token chunks by heads for one prefill on 8 GPUs, shaded by owning GPU. TP8: each GPU owns 8 heads of every token and computes the full indexer, so sparse MLA does 26 FLOPs per byte moved and is memory-bound on H200. PCP8: each GPU owns token chunks r and 15 minus r with all 64 heads and indexes only those tokens, so sparse MLA does 191 FLOPs per byte moved, near the H200 ridge of 206" width="100%">
<br>
<em>Figure 1. One prefill on 8 GPUs under TP8 and PCP8, shaded by the GPU that owns each piece. The bars show sparse MLA's arithmetic intensity; the line marks where the H200 becomes compute-bound in BF16.</em>
</p>

Figure 1 compares the two splits. PCP fixes both problems. Each GPU indexes only its own tokens, so the indexer is no longer replicated on every GPU and each GPU does an eighth of it. And each KV row read from memory now serves all 64 heads instead of 8, raising sparse MLA from about 26 to about 191 FLOPs per byte moved,[^intensity] close to where the H200 becomes compute-bound in BF16.[^ridge]

Not replicating the indexer matters most for long prefills. The indexer's cost grows quadratically with prompt length, while sparse MLA's grows only linearly, because each query attends to a fixed 2,048 tokens. Counting FLOPs for a prefill from scratch, the indexer overtakes sparse MLA under TP8 once the prompt reaches about 32K tokens; under PCP8, where no GPU replicates it, the crossover moves out to about 256K.

Together, these make PCP a great fit to prefill long prompts. But PCP still has two costs. Each GPU holds all the attention weights, while the MoE experts stay split across the eight GPUs with EP8. And because the new KV is all-gathered every layer, every GPU still ends up storing the full KV cache. That replicated cache, not compute, is what limits context length, and it is what DCP addresses.

## DCP: shard the KV cache too

To be more specific, a 1M-token GLM-5.3 request needs 53.4 GiB of FP8 KV,[^kvsize] but each GPU has room for only about 15 GiB. [Decode Context Parallelism](/blog/2026-08-07-decode-context-parallelism) (DCP) fixes this by splitting the cache itself: tokens are dealt round-robin across the eight GPUs, so each GPU stores every eighth token, and the same request needs only 6.7 GiB (=53.4/8) per GPU. Figure 2 compares this with DP8 and TP8.

<!--RLC: where is the 15 GiB number of KV room per GPU coming from? vLLM logs? -->

<p align="center">
<img src="/assets/figures/2026-10-23-glm53-part2-pcp-dcp-hisparse/pcp-dcp-kv-sharding.svg" alt="KV per GPU for one 1M-token request: 53.4 GiB on one GPU under DP8, 53.4 GiB on every GPU under TP8, 6.7 GiB per GPU under PCP8 with DCP8" width="100%">
<br>
<em>Figure 2. TP replicates MLA's latent KV, so eight GPUs buy no room. DCP divides it by eight.</em>
</p>

> [!NOTE]
> **PCP splits queries; DCP splits KV.** The names are historical. DCP was built for decode, where each request has a single query token, so the KV cache is the only thing worth splitting. PCP was built for prefill, where the prompt’s many query tokens are the work. Our setup runs both in the prefill engine.

As a result of this sharding, most of the KV that each GPU’s queries need now lives on other GPUs. In decode, DCP sends the queries to the KV: each GPU attends over its own shard, and the partial results are merged with their log-sum-exp. In this prefill engine it does the reverse: every layer, each GPU all-gathers the shards into a temporary full-context copy, attends for its own queries, and frees the copy. Only the persistent cache is divided by eight.

Figure 3 puts it all together. Step through one layer of PCP8 $\times$ DCP8 to see which KV each GPU stores, which queries it computes, and what crosses the interconnect. Switch to DP8, TP8 or PCP8 to compare. Note that the two layouts do not line up: GPU $g$ stores every eighth token but computes the queries of chunks $g$ and $15 − g$, which is why every layer needs the all-gather.

<div id="ppx" role="figure" aria-label="Figure 3: interactive view of one prefill layer on 8 GPUs under DP8, TP8, PCP8 and PCP8 × DCP8"></div>

<script src="/assets/figures/2026-10-23-glm53-part2-pcp-dcp-hisparse/prefill-explorer.js?v=3"></script>

<p align="center"><em>Figure 3. One prefill layer on 8 GPUs under each scheme. Pick a scheme and step through to see which queries each GPU computes, which KV it keeps, and what crosses the interconnect.</em></p>

That all-gather is not free: every layer, it re-sends the whole cached context, so its cost grows with context length. Figure 4 isolates it by running the same 2K-token prefill chunk on one PCP8 node with and without DCP8, at equal cached context. DCP8 adds about 6% per step up to 64K tokens of context, rising to 15% at 248K. That is the price of an 8× larger KV pool. This overhead could be reduced by overlapping each layer's all-gather with the previous layer's compute, an optimization we are currently working on.

<p align="center">
<img src="/assets/figures/2026-10-23-glm53-part2-pcp-dcp-hisparse/dcp-overhead.svg" alt="DCP8 overhead per prefill step as a percentage of the step without DCP, against cached context: about 6 percent up to 64K tokens, rising to 15 percent at 248K for a 2K-token chunk" width="75%">
<br>
<em>Figure 4. DCP8's per-step overhead for a 2K-token chunk on one 8× H200 PCP8 engine. The band spans 1K, 4K, and 8K chunks.</em>
</p>

Beyond roughly 250K tokens, PCP8 alone cannot hold the KV cache; there, DCP is not an overhead but what makes long prompts possible. Together, PCP8 $\times$ DCP8 gives the prefill engine both speed and room.

## The deployment

We serve it as a prefill/decode (P/D) disaggregated deployment on two 8× H200 nodes. Both engines use an FP8 KV cache and MTP speculative decoding with three draft tokens (MTP3):

* **Prefill:** one PCP8 × DCP8 × EP8 engine, with 512 GiB of CPU KV offload so later turns reuse their prefix.
* **Decode:** eight DP ranks with EP8 and Hybrid HiSparse from [Part 1](/blog/2026-09-08-glm53-part1-hybrid-sparse-offloading) (48 GiB host pool + 16 GiB offload per rank), so decode concurrency is bounded by host memory rather than HBM.

A prefix-aware [llm-d](https://github.com/llm-d/llm-d) router sends each request to the prefill engine, the NIXL connector moves its KV cache to a decode rank, and the decode rank streams the tokens back.

We compare PCP8 × DCP8 prefill against DP8 prefill (eight DP ranks with EP8 and 64 GiB of CPU KV offload each, the same 512 GiB in total), and Hybrid HiSparse against plain CPU KV offloading on the decode node, with the same 512 GiB of host memory either way (8 $\times$ 48 GiB of HiSparse pool + 8 $\times$ 16 GiB of offload for HiSparse). Every configuration uses the same 16 GPUs, host memory, and vLLM build.

<!--RLC: "(eight DP ranks with EP8 and 64 GiB of CPU KV offload each, the same 512 GiB in total)" Lucas, can you double-check this is right? It is a bit unclear -->

## Results

We ran the SemiAnalysis [AgentX](https://vllm.ai/blog/2026-09-08-vllm-agentx) benchmark: Claude Code traces replayed with prefix reuse, a fresh server per concurrency, one-hour runs, and MTP3 with a synthetic acceptance length fixed at 2.99, so every configuration gets the same speculative speedup. Throughput is logical tok/s per GPU (including cached): input plus output tokens, counting prefix-cached input, over all 16 GPUs. Interactivity is AgentX's P90 E2E-normalized output tokens/s per user, the x-axis of Figure 5.

<p align="center">
<img src="/assets/figures/2026-10-23-glm53-part2-pcp-dcp-hisparse/agentx-pareto.svg" alt="AgentX throughput per GPU against interactivity at 142K and 1M context for the four prefill and decode combinations" width="100%">
<br>
<em>Figure 5. One point per concurrency, each on a fresh deployment: c8–c64 at 142K and c8–c32 at 1M. Labels give the concurrency for PCP8 prefill with HiSparse decode; the other configurations run the same concurrencies, right to left. Hollow points are off their own configuration's Pareto frontier.</em>
</p>

Up to c32, the deployment is mostly prefill-bound, so the prefiller makes the bigger difference. With HiSparse decode in both setups, PCP8 × DCP8 prefill gives 1.7–3.2× the interactivity of DP8 at the same throughput and cuts median TTFT from 1.5–5.2 s to 0.8–0.9 s. The decoder makes little difference at this load: at c8 and c16 the two PCP8 × DCP8 configurations are within a few percent of each other, and offload-only decode is slightly ahead at c8.

From c32 the deployment becomes more decode-bound, and HiSparse decode pulls ahead. At c64 it nearly doubles throughput, to 8,340 logical tok/s/GPU at 29.6 tok/s/user, while every other configuration drops to 1–2 tok/s/user.

At 1M context, the deployment needs both. A DP8 prefill rank cannot hold one 1M-token request, and neither can an offload-only decode rank.

## Making it all compose

Much of the challenging work in vLLM is making features like these compose. Each changes how the KV cache is laid out, scheduled, or moved, and every combination has to stay correct and fast.

For P/D, the KV has to change layout between the two engines. The prefill engine shards each request's KV by token across eight GPUs. Each decode rank needs the whole KV, including the MTP layer's, in its own paged layout. The NIXL connector does this reassembly between the two nodes, and prefix reuse from CPU offload has to keep working alongside it.

Every configuration here runs on [Model Runner V2](/blog/2026-03-24-mrv2). It keeps model-specific state out of the common execution path and runs speculative decoding without CPU–GPU syncs. Hybrid HiSparse is built on it.

## Try it

Everything runs on the `vllm/vllm-openai:nightly` image, any build from October 6 on. The results above were measured on vLLM main at `9e38b4111a`. The [reproduction gist](https://gist.github.com/LucasWilkinson/b39792adeea1cc2b1505fb6b7a2a2654) pins `2537629eb2`. The prefill engine's key flags:

```bash
--prefill-context-parallel-size 8 \
--decode-context-parallel-size 8 \
--dcp-comm-backend ag_rs \
--enable-expert-parallel \
--attention-config '{"sparse_mla_force_mqa":true}'
```

Without `sparse_mla_force_mqa`, prefill batches whose prompts fit within the 2,048-token top-k take the dense MHA path. The flag keeps them on the sparse-MLA (MQA) kernel too, so every KV row serves all heads.

## Acknowledgements

This work was done by Roberto L. Castro (Red Hat), Simon Veitner (Red Hat), Matthew Bonanni (Red Hat), Lucas Wilkinson (Red Hat), and Nicolò Lucchesi (Mistral). We thank Maroon Ayoub (Red Hat) and Robert Shaw (Red Hat) for stress testing the deployment, reporting errors, and fixing bugs.

## Notes

[^intensity]: Per query, sparse MLA reads 2,048 selected KV rows of 660 bytes each (656 of FP8 KV in FlashMLA's `fp8_ds_mla` format, plus a 4-byte index), about 1.35 MB. Each head adds 2,176 bytes for its BF16 query and output, and does 2,176 FLOPs per row (QK over 576 dims, PV over 512). So 8 heads do 35.7 MFLOPs over 1.37 MB, about 26 FLOPs per byte, and 64 heads do 285 MFLOPs over 1.49 MB, about 191.
[^ridge]: For the H200 SXM: 989 dense BF16 TFLOP/s (the datasheet's 1,979 assumes 2:4 sparsity) $\div$ 4.8 TB/s $\approx$ 206 FLOPs per byte ([NVIDIA H200 datasheet](https://resources.nvidia.com/en-us-hopper-architecture/hpc-datasheet-sc23)).
[^kvsize]: Per token: 78 layers $\times$ 656 B of MLA KV, 21 indexer layers $\times$ 132 B of indexer keys (128 FP8 values plus a 4-byte scale), and 788 B for the MTP layer, or 54,728 B in total. For $2^{20}$ = 1,048,576 tokens, that is 54,728 MiB $\approx$ 53.4 GiB.
