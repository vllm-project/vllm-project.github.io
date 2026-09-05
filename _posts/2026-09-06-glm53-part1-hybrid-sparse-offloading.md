---
layout: post
title: "GLM-5.3 Optimizations, Part 1: Hybrid HiSparse Offloading in vLLM"
author: "vLLM Team"
summary: "vLLM integrates HiSparse as a pressure-driven memory tier that composes with the Hybrid Memory Allocator and KV offloading, letting GLM-5.3 requests keep decoding when their KV no longer fits in GPU memory, so concurrency stays high."
image: /assets/figures/2026-09-06-glm53-part1-hybrid-sparse-offloading/hisparse-residency.svg
tags:
  - glm
  - kv-cache
  - performance
---

<!-- DRAFT. -->

**TL;DR:** vLLM is on a mission to make inference faster and cheaper to serve; this two-part series covers new GLM-5.3 optimizations toward that goal, with Part 1 focused on deploying on memory-constrained hardware. GLM-5.3's sparse attention reads only the top-K token rows per decode step, which greatly reduces the cost of attention at long contexts. [HiSparse](https://arxiv.org/abs/2608.07009) exploits the locality of those selections to keep cold KV in CPU memory and the rows attention actually needs in a small GPU hot cache. This means we can keep concurrency high because single requests can keep decoding even when not fitting entirely in GPU memory.
Hybrid HiSparse builds upon vLLM's Hybrid Memory Allocator. GPU pages are released only if the shared pool is under pressure, and statically reserving blocks for requests is abandoned in favour of leasing out hot-cache blocks. Preparing completed prefix pages on the host while keeping their copies on the GPU keeps low-pressure scenarios on the fast GPU path, which means they incur little overhead under our mechanism. Part 2 puts this together with PCP, DCP, and adaptive verification in large-scale GLM-5.3 deployments.

With vLLM, we are on a mission to make inference easier and cheaper to serve. That means making inference both faster and easier to deploy on more constrained hardware. In this two-part series we cover new optimizations we've introduced for GLM-5.3 in pursuit of that goal: in Part 1 we demonstrate how Hybrid HiSparse assists with an aggregated deployment on a single 8× H200 node, which is tight on memory for a model of this size.

## Exploiting sparsity when we need to

Agentic workloads inherently involve many concurrent requests, each with a long context that keeps growing. Because the GPU block pool is fixed, the KV cache will eventually run out of room for concurrent requests to allocate new blocks.

So far there have been two main options for addressing this issue, each with its own tradeoffs:

* **Preemption** picks a request, drops its KV cache, and re-prefills it later. The request pays its full TTFT again on every eviction.
* **Offloading** moves blocks out to host memory, but dense attention requires every token to be resident on the GPU, so the number of concurrent requests remains bounded by GPU memory.

For sparse-MLA KV cache the indexer selects top-K tokens and only these needs to reside on GPU. [HiSparse](https://arxiv.org/abs/2608.07009) exploits this behaviour by keeping everything except the selected tokens on the CPU which gives us an effective upper bound for the GPU memory each request needs. The indexer KV does stay GPU-resident and still grows with context length, but it is much smaller overall, and GLM-5.3's [IndexShare](https://arxiv.org/abs/2603.12201) means there is only one index layer per four sparse-MLA layers. 

We introduce Hybrid HiSparse, which additionally keeps block on the GPU as long as there is enough capacity. Only when KV cache is under pressure we apply above described HiSparse mechanism to only keep on the GPU what the indexer selects.  Hot buffer pages are indexed by tokens and thus a page can hold tokens drawn from many different CPU block, which leads to reduction across a wide span of context. Thus Hybrid HiSparse only pays the cost of CPU-GPU memory transfers when the system is under KV-cache pressure, i.e. higher concurrency.

<figure>
  <img src="{{ '/assets/figures/2026-09-06-glm53-part1-hybrid-sparse-offloading/hisparse-two-requests.svg' | relative_url }}" alt="One pool, two growing requests: preempt vs offload" style="width: 100%;">
  <figcaption><em>Preemption: B's slots are freed and its KV is gone. Conventional offload: B's KV survives on host and we don't need to re-prefill but B still can't run until all of it fits on GPU again, so A decodes alone. Hybrid sparse offload: each request releases its coldest pages in place, the same slots are re-leased as new tails and hot pages, and both keep decoding.</em></figcaption>
</figure>

Only Hybrid HiSparse keeps both requests decoding. The hot pages are leased from the same block pool as the KV pages, and more importantly they live in the same KV-cache tensor, so they look like ordinary pages to the sparse MLA kernel. Unique to hybrid sparse, some tokens can exist in the hot buffers while some tokens can still exist in GPU-resident pages, reducing the amount of CPU reloading.

## How it works

<figure>
  <img src="{{ '/assets/figures/2026-09-06-glm53-part1-hybrid-sparse-offloading/hisparse-residency.svg' | relative_url }}" alt="Three KV residency states over one shared GPU block pool" style="width: 100%;">
  <figcaption><em>The same six top-K tokens are ringed in every panel; only their residency changes. Solid arrows: misses copying one row into a hot page. Dashed arrows: hot hits reused without a copy.</em></figcaption>
</figure>

Residency is tracked per page, so a request moves between three states as pressure rises and falls:
* **Full residency**: all sparse-MLA KV remains GPU-resident while completed prefix pages are proactively materialized in host memory.
* **Mixed residency**: the tail of the request stays on the GPU, older pages live only in CPU memory, and the rows the indexer wants from those pages sit in hot buffers. The block table holds real blocks and null placeholders side by side, and the tail is never evicted. One fused kernel resolves the top-K: resident tokens are read in place, hot tokens are read and their LRU entry refreshed, and a miss copies a single row from pinned host memory into an LRU slot. Nothing on the decode path waits on a CPU decision, so it stays CUDA-graph-capturable.
* **No residency**: a new request reusing a prefix that only exists in CPU memory starts with placeholders and a hot page. Rows arrive as the indexer selects them, so we pay for what the model attends to rather than the whole history.

All three states work because the hot buffers are not a separate allocation. A hot buffer page is an ordinary KV-cache block, leased from the same pool as the resident pages through vLLM's Hybrid Memory Allocator, taken when a request first needs one and returned when it does not. Both for rows sitting in resident page or the hot buffer the resolver hands it row IDs and it gathers them with one stride. A block freed by one request can become hot-buffer capacity for another.

HiSparse prepares for pressure before it arrives. When a cacheable prefix page is complete, HiSparse queues a copy to CPU memory while continuing to serve it from the GPU. If the GPU cache later fills up, that page can release its GPU slot without another copy. Even if pressure reaches a newer page first, its GPU slot becomes reusable as soon as the copy is queued, and the CPU copy becomes available for prefix reuse when the transfer completes.

The `hisparse-glm` branch keeps this path lightweight by copying all sparse-MLA layers together in one launch after the forward pass. The copy is ordered on the model's GPU stream, which keeps synchronization simple and safe.

## Composing with the rest of vLLM

Hybrid HiSparse is a residency policy over the shared HMA pool and a connector alongside vLLM's other KV machinery, so the rest of the stack keeps working as it did. Other cache groups still use normal prefix caching, transfer, and offloading, and the indexer KV in particular is untouched by HiSparse: the standard OffloadingConnector can offload it independently with ordinary block-granular storage. Imports from P/D disaggregation can land host-side when a prefix does not fit resident, and speculative decoding works through per-step replayable resolver plans that share the request's hot state.

Hot buffers default to 2x top-K rows per request, which ensures high hit rates while keeping the buffer size small. Since MLA KV is identical across TP ranks, the pinned host pool is allocated per DP replica and shared across its local TP ranks. TP rank 0 writes the shared copy, every rank can read it, and a CUDA event preserves stream ordering.

## Estimate the benefit for your configuration

The calculator below estimates ordinary GPU-resident KV and hybrid sparse offloading capacity using the same available HBM. Adjust the workload, GPU, parallelism, hot buffer, and host pool to approximate a deployment. Adjusting the values gives a sense of the potential increase in concurrency.

The calculator exposes two useful thresholds. The minimum host pool is the capacity required to keep CPU memory from limiting the concurrency that the GPU-side indexer and hot buffers can sustain. The plot assumes this non-limiting host capacity at each sequence length and exposes a second threshold: hot buffers add a fixed GPU cost per request, so at short contexts ordinary GPU-resident KV may fit more requests; beyond the crossover, bounding sparse-MLA residency outweighs that fixed cost and the multiplier rises above 1.0×. Increasing the hot buffer moves this crossover to longer sequences and reduces the concurrency multiplier, trading capacity for greater hot-cache coverage.

> [!NOTE]  
> These are planning estimates, not guaranteed serving limits: runtime workspaces, request-length skew, and scheduling behavior can lower the concurrency reached in practice.

<iframe
  src="{{ '/assets/interactive_pages/hisparse_concurrency_calculator.html' | relative_url }}"
  title="Hybrid sparse offloading concurrency calculator"
  loading="lazy"
  width="100%"
  height="1520"
  style="border: 0; border-radius: 12px;"
></iframe>

[Open the concurrency calculator full-screen]({{ '/assets/interactive_pages/hisparse_concurrency_calculator.html' | relative_url }})

## The numbers

We benchmarked GLM-5.3 on 8× H200 using an OpenHands-style agentic workload: 13-turn conversations with a 74,160-token first turn, 753-token later turns, and fixed 220-token outputs. Both TP8 deployments used MTP3, FP8 KV cache, a 142K admission limit, `max_num_batched_tokens=32768`, `max_num_seqs=256`, and `gpu_memory_utilization=0.92`. The offloading baseline used a 512 GiB offload pool; Hybrid HiSparse split the same host budget into a 384 GiB HiSparse pool and 128 GiB of offloading.

<figure>
  <img src="{{ '/assets/figures/2026-09-06-glm53-part1-hybrid-sparse-offloading/openhands-pareto-occupancy.svg' | relative_url }}" alt="GLM-5.3 interactivity-throughput Pareto and measured concurrent running requests for Hybrid HiSparse and KV offloading" style="width: 100%;">
  <figcaption><em>Top: the interactivity-throughput sweep.
</figure>


We are planning to make Hybrid HiSparse widely available in vLLM v0.30. In the meantime, the exact launch commands and benchmark client setup used for these results are in the [reproduction appendix](#appendix-reproducing-our-results) below.

## Offloading only where we need it

Hybrid HiSparse only offloads where we need it. KV starts on the GPU and stays there while there is room, then gives up residency page by page as the pool runs short. Hot buffers and resident pages share pool and tensor and thus a request under pressure keeps decoding at partial residency instead of waiting for a slot to free up or paying to prefill itself again.

This is the first post in a series on serving GLM-5.3 with vLLM. Hybrid HiSparse matters most on the decode side of a P/D deployment, where contexts are longest and KV pressure is highest. In Part 2 we put the pieces together on large-scale deployments, combining new and existing optimizations: Prefill Context Parallelism (PCP), Decode Context Parallelism (DCP), [adaptive verification](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification), and Hybrid HiSparse.

## Acknowledgements

vLLM's Hybrid HiSparse implementation was developed by Matthew Bonanni (Red Hat), Lucas Wilkinson (Red Hat), and Fares Obeid (Prime Intellect). The design was shaped through close collaboration with Chao Lei (Huawei) and Nicolò Lucchesi (Mistral). Simon Veitner (Red Hat) contributed to the performance evaluation and development of this blog. We thank the [HiSparse](https://arxiv.org/abs/2608.07009) authors for developing the sparse offloading concept employed as part of this work.

## Appendix: Reproducing our results

The results above use [vLLM `e8ef1e07bd`](https://github.com/neuralmagic/vllm/commit/e8ef1e07bd2f174bebfe34c3a3e35e952931efb1). We are planning to make Hybrid HiSparse widely available in vLLM v0.30; until then, build the pinned checkout above. Launch the Hybrid HiSparse configuration on one 8× H200 node with:

```bash
vllm serve zai-org/GLM-5.3 \
  --served-model-name glm-agentx \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 142000 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --attention-config '{"hisparse_config":{"host_pool_gib":384}}' \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":137438953472}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

`host_pool_gib` is per DP replica and is rounded to whole host blocks. The 128 GiB offloading pool stores cache groups that HiSparse does not manage, including the indexer KV. To reproduce HiSparse without MTP, omit `--speculative-config`. For the no-HiSparse MTP3 baseline shown in the figure, keep `--speculative-config`, omit `--attention-config`, and change `cpu_bytes_to_use` to `549755813888` (512 GiB). Omit both HiSparse and `--speculative-config` for the no-MTP baseline. HiSparse is currently implemented only for NVIDIA GPUs.

### Reproducing the OpenHands sweep

Everything the benchmark client needs ships with this blog so the recipe is self-contained: [`build_openhands_padded_dataset.py`]({{ '/assets/repro/2026-09-06-glm53-part1-hybrid-sparse-offloading/build_openhands_padded_dataset.py' | relative_url }}), [`install_evalscope_deps.sh`]({{ '/assets/repro/2026-09-06-glm53-part1-hybrid-sparse-offloading/install_evalscope_deps.sh' | relative_url }}), and [`evalscope-all-nodeps.txt`]({{ '/assets/repro/2026-09-06-glm53-part1-hybrid-sparse-offloading/evalscope-all-nodeps.txt' | relative_url }}). Download all three into one directory. EvalScope is pinned at `acd09b44384d53174768bb1063f675420f76fae9`. The following builds the deterministic 128-conversation dataset, then runs c1/c8/c16/c24/c32 with fresh conversations at every point:

```bash
python3.12 -m venv client-venv
source client-venv/bin/activate
bash install_evalscope_deps.sh
pip install 'modelscope[datasets]==1.34.0' 'lxml==6.0.2'
pip install 'evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@acd09b44384d53174768bb1063f675420f76fae9'

python build_openhands_padded_dataset.py \
  --model zai-org/GLM-5.3 \
  --pad-source openscience \
  --first-turn-length 74160 \
  --subsequent-turn-length 753 \
  --num-turns 13 \
  --number 128 \
  --output-path openhand-zai-org-GLM-5.3.json

evalscope perf \
  --model glm-agentx \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --dataset swe_smith \
  --dataset-path openhand-zai-org-GLM-5.3.json \
  --dataset-offset 52 \
  --max-tokens 220 \
  --multi-turn \
  --number 4 16 32 48 64 \
  --parallel 1 8 16 24 32 \
  --extra-args '{"ignore_eos":true}' \
  --name tp8-hisparse384-native128 \
  --outputs-dir results \
  --no-timestamp
```

For the figure, interactivity is `1000 / mean_TPOT_ms`; logical total-token throughput per GPU is EvalScope's total token throughput divided by eight. We scraped `/metrics` every 30 seconds during each point. Request occupancy is the mean of non-zero `vllm:num_requests_running` samples, and MTP acceptance length is `1 + Δ(vllm:spec_decode_num_accepted_tokens_total) / Δ(vllm:spec_decode_num_drafts_total)`.
