---
layout: post
title: "Serving DeepSeek-V4.1-Flash at Scale: vLLM Matches DeepInfer and Scales to GB300 NVL72"
author: "Inferact and the vLLM Team"
date: 2026-10-23 09:00:00 +0000
summary: "With P/D disaggregation and Wide-EP on a GB300 NVL72 rack, vLLM serves DeepSeek-V4.1-Flash on SemiAnalysis AgentX at 954K tokens/s per GPU at a P90 of 196 tokens/s per user, 7.9x our aggregated single-node result."
image: /assets/figures/2026-10-23-deepseek-v41-flash-pd-wide-ep/hero.png
social_image: /assets/figures/2026-10-23-deepseek-v41-flash-pd-wide-ep/hero.png
tags:
  - deepseek
  - performance
  - large-scale-serving
  - disaggregation
---

<p align="center">
<img src="/assets/figures/2026-10-23-deepseek-v41-flash-pd-wide-ep/hero.png" alt="7.9x throughput for DeepSeek-V4.1-Flash with P/D and Wide-EP on a GB300 NVL72 rack, measured on SemiAnalysis AgentX, and on par with DeepInfer" width="100%">
</p>

**TL;DR:** On one NVIDIA GB300 NVL72 rack, vLLM with P/D disaggregation and Wide-EP serves DeepSeek-V4.1-Flash on SemiAnalysis AgentX at **954K tokens/s per GPU** at a P90 of 196 tokens/s per user, **7.9×** our single-node result from the [first post](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash) and **$38.96 per GPU-hour** at DeepSeek's off-peak prices. A specialized decode path reaches **826 tokens/s** for a single user. Against DeepInfer, DeepSeek's own engine, vLLM's decode throughput is within **6%** across batch sizes 8–256, and its prefill tracks DeepInfer's published curve.

The efficiency comes from deliberate engineering on components that vLLM already has, built on the kernels DeepSeek open-sourced. This post walks through our performance study, breaks down each optimization, and shares the lessons we learned:

- **A general-purpose engine can match a model-specific one**
- **Understand the model before optimizing**
- **Small DEP instances work best for prefill**[^terms]
- **Wide-EP decode scales well with EPLB**
- **CPU overhead becomes critical at NVL72 scale**
- **A specialized decode path unlocks low latency**

## Performance overview

<iframe class="vllm-embed" src="/assets/interactive_pages/dsv41-pd-agentx-curve.html" title="SemiAnalysis AgentX on NVIDIA GB300 NVL72, P/D and Wide-EP vs. the aggregated single-node result" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*Figure 1: SemiAnalysis AgentX on NVIDIA GB300 NVL72: vLLM with P/D disaggregation and Wide-EP (gold) achieves a further 7.9× throughput increase over our aggregated single-node result from the [first post](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash). Throughput counts cached input, uncached input and output tokens ([source](https://inferencex.semianalysis.com/inference/deepseek-v41-flash?i_xmode=interactivity&i_seq=agentic-traces&g_model=DeepSeek-V4.1-Flash&i_best=0&i_active=gb300_dynamo-vllm)).*

### Compared to DeepInfer

[DeepInfer](https://github.com/deepseek-ai/DeepInfer) is DeepSeek's own inference engine for DeepSeek-V4.1-Flash, released this week as an offline engine with published prefill and decode curves. It is the most relevant reference for this model, so before going into our optimizations we show where vLLM stands against it.

**Decode.** Under DeepSeek's decode configuration (131K context, nextn=3), vLLM on GB200 achieves throughput within 6% of DeepInfer's published results across batch sizes 8 to 256 (Figure 2).

<iframe class="vllm-embed" src="/assets/interactive_pages/dsv41-pd-decode-vs-deepinfer.html" title="vLLM on GB200 DEP32 tracks DeepSeek's decode curve" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*Figure 2: Decode output throughput per GPU vs. batch size at 131K context, nextn=3. vLLM on GB200 vs. DeepInfer's published results, read from the charts in its README ([source](https://github.com/deepseek-ai/DeepInfer)).*

**Prefill.** DeepInfer's published prefill curve uses EP4DP4 with 32 redundant experts and a 140-token bounded replay, on hardware the chart does not state, and we read its values off the chart. Our run uses DEP2 on GB300 with no redundant experts, a 128-token replay and our own prompts. Because the setups differ, we do not read the gap between the two curves as an engine-to-engine speedup. What Figure 3 shows is that vLLM's DEP2 prefill tracks DeepSeek's own engine across the range they published.

<iframe class="vllm-embed" src="/assets/interactive_pages/dsv41-pd-prefill-vs-deepinfer.html" title="Prefill curve: vLLM DEP2 vs. DeepInfer EP4DP4" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*Figure 3: Prefill throughput vs. new tokens. vLLM DEP2 on GB300 vs. DeepInfer's published curve (EP4DP4, 32 redundant experts, 140-token replay, hardware not stated), read from the chart in its README ([source](https://github.com/deepseek-ai/DeepInfer)).*

The two engines land close because they share a lineage in both directions. DeepInfer was originally built on vLLM, and vLLM in turn has continuously integrated kernels from DeepSeek's libraries, including MegaAttn, MegaMoE and Mega-mHC, and has extensively optimized the rest of the stack around them. DeepInfer targets a single model in offline mode; the same vLLM build also serves this model online with P/D disaggregation and Wide-EP across a full NVL72 rack, which is what the rest of this post measures. The sections that follow describe that work.

## Under the hood: where the speed comes from

### Measure the model first

Before tuning, we started by quantifying DeepSeek-V4.1-Flash's efficient architectural designs:

- **Small KV cache footprint.** Only 890 bytes per token for global KV cache, thanks to cross-layer KV sharing and FP4 compression (see the [DeepSeek-V4.1-Flash Technical Report](https://arxiv.org/abs/2609.19969)).
- **Lightweight attention.** Layers 2, 8 and 14 use 2× compression for full-context indexing. With bounded replay, layer 20 writes global KV for all tokens but runs indexing and attention only on the last 128; layers 21–39 process only those tokens.
- **Decode is MoE-bound even in a long context.** See Figure 5 below. With DEP4 decoding at 128K context and 64 requests per rank, MoE takes 62% of the target step, while attention and the indexer together take only 13%.

[^terms]: DEP*n* is data-parallel attention plus expert parallelism over *n* GPUs. Target forward is the main model's full forward pass at each decode step, which verifies the draft tokens and produces the output; draft forward is the DSpark draft model's full forward pass, which proposes the tokens for the next step.

The table below shows the target model's forward time at DEP32, with 16 requests per rank, 6 tokens per request (one token plus five DSpark draft tokens), and every request at the same context length, on GB300. Going from 128K to 855K, 6.7× the context only adds 0.6 ms (+7%) to the target model decode latency.

| Context per request | 64K | 128K | 256K | 512K | 855K |
|---|---:|---:|---:|---:|---:|
| Target forward (ms) | 8.86 | 8.93 | 9.03 | 9.25 | 9.53 |
| vs. 128K | −0.8% | – | +1.1% | +3.6% | +6.7% |

These characteristics shape the rest of our optimization strategy:

- **Prefill should run at a small DEP size.** Even a DEP2 instance on B300 or GB300 can hold both the model weights and the KV cache for many concurrent sessions. A smaller world size greatly simplifies DP/EP load balancing.
- **KV capacity rarely limits batch size.** The KV cache fits comfortably in HBM, allowing us to push each operation toward its compute or memory bandwidth limits instead of working around KV capacity. This also keeps our sweep consistent and low-noise with no preemption or KV cache pressure.
- **Wide-EP is effective for decode.** MegaMoE decode is bound by weight reads, and Wide-EP directly cuts this dominating MoE cost.
- **Decode DP load balancing is simpler.** This model's decode step time varies little with context length, so DP load balancing can be simplified to least-load by request count per rank.
- **Synthetic workloads remain representative for decode.** Using average input and output lengths gives us a good estimate of the performance without replaying the full agentic trace.

#### How we measured

Measuring performance accurately without overcomplicating the methodology is not trivial. We break it into two questions: what should a prefill or decode step cost, and how much additional overhead comes from DP and EP imbalance?

- **Prefill and decode step time.** `vllm bench forward` ([#61339](https://github.com/vllm-project/vllm/pull/61339)) measures the model-forward time for an exact prefill or decode batch through the actual `execute_model` path across DP and EP ranks. This measures GPU time only and gives a clean trace. Further, vLLM's `DecodeBenchConnector` measures engine-level end-to-end decode performance at a given concurrency. The gap between GPU-only and serving step times reveals unexpected overhead.
- **Prefill DP imbalance: mirror vs. lag replays.** Agentic traces vary from run to run, so DP imbalance is hard to isolate. We therefore replay identical requests on every DP rank. In a *mirror* run, all ranks receive their requests at the same time. In a *lag* run, rank *r* receives the same requests *r* × 5 s later, so the ranks are out of step but do the same total work. The difference between the two runs is the cost of DP imbalance.
- **Decode EP imbalance: real vs. uniform routing.** For each MoE layer, we define balancedness as the mean rank load divided by the max rank load, per step and over EPLB's window. As the balanced reference, we run the same decode batch with tokens routed uniformly at random. This reference still has sampling noise: at DEP32 with 72 requests per GPU, uniform routing reaches about 0.96 balancedness for the target (top-6 routing) and 0.94 for the draft (top-3). The gap between real and uniform routing is the most that expert balancing can recover, and the EPLB run shows how much of that gap it actually recovers.

### Prefill: DEP2 works best

As discussed above, a small DEP size is appealing for prefill. In an AgentX prefill-only run at concurrency 256, DEP4 gave only 83% of DEP2's per-GPU throughput (53.3K vs. 64.4K tokens/s). Mirror and lag replays with prefix caching off point to DP imbalance: when the ranks fall out of step, DEP4 loses 13.6% of its throughput.[^earlier-build] This makes DEP2 a simple default for us.

| Prefill throughput per instance (tokens/s) | mirror (balanced) | lag (imbalanced) | loss |
|---|---:|---:|---:|
| DEP2 | 138.7K | 130.5K | 5.9% |
| DEP4 | 267.6K | 231.1K | 13.6% |

[^earlier-build]: Measured on an earlier build; the imbalance cost is a property of DEP size, not of the kernels.

#### Optimizations hidden behind the trace

Our [first post](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash) covered a broad set of prefill optimizations. Cleaner traces helped us identify the remaining bottlenecks, particularly those that grow with context length. For example, on two GB300 GPUs, an 8K-token prefill takes **87.9 ms without prior context**, but **189.4 ms with 512K cached context**. We addressed three additional bottlenecks:

<details markdown="1">
<summary><strong>Extend SWA bounded replay to layer 20</strong> (<a href="https://github.com/vllm-project/vllm/pull/59894">#59894</a>): +19.9% at 32K new tokens</summary>

SWA bounded replay already limits layers 21–39 to the last 128 tokens. Previously, layer 20 still performed indexer computation across all new tokens. Now it writes KV for all new tokens but runs its remaining computation only on the 128-token replay batch. This improves prefill throughput by **19.9% at 32K new tokens**, with gains increasing from 10.1% at 4K to 23.1% at 64K.

</details>

<details markdown="1">
<summary><strong>Use 2 MB pages for shared Engram host tables</strong> (<a href="https://github.com/vllm-project/vllm/pull/62359">#62359</a>): another +2.8% at 32K</summary>

Engram performs random reads from host tables shared across DP ranks. The previous `/dev/shm` mapping used 64 KB GPU pages, causing frequent GPU TLB misses. Moving the tables to host-NUMA cuMem allocations enables 2 MB GPU pages, improving prefill throughput by another **2.8% at 32K new tokens**.

</details>

<details markdown="1">
<summary><strong>Use DeepSelect top-k in the prefill indexer</strong> (<a href="https://github.com/vllm-project/vllm/pull/62240">#62240</a>): another +15.6% at 32K</summary>

The prefill indexer now uses DeepSelect to rank candidate blocks, matching the decode path. Its benefit grows with the number of keys being ranked, improving prefill throughput by another **15.6% at 32K new tokens** and 15.2% at 64K.

</details>

Together, these changes improve **new-token prefill throughput by 35.6–43.9%** across 4K–64K new tokens on top of a 128K cached prefix. At 32K, the combined gain is **42.5%**; at 64K, throughput increases from **71.2K to 102.5K tokens/s/GPU (+43.9%)**.

<iframe class="vllm-embed" src="/assets/interactive_pages/dsv41-pd-prefill-ablation.html" title="Prefill gains on DEP2 after a 128K cache hit" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*Figure 4: New-token prefill throughput per GPU on DEP2 after a 128K cache hit, adding layer 20 replay, Engram 2 MB pages and DeepSelect top-k one at a time.*

### Decode: scaling Wide-EP and balancing experts with EPLB

On the decode side, Wide-EP with EPLB is an established practice, described in [DeepSeek's inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md). vLLM has supported both for a while (see our [large-scale serving post](https://vllm.ai/blog/2025-12-17-large-scale-serving#wide-ep)), and we found that they still work well. This section focuses on our new findings and empirical lessons.

#### The scaling of Wide-EP

We measured decode scaling from DEP4 to DEP32 with 64 requests per GPU at 128K context (with EPLB enabled; see the next section). Target MegaMoE time drops from 13.4 to 6.0 ms as each rank holds fewer experts, while the remaining work stays around 10 ms. As a result, per-GPU throughput gains gradually diminish. At DEP32, the draft's communication overhead also starts to offset the benefits of wider parallelism.

<iframe class="vllm-embed" src="/assets/interactive_pages/dsv41-pd-wide-ep-scaling.html" title="Decode step time breakdown from DEP4 to DEP32 on GB200" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*Figure 5: Decode step time per rank from DEP4 to DEP32 on NVIDIA GB200 with 64 requests per GPU at 128K context, and output throughput per GPU.*

#### EPLB

We let EPLB rearrange the experts every 3,000 steps, based on the load of the previous 1,000 steps. EPLB raises balancedness from 0.35 to 0.94 for the target and from 0.21 to 0.89 for the draft, recovering about 97% and 93% of the gap to uniform routing.

<iframe class="vllm-embed" src="/assets/interactive_pages/dsv41-pd-eplb.html" title="EPLB at wide EP: balance, step time and throughput" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*Figure 6: Decode on GB300 NVL72 with EPLB off, EPLB for the target model only, and EPLB for both the target and the DSpark draft. The tabs show per-step balance, the per-step time breakdown at DEP32, and throughput (mean of three runs; bars show the range).*

Two changes make this work. [#62009](https://github.com/vllm-project/vllm/pull/62009) introduces an NVLink-based `symm_mem` backend that rebalances all of DEP32 in under a second without stopping decode, fuses EPLB's replica mapping and load recording into the MegaMoE kernels (saving about 0.2 ms per step), and removes host synchronization from the decode loop. [#61395](https://github.com/vllm-project/vllm/pull/61395) supports EPLB with heterogeneous target and draft models. The DSpark draft has 128 routed experts with top-3 routing, against the target's 384 with top-6, so EPLB tracks expert placement and load separately for each model and rebalances both.

### Overcome CPU overhead in fast large-scale decode

Wide-EP decode synchronizes all ranks at every layer, so one slow rank stalls the instance. Async scheduling hides CPU overhead well at a small scale, but more ranks make this harder in two ways: GPU steps get shorter, which leaves less slack for CPU work, and it becomes more likely that one rank falls behind.

In practice, we observed that an unoptimized DEP32 ran slower than DEP16 (17.57 vs. 13.84 ms per step), although its GPU execution was 1.1 ms faster. The table below lists every cause we found, ordered by effect. The effects come from different setups, so each row gives its own.

<details markdown="1">
<summary>Show the causes, fixes and their effects</summary>

| Cause | Fix | PRs | Effect | Setup |
|---|---|---|---|---|
| KV read posted synchronously on the decode worker thread | NIXL push with UCX scatter-gather; packed KV rows sent as one NIXL region | [NIXL push connector](https://vllm.ai/blog/2026-10-14-nixl-push-connector), [#61121](https://github.com/vllm-project/vllm/pull/61121) | 141K → 440K tokens/s per GPU; P90 ~40 → 159 tokens/s per user | GB300, 8P1D, concurrency 3,072 |
| Gloo all-reduce latency grows linearly with the rank count | Recursive-doubling CPU all-reduce | [#61767](https://github.com/vllm-project/vllm/pull/61767), building on [#60267](https://github.com/vllm-project/vllm/pull/60267) | DP sync 10.8 → 1.5 ms per step; step 16.10 → 13.15 ms | DEP32 decode |
| Admission bursts: up to 140 requests admitted in one step | Opt-in cap of 4 new KV loads per step | [#61725](https://github.com/vllm-project/vllm/pull/61725) | 460K → 516K tokens/s per GPU; P90 109 → 153 tokens/s per user | GB300, 7P1D, concurrency 4,800, with two other CPU fixes |
| Admission work that grows with the prompt length | Prompt as one int32 buffer; skip the null blocks of sliding-window block tables; chunked prompt copy kernel | [#59908](https://github.com/vllm-project/vllm/pull/59908), [#62702](https://github.com/vllm-project/vllm/pull/62702), [#61724](https://github.com/vllm-project/vllm/pull/61724), [#62698](https://github.com/vllm-project/vllm/pull/62698) | 855K prompt: 38.8 → 6.0 ms of CPU per admission | Engine microbenchmark |

</details>

### Low latency: a specialized decode path for small batches

We also extended low-latency kernel support for DeepSeek-V4.1-Flash ([#61868](https://github.com/vllm-project/vllm/pull/61868) and [#61869](https://github.com/vllm-project/vllm/pull/61869)), continuing the efforts from our [previous post](https://vllm.ai/blog/2026-10-16-kimi-k3-megakernel). On real AgentX sessions, the specialized decode path reaches 826 P90 tokens/s for a single user on four B300 GPUs, 44% higher than the standard decode path (575).

<iframe class="vllm-embed" src="/assets/interactive_pages/dsv41-pd-low-latency.html" title="P90 interactivity on SemiAnalysis AgentX with TP4 on B300 at concurrency 1, 8 and 16, before and after InferenceX #4332" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>

*Figure 7: SemiAnalysis AgentX with TP4 on four B300 GPUs at concurrency 1, 8 and 16, before and after [InferenceX #4332](https://github.com/SemiAnalysisAI/InferenceX/pull/4332) switched to the specialized decode path.*

## Acknowledgements

We would like to thank **DeepSeek** for open-sourcing and sharing DeepSeek-V4.1-Flash and its optimized kernels, which provided the foundation for this work, and for publishing DeepInfer and its performance results, which gave us a strong reference to learn and compare against. We also thank **SemiAnalysis** for the AgentX benchmark platform and all the help.

We are especially grateful to **Pei Li, Hanjie Qiu, Lisa Wang, Yaoyao Ding, Rohit Nagraj, and Xin Li** from **NVIDIA** for their close collaboration on performance optimization, continued support across vLLM and Dynamo, and assistance with the AgentX submissions. We also thank **Canlin Guo** from the **vLLM community** for his valuable prefill optimization.

This work was led by **Inferact**.

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
