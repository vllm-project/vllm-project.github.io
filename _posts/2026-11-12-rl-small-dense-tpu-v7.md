---
layout: post
title: "Reinforcement Learning: Optimizing 0.6B Dense Models on TPU v7 via vLLM"
author: "Loki Chen, Yang Song, Hongye Xie, Ming Lei, Greg Rehm, Chongyang Bai, Dre Olgiati, Bob Kamma, Tej Choudhary, Hang Cui, Imed Zitouni, Hamed Firooz (Meta ASA); Mingfu Liang, Kavosh Asadi, Yufei Li, Frank Shyu, Parish Aggarwal, Senthil Manickavelu, Xi Liu, Luke Simon (Meta Ranking AI Research)"
date: 2026-11-12 09:00:00 +0000
summary: "Co-designing RL for a Qwen3-0.6B dense policy on Google Cloud TPU v7 across vLLM, Tunix, MaxText, and Pallas: 23 upstream PRs and a 107x end-to-end speedup, from ~100 minutes to ~56 seconds per true-8K RL cycle."
image: /assets/figures/2026-10-09-rl-small-dense-tpu-v7/co-design-rl-0p6b-dense.webp
social_image: /assets/figures/2026-10-09-rl-small-dense-tpu-v7/co-design-rl-0p6b-dense.png
tags:
  - reinforcement-learning
  - performance
  - hardware
---

*Authors —*<br>
***ASA:** Loki Chen, Yang Song, Hongye Xie, Ming Lei, Greg Rehm, Chongyang Bai, Dre Olgiati, Bob Kamma, Tej Choudhary, Hang Cui, Imed Zitouni, Hamed Firooz.*<br>
***Ranking AI Research:** Mingfu Liang, Kavosh Asadi, Yufei Li, Frank Shyu, Parish Aggarwal, Senthil Manickavelu, Xi Liu, Luke Simon.*

**Takeaways**

- Small dense-model RL exposes new challenges across hardware architectures: decode scheduling, synchronization, attention geometry, and KV-cache lifecycle become first-order bottlenecks.
- By co-designing across the open-source TPU stack (**vLLM**, **Tunix**, **MaxText**, and **Pallas**) and upstreaming the work in **23 PRs**, developers spinning up Google Cloud TPU v7 today can reach the **56-second (~183K tokens/s)** path by enabling the upstreamed flags listed in [Appendix B](#appendix-b--how-to-reproduce)—without repeating this bring-up journey.
- The complete rollout-to-update cycle achieved a **107x end-to-end speedup** on Google Cloud TPU v7—falling from roughly 100 minutes to approximately 56 seconds—while preserving the full 8,192-token response budget.

## Why a small model matters for research

Generative reasoning is reshaping how ranking and decision systems work. Models like GR2 (Generative Reasoning Re-ranker) [[arXiv:2606.31984](https://arxiv.org/abs/2606.31984)] have demonstrated that generative reasoning can meaningfully improve re-ranking quality. The natural next question is: *for large-scale ranking and decision systems, how small can the policy model be without losing that generative reasoning capability?*

A 0.6B dense student distilled from stronger teachers could serve with dramatically lower latency and cost, enabling broader traffic coverage, faster model refreshes, and more RL experimentation loops per day. But realizing this requires that the *full training cycle*—not just model FLOPs—be fast. At 0.6B parameters, rollout generation, synchronization, and framework overhead increasingly dominate wall-clock time.

**This is where the system challenge begins.** Because a 0.6B model has far less matrix arithmetic per layer than frontier dense or MoE models, fixed system overheads—host dispatch, synchronization, and KV-cache lifecycle—shift the bottleneck from FLOPs to coordination.

## The model was small. The research challenge was not.

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-tpu-v7/why-small-dense-needs-co-design.webp" alt="Figure 1: Why small dense models need co-design" width="100%">
</p>

<p align="center">
<em>Figure 1: Why small dense models need co-design. Illustrative model-size comparison and RL-step composition.</em>
</p>

At first glance, a 0.6-billion-parameter model should be easy to run on a 64-chip Google Cloud TPU v7 (Ironwood) accelerator slice. The model has far fewer weights than the large dense and mixture-of-experts systems that usually motivate distributed training work. In practice, that is exactly what made this workload difficult.

For a large dense or MoE model, each layer performs enough matrix computation that device utilization is often dominated by the familiar questions: how to shard parameters, move activations, route experts, and keep large matrix units busy. Those problems are hard, but their costs scale visibly with the model.

A small dense model changes the balance. The useful matrix work per device shrinks, but many fixed costs do not:

- the host still schedules decode iterations;
- every token still traverses the attention stack;
- every layer still launches or enters compiled kernels;
- every rollout step still updates and synchronizes weights;
- the runtime still tracks thousands of sequences and their stopping conditions;
- the RL cycle still includes reward computation, training, cache invalidation, and distributed orchestration.

The smaller the model became, the more every boundary in the system mattered. A few milliseconds of dispatch overhead, an overly large attention compute block, or a cache allocation repeated on every update could dominate the work we wanted TPU v7 to perform.

This is the central thesis of the project: small dense-model RL is not simply a scaled-down version of large-model training. It is a coordination-sensitive workload that rewards hardware-software-model co-design.

## Why co-design mattered

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-tpu-v7/co-design-rl-0p6b-dense.webp" alt="Figure 2: Co-designing RL for a 0.6B dense model on Google Cloud TPU v7" width="100%">
</p>

<p align="center">
<em>Figure 2: Co-designing RL for a 0.6B dense model on Google Cloud TPU v7. Three stages of optimization: (1) configuration and mesh tuning (~4K → ~30K tok/s; ~6,000 s → 1,260 s → 616 s), (2) cross-stack decode co-design across the XLA device loop, Pallas VMEM pipeline, and host-device scheduler (~30K → ~130.5K tok/s; 616 s → 299 s), and (3) full-cycle optimization: sequence packing, reward process pool, and GC control (299 s → 88 s), then KV lifecycle, weight sync, training kernels, and the TP1×DP128 rollout shape (88 s → 56 s; ~183K tok/s whole-rollout average).</em>
</p>

As with any distributed accelerator stack, the first step was aligning basic configuration parameters—parallelism, batch concurrency, mesh shape, and HBM limits—with our workload geometry to establish a clean baseline. The remaining bottlenecks were embedded deeper in the open-source stack and were specific to long-context, high-concurrency small-model RL with frequent on-policy weight synchronization.

That motivated us to move from configuring the stack to co-designing it. We profiled the full RL cycle, identified the mechanism behind each new bottleneck, and then worked downward through MaxText, Tunix, vLLM/tpu-inference, Pathways/XLA, and Pallas—progressing from config tuning, to cross-stack decode co-design (XLA device loop, Pallas VMEM pipeline, and host-device scheduler), to Full-Cycle Optimization across host-side overheads, KV-cache lifecycle, training attention geometry, and rollout shape. Once configuration tuning plateaued at roughly 30K tokens/s, this cross-stack decode co-design pushed the full-batch decode rate to approximately 130.5K tokens/s, and the final configuration sustained ~183K tokens/s averaged over the whole rollout.

The lesson was simple: peak small-model RL performance on TPU v7 comes from preserving the workload contract while exposing hardware-specific fast paths across the open-source stack—and upstreaming those optimizations so future workloads benefit by default.

## Benchmark methodology and dataset

Because proprietary Ads ranking and generative recommendation (GR) models and user-interaction logs at Meta cannot be shared publicly, we designed **a fully open, reproducible benchmark proxy**—using Qwen3-0.6B on public math reasoning prompts—that mirrors the systems geometry of production generative ranking: a compact sub-billion-parameter dense policy generating long multi-step reasoning traces (up to 8,192 tokens) across thousands of concurrent candidates.

To make our workload contract, profiler methodology, and all **23** upstream vLLM, Tunix, and MaxText **PRs** **fully reproducible by the open-source community**, we evaluate both controlled ablations and our headline Qwen3-0.6B Group Relative Policy Optimization (GRPO) true-8K workload: 256 prompts × 8 generations (2,048 completions per rollout), up to 8,192 generated tokens, BF16, and synchronous on-policy execution. The bring-up and Phase I–II milestones (up to 299 s) ran with a 16,384-token prompt cap (24K maximum context) on NuminaMath and OpenMath prompts. From the 88-second milestone on, runs use [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) (`train_1M`) with a 4,096-token prompt cap (12,288-token context). Model, GRPO recipe, completions per step, and the 8,192-token response cap are identical, and real prompts are short in both (~170 tokens for NuminaMath, 149 tokens on average for OpenMathInstruct-2), so the generated work per step is comparable; the larger prompt cap mainly reserves more KV capacity per sequence. We report steady-state complete-cycle wall-clock time, excluding compilation, warmup, and checkpoint writes. We tracked rollout throughput and full-cycle latency across the campaign, then used controlled ablations to isolate individual mechanisms.

**Metric definitions.** *Step time* is the steady-state wall-clock time of one complete rollout-to-update cycle (TensorBoard `perf/train/global_step_time`, median over steps 3 and later of a 20-step run); for the final configuration the headline 56 s is its fastest steady-state steps (p10 55.7 s), next to the 57.9 s median. *Decode throughput (tokens/s)* counts generated (completion) tokens only, excludes prefill, and is aggregated over the whole 64-chip slice. For the milestones up to 299 seconds it is the full-batch decode rate measured in controlled runs. For the 88-second milestone and the final configuration it is the generated tokens of one rollout divided by the whole rollout phase (~9.2M tokens in 68 s; ~9.0M tokens in 49.1 s), which includes the low-occupancy tail and is therefore a conservative lower bound on the full-batch rate.

**Milestones at a glance.** Rollout dominated every step. Up to the 299-second milestone it took 80–90% of the step, so those rows give approximate (~) phase times; the 88-second and final splits are measured from per-phase step timers. Rows up to 299 s used the 16K-prompt-cap contract; the 88-second and final rows use the 4K-prompt-cap contract (see Benchmark methodology).

| Milestone | Step time | Decode throughput | Rollout | Everything else |
|---|---|---|---|---|
| Un-tuned bring-up | ~6,000 s | ~4K tok/s | ~5,100 s | ~900 s |
| Configuration and mesh tuning | 1,260 s | ~30K tok/s | ~1,070 s | ~190 s |
| Controlled baseline (colocated, TP/DP rebalanced) | 616 s | ~30K tok/s | ~525 s | ~90 s |
| + Continue Decode | 369 s | ~64.9K tok/s | ~315 s | ~55 s |
| + RPA block split + EOS-check interval | 299 s | ~130.5K tok/s | ~255 s | ~45 s |
| + Continuous-batching decode, sharded sampling, reward pool, packing | 88 s | ~135K tok/s | 68 s | 20 s (training 8 s, weight sync 7 s) |
| Final configuration (TP1×DP128) | **56 s** fastest steady-state steps (p10 55.7 s); 57.9 s 20-step median | **~183K tok/s** | 49.1 s | 8.4 s |

In the final configuration a rollout generates about 9.0M tokens (2,048 completions × ~4.4K mean completion length; 27.8% of completions reach the 8,192-token cap), and the median phase times are 49.1 s of rollout and 8.4 s for training, weight sync, reward, and cache lifecycle combined (85% rollout).

## Optimization journey on Google Cloud TPU v7

We ran the true-8K reference workload on a 64-chip Google Cloud TPU v7 (4×4×4) slice using **TP1×DP128** rollout parallelism: 128 independent vLLM engines, one per TPU device (two per chip), each holding the full 0.6B model (1.2 GB in BF16) and all eight KV heads, with requests spread across engines by round-robin data-parallel routing. Earlier in the campaign we used 4-way tensor parallelism to pool HBM for the KV caches; the move to TP1×DP128 is quantified in Phase III. The complete step included:

- rollout generation;
- reward computation and advantage construction;
- MaxText training;
- policy-weight synchronization;
- KV-cache and request-state lifecycle;
- preparation for the next rollout.

We intentionally began with a serial full-batch design. It made policy ownership, weight versions, cache invalidation, and timing boundaries explicit. Pathways managed the multi-host mesh and in-memory transitions, while the learner and sampler shared the 64-chip resource over the cycle. We later brought up an asynchronous RL prototype, but kept the main performance ladder on the serial path so the timing boundaries remained explicit and optimizations could be compared without adding hardware.

All milestones below reflect steady-state steps. To isolate sub-second kernel and runtime deltas from cluster topology variance, we used dedicated slices and same-window controls.

### Phase I: align basic workload configuration to establish a proper baseline

Our initial un-tuned functional bring-up—running 2,048 synchronous 24K-context rollouts before mesh sizing or continuous device-side decode—generated roughly 4K rollout tokens/s (~100 minutes per full RL cycle). Before modifying any core runtime or kernel code, we tuned the configuration space of the existing stack.

The work included:

- tensor- and data-parallel mesh sweeps;
- rollout batch size and concurrency;
- maximum active sequences and token budgets;
- HBM allocation and host-memory limits;
- prompt and response geometry;
- sampler and trainer resource balance;
- Pathways JobSet placement, compilation, checkpoint, and staging behavior.

This first phase increased decode throughput from roughly **4K to approximately 30K tokens/s** (**7.5x**), reducing the full cycle from **~100 minutes (6,000 s) to 21 minutes (1,260 s, 4.8x)**—and after colocation and TP/DP host-mesh rebalancing established our **616-second controlled baseline**, we moved into stack co-design.

These initial steps simply aligned basic configuration parameters—mesh topology, batch concurrency, and HBM limits—with a 24K-context, 2,048-rollout workload shape (rather than reflecting an out-of-the-box limitation of the TPU v7 stack), establishing a properly configured baseline and revealing the true runtime bottlenecks.

Decode throughput improved much more than full-cycle latency: the full-batch decode rate is only part of rollout time, and prefill, host returns, and the low-occupancy tail of long completions did not speed up with it. The system had not stopped scaling; the critical path had moved into how rollout used the hardware.

### Phase II: Cross-stack decode co-design—from MaxText and vLLM down to XLA and Pallas

At approximately 30K tokens/s, surface-level configuration sweeps hit a wall. Profiling revealed that while pure TPU v7 silicon execution took only milliseconds per decode step, end-to-end rollout time was dominated by host-device dispatch round trips, XLA compilation and runner overheads on the Flax NNX path, and Pallas kernel pipeline stalls over 24K-token contexts.

Fixing this required co-designing across every layer of the open-source stack—plumbing workload-aware execution semantics from the top-level trainer down into compiled XLA loops and Pallas VMEM pipelines:

**MaxText → Tunix → vLLM/tpu-inference (Runner & Scheduler) → Ragged Paged Attention (RPA v3)/Pallas → Pathways/XLA**

This phase raised stable production-scale decode throughput (attending over up to 24K context tokens per sequence across 2,048 synchronous rollouts) to approximately **130.5K tokens/s** (**4.35x** the 30K operating point), cutting full-step latency in the controlled ladder from **616 seconds to 299 seconds**.

#### Compiling multi-step device decode loops across Flax NNX, vLLM, and XLA

Autoregressive decoding is sequential across tokens, but re-entering the Python host scheduler on every token is fatal for a 0.6B model: the fixed cost of host-device dispatch, request-state bookkeeping, and trace serialization exceeds the matrix compute of the forward pass itself.

Continue Decode in vLLM moved many decode iterations into a single compiled XLA device loop, which required co-designing how MaxText, Flax NNX, and tpu-inference interact during compilation and execution. We repaired `continue_decode` on the `flax_nnx` serving path and prevented the runtime from dropping XLA compile options ([tpu-inference #3374](https://github.com/vllm-project/tpu-inference/pull/3374)), eliminated mutable metadata state leaks across compiled linear layers ([tpu-inference #3101](https://github.com/vllm-project/tpu-inference/pull/3101)), stopped the vLLM runner from over-padding steps past `max_num_batched_tokens` ([tpu-inference #3466](https://github.com/vllm-project/tpu-inference/pull/3466)), and capped per-step request-ID trace serialization overhead at 2,048 concurrent rollouts ([tpu-inference #3510](https://github.com/vllm-project/tpu-inference/pull/3510)).

In the controlled MaxText production ladder, this cross-stack path reduced a **616-second clean step to 369 seconds**, a **40.1% reduction**.

This result changed how we thought about the workload. The TPU v7 MXUs were not waiting for more arithmetic intensity from a larger model; the small model was waiting for the runtime and compiler stack to deliver uninterrupted device work.

#### Co-designing Pallas RPA v3 DMA-compute pipelining for 24K-context KV caches

Once host dispatch was amortized, on-device attention memory movement became the primary bottleneck. Ragged Paged Attention v3 (RPA v3), written in Pallas, overlaps HBM-to-VMEM DMA transfers with TensorCore compute using a double-buffered VMEM pipeline. However, its default heuristic coupled the KV fetch block size and the compute block size to a single monolithic value (~16K tokens). On our 24K-context RL workload, a 16K compute block starved the Pallas double-buffered pipeline: each compute stage had to wait for a large KV tile transfer to complete, reducing pipeline depth between DMA and MXU execution.

We refactored the kernel interface to decouple HBM fetch geometry from VMEM compute tiling across decode, prefill, and mixed paths ([tpu-inference #3102](https://github.com/vllm-project/tpu-inference/pull/3102)) and plumbed those kernel controls up through the tpu-inference MaxText serving bridge ([tpu-inference #3395](https://github.com/vllm-project/tpu-inference/pull/3395)) and MaxText RL trainer configs ([MaxText #5100](https://github.com/AI-Hypercomputer/maxtext/pull/5100)). Keeping a **16,384-token HBM fetch block** to maximize burst DMA bandwidth while slicing execution into **4,096-token VMEM compute blocks** restored deep DMA/MXU overlap—lifting decode throughput from **64.9K → 96.3K tokens/s** (64,929 → 96,273 tokens/s, **+48% decode throughput**).

This is a textbook co-design result: the hardware had the bandwidth, the kernel had the pipeline, and the workload exposed a block geometry the general heuristic did not capture.

The split above was measured with 4-way tensor parallelism. In the final TP1×DP128 configuration every chip holds all eight KV heads, so the tiles were re-tuned to 2,048-token decode blocks (`RPA_V3_DECODE_BLOCK_SIZES=1,2048,1,2048`, with matching prefill and mixed overrides; see Appendix B): at TP1, a 4,096-token decode tile is slower and an 8,192-token tile exceeds VMEM.

#### Co-designing on-device EOS reductions with host scheduler eviction

Even inside a fused device loop, Continue Decode still executed an on-device reduction (any-sequence-hit-EOS) and evaluated the early-exit condition on every token step. Across 2,048 concurrent rollouts, that per-token reduction created an unnecessary synchronization barrier inside the compiled loop.

We modified the compiled loop to skip the per-step on-device EOS reduction when the check is off ([tpu-inference #3509](https://github.com/vllm-project/tpu-inference/pull/3509)) and exposed cadence control to balance device execution against host scheduling ([tpu-inference #3174](https://github.com/vllm-project/tpu-inference/pull/3174)), while per-sequence masking preserved the true EOS boundary.

The trade-off was genuinely cross-layer: checking every step incurred on-device sync overhead, whereas checking every 32 steps delayed returning to the vLLM host scheduler to evict finished sequences and free batch slots. An interval of **8 steps** hit the sweet spot between uninterrupted XLA loop execution and timely scheduler retirement. Together with the RPA block split, this change took the step from 369 seconds to the **299-second** true-8K milestone; we did not retain a separate step-time measurement for each of the two.

In the final configuration we turn the in-loop EOS check off entirely (`CONTINUE_DECODE_EOS_CHECK_INTERVAL=0`) and instead lengthen the fused loop to 512 decode steps per return to the host scheduler (`max_decode_steps=512`); per-sequence masking still preserves each EOS boundary.

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-tpu-v7/decode-co-design.webp" alt="Figure 3: Cross-stack decode co-design across vLLM, XLA, and Pallas" width="100%">
</p>

<p align="center">
<em>Figure 3: Cross-stack decode co-design across vLLM, XLA, and Pallas. Three co-designed mechanisms across the runtime/compiler, attention kernel, and device-scheduler boundary that lifted decode throughput from 30K to 130.5K tokens/s on the 24K-context bring-up workload (616 s → 299 s full step). The RPA split and the EOS-check interval together took the step from 369 s to 299 s.</em>
</p>

### Phase III: optimize the whole RL cycle, not only decode

Once decode reached 130.5K tokens/s, the full-batch decode rate was no longer the limit: most of the 299-second step was still rollout, stretched by its low-occupancy tail, and the rest went to KV-cache reallocation and weight synchronization, host-side reward/detokenization/GC overhead, and the training attention kernel.

Phase III had two stages. The first took the step from 299 seconds to **88 seconds** by late August (rollout 68 s, training 8 s, weight sync 7 s). On the rollout side, a continuous-batching decode window kept the fused loop full by retiring finished sequences as inert masked slots instead of exiting on EOS, and vocab-sharded sampling removed a full-vocabulary logits all-gather (~78 MB per data-parallel rank) and the top-k/top-p sweeps from every decode iteration. Off the rollout path, serial reward scoring, measured at ~30 s per step, moved to a process pool, and sequence packing and configurable garbage collection shortened training and host time. Per-change contributions within this stage were not isolated. The move to the 4K prompt cap also happened here (see Benchmark methodology). KV persistence, faster weight synchronization and detokenization, the Splash Attention retune, and the final rollout configuration then took it from 88 seconds to **56 seconds**.

#### Separating logical cache invalidation from physical allocation

Every RL step changes the policy weights. KV values computed under the old policy must never be visible to the new policy. The conservative lifecycle reset the prefix cache, deleted the physical KV allocation, synchronized weights, and reallocated the cache.

We observed that resetting prefix mappings already made every old block logically unreachable. When HBM headroom allowed it, the physical allocation could persist safely across the weight update even though its contents were invalid. The next rollout re-prefilled and overwrote the blocks before reading them.

On the earlier TP4 stack, an opt-in KV-persistence path reduced the measured setup phase from **~37 seconds to 3 seconds**. On the final configuration, turning it on (`free_kv_cache_during_weight_sync=false`) still moves the 20-step median from 60.3 s to 57.9 s. In parallel, we unrolled scanned layers during weight sync ([Tunix #2232](https://github.com/google/tunix/pull/2232)), overlapped detokenization and log-prob extraction with decode ([Tunix #2233](https://github.com/google/tunix/pull/2233)), offloaded reward evaluation to a process pool ([Tunix #1904](https://github.com/google/tunix/pull/1904), [MaxText #5255](https://github.com/AI-Hypercomputer/maxtext/pull/5255)), and made per-step garbage collection configurable ([Tunix #2086](https://github.com/google/tunix/pull/2086), [MaxText #5236](https://github.com/AI-Hypercomputer/maxtext/pull/5236)).

The [public implementation](https://github.com/google/tunix/pull/2231) is default-off and requires two safety conditions: prefix mappings must be reset on every policy update, and the weight reshard must not depend on freeing the cache allocation for HBM headroom.

#### Retuning Splash Attention for the training shape

With rollout, weight-sync, and host lifecycle costs reduced, training accounted for a material share of the step. After enabling RL sequence packing ([MaxText #5100](https://github.com/AI-Hypercomputer/maxtext/pull/5100)) in the 299 → 88 s stage, we switched training to flash attention and retuned the Splash Attention block sizes for the production sequence geometry, passed as MaxText config overrides on the RL trainer command line:

```bash
attention='flash' \
  sa_block_q=1024 sa_block_q_dkv=1024 sa_block_q_dq=1024 \
  sa_block_kv=2048 sa_block_kv_compute=2048 \
  sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 sa_block_kv_dq=2048 \
  sa_use_base2_exp=false
```

This was an important transition in the program: together with KV persistence, the weight-sync and detokenization changes above, and the final rollout configuration below, it took the full true-8K cycle from **88 seconds to 56 seconds**.

#### Final rollout configuration

The last gains, from 88 seconds to 56 seconds, came from fine-grained tuning, aggressive host-side optimizations, and TP=1 rollout. Part of this came with the software itself: running the 88-second launcher unchanged on an image built at upstream head (which includes Tunix #2232/#2233 and MaxText #5235/#2309) measured 74 s on a second cluster, and launcher tuning on that image took it the rest of the way; cluster-to-cluster differences were not separated. Each change below was measured against a control run at the same time on an earlier image; sizes are approximate and were not re-measured one by one on the final image.

| Change | Effect |
|---|---|
| `max_decode_steps` 128 → 512 (fused decode steps per host return) | ~4 s less rollout (128 → 256: ~1.6 s; 256 → 512: ~2.6 s) |
| TP4 → TP1×DP128, with round-robin DP routing | ~1 s per step (−1.3 s and −0.9 s in two controlled pairs); the default least-loaded routing costs ~35 s per step at DP128 |
| Flash attention and Splash block sizes for training | ~2 s per step |
| KV persistence across the weight update | 60.3 s → 57.9 s 20-step median |
| Reward workers (`reward_num_workers=32`, `math_verify_num_procs=64`) | small |
| Weight-sync unroll and detokenization overlap ([Tunix #2232](https://github.com/google/tunix/pull/2232), [#2233](https://github.com/google/tunix/pull/2233)) | not isolated |

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-tpu-v7/bottleneck-migration.webp" alt="Figure 4: Bottleneck migration" width="100%">
</p>

<p align="center">
<em>Figure 4: Bottleneck migration. Steady-state step time per milestone (log scale, excluding compilation and checkpointing), with the rollout share of each step shown under its bar; the dashed line marks the move from the 16K to the 4K prompt cap. Up to 299 s, rollout took 80–90% of the step (~85% shown); the 88 s and final splits are measured (68 s of 88 s; 49.1 s + 8.4 s medians). 56 s is the fastest steady-state step time (p10 55.7 s); the 20-step median is 57.9 s.</em>
</p>

## What did not work—and why those learnings mattered

The campaign produced several negative results. They are part of the story because they prevented plausible but incorrect explanations from becoming roadmap commitments.

### Weight and KV byte reduction were not the answer

We verified a weight-only INT8 path and observed no meaningful decode improvement. For a 0.6B model with long contexts, model weights were a small fraction of the data movement and fixed execution cost.

We also verified an FP8 KV-cache path by checking that cache capacity doubled. The latency result was far smaller than a bandwidth-proportional speedup. That experiment closed an important hypothesis: at this operating point, decode was not simply saturated on HBM bytes. Dispatch, fusion, per-iteration fixed cost, and other kernels set a floor that byte reduction alone could not remove.

This distinction matters for TPU v7 optimization. Quantization can be transformative when weights or KV traffic dominate; it can be nearly irrelevant when the workload is controlled by a different roofline.

### Shared-prefix cascade attention was not a production win

Eight generations from the same prompt appear to be an ideal case for explicit shared-prefix attention. Early prototypes looked promising against a naive JAX baseline. When compared with the real pipelined RPA kernel at production scale, the advantage fell sharply. A composed prefix/suffix cascade measured no net gain because the stock paged-attention path already benefited substantially from shared physical pages and the extra calls and merge overhead consumed the remaining opportunity.

The negative result saved a larger kernel project. It also reinforced a benchmarking rule: **the correct baseline is the production kernel on the production image at the production batch size**—not a mathematically equivalent reference implementation.

### More microbatches did not mean less training time

A larger training microbatch fit after memory tuning, but halving the number of gradient-accumulation loops did not reduce the total training FLOPs. The step remained flat. Similarly, lighter rematerialization freed HBM but did not shorten the compute-bound training path.

### Shorter response caps had diminishing returns

Reducing the response cap from **8,192 to 6,144** removed a large low-occupancy tail and reached approximately **56 seconds**, but it changed the RL recipe and therefore remained a separate candidate. Reducing the cap further to **5,120** produced no reliable additional gain. Once the train and setup floor dominated, cutting more response budget no longer bought meaningful systems performance.

This was a separate experiment on the stack before the final Splash and rollout changes. The later Splash, KV-persistence, weight-sync, and final rollout changes reached approximately the same latency while preserving the true **8,192-token cap**; every headline number in this post uses the full 8,192-token cap.

## Our optimization loop

The performance curve was not produced by one heroic patch. It came from a repeated engineering loop:

- **Measure the full cycle.** Isolated tokens/s was useful, but the decision metric was rollout-to-update time.
- **Profile the current winner.** Every successful optimization changed the phase breakdown.
- **Form a mechanism hypothesis.** We asked what specific synchronization, allocation, block shape, or data movement created the wall time.
- **Change one mechanism.** Risky paths were environment-gated and default-off.
- **Use dedicated-slice or same-window controls.** Cluster placement and contention can overwhelm small deltas.
- **Validate semantics.** EOS boundaries, policy versions, token/log-prob alignment, finite loss, cache invalidation, and reward behavior were part of performance testing.
- **Bank negative results.** A null result that closed a large engineering branch was considered a successful experiment.
- **Profile again.** The next bottleneck was rarely the one that had mattered one round earlier.

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-tpu-v7/performance-evaluation-dashboard.webp" alt="Figure 5: Performance evaluation dashboard" width="100%">
</p>

<p align="center">
<em>Figure 5: Performance evaluation dashboard. (A) Full-step milestone ladder, log scale (56 s = fastest steady-state steps, p10 55.7 s; 20-step median 57.9 s). (B) Adopted and neutral optimizations from the campaign.</em>
</p>

## When systems optimization exposes a learning bottleneck

Notably, the optimized stack reached **56 seconds** per true-8K cycle under an intentional **worst-case systems stress test**: by evaluating the raw 0.6B checkpoint without an SFT warm-start, we forced a large fraction of rollouts to decode all the way to the 8,192-token ceiling (27.8% of the 2,048 completions hit the cap, and the mean completion length was ~4.4K tokens). Because production SFT warm-started policies converge to more concise reasoning traces, real-world RL training cycles run even faster than this 56-second upper bound.

Combining a lightweight format-and-task SFT warm-start and reward shaping with on-policy RL post-training (GRPO / Decoupled Clip and Dynamic Sampling Policy Optimization, DAPO) teaches the compact policy to solve reasoning and ranking tasks with shorter average completion lengths—compounding our systems speedups with algorithmic efficiency.

This creates the final co-design loop:

- runtime improvements make rollouts cheaper;
- cheaper rollouts allow more learning experiments;
- a better learning policy produces shorter, more useful completions;
- shorter completions further reduce systems cost.

The boundary between "model quality" and "systems performance" is therefore not fixed. In RL post-training, model behavior is part of the workload presented to the hardware.

## What comes next

With the sub-minute (56-second) milestone achieved and all 23 PRs merged upstream, we are continuing to invest in the open-source vLLM, Tunix, and MaxText ecosystem on Google Cloud TPU v7 across several priorities:

- **Scaling multi-stage SFT + RL (GRPO/DAPO) recipes** across broader generative ranking and reasoning model families;
- **Auto-tuning Pallas RPA v3 and Splash Attention heuristics** so optimal DMA/compute block sizes and layouts are selected automatically from model and context shape out of the box;
- **Upstreaming replay-exact sharded sampling** for deterministic, reproducible multi-host RL evaluation at scale;
- **Scaling to 128-chip (4×4×8) and larger TPU v7 slices** for even higher rollout concurrency and decode throughput;
- **Advancing asynchronous and agentic RL (AgenticAsync)** with strict policy-version and behavior-log-prob guarantees;
- **Enabling Multiple Program, Multiple Data (MPMD) disaggregated rollout and training** to overlap generation and learner updates across dedicated TPU sub-meshes.

Our goal is to make high-throughput, production-grade RL on Google Cloud TPU v7 turnkey for the entire open-source community through continued investment in vLLM, Tunix, and MaxText.

## Conclusion

The project started with a small dense model that appeared easy to run and a full RL cycle that took roughly 100 minutes.

The first **7.5x decode-throughput gain** came from disciplined configuration and mesh tuning. Further gains came from moving down the Google open-source stack and co-designing the decode path. The final full-cycle gains came from taking host-side reward, GC, and packing overheads off the critical path (299 → 88 s), then recognizing that cache lifetime, training attention, and rollout shape had become the new bottlenecks (88 → 56 s).

Across the program on Google Cloud TPU v7, decode throughput increased from approximately **4K to ~183K tokens/s** (the final value is a whole-rollout average, so this is a conservative comparison), while end-to-end cycle time fell from roughly **100 minutes to approximately 56 seconds**. That is about **46x higher throughput** and **107x faster full-cycle execution**.

The most important result is both the cross-stack co-design methodology behind those gains and the fact that **all 23 runtime, trainer, and kernel PRs are now merged upstream** across vLLM, Tunix, and MaxText—so teams adopting Google Cloud TPU v7 for RL today can reach the optimized 56-second (~183K tokens/s) operating point by enabling the upstream flags in Appendix B, without repeating this bring-up journey.

## Acknowledgments

We thank Tianyu Gu (@gutianyu-google), Alisa Chen, Pysith Vanuptikul, Summer Yang, Hossein Sarshar, Jianting Cao, Weiren Yu and the Google Cloud TPU, vLLM, Tunix, and MaxText engineering teams for their close collaboration in co-designing and upstreaming these optimizations.

## Appendix A — Merged upstream PRs

All 23 PRs below are merged upstream (the last on September 21, 2026), across **vllm-project/tpu-inference**, **google/tunix**, and **AI-Hypercomputer/maxtext** during this optimization campaign:

**vLLM / tpu-inference — 10 merged PRs**

- [#2507](https://github.com/vllm-project/tpu-inference/pull/2507) continuous decoding
- [#3068](https://github.com/vllm-project/tpu-inference/pull/3068) continuous decoding EOS
- [#3101](https://github.com/vllm-project/tpu-inference/pull/3101) fix(linear): mutable default kernel_metadata/bias_metadata dicts leak state
- [#3102](https://github.com/vllm-project/tpu-inference/pull/3102) feat(rpa-v3): env override for decode/prefill/mixed block sizes
- [#3174](https://github.com/vllm-project/tpu-inference/pull/3174) feat(continue_decode): env-gated EOS-check interval for the fused decode loop
- [#3374](https://github.com/vllm-project/tpu-inference/pull/3374) Fix continue_decode on the flax_nnx path and stop dropping compile options
- [#3395](https://github.com/vllm-project/tpu-inference/pull/3395) Forward RPA v3 block-size overrides on the MaxText serving path
- [#3466](https://github.com/vllm-project/tpu-inference/pull/3466) [Runner] Never pad a step past max_num_batched_tokens
- [#3509](https://github.com/vllm-project/tpu-inference/pull/3509) [continue_decode] Skip the per-step EOS reduction when the EOS check is off
- [#3510](https://github.com/vllm-project/tpu-inference/pull/3510) [Runner] Cap the request ids attached to the execute_model trace annotation

**Tunix — 7 merged PRs**

- [#1904](https://github.com/google/tunix/pull/1904) Add optional process-pool evaluation of reward functions
- [#2086](https://github.com/google/tunix/pull/2086) Make the per-step gc.collect() in RLEngine configurable
- [#2088](https://github.com/google/tunix/pull/2088) Disable the profiler on an empty profiling window instead of raising
- [#2231](https://github.com/google/tunix/pull/2231) Make freeing the vLLM KV cache around the weight sync optional
- [#2232](https://github.com/google/tunix/pull/2232) Unroll scanned layers with one jitted call per parameter in the weight sync
- [#2233](https://github.com/google/tunix/pull/2233) Overlap detokenization and logprob extraction with decode in VllmSampler
- [#2309](https://github.com/google/tunix/pull/2309) Fix packed-row conversion failing on TPU/GPU with copy=False

**MaxText — 6 merged PRs**

- [#5100](https://github.com/AI-Hypercomputer/maxtext/pull/5100) Add RL sequence-packing and vLLM block-size knobs
- [#5235](https://github.com/AI-Hypercomputer/maxtext/pull/5235) Make extract_answer linear in the response length
- [#5236](https://github.com/AI-Hypercomputer/maxtext/pull/5236) Plumb tunix gc_collect_after_weight_sync into the RL trainer
- [#5255](https://github.com/AI-Hypercomputer/maxtext/pull/5255) Plumb tunix reward_num_workers into the RL trainer with a picklable reward wrapper
- [#5270](https://github.com/AI-Hypercomputer/maxtext/pull/5270) Restore needs_layout_passes=False on the SC ragged_gather_reduce v2 kernel (fixes TPU unit tests on main)
- [#5289](https://github.com/AI-Hypercomputer/maxtext/pull/5289) Add free_kv_cache_during_weight_sync to the RL config and forward it to tunix

Several optimizations required both a runtime change and trainer configuration: reward workers (Tunix #1904 / MaxText #5255), GC control (#2086 / #5236), and KV-cache lifetime (#2231 / #5289).

## Appendix B — How to reproduce

**Software.** All 23 PRs are included in [MaxText v0.2.6](https://github.com/AI-Hypercomputer/maxtext/releases/tag/maxtext-v0.2.6) together with the post-training dependencies it pins: Tunix [`385551df`](https://github.com/google/tunix/commit/385551df4643ee3ac3205e813a93c2ff217dffa5), tpu-inference [`73d7342d`](https://github.com/vllm-project/tpu-inference/commit/73d7342da057ecc4d4f453df85cb036976701efd), vLLM [`5df63a72`](https://github.com/vllm-project/vllm/commit/5df63a7203cdb9328d824a27b682e77c7b9928be), JAX/jaxlib 0.11.0, and libtpu 0.0.44. The numbers in this post were measured on an earlier build with the same changes (MaxText `78aa1555`, Tunix `28ad56f7`, tpu-inference `1224e9fb`, vLLM `51da0ca6`, JAX 0.11.0, libtpu 0.0.44, Python 3.12).

**Hardware.** One Google Cloud TPU v7 4×4×4 slice (64 chips, 16 hosts), with Pathways managing the multi-host mesh.

**Workload.** Qwen3-0.6B, [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) `train_1M`, GRPO with 256 prompts × 8 generations per step, a 20-step run, and step time read as the median over steps 3 and later.

**Environment variables** (tpu-inference):

```bash
export RPA_V3_DECODE_BLOCK_SIZES="1,2048,1,2048"
export RPA_V3_MIXED_BLOCK_SIZES="512,1024,256,512"
export RPA_V3_PREFILL_BLOCK_SIZES="512,1024,256,512"
export CONTINUE_DECODE_EOS_CHECK_INTERVAL="0"
export DP_SCHED_ROUTING=round_robin
```

**MaxText RL trainer command**, run from the MaxText repository root (add your checkpoint path with `load_parameters_path=...`):

```bash
PYTHONPATH=src python3 -m maxtext.trainers.post_train.rl.train_rl src/maxtext/configs/post_train/rl.yml \
  model_name=qwen3-0.6b tokenizer_path=Qwen/Qwen3-0.6B \
  dataset_name=nvidia/OpenMathInstruct-2 train_split=train_1M \
  hf_train_files='hf://datasets/nvidia/OpenMathInstruct-2/data/train_1M-*.parquet' \
  batch_size=256 rl.num_generations=8 \
  rl.grpo_beta=0.0 rl.grpo_epsilon=0.2 rl.epsilon_high=0.28 learning_rate=1e-6 \
  max_prefill_predict_length=4096 max_target_length=12288 kv_cache_buffer=4352 \
  decode_sampling_temperature=0.8 decode_sampling_top_k=50 decode_sampling_nucleus_p=0.95 \
  rollout_data_parallelism=128 rollout_tensor_parallelism=1 \
  max_num_seqs=256 max_num_batched_tokens=33280 hbm_utilization_vllm=0.3 \
  vllm_block_size=128 enable_prefix_caching=false async_scheduling=False \
  vllm_additional_config='{"enable_continue_decode": true, "max_decode_steps": 512}' \
  max_seq_token_per_tpu=12288 train_micro_batch_size=32 \
  free_kv_cache_during_weight_sync=false gc_collect_after_weight_sync=false \
  reward_num_workers=32 math_verify_num_procs=64 \
  attention=flash \
  sa_block_q=1024 sa_block_q_dkv=1024 sa_block_q_dq=1024 \
  sa_block_kv=2048 sa_block_kv_compute=2048 \
  sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 sa_block_kv_dq=2048 \
  sa_use_base2_exp=false
```

Keep `async_scheduling=False` when enabling Continue Decode, as synchronous host scheduling is required for the fused device loop path.

