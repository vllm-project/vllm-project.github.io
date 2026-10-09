---
layout: post
title: "Reinforcement Learning: Optimizing 0.6B Dense Models on Heterogeneous Hardware via vLLM"
author: "Loki Chen, Yang Song, Hongye Xie, Ming Lei, Greg Rehm, Chongyang Bai, Dre Olgiati, Bob Kamma, Tej Choudhary, Hang Cui, Imed Zitouni, Hamed Firooz (Meta ASA); Mingfu Liang, Kavosh Asadi, Yufei Li, Frank Shyu, Parish Aggarwal, Senthil Manickavelu, Xi Liu, Luke Simon (Meta Ranking AI Research)"
date: 2026-10-09 09:00:00 +0000
summary: "Co-designing RL for a Qwen3-0.6B dense policy across GB200, MI350X, and TPU v7x with vLLM: one workload contract, hardware-specific fast paths, and a 107x end-to-end speedup from ~100 minutes to ~56 seconds per true-8K RL step."
image: /assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/program-journey-107x.png
social_image: /assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/program-journey-107x.png
tags:
  - reinforcement-learning
  - performance
  - hardware
---

*Authors — **ASA:** Loki Chen, Yang Song, Hongye Xie, Ming Lei, Greg Rehm, Chongyang Bai, Dre Olgiati, Bob Kamma, Tej Choudhary, Hang Cui, Imed Zitouni, Hamed Firooz. **Ranking AI Research:** Mingfu Liang, Kavosh Asadi, Yufei Li, Frank Shyu, Parish Aggarwal, Senthil Manickavelu, Xi Liu, Luke Simon.*

**TL;DR**

- Small dense-model RL exposes new challenges for every type of hardware: decode scheduling, synchronization, attention geometry, and KV-cache lifecycle become first-order bottlenecks.
- A common workload contract across accelerator backends allows the **vLLM** community to preserve RL semantics while developing hardware-specific fast paths.
- The complete rollout-to-update cycle achieved a **107× end-to-end speedup** on certain chips, from roughly 100 minutes to approximately 56 seconds, while preserving the true 8,192-token response budget.

## Why a small model

Generative reasoning is reshaping how ranking and decision systems work. Models like GR2 (Generative Reasoning Re-ranker) [[arXiv:2606.31984](https://arxiv.org/abs/2606.31984)] have demonstrated that generative reasoning can meaningfully improve re-ranking quality. The natural next question is: *how small can the policy model be without losing that capability?*

A 0.6B dense student distilled from stronger teachers could serve with dramatically lower latency and cost and enabling broader traffic coverage, faster model refreshes, and more RL experimentation loops per day. But realizing this requires that the *full training cycle* — not just model FLOPs be fast. At 0.6B parameters, rollout generation, synchronization, and framework overhead increasingly dominate wall-clock time.

**This is where the system challenge begins.** Because the model is so small, the useful compute per device shrinks while fixed system costs: dispatch, synchronization, cache lifecycle remain constant. The bottleneck shifts from arithmetic to coordination.

## The model was small. The system problem was not.

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/why-small-dense-needs-co-design.png" alt="Figure 1: Why small dense models need co-design" width="100%">
</p>

<p align="center">
<em>Figure 1: Why small dense models need co-design</em>
</p>

At first glance, a 0.6-billion-parameter model should be easy to run on a GB200/MI350x/ASIC accelerator slice. The model has far fewer weights than the large dense and mixture-of-experts systems that usually motivate distributed training work. In practice, that is exactly what made this workload difficult.

For a large dense or MoE model, each layer performs enough matrix computation that device utilization is often dominated by the familiar questions: how to shard parameters, move activations, route experts, and keep large matrix units busy. Those problems are hard, but their costs scale visibly with the model.

A small dense model changes the balance. The useful matrix work per device shrinks, but many fixed costs do not:

- the host still schedules decode iterations;
- every token still traverses the attention stack;
- every layer still launches or enters compiled kernels;
- every rollout step still updates and synchronizes weights;
- the runtime still tracks thousands of sequences and their stopping conditions;
- the RL cycle still includes reward computation, training, cache invalidation, and distributed orchestration.

The smaller the model became, the more every boundary in the system mattered. A few milliseconds of dispatch overhead, an overly large attention compute block, or a cache allocation repeated on every update could dominate the work we actually wanted to do.

This is the central thesis of the project: small dense-model RL is not simply a scaled-down version of large-model training.** It is a coordination-sensitive workload that rewards hardware-software-model co-design.**

## Why heterogeneous hardware matters

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/accelerator-market-landscape.png" alt="Figure 2: AI accelerator market landscape (illustrative, based on public estimates)" width="100%">
</p>

<p align="center">
<em>Figure 2: AI accelerator market landscape (illustrative, based on public estimates)</em>
</p>

AI workloads span training, RL rollout, and latency-critical inference: each with different compute, memory, and communication profiles. By running the same frozen workload across GB200, MI350X, and other hardware types, we can distinguish general system behavior from backend-specific effects.

- **GB200/MI350X** represent popular GPU stacks.
- ASIC (MTIA, NPU, TPU etc.)

The goal is scientific validation and software portability: not vendor preference or procurement comparison. We hold the model, prompts, response budget, rollout count, numerical mode, and complete RL semantics constant; only the hardware and its co-designed software stack vary.

## Why co-design mattered

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/co-design-rl-0p6b-dense.png" alt="Figure 3: Co-designing RL for a 0.6B dense model" width="100%">
</p>

<p align="center">
<em>Figure 3: Co-designing RL for a 0.6B dense model</em>
</p>

Parameter tuning was necessary, but it was not enough. Early sweeps over parallelism, batch size, mesh shape, concurrency, and memory limits released the performance already available in the system; after that, gains quickly diminished. The remaining bottlenecks were embedded deeper in the open-source stack and were specific to our workload: a 0.6B dense policy, long prompts, 2,048 rollouts per step, an 8K response budget, and frequent weight synchronization.

That forced us to move from **configuring the stack to co-designing it**. We profiled the full RL cycle, identified the mechanism behind each new bottleneck, and then worked downward through MaxText, Tunix, vLLM/tpu-inference, Pathways, and Pallas—changing scheduling, decode behavior, cache lifecycle, and kernel geometry around the actual workload. Once configuration tuning plateaued at roughly 30K tokens/s, this workload-driven OSS optimization pushed decode to approximately 130.5K tokens/s.

The lesson was simple: performance portability does not come from finding one universal parameter set. **It comes from preserving the workload contract while allowing each hardware/software stack to expose and optimize its own fast path.**

## A heterogeneous infrastructure lens

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/heterogeneous-rl-architecture.png" alt="Figure 4: Meta's heterogeneous RL architecture" width="100%">
</p>

<p align="center">
<em>Figure 4: Meta's heterogeneous RL architecture</em>
</p>

This work also fits a broader Meta infrastructure principle: no single accelerator architecture is optimal for every AI workload.

Meta’s public infrastructure strategy combines partner accelerators, including NVIDIA and AMD GPUs, with custom silicon such as [MTIA](https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/). The goal is not portability at the lowest common denominator. It is a common workload contract with hardware-specific fast paths: the RL algorithm, data semantics, policy versions, rewards, and correctness checks should remain stable, while scheduling, parallelism, kernels, precision, and lifecycle management are co-designed for each accelerator.

In that model, rollout and training are also separable roles. High-concurrency autoregressive generation, dense training, reward inference, and agentic orchestration may eventually run on different accelerator types or different partitions of the same system. This project is one concrete example of how Meta can bring a production workload to a new accelerator family, preserve end-to-end semantics, and then push beyond out-of-the-box performance through co-design.

We plan to apply the same measurement contract to NVIDIA H100, AMD MI300X, and relevant MTIA configurations. Those comparisons will be published only after we can hold the model, prompts, response budget, rollout count, numerical mode, and complete RL semantics constant.

## Benchmark methodology and dataset

We evaluate two complementary workload regimes. Controlled ablations use public [**NuminaMath**](https://github.com/project-numina)** and **[**OpenMath**](https://github.com/openmath)** **math-reasoning datasets, primarily with decode-light or moderate generation lengths, to isolate framework, scheduling, and orchestration overheads. Our headline end-to-end results use a frozen **Qwen3-0.6B GRPO true-8K workload**: 256 prompts × 8 generations, 2,048 completions per rollout, up to 16K prompt tokens and 8K generated tokens, BF16, and synchronous on-policy execution. Across systems, we keep the model, workload semantics, token budget, and RL objective fixed and report steady-state **TRUE CYCLE** wall-clock time after excluding compilation and warmup.** **

## Optimization journey on NVDA GB200

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/gb200-optimization-journey.png" alt="Figure 5: GB200 optimization journey via vLLM" width="100%">
</p>

<p align="center">
<em>Figure 5: GB200 optimization journey via vLLM</em>
</p>

On GB200, the same co-design pattern appeared in a different form: **the biggest gains came from reducing framework work per RL step **rather than changing the silicon or learning semantics. With the 256-prompt × 8-generation, true-8K, BF16 workload frozen, increasing the per-device training batch reduced optimizer micro-steps from 8 to 2 and cut the 16-GPU cycle from **143 s to 69.7 s (2.05×)**, while a larger batch hit the HBM limit. Removing an unnecessary rollout-logging path shaved another ~22% from the training wall. At 64 GPUs, the tuned stack reached ~**48s end-to-end**, with only 6% of the cycle left in training compute—the rest was rollout and framework coordination.

## Optimization journey on AMD MI350x

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/mi350x-optimization-journey.png" alt="Figure 6: MI350X optimization journey via vLLM" width="100%">
</p>

<p align="center">
<em>Figure 6: MI350X optimization journey via vLLM</em>
</p>

Applying the same tuned RL configuration to AMD MI350X exposed a very different bottleneck. The framework-amortization changes still reduced per-step overhead, but rollout generation quickly became the dominant floor: at 64 GPUs, generation alone consumed 100.2 s of the 133.4 s full cycle, or** 75% of total time**. More hardware provided surprisingly little relief—the rollout wall fell only from **121.8 s at 8 GPUs **to 93.4 s at 128 GPUs, just **1.3× faster with 16× more GPUs**. This changed the optimization strategy entirely: once framework overhead was reduced, further configuration tuning could no longer move the critical path. **The next gains have to come from the ROCm/vLLM generation path itself—**through engine-, scheduling-, or kernel-level co-design rather than another configuration knob.

## Optimization journey on Google OSS & hardware

Our true-8K reference workload used Qwen3-0.6B on a 64-chip TPU v7x 4×4×4 slice with DP16×TP4 parallelism. Each step started from 256 prompts and generated eight responses per prompt, producing 2,048 rollouts. Prompts were approximately 16K tokens long, and responses retained the full 8,192-token cap with normal EOS handling.

The complete step included:

- rollout generation;
- reward computation and advantage construction;
- MaxText training;
- policy-weight synchronization;
- KV-cache and request-state lifecycle;
- preparation for the next rollout.

We intentionally began with a serial full-batch design. It made policy ownership, weight versions, cache invalidation, and timing boundaries explicit. Pathways managed the multi-host mesh and in-memory transitions, while the learner and sampler shared the 64-chip resource over the cycle. We later brought up an asynchronous RL prototype, but kept the main performance ladder on the serial path so that every optimization remained directly comparable and did not hide work behind additional hardware.

We measured warm, compile-free, checkpoint-free steps. Risky changes were environment-gated and default-off. When possible, we used dedicated slices and same-window controls; an important lesson from the program was that parallel performance experiments on a shared compute pool can produce misleading placement and contention artifacts.

### Phase I: release the performance already present in the system

The earliest Tunix deployment generated only a few thousand rollout tokens per second, and a complete RL cycle took roughly 100 minutes. Before modifying core runtime or kernel code, we exhausted the configuration space of the existing system.

The work included:

- tensor- and data-parallel mesh sweeps;
- rollout batch size and concurrency;
- maximum active sequences and token budgets;
- HBM allocation and host-memory limits;
- prompt and response geometry;
- sampler and trainer resource balance;
- Pathways JobSet placement, compilation, checkpoint, and staging behavior.

This first phase increased rollout throughput from roughly **4K to approximately 30K tokens/s**, a **7.5× improvement**, and reduced the full cycle from approximately **100 minutes to 21 minutes**, a **4.8× speedup**.

That result was more than parameter tuning. It established the operating envelope of the workload and revealed the next bottleneck. Rollout throughput improved much more than full-cycle latency because training, cache lifecycle, and control-plane overhead had become visible. The system had not stopped scaling; the critical path had moved.

### Phase II: move down the Google open-source stack

At approximately 30K tokens/s, more configuration sweeps delivered diminishing returns. We shifted from tuning inputs to reading and instrumenting the stack itself:

**MaxText → Tunix → vLLM/tpu-inference → Continue Decode → RPA v3/Pallas → Pathways/XLA**

This phase raised stable production-scale decode throughput to approximately **130.5K tokens/s**, another **4.4×**, and reduced the controlled full-step ladder from 616 seconds to 299 seconds.

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/program-journey-107x.png" alt="Figure 7: The 107× program journey" width="100%">
</p>

<p align="center">
<em>Figure 7: The 107× program journey</em>
</p>

#### Amortizing token-by-token dispatch

Autoregressive decoding is sequential, but that does not mean the host must re-enter the scheduling path for every token. On a small model, the fixed cost of each host-device round trip can exceed the useful compute between round trips.

Continue Decode moved many decode iterations into one compiled device-side loop. In the controlled MaxText production ladder, enabling this path reduced a 616-second clean step to 369 seconds—a 40% reduction and the largest single measured step in the campaign.

This result changed how we thought about the workload. The accelerator was not waiting for more arithmetic intensity from a bigger model; the small model was waiting for the software stack to give it enough uninterrupted work.

#### Matching RPA’s block geometry to a long-context RL workload

Ragged Paged Attention v3 was already the optimized default attention family on the decode path. The issue was not that the stack had selected the wrong kernel. The issue was that a broad default heuristic selected the same large block size for both KV fetching and computation.

For our long-prefix workload, using an approximately 16K compute block reduced the depth of the Pallas double-buffered pipeline. We added independent environment overrides for decode, prefill, and mixed cases and swept production shapes. The best configuration kept a **16,384-token fetch block** while reducing the **compute block to 4,096 tokens**.

The result was **64,929 → 130,573 tokens/s**, a **2.01× increase**, reproduced across multiple runs. This is a textbook co-design result: the hardware had the bandwidth, the kernel had the pipeline, and the workload exposed a block geometry the general heuristic did not capture.

#### Amortizing stopping-condition synchronization

Continue Decode still evaluated the any-sequence-hit-EOS early-exit condition on every fused iteration. At the scale of 2,048 rollouts, the check became a measurable synchronization point.

We added an environment-gated EOS-check interval. An interval of eight allowed the compiled loop to make more progress before evaluating the global early-exit condition, while per-sequence masking preserved the true EOS boundary. The response was not monotonic: interval eight won, while interval 32 regressed. The optimum sat between two competing needs—amortizing control overhead and returning to the scheduler frequently enough to retire completed work.

Stacked with Continue Decode and the RPA block split, this change produced the banked **299-second true-8K result**.

On Ironwood (TPU v7x), the cycle started at 250~290 s. Retuning the block sizes of vLLM's ragged paged attention kernel for this workload's decode shapes brought it to **218s (1.19×)**. Colocating the vLLM rollout engine with the MaxText trainer on the same chips and rebalancing tensor and data parallelism across the slice reached 189 s **(1.38× cumulative)**. Keeping the decode loop on-device across generation steps, instead of returning to the host after every token, cut it to **157 s (1.66×)**. Moving reward computation off the critical path into a parallel worker pool gave **129 s (2.02×)**. Packing variable-length completions into dense training sequences, rather than padding every sample to the longest one, gave **103s (2.52×)**. Finally, sampling directly from tensor-parallel-sharded logits, instead of gathering the full vocabulary onto every chip at each decode step, brought the cycle to** 56s (3.51×)**. Reward and completion-length distributions were unchanged at every stage. As on GB200, what remains is dominated by rollout generation and framework coordination; training compute is now roughly a tenth of the cycle.

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/decode-co-design.png" alt="Figure 8: Decode co-design" width="100%">
</p>

<p align="center">
<em>Figure 8: Decode co-design</em>
</p>

### Phase III: optimize the whole RL cycle, not only decode

Once decode became faster, two costs that had previously been secondary moved onto the critical path: the lifetime of KV-cache allocations across weight updates and the training attention kernel.

#### Separating logical cache invalidation from physical allocation

Every RL step changes the policy weights. KV values computed under the old policy must never be visible to the new policy. The conservative lifecycle reset the prefix cache, deleted the physical KV allocation, synchronized weights, and reallocated the cache.

We observed that resetting prefix mappings already made every old block logically unreachable. When HBM headroom allowed it, the physical allocation could persist safely across the weight update even though its contents were invalid. The next rollout re-prefilled and overwrote the blocks before reading them.

An opt-in KV-persistence path reduced the measured setup phase from approximately **37 seconds to 3 seconds**. The public implementation is default-off and requires two safety conditions: prefix mappings must be reset on every policy update, and the weight reshard must not depend on freeing the cache allocation for HBM headroom.

#### Retuning Splash Attention for the training shape

With rollout and lifecycle costs reduced, training accounted for a material share of the step. We then retuned five train-side Splash Attention parameters for the production sequence geometry:

```
sa_block_q      = 4096
sa_block_q_dkv  = 2048
sa_block_kv_dkv = 2048
sa_q_layout     = SEQ_MINOR
sa_v_layout     = SEQ_MINOR
```

This was an important transition in the program. The final true-8K improvement did not come from generation. It came from a training attention kernel that had become visible only after decode and cache lifecycle were improved.

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/bottleneck-migration.png" alt="Figure 9: Bottleneck migration" width="100%">
</p>

<p align="center">
<em>Figure 9: Bottleneck migration</em>
</p>

## What did not work—and why those results mattered

The campaign produced several negative results. They are part of the story because they prevented plausible but incorrect explanations from becoming roadmap commitments.

#### Weight and KV byte reduction were not the answer

We verified a weight-only INT8 path and observed no meaningful decode improvement. For a 0.6B model with long contexts, model weights were a small fraction of the data movement and fixed execution cost.

We also verified a lower-precision KV storage path by checking that cache capacity doubled. The latency result was far smaller than a bandwidth-proportional speedup. That experiment closed an important hypothesis: at this operating point, decode was not simply saturated on HBM bytes. Dispatch, fusion, per-iteration fixed cost, and other kernels set a floor that byte reduction alone could not remove.

This distinction matters for heterogeneous optimization. Quantization can be transformative when weights or KV traffic dominate; it can be nearly irrelevant when the workload is controlled by a different roofline.

#### Shared-prefix cascade attention was not a production win

Eight generations from the same prompt appear to be an ideal case for explicit shared-prefix attention. Early prototypes looked promising against a naïve JAX baseline. When compared with the real pipelined RPA kernel at production scale, the advantage fell sharply. A composed prefix/suffix cascade measured no net gain because the stock paged-attention path already benefited substantially from shared physical pages and the extra calls and merge overhead consumed the remaining opportunity.

The negative result saved a larger kernel project. It also reinforced a benchmarking rule: **the correct baseline is the production kernel on the production image at the production batch size—not a mathematically equivalent reference implementation.**

#### More microbatches did not mean less training time

A larger training microbatch fit after memory tuning, but halving the number of gradient-accumulation loops did not reduce the total training FLOPs. The step remained flat. Similarly, lighter rematerialization freed HBM but did not shorten the compute-bound training path.

#### Shorter response caps had diminishing returns

Reducing the response cap from 8,192 to 6,144 removed a large low-occupancy tail and reached approximately 56 seconds, but it changed the RL recipe and therefore remained a separate candidate. Reducing the cap further to 5,120 produced no reliable additional gain. Once the train and setup floor dominated, cutting more response budget no longer bought meaningful systems performance.

The later Splash result reached approximately the same latency while preserving the true 8,192-token cap, allowing the public headline to remain a same-workload system result.

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
<img src="/assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/performance-evaluation-dashboard.png" alt="Figure 10: Performance evaluation dashboard" width="100%">
</p>

<p align="center">
<em>Figure 10: Performance evaluation dashboard</em>
</p>

## When systems optimization exposes a learning bottleneck

The optimized stack now completes the true-8K cycle in approximately 56 seconds, but the current 0.6B policy often emits repetitive, low-value long completions. That behavior is not a kernel defect; it is a model and training-recipe problem. It also has a direct system cost because every unnecessary autoregressive token extends the rollout tail.

A few-shot prompt experiment improved the surface shape of the response but did not reliably teach the small model to produce a valid task solution. The next step is a small format-and-task warm-start SFT before resuming DAPO, with shaped reward as an owner-gated fallback.

This creates the final co-design loop:

- runtime improvements make rollouts cheaper;
- cheaper rollouts allow more learning experiments;
- a better learning policy produces shorter, more useful completions;
- shorter completions further reduce systems cost.

The boundary between “model quality” and “systems performance” is therefore not fixed. In RL post-training, model behavior is part of the workload presented to the hardware.

## Cross-Accelerator Validation at the Optimized Endpoint

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/gb200-vs-mi350x.png" alt="Figure 11: GB200 vs. MI350X: Qwen3-0.6B GRPO, 256 prompts × 8 generations, 16K/8K caps, strict BF16, synchronous on-policy on vLLM" width="100%">
</p>

<p align="center">
<em>Figure 11: GB200 vs. MI350X: Qwen3-0.6B GRPO, 256 prompts × 8 generations, 16K/8K caps, strict BF16, synchronous on-policy on vLLM</em>
</p>

<p align="center">
<img src="/assets/figures/2026-10-09-rl-small-dense-heterogeneous-hardware/gb200-vs-mi350x-vs-tpu7x.png" alt="Figure 12: GB200 vs. MI350X vs. TPU v7x on Qwen3-0.6B GRPO after the tuning journey" width="100%">
</p>

<p align="center">
<em>Figure 12: GB200 vs. MI350X vs. TPU v7x on Qwen3-0.6B GRPO after the tuning journey</em>
</p>

After completing the full-stack optimization campaign, we compared the best 64-accelerator results under the same frozen Qwen3-0.6B GRPO workload contract. GB200 completed the full cycle in 48 seconds, MI350X in 133.4 seconds, and TPU v7x in approximately 56 seconds.

At the time of publication, TPU v7x was not currently the fastest system for this workload. However, its final result represents a roughly 107× improvement from the initial full-cycle baseline and completes an important cross-accelerator validation: the same model, sequence budget, rollout count, numerical mode, and RL semantics now execute reproducibly across three materially different hardware–software stacks, and we continue to make further improvements.

The figure should therefore be read not as an intrinsic silicon ranking, but as the endpoint of the co-design process—and as evidence that portable RL systems require both a common workload contract and hardware-specific optimization.

## What comes next

The current result closes the primary sub-80-second goal for the true-8K 64-chip workload, but several directions remain:

- complete the closing A-B-A validation for the approximately 56-second Splash configuration;
- validate the optimized stack under a policy that is demonstrably learning, not only executing the full system path;
- automate RPA and Splash block/layout selection from model and sequence shape;
- complete end-to-end integration of replay-exact sharded sampling;
- investigate remaining dispatch/fusion overhead and the 128-chip 4×4×8 path for higher decode throughput;
- evaluate AgenticAsync and disaggregated rollout/training designs with explicit policy-version and behavior-log-prob correctness;
- MPMD rollout enablement and off-policy overlap optimization

The objective is not to declare one accelerator universally best. It is to build a Meta RL platform that can match each stage of the workload to the right hardware while preserving one set of learning semantics and one standard of evidence.

## Conclusion

The project started with a small dense model that appeared easy to run and a full RL cycle that took roughly 100 minutes. The first 7.5× rollout gain came from disciplined configuration and mesh tuning. The next 4.4× came from moving down the Google open-source stack and co-designing the decode path. The final full-cycle gains came from recognizing that cache lifetime and training attention had become the new bottlenecks.

Across the program, stable rollout throughput increased from approximately 4K to 130.5K tokens/s, while end-to-end cycle time fell from roughly 100 minutes to approximately 56 seconds. That is approximately **33× higher rollout throughput** and a **107× faster end-to-end cycle**.

The most important result is not the headline numbers. It is the engineering pattern behind them. Small dense-model RL performance emerged only when we stopped treating the model, framework, runtime, kernels, mesh, and generated sequence distribution as separate systems.

**The model was small. The co-design surface was the entire stack.**
