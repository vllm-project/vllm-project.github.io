---
layout: post
title: "vime × RL-Kernel × AMD: Achieving Bitwise-Consistent Training and Rollout on ROCm"
author: "RL-Kernel Team, vime Team, and AMD Team"
date: 2026-09-14
summary: "vime and RL-Kernel align selected-token logprobs bit for bit across Megatron training and vLLM rollout on AMD Instinct MI300X, with zero mismatch across 200 GRPO steps."
image: /assets/figures/2026-09-14-rl-kernel-v0-1-0/rocm-bitwise-consistency.png
social_image: /assets/figures/2026-09-14-rl-kernel-v0-1-0/rocm-bitwise-consistency.png
tags:
  - reinforcement-learning
  - post-training
  - performance
  - hardware
  - ecosystem
---

vime brings Megatron training, vLLM rollout, and the Data Buffer together in a complete RL post-training workflow. It coordinates sample generation, training, weight updates, and subsequent rollouts so that both engines advance along the same policy timeline.

In a real system, however, the same set of weights does not necessarily produce the same logprob. Training and rollout serve different workloads and may select different kernels, partitioning strategies, reduction orders, and intermediate precision. Even when the model, weights, inputs, and tokens are identical, the two sides can still produce different floating-point results.

RL-Kernel adds an optional strict execution path to vime. vime continues to manage the complete training-and-rollout workflow and weight lifecycle, while RL-Kernel makes Megatron and vLLM follow the same numerical execution contract when computing logprobs.

In an end-to-end Qwen3-8B GRPO experiment on AMD Instinct MI300X, the strict vime + RL-Kernel path ran for 200 consecutive steps. At every step, the logprobs recomputed by the training engine and those recorded by the rollout engine satisfied:

<p style="text-align:center;"><strong>200 steps · mismatch_count = 0 · max_abs_diff = 0</strong></p>

This post focuses on three questions: why mismatch occurs, which parts are handled by vime and RL-Kernel, and how we verify bitwise consistency while preserving native ROCm execution paths.

## Why RL Training Requires Train–Rollout Numerical Consistency

Training and rollout typically use different execution engines and operators. Even with the same model, weights, and inputs, different kernels, parallelization strategies, and reduction orders can still produce different logprobs.

This is a long-standing problem in RL systems because it affects importance ratios, KL divergence, and clipping. Systematic end-to-end investigation of this issue on ROCm remains comparatively limited.

Building on vime × RL-Kernel, we align the Attention, FFN, logprob, and communication paths on AMD Instinct MI300X to achieve bitwise-consistent training and rollout.

The rollout engine generates token *a<sub>t</sub>* from prefix *h<sub>t</sub>* and records:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/equation-rollout-logprob.png" alt="Rollout logprob definition" style="display:block;margin:0 auto;width:2.35083in;max-width:100%;" />

Before training begins, Megatron scores the same token again using the same weight version:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/equation-training-logprob.png" alt="Training logprob definition" style="display:block;margin:0 auto;width:2.19917in;max-width:100%;" />

On-policy RL uses the two values to construct the importance ratio:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/equation-importance-ratio.png" alt="Importance ratio" style="display:block;margin:0 auto;width:2.00417in;max-width:100%;" />

If the policy has not yet changed and both sides are evaluating the same logical object, then ideally:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/equation-ideal-ratio.png" alt="Ideal equality before a policy update" style="display:block;margin:0 auto;width:2.01139in;max-width:100%;" />

Let:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/equation-logprob-delta.png" alt="Logprob difference" style="display:block;margin:0 auto;width:1.44444in;max-width:100%;" />

When δ<sub>t</sub> is small, ρ<sub>t</sub> ≈ 1 + δ<sub>t</sub>. This error also enters the PPO or GRPO clipped objective:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/equation-clipped-objective.png" alt="Clipped policy objective" style="display:block;margin:0 auto;width:4.39472in;max-width:100%;" />

The total discrepancy can be decomposed further. Let *q<sub>t</sub> = exp(ℓ<sub>t</sub><sup>T</sup>)* be the probability from training scoring and *μ<sub>t</sub> = exp(ℓ<sub>t</sub><sup>R</sup>)* be the probability recorded by rollout. Let *s<sub>t</sub><sup>P</sup>* and *s<sub>t</sub><sup>D</sup>* be the probabilities assigned to the same token by serving prefill and an independent decode replay:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/equation-error-decomposition.png" alt="Train-rollout error decomposition" style="display:block;margin:0 auto;width:1.67194in;max-width:100%;" />

The first term compares training scoring with serving prefill, the second compares prefill with decode, and the third checks the weight version, KV-cache state, and rollout-record identity. This decomposition matters: a single ratio spans two engines, two internal vLLM execution paths, and a system-state boundary. If any one of these is not fixed, the aggregate difference should not be attributed loosely to a kernel.

Train–rollout mismatch can therefore make the training objective observe an extra policy shift before the parameters are actually updated. If the error is large enough, it may also change which clipping branch is selected.

This integration establishes a stricter target: bit-for-bit equality of independently computed logprobs.

## Why the Same Model Can Produce Different Results

GPUs perform finite-precision floating-point arithmetic, so rounding can occur at every step. As a simple example, let *a = 100000000*, *b = −100000000*, and *c = 1*. Computing *(a + b) + c* gives 1. Computing *a + (b + c)* may instead give 0 because *b + c* can round back to −100000000 in finite precision.

Training and inference face the same issue. Training is optimized for packed sequences, backward propagation, and cross-device parallelism. Inference is optimized for prefill, decode, dynamic batching, and paged KV cache. Even with identical models, weights, and inputs, the two sides may choose different block sizes, Split-K or Split-KV strategies, reduction orders, fusion boundaries, and intermediate precision.

The same model therefore does not imply that training and rollout execute the same floating-point program, and it does not guarantee identical logprobs.

We can write the actual execution of a computation node *v* as:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/equation-execution-node.png" alt="Execution under a numerical contract" style="display:block;margin:0 auto;width:1.53833in;max-width:100%;" />

Here, *C<sub>v</sub>* is the node's numerical execution contract:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/equation-numerical-contract.png" alt="Numerical execution contract" style="display:block;margin:0 auto;width:2.85278in;max-width:100%;" />

- **D<sub>v</sub>:** The elements and reduction domain on which the output actually depends.
- **Π<sub>v</sub>:** How the reduction domain is partitioned into partial results.
- **T<sub>v</sub>:** The order in which partial results are merged.
- **P<sub>v</sub>:** The precision of inputs, accumulators, intermediate states, and outputs.
- **Q<sub>v</sub>:** Where rounding or downcasting occurs.
- **A<sub>v</sub>:** The numerical primitives used for exp, log, rsqrt, SiLU, FMA, and related operations.

The candidate numerical divergence between training and rollout can be written as:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/equation-contract-difference.png" alt="Difference between training and rollout numerical contracts" style="display:block;margin:0 auto;width:2.48083in;max-width:100%;" />

Bitwise consistency means that both sides follow the same observable numerical contract along the forward path being compared.

## vime Aligns the Timeline; RL-Kernel Aligns Numerical Execution

Their respective roles are:

- **vime aligns the training timeline:** which token batch belongs to which step, which weight version generated it, which rollout record enters an update, and when new weights are synchronized to vLLM.
- **RL-Kernel aligns numerical execution:** which values participate in a computation, how they are partitioned and merged, which intermediate precision is used, and where rounding occurs.

Without vime's timeline synchronization, even deterministic kernels may compare different weight versions or different tokens. Without RL-Kernel's numerical alignment, two engines may interpret the same model through different floating-point paths even when the weight version is identical.

On ROCm, these numerical rules must ultimately be implemented in the operators, compiler, and communication stack. To complete that work, we added deterministic GEMM using AMD MFMA, aligned vocabulary reduction and HIP IPC communication, fixed the execution schedule for AITER/CK Attention, addressed last-bit differences introduced by math functions and compiler fusion, and corrected state issues in paged-KV layout and HIP Graph replay. With these adaptations, the weights and tokens aligned by vime follow a consistent numerical path through training and rollout. On 8× AMD Instinct MI300X with Qwen3-8B, logprobs remained bitwise identical for 200 consecutive training and rollout steps.

## Confirming That Both Sides Compute the Same Object

Before comparing floating-point results, we verify that the following match:

- checkpoint and weight version;
- prefix, token, and active mask;
- position, RoPE, causal mask, and padding mask;
- logical K/V after paged-KV-cache mapping;
- sequence, head, and vocabulary ownership;
- the true vocabulary range and any random state relevant to the comparison.

If any condition differs, the sample should be marked comparable = false; the final difference cannot be attributed directly to a kernel.

## Why Numerical Divergence Across the Transformer Must Be Addressed Together

RMSNorm, GEMM, Attention, linear logp, and distributed collectives may appear to be separate modules, but all of them contain reductions:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/equation-nested-reduction.png" alt="Nested reductions across Transformer operations" style="display:block;margin:0 auto;width:4.42361in;max-width:100%;" />

Under finite precision, block size, Split-K or Split-KV, reduction trees, collective trees, exp and log implementations, intermediate precision, and fusion can all change the merge order and rounding boundaries of partial results. Attention's Split-KV merge, linear logprob's cross-TP vocabulary merge, GEMM's K-dimension reduction, and ROCm collectives are all part of the same numerical chain.

The RL-Kernel strict path therefore fixes the reduction domain, partition and merge order, intermediate precision, numerical primitives, and fallback behavior together. Fixing only one kernel or collective is not sufficient to guarantee bitwise-consistent logprobs in vime.

## What RL-Kernel Fixes in vime

The RL-Kernel strict path covers the main forward boundaries that determine logprob:

| **Computation boundary** | **What the strict path fixes** |
|---|---|
| RMSNorm | Reduction range, epsilon, residual-add behavior, and output boundary |
| Attention | Position, mask, logical paged KV, split policy, LSE precision, and final cast |
| GEMM and SwiGLU | K-dimension reduction, accumulation precision, epilogue, activation, and materialization boundary |
| Linear logprob | True vocabulary range, target ownership, local reduction, and cross-rank LSE merge |
| Distributed collectives | Payload ownership, dtype, and fixed rank-reduction order |

This contract does not require training and rollout to share every memory layout or scheduling policy. The two engines may still optimize independently as long as those optimizations do not change the numerical semantics of the compared result.

## Using Ablations to Locate the First Divergence

In operator-ablation experiments, **P** denotes the production path and **R** denotes the RL-Kernel strict path. The left side of the slash is training and the right side is rollout:

| **Combination** | **Training** | **Rollout** | **Purpose** |
|---|---|---|---|
| P/P | Production | Production | Observe native vime execution |
| P/R | Production | RL-Kernel | Replace only the rollout path and check for divergence |
| R/P | RL-Kernel | Production | Replace only the training path and check for divergence |
| R/R | RL-Kernel | RL-Kernel | Fully aligned strict control path |

When isolating Attention, the remaining FFN, logprob, and collective paths must stay on the same baseline. The same principle applies when isolating FFN. Changing one boundary at a time makes it possible to trace the final logprob difference back to the first nonzero output.

## A 200-Step vime Alignment Experiment on ROCm

We completed a strict 200-step validation in a full vime workflow composed of Megatron training and vLLM rollout, with zero mismatch throughout. This is an end-to-end validation of vime + RL-Kernel. vime manages rollout, training, weight synchronization, and the sample lifecycle; RL-Kernel aligns the numerical execution path for logprob within the same workflow.

### Experimental Configuration

| **Item** | **Configuration** |
|---|---|
| Model / dtype | Qwen3-8B / BF16 |
| Hardware | 1 node, 8× AMD Instinct MI300X 192GB |
| Megatron | TP4 / CP2 / PP1 across 8 GPUs |
| Rollout | 2 vLLM engines, each using TP4 |
| Placement | Actor and rollout colocated |
| Horizon | 200 rollout/training steps |
| Dataset | dapo-math-17k |
| Seeds | Training 1234, rollout 1234 |
| Sampling | 1 prompt × 8 samples per step; global batch 8 |
| Response limit | 7,168 tokens |
| Dynamic batching | Maximum 4,096 tokens/GPU |
| vLLM memory utilization | 0.38 |
| HIP Graph | FULL_AND_PIECEWISE, preserving the production graph path |
| KL loss | Enabled, coefficient 0.001 |
| Validation requirement | Frozen inputs and sources remain identical before and after the run; every step passes runtime-provenance and mismatch validation |

Across all 200 steps of the strict path, mismatch_count and max_abs_diff are both zero.

### 200-Step Training Trajectory

Figure 1 plots the train–rollout mismatch count and maximum absolute Δlogp on the same 200-step timeline. The RL-Kernel strict path remains at zero mismatch throughout, while the native vime path shows a mismatch at every step.

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/rocm-bitwise-consistency.png" alt="Training and bitwise consistency across 200 ROCm steps" style="display:block;margin:0 auto;width:6.5in;max-width:100%;" />

<p style="text-align:center;opacity:0.7;font-size:0.95em;"><em>Figure 1: Consistency comparison between native vime and vime + RL-Kernel.</em></p>

These signals emerge over the same interval and are consistent with persistent train–rollout mismatch. Together, they provide end-to-end evidence for strict alignment. The experiment directly shows that vime + RL-Kernel can maintain verifiable bitwise consistency and a more stable training trajectory throughout the 200-step run.

Figure 2 shows the mean absolute train–rollout logprob difference over 200 steps. vime + RL-Kernel remains at zero throughout.

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/rocm-mean-abs-logprob-diff.png" alt="Mean absolute train-rollout logprob difference across 200 ROCm steps" style="display:block;margin:0 auto;width:6.5in;max-width:100%;" />

<p style="text-align:center;opacity:0.7;font-size:0.95em;"><em>Figure 2: Mean absolute train–rollout logprob difference across 200 steps.</em></p>

Figure 3 compares the performance of native vime and vime + RL-Kernel across the 200-step run.

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/rocm-performance-matrix.png" alt="Performance comparison between native vime and vime plus RL-Kernel" style="display:block;margin:0 auto;width:6.5in;max-width:100%;" />

<p style="text-align:center;opacity:0.7;font-size:0.95em;"><em>Figure 3: Performance matrix for native vime and the strict vime + RL-Kernel path.</em></p>

## What vime × RL-Kernel Achieves on ROCm

- **Bitwise correctness:** Training and rollout logprobs match exactly on ROCm. Across all 200 steps, mismatch_count remains zero and the maximum logprob difference is also zero.
- **Controlled end-to-end overhead:** The mean end-to-end step time is 110.76 seconds for RL-Kernel + vime, compared with 94.66 seconds for native vime—an overhead of approximately 17% for the strict consistency path.
- **Stable consistency guarantees:** Zero mismatch is maintained throughout the 200-step end-to-end training run, making results easier to verify and reproduce.
- **Complete ROCm execution evidence:** The validation records the kernels, HIP Graph execution, paged KV, collectives, and fallback paths actually used at runtime.
- **Fast failure localization:** Operator ablations identify the specific operator or system boundary where train–rollout divergence begins.

On 8× AMD Instinct MI300X, RL-Kernel + vime maintained zero mismatch across all 200 steps with approximately 17% end-to-end overhead. The result moves strict train–rollout consistency beyond correctness validation toward a ROCm implementation with quantifiable performance cost, traceable execution paths, and reproducible experimental results—providing a foundation for production deployment and further optimization.

## Current Scope and Next Steps

The current end-to-end validation covers Qwen3-8B Dense, vime, vLLM, Megatron-LM, and AMD Instinct MI300X. Next, we plan to extend the work to additional MoE and multimodal models, more AMD GPU architectures, and further performance optimization of the strict ROCm path.

## Acknowledgements

This collaboration between vime, RL-Kernel, and AMD would not have been possible without the support of our hardware partners, open-source ecosystem collaborators, and development teams. We thank the vLLM community for working closely with RL-Kernel. We especially thank Ao Shen, a vime maintainer at Inferact, for supporting the vime integration, community coordination, and ongoing maintenance.

We sincerely thank Liz Li and Yuhan Yang from AMD for providing AMD Instinct GPU compute resources, deep technical collaboration, and long-term support, enabling vime + RL-Kernel to complete end-to-end train–rollout consistency validation on ROCm. We also thank Lei Ding from Moore Threads for advancing RL-Kernel support for MUSA, Yang Chen from Huawei for advancing RL-Kernel support for Ascend, and Embedded LLM for supporting project development and community collaboration.

The core contributors to dense-model train–rollout consistency in RL-Kernel v0.1.0 are Chutian Wang, Jiajie Li, Siru He, Xiaosong Ma, Kaijie Lin, Jian Zhang, Huihong Lu, Yunxiang Cai, Bosong Yang, Zhewei Liu, Houhong Liang, Ryan Huang, and Vensen Mu.

We also thank the contributors whose pull requests were merged into v0.1.0: Xiaopeng Du, Yuepeng Pan, Yiyang Fei, Ziying Tao, Zhifu Liu, Zhengtao Chen, Mengjie Li, Zien Liu, and GitHub users haoruilee, luoyueyuguang, hongleng, and smarslou.
