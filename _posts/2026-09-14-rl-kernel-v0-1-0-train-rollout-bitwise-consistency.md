---
layout: post
title: "vime × RL-Kernel × AMD: Bitwise Train–Rollout Consistency on ROCm"
author: "RL-Kernel Team, vime Team, and AMD Team"
date: 2026-09-14
summary: "vime and RL-Kernel align selected-token logprobs bit for bit across Megatron training and vLLM rollout on AMD Instinct MI300X, with zero mismatches across 200 GRPO steps."
image: /assets/figures/2026-09-14-rl-kernel-v0-1-0/rocm-bitwise-consistency.png
social_image: /assets/figures/2026-09-14-rl-kernel-v0-1-0/rocm-bitwise-consistency.png
math: true
tags:
  - reinforcement-learning
  - post-training
  - performance
  - hardware
  - ecosystem
---

## Why RL Training Needs Train–Rollout Numerical Consistency

Training and rollout usually use different execution engines and operators. Even with the same model, weights, and inputs, differences in kernels, parallelism, and reduction order can produce different logprobs.

This is a long-standing issue in RL systems because it affects the importance ratio, KL, and clipping. Existing work has focused mainly on NVIDIA platforms, while systematic exploration on ROCm has been more limited.

Building on vime × RL-Kernel, this work aligns the Attention, FFN, logprob, and communication paths on AMD Instinct MI300X to achieve bitwise train–rollout consistency.

Concretely, the rollout engine generates token $a_t$ from prefix $h_t$ and records logprob $\ell_t^R$. The training engine then recomputes the same token under the same weight version to obtain $\ell_t^T$. We track the difference $\delta_t = \ell_t^T - \ell_t^R$ and the corresponding importance ratio $\rho_t = \exp(\delta_t)$.

Before the policy update, both sides should be scoring the same token with the same weights. The ideal target is therefore $\delta_t = 0$ and $\rho_t = 1$. Any non-zero difference introduces an additional numerical policy shift, which can propagate into KL and clipping. The strict target of this integration is to make the logprob difference bitwise zero.

## Why the Same Model Can Produce Different Results

GPUs use finite-precision floating-point arithmetic, so rounding can occur at every step. Consider a simple example with $a = 100000000$, $b = -100000000$, and $c = 1$. Computing $(a + b) + c$ gives 1, while $a + (b + c)$ may give 0 because $b + c$ can round back to $-100000000$ at finite precision.

Training and inference encounter the same issue. Training is optimized for packed sequences, backpropagation, and multi-GPU parallelism; inference is optimized for prefill, decode, dynamic batching, and paged KV cache. Even when the model, weights, and inputs are identical, the two sides may use different block sizes, Split-K or Split-KV strategies, reduction orders, fusion patterns, and intermediate precision.

Therefore, the same model does not necessarily execute the same floating-point computation in training and rollout, nor does it guarantee identical logprobs.

We describe the observable numerical behavior of a compute node $v$ with a numerical execution contract:

$$
C_v = (D_v, \Pi_v, T_v, P_v, Q_v, A_v)
$$

- $D_v$: the elements on which the output actually depends, together with the reduction domain.
- $\Pi_v$: how the reduction domain is partitioned into partial results.
- $T_v$: the order in which partial results are merged.
- $P_v$: the precision of inputs, accumulators, intermediate states, and outputs.
- $Q_v$: where rounding or downcasting occurs.
- $A_v$: the numerical primitives actually used, such as exp, log, rsqrt, SiLU, and FMA.

Bitwise consistency requires training and rollout to follow the same observable numerical contract along the forward path being compared.

## vime Aligns the Timeline; RL-Kernel Aligns Numerical Execution

Their respective roles are:

- **vime aligns the training timeline:** which token batch belongs to which step, which weight version generated it, which rollout record enters an update, and when new weights are synchronized to vLLM.
- **RL-Kernel aligns numerical execution:** which values participate in a computation, how they are partitioned and merged, which intermediate precision is used, and where rounding occurs.

Without vime's timeline synchronization, even deterministic kernels may compare different weight versions or different tokens. Without RL-Kernel's numerical alignment, two engines may interpret the same model through different floating-point paths even when the weight version is identical.

This division of labor is the same on CUDA and ROCm, but the numerical rules must ultimately be implemented in each platform's operators, compiler, and communication stack. Paths already validated on CUDA therefore had to be adapted and revalidated on ROCm. We added deterministic AMD MFMA GEMM, vocabulary reduction, and HIP IPC communication; fixed the execution schedule of AITER/CK Attention; addressed last-bit differences caused by math functions and compiler fusion; and fixed state issues in paged KV layout and HIP Graph replay.

After these adaptations, the weights and tokens aligned by vime could pass through training and inference along the same numerical path. On an 8× MI300X Qwen3-8B configuration, the resulting logprobs remained bitwise identical for 200 consecutive training and rollout steps.

### Confirming That Both Sides Compute the Same Object

Before comparing floating-point results, we verify that the following match:

- checkpoint and weight version;
- prefix, token, and active mask;
- position, RoPE, causal mask, and padding mask;
- logical K/V after paged KV cache mapping;
- sequence, head, and vocabulary ownership;
- the true vocabulary range and any random state relevant to the comparison.

If any condition differs, the sample is marked $\mathrm{comparable} = \mathrm{false}$, and the final difference is not attributed to kernels.

### Why Transformer Numerical Divergences Must Be Handled Together

RMSNorm, GEMM, Attention, linear logprob, and distributed collectives look like separate modules, but all of them merge partial results. Block size, Split-K or Split-KV, reduction trees, collective trees, exp and log implementations, intermediate precision, and fusion can all change the merge order and rounding boundaries.

Attention's Split-KV merge, linear logprob's cross-TP vocabulary merge, GEMM's K-dimension reduction, and ROCm collectives are therefore parts of one numerical chain. The RL-Kernel strict path fixes these boundaries together; pinning down a single kernel or collective is not enough to guarantee bitwise-identical logprobs.

## Which Boundaries RL-Kernel Fixes in vime

The RL-Kernel strict path covers the main forward boundaries that determine logprobs:

| Compute boundary | What the strict path fixes |
|---|---|
| RMSNorm | Reduction domain, epsilon, residual addition, and output boundary |
| Attention | Position, mask, logical paged KV, split policy, LSE precision, and final cast |
| GEMM and SwiGLU | K-dimension reduction, accumulation precision, epilogue, activation, and materialization boundary |
| Linear logprob | True vocabulary range, target ownership, local reduction, and cross-rank LSE merge |
| Distributed collectives | Payload ownership, dtype, and a fixed rank reduction order |

This contract does not require training and rollout to share every memory layout or scheduling policy. The two engines can still optimize independently as long as those optimizations do not change the numerical semantics of the compared results.

## Locating the First Divergence with Ablations

In operator ablations, P denotes the production path and R denotes the RL-Kernel strict path. The left side of the slash is training; the right side is rollout:

| Combination | Training | Rollout | Purpose |
|---|---|---|---|
| P/P | Production | Production | Observe native vime behavior |
| P/R | Production | RL-Kernel | Replace only the rollout side |
| R/P | RL-Kernel | Production | Replace only the training side |
| R/R | RL-Kernel | RL-Kernel | Run the fully aligned strict control path |

When localizing Attention, FFN, logprob, or collectives, all other boundaries remain on the same baseline. Changing one boundary at a time lets us trace the final logprob difference back to the first non-zero output.

## 200-Step Alignment Experiment with vime on ROCm

We completed a strict 200-step validation on ROCm within the full vime workflow of Megatron training and vLLM rollout, with zero mismatches throughout. vime handled rollout, training, weight synchronization, and the sample lifecycle, while RL-Kernel aligned the numerical execution path used to compute logprobs in the same pipeline.

### Experimental Setup

| Item | Configuration |
|---|---|
| Model / dtype | Qwen3-8B / BF16 |
| Hardware | 1 node, 8× AMD Instinct MI300X 192GB |
| Megatron | TP4 / CP2 / PP1, using 8 GPUs |
| Rollout | 2 vLLM engines, TP4 each |
| Placement | Actor and rollout colocated |
| Horizon | 200 rollout/training steps |
| Dataset | dapo-math-17k |
| Seeds | Training 1234, rollout 1234 |
| Sampling | 1 prompt × 8 samples per step, global batch 8 |
| Response limit | 7,168 tokens |
| Dynamic batching | Maximum 4,096 tokens/GPU |
| vLLM memory utilization | 0.38 |
| HIP Graph | FULL_AND_PIECEWISE, preserving the production graph execution path |
| KL loss | Enabled, coefficient 0.001 |
| Validation | Inputs and source values were frozen before and after the run; every step had to pass provenance and mismatch checks |

Across all 200 steps of the strict path, both mismatch_count and max_abs_diff remained zero.

### 200-Step Training Trajectory

Figure 1 plots the train–rollout mismatch count and maximum absolute $\Delta\log p$ on the same 200-step timeline. The RL-Kernel strict path remains at zero throughout, while native vime shows a mismatch at every step.

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/rocm-bitwise-consistency.png" alt="Consistency comparison between native vime and vime plus RL-Kernel" style="display:block;margin:0 auto;width:6.5in;max-width:100%;" />

<p style="text-align:center;opacity:0.7;font-size:0.95em;"><em>Figure 1: Consistency comparison between native vime and vime + RL-Kernel.</em></p>

These signals appear over the same interval and are consistent with persistent train–rollout mismatch. Together, they provide end-to-end evidence for strict alignment. The experiment shows that vime + RL-Kernel can maintain verifiable bitwise consistency and a more stable training trajectory throughout the 200-step run.

Figure 2 shows the mean absolute train–rollout logprob difference over 200 steps. vime + RL-Kernel remains at zero throughout.

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/rocm-mean-abs-logprob-diff.png" alt="Mean absolute train-rollout logprob difference across 200 ROCm steps" style="display:block;margin:0 auto;width:6.5in;max-width:100%;" />

<p style="text-align:center;opacity:0.7;font-size:0.95em;"><em>Figure 2: Mean absolute train–rollout logprob difference across 200 steps.</em></p>

## What vime × RL-Kernel Achieves on ROCm

- **Bitwise correctness:** Training and rollout logprobs match exactly on ROCm. Across all 200 steps, mismatch_count remains zero and the maximum logprob difference is also zero.
- **Stable consistency guarantees:** Zero mismatch is maintained throughout the 200-step end-to-end training run, making results easier to verify and reproduce.
- **Complete ROCm execution evidence:** The validation records the kernels, HIP Graph execution, paged KV, collectives, and fallback paths actually used at runtime.
- **Fast failure localization:** Operator ablations identify the specific operator or system boundary where train–rollout divergence begins.

On 8× AMD Instinct MI300X, RL-Kernel + vime maintained zero mismatch across all 200 steps. This result moves strict train–rollout consistency beyond correctness validation toward a ROCm implementation with quantifiable performance cost, traceable execution paths, and reproducible experimental results, providing a foundation for production deployment and further optimization.

## Current Scope and Next Steps

The current end-to-end validation covers Qwen3-8B Dense, vime, vLLM, Megatron-LM, and AMD Instinct MI300X. Next, we plan to extend the work to more MoE and multimodal models and additional AMD GPU architectures, while continuing to optimize the ROCm strict path.

## Acknowledgments

This integration of vime, RL-Kernel, and AMD would not have been possible without the support of our hardware partners, open-source ecosystem collaborators, and development teams. We thank the vLLM community for its close collaboration with RL-Kernel, and especially Ao Shen, vime maintainer at Inferact, for his trust and support in the vime integration, community coordination, and ongoing maintenance.

We sincerely thank Liz Li and Yuhan Yang of AMD for providing AMD Instinct GPU compute resources, in-depth technical collaboration, and long-term support for RL-Kernel. Their support made the end-to-end validation of vime + RL-Kernel on ROCm possible. We also thank Lei Ding of Moore Threads for advancing RL-Kernel's MUSA support, Yang Chen of Huawei for advancing its Ascend support, and Embedded LLM for supporting the project's development and community collaboration.

Core contributors to dense-model train–rollout consistency in RL-Kernel v0.1.0: Chutian Wang, Jiajie Li, Siru He, Xiaosong Ma, Kaijie Lin, Jian Zhang, Huihong Lu, Yunxiang Cai, Bosong Yang, Zhewei Liu, Houhong Liang, Ryan Huang, and Vensen Mu.

We also thank contributors whose PRs were merged into v0.1.0: Xiaopeng Du, Yuepeng Pan, Yiyang Fei, Ziying Tao, Zhifu Liu, Zhengtao Chen, Mengjie Li, Zien Liu, and GitHub users haoruilee, luoyueyuguang, hongleng, and smarslou.
