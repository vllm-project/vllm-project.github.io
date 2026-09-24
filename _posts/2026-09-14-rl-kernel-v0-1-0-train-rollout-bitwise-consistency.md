---
layout: post
title: "vime × RL-Kernel × AMD: Bitwise Train–Rollout Consistency on ROCm"
author: "RL-Kernel Team, vime Team, and AMD Team"
date: 2026-09-14
summary: "vime and RL-Kernel align selected-token logprobs bit for bit across Megatron training and vLLM rollout on AMD Instinct MI300X, with zero mismatch across 200 GRPO steps."
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

vime connects Megatron training, vLLM rollout, and a Data Buffer into a complete RL post-training workflow. It orchestrates sample generation, training, weight updates, and the next round of rollout, so that both engines advance along the same policy timeline.

In real systems, however, the same weights do not necessarily produce the same logprobs. Training and rollout serve different workloads and therefore choose different kernels, tiling schemes, reduction orders, and intermediate precisions. Even when the model, weights, inputs, and tokens are all identical, the two sides can still produce different floating-point results.

RL-Kernel adds an optional strict execution path to vime. vime remains responsible for end-to-end train/rollout orchestration and the weight lifecycle, while RL-Kernel makes Megatron and vLLM follow the same numerical execution contract when computing logprobs.

In an end-to-end GRPO experiment with Qwen3-8B on AMD Instinct MI300X, the vime + RL-Kernel strict path ran for 200 consecutive steps. At every step, the logprobs recomputed on the training side matched those recorded on the rollout side:

<p style="text-align:center;"><strong>200 steps · mismatch_count = 0 · max_abs_diff = 0</strong></p>

This post focuses on three questions: why mismatches occur, which part of the problem vime and RL-Kernel each solve, and how we verified bitwise consistency while preserving ROCm's native execution path.

---

## Why RL Training Needs Train–Rollout Numerical Consistency

Training and rollout usually run on different execution engines and operators. Even with the same model, weights, and inputs, differences in kernels, parallelism, and reduction order can produce different logprobs.

This is a long-standing problem in RL systems, affecting the importance ratio, KL, and clipping. Existing work has focused mainly on NVIDIA platforms, with relatively little systematic exploration on ROCm.

Building on vime × RL-Kernel, this work aligns the Attention, FFN, logprob, and communication paths on AMD MI300X to achieve bitwise train–rollout consistency.

Concretely, the same token is both recorded by rollout and recomputed by training, and the discrepancy can be expressed as follows.

The rollout engine generates token $a_t$ given prefix $h_t$ and records:

$$
\ell_t^R = \log p_{\text{rollout}}(a_t \mid h_t)
$$

Before training, Megatron rescores this token with the same weight version:

$$
\ell_t^T = \log p_{\text{train}}(a_t \mid h_t)
$$

On-policy RL uses the two to construct the importance ratio:

$$
\rho_t = \exp\left(\ell_t^T - \ell_t^R\right)
$$

If the policy has not yet been updated and both sides compute the same logical object, then ideally:

$$
\ell_t^T = \ell_t^R, \qquad \rho_t = 1
$$

Let

$$
\delta_t = \ell_t^T - \ell_t^R
$$

When $\delta_t$ is small, $\rho_t \approx 1 + \delta_t$. This error then enters the clipped objective of PPO or GRPO:

$$
L_t = \min\left(\rho_t \hat{A}_t,\ \operatorname{clip}(\rho_t,\ 1-\varepsilon_{\text{low}},\ 1+\varepsilon_{\text{high}})\,\hat{A}_t\right)
$$

The total error can be further decomposed. Let the training-scoring probability be $q_t = \exp(\ell_t^T)$, the rollout-recorded probability be $\mu_t = \exp(\ell_t^R)$, and let $s_t^P$ and $s_t^D$ denote the probabilities of the same token under serving prefill and an independent decode replay. Then:

$$
\frac{q_t}{\mu_t} = \frac{q_t}{s_t^P} \times \frac{s_t^P}{s_t^D} \times \frac{s_t^D}{\mu_t}
$$

The first term compares training scoring with serving prefill; the second compares prefill with decode; the third checks the weight version, KV cache state, and rollout record identity. This decomposition matters: there is only one final ratio, but behind it lie two engines (training and inference), two execution paths inside vLLM, and system-state boundaries. If any of these is not pinned down, the total discrepancy should not be blindly attributed to kernels.

As a result, train–rollout mismatch can make the training objective observe an extra policy shift before the parameters have actually been updated. When the error is large enough, it can even change which branch of the clipping is taken.

This integration goes further and sets a stricter target: driving the logprob difference all the way to bitwise equality.

---

## Why the Same Model Produces Different Results

GPUs compute with finite-precision floating point, and rounding can occur at every step. So even with identical inputs, a different computation order can yield a different final result. A simple example: let $a = 100000000$, $b = -100000000$, $c = 1$. Computing $(a + b) + c$ gives 1; but computing $a + (b + c)$, since $|b|$ is far larger than $c$, $b + c$ may round to $-100000000$ in finite precision, and the final result may become 0.

Training and inference face exactly the same problem. The training side is built for packed sequences, backpropagation, and multi-GPU parallelism; the inference side is built for prefill, decode, dynamic batching, and paged KV cache. Even with identical model, weights, and inputs, the two sides may use different block sizes, Split-K / Split-KV, reduction orders, fusion, and intermediate precision.

Therefore, "the same model" does not mean training and rollout execute exactly the same floating-point computation, nor does it guarantee identical logprobs.

The actual execution of a compute node $v$ can be written as:

$$
\hat{y}_v = \hat{f}_v(x;\ C_v)
$$

where $C_v$ is the node's numerical execution contract:

$$
C_v = (D_v,\ \Pi_v,\ T_v,\ P_v,\ Q_v,\ A_v)
$$

- $D_v$: the elements the output actually depends on, and the reduction domain.
- $\Pi_v$: how the reduction domain is partitioned into multiple partial results.
- $T_v$: the order in which partial results are merged.
- $P_v$: the precision of inputs, accumulators, intermediate states, and outputs.
- $Q_v$: where rounding or downcasting happens.
- $A_v$: the numerical primitives actually used: exp, log, rsqrt, SiLU, FMA, etc.

The candidate numerical divergence between training and rollout can be written as:

$$
\Delta C_v = C_v^{\text{train}} \ \triangle\ C_v^{\text{rollout}}
$$

Bitwise consistency means making both sides follow the same observable numerical contract along the forward path being compared.

---

## vime Aligns the Timeline, RL-Kernel Aligns Numerical Execution

The division of labor can be summarized as:

- **vime aligns the training timeline:** which batch of tokens belongs to which step, which weight version generated them, which rollout record feeds which update, and when new weights are synced to vLLM.
- **RL-Kernel aligns numerical execution:** which values participate in the computation, how they are tiled and merged, what intermediate precision is used, and at which boundary rounding occurs.

This division of labor is the same on CUDA and ROCm, but the numerical rules ultimately have to be realized in each platform's operators, compilers, and communication implementations. Paths already validated on CUDA therefore need to be adapted and re-verified item by item on ROCm. To do so, we added deterministic AMD MFMA GEMM, vocabulary reduction, and HIP IPC communication; fixed the compute schedule of AITER/CK Attention; handled last-bit differences caused by math functions and compiler fusion; and fixed state issues in paged KV layout and HIP Graph replay. Only after these adaptations can the weights and tokens aligned by vime flow through training and inference along a consistent numerical path, ultimately keeping logprobs bitwise identical for 200 consecutive training and rollout steps on the 8×MI300X, Qwen3-8B validation configuration.

### Confirming Both Sides Compute the Same Object

Before comparing floating-point results, check:

- checkpoint and weight version;
- prefix, tokens, and active mask;
- position, RoPE, causal mask, and padding mask;
- the logical K/V after paged KV cache mapping;
- sequence, head, and vocabulary ownership;
- the true vocabulary range and any random state relevant to the comparison.

If any of the above is inconsistent, the sample should be marked comparable = false, and the final difference must not be attributed directly to kernels.

### Why Numerical Divergences in a Transformer Must Be Handled Together

RMSNorm, GEMM, Attention, linear logp, and distributed collectives look like separate modules, but they all contain reductions:

$$
\operatorname{Agg}(R) = \operatorname{Merge}\left(\operatorname{Agg}(R^{(1)}),\ \ldots,\ \operatorname{Agg}(R^{(m)})\right)
$$

Under finite precision, block size, Split-K/Split-KV, reduction tree, collective tree, exp/log implementation, intermediate precision, and fusion all change the merge order and rounding boundaries of partial results. Attention's Split-KV merge, linear logprob's cross-TP vocabulary merge, GEMM's K-dimension reduction, and ROCm collectives are essentially one numerical chain.

The RL-Kernel strict path therefore pins down, all at once, the reduction domain, tiling and merge order, intermediate precision, numerical primitives, and fallback behavior. Pinning down a single kernel or collective alone is still not enough to guarantee bitwise-consistent logprobs in vime.

---

## Which Boundaries RL-Kernel Pins Down in vime

The RL-Kernel strict path covers the main forward boundaries that determine logprobs:

| Compute boundary | What the strict path pins down |
|---|---|
| RMSNorm | Reduction range, epsilon, residual-add behavior, and output boundary |
| Attention | Position, mask, logical paged KV, split policy, LSE precision, and final cast |
| GEMM & SwiGLU | K-dimension reduction, accumulation precision, epilogue, activation, and materialization boundary |
| Linear logprob | True vocabulary range, target ownership, local reduction, and cross-rank LSE merge |
| Distributed collectives | Payload ownership, dtype, and a fixed rank reduction order |

This contract does not require training and rollout to share all memory layouts and scheduling policies. The two engines can still optimize independently, as long as those optimizations do not change the numerical semantics of the compared results.

---

## Locating the First Divergence with Ablations

In operator ablations, P denotes the production path and R denotes the RL-Kernel strict path. The left of the slash is training; the right is rollout:

| Combination | Training | Rollout | Purpose |
|---|---|---|---|
| P/P | Production | Production | Observe results of vime's native path |
| P/R | Production | RL-Kernel | Replace only the rollout side; check for divergence |
| R/P | RL-Kernel | Production | Replace only the training side; check for divergence |
| R/R | RL-Kernel | RL-Kernel | Fully aligned strict control path |

When localizing Attention, FFN, logprob, and collectives should remain at the same baseline; the same principle applies when localizing FFN. Only by changing one boundary at a time can the final logprob difference be traced back to the first non-zero output.

---

## 200-Step Alignment Experiment with vime on ROCm

We completed a strict 200-step validation on ROCm within the full vime workflow composed of Megatron training and vLLM rollout, with zero mismatches throughout. This is an end-to-end validation of vime + RL-Kernel: vime handles rollout, training, weight synchronization, and the sample lifecycle, while RL-Kernel aligns the numerical execution path of logprobs within the same pipeline.

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
| Validation requirement | Frozen inputs and frozen sources remain unchanged before and after the run; every step must pass runtime provenance and mismatch validation |

Across all 200 steps of the strict path, both mismatch_count and max_abs_diff were 0.

### 200-Step Training Trajectory

Figure 1 places the train/rollout mismatch count and maximum absolute Δlogp on the same 200-step timeline. The RL-Kernel strict path stays at 0 mismatches throughout, while vime's native path has mismatches at every step.

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/rocm-bitwise-consistency.png" alt="Consistency comparison between native vime and vime plus RL-Kernel" style="display:block;margin:0 auto;width:6.5in;max-width:100%;" />

<p style="text-align:center;opacity:0.7;font-size:0.95em;"><em>Figure 1: Consistency comparison between native vime and vime + RL-Kernel (Qwen3-8B · TP4/CP2 · temperature 0.7 · top_p 0.95 · max response 6912 · ROCm MI300X).</em></p>

These signals appear together over the same period, consistent with the continued accumulation of train–rollout mismatch, providing end-to-end evidence for strict alignment. What it directly demonstrates is that vime + RL-Kernel can maintain both verifiable bitwise consistency and a more stable training trajectory over the full 200 steps.

Figure 2 shows the mean absolute train/rollout logprob difference over 200 steps on its own; vime + RL-Kernel stays at 0 throughout.

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/rocm-mean-abs-logprob-diff.png" alt="Mean absolute train-rollout logprob difference across 200 ROCm steps" style="display:block;margin:0 auto;width:6.5in;max-width:100%;" />

<p style="text-align:center;opacity:0.7;font-size:0.95em;"><em>Figure 2: Mean absolute train/rollout logprob difference over 200 steps (vime vs RL-Kernel + vime · Qwen3-8B · ROCm MI300X).</em></p>

Figure 3 compares the performance of native vime and vime + RL-Kernel over 200 steps.

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/rocm-performance-matrix.png" alt="VIME Qwen3-8B 200-step performance matrix" style="display:block;margin:0 auto;width:6.5in;max-width:100%;" />

<p style="text-align:center;opacity:0.7;font-size:0.95em;"><em>Figure 3: Aligned configuration · 1 node · 8× AMD Instinct MI300X 192GB · TP4/CP2 · global batch 8 · seed 1234.</em></p>

**Mean performance over 200 paired steps**

| Metric | Unit | vime | RL-Kernel + vime | RL-Kernel + vime vs vime | Finding |
|---|---|---:|---:|---:|---|
| Rollout time | s / step | 81.47 | 68.02 | +16.5% | RL-Kernel + vime faster |
| Rollout throughput | tok/GPU/s | 76.8 | 86.4 | +12.5% | RL-Kernel + vime higher |
| Reference logp | s / step | 2.60 | 6.36 | -144.4% | vime faster |
| Actor train | s / step | 8.95 | 14.91 | -66.6% | vime faster |
| Actor throughput | tok/s | 5,694 | 3,209 | -43.7% | vime higher |
| Total train | s / step | 14.78 | 27.66 | -87.2% | vime faster |
| End-to-end step | s / step | 99.24 | 99.04 | +0.2% | Parity |

**Quality and strict train/rollout consistency**

| Metric | Unit | vime | RL-Kernel + vime | RL-Kernel + vime vs vime | Finding |
|---|---|---:|---:|---:|---|
| Mean raw reward | score | 0.321875 | 0.385625 | +0.063750 | RL-Kernel + vime higher |
| Mean KL loss | loss | 0.007687 | 0.001091 | +85.8% | RL-Kernel + vime lower |
| Mismatch count | mean / step | 3,829.2 | 0.0 | exactly zero | RL-Kernel + vime PASS |
| Max \|Delta logp\| | max / 200 | 13.872876 | 0.0 | exactly zero | RL-Kernel + vime PASS |

*Arithmetic means over all 200 logged steps; positive percentages mean RL-Kernel + vime is better. temperature 0.7 · top_p 0.95 · max response 6912 · learning rate 5e-7 · KL coefficient 0.01*

---

## What vime × RL-Kernel Achieves on ROCm

- **Bitwise correctness:** Rollout and training logprobs are exactly identical on ROCm. Across 200 steps, mismatches stayed at 0 and the maximum logprob difference was 0.
- **Controlled end-to-end overhead:** The mean step time of RL-Kernel + vime is 99.04 s versus 99.24 s for native vime. The end-to-end overhead of the strict consistency path is about 0.2%.
- **Stable consistency guarantee:** Zero mismatches are maintained throughout 200 steps of end-to-end training, making results easier to verify and reproduce.
- **Complete ROCm execution evidence:** The kernels, HIP Graph, paged KV, collectives, and fallback paths actually executed are recorded, ensuring results come from the intended ROCm backend.
- **Fast fault localization:** Ablations pinpoint the specific operators and system boundaries involved in train–rollout inconsistency.

On 8 AMD Instinct MI300X GPUs, RL-Kernel + vime achieves zero mismatches across all 200 steps at roughly 0.2% end-to-end performance overhead. The results show that strict train–rollout consistency has progressed from correctness validation to a ROCm implementation with quantifiable performance overhead, traceable execution paths, and reproducible results, laying the groundwork for production deployment and further performance optimization.

---

## Current Scope and Next Steps

The current end-to-end validation covers Qwen3-8B Dense, vime, vLLM, Megatron-LM, and AMD Instinct MI300X. Next, we will extend to more MoE and multimodal models and AMD GPU architectures, and continue optimizing the performance of the ROCm strict path.

---

## Acknowledgments

This integration of vime, RL-Kernel, and AMD would not have been possible without the support of our hardware partners, open-source ecosystem collaborators, and development teams. We thank the vLLM community for its close collaboration with RL-Kernel, and especially Ao Shen, vime maintainer from Inferact, for his trust and support in the vime integration, community coordination, and ongoing maintenance.

We sincerely thank Liz Li and Yuhan Yang of AMD for providing AMD Instinct GPU compute resources, in-depth technical collaboration, and long-term support for RL-Kernel, which made the end-to-end train–rollout consistency validation of vime + RL-Kernel on ROCm possible. We also thank Lei Ding of Moore Threads for driving RL-Kernel's MUSA support, Yang Chen of Huawei for driving RL-Kernel's Ascend support, and Embedded LLM for its support of the project's development and community collaboration.

Core contributors to dense-model train–rollout consistency in RL-Kernel v0.1.0: Chutian Wang, Jiajie Li, Siru He, Xiaosong Ma, Kaijie Lin, Jian Zhang, Huihong Lu, Yunxiang Cai, Bosong Yang, Zhewei Liu, Houhong Liang, Ryan Huang, and Vensen Mu.

We also thank contributors whose PRs were merged into v0.1.0: Xiaopeng Du, Yuepeng Pan, Yiyang Fei, Ziying Tao, Zhifu Liu, Zhengtao Chen, Mengjie Li, Zien Liu, and GitHub users haoruilee, luoyueyuguang, hongleng, and smarslou.
