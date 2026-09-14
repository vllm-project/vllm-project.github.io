---
layout: post
title: "RL-Kernel × vime × AMD: Bitwise-Consistent Training and Rollout"
author: "RL-Kernel Team"
date: 2026-09-14
summary: "RL-Kernel and vime align selected-token logprobs bit for bit across training and rollout on CUDA and ROCm, with 200-step comparisons against native execution paths."
image: /assets/figures/2026-09-14-rl-kernel-v0-1-0/image35.png
social_image: /assets/figures/2026-09-14-rl-kernel-v0-1-0/image35.png
tags:
  - reinforcement-learning
  - post-training
  - performance
  - hardware
  - ecosystem
---

On-policy reinforcement learning assumes that the rollout engine and the training engine evaluate the same policy before any parameter update. In production systems, however, generation and training are usually handled by two separate engines. Identical models, weights, and inputs do not imply identical execution paths: kernels, batch shapes, parallel layouts, reduction orders, and intermediate precision can all change token probabilities, ultimately creating train–rollout mismatch.

vime and RL-Kernel address opposite ends of this path. vime manages the lifecycle of tokens, state, and weight versions, while RL-Kernel aligns the reduction and rounding boundaries in RMSNorm, Attention, GEMM, SwiGLU, linear logp, and distributed collectives. The former keeps both engines on the same training timeline; the latter makes both engines follow the same numerical execution contract.

## Introduction

Training and rollout optimize their execution engines for different goals. Rollout prioritizes throughput for sampling, prefill, decode, and KV-cache operations. Training must handle forward and backward passes, optimizer state, and multidimensional parallelism. The two systems implement the same mathematical model, but they do not necessarily execute the same floating-point program.

Once a discrepancy enters the importance ratio and clipped objective before the first parameter update, it behaves like a genuine policy shift. Reusing rollout logprobs can avoid part of the impact, but it cannot establish whether two engines independently produce the same result. Demonstrating bitwise consistency requires fixing the comparison target, numerical contract, and actual execution path at the same time.

We begin with the non-associativity of floating-point arithmetic and place Attention, logp, RMSNorm, GEMM, and collectives within a single nested-reduction framework. We then describe how vime and RL-Kernel divide responsibilities and how P/P, P/R, R/P, and R/R comparisons isolate the first divergence. Finally, we present separate 200-step results for ROCm and CUDA.

### Questions This Post Addresses

- Why can training and rollout produce different logprobs even when the model, weights, and inputs are identical?

- How can a comparability gate, a numerical execution contract, and P/P, P/R, R/P, and R/R comparisons identify the first divergence?

- Which parts of timeline synchronization and numerical alignment are handled by vime and RL-Kernel, respectively?

- How can we verify that zero mismatch actually comes from the intended backend?

- What do the 200-step ROCm and CUDA results establish?

## Eliminating Train–Rollout Mismatch

### Why the Same Model Is Not Necessarily the Same Computation

In our Qwen3-8B experiments, the strict R/R path integrating RL-Kernel with vime ran for 200 GRPO steps. At every step, the selected-token logprobs recomputed by the training engine and recorded by the rollout engine satisfied mismatch_count = 0 and max_abs_diff = 0. Aggregated using the validator's accounting method, the experiment covered 147,379,363 active tokens.

Here, bitwise consistency means elementwise equality with zero tolerance at runtime. It compares the probability of the same token, under the same weight version and within the same run, as computed independently by the two engines.

Why can the results differ when the model, weights, and inputs are identical? And why must seemingly separate components—RMSNorm, Attention, GEMM, logp, and communication—be handled together?

This post answers both questions from one perspective:

> Model equations define a mathematical result, but they do not uniquely specify an execution path. Finite precision breaks some equivalences between paths. Bitwise consistency therefore requires training and inference to follow the same numerical contract.

## A Spurious Policy Shift Before Parameter Updates

The rollout engine generates token aₜ from prefix hₜ and records

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image14.png"
style="width:1.25667in;height:0.25333in" />

Before training begins, the training engine scores the token again using the same weight version:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image21.png"
style="width:1.23333in;height:0.25in" />

If the policy has not yet been updated and both sides are evaluating the same logical object, the importance ratio should satisfy

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image27.png"
style="width:1.89in;height:0.36667in" />

Let δₜ = ℓₜᵀ − ℓₜᴿ. When δₜ is small, ρₜ ≈ 1 + δₜ. It also enters the PPO and GRPO clipped objective:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image28.png"
style="width:2.84667in;height:0.34in" />

Here, Âₜ is the advantage estimate. When δₜ exceeds log(1 + εhigh) or falls below log(1 − εlow), the mismatch may even change which clipping branch is selected. In effect, it creates another policy shift before any parameter update.

The total error can be decomposed further. Let sₜᴾ and sₜᴰ denote the probabilities assigned to the same token by serving prefill and an independent decode replay:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image5.png"
style="width:1.5in;height:0.45667in" />

The first term compares training scoring with serving prefill; the second compares prefill with decode; and the third checks weight version, cache state, and record identity. This decomposition matters. Although the final system exposes only one ratio, that ratio spans three interfaces: arithmetic across engines, two internal inference paths, and system state. If any one of them is not fixed, the aggregate discrepancy should not be loosely attributed to a kernel error.

The root cause can be reduced to one fact: floating-point addition is not associative. For example, in BF16 using round-to-nearest-even,

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image11.png"
style="width:1.64in;height:0.26333in" />

whereas

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image9.png"
style="width:2.01in;height:0.26333in" />

The two expressions differ only in their parenthesization over the reals, yet they produce different answers in BF16. Training is designed around packed sequences, backpropagation, and multi-GPU parallelism. Inference is designed around prefill, decode, dynamic batching, and the KV cache. Even with shared parameters, these systems may select different partitions, reduction orders, and intermediate precision because they optimize for different goals.

In other words, identical weights determine which values are added, but not the order in which they are added.

Two issues that are often conflated should also be separated. Reusing rollout logprobs determines which recorded values are used by the loss. Operator alignment tests whether two engines independently compute the same result. The former can bypass some consequences of mismatch, but it cannot establish the latter.

## Establishing Comparability: Fixing the Mathematical Object and Numerical Contract

Before discussing floating-point error, we must first determine whether the two engines are answering the same question. Write the ideal mathematical object as

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image24.png"
style="width:1.09333in;height:0.22667in" />

where x is the input, θ the weights, s state such as the KV cache, and ξ the random state involved in the computation. What training and rollout actually execute is

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image19.png"
style="width:3.42667in;height:0.26333in" />

where C is the numerical execution contract.

For selected-token logprobs, at least the following logical objects must match:

- checkpoint and weight version;

- prefix, target token, and active mask;

- position, RoPE, and causal/padding mask;

- logical K/V after KV-cache mapping;

- head, sequence, and vocabulary ownership;

- the true vocabulary range and any random state involved in the computation being compared.

If any of these differ, the comparison should be marked comparable = false. Physical page numbers or shard layouts may differ as long as they map back to the same logical tensor. When replaying fixed sampled tokens, there is also no need to reproduce the entire consumption history of the sampling RNG.

After this gate passes, we describe each computation node layer by layer. First, write

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image10.png"
style="width:0.91667in;height:0.23333in" />

where Dᵢ is the set of input elements on which output yᵢ actually depends. If the node can also be written as

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image1.png"
style="width:1.09333in;height:0.39in" />

then Rᵢ is its reduction domain. The two sides must first satisfy

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image3.png"
style="width:1.67in;height:0.25in" />

Next, partition the reduction domain:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image18.png"
style="width:3.82667in;height:0.28in" />

Πᵢ determines which partial summaries are produced first, while the cross-device reduction tree determines how values are combined within and across partitions. Even when Rᵢ is identical, a different cross-device reduction tree may produce different bits.

We then record the node's precision tuple

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image22.png"
style="width:2.82in;height:0.23667in" />

and represent each rounding operation as

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image26.png"
style="width:0.78333in;height:0.23667in" />

Collecting the quantities that can independently change a result, the minimal arithmetic contract can be written as

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image15.png"
style="width:1.87333in;height:0.23in" />

Here, Dᵥ summarizes the dependency sets and reduction domains for the node's outputs; Πᵥ and Tᵥ record the reduction partition and ordered merge tree; Pᵥ is the precision tuple; Rᵥ records where Qₚ occurs; and Aᵥ records the exact numerical primitives used for exp, log, rsqrt, SiLU, and related operations.

Fusion, materialization, and recomputation boundaries are not listed separately in this arithmetic contract. They change the numerical result only when they alter Tᵥ, Pᵥ, Rᵥ, or Aᵥ, so they are better recorded as execution mechanisms. State and control are also kept out of the tuple: state such as the cache and RNG is checked by the comparability gate, while control conditions such as dispatch and CUDA Graph are trigger axes. This separation avoids describing the same cause at multiple levels.

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image29.png"
style="width:1.44in;height:0.25333in" />

The expression above denotes the set difference between the training and inference arithmetic contracts. That difference generates candidate root causes for mismatch.

## 3. Nested Reductions in Transformers

RMSNorm, Attention, GEMM, linear logp, and collectives all perform reductions of the following form:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image25.png"
style="width:1.18in;height:0.38in" />

The reduction domain R is partitioned, local Agg(R⁽ʲ⁾) values are computed, and the partial results are merged. Over the reals, different partitions and parenthesizations are normally treated as equivalent. Under finite-precision execution, they are not.

| **Module**               | **Reduction domain** | **Reduction semantics**                              |
|--------------------------|----------------------|------------------------------------------------------|
| RMSNorm                  | Hidden dimension     | Sum of squares that determines the vector scale      |
| GEMM                     | K dimension          | Sum of products that determines one output element   |
| Attention                | Visible keys         | Max, sum-exp, and weighted values                     |
| Linear logp              | Vocabulary           | Max, sum-exp, and the target logit                    |
| AllReduce, ReduceScatter | Ranks                | Partial contributions produced by individual devices |

From this perspective, Split-K, Split-KV, vocabulary sharding, context parallelism, and rank trees simply partition different mathematical axes.

### 3.1 The Shared Normalization Structure of Attention and Logprob

Given scores sᵢ, define

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image6.png"
style="width:1.96667in;height:0.37333in" />

Here, m is fixed to the maximum over the domain, while l records the exponential sum relative to m. For the final real-valued result, the two can be combined into a single log-sum-exp (LSE). In an actual kernel, however, they are updated and merged separately, so the bitwise contract must preserve both intermediate states.

Linear logp only needs to retain (m, l) and the target logit:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image16.png"
style="width:1.76in;height:0.26in" />.

Attention carries one additional vector:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image30.png"
style="width:2.57667in;height:0.28667in" />.

The two operations therefore perform the same kind of LSE aggregation in different spaces. Attention normalizes over the context space to determine which tokens to attend to; logp normalizes over the vocabulary space to determine which token to select. Attention's Split-KV merge and logp's cross-TP vocabulary merge are two instances of the same mathematical problem.

To merge two blocks (m₁, l₁, o₁) and (m₂, l₂, o₂), first set m = max(m₁, m₂), then compute

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image8.png"
style="width:3.23333in;height:0.21333in" />.

Over the reals, this merge operation is associative, so the same global result can be recovered from any partition. For an arbitrary number of blocks, the merged result can be written directly as

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image4.png"
style="width:3.44333in;height:0.39333in" />

The right-hand side depends only on the set of all blocks, providing a short proof of associativity.

Actual computation, however, uses the rounded and approximated merge ⊕̂. In general,

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image17.png"
style="width:1.82667in;height:0.25in" />

where σⱼ = (mⱼ, lⱼ, oⱼ). The partition Πᵢ, reduction tree Tᵢ, exponential primitive Aᵥ, and precision Pᵥ of m, l, and o are therefore part of the normalization itself.

### 3.2 Reduction Order in RMSNorm, GEMM, and Communication

RMSNorm computes Σᵢ xᵢ²; GEMM computes Σₖ aᵢₖbₖⱼ; and after a row-parallel GEMM, AllReduce continues summing the local contributions produced by each rank.

Suppose the K dimension is divided among ranks into disjoint sets K₀, …, Kₚ₋₁:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image2.png"
style="width:2.29667in;height:0.75in" />

The inner Yᵢⱼ⁽ʳ⁾ is produced by a local GEMM, while the outer ΣᵣYᵢⱼ⁽ʳ⁾ is completed by AllReduce. Mathematically, both are parts of one summation; operationally, the kernel boundary splits that sum into two levels of reduction. The collective is therefore the portion of the same overall reduction tree that extends beyond one GPU.

RMSNorm and softmax share another structural property. Both compress a domain into a small set of global statistics and then broadcast those statistics back to each local output. For RMSNorm,

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image13.png"
style="width:2.68667in;height:0.53in" />

Softmax applies the same (m, l) pair to every score. A one-bit difference in the reduction can therefore propagate through the shared scale and couple an entire hidden vector, Attention row, or vocabulary distribution. The three operations have different semantics, but the same numerical structure.

This also shows why fixing only one layer is insufficient. Disabling Split-K in GEMM removes one class of partial merge inside the kernel. If TP AllReduce still uses a different rank tree, the full parenthesization of the sum remains different. Conversely, fixing the rank tree cannot replace the warp/CTA reduction contract inside the kernel.

### 3.3 Fusion and Rounding Boundaries

Suppose one side executes

GEMM -> write BF16 -> SiLU -> multiply

while the other executes

GEMM -> retain FP32 accumulator -> SiLU -> multiply -> write BF16

Both are SwiGLU on paper, but they already differ in whether the intermediate tensor is materialized. Fusion, recomputation, and communication staging can affect the result because they can move Rᵥ—the points at which rounding occurs in the contract.

Likewise, two implementations both labeled FP32 may still produce different bits because they use different exp, log, rsqrt, SiLU, FMA, or fast-math primitives. A dtype describes the storage container, not the complete computation.

### 3.4 From Continuous Numerical Error to Discrete Path Divergence

A tiny difference in a score can change sampling, argmax, top-k, or a threshold decision, after which the two trajectories are no longer comparable. The relevant discrete boundaries are the mask, position, cache lookup, selected token, and vocabulary ownership.

This is why fixed replay matters: freeze the token first, then separate "a different token was sampled" from "the same token was assigned a different probability."

## 4. From Timeline Synchronization to Algebraic Alignment: vime and RL-Kernel

With the formalism above, the relationship between vime and RL-Kernel becomes clear.

One training timeline:

prompt -> vLLM rollout -> token / rollout logp -> Megatron scoring -> backward -> update

vime determines whether both sides are at the same point in time: which token batch, weight version, and rollout record enter each update. RL-Kernel determines whether both sides use the same algebraic implementation at that point: how contributions are partitioned, merged, and rounded. In short, vime aligns causal time, while RL-Kernel aligns numerical algebra.

Without the former, even a fully deterministic kernel may compare different weight versions. Without the latter, the same weights may still be interpreted by two different implementations. Together, they constrain the state degrees of freedom in F and the implementation degrees of freedom in F̂C.

The strict Qwen3/H100 path applies this idea at five points:

| **Module**      | **What the current path fixes**                                                                                                               | **Object**                 |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| RMSNorm         | Shared primitive, hidden width, epsilon, residual-add boundary, and output boundary                                                            | Hidden dimension           |
| Attention       | Position/mask/GQA/cache identity; common FA4 arithmetic identity; num_splits=1; FP32 LSE; final-write downcast                                  | (m, l, o) over keys        |
| GEMM / SwiGLU   | No Split-K; BF16 operands and FP32 accumulation; aligned epilogue and activation-materialization boundaries                                    | Product sum over K         |
| Linear logp     | True vocabulary size of 151,936; 128 padded lanes masked; fixed local tree and cross-rank LSE merge; FP32 logp                                 | (m, l) over the vocabulary |
| Collectives     | Logical ownership, payload dtype, and explicit rank tree                                                                                        | Per-device partial summary |

num_splits=1 and no Split-K are currently the easiest choices to audit. Other choices can form an equally valid contract as long as both sides completely fix the partition, partial state, and merge tree. Consistency requires only that these choices do not silently change the observable numerical semantics.

### Backward as a Separate Execution Graph

Rollout has no backward pass, so forward train–rollout parity cannot imply cross-engine backward parity. Backpropagation also introduces new reduction axes. For example,

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image7.png"
style="width:1.17in;height:0.28667in" />ᵀ

The forward GEMM reduces over the hidden/K dimension, while this operation reduces over the token dimension. Microbatch partitioning, gradient accumulation order, saved values used for recomputation, atomics, and gradient collectives can all change the parenthesization again.

Written as a vector–Jacobian product (VJP),

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image12.png"
style="width:1.50667in;height:0.25in" />.

The precise claim is therefore that the evidence in this post validates cross-engine forward logprobs. Reproducibility of the training backward pass requires treating the VJP as a separate computation graph and checking its domain, partition, tree, saved precision, and communication. The deterministic_backward=true setting is one part of that contract.

## 5. Key Trigger Axes for Train–Rollout Mismatch

Batch size, sequence length, prefill/decode mode, workspace, CUDA Graph, GPU model, and topology often vary together with mismatch, but they are usually trigger conditions rather than root causes.

Trigger condition → path selection → ΔCᵥ → first numerical divergence

For example:

batch size changes

-> cuBLASLt heuristic changes

-> Split-K count changes

-> K-dimension partial-merge tree changes

-> GEMM output bits change

Saying that batch size caused the mismatch describes only a correlation. Identifying that batch size selected a different Split-K reduction tree is much closer to the root cause.

### Single-Variable Ablation: Change One Trigger Axis at a Time

Let C be a fully aligned baseline contract, and change only its k-th field:

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image23.png"
style="width:2.90333in;height:0.26in" />.

First identify the earliest nonzero Δₖ, then trace how it propagates through the computation graph into the hidden states, logits, selected-token logprobs, and ρₜ. This is more informative than observing a difference in the final logp and then suspecting modules one by one.

Operator replacements on the training and rollout sides form a matrix, where P denotes the production implementation and R denotes RL-Kernel:

| **Combination** | **Training** | **Rollout** | **Question answered**                                                      |
|-----------------|--------------|-------------|----------------------------------------------------------------------------|
| R/R             | R            | R           | Fully aligned control baseline                                             |
| P/R             | P            | R           | Does replacing only the training side introduce divergence?                |
| R/P             | R            | P           | Does replacing only the rollout side introduce divergence?                 |
| P/P             | P            | P           | Native-path behavior; a diagonal difference alone cannot identify one side |

When isolating Attention, FFN and logp must remain R/R; the same principle applies when isolating FFN. Any silent fallback on an R side should fail the experiment. Otherwise, a nominally aligned result may never have executed the aligned implementation.

Here, R/R, P/R, R/P, and P/P describe the operator-implementation matrix across the two engines. The G00–G11 labels used below encode a separate system-level matrix: whether rollout logprobs are reused × whether aligned operators are enabled.

### Strict Bitwise Consistency Requires Execution Provenance

A credible zero-mismatch result requires at least five layers of evidence:

1. The comparability gate establishes that the compared objects, state, and ownership are identical.

2. Fixed-input tests establish that the same path is repeatable.

3. Training and rollout run cross-engine parity checks on the same operator inputs.

4. Selected-token logprobs are compared online under the same weight version.

5. The complete workflow archives per-step results and records the backend, device, fallback status, and CUDA/HIP Graph route.

A configuration file that enables RL-Kernel does not by itself prove that RL-Kernel executed at runtime. Likewise, the presence of NCCL in an application log does not establish that the target payload used the fixed tree. Numerical results answer what was computed; execution provenance answers which path computed it. Both forms of evidence are necessary.

## Qwen3-8B Results and Validation Scope

The CUDA results below are accompanied by the corresponding experimental configuration, 200-step records, aggregate tables, bootstrap statistics, validation artifacts, and plotting scripts.

### Experimental Configuration

| **Item**                | **Configuration**                                      |
|-------------------------|--------------------------------------------------------|
| Model / dtype           | Qwen3-8B / BF16                                        |
| Hardware                | 1 node, 8× NVIDIA H100 80GB                            |
| Megatron                | TP4 / CP2 / PP1 using 8 GPUs                           |
| Rollout                 | 2 vLLM engines, each using TP4                         |
| Placement               | Actor and rollout colocated                            |
| Horizon                 | 200 rollout/training steps                             |
| Seeds                   | Training 1234, rollout 1234                            |
| Sampling                | 8 prompts × 16 samples per step; global batch 128      |
| Response limit          | 7,168 tokens                                           |
| Dynamic batching        | Maximum 4,096 tokens/GPU                               |
| vLLM memory utilization | 0.4                                                    |
| CUDA Graph              | FULL_DECODE_ONLY, preserving the production graph path |
| KL loss                 | Enabled, coefficient 0.001                             |
| Snapshot requirement    | Exactly 8 rank files must be present for every step    |

### Bitwise Consistency

| **Experiment** | **Path**       | **Active-token comparisons** | **Mismatches** | **Agreement** |
|----------------|----------------|------------------------------|----------------|---------------|
| G10            | Production P/P | 140,601,694                  | 58,230,217     | 58.58%        |
| Optimized G11  | Strict R/R     | 147,379,363                  | 0              | 100%          |

For G11, both runtime mismatch_count and max_abs_diff are zero at every step. G10 has a nonzero mismatch at every step, with a maximum per-step max_abs_diff of 1.591547. The aggregate counts are reconstructed according to the runtime validator's accounting method by multiplying each step's mean sample count in rounds.csv by the global batch size of 128 and then summing across steps.

Both runs use the same workload configuration, but their sampled trajectories differ, so their active-token counts are not identical.

**Figure: G10 vs. Optimized G11 — Training and Bitwise Consistency (CUDA)**

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image35.png"
style="width:6.5in;height:4.16667in" />

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image32.png"
style="width:6.5in;height:3.31944in" />

## Progress on Bitwise Alignment for ROCm

We completed a 200-step strict R/R validation on ROCm in a full training-and-rollout system combining Megatron training with vLLM rollout, with zero mismatch throughout. This is an end-to-end validation of the cumulative integration stack, not an isolated test of any single pull request. The ROCm path uses the same correctness boundary while preserving native execution paths for AITER, CK, paged KV, HIP Graph, and ROCm collectives. Runtime readback verifies the actual backend, execution path, and fallback state.

### Experimental Configuration

| **Item**                | **Configuration**                                                                                                             |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Model / dtype           | Qwen3-8B / BF16                                                                                                               |
| Hardware                | 1 node, 8× AMD Instinct MI300X 192GB                                                                                          |
| Megatron                | TP4 / CP2 / PP1 using 8 GPUs                                                                                                  |
| Rollout                 | 2 vLLM engines, each using TP4                                                                                                |
| Placement               | Actor and rollout colocated                                                                                                   |
| Horizon                 | 200 rollout/training steps                                                                                                    |
| Seeds                   | Training 1234, rollout 1234                                                                                                   |
| Sampling                | 1 prompt × 8 samples per step; global batch 8                                                                                 |
| Response limit          | 7,168 tokens                                                                                                                  |
| Dynamic batching        | Maximum 4,096 tokens/GPU                                                                                                      |
| vLLM memory utilization | 0.38                                                                                                                          |
| HIP Graph               | FULL_AND_PIECEWISE, preserving the production graph path                                                                      |
| KL loss                 | Enabled, coefficient 0.001                                                                                                    |
| Validation requirement  | Frozen inputs and sources must match before and after the run; every step must pass runtime-provenance and mismatch validation |

### Bitwise Consistency

| **Experiment** | **Path**       | **Active-token comparisons** | **Mismatches** | **Agreement** |
|----------------|----------------|------------------------------|----------------|---------------|
| G10            | Production P/P | 10,912,549                   | 8,662,719      | 20.62%        |
| Optimized G11  | Strict R/R     | 9,400,614                    | 0              | 100%          |

Across all 200 steps of Optimized G11, runtime mismatch_count and max_abs_diff are both zero. The strict validator confirms torch.equal == true for 9,400,614 active-token comparisons from 1,600 samples. G10 also completes all 200 steps, but every step contains a nonzero mismatch: 8,662,719 of 10,912,549 active-token comparisons disagree, for an agreement rate of 20.62%. Under the per-step accounting in rounds.csv, G10 reaches a maximum per-step max_abs_diff of 17.970963.

G10 does not pass the strict validator because the native P/P path does not generate RL-Kernel operator readbacks; the 200-step training run itself completes. Ray submissions for both experiments successfully record 200/200 rollout and training steps.

Both runs use the same frozen workload and source configuration, but their sampled trajectories differ, so their active-token comparison counts are not identical. The aggregate counts are reconstructed according to the validator's accounting method by multiplying each step's mean sample count in rounds.csv by the global batch size of 8 and then summing across steps. The frozen-input and frozen-source audits remain identical before and after both runs. No missing-value imputation or row deletion is performed during aggregation.

The figure below shows G10's reward collapsing after step 75 because its generated responses exceed the 7,168-token generation limit. The native vime run also exhibits more aggressive loss and gradient-norm behavior than RL-Kernel. This training instability is caused by train–rollout mismatch and demonstrates why full alignment matters.

**Figure: G10 vs. Optimized G11 — Training and Bitwise Consistency (ROCm)**

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image34.png"
style="width:6.5in;height:4.08333in" /><img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image31.png"
style="width:3.4208in;height:1.78969in" /><img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image20.png"
style="width:3.40417in;height:1.81556in" />

<img src="/assets/figures/2026-09-14-rl-kernel-v0-1-0/image33.png"
style="width:6.5in;height:3.25in" />

## Connecting the Entire Path

For readers who are less familiar with distributed kernels, the modules above may appear to span a wide range of concerns. Rearranged into one sequence, the logic is compact:

1. The same model equation defines only a real-valued function F; it does not uniquely specify the actual implementation F̂C under a numerical contract.

2. Before any parameter update, a difference between training and rollout logprobs creates a spurious policy ratio.

3. Before measuring error, apply the comparability gate to ensure that the token, weights, position, mask, cache, and ownership describe the same object.

4. View the Transformer as a set of nested reductions: RMSNorm reduces over hidden dimensions, GEMM over features, Attention over keys, logp over the vocabulary, and collectives over ranks.

5. Ask the same questions for every reduction: which values participate, how they are partitioned, which tree merges them, where rounding occurs, and which primitives are used.

6. Attention and logp both contain LSE operations, while collectives perform cross-device reductions. These modules must therefore form one end-to-end contract.

7. vime fixes the temporal relationship among tokens, state, and weight versions; RL-Kernel fixes the numerical algebra of critical operators. Together, they make "the same policy" an executable condition.

8. Batch size, sequence length, and CUDA Graph are only trigger axes. Diagnosis must continue until it reaches a specific partition, merge tree, or rounding boundary.

9. Finally, use single-variable replacements to locate the first divergence and archive both outputs and execution provenance before calling a result bitwise consistent.

From this perspective, train–rollout mismatch is an abstraction leak: the upper layer treats mathematical equivalence as numerical equivalence. RL-Kernel's value is not limited to providing deterministic operators; it elevates the numerical contract into a cross-engine interface. More broadly, the numerical contract defines an observable boundary for optimization. Training and inference may still use different memory layouts, parallel strategies, and scheduling policies. As long as those changes do not alter dependency domains or rounding boundaries, they remain part of the same verifiable implementation. What must be eliminated is undeclared arithmetic divergence.

## Next Steps

- Extend support to more models and multimodal architectures.

- Continue adapting RL-Kernel to MUSA, Ascend, and additional hardware platforms.

- Advance integrations with Miles and AReaL.

Models, hardware, and execution frameworks for RL post-training will continue to evolve. RL-Kernel aims to preserve the correctness boundary inside the system so that every kernel replacement, framework upgrade, or hardware migration can answer two questions explicitly: whether the numerical semantics were preserved and where any divergence began.

## Acknowledgements

The release of RL-Kernel v0.1.0 would not have been possible without the support of our hardware partners, open-source ecosystem collaborators, and core development team.

#### Hardware and Compute Partners

We sincerely thank Liz Li and Yuhan Yang from AMD for providing AMD Instinct GPU compute resources, deep technical collaboration, and long-term support for RL-Kernel. We look forward to continuing cross-platform consistency validation, kernel-level performance optimization, and large-scale RL workload deployment on ROCm.

We also thank Lei Ding from Moore Threads for advancing MUSA support and Yang Chen from Huawei for advancing Ascend support. We thank Embedded LLM for supporting project development and community collaboration.

Consistent execution and generalization across heterogeneous hardware platforms are long-term priorities for RL-Kernel. We welcome collaboration with additional hardware vendors and open-source communities to build open and efficient RL-Kernel infrastructure.

#### Open-Source Ecosystem and Framework Collaboration

We thank the vLLM community for its close collaboration with RL-Kernel. We especially thank Ao Shen, vime maintainer at Inferact, for the trust and support provided throughout the RL-Kernel and vime integration, community coordination, and ongoing maintenance. This work builds on the open-source ecosystem formed by vLLM Rollout, vime orchestration, and Megatron training.

#### Core Contributors — v0.1.0

We especially thank the RL-Kernel core contributors for their work on the v0.1.0 architecture, kernel implementation, operator-level train–rollout consistency for dense models, distributed validation, and community development.

Chutian Wang:

- Developed cross-node and inter-GPU communication modules for CUDA.

- Built the cross-platform ablation-matrix infrastructure.

Jiajie Li:

- Led the evaluation of the vime framework, roadmap planning for the fork, and PR delivery.

- Led the integration of vime and RL-Kernel on both CUDA and ROCm.

- Led Distributed Attention development.

- Completed the linear_logp replacement experiments, including TP parallelization and technical blog writing.

- Led end-to-end training-and-rollout testing and performance tuning of native vime and RL-Kernel + vime on both CUDA and ROCm.

Siru He:

- Led development of the WS1 GTest unit-testing framework.

- Led distributed adaptation and PR delivery for GEMM operators on CUDA and ROCm.

Xiaosong Ma:

- Developed the WS1 Attention operator.

- Led end-to-end development and testing of individual WS1 operators and construction of the GTest framework.

- Optimized GEMM performance for end-to-end training-and-rollout tests on CUDA.

- Delivered the communication PR for ROCm.

- Participated in reviews and assisted with end-to-end training-and-rollout testing and performance tuning of native vime and RL-Kernel + vime on ROCm.

Kaijie Lin:

- Led development of the standalone and distributed Logprob operator PRs.

- Implemented a deterministic fused Linear-Logp operator in Triton for ROCm.

- Implemented Logprob TP parallelization for the vime integration experiments.

Jian Zhang:

- Developed the RoPE operator.

- Contributed to Distributed Attention development.

- Led operator adaptation and PR delivery for Ascend.

Huihong Lu:

- Developed the Triton Logprob operator, including ROCm support.

- Contributed to distributed Logprob adaptation.

Yunxiang Cai:

- Developed the standalone RMSNorm operator.

- Implemented the standalone Attention operator for the Triton path.

Vensen Mu:

- Developed the standalone GEMM operator and optimized its performance in CUDA training-and-rollout tests.

- Led deep adaptation for ROCm.

- Delivered the communication PR for ROCm.

- Led end-to-end training-and-rollout benchmarking and performance tuning of native vime and RL-Kernel + vime on ROCm.

Bosong Yang:

- Contributed to Distributed Attention development.

Zhewei Liu:

- Contributed to Distributed Attention development.

Houhong Liang:

- Profiled and tuned the end-to-end training-and-rollout path on CUDA.

Ryan Huang:

- Contributed to development of the standalone and distributed Logprob operator PRs.

Finally, we thank the community contributors Xiaopeng Du, Yuepeng Pan, Yiyang Fei, Ziying Tao, Zhifu Liu, Zhengtao Chen, Mengjie Li, Zien Liu, and GitHub users haoruilee, luoyueyuguang, hongleng, and smarslou.
