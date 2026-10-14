---
layout: post
title: "Finish RL Training Faster: Bitwise-Consistent, High-Performance, Agent-Driven"
author: "Bai Yan (NVIDIA), Ao Shen (Inferact)"
date: 2026-10-14 09:00:00 +0000
summary: "We make vLLM rollout and the training forward give bitwise identical logprobs on DeepSeek-V4-Flash and Nemotron-H NVFP4, keep 91% of rollout throughput on DeepSeek-V4-Flash, and open-source an agent skill for other models."
image: /assets/figures/2026-10-14-train-inference-consistency/ds4-reward.png
social_image: /assets/figures/2026-10-14-train-inference-consistency/ds4-reward.png
tags:
  - rl
  - batch-invariance
  - deepseek
---

Rollout dominates RL training time, so training systems adopt low-precision compute, low-precision KV cache, and communication and kernel optimizations. These change the inference-side numerics. The inference engine generates a response; the training engine recomputes its probabilities and updates the model. With identical weights, the two sides now assign different probabilities to the same token, so training optimizes a policy other than the one that sampled. The gap can grow with updates and destabilize training. In our DeepSeek-V4-Flash-Base experiment (no algorithmic correction such as TIS), the high-performance baseline's train-inference KL rose from 2e-3 to 29.9 while reward fell back; the consistent version had a difference of 0 at every step and no such drop.

Stable training means fewer crashes, rollbacks and reruns, and less total time to reach the same result. Our goal is to **finish effective RL training faster**: faster end to end, not per step. This post measures numerical consistency, training curves and rollout throughput; the end-to-end wall-clock comparison is left for later. The goal needs three properties at once:

- **A, high performance**: fast rollout and training, with low-precision compute, low-precision KV cache and communication optimizations all usable;
- **B, high consistency**: training and inference compute the same policy for the same weights;
- **C, low cost and generality**: getting A and B on a new model, precision or hardware costs acceptable effort.

These rarely hold together. Drop B and you optimize freely but risk unstable training. Drop A and you can use consistent kernels, but on the workload reported by [Thinking Machines Lab](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference) time went from 26 to 55 seconds, or 42 seconds with an improved attention kernel. Keeping A and B means aligning model by model and operator by operator, which used to need senior engineers, so C fails.

We make B strict on the high-performance numerical path: with the same weights and input, training and inference produce a **bitwise identical** logprob for every sampled token (diff = 0; Section 2 defines it exactly). This is true on-policy: training optimizes the numerical policy that actually sampled. It holds on our accepted tasks (math reasoning, multi-turn tool calling) and parallel layouts, including different layouts per side. Agent + skill then cuts the cost of building this path. The post covers:

1. How existing approaches trade off the three, and why we chose this one (Section 1);
2. What train-inference consistency requires (Section 2);
3. **Reference recipe**: DeepSeek-V4-Flash-Base combines low-precision compute (block-FP8), sparse attention and multi-node expert parallelism. We reach 0 diff on it with whole-run average rollout throughput at 91.3% of the high-performance baseline (Section 3);
4. **Module-by-module deep dive**: per module and kernel, what each side does, the trade-offs, and who pays (Section 4);
5. Encoding the method as an agent skill to lower C (Section 5);
6. **Transfer**: using agent + skill to support a different kind of model, Nemotron, a Mamba hybrid with NVFP4. The agent did most reproduction, experiments and hypothesis elimination; humans made the key judgments (Section 6);
7. Open-sourcing the skill and code (Section 7).

The second case tests whether the recipe and skill transfer; aligning the Mamba recurrence and routing drift under multiple updates per step were both new.

## 1. Existing approaches: how they trade off the three

Train-inference mismatch is not new; existing approaches trade off A, B and C differently:

| Approach | What it keeps | What it gives up or pays |
|---|---|---|
| **Algorithmic correction**: TIS, MIS, FlashREINFORCE, Score Centering | A and C: leaves the inference engine alone, model-agnostic, cheap to adopt | B: the numerical difference stays; only its effect on the update is limited. Costs: bias, discarded samples, and correction gets harder as low precision widens the gap |
| **R3** (replay rollout expert routing) | A, plus B for MoE routing | Removes only routing flips; other numerical differences remain |
| **QAT** (training forward simulates deployment quantization) | A: rollout can safely use low precision; shrinks the quantization-induced part of the difference (improves B) | Training must simulate quantization, and backward uses the straight-through estimator (STE: backward treats quantization as identity); simulating weights only leaves differences from activations, KV, kernels and reduction |
| **Strict numerical alignment** (Thinking Machines, slime, IsoExec, this post) | B | A or C: generic batch-invariant kernels cost A (Thinking Machines workload: 26 s to 55 s, 42 s with an improved attention kernel); aligning on the high-performance path keeps A and costs C (slime, IsoExec, this post: per-model, per-operator alignment and verification) |

Algorithmic correction covers several mechanisms:

- **TIS** (Truncated Importance Sampling, [Yao et al.](https://fengyao.notion.site/off-policy-rl)): weights the gradient by the training/rollout probability ratio, truncated at an upper bound, trading a little bias for bounded variance;
- **MIS** (Masked Importance Sampling, [When Speed Kills Stability](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda)): no truncation; drops whole sequences whose sequence-level probability ratio exceeds a threshold;
- **FlashREINFORCE** ([Hu, Zhang et al.](https://github.com/yifanzhang-pro/FlashREINFORCE)): for asynchronous agentic RL; trajectory-level drift filtering (sequence trust region) removes trajectories that deviate too far;
- **Score Centering** ([Marek & Ryabinin](https://arxiv.org/abs/2609.20807)): cancels train-inference drift accumulated with each update via an additive correction term; stacks with importance sampling.

R3 (Rollout Routing Replay, [Ma et al.](https://arxiv.org/abs/2510.11370)) targets MoE: it records expert routing during rollout and replays it in training, removing the difference caused by routing flips.

Several teams work on the last row, paying with either A or C. In [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference), Thinking Machines Lab traces inference nondeterminism mainly to batch size varying with load and achieves true on-policy RL with batch-invariant kernels, at the inference-time cost in the table (the post also notes that much of the overhead comes from a FlexAttention integration that is not yet well optimized).

[slime](https://github.com/THUDM/slime) aligns training and rollout logprobs on the GLM-5 architecture (MLA + DSA sparse attention, same structure as DeepSeek-V3.2).

In the vLLM community, [IsoExec](https://vllm.ai/blog/2026-08-21-isoexec) by Alexander Jiang and the SkyRL team does the same for linear attention (recurrent structures such as Qwen3.5-35B-A3B's Gated DeltaNet) (details in Section 6).

These approaches target different differences, and most stack. Two kinds must be distinguished: the numerical difference between engines on the same weight version, and the policy difference between weight versions.

With staleness, as in asynchronous training, rollout's old weights lag the trained policy. That algorithm-level off-policy still goes to TIS, MIS or Score Centering, yet the old weights can still be numerically aligned across engines. Removing engine-induced differences leaves only the algorithm's own off-policy, free of implementation noise.

QAT is also part of strict alignment: for low-precision rollout and training to match bitwise, the training forward must apply the same quantization; our backward uses STE.

We take the last-row path that keeps A, with this bar for A: **rollout throughput at least 70% of an equally optimized high-performance baseline** (rollout is the most time-consuming part of RL training; training-side overhead was not measured separately, see 3.3). C is this path's weak point; Section 5 covers how Agent + skill makes up for it.

## 2. What train-inference consistency requires

"Training and inference compute the same thing" breaks into four dimensions plus an aligned numerical contract:

```
train-inference consistency ⇐ operator determinism + batch independence + phase consistency + cross-GPU consistency + aligned numerical contract on both sides
```

**Operator determinism (single GPU).** The same operator on the same input always gives the same output. GPUs do not guarantee this by default. Split-K matrix multiplication spreads one multiplication across compute units that accumulate in parallel; if the merge uses atomic adds, accumulation order follows scheduling, and since floating-point addition is not associative, the result varies (split-K can use a fixed-order two-stage reduction; the default implementation does not). Thinking Machines separates "repeated runs on the same input agree" (this dimension) from "the result does not change with batch" (the next).

**Batch independence.** A request's result must not depend on its batch-mates. Inference is worst: the engine merges and splits batches dynamically for throughput. In training, micro-batch size, packing and per-expert MoE token counts also change the path, but more controllably. Many optimizations (kernel choice by batch size, dynamic split count, CUDA Graph on or off) make the path depend on "neighbors".

**Phase consistency.** A token must get the same result in prefill (many tokens at once) and decode (one at a time). Batch independence is "horizontal"; phase consistency is "vertical", one request across phases. Inference engines give each phase its fastest kernel; forcing a shared path is a main source of the consistent version's slowdown and a focus of optimization.

**Cross-GPU consistency.** Cross-GPU reductions (AllReduce, reduce-scatter, and merging expert outputs after the MoE all-to-all) must also be fixed, at two levels.

- With the **same parallel layout** on both sides, communication results must not vary with timing and scheduling (fix the communication library's channel count, thread count and algorithm, plus reduction chunking and precision and expert merge order; this is not cheap, see 4.5).
- With **different parallel layouts** (e.g. different TP/EP degrees), results must also stay the same as the degree of parallelism changes, which needs a fixed reduction tree and costs more. The DeepSeek-V4 recipe instead moves cross-GPU reduction out of the compared path (4.5).

The four dimensions do not suffice: two engines can each be deterministic, batch-invariant, phase-consistent and cross-GPU consistent yet take different numerical paths and disagree. So we also require **aligned numerical contracts on both sides**: same quantization and rounding, same accumulation precision, same kernel arithmetic.

Every mismatch has one of two root causes.

- **Different order of operations**: both sides take the same path, but parallel accumulation or cross-GPU reduction order varies with scheduling, and non-associative floating-point addition changes the result.
- **Different computation path**: the sides run different algorithms. Statically, the sides use separate kernels; dynamically, batch shape picks the kernel or split count, or KV cache / prefix cache state picks the branch.

Discrete choices such as Top-K and MoE routing are an amplifier, not a third root cause: a one-ulp upstream difference (one unit in the last place of a floating-point number) becomes a different token set or expert combination.

To reach 0 diff: remove static path and order-of-operations differences, then pin dynamic path selection so it ignores load. With upstream aligned, discrete selection must also match: same tie-breaking, output ordering and merging rules on both sides.

Here diff is, before each update with the same weight version and token prefix, the difference between rollout's logprob for each valid token (sampled tokens inside the response mask) and the training side's recomputed old-logprob. Acceptance requires the max, mean and standard deviation of the corresponding-probability difference, and k3 KL, all to be exactly 0. This is exact equality, not a threshold, checked before each update; a mini-batch update changes the policy anyway, which the definition excludes.

Hence: **a train-inference consistent recipe does only 1 update per step (step batch equals mini-batch).** With multiple mini-batch updates per step, from the second on, training weights differ from sampling weights; diff = 0 does not cover this, and training stops being strictly true on-policy (example in 6.2).

Two easily missed points. First, the four dimensions guarantee only a consistent forward; training also needs a correct backward, and the first step after checkpoint restore must still be consistent. Second, **diff = 0 says the two sides compute the same thing, not that it suits training**: with multiple updates per step, diff = 0 holds while routing drifts between updates (6.2). diff = 0 is our engineering target, not a sufficient condition for training quality, which needs separate acceptance.
## 3. Reference recipe: DeepSeek-V4-Flash

DeepSeek-V4-Flash combines low-precision compute, sparse attention and multi-node expert parallelism; it is one of the hardest consistency tasks we have done. From the FP8 base we trained two 100-step runs on GB200: an optimized high-performance baseline and a consistent version with diff = 0.

Below, **BI=0 means the baseline version and BI=1 means the consistent version** (BI is batch invariance, Section 2's "batch independence", borrowed as a label).

Task: math reasoning with DAPO-format data, GRPO, and a reward with an over-length penalty (so it can be negative). Each run uses 8 machines (32 GPUs), 128 prompts × 8 responses per step, and 1 update per step (step batch = mini-batch). Neither run uses TIS, MIS or similar corrections, exposing the engine difference directly.

Parallelism: DP data, EP expert, CP context (sequence split), TP tensor. Consistent-version rollout runs 4 vLLM engines, each DP8/EP8, TP=1; training uses EP32, CP4, TP=1. **The two sides use different parallel layouts** (4.5).

Relevant structures: **MLA** (multi-head latent attention, which compresses KV before caching); **DSA** (DeepSeek sparse attention: a lightweight indexer scores history positions, Top-K keeps at most 2048, and MLA attends only to them); the **CSA compressor** (compresses history KV into fewer rows); **mHC** (mixing across residual streams, with the mixing matrix normalized by Sinkhorn iterations); and 43 MoE layers. SM100 is Blackwell (the GB200 used here); SM90 is Hopper.

Section 4 examines each key choice in the 3.2 table.

### 3.1 Division of work: inference kernels define the values, training adds backward

Two engines agree most easily by running the same code; the question is which side adapts. Rollout dominates RL time, with every operator tuned for throughput. With training as reference, every inference optimization would have to accommodate it, and A would fail. Inference as reference has a cost too: training still owns backward and saved tensors, and each layout entry point (e.g. masked / contiguous in 4.4) needs its own verification. So **inference kernels are the numerical reference, training forward calls the same vLLM kernels for its values, and training only adds backward.**

But inference kernels lack backward, and FP8 quantization is not differentiable. We use two kinds of wiring:

- **Most modules use a custom `autograd.Function`**: forward calls the vLLM kernel without gradients and saves what backward needs; backward is a hand-written BF16 / FP32 VJP (input and parameter gradients from the upstream gradient) w.r.t. the BF16 master weights, with FP8 quantization as identity (STE). Linear layers, norm, mHC, MoE and o_proj use it.
- **A few places use the straight-through form** `visible + (graph - graph.detach())`: forward takes the inference kernel's value (visible); backward runs through a differentiable PyTorch graph (graph, unrelated to CUDA Graph).

  Only attention q/k, compressed KV and the final logprob use it. These inference kernels are stateful fused operators that quantize and write the KV cache (or return only the selected token's logprob), with no differentiable output. Their math (RoPE, norm, compression pooling, log-softmax) is easy to express in PyTorch for autograd, with FP8 store/load as STE, so no hand-written backward is needed.

Each backward first checks the master weights are unchanged since forward, so recomputation or async updates cannot mix versions.

Both costs land on training. First, the gradient is a surrogate, not the quantized operator's exact derivative; a bitwise-identical forward proves nothing about gradients, so the skill checks backward against an independent reference (Section 5). Second, training forward must save the inference kernels' inputs and outputs for backward; the memory pressure is why the recipe enables full recomputation.

### 3.2 Per-operator master table

Per module: what each operator does in rollout and training, what is fixed or given up, and who pays. "Same as left" means training forward calls the same vLLM kernel as rollout. Cost sits mostly in MoE communication and backend (4.4, 4.5), then attention gather and dequantization (4.2, 4.3); dense is cheap.

**Dense, norm and mHC**

| Operator | rollout (BI=1) | Training | What is fixed / given up; cost |
|---|---|---|---|
| Embedding | Table lookup | Forward: Same as left<br>Backward: PyTorch deterministic | Naturally consistent |
| mHC residual mixing (pre / post / Sinkhorn) | Split TileLang mHC kernels, `split_tokens=1` | Forward: Same as left<br>Backward: Hand-written VJP, Sinkhorn iterations unrolled, FP32 | Gives up BI=0's fused post+pre kernel (split count varies with token count)<br>Cost: Both sides lose one fusion |
| RMSNorm | Batch-invariant RMSNorm (one program per row) | Forward: Same as left<br>Backward: FP32 VJP | Per-row reduction order fixed<br>Cost: Small |
| FP8 linear layers (q_a/kv, q_b, wo_b, shared expert) | DeepGEMM FP8 GEMM, batch-invariant mode; weights in 128×128 blocks, activations quantized dynamically per token in groups of 128 | Forward: Quantizes BF16 master weights on the fly to the same block FP8, same DeepGEMM<br>Backward: Transformer Engine (TE) BF16 GEMM w.r.t. BF16 master weights (STE) | Gives up BI=0's FlashInfer switch at M<32; scale is UE8M0 on SM100 (8-bit, exponent only, i.e. a power of 2), FP32 on SM90<br>Cost: rollout: no faster small-batch kernel; training: FP8 weights repacked every forward (including recomputation) |
| Small linear layers (compressor, indexer weight projection) | FP32-output cuBLAS GEMM / batch-invariant persistent Triton matmul | Forward: Same as left<br>Backward: BF16 | Summation order along K fixed<br>Cost: Small |

**Attention (DSA sparse attention + FP8 KV cache)**

| Operator | rollout (BI=1) | Training | What is fixed / given up; cost |
|---|---|---|---|
| q-norm + RoPE + KV write | Fused kernel quantizes K and writes the FP8 cache | Forward: Same kernel writes a private training-side FP8 cache, then reads it back<br>Backward: FP32 RoPE / norm in PyTorch (straight-through) | K takes the same FP8 round trip<br>Cost: Training: one extra cache and dequantization |
| FP8 KV cache | Stored as FP8; prefill and decode dequantize to BF16 before attention | Forward: Same as left<br>Backward: — | Gives up the default FP8-reading decode kernel (4.2)<br>Cost: rollout decode |
| CSA compressor | vLLM compression kernel, state reset per request | Forward: Same kernel's values, functional graph's gradients<br>Backward: autograd | In-request state starts as in prefill<br>Cost: Training: loop over requests |
| DSA indexer scoring | DeepGEMM FP8 MQA logits; prefill reads contiguous KV, decode reads paged KV | Forward: Same as left, per request<br>Backward: None (indexer not trained) | Paged and contiguous reads bitwise identical |
| Top-K selection + index arrangement | Batch-invariant Top-K (64-bit key + radix select), sorted descending, invalid entries last | Forward: Same as left<br>Backward: None | Ties go to the larger in-request index (4.3)<br>Cost: One extra sort per side, recovered by fusion |
| Sparse MLA | `flash_mla_sparse_fwd`; prefill one segment per request, decode same sparse kernel | Forward: Same as left, once per request<br>Backward: cuDNN frontend DSA backward | In-request geometry ignores neighbors; decode emits 1 token per request per call<br>Cost: rollout: multiple prefill launches; training: serial over requests |
| o_proj | SM100: DeepGEMM FP8 einsum; SM90: grouped Triton W8A8 | Forward: Same as left<br>Backward: TE BF16 grouped GEMM + inverse-RoPE VJP | Aligned per architecture; no bitwise identity across architectures<br>Cost: Loop over groups on SM90 |

**MoE**

| Operator | rollout (BI=1) | Training | What is fixed / given up; cost |
|---|---|---|---|
| Router scoring + expert selection | Same FP32-output gate GEMM for all batch sizes; sqrt-softplus + bias selects top-k | Forward: Same as left, routing recomputed bitwise<br>Backward: Expert IDs fixed, FP32 gradient w.r.t. routing weights | Gives up the small-batch gate kernel; **R3 is not needed** (4.4)<br>Cost: Small |
| Dispatch | DeepEP low-latency (LL below), fixed per-rank capacity, all ranks segment in sync; FP8 quantization uses elementwise round-to-nearest | Forward: DeepEP normal, plus one dispatch of routing metadata (fingerprint, expert, weight)<br>Backward: DeepEP backward | Experts get bitwise-identical FP8 values and scales either way<br>Cost: rollout: LL communication cannot overlap compute; training: metadata dispatch and hash matching |
| SiLU · mul · quantization | Batch-invariant fused kernel: product rounded once to BF16, then quantized in groups of 128 | Forward: Same as left (mandatory)<br>Backward: FP32 VJP | Rounding position fixed (4.4)<br>Cost: Small after optimization |
| Routed expert GEMM | DeepGEMM masked grouped GEMM, static capacity as scheduling hint | Forward: DeepGEMM contiguous grouped GEMM, per-expert rows aligned to 128<br>Backward: TE BF16 grouped GEMM | Gives up BI=0 FlashInfer TRT-LLM MoE (not batch-invariant)<br>Cost: rollout: a main throughput gap; training: alignment padding |
| Combine | DeepEP LL combine, fixed merge order | Forward: FP32 accumulation in top-k slot order on the token's owning rank, then BF16<br>Backward: Ordered gradient scatter | Merge order by slot, not arrival order or rank<br>Cost: 43 layers × 2 serial all-to-all per step |
| Shared expert | FP8 linear layer; auxiliary stream under LL + MNNVL (multi-node NVLink) | Forward: Same as left<br>Backward: BF16 (STE) | Stream changes overlap, not values<br>Cost: A source of recovered speed |

**Output layer and infrastructure**

| Item | rollout (BI=1) | Training | What is fixed / given up; cost |
|---|---|---|---|
| lm_head + logprob | Batch-invariant matmul; FP32 batch-invariant log-softmax | Logits per 8192 tokens, vLLM logprob value; backward via PyTorch log-softmax on the same logits (straight-through) | temperature must be a scalar<br>Cost: Training: each chunk's FP32 log-softmax kept for backward |
| Cross-GPU reduction | TP=1, no all-reduce; NCCL channel count and algorithm pinned as a guard | TP=1; CP exchanges boundaries only, no reduction | TP>1 not supported (4.5)<br>Cost: Cannot split with TP |
| CUDA Graph | Kept, with the right mode (4.6) | — | Graph and eager paths bitwise identical |
| Weight sync | Reload copies new values to original addresses, refreshes derived weights | Exports block-FP8 weights | Graph-referenced addresses unchanged<br>Cost: One export and transfer per step |

### 3.3 Results and who pays

![DeepSeek-V4-Flash 100-step training reward](/assets/figures/2026-10-14-train-inference-consistency/ds4-reward.png)

*Per-step training reward. Orange: BI=0 baseline; blue: BI=1 consistent. Thick line: moving average of the current and previous 4 steps; thin line: raw value.*

![DeepSeek-V4-Flash per-step train-inference difference](/assets/figures/2026-10-14-train-inference-consistency/ds4-train-infer-mismatch.png)

*Top: per-step max train-inference probability difference; bottom: train-inference KL (k3 estimator r − 1 − log r, used throughout; symmetric log scale, linear near zero). Max difference reacts to one routing flip or Top-K swap, so it is sizable from baseline step 1; KL shows the overall level. Orange: BI=0; blue: BI=1. Both BI=1 metrics are 0 at every step.*

Over 100 steps, BI=1 rollout averages 24.22 token/ms, 91.3% of BI=0 (26.53):

![DeepSeek-V4-Flash rollout throughput](/assets/figures/2026-10-14-train-inference-consistency/ds4-rollout-throughput.png)

*Per-step rollout throughput (output token/ms, summed over 32 GPUs). Orange: BI=0; blue: BI=1. Thick line: moving average of the current and previous 4 steps; thin line: raw value.*

Baseline train-inference KL grows (2e-3 to 29.9) while its reward falls back; BI=1 stays at diff 0 and its reward holds. Another potential gain is one fewer forward: we recompute old-logprob each step only to measure diff; after acceptance, training can use the rollout logprob instead.

**Rollout side**, measured in this run: average throughput is 91.3% of baseline (about 9% lower), above Section 1's 70% floor. Baseline slowdown after step 90 inflates this; the first 89 steps give 88.6%.

On the inference benchmark the consistent version matches or beats the baseline (4.5); the training-time gap is not yet attributed to operators; 3.2 points to MoE LL communication and backend, attention gather and dequantization, and per-request launches.

**Training side**, not measured separately: per 3.2, overhead is repacking FP8 weights every forward, serial per-request attention and Top-K, one extra routing-metadata dispatch, and memory pressure from saving intermediates for backward (hence full recomputation: forward runs twice).

A separate verl recipe PR preview on 32 GB200 GPUs reports about 10% end-to-end overhead: rollout near parity, the gap mostly in training. Training-side optimization is next.
## 4. Module by module: technical choices in the DeepSeek-V4 recipe

This section expands the key choices in the 3.2 master table. Each subsection covers the bottleneck, the choice and why, the numerical problem it introduces, the fix, and who pays.

### 4.1 W8A8 and dense GEMM: same precision, same accumulation

**Bottleneck and choice.** DeepSeek-V4-Flash-Base ships in block-FP8, so the natural deployment path is W8A8: weights in FP8 E4M3 with 128×128 blocks, activations quantized dynamically per token in groups of 128, and DeepGEMM for the GEMM.

A BF16 rollout would cost compute and weight bandwidth (we did not measure the loss). A separate training-side scheme such as MXFP8 would train a policy closer to, but not the same as, the deployed one: the limitation of the QAT row in Section 1.

So the consistent version's training forward uses deployment-time W8A8 values: each forward quantizes the BF16 master weights to block FP8 by the deployment rule (checkpoint scales before the first update, recomputed from amax afterwards) and calls the inference GEMM.

**Not everything is 8-bit.** DeepGEMM scales are powers of 2: UE8M0-packed on SM100, FP32 on SM90. The router gate and compressor projection output FP32, the indexer weight projection and lm_head are BF16, the KV cache is FP8 (4.2), and the routed experts' FP8 path is in 4.4. Backward is BF16 / FP32 against the BF16 master weights; optimizer state is FP32.

**The numerical problem: same precision, different accumulation.** The inference engine picks kernels by shape: the FP8 linear layer uses FlashInfer when M<32 and DeepGEMM otherwise; the router gate switches to a small-batch kernel at 16 tokens or fewer; the mHC split count grows with token count. Floating-point addition is not associative, so each makes a token's result depend on its neighbors.

![Dense GEMM: which kernel the same request uses in steps of different sizes](/assets/figures/2026-10-14-train-inference-consistency/ds4-dense-gemm.png)

*Horizontal axis: tokens per step (schematic). The default path switches kernels at 32, so the same request in an 8-token and a 200-token step goes through FlashInfer and DeepGEMM and differs in the last bit. Batch-invariant mode has one summation order.*

**Fix.** Under BI=1, each output element has exactly one accumulation order:

- the FP8 linear layer always uses DeepGEMM in batch-invariant mode (a DeepGEMM version without that mode raises an error rather than silently falling back);
- the router gate uses the same FP32-output GEMM at every batch size;
- the small BF16 matmuls (indexer weight projection, lm_head) use a persistent Triton kernel with a fixed summation order along K; tile shape may vary with M, summation order does not;
- the mHC split is fixed at 1, and the fused pre/post kernel is split.

Training calls the same kernels and packs FP8 weights by the deployment rule: a Triton kernel emits packed UE8M0 weights on SM100; SM90 uses vLLM's weight post-processing.

**Cost.** Rollout loses faster small-batch kernels and one mHC fusion, but Tensor Cores stay in use. Training does not cache FP8 weights by default, so every forward (including the recompute forward) repacks them; the cache saves that time for one more resident FP8 copy.

Because the BI=0 mHC split count varies with token count, each new prefill size triggers a 10–14 s recompilation. Rule out this stall before comparing the two runs.

### 4.2 FP8 KV cache: FP8 storage, unified read path

**Bottleneck and choice.** The DeepSeek-V4 KV cache is FP8. Per token per layer it stores 448 FP8 NoPE (no positional encoding) dimensions, 64 BF16 RoPE dimensions, one UE8M0 scale per 64 elements and 1 alignment byte: 584 bytes, about 57% of BF16 (1024 bytes), which directly sets concurrency. The consistent version keeps it, and both runs share one fused write kernel (q-norm, RoPE, K quantization, write).

**The numerical problem** is in the read. Default decode uses a FlashMLA kernel that reads FP8 directly with tile scheduling, partitioning work by batch and merging partial results. Prefill dequantizes to BF16 and runs sparse attention. Two algorithms rule out phase consistency, and decode's batch-dependent partition breaks batch independence.

**Fix.** Route decode through the prefill path: dequantize and gather the selected KV into a BF16 workspace, then call the same `flash_mla_sparse_fwd`. This needs the sparse decode kernel selected explicitly: the consistency switch alone does not change the default decode path, an easy setting to miss.

The workspace is per request, sized only by max length, compression ratio and sliding window, not by other requests. Each request generates one token per step; multi-token prediction (MTP) and speculative decoding are not covered.

![FP8 KV cache: prefill, decode and training share one read path](/assets/figures/2026-10-14-train-inference-consistency/ds4-fp8-kv.png)

*Top: before, prefill and decode read the same FP8 history with two algorithms, and decode's partition depends on the batch. Bottom: prefill, decode and training (with a private cache copy) share one read path.*

Training has no real cache, but K must make the same FP8 round trip. The training forward writes K into a private FP8 cache with the same fused kernel and reads it back with the same dequantization; backward is straight-through, via an FP32 PyTorch RoPE / norm. The CSA compressor likewise takes values from the vLLM compression kernel and zeroes state at each request start, matching rollout prefill.

**Cost.** Decode loses the direct-FP8 kernel and adds gather and dequantization; training adds a cache and a dequantization, and the compressor loops over requests. This is the price of phase consistency; storage precision is untouched.

### 4.3 Top-K in sparse attention

**Bottleneck.** The DSA indexer scores every history position, and Top-K picks at most 2048 for attention. This discrete choice turns tiny numerical differences into a different context: one ulp upstream can swap a position and change the whole layer after it.

![Top-K: same scores, same selection, same order](/assets/figures/2026-10-14-train-inference-consistency/ds4-topk.png)

*Top: layout of the 64-bit selection key. Bottom: an example with K = 2 where positions 17 and 42 tie and the position bits in the key select 42; selected positions are then sorted by descending position, invalid entries last.*

**Align the scoring first.** Indexer scoring uses DeepGEMM's FP8 MQA logits kernel; prefill reads contiguous KV and decode reads paged KV. Our tests confirm the two reads are bitwise identical across context lengths and when mixed with other requests. Training calls the same kernel per request, so lengths and strides match rollout.

**Then make selection deterministic.** Default Top-K breaks ties by thread scheduling. The batch-invariant Top-K encodes each candidate as a 64-bit key: high 32 bits are the score after an order-preserving transform, low 32 bits the in-request position index. Unique keys give a unique selection, ties going to the larger index; a radix select at 8 bits per round finds the K-th largest key. The in-request index, not the global batch position, is deliberate: otherwise moving a request within the batch would change tie results.

**The same set is not enough; the order must match too.** The indices set the gather order and thus attention's accumulation order, so they are sorted by descending position, invalid entries (-1) last, then concatenated with the sliding-window indices.

The R3 alternative records the indices during rollout and replays them in training, but that is up to 2048 indices per token per layer: GB-scale per sequence for long responses.

**Training-side trade-off.** The training forward selects the same positions with the same scoring and Top-K, but the indexer is not trained: its input is detached and the auxiliary loss coefficient is 0, mainly for simplicity.

This matches existing practice. [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) trains DSA with the indexer's input detached, the indexer trained only by a separate KL alignment loss and the main model only by the language-model loss, so the RL loss never reaches the indexer. [GLM-5](https://arxiv.org/abs/2602.15763) also freezes the indexer by default in RL, to speed up training and avoid unstable indexer learning. Our auxiliary loss also does not yet support CP>1.

**Cost and recovery.** The deterministic Top-K takes about 5.9 µs per call, an order of magnitude faster than another radix select implementation with a built-in histogram (56.7 µs). Fusing Top-K with the index sort raises the consistent version's inference-benchmark throughput by 6.7% (configuration in 4.4); separately the two changes give only +2.3% and +0.1%. The interaction is unexplained, so measure them together. The remaining cost is multiple per-request launches in prefill.

**Not yet covered**: under decode context parallel (DCP>1), the tie rule for merging candidates across GPUs is the opposite of the local rule, so this combination is unsupported.

### 4.4 Low-precision expert GEMM: two layouts, same row result

**Bottleneck.** The routed experts' FP8 grouped GEMM dominates the MoE layer, and its input layout depends on the communication mode. Rollout uses DeepEP low-latency (LL) mode (why: 4.5), giving a masked layout chunked by expert with fixed capacity per chunk. Training processes many tokens with DeepEP normal mode, giving a contiguous layout packed by expert.

**Consistency locks the backend.** Of the block-FP8 MoE backends in our vLLM version, only DeepGEMM's grouped GEMM passed batch-invariance acceptance: cutlass lacks block-FP8, and triton falls into a batched implementation that does not declare batch invariance. So both consistent sides use DeepGEMM in batch-invariant mode; the baseline can use the faster FlashInfer TRT-LLM MoE.

Rollout's masked grouped GEMM takes static capacity as the scheduling hint instead of reading the per-expert max token count back to the host, which would synchronize and cannot be graph-captured (4.6). Training uses the contiguous grouped GEMM, with each expert's row count aligned to 128.

**Numerical problems and fixes.** For a token to get the same expert output in both layouts, three conditions must hold:

1. **Experts receive the same FP8 input.** LL quantizes activations at dispatch and normal mode after it, originally with different rounding. We switched LL to elementwise round-to-nearest, bitwise identical to training-side quantization, which also runs on the contiguous layout under the "LL dispatch quantization convention".
2. **Each row's result is independent of layout.** Quantization is per token and batch-invariant DeepGEMM accumulates each row in a fixed way, so a row's result depends neither on its neighbors nor on masked vs contiguous layout. Bitwise router + MoE tests confirm this.
3. **The activation is rounded once.** Between the two GEMMs, SiLU(gate) × up is computed and quantized to FP8. We multiply in FP32, round the product to BF16 once, take amax per group of 128, compute the scale and convert to FP8. Rounding SiLU to BF16 before the multiply can differ by one ulp, which FP8 quantization amplifies into a code flip. The fused kernel has fixed launch geometry and is mandatory on both sides; training refuses to start without it. Its SiLU formulation differs on SM100 and SM90; each is aligned separately, with no cross-architecture bitwise promise.

![Expert GEMM: two layouts, the same row result](/assets/figures/2026-10-14-train-inference-consistency/ds4-expert-gemm.png)

*The same token (orange) sits at different positions in rollout's masked and training's contiguous layouts, yet the three conditions make its expert GEMM result bitwise identical. Gray: empty slots; hatching: alignment padding.*

**Training backward.** Forward values come from DeepGEMM; backward uses TE's BF16 grouped GEMM against the BF16 master weights: dgrad / wgrad of the second GEMM, then SwiGLU recomputed from saved gate_up and differentiated in FP32, then the first GEMM. FP8 quantization is treated as identity in backward.

**Why R3 is unnecessary here.** R3 (Section 1) replays rollout routing because the two sides may compute different routing. Here the router's scoring GEMM uses one FP32-output kernel at every batch size on bitwise-identical inputs, so recomputed routing matches rollout. Backward fixes the expert indices and differentiates only routing weights.

This assumes 1 update per step, as in the DeepSeek-V4 experiment. With several mini-batch updates per step (step batch larger than mini-batch), weights change after the first, later mini-batches' routing drifts from sampling time, and training is no longer strictly true on-policy; hence our recommendation of 1 update per step (Section 2). The math task in 6.2, with 4 updates per step, is exactly this case.

**Cost.** Rollout gives up FlashInfer TRT-LLM MoE, but the GEMM is not most of the gap. On the inference benchmark (GB200, DP8/EP8, 256 concurrency, 150 input / 3000 output per request) the consistent version started 33% slower than the baseline, about 83% of it from non-overlappable LL communication (4.5).

We also tried other parallel layouts for performance only (the final recipe is restricted to TP=1): TP2/DP4 is 17% slower, TP4/DP2 28% slower, and DP8 remains best for the consistent version. The inference-benchmark numbers below come from different optimization stages: numbers in one sentence share a configuration; do not compare across paragraphs.

Training must save the input and gate_up for backward and pays for padding each expert to 128 rows.

The SiLU·mul·quantization kernel also taught a fairness lesson. vLLM enables it whenever the shared library is present, ignoring the consistency switch, so any image shipping the library runs this batch-invariant kernel under BI=0 too. Unoptimized, it took 35.7 µs per call and 9.9% of GPU time, slowing the baseline. Optimization cut it to 5.3 µs and lifted the benchmark of the time from 16703 to 18313 tok/s for both runs. Confirm which kernel the baseline runs before comparing.

### 4.5 Cross-GPU reduction: off the compared path

**Bottleneck.** Rollout wants low latency and uses DP8/EP8; training must fit long-sequence activations and uses EP32 plus CP4. As Section 2 noted, different layouts need results invariant to parallelism degree, usually via a fixed, expensive reduction tree.

**Choice.** This recipe does not build a layout-invariant reduction tree. Instead it **keeps cross-GPU reduction out of the numerical path being compared**, and makes the remaining communication move data without changing values:

- **TP=1.** Neither side uses TP, so linear layers have no all-reduce. The recipe rejects TP>1 at startup, since vLLM's custom and FlashInfer all-reduce pick algorithms by message size.
- **EP merge is local.** The all-to-all only moves data. Expert outputs return to the token's owning rank, are weighted and accumulated in FP32 in top-k slot order, and stored as BF16. Order depends only on slot, not on EP=8 vs 32 or arrival order. Rollout's DeepEP LL combine also uses a fixed merge order, not first-come-first-added.
- **CP does no reduction.** Training pads each sequence and splits it contiguously across CP ranks; neighbors exchange only boundary hidden states, and compressed KV rows are all-gathered in the CP group. This is concatenation, not summation, and per-request geometry (length, stride, compression boundaries) matches rollout.

![Different parallel layouts, bitwise identical results](/assets/figures/2026-10-14-train-inference-consistency/ds4-cross-rank.png)

*One token's path: the all-to-all only moves data, and the only summation happens on the token's owning GPU, in FP32 in top-k slot order, so EP8 and EP32 give the same result.*

**Numerical problems and fixes: making the data movement deterministic.**

- **Rollout: fixed shape.** Each rank sends at most a fixed number of tokens per call and all ranks segment in lockstep; ranks with few tokens join every dispatch and combine with empty segments, and the shared expert launches once, in the first segment. Only LL implements this, and DeepEP normal mode disables CUDA Graph when DP>1 (4.6), so rollout must use LL.
- **Training: recover each result by slot.** Normal mode has no fixed arrival order, so training dispatches routing metadata once more (per route: a 16-element token fingerprint, expert index and FP32 routing weight), matches each expert output to its token and slot by bitwise hash, and scatters into expert-ordered rows with a deterministic Triton kernel. Backward follows the same mapping, with routing-weight gradients in FP32.
- **Communication library pinned as a guard.** NCCL channel count, thread count, protocol and algorithm are pinned, and both symmetric-memory all-reduce and custom all-reduce are off. Under TP=1 they are off the compared path; this is a guard, not the consistency mechanism.

**Cost.** Rollout: in LL mode each step has 43 layers × (dispatch + combine) = 86 serial round trips. The cross-layer overlap schemes we tried compute before inputs are ready and break batch independence, so we dropped them. Interleaving two micro-batches (DBO) was 39% slower: each layer's expert weights are read twice, and weight reads already use 84% of HBM bandwidth per the profile.

We eventually closed the communication gap on the inference benchmark. Confirm both runs use the same transport first: we lost two days because a script pinned the consistent version's DeepEP to Ethernet RDMA (RoCE) while the baseline used the much faster MNNVL. With matched transport and the shared expert on an auxiliary stream, the consistent version was 3.2% faster than the baseline (16834 vs 16313 tok/s), with bitwise-identical outputs before and after the transport change.

Training: one extra metadata dispatch plus hash matching. **Not covered**: TP>1 and decode context parallel, which would put cross-GPU reduction or merging back on the compared path.

### 4.6 CUDA Graph: required for rollout, and must coexist with consistency

**Bottleneck.** Decode runs dozens of small kernels per layer per step at one token per request, so launch overhead would dominate; we keep CUDA Graph to hit the performance target. Graphs conflict with batch invariance in three places: a step may run in the graph or fall back to eager; graphs forbid host syncs and batch-dependent shapes; and graphs capture addresses while RL swaps weights every step.

![Three conflicts between CUDA Graph and batch invariance](/assets/figures/2026-10-14-train-inference-consistency/ds4-cuda-graph.png)

*The path each mode takes for three kinds of step, with inference-benchmark throughput on the right. A request switches between graph and eager depending on its batch, so both paths must be bitwise identical.*

**Conflict 1: graph or eager.** vLLM has three modes. PIECEWISE captures everything except attention, in pieces. FULL captures the whole step, but on DeepSeek-V4 applies only to pure decode steps. FULL_AND_PIECEWISE uses FULL for pure decode steps and PIECEWISE for mixed steps.

On the inference benchmark with the same configuration, FULL_AND_PIECEWISE gives 16409 tok/s, FULL 16068 and PIECEWISE 10423. In every mode, a step runs in the graph if its total token count fits the max capture size and in eager otherwise, which depends on the other tokens in the step, so the two must be bitwise identical.

That was our first mismatch: an indexer short-context shortcut was baked into the graph at capture based on the dummy input's length, so graph and eager steps selected different Top-K. Making the check depend only on a static upper bound fixed it.

We use FULL_AND_PIECEWISE, the fastest of the three, and diff stays 0 across the 100-step training run. FULL captures only pure decode steps, so every step mixed with prefill runs eager, and time to first token rises from 764 ms to 1792 ms.

**Conflict 2: no host sync or batch-dependent shapes in the graph.** This aligns with consistency, and many designs serve both:

- The MoE masked GEMM uses static capacity as the scheduling hint instead of reading the per-expert max token count back to the host (4.4);
- LL's fixed per-rank capacity and lockstep segmentation fix the communication shape (4.5); DeepEP normal mode disables the graph when DP>1, another reason rollout can only use LL;
- The sparse attention decode workspace size depends only on max length, compression ratio and window (4.2), and the row starts Top-K uses live in a resident zero buffer;
- All workspaces are allocated once in warmup, before capture, and locked afterwards;
- Performance-only shortcut branches (for example skipping indexer zeroing for short contexts) are disabled at capture time so no step's branch gets baked into the graph (the Conflict 1 bug came from such a branch).

**Conflict 3: graphs capture addresses, weights change every step.** After each step, training quantizes the updated BF16 weights to block FP8 by the deployment rule, exports them and transfers them to rollout in buckets. Rollout must copy new values into the original storage rather than new memory that leaves the graph on old addresses, and recompute weight-derived mHC parameters in place without new tensors. Matching at the first step is not enough, so the skill's acceptance compares again after update, export and reload.

**Cost.** Mainly mode choice: PIECEWISE is slowest, and FULL has worse time to first token. Workspaces reserved at their upper bound cost some memory. Other speed recoveries (removing needless debug padding and cold start from rollout, faster KV cache zeroing) apply to both runs; the baseline must get them too, or the performance-loss number is meaningless.
## 5. Lowering engineering cost with Agent + skill

A and B can coexist, but the key choices in the 3.2 master table and Section 4 take expert judgment: which backend is batch-invariant, that discrete choices need aligned ties and order, how to align quantization and merge order across different communication modes, and that a communication gap first requires ruling out a transport difference. A new model, precision or hardware means starting over, so C does not hold.

After DeepSeek-V4-Flash, we saw that most of this work is two kinds of task, each with a measurable goal:

- **Bring diff to 0.** Fix weights and input, find the first operator where the two sides diverge, reduce it to a minimal rerunnable repro, fix it, then verify in real training. Outputs are compared directly.
- **Optimize performance while keeping diff at 0.** Profile hotspots, A/B single operators, and re-verify consistency after every change. Speedups that change results or that show up only in micro benchmarks, not real rollout, do not count; slowing the baseline to narrow the gap is not allowed.

Reading code, tracing call chains, locating divergences, writing repros, changing kernels and rerunning regressions is long-running work suited to an agent, with success decided by consistency comparison and performance measurement.

We packaged this as the [train-inference-consistency skill](https://github.com/vllm-project/vllm/pull/61551). A skill is an operating manual for agents: what to do in what order, and what each step must reach to pass. It is not one experiment's launch command; it is meant to work for other models too.

The manual has 11 gates (C0–C10) with explicit pass conditions. To speed up cold start, we **first cut a few layers from the full model into a small model, solve most problems there, then move to the full model**. Five steps:

1. **Set the target and baseline (C0–C1).** List the model's modules with their precision and parallelism. Tune the ordinary, non-consistent version (the baseline) to normal high performance for human review. It is the denominator for "how much slower the consistent version is".
2. **Cut a small model and make inference self-consistent (C2–C4).** The small model (usually 4–6 layers picked from the full model) must contain every operator class, keep adjacent-layer interaction, and match per-layer size and quantization, so it reproduces most local numerical problems. Full depth, long context, multi-GPU layout and memory pressure must still be accepted on the full model. First make the inference engine internally consistent: a request gives the same result alone and in a batch, and whole-input processing matches token-by-token. Meanwhile, look for structural problems that would slow the consistent version.
3. **Connect training (C5–C7).** Training calls the same operators as inference, recomputes inference-generated data, and compares token by token; on a mismatch, compare layer by layer for the first divergent layer. Then add backward and check gradients against a reference implementation independent of the code under test: a consistent forward does not imply correct gradients.
4. **Run RL, then switch to the full model (C8–C9).** On the small model, run two consecutive updates in the real training framework, comparing under the same weight version before and after each. After human review, switch to the full model and scale up stepwise, including checking the first step after checkpoint restore.
5. **Recover performance (C10).** Optimize the most expensive operators one by one, rechecking consistency each time, then run a paired comparison against the baseline on the delivery image.

Two designs matter most. First, **two agents advance the baseline and consistency in parallel**: one makes the baseline fast, the other builds the consistent version. The tuned baseline is frozen for human review, so the agent cannot slow it to narrow the gap. Second, **our pitfalls are written into the process**: the running program must match the source being read; checkpoints need the first post-restore step verified, not just a save; every performance conclusion needs a fair baseline and traceable raw data; before comparing communication bottlenecks, confirm both runs use the same transport (the lesson from 4.5).

The skill lowers the cost of localization and regression (savings not yet quantified here); it does not replace human judgment: humans still review the baseline, precision trade-offs and recipe. 6.2 is an example: the agent builds a repro from the symptom, runs discriminating experiments and eliminates hypotheses, and a human sets the direction at a specific step.

## 6. Transfer: Nemotron, a Mamba hybrid architecture + NVFP4

We then tested the reference recipe and skill on a structurally very different model: Nemotron.

Recurrent structures have precedent: in the vLLM community, Alexander Jiang and the SkyRL team designed chunkwise-parallel recurrent (CPR) in [IsoExec](https://vllm.ai/blog/2026-08-21-isoexec) for the Gated DeltaNet (GDN) in Qwen3.5-35B-A3B. It keeps the recurrent form but computes chunk-parallel, so training, prefill and decode share one rounding order; it is also validated on MiMo-7B, GLM-4.7-Flash and Qwen3.5-9B.

IsoExec's "zero mismatch" covers only operators within its execution contract; its end-to-end experiments still report a tiny nonzero logprob difference (mean absolute difference around 1e-7), unlike this post's "bitwise identical".

IsoExec also contributes the **unified model**: training and inference share one model definition and one set of aligned kernels (batch-invariant GEMM, attention, norm), so training, prefill and decode use the same numerics by construction.

Mamba-2 and GDN are both linear recurrences: state recurs per token, training and prefill run chunk-parallel, decode runs one step at a time, and all three need one rounding order. We make decode exactly replay prefill's chunked arithmetic instead of its own single-step recurrence, and training imports the same operators as inference, following the unified model's kernel sharing. Agent + skill did this alignment.

The target is **Nemotron-3.5-Lightning-30B-A3B (NVFP4)**, a Mamba-2 / attention / MoE hybrid (NVFP4 W4A16 routed experts, FP8 Mamba projections, FP8 KV cache for attention) on GB200. BI=0 uses NVFP4 QAT training (quantization rule and export from NVIDIA ModelOpt) + R3 route replay; BI=1 shares vLLM's forward, without route replay. Both runs use BF16 master weights and 1 update per step.

In vLLM, every operator that might depend on batch composition gets a batch-invariant implementation, and training (a lightweight Megatron-LM fork) imports them:

| Operator | What BI=1 does |
|---|---|
| Mamba-2 SSD scan | decode exactly replays prefill's chunked arithmetic, fixed autotune config, fp32 state |
| Attention | FlashAttention with fixed split-KV, split count independent of batch |
| Routed experts (NVFP4 W4A16: 4-bit weights, 16-bit activations) | CuTe-DSL grouped GEMM, single tactic, no split-K, fp32 merge in slot order |
| MoE all-to-all | one-sided communication, fixed merge order |
| Router GEMM | Batch-invariant persistent matmul, fp32 output |
| Norm / FP8 quantization | Unified batch-invariant kernels; value-changing fusions forbidden |

### 6.1 Results: training on SWE-bench Pro

We run agentic RL on SWE-bench Pro in remote sandboxes: 32 tasks × 8 trajectories per step, about 57K tokens per trajectory on average, GRPO, 8 GB200 nodes per run. Both runs use the same image and recipe, without corrections such as TIS or MIS. With 1 update per step, the KL between pre-update rollout logprobs and the training-side recomputation is that step's train-inference KL.

![Nemotron SWE-bench Pro: train-inference KL, training score and entropy](/assets/figures/2026-10-14-train-inference-consistency/nemotron-swe.png)

*Top: per-step train-inference KL; middle: training score (moving average over the current and previous 4 steps); bottom: actor entropy. Orange: BI=0; blue: BI=1. At BI=1 step 10 only a few trajectories returned from the sandbox, so its score is excluded.*

- **BI=1**: train-inference KL is exactly 0 every step, entropy stays around 0.05, and the score goes from 0.56 (first 10 steps) to 0.59 (last 10), with no drop.
- **BI=0**: train-inference KL falls from ~0.015 to ~0.002, but entropy rises from 0.05 to 2.6 and the score drops from 0.56 to 0.22 (first 10 steps vs. last 10).
- **rollout speed**: over the first 30 steps, while both runs are healthy, BI=1 rollout throughput is 87.5% of BI=0, only 12.5% slower.

![Nemotron SWE-bench Pro: rollout throughput](/assets/figures/2026-10-14-train-inference-consistency/nemotron-swe-rollout.png)

*Per-step rollout throughput (token/ms, moving average over the current and previous 4 steps). Orange: BI=0; blue: BI=1.*

### 6.2 Lesson: with multiple updates per step, consistency is not enough

Earlier, we reused a math-task recipe with 4 mini-batch updates per step. Consistency only guarantees identical routing under the same weight version. After the first update the weights change, so routing recomputed for later mini-batches deviates from sampling-time routing: about 7% of tokens switch expert combination after each update. Hence we recommend 1 update per step (Section 2), as in 6.1. The agent located this with a small-scale repro; a human set the direction.

## 7. Open source

We open-source this so others can reach train-inference consistency with less reliance on senior engineers checking operator by operator, using the skill, ready-made code and an agent on their own model. The release:

- **[train-inference-consistency skill](https://github.com/vllm-project/vllm/pull/61551)**: the full workflow (C0–C10), ten guides, and key practices such as logprob comparison and gradient cross-checking;
- **DeepSeek-V4**: a true on-policy RL recipe reaching 0 diff on Megatron and vLLM, [verl-project/verl#7688](https://github.com/verl-project/verl/pull/7688);
- **Nemotron-3.5-Lightning-30B-A3B** (BF16 master weights, quantized to the deployment NVFP4 in the training forward and at weight sync):
  - vLLM batch-invariant inference aligned with training: [aoshen02/vllm#85](https://github.com/aoshen02/vllm/pull/85);
  - A training actor aligned with vLLM: [aoshen02/Megatron-LM#7](https://github.com/aoshen02/Megatron-LM/pull/7);
  - Two verl changes: [aoshen02/verl#11](https://github.com/aoshen02/verl/pull/11);
  - Train-inference consistency recipe and unified image Dockerfile: [aoshen02/verl#12](https://github.com/aoshen02/verl/pull/12).

Issues, PRs and feedback from using the skill on your own model are welcome.

Next: train-inference consistency for DeepSeek-V4.1 has reached 0 diff; performance optimization is in progress and will get its own post.


## Acknowledgments

Thanks to (in alphabetical order): [Jian Hu](https://hijkzzz.notion.site/) of NVIDIA Research, [Nolen Liang](https://github.com/NolenLiang) and [Tomer](https://github.com/TomerBN-Nvidia) of the Nemotron team, [Verda](https://verda.com/) for sandbox support, the [verl](https://github.com/verl-project/verl) team, and the vLLM community.
