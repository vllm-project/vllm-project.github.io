---
layout: post
title: "Finish RL Training Faster: Bitwise-Consistent, High-Performance, Agent-Driven"
author: "Yan Bai (NVIDIA DevTech), Ao Shen (Inferact)"
date: 2026-10-14 09:00:00 +0000
summary: "We make vLLM rollout and the training forward give bitwise identical logprobs on DeepSeek-V4-Flash and Nemotron-3.5-Lightning-30B-A3B (NVFP4), keep 91% of rollout throughput on DeepSeek-V4-Flash, and open-source the agent skill."
image: /assets/figures/2026-10-14-train-inference-consistency/ds4-reward.png
social_image: /assets/figures/2026-10-14-train-inference-consistency/ds4-reward.png
tags:
  - reinforcement-learning
  - deepseek
---

Rollout dominates RL training time, so training systems adopt low-precision compute, low-precision KV cache, and communication and kernel optimizations. These change the inference-side numerics. The inference engine generates a response; the training engine recomputes its probabilities and updates the model. With identical weights, the two sides now assign different probabilities to the same token, so training optimizes a policy other than the one that sampled. The gap can grow with updates and destabilize training. In our DeepSeek-V4-Flash-Base experiment (no algorithmic correction such as TIS), the high-performance baseline's train-inference KL rose from 2e-3 to 29.9 while reward fell back.

Unstable training means more crashes, rollbacks and reruns, a smaller share of effective training, and more total time to reach the same result. Our goal is to **finish effective RL training faster**: by cutting crashes and reruns, reach a result in less end-to-end time.

Our approach: on the high-performance numerical path, training and inference produce a **bitwise identical** logprob for every sampled token (diff = 0; Section 2 defines it exactly), which is true on-policy RL. We get there by having the training forward call vLLM's inference kernels directly and modifying those high-performance kernels so results do not change with batch composition, prefill or decode phase, or parallel layout (Sections 3 and 4). An agent skill then lowers the cost of doing this on a new model and makes the process transferable: it tells an agent how to find the first operator where the two sides diverge, reduce it to a minimal repro, fix it and rerun the regressions. On DeepSeek-V4-Flash we reach 0 diff at 91.3% of the high-performance baseline's rollout throughput, and we carried the method over to Nemotron, a Mamba hybrid with NVFP4.

This post measures numerical consistency, training curves and rollout throughput. The end-to-end gain comes from fewer crashes and reruns, which shows only over long runs; that wall-clock comparison is left for later.

## 1. Existing approaches: how they trade off the three

Finishing effective RL training faster needs three properties at once:

- **A, high performance**: fast rollout and training, with low-precision compute, low-precision KV cache and communication optimizations all usable;
- **B, high consistency**: training and inference compute the same policy for the same weights;
- **C, low cost and generality**: getting A and B on a new model, precision or hardware takes acceptable development time.

These rarely hold together. Drop B and you optimize freely but risk unstable training. Drop A and you can use consistent kernels, but inference gets much slower (numbers in the table below). Keeping A and B means aligning model by model and operator by operator, so C fails.

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

These methods reduce the effect of the mismatch but do not remove it: the mismatch turns on-policy training into off-policy training ([Yao et al.](https://fengyao.notion.site/off-policy-rl)); when it is large and has several sources, token-level importance sampling is biased and training is more prone to collapse ([Li & Liu](https://yingru.notion.site/Part-1-Why-Mismatch-Breaks-LLM-RL-295211a558b780b8bb4cc8e367d906d0)); in [P1-VL](https://arxiv.org/abs/2602.09443), truncated importance sampling only delayed collapse. R3 (Rollout Routing Replay, [Ma et al.](https://arxiv.org/abs/2510.11370)) records rollout expert routing and replays it in training, removing only the routing-flip part.

On the last row, Thinking Machines Lab achieves true on-policy RL with batch-invariant kernels ([Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference)) at the inference-time cost in the table, much of it from a FlexAttention integration not yet well optimized; [vLLM and TorchTitan](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference) reuse vLLM's forward kernels in training, and their bitwise-consistent run reached a higher reward in fewer steps; [slime](https://github.com/THUDM/slime) aligns logprobs on the GLM-5 architecture (MLA + DSA, same structure as DeepSeek-V3.2); and [IsoExec](https://vllm.ai/blog/2026-08-21-isoexec) by Alexander Jiang and the SkyRL team covers linear attention (Section 6).

Most of these approaches stack: some target the numerical difference between engines on one weight version, others the policy difference between weight versions. In asynchronous training, the off-policy from stale weights still goes to TIS, MIS or Score Centering, but the engines can still be numerically aligned on those weights, and the bitwise-consistent run becomes the baseline for measuring such corrections. QAT also belongs to strict alignment: low-precision rollout and training match bitwise only if the training forward applies the same quantization; our backward uses STE.

We take the last-row path that keeps A, with this bar for A: **rollout throughput at least 70% of an equally optimized high-performance baseline** (rollout is the most time-consuming part of RL training; training-side overhead was not measured separately, see 3.2). C is this path's weak point; Section 5 covers how Agent + skill makes up for it.

## 2. What train-inference consistency requires

In this post, diff is the difference, before each update with the same weight version and token prefix, between rollout's logprob for each valid token (sampled tokens inside the response mask) and the training side's recomputed old-logprob. Acceptance requires the max, mean and standard deviation of the per-token difference |p_rollout − p_train|, and the k3 KL, all to be exactly 0. This is exact equality, not a threshold, checked before each update; a mini-batch update changes the policy anyway, which the definition excludes. It holds on our accepted tasks (math reasoning, multi-turn tool calling) and parallel layouts, including different layouts per side.

"Training and inference compute the same thing" breaks into four dimensions plus a shared numerical contract:

```
train-inference consistency ⇐ kernel determinism + batch independence + phase consistency + cross-GPU consistency + same numerical contract on both sides
```

**Kernel determinism (single GPU).** The same kernel on the same input always gives the same output. GPUs do not guarantee this by default. An example is split-K matrix multiplication: a different accumulation order gives a different result because floating-point addition is not associative.

**Batch independence.** A request's result must not depend on its batch-mates. During inference, the engine merges and splits batches dynamically for throughput. In training, micro-batch size, packing and per-expert MoE token counts also change the path. Various optimizations (kernel choice by batch size, dynamic split count) make a given request depend on others in its batch.

**Phase consistency.** A token must get the same result in prefill (many tokens at once) and decode (one at a time). Inference engines give each phase its fastest kernel; forcing a shared path is a main source of the consistent version's slowdown and a focus of optimization.

**Cross-GPU consistency.** Cross-GPU reductions (AllReduce, reduce-scatter, and merging expert outputs after the MoE all-to-all) must also be fixed, at two levels.

- With the **same parallel layout** on both sides, communication results must not vary with timing and scheduling (fix the communication library's algorithm, reduction chunking and precision, and expert merge order).
- With **different parallel layouts** (e.g. different TP/EP degrees), results must also stay the same as the degree of parallelism changes, which needs a fixed reduction tree and costs more. The DeepSeek-V4 recipe instead moves cross-GPU reduction out of the compared path (4.5).

Each dimension guards against one of two root causes: a different order of operations (parallel accumulation or reduction order follows scheduling) or a different computation path (the two sides, or different batch shapes, run different kernels or branches). Reaching 0 diff means first removing the static differences, then pinning dynamic path selection so it ignores load.

Besides these four dimensions, reaching diff = 0 takes a few more things:

- **The same numerical contract on both sides**: same quantization and rounding, same accumulation precision, same kernel arithmetic. Two engines can satisfy all four dimensions and still disagree if they compute differently.
- **The same discrete choices**: Top-K and MoE routing turn a one-ulp upstream difference (one unit in the last place of a floating-point number) into a different token set or expert combination. Once upstream matches, tie-breaking, output order and merging rules must match too.
- **1 update per step (step batch equals mini-batch)**: with several mini-batch updates per step, from the second on, training weights differ from sampling weights, which diff = 0 does not cover, so training stops being strictly true on-policy (on a math task with 4 updates per step, about 7% of tokens switched expert combination after each update).
- **A correct backward and restore**: the four dimensions give only a consistent forward. Backward must still be correct, and the first step after checkpoint restore must still be consistent.

The main operators in Section 4 each meet one of these requirements: dense GEMM pins kernel selection for batch independence (4.1); the FP8 KV cache unifies the prefill and decode reads for phase consistency (4.2); Top-K fixes ties and order so discrete choices match (4.3); the expert GEMM keeps one numerical contract across two layouts (4.4); and cross-GPU reduction moves off the compared path for cross-GPU consistency (4.5).

**diff = 0 says the two sides compute the same thing, not that it suits training**: with several updates per step, diff = 0 holds while routing drifts between updates. diff = 0 is our engineering target, not a sufficient condition for training quality, which needs separate acceptance.

## 3. Reference recipe: DeepSeek-V4-Flash

DeepSeek-V4-Flash combines low-precision compute, sparse attention and multi-node expert parallelism, making it a challenging configuration. From the FP8 base we trained two 100-step runs on GB200: an optimized high-performance baseline and a consistent version with diff = 0.

Task: math reasoning with DAPO-format data, GRPO, and a reward with an over-length penalty (so it can be negative). Each run uses 8 machines (32 GPUs), 128 prompts × 8 responses per step, and 1 update per step (step batch = mini-batch). Neither run uses TIS, MIS or similar corrections, exposing the engine difference directly.

Parallelism: DP data, EP expert, CP context (sequence split), TP tensor. Consistent-version rollout runs 4 vLLM engines, each DP8/EP8, TP=1; training uses EP32, CP4, TP=1. **The two sides use different parallel layouts** (4.5).

Relevant structures: **MLA** (multi-head latent attention, which compresses KV before caching); **DSA** (DeepSeek sparse attention: a lightweight indexer scores history positions, Top-K keeps at most 2048, and MLA attends only to them); the **CSA compressor** (compresses history KV into fewer rows); **mHC** (mixing across residual streams, with the mixing matrix normalized by Sinkhorn iterations); and 43 MoE layers. SM100 is Blackwell (the GB200 used here); SM90 is Hopper.

### 3.1 Core decision: inference kernels define the values, training adds backward

Two engines agree most easily by running the same code; the question is which side adapts. Rollout dominates RL time, with every operator tuned for throughput; making training the reference would give up those optimizations. Thus, **inference kernels are the numerical reference: training forward calls the same vLLM kernels for its values, and training only adds backward.** Inference does not switch to generic batch-invariant kernels either: it keeps the high-performance deployment kernels and modifies them to meet the Section 2 requirements. Every operator in Section 4 follows this trade-off.

The challenge with this decision is that inference kernels have no backward, and FP8 quantization is not differentiable. So forward and backward are handled separately, and forward values always come from the vLLM kernel. Each module wraps the kernel in a custom `autograd.Function` whose backward differentiates against the BF16 master weights, with FP8 quantization treated as identity (the straight-through estimator, STE). Backward mostly reuses existing kernels: Transformer Engine BF16 GEMMs for linear layers and expert GEMMs, the cuDNN frontend DSA backward for sparse attention, and DeepEP's backward for the all-to-all; only small operators such as mHC, RMSNorm and SwiGLU have hand-written FP32 derivatives. A few fused kernels have outputs that cannot be differentiated this way (attention q/k, compressed KV, routing weights, the final logprob); for these we rerun the same computation in PyTorch only for its gradient and keep the kernel's values: `visible + (graph - graph.detach())`. Each backward first checks that the master weights have not changed since forward, so recomputation or asynchronous updates cannot mix weight versions.

All the costs land on training, in two kinds. **Engineering and verification:** the gradient is a surrogate, not the quantized operator's exact derivative, so a bitwise-identical forward proves nothing about gradients and the skill checks backward against an independent reference (Section 5); backward only has to be correct, not deterministic. Each layout entry point (e.g. masked / contiguous in 4.4) also needs its own verification. **Performance and memory:** training forward must save the inference kernels' inputs and outputs for backward; the memory pressure is why the recipe enables full recomputation, which runs forward twice.

### 3.2 Results and who pays

![DeepSeek-V4-Flash 100-step training reward](/assets/figures/2026-10-14-train-inference-consistency/ds4-reward.png)

*Per-step training reward. Orange: baseline; blue: consistent version. Thick line: moving average of the current and previous 4 steps; thin line: raw value.*

![DeepSeek-V4-Flash per-step train-inference difference](/assets/figures/2026-10-14-train-inference-consistency/ds4-train-infer-mismatch.png)

*Top: per-step max train-inference probability difference; bottom: train-inference KL (k3 estimator r − 1 − log r, used throughout; symmetric log scale, linear near zero). Max difference reacts to one routing flip or Top-K swap, so it is sizable from baseline step 1; KL shows the overall level. Orange: baseline; blue: consistent version. Both consistent-version metrics are 0 at every step.*

Over 100 steps, the consistent version's rollout averages 24.22 token/ms, 91.3% of the baseline (26.53):

![DeepSeek-V4-Flash rollout throughput](/assets/figures/2026-10-14-train-inference-consistency/ds4-rollout-throughput.png)

*Per-step rollout throughput (output token/ms, summed over 32 GPUs). Orange: baseline; blue: consistent version. Thick line: moving average of the current and previous 4 steps; thin line: raw value.*

Baseline train-inference KL grows (2e-3 to 29.9) while its reward falls back; the consistent version stays at diff 0 and its reward holds. Another potential gain is one fewer forward: we recompute old-logprob each step only to measure diff; after acceptance, training can use the rollout logprob instead.

**Rollout side**, measured in this run: average throughput is 91.3% of baseline (about 9% lower), above Section 1's 70% floor. After step 90 the baseline's throughput drops, which inflates the ratio; over the same steps its mean response length falls from about 3,000 to about 1,600 tokens. the first 89 steps give 88.6%. The two runs' response lengths differ throughout training, so the ratio also reflects different workloads, not only the engines.

This gap is not yet attributed to operators; Section 4 points to MoE LL communication and backend, attention gather and dequantization, and per-request launches.

**Training side**, not measured separately: per Section 4, overhead is requantizing FP8 weights every forward, serial per-request attention and Top-K, one extra routing-metadata dispatch, and full recomputation (3.1), which runs forward twice.

A separate verl recipe PR preview on 32 GB200 GPUs reports about 10% end-to-end overhead: rollout near parity, the gap mostly in training. Training-side optimization is next.

## 4. Operator by operator: technical choices in the DeepSeek-V4 recipe

Each subsection covers what is kept, what is given up, how speed is recovered, and what cost remains. Which kernel each side uses per operator, and how backward is written, are in the [skill](https://github.com/vllm-project/vllm/pull/61551).

### Per-operator master table

Cost sits mostly in MoE communication and backend (4.4, 4.5), then attention gather and dequantization (4.2, 4.3); dense is cheap.

| Module | Kept | Given up | Cost |
|---|---|---|---|
| Dense GEMM (4.1) | Deployment W8A8 and DeepGEMM; one accumulation order per output | FlashInfer branch at M<32, small-batch gate kernel, fused mHC kernel | rollout: no faster small-batch kernel; training: weights requantized every forward |
| FP8 KV cache (4.2) | FP8 storage | Decode kernel that reads FP8 directly | rollout: extra gather and dequantization in decode; training: one private cache |
| Top-K (4.3) | DSA sparse attention | Default Top-K tie-breaking and output order | One extra sort per side, recovered by fusion; multiple per-request prefill launches; training: attention and Top-K serial per request |
| Expert GEMM (4.4) | FP8 grouped GEMM; routing matches by construction, no R3 | FlashInfer TRT-LLM MoE | rollout: a main throughput gap; training: alignment padding |
| Cross-GPU reduction (4.5) | Different parallel layouts for training and rollout | TP>1 | rollout: LL communication cannot overlap compute; training: one extra routing-metadata dispatch |
| CUDA Graph (4.6) | Graphs | Performance-only shortcut branches | Workspaces reserved at their upper bound |
| Others (embedding, RMSNorm, o_proj, lm_head + logprob) | Existing kernels, fixed reduction order | — | Small |

### 4.1 Dense GEMM: keep FP8, one accumulation order for every M

**Kept.** DeepSeek-V4-Flash-Base ships in block-FP8, and the deployment path is W8A8 with DeepGEMM. The consistent version uses it on both sides: each training forward quantizes the BF16 master weights to block FP8 by the deployment rule and calls the inference GEMM. A BF16 rollout would cost compute and bandwidth; a separate training-side scheme such as MXFP8 only gets close to the deployed numerics, the limitation of the QAT row in Section 1.

**Given up.** The inference engine picks kernels by shape: the FP8 linear layer uses FlashInfer when M<32 and DeepGEMM otherwise; the router gate switches to a small-batch kernel at 16 tokens or fewer; the mHC split count grows with token count. Floating-point addition is not associative, so each makes a request's result depend on its neighbors. The consistent version removes these branches: the FP8 linear layer always uses DeepGEMM in batch-invariant mode, the router gate uses one FP32-output GEMM at every batch size, small BF16 matmuls use a Triton kernel with fixed summation order, and the mHC split is fixed at 1. We unify on DeepGEMM rather than FlashInfer because DeepGEMM has a batch-invariant mode, is already the deployment kernel, and training M is large.

![Dense GEMM: which kernel the same request uses in steps of different sizes](/assets/figures/2026-10-14-train-inference-consistency/ds4-dense-gemm.png)

*Horizontal axis: tokens per step (schematic). The default path switches kernels at 32, so the same request in an 8-token and a 200-token step goes through FlashInfer and DeepGEMM and differs in the last bit. Batch-invariant mode has one summation order.*

**Cost.** Rollout loses faster small-batch kernels and one mHC fusion, but Tensor Cores stay in use. Training requantizes weights every forward, including recomputation; caching the FP8 weights saves that time for one more resident copy.

### 4.2 FP8 KV cache: keep FP8 storage, unify the read path

**Kept.** The KV cache is stored in FP8, about 57% of BF16, which directly sets concurrency. Both runs share the write path.

**Given up.** Default decode uses a FlashMLA kernel that reads FP8 directly and partitions work by batch; prefill dequantizes to BF16 and runs sparse attention. The two phases run different algorithms, and decode's partition depends on its batch. The consistent version routes decode through the prefill path: dequantize and gather the selected KV into a BF16 workspace, then call the same `flash_mla_sparse_fwd`. The workspace is per request and sized only by model limits. This decode kernel must be selected explicitly; the consistency switch alone does not change it.

![FP8 KV cache: prefill, decode and training share one read path](/assets/figures/2026-10-14-train-inference-consistency/ds4-fp8-kv.png)

*Top: before, prefill and decode read the same FP8 history with two algorithms, and decode's partition depends on the batch. Bottom: prefill, decode and training (with a private cache copy) share one read path.*

Training has no real cache, but K must make the same FP8 round trip: the same fused kernel writes it into a private FP8 cache, and the same dequantization reads it back.

**Cost.** Decode adds gather and dequantization, and training adds a cache. This is the price of phase consistency; storage precision is untouched. Each request currently generates one token per step; multi-token prediction (MTP) and speculative decoding are not covered.

### 4.3 Top-K in sparse attention: fix ties and order

**Problem.** The DSA indexer scores every history position, and Top-K picks at most 2048 for attention. This discrete choice turns a one-ulp upstream difference into a different context. Scoring is aligned first: the indexer uses the same DeepGEMM kernel, prefill reads contiguous KV and decode reads paged KV, and the two reads are bitwise identical.

![Top-K: same scores, same selection, same order](/assets/figures/2026-10-14-train-inference-consistency/ds4-topk.png)

*Top: layout of the 64-bit selection key. Bottom: an example with K = 2 where positions 17 and 42 tie and the position bits in the key select 42; selected positions are then sorted by descending position, invalid entries last.*

**Fix selection and order.** Default Top-K breaks ties by thread scheduling. The consistent version combines score and in-request position into a unique key, so ties go to the larger position. It uses the in-request position, not the global batch position; otherwise moving a request within the batch would change tie results. The same set is not enough: index order sets attention's accumulation order, so selected indices are sorted by descending position, invalid entries last.

The R3-style alternative records rollout's indices and replays them in training, but that is up to 2048 indices per token per layer: GB-scale per sequence for long responses.

**Training side.** The training forward selects the same positions with the same scoring and Top-K, but the indexer is not trained, mainly for simplicity. This matches existing practice: [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) detaches the indexer's input when training DSA, so the RL loss never reaches it, and [GLM-5](https://arxiv.org/abs/2602.15763) freezes the indexer by default in RL.

**Cost and recovery.** Each side does one extra sort. The deterministic Top-K takes about 5.9 µs per call; fusing it with the index sort raises the consistent version's inference-benchmark throughput by 6.7% (configuration in 4.4). The remaining cost is multiple per-request launches in prefill.

### 4.4 Expert GEMM: two layouts, same row result

**Kept.** The routed experts' FP8 grouped GEMM dominates the MoE layer, and the two sides feed it different layouts. Rollout uses DeepEP low-latency (LL) mode (why: 4.5), giving a masked layout chunked by expert with fixed capacity per chunk. Training processes many tokens with DeepEP normal mode, giving a contiguous layout packed by expert.

**Backend choice.** Both consistent sides use DeepGEMM in batch-invariant mode: it is open source, has a batch-invariant mode, and is the high-performance deployment kernel. The baseline's FlashInfer TRT-LLM MoE ships as precompiled cubins, so we cannot modify it or verify whether it is batch-invariant; cutlass lacks block-FP8, and triton falls into a batched implementation that does not declare batch invariance. A single expert GEMM takes about the same time in both; the baseline is faster because TRT-LLM MoE overlaps communication with compute in a multi-stream pipeline.

**Same row result from two layouts.** Three conditions must hold:

1. **Experts receive the same FP8 input.** LL quantizes at dispatch and normal mode after it, originally with different rounding; we unified both on elementwise round-to-nearest.
2. **Each row's result is independent of layout.** Quantization is per token and batch-invariant DeepGEMM accumulates each row in a fixed way, so a row's result does not depend on its neighbors. Bitwise router + MoE tests confirm this.
3. **The activation is rounded once.** SiLU(gate) × up between the two GEMMs is computed in FP32 and only the product is rounded to BF16, then quantized to FP8. Rounding SiLU to BF16 before the multiply can differ by one ulp, which FP8 quantization amplifies into a code flip.

![Expert GEMM: two layouts, the same row result](/assets/figures/2026-10-14-train-inference-consistency/ds4-expert-gemm.png)

*The same token (orange) sits at different positions in rollout's masked and training's contiguous layouts, yet the three conditions make its expert GEMM result bitwise identical. Gray: empty slots; hatching: alignment padding.*

**No R3 needed.** The router's scoring GEMM uses one FP32-output kernel at every batch size on bitwise-identical inputs, so recomputed routing matches rollout. This assumes 1 update per step (Section 2).

**Cost.** Rollout gives up TRT-LLM MoE, but the GEMM is not most of the gap. On the inference benchmark (GB200, DP8/EP8, 256 concurrency, 150 input / 3000 output per request) the consistent version started 33% slower than the baseline, about 83% of it from non-overlappable LL communication (4.5). Training saves intermediates for backward and pads each expert's rows. Inference-benchmark numbers below come from different optimization stages; compare them only within a sentence.

Before comparing performance, also confirm which kernel the baseline runs: vLLM enables the batch-invariant SiLU·mul·quantization kernel whenever its shared library is present, ignoring the consistency switch, so any image shipping the library runs it in the baseline too.

### 4.5 Cross-GPU reduction: off the compared path

**Problem.** Rollout wants low latency and uses DP8/EP8; training must fit long-sequence activations and uses EP32 plus CP4. As Section 2 noted, different layouts need results invariant to parallelism degree, usually via a fixed, expensive reduction tree.

**Choice.** This recipe **keeps cross-GPU reduction out of the numerical path being compared**, and the remaining communication moves data without changing values:

- **TP=1.** Linear layers have no all-reduce. vLLM's all-reduce picks algorithms by message size, so TP would bring that dependence back; the recipe rejects TP>1 at startup.
- **EP merge is local.** The all-to-all only moves data. Expert outputs return to the token's owning rank and are accumulated in FP32 in top-k slot order, independent of EP=8 vs 32 or which rank answers first.
- **CP does no reduction.** Neighboring ranks exchange only boundary hidden states, and compressed KV is all-gathered in the CP group: concatenation, not summation.

![Different parallel layouts, bitwise identical results](/assets/figures/2026-10-14-train-inference-consistency/ds4-cross-rank.png)

*One token's path: the all-to-all only moves data, and the only summation happens on the token's owning GPU, in FP32 in top-k slot order, so EP8 and EP32 give the same result.*

**Making the data movement deterministic.** In rollout, each rank sends at most a fixed number of tokens per call and all ranks segment in lockstep, so the communication shape is fixed. Only LL implements this, and DeepEP normal mode disables CUDA Graph when DP>1, so rollout must use LL. In training, normal mode has no fixed arrival order, so routing metadata is dispatched once more and a bitwise hash matches each expert output to its token and slot.

**Cost and recovery.** LL mode runs 43 layers × 2 = 86 serial all-to-alls per step that cannot overlap compute. The cross-layer overlap schemes we tried compute before inputs are ready and break batch independence. Interleaving two micro-batches (DBO) was 39% slower: each layer's expert weights are read twice, and weight reads already use 84% of HBM bandwidth.

We eventually closed the communication gap on the inference benchmark. Confirm both runs use the same transport first: we lost two days because a script pinned the consistent version's DeepEP to Ethernet RDMA (RoCE) while the baseline used the much faster MNNVL. With matched transport and the shared expert on an auxiliary stream, the consistent version was 3.2% faster than the baseline (16834 vs 16313 tok/s).

TP>1 and decode context parallel would put cross-GPU reduction back on the compared path and are not supported yet.

### 4.6 CUDA Graph: keep graphs and make them coexist with consistency

**Why keep them.** Decode runs dozens of small kernels per layer per step at one token per request, so launch overhead would dominate; we keep CUDA Graph to hit the performance target. Graphs conflict with bitwise consistency in three places.

![Three conflicts between CUDA Graph and bitwise consistency](/assets/figures/2026-10-14-train-inference-consistency/ds4-cuda-graph.png)

*The path each mode takes for three kinds of step, with inference-benchmark throughput on the right. A request switches between graph and eager depending on its batch, so both paths must be bitwise identical.*

**Conflict 1: graph or eager.** A step runs in the graph if its total token count fits the max capture size and in eager otherwise, which depends on the other tokens in the step, so the two paths must be bitwise identical. That was our first mismatch: an indexer short-context shortcut was baked into the graph at capture based on the dummy input's length. Making the check depend only on a static upper bound fixed it.

The mode matters too. PIECEWISE does not capture attention, and FULL on DeepSeek-V4 captures only pure decode steps. On the inference benchmark FULL_AND_PIECEWISE gives 16409 tok/s, FULL 16068 and PIECEWISE 10423; FULL runs every step mixed with prefill in eager, raising time to first token from 764 ms to 1792 ms. We use FULL_AND_PIECEWISE: full graphs for pure decode steps, piecewise capture for mixed steps; diff stays 0 across the 100-step run.

**Conflict 2: no host sync or batch-dependent shapes in the graph.** This aligns with consistency, and many designs serve both: the MoE scheduling hint uses static capacity instead of reading per-expert token counts back to the host (4.4); LL fixes the communication shape (4.5); workspaces are sized only by model limits, allocated before capture and then locked; performance-only shortcut branches are disabled at capture time.

**Conflict 3: graphs capture addresses, weights change every step.** Rollout must copy new weights into their original storage and recompute weight-derived parameters in place, rather than allocate new memory that leaves the graph on old addresses. Matching at the first step is not enough, so the skill compares again after update, export and reload.

**Cost.** Workspaces reserved at their upper bound cost some memory. Other speed recoveries (removing needless debug padding and cold start from rollout, faster KV cache zeroing) apply to both runs; the baseline must get them too, or the performance-loss number is meaningless.

## 5. Lowering engineering cost with Agent + skill

A and B can coexist, but the Section 4 master table and the key choices in it take expert judgment: which backend is batch-invariant, that discrete choices need aligned ties and order, how to align quantization and merge order across different communication modes, and that a communication gap first requires ruling out a transport difference. A new model, precision or hardware means starting over, so C does not hold.

After DeepSeek-V4-Flash, we saw that most of this work falls into two categories, each with a measurable goal:

- **Bring diff to 0.** Fix weights and input, find the first operator where the two sides diverge, reduce it to a minimal rerunnable repro, fix it, then verify in real training. Outputs are compared directly.
- **Optimize performance while keeping diff at 0.** Profile hotspots, A/B single operators, and re-verify consistency after every change. Speedups that change results or that show up only in micro benchmarks, not real rollout, do not count; slowing the baseline to narrow the gap is not allowed.

Reading code, tracing call chains, locating divergences, writing repros, changing kernels and rerunning regressions is long-running work suited to an agent, with success decided by consistency comparison and performance measurement.

We packaged this as the [train-inference-consistency skill](https://github.com/vllm-project/vllm/pull/61551): it tells an agent what to do in what order and what each step must reach to pass, so the process carries over to other models.

The skill has 11 gates (C0–C10) with explicit pass conditions. To speed up cold start, we **first cut a few layers from the full model into a small model, solve most problems there, then move to the full model**. Five steps:

1. **Set the target and baseline (C0–C1).** List the model's modules with their precision and parallelism. Tune the ordinary, non-consistent version (the baseline) to normal high performance for human review. It is the denominator for "how much slower the consistent version is".
2. **Make inference self-consistent on a small model (C2–C4).** The small model (usually 4–6 layers picked from the full model) must contain every operator class, keep adjacent-layer interaction, and match per-layer size and quantization, so it reproduces most local numerical problems. Full depth, long context, multi-GPU layout and memory pressure must still be accepted on the full model. First make the inference engine internally consistent: a request gives the same result alone and in a batch, and whole-input processing matches token-by-token. Meanwhile, look for structural problems that would slow the consistent version.
3. **Connect training (C5–C7).** Training calls the same operators as inference, recomputes inference-generated data, and compares token by token; on a mismatch, compare layer by layer for the first divergent layer. Then add backward and check gradients against a reference implementation independent of the code under test: a consistent forward does not imply correct gradients.
4. **Run RL, then switch to the full model (C8–C9).** On the small model, run two consecutive updates in the real training framework, comparing under the same weight version before and after each. After human review, switch to the full model and scale up stepwise, including checking the first step after checkpoint restore.
5. **Recover performance (C10).** Optimize the most expensive operators one by one, rechecking consistency each time, then run a paired comparison against the baseline on the delivery image.

Two designs matter most. First, **two agents advance the baseline and consistency in parallel**: one makes the baseline fast, the other builds the consistent version. The tuned baseline is frozen for human review, so the agent cannot slow it to narrow the gap. Second, **our pitfalls are written into the process**: the running program must match the source being read; checkpoints need the first post-restore step verified, not just a save; every performance conclusion needs a fair baseline and traceable raw data; before comparing communication bottlenecks, confirm both runs use the same transport (the lesson from 4.5).

The skill lowers the cost of localization and regression (savings not yet quantified here); it does not replace human judgment: humans still review the baseline, precision trade-offs and recipe. An example is the routing drift with several updates per step: the agent built a small-scale repro from the symptom, ran discriminating experiments and eliminated hypotheses, and a human decided to move to 1 update per step.

## 6. Transfer: Nemotron, a Mamba hybrid architecture + NVFP4

We then tested the reference recipe and skill on a structurally very different model.

**Nemotron-3.5-Lightning-30B-A3B (NVFP4)** is a Mamba-2 / attention / MoE hybrid (NVFP4 W4A16 routed experts, FP8 Mamba projections, FP8 KV cache for attention) on GB200. The baseline uses NVFP4 QAT training (quantization rule and export from NVIDIA ModelOpt) + R3 route replay; the consistent version shares vLLM's forward, without route replay. Both runs use BF16 master weights and 1 update per step.

Recurrent structures have precedent. In the vLLM community, Alexander Jiang and the SkyRL team designed chunkwise-parallel recurrent (CPR) in [IsoExec](https://vllm.ai/blog/2026-08-21-isoexec) for the Gated DeltaNet (GDN) in Qwen3.5-35B-A3B: it keeps the recurrent form but computes chunk-parallel, so training, prefill and decode share one rounding order, and its **unified model** gives training and inference one model definition and one set of aligned kernels. IsoExec's "zero mismatch" holds within its execution contract; end to end it still reports a tiny nonzero logprob difference (mean absolute difference around 1e-7), unlike this post's "bitwise identical". Mamba-2 is the same kind of linear recurrence. We make decode exactly replay prefill's chunked arithmetic instead of its own single-step recurrence, and training imports the same operators as inference, as in the unified model. Agent + skill did this alignment.

In vLLM, every operator that might depend on batch composition gets a batch-invariant implementation, and training (a lightweight Megatron-LM fork) imports them:

| Operator | Consistent version |
|---|---|
| Mamba-2 SSD scan | decode exactly replays prefill's chunked arithmetic, fixed autotune config, fp32 state |
| Attention | FlashAttention with fixed split-KV, split count independent of batch |
| Routed experts (NVFP4 W4A16: 4-bit weights, 16-bit activations) | CuTe-DSL grouped GEMM, single tactic, no split-K, fp32 merge in slot order |
| MoE all-to-all | one-sided communication, fixed merge order |
| Router GEMM | Batch-invariant persistent matmul, fp32 output |
| Norm / FP8 quantization | Unified batch-invariant kernels; value-changing fusions forbidden |

### Results: training on SWE-bench Pro

We run agentic RL on SWE-bench Pro in remote sandboxes: 32 tasks × 8 trajectories per step, about 57K tokens per trajectory on average, GRPO, 8 GB200 nodes per run. Both runs use the same image and recipe, without corrections such as TIS or MIS. With 1 update per step, the KL between pre-update rollout logprobs and the training-side recomputation is that step's train-inference KL.

![Nemotron SWE-bench Pro: train-inference KL, training score and entropy](/assets/figures/2026-10-14-train-inference-consistency/nemotron-swe.png)

*Top: per-step train-inference KL; middle: training score (moving average over the current and previous 4 steps); bottom: actor entropy. Orange: baseline; blue: consistent version. At consistent-version step 10 only a few trajectories returned from the sandbox, so its score is excluded.*

- **Consistent version**: train-inference KL is exactly 0 every step, entropy stays around 0.05, and the score goes from 0.56 (first 10 steps) to 0.59 (last 10), with no drop.
- **Baseline**: train-inference KL falls from ~0.015 to ~0.002, but entropy rises from 0.05 to 2.6 and the score drops from 0.56 to 0.22 (first 10 steps vs. last 10).
- **rollout speed**: over the first 30 steps, while both runs are healthy, the consistent version's rollout throughput is 87.5% of the baseline's, only 12.5% slower.

![Nemotron SWE-bench Pro: rollout throughput](/assets/figures/2026-10-14-train-inference-consistency/nemotron-swe-rollout.png)

*Per-step rollout throughput (token/ms, moving average over the current and previous 4 steps). Orange: baseline; blue: consistent version.*

## 7. Open source

We open-source this so others can reach train-inference consistency with less reliance on senior engineers checking operator by operator, using the skill, ready-made code and an agent on their own model. The release:

- **[train-inference-consistency skill](https://github.com/vllm-project/vllm/pull/61551)**: the full workflow (C0–C10), ten guides, and key practices such as logprob comparison and gradient cross-checking;
- **DeepSeek-V4**: a true on-policy RL recipe reaching 0 diff on Megatron and vLLM, [verl-project/verl#7688](https://github.com/verl-project/verl/pull/7688);
- **Nemotron-3.5-Lightning-30B-A3B** (BF16 master weights, quantized to the deployment NVFP4 in the training forward and at weight sync):
  - vLLM bitwise-consistent inference aligned with training: [aoshen02/vllm#85](https://github.com/aoshen02/vllm/pull/85);
  - A training actor aligned with vLLM: [aoshen02/Megatron-LM#7](https://github.com/aoshen02/Megatron-LM/pull/7);
  - Two verl changes: [aoshen02/verl#11](https://github.com/aoshen02/verl/pull/11);
  - Train-inference consistency recipe and unified image Dockerfile: [aoshen02/verl#12](https://github.com/aoshen02/verl/pull/12).

Issues, PRs and feedback from using the skill on your own model are welcome.

Next: train-inference consistency for DeepSeek-V4.1 has reached 0 diff; performance optimization is in progress and will get its own post.

## Acknowledgments

Thanks to (in alphabetical order): [Jian Hu](https://hijkzzz.notion.site/) of NVIDIA Research, [Mingrui](https://github.com/mingruimingrui) of TikTok, [Nolen Liang](https://github.com/NolenLiang) and [Tomer](https://github.com/TomerBN-Nvidia) of the Nemotron team, the [slime](https://github.com/THUDM/slime) community, [Verda](https://verda.com/) for sandbox support, the [verl](https://github.com/verl-project/verl) team, and the vLLM community.
