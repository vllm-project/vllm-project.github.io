---
layout: post
title: "Parallel All the Way Down: Beyond Single-Token Generation with Speculative Decoding"
author: "Alexandre Marques, Megan Flynn, Helen Zhao, Krishna Teja Chitty Venkata, Chibu Ukachi (Red Hat AI)"
summary: "Speculators and vLLM now support P-EAGLE, DFlash, and DSpark — three parallel drafting algorithms that move beyond sequential token generation to deliver faster, simpler, and more scalable speculative decoding for LLM serving."
image: /assets/figures/2026-07-24-speculator-parallel-drafting/ar_vs_parallel.jpg
tags:
  - speculators
  - speculative_decoding
  - peagle
  - dflash
  - dspark
---


# 1. Introduction

Speculative decoding has emerged as a core optimization technique for mitigating memory-bandwidth bottlenecks in Large Language Model (LLM) serving. By validating multiple candidate tokens in a single verifier-model forward pass, it allows production systems to achieve substantial inference speedups.

However, as serving infrastructure evolves, traditional speculative frameworks face a structural ceiling rooted in the way draft tokens are generated. Today, we are excited to showcase how [Speculators](https://github.com/vllm-project/speculators) and [vLLM](https://github.com/vllm-project/vllm) are moving beyond these limitations by providing full open-source support for three state-of-the-art parallel drafting algorithms: [P-EAGLE](https://arxiv.org/abs/2602.01469), [DFlash](https://arxiv.org/abs/2602.06036) and [DSpark](https://arxiv.org/abs/2607.05147).


<p align="center">
<div class="artifact-image-grid">
<img src="/assets/figures/2026-07-24-speculators-parallel-drafting/compare_interactivity_qwen38b_math.png" width="31%">
<img src="/assets/figures/2026-07-24-speculators-parallel-drafting/compare_interactivity_qwen330b_humaneval.png" width="31%">
<img src="/assets/figures/2026-07-24-speculators-parallel-drafting/compare_interactivity_gemma431b_humaneval.png" width="31%">
</div>
<br>
<em>Figure 1. Parallel drafting algorithms, such as P-EAGLE, DFlash and DSpark, provide significant performance gains when compared to autoregressive drafting algorithms such as EAGLE-3. Speculator models mentioned above can be found in the [Speculators Collection](https://huggingface.co/collections/RedHatAI/speculator-models) at the RedHatAI HuggingFace Hub.</em>
</p>
  
# 2. The Limits of Recursive Drafting

The introduction of frameworks like [EAGLE](https://arxiv.org/abs/2401.15077) and [MTP](https://arxiv.org/abs/2404.19737) marked a major paradigm shift in speculative decoding. Instead of forcing the speculator model to guess blindly from surface-level text, EAGLE demonstrated that a speculator architecture could tap directly into the verifier model’s rich internal hidden states, dramatically increasing token acceptance rates.

Despite this breakthrough, advanced iterations like [EAGLE-3](https://arxiv.org/abs/2503.01840) still operate under a fundamental constraint: **auto-regressive drafting**. To propose a sequence of candidate tokens, the speculator architecture must generate them sequentially, executing a separate forward pass for every single token.

This auto-regressive design introduces two major trade-offs in production:

- **Constraints on Model Size:** Because the drafting cost scales linearly with the speculation length, speculator models are forced to remain extremely small and lightweight to avoid consuming the execution time saved during verifier-model verification.  
- **Complex Operational Tuning:** Linear scaling heavily limits the number of drafted tokens in practice. Choosing the optimal speculation length (K) becomes a sensitive variable that engineering teams must constantly adjust depending on the specific use case and real-time server loading.

<p align="center">
<img src="/assets/figures/2026-07-24-speculators-parallel-drafting/ar_vs_parallel.jpg" width="100%">
</p>

# 3. The Shift to Parallel Drafting

Parallel drafting fundamentally re-engineers this trade-off by eliminating sequential execution from the drafting phase entirely. Rather than looping through single-token generation steps, parallel drafting algorithms predict an entire candidate block of tokens concurrently.

By flattening the drafting phase into a single forward pass, the latency of generating proposals is decoupled from the number of tokens speculated. This architectural shift simplifies production serving in two distinct ways:

- **Capacity for Expressiveness:** Because the speculator model only runs once per block, developers can utilize larger, more robust, and more expressive draft architectures. These deeper speculator models capture more complex context and yield higher acceptance rates without introducing a sequential latency penalty.

- **Simplified Parameter Tuning:** Decoupling drafting cost from block length removes he operational burden of hyper-tuning speculation parameters based on fluctuating server loads.

Parallel drafting as a concept has been explored before — [Medusa](https://arxiv.org/abs/2401.10774) and [PARD](https://arxiv.org/abs/2504.18583) are notable earlier examples. P-EAGLE, DFlash, and DSpark build on this foundation by combining parallel execution with deep verifier-state conditioning, the insight that made EAGLE so successful.

# 4. Under the Hood: Inference & Training Architecture

**P-EAGLE**, **DFlash**, and **DSpark** all build upon the verifier model's hidden states to generate draft tokens in parallel, but each takes a different path to get there. Below we outline what makes each architecture unique.

A shared challenge across all three is training. Any parallel speculator must perform next-K prediction at every token position along a training sequence. For a sequence of length N and a lookahead window of K, naively computing losses across the full matrix causes memory and compute costs to scale prohibitively. Each algorithm addresses this differently.

## **P-EAGLE**

P-EAGLE builds directly on EAGLE's foundation of using the verifier model's hidden states as input features. Instead of consuming those features to predict tokens sequentially, P-EAGLE maps them across multiple future positions simultaneously, outputting an entire sequence of candidate tokens in a single parallel step.

To keep training tractable, P-EAGLE implements draft block sparsification: it drops tokens along the lookahead dimension (K) according to a decaying rate, concentrating optimization on the most critical immediate tokens while pruning distant future positions from the loss calculation.

## **DFlash**

DFlash routes verifier features differently. Rather than feeding hidden states in as standard inputs, DFlash projects them and injects them directly into the KV-cache of the speculator model. This tightly conditions the speculator's attention mechanism on the verifier's exact state without expanding the input sequence length, enabling it to generate a highly accurate block of candidate tokens via block diffusion.

For training, DFlash implements sequence length sparsification. Instead of calculating block loss at every token position across a sequence of length N, it selects random anchor points along the timeline and computes block predictions exclusively at these intersections — preserving GPU memory while maintaining representative coverage.

## **DSpark**

DSpark takes DFlash's parallel backbone and layers two additional innovations on top. First, it augments the architecture with a lightweight autoregressive correction head, allowing future tokens to be more strongly conditioned on past tokens. This combines the throughput benefits of parallel generation with the sequential coherence of autoregressive refinement.

Second, DSpark addresses a downstream bottleneck: verification cost. Parallel drafting can generate many draft tokens inexpensively, but the verifier must still process all of them. DSpark introduces a confidence head that scores draft tokens before they reach the verifier, selectively forwarding only those likely to be accepted. This reduces wasted verification compute and improves end-to-end throughput.

# 5. Inference Performance

Figure 1 illustrates the performance gains provided by parallel drafting algorithms when compared to EAGLE-3. Three distinct models and parallel drafting algorithms are displayed:


| Model          | Algorithm                                                                  | Use case               | Hardware |
| -------------- | -------------------------------------------------------------------------- | ---------------------- | -------- |
| Qwen3-8B       | [P-EAGLE](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.peagle)      | Math reasoning (GSM8k) | 1xA100   |
| Qwen3-30B-A3B  | [DFlash](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-speculator.dflash)  | Coding (HumanEval)     | 2xA100   |
| gemma-4-31B-it | [DSpark](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dspark) | Coding (HumanEval)     | 2xA100   |


In all cases, parallel drafting shows significant improvement over EAGLE-3. Performance will vary across models, tasks, and hardware configurations — we encourage the community to benchmark on their own workloads.

# 6. Production Serving with vLLM and Speculators

Integrating state-of-the-art parallel drafting algorithms into production requires a stable, optimized infrastructure stack. The Speculators repository provides a unified ecosystem to train and evaluate these next-gen models, fully integrated with **vLLM**.

Launching a parallel-backed speculative engine is as straightforward as passing the appropriate configuration flags at initialization:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 2 \
  --reasoning-parser qwen3 \
  --speculative-config '{
    "model": "RedHatAI/Qwen3-30B-A3B-speculator.dflash",
    "num_speculative_tokens": 7,
    "method": "dflash"
  }'
```

By moving from single-token generation to block-level parallel drafting, your inference pipeline becomes parallel all the way down—maximizing hardware utilization and delivering sustained, lossless acceleration. (Speculative decoding preserves the verifier model's output distribution exactly via rejection sampling, so quality is mathematically identical to standard decoding.)

# 7. Get Started

Parallel drafting is fully supported, open-source, and production-ready today. We invite the community to explore the repository, utilize our documented training pathways to build your own parallel speculators, and benchmark them natively in vLLM.

- Repository: [Speculators](http://github.com/RedHatAI/speculators)  
- Pre-trained speculators: [Speculators Collection on HuggingFace](https://huggingface.co/collections/RedHatAI/speculator-models)  
- Training guides: [Speculator tutorials](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/tutorials/index.md)

