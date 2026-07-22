---
layout: post
title: "A Preview of Production-Scale Kimi K3 Support on vLLM"
author: "vLLM Team"
summary: "A preview of production-scale Kimi K3 support in vLLM, including KDA-aware prefix caching, fused kernels, optimized MXFP4 MoE, multimodal integration, and initial NVIDIA and AMD paths."
image: /assets/figures/2026-07-22-kimi-k3-preview/social-preview.png
social_image: /assets/figures/2026-07-22-kimi-k3-preview/social-preview.png
tags:
  - models
  - performance
  - prefix caching
  - multimodal
---

Last week, Moonshot AI [introduced Kimi K3](https://www.kimi.com/blog/kimi-k3), a 2.8-trillion-parameter model with native vision support, a 1-million-token context window, Kimi Delta Attention (KDA), Attention Residuals (AttnRes), and a highly sparse Mixture-of-Experts architecture. The announcement immediately drew [global attention](https://x.com/Kimi_Moonshot/status/2077830229968683203), and the open-source community is extremely excited that open-weight models are advancing quickly to catch up with the best proprietary models.

Moonshot AI has announced that the full model weights will be released by July 27, 2026. In the meantime, vLLM, Moonshot AI, NVIDIA, AMD, and the broader community are working through the final integration and validation so the open-source community can serve Kimi K3 from day 0.

This post is a preview and performance optimization is ongoing, but the core model path, KDA-aware prefix caching, multimodal integration, tool calling parsers, and hardware-specific optimizations are already taking shape. Selected trusted partners, approved by both Moonshot AI and the vLLM/Inferact team, have also begun deployment validation using the same code that is being prepared for open source.

As stated in the announcement blog, KDA poses new challenges for conventional prefix caching, and the Moonshot AI team has contributed a corresponding implementation to the vLLM project, to be released alongside the model weights. We will dedicate a future blog post to explaining the design.

## TL;DR

- **Day-0 open-source serving:** vLLM is preparing model implementation, Docker images, deployment recipes, and production validation for the Kimi K3 weight release.
- **A new hybrid architecture:** Kimi K3 combines KDA-dominant linear attention with periodic full-attention layers, AttnRes across depth, Stable LatentMoE, and native vision support.
- **Prefix caching required core changes:** vLLM now separates the physical KDA state-block size from prefix-match granularity, enabling useful partial prefix-cache hits without storing recurrent state at every small attention block.
- **Kernel work across the stack:** the release branch includes FlashKDA integration, fused KDA decode, fused KDA projections and convolution, fused AttnRes, reimplemented MLA module, SiTU-enabled MXFP4 MoE execution, and optimized expert routing.
- **NVIDIA and AMD support:** NVIDIA-specific kernels are under final tuning, while an initial AMD implementation with a FlyDSL MoE kernel is already in place and moving through broader validation.

## Kimi K3 at a Glance

Kimi K3 is not a larger version of Kimi K2. Kimi K3 changes the serving problem in several dimensions at once.

| Property | Kimi K3 configuration | Serving implication |
| :---- | :---- | :---- |
| **Model scale** | **2.8T parameters** | Requires large-scale expert parallelism and high-bandwidth accelerator domains |
| **Context length** | **1M tokens** | Makes cache capacity, prefix reuse, chunked prefill, and prefill/decode disaggregation first-order concerns |
| **Attention** | **Hybrid KDA and full attention** | Requires both recurrent state caches and paged KV caches to advance on exactly the same logical prefix |
| **Depth** | **Attention Residual** | Adds cross-layer representation reads and writes that need dedicated kernels |
| **MoE** | **896 routed experts, 16 active per token, plus shared experts** | Makes routing, dispatch, load balance, and MoE kernels central to end-to-end performance |
| **Quantization** | **MXFP4 weights in the provided release configuration** | Needs an efficient FP4 MoE path with Kimi K3’s SiTU activation |
| **Multimodality** | **Native vision with a vision tower** | Requires multimodal preprocessing (image-only) and a robust vision parallelism strategy |

For inference systems, each of these choices moves cost somewhere new. KDA reduces the need to retain a conventional KV pair for every past token, but introduces a large recurrent state. AttnRes reduces the limitations of a single residual stream, but creates additional cross-layer memory traffic. Extreme MoE sparsity avoids activating all 2.8T parameters for every token, but raises the stakes for routing and communication. vLLM's job is to make all of these pieces work together behind one familiar serving API.

## A Collaboration Built Over Multiple Kimi Generations

Kimi K3 continues a long collaboration between Moonshot AI and the vLLM community.

- At [GOSIM 2024](https://china2024.gosim.org/schedules/vllm-in-moonshot.html), Moonshot AI engineers presented how vLLM was used at scale inside Moonshot AI and discussed the vLLM + Mooncake prefill/decode-disaggregated architecture.
- Moonshot AI later shared Kimi K2 training and inference practices at the [vLLM Beijing Meetup](https://pytorch.org/blog/vllm-beijing-meetup-advancing-large-scale-llm-deployment/), including operating under strict SLOs while serving online traffic and supporting reinforcement-learning workloads.
- vLLM has been a day-0 launch partner for Kimi K2, Kimi K2-Thinking, Kimi K2.5, Kimi Linear, and so on.
- vLLM has deep technical collaboration with Moonshot AI engineers, including [Kimi K2 tool-calling accuracy](https://vllm.ai/blog/Kimi-K2-Accuracy) for correctness, [improved CUDA debugging](https://vllm.ai/blog/improved-cuda-debugging) for development, [decode context parallelism](https://github.com/vllm-project/vllm/pull/23734), Mooncake-based PD disaggregation, and large-scale performance validation. Kimi K2.5 has also appeared in public [InferenceX serving results](https://inferencex.semianalysis.com/inference?g_rundate=2026-04-07&g_model=Kimi-K2.5&g_runid=24100518225&i_gpus=gb200_dynamo-vllm&i_dstart=2026-04-07&i_dend=2026-04-07).

That history matters. Day-0 support is rarely one pull request written after a release announcement. It comes from model and inference teams sharing architecture details early, testing real checkpoints under realistic parallelism, identifying gaps in the serving engine, and upstreaming improvements that remain useful after one launch. **vLLM is proud to be a long-term partner of Moonshot AI and a popular inference engine for Kimi-series models.**

Now, let’s dive into one of the most interesting technical challenges we ran into.

## The Hardest Part: Prefix Caching for KDA

Conventional full attention and KDA remember a prefix in very different ways.

In full attention, a prefix is represented by per-token key and value vectors. vLLM stores those vectors in paged blocks, hashes complete token blocks, and can reuse a matching sequence of blocks for another request.

KDA is recurrent. Instead of retaining a conventional KV pair for every token, each KDA layer advances a matrix-like recurrent state, together with a short convolution state. To resume from a cached prefix, the engine needs the KDA state *at the exact prefix boundary*. Replaying an earlier state to reach that boundary would erase much of the benefit of prefix caching.

![How conventional attention and KDA represent cached prefixes](/assets/figures/2026-07-22-kimi-k3-preview/kda-prefix-state.png)

The straightforward solution—store KDA state at every small attention-cache boundary—is too expensive. A KDA state is much larger than one ordinary token's KV entry, so implementations use a relatively large physical state block to amortize storage. Before the current work, that physical block size also constrained where a prefix-cache hit could land. With a multi-thousand-token state block, two requests sharing almost the entire prompt could still miss the reusable prefix because their common boundary did not fill the same physical block.

The new vLLM design separates three concepts that used to move together:

- **Physical block size:** how KDA state and full-attention KV are allocated on the GPU.
- **Scheduler alignment:** where execution must stop so all cache groups remain consistent.
- **Prefix-match unit:** the finer token interval at which a shared prefix is hashed and may be matched.

![Fine-grained prefix matching inside a larger physical KDA state block](/assets/figures/2026-07-22-kimi-k3-preview/fine-grained-prefix-cache.png)

This lets vLLM register a valid KDA state at a fine-grained boundary inside a larger physical state block. When a later request hits that partial block, the cached state is copied into a private destination before the request extends it. This copy-on-write rule preserves the shared cached prefix while allowing the new request to continue generation safely.

The implementation also handles details that are easy to miss:

- The scheduler stops at the right block and hash boundaries so the recurrent state being registered really corresponds to the advertised token prefix.
- Full-attention and KDA cache groups agree on one `num_computed_tokens`, even though their physical block sizes differ.
- Partial cache entries use chained, fine-grained hashes so a boundary identifies the entire prefix, not only the tail tokens.
- Same-step reuse is deferred until the state copy is safe, avoiding races between cache registration and extension.
- Cache transfer and disaggregated prefill/decode paths can carry the same logical prefix across workers.

This work was motivated by Kimi K3 and many other hybrid attention models, but it is core vLLM infrastructure rather than a model-specific shortcut. The vLLM team and the Moonshot AI team collaborated deeply on the design. The two teams will publish a separate post with the design, invariants, and benchmarks in more detail.

## Performance Work: Removing the New Bottlenecks

Our current progress can be summarized into this table:

| Area | Current status |
| :---- | :---- |
| **Model and configuration** | Kimi K3 language and vision model definitions are integrated, with separate **NVIDIA** and **AMD** implementations where hardware paths differ |
| **Optimized MLA module for native PD disaggregation deployment** | Optimized MLA module with manual kernel fusion and separate prefill/decode paths. Gate projection runs in parallel with attention, with multi-stream support in decode and a fused epilogue in prefill—highly optimized for PD disaggregation deployment. |
| **Serving semantics** | Kimi K3 chat rendering, tokenizer integration, streaming parsing, tool calls, reasoning output, and structured-output paths are implemented and under **final end-to-end validation** |
| **KDA prefill** | FlashKDA and Triton paths are integrated; final backend selection and numerical validation are **in progress** |
| **KDA decode** | A fused **NVIDIA** decode kernel covering convolution, the recurrent KDA update, gating, and normalization is integrated, with portable fallback paths retained |
| **Prefix caching** | Fine-grained partial prefix hits for hybrid full-attention + recurrent-state caches are integrated; disaggregated and offload scenarios are **being validated** |
| **Attention Residuals** | Triton and **NVIDIA** kernels are integrated, including fusion of residual addition and output RMSNorm on supported shapes |
| **MoE** | Kimi K3’s **SiTU** activation is wired into **MXFP4 TRTLLM-Gen** and **DeepGEMM** paths; optimized grouped top-k routing is integrated. **AMD** implements FlyDSL’s **MLIR** kernel stack with hardware-tuned **A16W4/A8W4** fused operators and **SiTU** activation |
| **Production stack** | Non-disaggregated serving is working; Dynamo + vLLM + Mooncake disaggregated serving, expert parallelism, and vendor verification are in the **final validation loop** |

Kimi K3 changes the hot path, so the team has optimized more than the attention kernel itself. Below are details of the progress in each area.

### KDA prefill and decode

The prefill path integrates FlashKDA and Flash Linear Attention (FLA). Around the core recurrence, vLLM fuses the input projections and causal convolution, and gathers initial recurrent states in one operation.

Decode uses a fused NVIDIA kernel on supported architectures and shapes. Instead of launching separate operations for the short convolution, KDA state update, output gate, and normalization for every generated token, the fused path performs them together. This is especially important because Kimi K3 contains many KDA layers; a small per-layer launch or memory penalty quickly becomes a large TPOT penalty.

### Attention Residuals

AttnRes retrieves from representations written by earlier layer blocks rather than relying on only one uniformly accumulated residual stream. A naive implementation creates extra reads, writes, reductions, and normalization launches throughout the 93-layer network.

The release branch includes a Triton implementation and an NVIDIA kernel that fuse residual update, AttnRes mixing, and output RMSNorm for supported cases. Sequence-parallel work also shards the attention-residual traffic across ranks. Early kernel-level results are encouraging, while end-to-end gains are still being measured across prefill lengths and parallel configurations.

### Optimized MLA module for native PD disaggregation deployment

Kimi K3 still uses MLA attention every four layers. In the previous model, vLLM relied heavily on a `torch.compile` custom-fusion path to map small kernels into fused kernels, which slowed startup and still left many kernels unfused. In this release, we implement a new MLA module that fuses these kernels manually. MLA also requires different kernel launch orders for prefill and decode, so we implement two code paths with different fusion patterns, specialized for PD-disaggregated deployment. Furthermore, Kimi K3 introduces a gate projection that can execute in parallel with the main attention path. We optionally add multi-stream support for the gate projection in the decode path, while in the prefill path—where multi-stream overlap is not optimal—we fuse the elementwise multiply and sigmoid into the gate-projection epilogue.

### MXFP4 MoE

Kimi K3's release configuration uses MXFP4 weights and the SiTU activation. Before this work, the MXFP4 TRTLLM-Gen path did not support SiTU and would fall back to a slower implementation. vLLM now maps Kimi K3's SiTU parameters into the optimized FP4 expert path and also handles large token-by-top-k launch grids by safely chunking the workload.

This has already been validated on a 16-GPU DP16+EP16 configuration, where all ranks selected the optimized MXFP4 backend and passed correctness checks.

On the AMD side, Kimi K3 MoE is supported on FlyDSL's MLIR Python kernel stack. This includes hardware-tuned A16W4/A8W4 quantized fused operators and a SiTU activation implementation, all built on FlyDSL's modular abstractions.

## What to Expect on Open-Source Day

The planned day-0 package includes:

- vLLM model, parser, cache, and kernel integration;
- initial open-source Docker images;
- validated launch recipes for NVIDIA configurations;
- an initial AMD path with FlyDSL MoE kernel, with more ROCm tuning to follow;
- multimodal, tool-use, reasoning, and structured-output examples;
- initial performance results.

Trusted deployment partners are already exercising the release candidate under a dual-approval process from Moonshot AI and vLLM/Inferact. This provides real production feedback without distributing prerelease model artifacts broadly. It also gives us a chance to test the complete serving system—frontend semantics, batching, cache transfer, expert parallelism, observability, and failure handling—not only isolated kernels.

## Acknowledgements

Kimi K3 day-0 support is a joint effort across the model vendor, inference engine, and hardware communities.

We thank the **Moonshot AI team** for creating Kimi K3, sharing architecture details ahead of the weight release, contributing the initial model integration and KDA prefix-caching work, and collaborating closely on correctness and production validation.

We thank the **Inferact team** for integrating the model into vLLM, extending the core cache manager for partial hybrid prefix hits, implementing serving semantics and multimodal support, building deployment recipes, and driving end-to-end performance optimization.

We thank the **NVIDIA team** for KDA decode and Attention Residual kernels, MXFP4 MoE collaboration, and performance work across the board.

We thank the **AMD team** for initial day-0 ROCm support and for continuing to expand Kimi K3 across AMD GPUs.

Most importantly, we thank the broader open-source community for the anticipation, testing, and feedback already surrounding Kimi K3. We look forward to putting the weights and the inference engine support in your hands.

## One More Thing: Why the Announcement and Open-Source Release Are Separated

Kimi K3 also features a release process that we hope more model vendors will consider: announce the model first, then release the weights and inference engine support later.

The vLLM team proposed this separation, and Moonshot AI agreed and executed. The reason is practical. A frontier-model announcement has unavoidable last-mile uncertainty. The model team is simultaneously stabilizing its own products, APIs, evaluations, safety work, documentation, and commercial launch. If open-source weights and open-source support must land at the exact same moment, a community project such as vLLM suffers from the moving deadline.

Separating the two timelines gives both sides a better contract:

1. The model vendor can concentrate on its product launch and freeze the final checkpoint, configuration, tokenizer, and serving semantics.
2. The open-source inference engine team gets a stable integration window for correctness tests, performance tuning, Docker builds, and recipe validation.
3. The community gets a public, bounded expectation instead of an ambiguous “coming soon.”

The separation is not a retreat from day-0 support. It is a more sustainable way to deliver day-0 support against the artifact that users will actually download. We encourage more model vendors to follow!
