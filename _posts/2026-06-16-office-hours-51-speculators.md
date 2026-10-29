---
layout: post
title: "Speculators v0.5 and Accelerating Sparse MLA: A Recap of Office Hours #51"
author: "Sawyer Bowerman"
tags:
  - community
  - office-hours
  - speculative-decoding
  - speculators
  - inference
---

At [vLLM Office Hours #51](https://www.youtube.com/watch?v=FfaBFddcj_4), the Red Hat AI team presented two deep technical topics: a walkthrough of the Speculators v0.5 release with DFlash support and online training, and an engineering spotlight on accelerating sparse Multi-head Latent Attention (MLA) using masked MHA. We also covered project updates from across the vLLM ecosystem.

Here is a summary of the key points from the session.

---

## vLLM project updates

Before the deep dives, Michael Goin shared what has been happening across the vLLM ecosystem. Highlights include:

- **Speculators v0.5** shipped with DFlash support and online training. On math reasoning workloads, combining DFlash with FP8 block quantization delivers up to 6x lower median inter-token latency compared to baseline FP16 inference without speculative decoding. Read the [release post](https://vllm.ai/blog/2026-05-28-speculators-v050).
- **vLLM-Omni v0.22.0** added Day-0 NVIDIA Cosmos 3 world-model support for omnimodal serving across text, image, audio, video, and action. The release included 339 commits from 124 contributors, 52 of whom were new.
- **Native RL APIs** landed in vLLM with a controller, trainer, and generator architecture. Weight updates flow through NCCL. Alongside this, [vime](https://github.com/vllm-project/vime) provides a stable RL framework for LLMs with R3 routing replay to reduce train-inference mismatch for MoE models.
- **DiffusionGemma** became the first diffusion language model natively supported in vLLM, reaching up to 1,288 generation tokens per second on H200 with FP8.
- **NVIDIA Nemotron 3 Ultra** received Day-0 vLLM support with 5x faster inference while maintaining competitive accuracy among open models.
- **vLLM v0.22** landed with 459 commits from 230 contributors. Key additions include DeepSeek V4 hardening, DSv4 MTP spec decode on ROCm (+75% tok/s), Model Runner V2 as default for Qwen3 dense, batch invariance with faster Cutlass, bring-your-own drafter for spec decode, MXFP4 quantization, an experimental Rust frontend, multi-tier KV cache offloading, and Blackwell SM120/121 support.

![Speculators v0.5 release overview with DFlash support, online training, and 6x latency improvement on math reasoning benchmarks](/assets/figures/office-hours-51-speculators/speculators-v05-overview.png)

## What's new in Speculators

Fynn Schmitt-Ulms and Helen Zhao walked through the latest additions to the [Speculators library](https://github.com/vllm-project/speculators), which provides an end-to-end workflow for speculative decoding: train, convert, evaluate, and serve.

For context, speculative decoding is a lossless technique for speeding up LLM inference. A small draft model (the speculator) proposes multiple tokens, and the larger target model verifies them all in a single forward pass. Accepted tokens pass through as-is; rejected tokens are corrected by the verifier. The output is mathematically identical to standard decoding, but latency drops because multiple tokens can be accepted per verification step.

![Speculative decoding compared to vanilla decoding: the draft model proposes tokens that the verifier checks in one forward pass](/assets/figures/office-hours-51-speculators/speculative-decoding-overview.png)

### New drafting algorithms

Speculators v0.5 adds support for several drafting algorithms alongside the existing Eagle3 architecture:

**Eagle3** takes hidden states from three layers of the target model (concatenated and embedded) plus token embeddings as input and autoregressively generates candidate draft tokens. Based on [Li et al. 2025](https://arxiv.org/abs/2503.01840).

**P-Eagle** inherits Eagle3's architecture but introduces parallel prediction. Instead of sequential test-time drafting steps, P-Eagle trains multiple token predictions in parallel through Conditional Drop-Token (COD) sampling. Based on [Hui et al. 2026](https://arxiv.org/abs/2602.01469).

**DFlash** uses a lightweight block diffusion model to draft an entire block of tokens in a single parallel forward pass. On Qwen3-8B, DFlash achieved up to 3.5x speedup in median inter-token latency. Combining DFlash with FP8-Dynamic quantization pushed latency even lower across a range of request rates. Based on [Chen et al. 2026](https://arxiv.org/abs/2602.06036).

![DFlash architecture: a lightweight block diffusion model drafts an entire block of tokens in a single parallel forward pass](/assets/figures/office-hours-51-speculators/dflash-architecture.png)

![Qwen3-8B speculative decoding benchmarks showing DFlash speedup and the additional gains from combining DFlash with FP8-Dynamic quantization](/assets/figures/office-hours-51-speculators/dflash-benchmarks.png)

**Fast MTP** fine-tunes only the MTP (multi-token prediction) head using a weighted cross-entropy loss with exponential decay for distant token predictions. All main model components stay frozen, keeping training cost low.

### Hidden states extraction

Training speculator models requires hidden states from the target model. Speculators now supports a new extraction system with two primary modes:

**Online training:** The vLLM server extracts hidden states and sends them directly to the training process in real time via NCCL. This keeps the pipeline simple and avoids disk I/O.

**Offline training:** A separate datagen process sends requests to the vLLM server, which writes hidden states to local disk. The training process reads from disk independently, decoupling data generation from training.

The extraction system uses a dummy Eagle3 draft model with `FakeAttentionLayer` modules inside the vLLM server. Hidden states from the verifier model are captured through a KV Cache Connector and routed to the training process via disk or NCCL.

### Evaluation and documentation

Speculators now integrates with GuideLLM for evaluation. You can run a sweep against a running vLLM endpoint:

```bash
python evaluate.py sweep --target http://localhost:8000/v1
```

Full documentation with tutorials for each algorithm is available at [docs.vllm.ai/projects/speculators](https://docs.vllm.ai/projects/speculators), covering online and offline training for Eagle3, DFlash, P-Eagle, and MTP models.

## Engineering spotlight: accelerating sparse MLA via masked MHA

Matt Bonanni presented work on accelerating sparse Multi-head Latent Attention (MLA) in DeepSeek models by introducing a masked MHA path for moderate sequence lengths.

### How MLA works

MLA compresses the KV cache by projecting keys and values into a shared latent space. In DeepSeek models, this reduces the cached representation from dimension 16,384 down to 512, a 57x reduction. There are two mathematically equivalent ways to compute MLA:

- **MHA-style:** Decompress the latent KV back into full keys and values, then run standard multi-head attention. This is compute-friendly and works well during prefill.
- **MQA-style:** Compress the query into the latent space and attend directly over the compressed KV. This requires 3.4x more FLOPs but moves far less memory, making it better suited for decode.

### The sequence length gap

DeepSeek V3.2 introduced sparse attention, where each query token attends only to its top-k most relevant past tokens. MHA-style attention wins at shorter sequences and MQA-style wins at longer sequences, but neither is optimal in the middle band. Matt Bonanni's insight was that masked MHA fills that gap.

The solution splits sparse MLA computation into three tiers based on sequence length:

| Sequence length | Strategy | Rationale |
|---|---|---|
| seq ≤ top-k | Dense MHA | Every token is already in top-k, no mask needed |
| top-k < seq ≤ 8,192 | Masked MHA | FlashAttention 4 with block-sparse masks from top-k |
| seq > 8,192 | Sparse MQA | Gather top-k latents, run decode-friendly MQA path |

In the moderate sequence length band (roughly 136 to 7,740 tokens), masked MHA delivers up to 2x speedup over the MQA path for prefill.

![Sparse MLA benchmark results showing MHA versus MQA latency across sequence lengths and the three-tier strategy](/assets/figures/office-hours-51-speculators/sparse-mla-results.png)

## Get involved

- **Speculators repo:** [github.com/vllm-project/speculators](https://github.com/vllm-project/speculators)
- **Speculators docs:** [docs.vllm.ai/projects/speculators](https://docs.vllm.ai/projects/speculators)
- **Pre-trained speculator models:** [RedHatAI/speculator-models on Hugging Face](https://huggingface.co/collections/RedHatAI/speculator-models)
- **vLLM Slack:** [#sig-spec-decode](https://slack.vllm.ai) and [#speculators](https://slack.vllm.ai)
- **Spec Decode SIG meeting:** Weekly on Thursdays at 3:30pm ET

## Watch the full session

The recording from Office Hours #51 is available [here](https://www.youtube.com/watch?v=FfaBFddcj_4). The slide deck is also [available](https://docs.google.com/presentation/d/146fadQKacCEBnVmCRwPPFS-OXu34yoyvDGCVyWjhg7o/edit).

