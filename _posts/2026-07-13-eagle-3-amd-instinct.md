---
layout: post
title: "EAGLE-3 Speculative Decoding on AMD Instinct GPUs: Training and Serving with vLLM and AMD Quark"
author: "Larry Li, Chao Li, and Haichen Zhang"
summary: "How AMD Quark trains, quantizes, and serves EAGLE-3 speculative-decoding drafts with vLLM on AMD Instinct GPUs, delivering up to 2.00x throughput gains for Kimi-K2.5 and 1.79x for MiniMax-M2.5."
image: /assets/figures/2026-07-13-eagle-3-amd-instinct/figure5.png
social_image: /assets/figures/2026-07-13-eagle-3-amd-instinct/figure5.png
tags:
  - performance
  - hardware
---

Large language model (LLM) inference is increasingly constrained by autoregressive decoding. Even when prefill is highly optimized, the decode phase still generates tokens one step at a time, and each step typically requires running the full target model. For large mixture-of-experts and attention-heavy models such as Kimi-K2.5 and MiniMax-M2.5, this sequential pattern limits serving throughput and increases latency for real-time applications.

Speculative decoding is one of the most practical ways to address this bottleneck. It is a lossless LLM inference acceleration technique that preserves the exact output distribution of the target model while improving decoding efficiency. It uses a smaller or lighter-weight draft model to propose multiple future tokens, then asks the original target model to verify those tokens in a single forward pass. When the draft model predicts tokens that the target model would also produce, those tokens can be accepted together, reducing the number of expensive target-model decode iterations.

Common speculative decoding approaches include small draft models, multi-token prediction (MTP), Medusa-style multi-head prediction, and feature-level drafting methods such as EAGLE-3, DFlash, and the recently introduced DSpark. Among existing speculative decoding methods, EAGLE-3 is particularly attractive due to its strong draft quality, high acceptance rate, and consistently competitive inference speedups.

In this blog, we walk through the complete EAGLE-3 lifecycle on AMD Instinct GPUs, all built on vLLM and delivered end to end by the AMD Quark team: (1) training EAGLE-3 draft models, where vLLM serves the target to synthesize on-policy data, extract training-time hidden states, and run in-the-loop acceptance evaluation; (2) AMD Quark quantization, which provides day-0 MXFP4 and FP8 support for both the target and the draft; and (3) inference acceleration on ROCm/vLLM for Kimi-K2.5 and MiniMax-M2.5 on AMD Instinct™ MI355X GPUs, benchmarked with InferenceX. The same pipeline was used to train our MiniMax-M3 EAGLE-3 draft, which we use as the running example in the training section.

## Why Speculative Decoding and EAGLE-3 Matter

Standard autoregressive decoding emits one token per target-model step. If a model needs to generate 1,000 output tokens, the serving engine typically performs roughly 1,000 target-model decode iterations after prefill. This is expensive because each decode iteration touches the model weights, attention state, scheduler, and KV cache machinery.

Speculative decoding changes this process:

1. A draft model proposes several candidate next tokens.
2. The target model verifies those candidates in one pass.
3. Tokens that match the target model distribution are accepted.
4. Generation continues from the first rejected token, or from the end of the accepted block.

The key metric is the acceptance rate: how many draft tokens the target model can accept. Imagine the draft model proposes the next four tokens: if the target model agrees with all four, they can be accepted in a single verification step; if it disagrees on the third token, only the first two are accepted, and the remaining tokens must be regenerated. The more tokens that are accepted at a time, the fewer expensive decoding steps the target model needs to perform, resulting in higher throughput. Importantly, every draft token is checked by the target model before being emitted, so speculative decoding speeds up inference without changing the final output quality. (Figure 1)

<img src="/assets/figures/2026-07-13-eagle-3-amd-instinct/figure1.png" alt="Speculative decoding proposal and verification flow" style="display:block;margin:0 auto;width:60%;height:auto" />

*Figure 1: How speculative decoding works — the draft proposes γ tokens, the target verifies them in a single pass, the matching prefix (α) is accepted and α+1 tokens are emitted, and drafting resumes from the first rejected token. Every emitted token is verified, so the speedup is lossless.*

EAGLE has been continuously improving over the past few years. It started with feature-level speculative decoding in EAGLE, improved draft quality and acceptance rates in EAGLE-2, and further increased accuracy and speedups in EAGLE-3 by leveraging multi-layer features from the target model. Instead of relying on an unrelated small language model, it trains a draft module that is closely aligned with the target model. It uses training-time testing techniques and combines low-, mid-, and high-level semantic features from the target model, helping the draft model propose candidates that the verifier is more likely to accept.

For production inference, the important point is simple: EAGLE-3 can improve generation throughput while preserving the target model output behavior through verification.



## AMD Quark MXFP4: Day-0 Quantization for Mainstream LLMs

MXFP4 is the Open Compute Project (OCP) Microscaling 4-bit floating-point format: 4-bit elements are grouped into small blocks that share a scale factor, giving a memory footprint close to INT4 while keeping far better numerical behavior. AMD Instinct MI350-series GPUs (MI350X/MI355X) provide native FP4 matrix acceleration, so MXFP4 weights map directly onto the hardware and relieve the memory-bandwidth and capacity pressure that dominates large mixture-of-experts decoding.

AMD Quark is AMD’s model-quantization toolkit, and the Quark team provides day-0 MXFP4 quantized checkpoints for mainstream LLMs, published on Hugging Face (for example, amd/Kimi-K2.5-MXFP4 and amd/MiniMax-M3-MXFP4). Day-0 means that when a major model is released, the Quark team ships a hardware-ready MXFP4 (and FP8) build that runs on ROCm/vLLM out of the box, rather than waiting for third-party quantization to catch up. These published checkpoints are ready to use directly as the target for both EAGLE-3 draft training and speculative-decoding inference.

These checkpoints are consumed directly by vLLM on ROCm through the FP4 ASM GEMM path (VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1) and AITER MoE kernels, so users get the memory savings of MXFP4 together with production-grade throughput. Speculative decoding is lossless: every draft token is verified against the served target, so it leaves the target’s output distribution unchanged. (Figure 2)

<img src="/assets/figures/2026-07-13-eagle-3-amd-instinct/figure2.png" alt="AMD Quark day-0 quantization flow" style="display:block;margin:0 auto;width:55%;height:auto" />

*Figure 2: AMD Quark day-0 quantization flow — when a mainstream LLM ships, the Quark team releases an MXFP4/FP8 build on Hugging Face that vLLM on ROCm serves directly.*

## Training EAGLE-3 Draft Models with vLLM

A high-acceptance draft is what makes speculative decoding fast, and training one is as much a systems problem as a modeling problem. In our pipeline, vLLM is not just the inference engine — it sits at the center of training too. We use the MiniMax-M3 EAGLE-3 draft we trained on AMD Instinct GPUs as the running example. (The Kimi-K2.5 and MiniMax-M2.5 EAGLE-3 drafts in the inference results below are open-source community drafts from Hugging Face, not trained by us.) (Figure 3)

<img src="/assets/figures/2026-07-13-eagle-3-amd-instinct/figure3.png" alt="vLLM-centric EAGLE-3 training and serving pipeline" style="display:block;margin:0 auto;width:80%;height:auto" />

*Figure 3: The vLLM-centric EAGLE-3 pipeline. A single vLLM-on-ROCm runtime powers the whole loop across five stages — it serves the AMD Quark MXFP4/FP8 target to synthesize on-policy data (Stage 1), streams the target hidden states to the FSDP2 cold-start draft trainer and runs in-loop serve-eval to select the best checkpoint (Stage 2), after which the draft is exported (Stage 3), deployed for EAGLE-3 speculative decoding (Stage 4), and evaluated for acceptance length and per-GPU throughput (Stage 5).*

1. On-policy data synthesis, served by vLLM. EAGLE-3 drafts learn best from data in the target’s own distribution. We stand up the AMD Quark MXFP4 target as a vLLM-ROCm server and generate on-policy responses through it — both chat (`/v1/chat/completions`, using the exact serving chat template) and raw `/v1/completions` (template-bypassed) for non-chat and out-of-distribution robustness. Generating data with the same engine and template we later serve with keeps training and serving consistent.

2. Hidden-state extraction, provided by vLLM. EAGLE-3 conditions the draft on the target’s internal features — low-, mid-, and high-level hidden states plus an `fc_norm` — rather than on an unrelated small model. vLLM’s hidden-state extraction hook exposes these auxiliary layers directly from the running target engine. We support three interchangeable modes: online (target co-located with the trainer), offline (hidden states dumped to disk), and streaming (hidden states streamed from a live vLLM serve to the trainer with no disk dump). Streaming is what makes training a 420B MXFP4 MoE target practical on a single node.

3. Cold-start FSDP2 training. The single-layer EAGLE-3 draft head is trained from scratch with a training-time-test (TTT) loss and position-decay weighting under FSDP2. Because the verifier is the AMD Quark MXFP4 target, the draft learns against exactly the activation space it will face at deploy time.

4. Serve-eval in the loop, again on vLLM. The in-training loss overstates real acceptance, so we periodically export the current checkpoint, serve it under vLLM speculative decoding, measure the true acceptance length, and select the best checkpoint by that served metric. The engine that will run in production is the same engine that picks the draft.

5. Export and deployment on vLLM. The selected draft is exported to Hugging Face format, folded into a vLLM-ready draft directory, and deployed with vLLM-ROCm EAGLE speculative decoding — the exact path measured in the next section.

Because vLLM drives data generation, hidden-state extraction, in-loop evaluation, and serving, the draft is trained and validated against its production engine end to end — which is exactly why the acceptance we measure in training holds up in production.

### Draft quality on SPEED-Bench: 11 domains and long context

We evaluate the trained MiniMax-M3 EAGLE-3 draft on SPEED-Bench, a multi-domain speculative-decoding benchmark, using acceptance length (AL) — the mean number of tokens emitted per target verification step (higher is better; AL = 1 means no speedup).

**Acceptance length by domain (SPEED-Bench qualitative):**

| Domain        | Acceptance length (AL) |
|---------------|------------------------|
| Coding        | 3.16                   |
| Math          | 3.12                   |
| RAG           | 3.11                   |
| Multilingual  | 3.07                   |
| Summarization | 2.88                   |
| Reasoning     | 2.87                   |
| STEM          | 2.78                   |
| Humanities    | 2.67                   |
| QA            | 2.57                   |
| Writing       | 2.29                   |
| Roleplay      | 2.01                   |
| **Average**   | **2.77**               |

Across 11 domains the draft averages AL 2.77 — roughly 2.8 accepted tokens per target step. It is strongest on structured, technical content — coding (3.16), math (3.12), RAG (3.11), and multilingual (3.07) — and still holds AL 2.0-2.9 on open-ended writing and roleplay, the hardest cases for any draft to predict. Just as important, acceptance length is essentially flat as the prompt grows from 1K to 32K tokens (2.64 to 2.63), so the speedup does not decay on long context. At three speculative tokens, the first, second, and third draft positions are accepted about 75%, 55%, and 41% of the time (cumulative). These results are the payoff of our vLLM-centric recipe: on-policy data generated through the target, hidden-state supervision from the target’s own features, cold-start training against the exact AMD Quark MXFP4 verifier, and checkpoint selection by real served acceptance. (Figure 4)

<img src="/assets/figures/2026-07-13-eagle-3-amd-instinct/figure4.png" alt="MiniMax-M3 EAGLE-3 acceptance length by input length" style="display:block;margin:0 auto;width:100%;height:auto" />

*Figure 4: MiniMax-M3 EAGLE-3 acceptance length is essentially flat from 1K to 32K context on SPEED-Bench (2.64 at 1K to 2.63 at 32K) — the speedup does not decay on long prompts.*

## One Team, End to End: The AMD Quark Advantage

A distinctive aspect of this work is that a single team — AMD Quark — owns the entire stack for both the target and the draft model:

- Target model: day-0 MXFP4/FP8 quantization and ROCm/vLLM deployment.
- Draft model: EAGLE-3 training, FP8/MXFP4 quantization, and ROCm/vLLM deployment.
- End-to-end integration: on-policy data synthesis, hidden-state extraction, serve-eval, export, and speculative serving are all wired through vLLM and validated together.

This one-stop ownership means that customers on AMD Instinct receive a quantized target, a matching high-acceptance draft, and a tuned vLLM speculative-decoding deployment from a single team — with day-0 support for new models — instead of hand-assembling CUDA-centric tooling.

## Acceleration Results

The following draft results section lists only the 1K/1K workload, with ISL=1024 and OSL=1024. Speedup is computed as EAGLE-3 throughput divided by the corresponding no-speculative-decoding baseline throughput. Kimi-K2.5 results use AMD Instinct MI355X, TP=4, random prompts, `num_prompts=10 x concurrency`, `num_warmups=2 x concurrency`, and 10 seeds per cell. The Kimi chart shows the BF16 and FP8 draft paths together; the BF16 vLLM v0.19.0 sweep uses MML=2248, while the FP8 sweep uses MML=2304. Here, MML (`max-model-len`) is the maximum context length — the total number of tokens (prompt + generated output) that a vLLM model can process in a single request.

### Kimi K2.5 EAGLE-3: BF16 and AMD Quark FP8 Drafts

Docker images: BF16 sweep uses `vllm/vllm-openai-rocm:v0.19.0` (MML=2248); FP8 sweep uses `vllm/vllm-openai-rocm:nightly-fb1ac806c55a6dc96fe92261b80c8550e9c39d2f` (MML=2304).

Target model: [amd/Kimi-K2.5-MXFP4](https://huggingface.co/amd/Kimi-K2.5-MXFP4). BF16 draft model: [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3). FP8 draft model: [amd/kimi-k2.5-eagle3-fp8](https://huggingface.co/amd/kimi-k2.5-eagle3-fp8), quantized with AMD Quark FP8 metadata and sharing the BF16 target LM head. In this setup, the FP8 draft path dispatches through vLLM `RowWiseTorchFP8ScaledMMLinearKernel`, i.e. `torch._scaled_mm` over hipBLASLt row-wise scaled FP8 GEMM, rather than the AITER preshuffled FP8 path. The target MXFP4 model uses the ROCm FP4 ASM path via `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1`.

<img src="/assets/figures/2026-07-13-eagle-3-amd-instinct/figure5.png" alt="Kimi-K2.5 EAGLE-3 throughput on AMD Instinct MI355X" style="display:block;margin:0 auto;width:100%;height:auto" />

*Figure 5: Kimi-K2.5 EAGLE-3 output throughput (tok/s/GPU) at 1K/1K on AMD Instinct MI355X (TP=4). Both the BF16 and AMD Quark FP8 draft paths beat the no-speculative baseline (1.69x-1.90x and 1.76x-2.00x respectively); the gain is largest at low concurrency.*



### MiniMax M2.5 BF16 EAGLE-3

Docker image: `vllm/vllm-openai-rocm:nightly-4eafc729285e459a5fc96efd6f7b313b155cad48`

Target model: [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5). Draft model: [thoughtworks/MiniMax-M2.5-Eagle3](https://huggingface.co/thoughtworks/MiniMax-M2.5-Eagle3/tree/main), BF16 draft path with `num_speculative_tokens=3` and `draft_tensor_parallel_size=1`. The numbers below use 1K/1K random prompts, TP=4/EP enabled, and 5 seeds per concurrency, with the same baseline-vs-EAGLE-3 setup used for the MiniMax reproduction.

<img src="/assets/figures/2026-07-13-eagle-3-amd-instinct/figure6.png" alt="MiniMax-M2.5 EAGLE-3 throughput on AMD Instinct" style="display:block;margin:0 auto;width:100%;height:auto" />

*Figure 6: MiniMax-M2.5 EAGLE-3 output throughput (tok/s/GPU) at 1K/1K (TP=4). The BF16 draft path delivers 1.38x-1.79x over the no-speculative baseline, again strongest at low concurrency.*



Across the Kimi-K2.5 1K/1K sweep, the BF16 EAGLE-3 draft path delivers 1.69x to 1.90x throughput over the no-speculative-decoding baseline, while the AMD Quark FP8 EAGLE-3 draft path delivers 1.76x to 2.00x throughput (Figure 5). For MiniMax-M2.5, the BF16 EAGLE-3 draft path delivers 1.38x to 1.79x throughput over baseline across concurrency 64 to 4 (Figure 6). The FP8 draft path for Kimi uses the currently integrated RowWise hipBLASLt FP8 draft-kernel path.

## Summary

Speculative decoding with EAGLE-3 delivers lossless, significant throughput gains on AMD Instinct GPUs — 1.69x to 2.00x for Kimi-K2.5 and up to 1.79x for MiniMax-M2.5 in our 1K/1K sweeps. What makes this practical end to end is the combination of (1) AMD Quark day-0 MXFP4/FP8 quantization for both target and draft, (2) a vLLM-centric training pipeline that synthesizes on-policy data, extracts hidden states, and selects checkpoints by real served acceptance, and (3) ROCm/vLLM speculative serving. Because a single team — AMD Quark — owns quantization, training, and deployment for both target and draft, AMD Instinct users get a coherent, day-0 speculative-decoding stack rather than a hand-assembled one.

## Acknowledgements

This work strengthens the vLLM ecosystem on AMD: it adds ROCm backend coverage for EAGLE-3 speculative decoding, reproducible training and serving recipes, a ready-to-use draft model, and clear paths for upstream collaboration.

We would like to thank the AMD Quark team, the AMD ROCm and vLLM contributors, the InferenceX maintainers and reviewers, and the EAGLE-3 research community for their work and feedback. Special thanks to AMD Andy Luo’s team members including Haichen Zhang, Chun Fang, and Chang Liu for their contributions to the InferenceX EAGLE-3 benchmark integration.

## Additional Resources

- [EAGLE-3 project](https://github.com/SafeAILab/EAGLE)
- [AMD Quark](https://github.com/amd/Quark)
- [vLLM](https://github.com/vllm-project/vllm)
