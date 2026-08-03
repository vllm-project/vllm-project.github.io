---
layout: post
title: "EAGLE3 Speculative Decoding on AMD Instinct GPUs: Training and Serving with vLLM and AMD Quark"
author: "Larry Li, Chao Li, Haichen Zhang, Chun Fang, Andy Luo, Spandan Tiwari, and Ashish Sirasao"
summary: "How AMD Quark trains, quantizes, and serves EAGLE3 speculative-decoding drafts with vLLM on AMD Instinct GPUs, delivering up to 2.00x throughput gains for Kimi-K2.5 and 1.79x for MiniMax-M2.5."
image: /assets/figures/2026-07-13-eagle-3-amd-instinct/figure4.png
social_image: /assets/figures/2026-07-13-eagle-3-amd-instinct/figure4.png
tags:
  - performance
  - hardware
---

Large language model (LLM) inference is increasingly constrained by autoregressive decoding. Even when prefill is highly optimized, the decode phase still generates tokens one step at a time, and each step typically requires running the full target model. For large mixture-of-experts and attention-heavy models such as Kimi-K2.5 and MiniMax-M2.5, this sequential pattern limits serving throughput and increases latency for real-time applications.

Speculative decoding is one of the most practical ways to address this bottleneck. It is a lossless LLM inference acceleration technique that preserves the exact output distribution of the target model while improving decoding efficiency. It uses a smaller or lighter-weight draft model to propose multiple future tokens, then asks the original target model to verify those tokens in a single forward pass. When the draft model predicts tokens that the target model would also produce, those tokens can be accepted together, reducing the number of expensive target-model decode iterations.

Common speculative decoding approaches include small draft models, multi-token prediction (MTP), Medusa-style multi-head prediction, and feature-level drafting methods such as EAGLE3, DFlash, and the recently introduced DSpark. Among existing speculative decoding methods, EAGLE3 is particularly attractive due to its strong draft quality, high acceptance rate, and consistently competitive inference speedups.

In this blog, we walk through three parts of the EAGLE3 workflow on AMD Instinct GPUs, with contributions from the AMD Quark team: (1) training EAGLE3 draft models, where vLLM serves the target to synthesize on-policy data, extract training-time hidden states, and run in-the-loop acceptance evaluation; (2) [AMD Quark](https://quark.docs.amd.com/latest/) quantization, which provides day-0 MXFP4 and FP8 support for both the target and the draft; and (3) inference acceleration on ROCm/vLLM for Kimi-K2.5 and MiniMax-M2.5 on AMD Instinct™ MI355X GPUs, benchmarked with InferenceX. The same pipeline was used to train our MiniMax-M3 EAGLE3 draft, which we use as the running example in the training section.

## Why Speculative Decoding and EAGLE3 Matter

Standard autoregressive decoding emits one token per target-model step. If a model needs to generate 1,000 output tokens, the serving engine typically performs roughly 1,000 target-model decode iterations after prefill. This is expensive because each decode iteration touches the model weights, attention state, scheduler, and KV cache machinery.

Speculative decoding changes this process:

1. A draft model proposes several candidate next tokens.

2. The target model verifies those candidates in one pass.

3. Under greedy decoding, matching draft tokens are accepted; under sampling, draft tokens are accepted or corrected according to the target and draft probabilities.

4. At the first rejection, the verifier emits a correction token and drafting resumes from it; if all draft tokens are accepted, the verifier emits one bonus token.

Conditional acceptance rate measures the probability of accepting a draft position given that the preceding positions were accepted. Acceptance length measures the number of tokens emitted per verification cycle. Higher acceptance length can reduce the number of target-model verification steps, but realized throughput also depends on drafting and verification overhead. (Figure 1)

<img src="/assets/figures/2026-07-13-eagle-3-amd-instinct/figure1.png" alt="Greedy speculative decoding proposal and verification flow" style="display:block;margin:0 auto;width:60%;height:auto" />

*Figure 1: Greedy speculative decoding with γ=5: the target accepts an α=3-token prefix, rejects the first mismatch, discards later draft tokens, and emits a correction token, returning α+1=4 tokens. If all γ draft tokens are accepted, the extra token is a target-generated bonus token.*

[EAGLE](https://github.com/SafeAILab/EAGLE) has been continuously improving over the past few years. It started with feature-level speculative decoding in EAGLE, improved draft quality and acceptance rates in EAGLE2, and further increased accuracy and speedups in EAGLE3 by leveraging multi-layer features from the target model. Instead of relying on an unrelated small language model, it trains a draft module that is closely aligned with the target model. It uses training-time testing techniques and combines low-, mid-, and high-level semantic features from the target model, helping the draft model propose candidates that the verifier is more likely to accept.

For production inference, the important point is simple: EAGLE3 can improve generation throughput while preserving the target model output behavior through verification.

## AMD Quark MXFP4: Day-0 Quantization for Mainstream LLMs

MXFP4 is the Open Compute Project (OCP) Microscaling 4-bit floating-point format: 4-bit elements are grouped into small blocks that share a scale factor, giving a memory footprint close to INT4 while keeping far better numerical behavior. AMD Instinct MI350-series GPUs (MI350X/MI355X) provide native FP4 matrix acceleration, so MXFP4 weights map directly onto the hardware and relieve the memory-bandwidth and capacity pressure that dominates large mixture-of-experts decoding.

AMD Quark is AMD's model-quantization toolkit, and the AMD Quark team provides Day-0 MXFP4 quantized checkpoints for mainstream LLMs, published on Hugging Face (for example, [amd/Kimi-K2.5-MXFP4](https://huggingface.co/amd/Kimi-K2.5-MXFP4) and [amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4)). Day-0 means that when a major model is released, the Quark team ships a hardware-ready MXFP4 (and FP8) build that runs on ROCm/vLLM out of the box, rather than waiting for third-party quantization to catch up. These published checkpoints are ready to use directly as the target for both EAGLE3 draft training and speculative-decoding inference.

These checkpoints are consumed directly by vLLM on ROCm through the supported MXFP4 execution path and AITER MoE kernels, so users get the memory savings of MXFP4 together with production-grade throughput. Speculative decoding is lossless: every draft token is verified against the served target, so it leaves the target's output distribution unchanged.

## Training EAGLE3 Draft Models with vLLM

A high-acceptance draft is what makes speculative decoding fast, and training one is as much a systems problem as a modeling problem. In our pipeline, vLLM is not just the inference engine — it sits at the center of training too. The AMD Quark team developed and validated the MiniMax-M3 EAGLE3 training workflow on AMD Instinct GPUs, which we use as the running example. (The Kimi-K2.5 and MiniMax-M2.5 EAGLE3 drafts in the inference results below are open-source community drafts from Hugging Face, not trained by us.) (Figure 2)

<img src="/assets/figures/2026-07-13-eagle-3-amd-instinct/figure2.png" alt="vLLM-centric EAGLE3 training and serving pipeline" style="display:block;margin:0 auto;width:80%;height:auto" />

*Figure 2: The vLLM-centric EAGLE3 training pipeline. One vLLM-on-ROCm runtime drives the whole loop: it serves the AMD Quark MXFP4/FP8 target model to synthesize on-policy data (Stage 1), streams the target’s low-, mid-, and high-level hidden states to the trainer (Stage 2), cold-starts the single-layer EAGLE3 draft head under FSDP2 (Stage 3), runs in-loop serve-eval to select the best checkpoint by measured acceptance length (Stage 4), then exports the draft and deploys it for EAGLE3 speculative decoding (Stage 5).*

1. On-policy data synthesis, served by vLLM. EAGLE3 drafts learn best from data in the target’s own distribution. We stand up the AMD Quark MXFP4 target as a vLLM-ROCm server and generate on-policy responses through it — both chat (`/v1/chat/completions`, using the exact serving chat template) and raw `/v1/completions` (template-bypassed) for non-chat and out-of-distribution robustness. Generating data with the same engine and template we later serve with keeps training and serving consistent.

2. Hidden-state extraction, provided by vLLM. EAGLE3 conditions the draft on the target’s internal features — low-, mid-, and high-level hidden states plus an `fc_norm` — rather than on an unrelated small model. vLLM’s hidden-state extraction hook exposes these auxiliary layers directly from the running target engine. We support three interchangeable modes: online (target co-located with the trainer), offline (hidden states dumped to disk), and streaming (hidden states streamed from a live vLLM serve to the trainer with no disk dump). Streaming is what makes training a 420B MXFP4 MoE target practical on a single node.

3. Cold-start FSDP2 training. The single-layer EAGLE3 draft head is trained from scratch with a training-time-test (TTT) loss and position-decay weighting under FSDP2. Because the verifier is the AMD Quark MXFP4 target, the draft learns against exactly the activation space it will face at deploy time.

4. Serve-eval in the loop, again on vLLM. The in-training loss overstates real acceptance, so we periodically export the current checkpoint, serve it under vLLM speculative decoding, measure the true acceptance length, and select the best checkpoint by that served metric. The engine that will run in production is the same engine that picks the draft.

5. Export and deployment on vLLM. The selected draft is exported to Hugging Face format, folded into a vLLM-ready draft directory, and deployed with vLLM-ROCm EAGLE speculative decoding — the exact path measured in the next section.

### Draft quality on SPEED-Bench: 11 domains and long context

We evaluate the trained MiniMax-M3 EAGLE3 draft on SPEED-Bench, a multi-domain speculative-decoding benchmark, using acceptance length (AL) — the mean number of tokens emitted per target verification step (higher is better; AL = 1 means one emitted token per target verification step, before accounting for drafting overhead).

**Acceptance length by domain (SPEED-Bench qualitative):**

| Domain        | Acceptance length (AL) |
|---------------|------------------------|
| Coding        | 3.32                   |
| Math          | 3.14                   |
| RAG           | 3.12                   |
| Multilingual  | 3.04                   |
| Reasoning     | 2.89                   |
| STEM          | 2.86                   |
| Summarization | 2.86                   |
| Humanities    | 2.71                   |
| QA            | 2.55                   |
| Writing       | 2.33                   |
| Roleplay      | 2.01                   |
| **Average**   | **2.80**               |

Across 11 domains the draft averages AL 2.80 — roughly 2.8 emitted tokens per target verification step. It is strongest on structured, technical content — coding (3.32), math (3.14), RAG (3.12), and multilingual (3.04) — and still holds AL 2.01-2.33 on open-ended writing and roleplay, the hardest cases for any draft to predict. Just as important, acceptance length is essentially flat as the prompt grows from 1K to 32K tokens (2.69 to 2.65), indicating stable draft acceptance across context lengths. At three speculative tokens, the first, second, and third draft positions are accepted about 76%, 56%, and 43% of the time (cumulative). These results are the payoff of our vLLM-centric recipe: on-policy data generated through the target, hidden-state supervision from the target’s own features, cold-start training against the exact AMD Quark MXFP4 verifier, and checkpoint selection by real served acceptance. (Figure 3)

<img src="/assets/figures/2026-07-13-eagle-3-amd-instinct/figure3.png" alt="MiniMax-M3 EAGLE3 acceptance length by input length" style="display:block;margin:0 auto;width:100%;height:auto" />

*Figure 3: MiniMax-M3 EAGLE3 acceptance length is essentially flat from 1K to 32K context on SPEED-Bench (2.69 at 1K to 2.65 at 32K). The dashed AL=1 line marks one emitted token per verification cycle.*

The trained draft is published as [amd/MiniMax-M3-EAGLE3.1](https://huggingface.co/amd/MiniMax-M3-EAGLE3.1) and can be served with vLLM speculative decoding against the [amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4) target:

```bash
export VLLM_ROCM_USE_AITER=1
vllm serve amd/MiniMax-M3-MXFP4 --trust-remote-code --tensor-parallel-size 8 \
--block-size 128 --attention-backend TRITON_ATTN --moe-backend emulation \
--speculative-config '{"method":"eagle3","model":"amd/MiniMax-M3-EAGLE3.1","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'
```

## End-to-End Solution

The AMD Quark team handles the entire stack end to end:

- Target model: day-0 MXFP4/FP8 quantization and ROCm/vLLM deployment.

- Draft model: EAGLE3 training performed in this work, FP8/MXFP4 quantization with AMD Quark, and ROCm/vLLM deployment.

- End-to-end integration: on-policy data synthesis, hidden-state extraction, serve-eval, export, and speculative serving are all wired through vLLM and validated together.

Together, these components provide a quantized target, a matching high-acceptance draft, and a validated vLLM speculative-decoding deployment for AMD Instinct GPUs.

## Acceleration Results

The following draft results section lists only the 1K/1K workload, with ISL=1024 and OSL=1024. Speedup is computed as EAGLE3 throughput divided by the corresponding no-speculative-decoding baseline throughput. Each draft result is compared only with the no-speculation baseline from the same vLLM build and MML setting. Kimi-K2.5 results use AMD Instinct MI355X, TP=4, random prompts, `num_prompts=10 x concurrency`, `num_warmups=2 x concurrency`, and 10 seeds per cell. Each plotted value is the arithmetic mean of 10 runs with different random seeds. These random-prompt sweeps are throughput microbenchmarks, not application-level workload benchmarks. The Kimi chart shows the BF16 and FP8 draft paths together; the BF16 vLLM v0.19.0 sweep uses MML=2248, while the FP8 sweep uses MML=2304. Because the builds and MML settings differ, the two paths are not a controlled precision comparison. Here, MML (`max-model-len`) is the maximum context length - the total number of tokens (prompt + generated output) that a vLLM model can process in a single request.

### Kimi K2.5 EAGLE3: BF16 and AMD Quark FP8 Drafts

Docker images: BF16 sweep uses `vllm/vllm-openai-rocm:v0.19.0` (MML=2248); FP8 sweep uses `vllm/vllm-openai-rocm:nightly-fb1ac806c55a6dc96fe92261b80c8550e9c39d2f` (MML=2304).

Target model: [amd/Kimi-K2.5-MXFP4](https://huggingface.co/amd/Kimi-K2.5-MXFP4). BF16 draft model: [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3). FP8 draft model: [amd/kimi-k2.5-eagle3-fp8](https://huggingface.co/amd/kimi-k2.5-eagle3-fp8), produced by the AMD Quark team using the released AMD Quark FP8 quantization workflow and metadata; it shares the target's BF16 LM head. In this setup, the FP8 draft path dispatches through vLLM `RowWiseTorchFP8ScaledMMLinearKernel`, i.e. `torch._scaled_mm` over hipBLASLt row-wise scaled FP8 GEMM, rather than the AITER preshuffled FP8 path.

<img src="/assets/figures/2026-07-13-eagle-3-amd-instinct/figure4.png" alt="Kimi-K2.5 EAGLE3 throughput on AMD Instinct MI355X" style="display:block;margin:0 auto;width:100%;height:auto" />

*Figure 4: Kimi-K2.5 EAGLE3 output throughput (tok/s/GPU) at 1K/1K on AMD Instinct MI355X (TP=4). Both the BF16 and AMD Quark FP8 draft paths beat the no-speculative baseline (1.69x-1.90x and 1.76x-2.00x respectively); the gain is largest at low concurrency. Each speedup uses its matching no-speculation baseline; the BF16 and FP8 sweeps use different vLLM builds and MML settings.*

### MiniMax M2.5 BF16 EAGLE3

Docker image: `vllm/vllm-openai-rocm:nightly-4eafc729285e459a5fc96efd6f7b313b155cad48`

Target model: [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5). Draft model: [thoughtworks/MiniMax-M2.5-Eagle3](https://huggingface.co/thoughtworks/MiniMax-M2.5-Eagle3), BF16 draft path with `num_speculative_tokens=3` and `draft_tensor_parallel_size=1`. The numbers below use 1K/1K random prompts, TP=4 with expert parallelism enabled, and five seeds per concurrency. Each plotted value is the arithmetic mean of five runs with different random seeds, and each EAGLE3 result is paired with a no-speculation baseline from the same build and configuration.

<img src="/assets/figures/2026-07-13-eagle-3-amd-instinct/figure5.png" alt="MiniMax-M2.5 EAGLE3 throughput on AMD Instinct" style="display:block;margin:0 auto;width:100%;height:auto" />

*Figure 5: MiniMax-M2.5 EAGLE3 output throughput (tok/s/GPU) at 1K/1K on AMD Instinct MI355X (TP=4). Each EAGLE3 result uses the matching no-speculation baseline; the largest relative gain occurs at low concurrency.*

Across the 1K/1K sweeps, EAGLE3 increases output throughput by 1.69x–2.00x for Kimi-K2.5 and 1.38x–1.79x for MiniMax-M2.5 relative to the matching no-speculation baselines (Figures 4 and 5).

## Summary

Speculative decoding with EAGLE3 delivers throughput gains on AMD Instinct GPUs while preserving target-model decoding semantics - 1.69x to 2.00x for Kimi-K2.5 and up to 1.79x for MiniMax-M2.5 in our 1K/1K sweeps. What makes this practical end to end is the combination of (1) AMD Quark MXFP4/FP8 quantization for the target and selected draft checkpoints, (2) a vLLM-centric training pipeline that synthesizes on-policy data, extracts hidden states, and selects checkpoints by real served acceptance, and (3) ROCm/vLLM speculative serving. The released AMD Quark toolkit provides the quantization workflows; EAGLE3 draft-training support on AMD Instinct GPUs is planned for the next AMD Quark release.

## Acknowledgements

We would like to thank the AMD Quark team, the AMD ROCm and vLLM contributors, the InferenceX maintainers and reviewers, and the EAGLE3 research community for their work and feedback. Special thanks to Chang Liu, Xinjun Niu, Wei Luo, Lin Zhao.

## Additional Resources

- [EAGLE3 project](https://github.com/SafeAILab/EAGLE)
- [EAGLE3 paper](https://arxiv.org/abs/2503.01840)
- [SPEED-Bench](https://arxiv.org/abs/2604.09557)
- [InferenceX](https://github.com/SemiAnalysisAI/InferenceX)
- [AMD Quark](https://github.com/amd/Quark)
- [vLLM](https://github.com/vllm-project/vllm)
