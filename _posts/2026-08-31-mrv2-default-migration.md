---
layout: post
title: "Model Runner V2 Is Now the Default in vLLM"
author: "Wentao Ye, Nick Hill, Woosuk Kwon"
summary: "Model Runner V2 rollout across vLLM models and features, with capability-based fallback to Model Runner V1."
image: /assets/figures/2026-08-31-mrv2-default-migration/rollout.svg
tags:
  - engineering
---

[Model Runner V2 (MRV2)](https://vllm.ai/blog/2026-03-24-mrv2), introduced as an experimental runner in March, is now the default across model families on `main`. Unsupported configurations still fall back to MRV1. Default switch in [#53183](https://github.com/vllm-project/vllm/pull/53183); migration tracked in [#41286](https://github.com/vllm-project/vllm/issues/41286).

![MRV2 rollout from opt-in to the default model runner](/assets/figures/2026-08-31-mrv2-default-migration/rollout.svg)

## Runner Selection and Fallback

[#39337](https://github.com/vllm-project/vllm/pull/39337) added three-state selection through `VLLM_USE_V2_MODEL_RUNNER`: `1` for MRV2, `0` for MRV1, unset for capability-based selection.

![The capability-aware Model Runner selection oracle](/assets/figures/2026-08-31-mrv2-default-migration/oracle.svg)

The capability oracle enabled model groups one at a time, with MRV1 kept for unsupported configurations and A/B testing.

## Rollout Timeline

Qwen3 first: broad Qwen3/OPT test coverage at manageable CI cost. OPT initially stayed on MRV1. The first gap, per-request prompt-logprob count under chunked prefill and preemption, was fixed in [#39937](https://github.com/vllm-project/vllm/pull/39937).

| Date | Milestone | What it validated |
| --- | --- | --- |
| March 24 | [MRV2 architecture announcement](https://vllm.ai/blog/2026-03-24-mrv2) | Experimental, opt-in design and early performance |
| May 14 | [Qwen3 canary with Qwen3/OPT test coverage](https://github.com/vllm-project/vllm/pull/39337) | Capability-based selection and broad existing CI coverage |
| June 2 | [Llama and Mistral dense models](https://github.com/vllm-project/vllm/pull/43458) | A second group of widely used dense architectures |
| June 12 | [Qwen and DeepSeek-V2 MoE rollout](https://github.com/vllm-project/vllm/pull/42667) | Moving beyond the first dense-model path |
| June 16 | [Granite MoE](https://github.com/vllm-project/vllm/pull/45461) | Another MoE family with different model integration details |
| June 18 | [Quantized models](https://github.com/vllm-project/vllm/pull/44446) | Quantized variants of already-enabled model families |
| July 2 | [All dense models enabled by default](https://github.com/vllm-project/vllm/pull/44443) | Broad model-family and feature compatibility |
| August 14 | [Attention-free model support](https://github.com/vllm-project/vllm/pull/52374) | Mamba-style state without a conventional attention path |
| August 19 | [Pooling models enabled by default](https://github.com/vllm-project/vllm/pull/48290) | Embedding, classification, reranking, reward, and multimodal pooling workloads |
| August 27 | [MRV2 selected by default for all model families](https://github.com/vllm-project/vllm/pull/53183) | The default boundary reached the full model registry, subject to capability fallbacks |

## Testing Both Runners

Existing tests, run against both runners; focused regressions for each new failure. The [migration tracker](https://github.com/vllm-project/vllm/issues/41286) records **47 completed PRs**: nine rollout milestones and 38 compatibility changes.

### Request State Needs an Explicit Lifecycle

[#39937](https://github.com/vllm-project/vllm/pull/39937): MRV2 did not retain the requested top-k prompt-logprob count per request. [#48132](https://github.com/vllm-project/vllm/pull/48132): a reused `MambaHybridModelState` slot could keep the previous request's `num_accepted_tokens`. Both needed explicit per-request initialization.

### Ordering Is Part of the Runner Contract

[#42676](https://github.com/vllm-project/vllm/pull/42676): bind KV connector metadata after preemption handling. [#43719](https://github.com/vllm-project/vllm/pull/43719): run connector post-forward work after sampling and draft-token proposal. Scheduler, forward, sampling, and connector order is part of the runner contract.

### Buffer Shapes Are Backend Contracts

[#39353](https://github.com/vllm-project/vllm/pull/39353): size FlexAttention buffers by scheduled-token limit, not `max_model_len`. [#46753](https://github.com/vllm-project/vllm/pull/46753): larger cross-attention block tables when encoder inputs exceed decoder length. [#46746](https://github.com/vllm-project/vllm/pull/46746): bounded top-k logprob working set. Buffer shapes follow scheduled work and backend layout, not one global maximum.

Presubmit gap: MRV2 silently ignored `--cpu-offload-gb`, reported in [#51396](https://github.com/vllm-project/vllm/issues/51396). Support in [#51413](https://github.com/vllm-project/vllm/pull/51413), regression coverage in [#51440](https://github.com/vllm-project/vllm/pull/51440).

[#39337](https://github.com/vllm-project/vllm/pull/39337) passed 94 checks but missed a P/D case. [#42846](https://github.com/vllm-project/vllm/issues/42846) later found a Qwen3/NIXL/FlashInfer KV-cache layout bug. Temporary MRV1 fallback in [#42955](https://github.com/vllm-project/vllm/pull/42955); `kernel_block_size` fix and fallback removal in [#42766](https://github.com/vllm-project/vllm/pull/42766).

Before [#53183](https://github.com/vllm-project/vllm/pull/53183): full NVIDIA CI and one AMD nightly. Four days on `main` when drafted. A **default-on-main milestone**, with MRV1 fallbacks still in place.

## Remaining MRV1 Fallbacks

As of August 31, 2026, automatic MRV1 fallbacks in this [fixed `vllm/config/vllm.py` snapshot](https://github.com/vllm-project/vllm/blob/e0d27040ddcc5ac31cf01c5b04a7d764ccba656d/vllm/config/vllm.py):

- Environments without Triton.
- Some models on ROCm.
- Stock `torch.compile`, sequence parallelism with tensor parallelism, and pipeline parallelism with `external_launcher`.
- N-gram speculative decoding; other MRV2-unsupported speculative methods, including generic draft-model speculation; EAGLE parallel drafting; and EAGLE3 with pipeline parallelism.
- Dual Batch Overlap and Elastic Expert Parallelism.
- Custom logits processors, whether explicitly configured or registered as entry-point plugins, and KV-sharing fast prefill.

Remaining work: [MRV2 parity tracker](https://github.com/vllm-project/vllm/issues/47172). Q3 goals: close parity and backend gaps, expand release gating, and put new day-zero models on MRV2 only ([roadmap](https://github.com/vllm-project/vllm/issues/48168)).

## For Users

No API migration. Builds after [`4aab2b0`](https://github.com/vllm-project/vllm/commit/4aab2b0ebed20343efe543c633f71b3c1336d5b8) select MRV2 for supported configurations. First tagged release: v0.29.0.

Startup log: `Using V2 Model Runner`. On fallback, the warning names the blocker. For debugging:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1  # MRV2
# or
export VLLM_USE_V2_MODEL_RUNNER=0  # MRV1
```

## Acknowledgments

MRV2 architecture and implementation: [Woosuk Kwon](https://github.com/WoosukKwon) and contributors listed in the [original announcement](https://vllm.ai/blog/2026-03-24-mrv2). Pooling migration: [Taneem Ibrahim](https://github.com/taneem-ibrahim). Supporting features: [Michael Goin](https://github.com/mgoin) and [Giancarlo Delfin](https://github.com/gcanlin). Default-boundary changes: [Kaichao You](https://github.com/youkaichao). Regression reports: [malaiwah](https://github.com/malaiwah). Also thanks to the model, platform, and CI contributors who tested the broader matrix.
