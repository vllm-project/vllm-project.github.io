---
layout: post
title: "Making Model Runner V2 the Default in vLLM"
author: "Wentao Ye, Nick Hill, Woosuk Kwon"
summary: "How we rolled out Model Runner V2 across vLLM models and features while keeping known incompatibilities on Model Runner V1."
image: /assets/figures/2026-08-31-mrv2-default-migration/rollout.svg
tags:
  - engineering
---

In March, we [introduced Model Runner V2 (MRV2)](https://vllm.ai/blog/2026-03-24-mrv2), a ground-up rewrite of vLLM's execution core. That post covered its architecture and early performance. At the time, MRV2 was experimental and opt-in, with only part of vLLM's model and feature surface supported.

As of August 31, 2026, MRV2 is the default model runner across vLLM's model families on `main`, with capability-aware fallbacks for configurations that still require MRV1. The change landed in [#53183](https://github.com/vllm-project/vllm/pull/53183) after several months of work tracked in [#41286](https://github.com/vllm-project/vllm/issues/41286).

![MRV2 rollout from opt-in to the default model runner](/assets/figures/2026-08-31-mrv2-default-migration/rollout.svg)

## Runner Selection and Fallback

[#39337](https://github.com/vllm-project/vllm/pull/39337) introduced a three-state policy for `VLLM_USE_V2_MODEL_RUNNER`

![The capability-aware Model Runner selection oracle](/assets/figures/2026-08-31-mrv2-default-migration/oracle.svg)

We called this policy the capability oracle. It let us enable model groups one at a time while keeping MRV1 available for comparison.

## Rollout Timeline

We started with Qwen3 because Qwen3 and OPT-based tests covered much of the MRV1 suite without making CI prohibitively expensive. OPT itself initially remained on MRV1. One early gap was the per-request prompt-logprob count under chunked prefill and preemption, fixed in [#39937](https://github.com/vllm-project/vllm/pull/39937).

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

We used the existing vLLM test suite rather than building a separate MRV2 suite. For each rollout step, we selected MRV2 to find assumptions inherited from MRV1, reran relevant tests on MRV1 to protect the old path, and added focused tests for new failures. The [migration tracker](https://github.com/vllm-project/vllm/issues/41286) records **47 completed PRs**: nine rollout milestones and 38 compatibility changes.

### Request State Needs an Explicit Lifecycle

The prompt-logprob bug in [#39937](https://github.com/vllm-project/vllm/pull/39937) was not about accumulating results across chunks; that already worked. MRV2 did not retain the requested top-k count for each request. The fix added per-request state and tested it with chunked prefill and preemption. A similar issue appeared in [#48132](https://github.com/vllm-project/vllm/pull/48132): when `MambaHybridModelState` reused a request slot, `num_accepted_tokens` could be left over from the previous request unless it was reset in `add_request`. GPU-resident request fields need explicit initialization and reuse semantics.

### Ordering Is Part of the Runner Contract

In [#42676](https://github.com/vllm-project/vllm/pull/42676), MRV2 bound KV connector metadata before handling preemptions, reversing the expected lifecycle. [#43719](https://github.com/vllm-project/vllm/pull/43719) moved KV connector post-forward work until after sampling and draft-token proposal for speculative decoding. These were not model-output bugs: they showed that compatibility also includes the ordering of scheduler events, forward execution, sampling, and connector side effects.

### Buffer Shapes Are Backend Contracts

Several failures came from treating one model-level bound as correct for every execution path. [#39353](https://github.com/vllm-project/vllm/pull/39353) corrected a FlexAttention allocation that used `max_model_len` instead of the scheduled-token limit. [#46753](https://github.com/vllm-project/vllm/pull/46753) expanded cross-attention block tables for encoder inputs that can exceed the decoder's `max_model_len`. [#46746](https://github.com/vllm-project/vllm/pull/46746) bounded the top-k logprob kernel's working set rather than padding it to an arbitrarily large requested k. Buffer shapes need to follow the workload and the backend's physical layout, not a convenient global maximum.

---

Not every gap appeared in presubmit. [#51396](https://github.com/vllm-project/vllm/issues/51396) reported that MRV2 silently ignored `--cpu-offload-gb`; [#51413](https://github.com/vllm-project/vllm/pull/51413) added support and [#51440](https://github.com/vllm-project/vllm/pull/51440) added regression coverage.

The number of passing checks was useful, but it was not a coverage metric. [#39337](https://github.com/vllm-project/vllm/pull/39337) had 94 passing checks and still missed a prefill/decode (P/D) issue. Later, [#42846](https://github.com/vllm-project/vllm/issues/42846) found a KV-cache layout problem with Qwen3, NIXL, and FlashInfer. [#42955](https://github.com/vllm-project/vllm/pull/42955) temporarily sent KV connector configurations to MRV1; [#42766](https://github.com/vllm-project/vllm/pull/42766) fixed the `kernel_block_size` handling and removed that fallback.

Before merging the all-model change, [#53183](https://github.com/vllm-project/vllm/pull/53183), we ran the full NVIDIA CI and an AMD nightly. It had been on `main` for only four days when this post was drafted, so we are treating it as a **default-on-main milestone**, not claiming that every MRV1 fallback is ready to be removed.

## Remaining MRV1 Fallbacks

MRV2 is now the default across model families, but known unsupported configurations still route to MRV1. As of August 31, 2026, the automatic fallback cases in this [fixed snapshot of `vllm/config/vllm.py`](https://github.com/vllm-project/vllm/blob/e0d27040ddcc5ac31cf01c5b04a7d764ccba656d/vllm/config/vllm.py) are:

- Environments without Triton.
- Some models on ROCm.
- Stock `torch.compile`, sequence parallelism with tensor parallelism, and pipeline parallelism with `external_launcher`.
- N-gram speculative decoding; other MRV2-unsupported speculative methods, including generic draft-model speculation; EAGLE parallel drafting; and EAGLE3 with pipeline parallelism.
- Dual Batch Overlap and Elastic Expert Parallelism.
- Custom logits processors, whether explicitly configured or registered as entry-point plugins, and KV-sharing fast prefill.

The [MRV2 parity tracker](https://github.com/vllm-project/vllm/issues/47172) lists the remaining design and implementation work as of the date of this post.

The [Q3 roadmap](https://github.com/vllm-project/vllm/issues/48168) calls for closing the remaining parity and backend gaps, expanding release-gating coverage, and supporting new day-zero models only on MRV2.

## For Users

No API migration is required. Builds from `main` after commit [`4aab2b0`](https://github.com/vllm-project/vllm/commit/4aab2b0ebed20343efe543c633f71b3c1336d5b8) select MRV2 for supported configurations; the first tagged release with this default will be v0.29.0.

At startup, MRV2 logs `Using V2 Model Runner`. If vLLM falls back, the warning names the blocker and says that MRV1 was selected. You can also select a runner explicitly while debugging:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1 / 0
```

## Acknowledgments

The original MRV2 architecture and implementation made this rollout possible. Thanks to [Woosuk Kwon](https://github.com/WoosukKwon) and the other contributors listed in the [MRV2 announcement](https://vllm.ai/blog/2026-03-24-mrv2).

[Taneem Ibrahim](https://github.com/taneem-ibrahim) led the pooling-model migration, while [Michael Goin](https://github.com/mgoin) and [Giancarlo Delfin](https://github.com/gcanlin) contributed supporting features. Thanks also to [Kaichao You](https://github.com/youkaichao) for shepherding key default-boundary changes, to the model, platform, and CI contributors who tested the broader matrix, and to users such as [malaiwah](https://github.com/malaiwah), whose reports became regression tests.
