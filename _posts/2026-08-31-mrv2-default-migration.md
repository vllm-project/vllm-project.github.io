---
layout: post
title: "From Opt-In to Default: Migrating vLLM to Model Runner V2"
author: "Wentao Ye"
summary: "How vLLM migrated its execution core from Model Runner V1 to Model Runner V2 through capability-aware routing, staged rollouts, and compatibility-driven testing."
image: /assets/figures/2026-08-31-mrv2-default-migration/rollout.svg
tags:
  - performance
  - engineering
---

In March, we [introduced Model Runner V2 (MRV2)](https://vllm.ai/blog/2026-03-24-mrv2), a ground-up rewrite of vLLM's execution core. That post covered its architecture and early performance. At the time, MRV2 was experimental and opt-in, with only part of vLLM's model and feature surface supported.

As of August 31, 2026, MRV2 is the default model runner across vLLM's model families on `main`, with capability-aware fallbacks for configurations that still require MRV1. The change landed in [#53183](https://github.com/vllm-project/vllm/pull/53183) after several months of work tracked in [#41286](https://github.com/vllm-project/vllm/issues/41286).

![MRV2 rollout from opt-in to the default model runner](/assets/figures/2026-08-31-mrv2-default-migration/rollout.svg)

This post explains how we changed the execution core of a fast-moving inference system without requiring API changes—and what we learned about safely shipping a rewrite at vLLM's scale.

## The Hard Part Was Not Changing the Default

MRV2 has no new user-facing serving API, but the model runner sits at the intersection of nearly every important dimension in vLLM:

- Models: dense, MoE, hybrid-attention, attention-free, encoder-only, decoder-only, and multimodal architectures.
- Features: CUDA graphs, asynchronous scheduling, speculative decoding, LoRA, KV caching and connectors, and weight offloading.
- Deployment and output: parallelism across hardware backends, sampling, prompt logprobs, structured outputs, pooling, and reward models.

A runner can pass greedy decoding on a dense model and still fail when prompt logprobs meet chunked prefill and preemption, LoRA meets CUDA graph capture, or a KV connector changes lifecycle ordering. The real requirement was not simply loading models, but preserving behavior across a large and changing product surface.

## A Capability Oracle Made the Rollout Reversible

The key migration mechanism was the selection policy introduced in [#39337](https://github.com/vllm-project/vllm/pull/39337). `VLLM_USE_V2_MODEL_RUNNER` became a three-state control:

- Unset: let vLLM select the runner from the model and requested features.
- `1`: explicitly select MRV2; known unsupported configurations fail validation instead of falling back.
- `0`: explicitly select MRV1 where supported, providing a debugging and compatibility escape hatch.

When unset, vLLM checks the configuration against known MRV2 compatibility constraints. With no known blocker it selects MRV2; otherwise it can select MRV1 and emit a warning. This is a conservative guardrail, not proof of complete compatibility: gaps that have not yet been identified or encoded can still surface.

![The capability-aware Model Runner selection oracle](/assets/figures/2026-08-31-mrv2-default-migration/oracle.svg)

The oracle separated known incompatibilities from rollout readiness. That let us expand the default boundary in observable, testable, and reversible stages, while contributors could explicitly select either runner where both were supported and compare behavior.

## Expanding the Compatibility Frontier

Qwen3 was the first default-on canary. Qwen3 and OPT-based tests were inexpensive and covered much of the MRV1 suite, although OPT itself initially remained on MRV1. They exposed implicit behavior: [#39353](https://github.com/vllm-project/vllm/pull/39353) fixed a FlexAttention allocation bound, while [#39937](https://github.com/vllm-project/vllm/pull/39937) preserved each request's prompt-logprob count under chunked prefill and preemption. We then expanded MRV2 in stages rather than switching the entire registry at once.

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

## Long-Tail Features Are Production Features

Some of the most useful signals came from configurations that are difficult to cover exhaustively in presubmit. [A user report](https://github.com/vllm-project/vllm/issues/51396) found that MRV2 could silently ignore `--cpu-offload-gb`, consume full GPU memory, and eventually fail with an out-of-memory error even though MRV1 worked.

[#51413](https://github.com/vllm-project/vllm/pull/51413) added MRV2 weight offloading by reusing the MRV1 offloader, and [#51440](https://github.com/vllm-project/vllm/pull/51440) added a regression test. Unsupported configurations should fall back or fail loudly, and production bugs should become focused tests.

## CI Became the Migration Specification

While both runners existed, we forced existing tests onto MRV2 to find MRV1 assumptions, reran relevant tests on MRV1 to prevent regressions, and expanded automatic selection only after compatibility blockers landed.

Rather than build a separate “MRV2 test suite,” we reused the broader vLLM suite against both runners and added focused tests for rollout failures. The [migration tracker](https://github.com/vllm-project/vllm/issues/41286) records **47 completed PRs**: nine rollout milestones and 38 compatibility changes.

Check totals signaled breadth, not behavioral completeness. The first oracle rollout, [#39337](https://github.com/vllm-project/vllm/pull/39337), reported 94 passing checks but still missed a prefill/decode (P/D) issue. Later, a Qwen3 configuration with NIXL and FlashInfer exposed a KV-cache layout incompatibility in [#42846](https://github.com/vllm-project/vllm/issues/42846). [#42955](https://github.com/vllm-project/vllm/pull/42955) temporarily routed KV connector configurations to MRV1 until [#42766](https://github.com/vllm-project/vllm/pull/42766) fixed explicit `kernel_block_size` handling and removed the fallback.

For the final all-model change, [#53183](https://github.com/vllm-project/vllm/pull/53183), the team ran the full NVIDIA CI and an AMD nightly. But the change had been on `main` for only four days when this post was drafted, so release-level soak data was not yet available. This is the **default-on-main milestone**, not proof of completeness or a reason to remove every MRV1 fallback.

## Default Does Not Mean MRV1 Is Gone

MRV2 is now the default across model families, but known unsupported configurations still route to MRV1. As of August 31, 2026, the automatic fallback cases in this [fixed snapshot of `vllm/config/vllm.py`](https://github.com/vllm-project/vllm/blob/e0d27040ddcc5ac31cf01c5b04a7d764ccba656d/vllm/config/vllm.py) are:

- Environments without Triton.
- `DeepseekV32ForCausalLM` and `DeepseekV4ForCausalLM` on ROCm, where MRV2 is unsupported or currently slower.
- Stock `torch.compile`, sequence parallelism with tensor parallelism, and pipeline parallelism with `external_launcher`.
- N-gram speculative decoding, unrecognized speculative methods, EAGLE parallel drafting, and EAGLE3 with pipeline parallelism.
- Dual Batch Overlap and Elastic Expert Parallelism.
- Custom logits processors, whether explicitly configured or registered as entry-point plugins, and KV-sharing fast prefill.

The [MRV2 parity tracker](https://github.com/vllm-project/vllm/issues/47172) records further work around per-request OCR n-grams, generic draft-model speculative decoding and Token-Level Intersection (TLI), rejection sampling, and MTP with pipeline parallelism. This is a dated snapshot; coverage and behavioral parity will improve as these gaps close.

Keeping fallbacks is intentional. “Default” gives MRV2 the first choice for normal operation. “Deprecated” and “removed” require the remaining fallbacks to disappear, downstream backends to have a migration path, and users to have time to report configurations absent from CI.

## What Comes Next

Our immediate focus is to close the remaining parity and backend gaps, expand release-gating coverage, and make new day-zero models MRV2-only, as specified in the [Q3 roadmap](https://github.com/vllm-project/vllm/issues/48168).

## What Users Should Know

No API migration is required. Builds from `main` after commit [`4aab2b0`](https://github.com/vllm-project/vllm/commit/4aab2b0ebed20343efe543c633f71b3c1336d5b8) select MRV2 for supported configurations; the first tagged release with this default will be v0.29.0.

At startup, MRV2 logs `Using V2 Model Runner`; fallback warnings identify the blocker and state that MRV1 was selected. The environment override remains available for diagnosis:

```bash
# Explicitly select MRV2
export VLLM_USE_V2_MODEL_RUNNER=1

# Explicitly select MRV1 where supported
export VLLM_USE_V2_MODEL_RUNNER=0
```

If the runners behave differently, please open an issue with the model, feature flags, hardware/backend, and a minimal reproduction. Reports from real deployments remain essential to this migration.

MRV2 began as a cleaner and faster execution core. Making it the default also required a capability model, a reversible rollout, shared behavioral tests, and months of community work. That less-visible work turns a promising architecture into dependable infrastructure.

## Acknowledgments

This rollout built on the original MRV2 architecture and implementation. Thanks to [Woosuk Kwon](https://github.com/WoosukKwon) and all contributors acknowledged in the [MRV2 announcement](https://vllm.ai/blog/2026-03-24-mrv2) for creating that foundation.

Special thanks to [Nick Hill](https://github.com/njhill), who co-drove the rollout and much of its compatibility work; [Taneem Ibrahim](https://github.com/taneem-ibrahim), who led the pooling-model migration; and [Michael Goin](https://github.com/mgoin) and [Giancarlo Delfin](https://github.com/gcanlin) for supporting features. Thanks also to [Kaichao You](https://github.com/youkaichao) for shepherding key default-boundary changes; to the model, platform, and CI contributors who validated the broader matrix; and to users such as [malaiwah](https://github.com/malaiwah), whose reports became regression tests.
