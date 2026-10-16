---
layout: post
title: "From MXFP4 to FP8: Serving DeepSeek V4.1 Flash on 8×H20"
author: "Anzhe Zhang"
summary: >-
  An 8×H20 DeepSeek V4.1 Flash case study: FP8 conversion, Engram offload,
  routed-row MoE dispatch, a 3.24× operator result, and three-run Decode and
  fixed-1K Prefill comparisons against Humming W4A8.
image: /assets/figures/deepseek-v41-h20/optimization-overview.svg
tags:
  - performance
  - moe
  - deepseek
---

**TL;DR.** On eight H20 GPUs, an EP8 FP8 MoE dispatch fix reached up to
**3.24×** speedup in isolated operator tests. Across three serving runs, the
FP8 recipe averaged **6,459 output tok/s** at Decode concurrency 256 (+36.7%
versus Humming W4A8) and **15,048 input tok/s** at fixed-1K Prefill
concurrency 64 (+52.4%). The operator and serving gains should not be
multiplied.

We had eight NVIDIA H20 GPUs and wanted to serve DeepSeek V4.1 Flash
efficiently. That meant choosing an expert-compute path, fitting the
converted checkpoint, and understanding why FP8 MoE still slowed down for
some routed shapes.

![Overview of the deployment and optimization path](/assets/figures/deepseek-v41-h20/optimization-overview.svg)

## Start with the model and the actual kernel path

DeepSeek describes a 552B-parameter Mixture-of-Experts (MoE) model that
activates about 8B parameters per prefill token and 16B per decode token,
with a 40-layer causal encoder-decoder and Compressed Sparse Attention 2
(CSA2). [The technical report](https://arxiv.org/abs/2609.19969) explains
these choices; here we follow the deployment on one eight-H20 node.

The released checkpoint stores routed experts in MXFP4. On H20 (SM90), our
initial vLLM run selected a Marlin W4A16 expert path. We also tried Humming's
W4A8 path, which keeps low-bit weights and uses lower-precision activations.
The [vLLM quantization guide](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/README.md)
lists Humming among its W4A8/W4A4 implementations. It did not deliver the
throughput we expected, so we profiled the running service.

In a separate Torch-profile run before the FP8 conversion, a 30-iteration
trace across all eight ranks attributed roughly 49–52% of summed CUDA-kernel
time to Humming MoE GEMMs; collective kernels accounted for about 4.5–9.8%.
Expert computation was the largest measured hotspot. The trace also showed
rank-to-rank variation.

W4A8 kernels must unpack or convert low-bit weights and prepare activations;
that work can offset faster matrix multiplication. [Humming has discussed
this trade-off](https://github.com/vllm-project/humming/issues/26) on H200.

We checked whether FP8 enabled more efficient expert and linear kernels on
H20. The final serving run selected `FLASHINFER_CUTLASS` for FP8 MoE and
`FlashInferFp8DeepGEMMDynamicBlockScaledKernel` for eligible FP8 linear
layers. [DeepGEMM's documentation](https://github.com/deepseek-ai/DeepGEMM)
describes its SM90 FP8 and grouped-GEMM support.

## A faster compute path brought a separate memory constraint

The first conversion attempt produced reference-inference shards, not a
standalone vLLM checkpoint: it had no `config.json`, and its 32×32 scale
layout did not match the 128×128 block-scaled expert path we were targeting.
Changing metadata could not change the underlying scale data. We therefore
requantized eligible expert and non-expert GEMM weights to FP8 E4M3 with
128×128 FP32 scales, while preserving Engram lookup tables, embeddings,
router weights, norms, and other tensors that were not suitable GEMM targets.
The conversion covered 47,232 routed-expert weight tensors and 355
non-expert weight/scale pairs. Sampled tensor checks measured about 2.69%
relative L2 error for experts and 2.64% for converted non-experts.

The conversion increased the checkpoint's disk footprint: the
original model occupied 475.24 GiB, the FP8 expert overlay alone was 519.03
GiB, and the resulting standalone checkpoint was 718.60 GiB. We kept the
large Engram tables in pinned host memory to fit the converted model on eight
H20s.

Each H20 reported 97,871 MiB of device memory. We used TP8 for dense layers
and EP8 for routed experts, kept the large Engram embedding tables in pinned
host memory, limited the configured context to 262,144 tokens, and bounded
CUDA Graph capture at 768. The model loader reported 70.94 GiB of weights per
rank; the startup memory profiler reported 76.11 GiB for weights plus
non-Torch allocations, a 10.65 GiB peak activation, a 2.60 GiB CUDA Graph
pool, and 3.57 GiB available for KV cache. The resulting KV capacity was
1,041,984 tokens. The serving tests used much shorter requests than the
configured context limit.

We also tried globally setting `expert_dtype=fp8` during an intermediate
overlay validation. That changed ordinary linear-layer dispatch as well and
hit a Marlin shape assertion. The final conversion and serving path therefore
handled expert and non-expert tensor groups deliberately rather than treating
the entire model as one quantization class.

## Small routed M selected the wrong FP8 MoE kernel

Once the FP8 service was running, profiling and source inspection narrowed
the next problem to MoE dispatch. For a grouped expert GEMM, the relevant M is
the number of rows routed to an expert—not the global input-token count.
However, the original SM90 FP8 dispatch estimate used global input M when
choosing a GEMM variant. With top-k routing, that can select a kernel for a
much larger per-expert matrix than the one each expert actually receives.

For the main model, 384 routed experts plus 16 EPLB replicas gave 400 physical
expert slots across EP8. With top-k 6 and 288 input tokens, the average was
`288 × 6 / 400 = 4.32` rows per physical expert. We used a rounded-up estimate
of five instead of global M=288:

```text
expected_m = max(1, ceil(tokens * top_k / (local_experts * ep_size)))
```

The SM90 grouped FP8 GEMM dispatcher uses a per-expert-M threshold of 64 on
H20. For this 288-token case, the original estimate of 288 selects its regular
grouped GEMM path; the estimate of five selects the `swap-A/B` path. Swapping
the operands makes the larger expert output dimension the tiled M dimension,
improving how the kernel schedules work when each expert receives only a few
rows. The dispatcher source establishes these branch choices; the measured
operator latency below covers the full MoE call, not just the GEMM kernel.

We measured both uniform and skewed routing because this estimate uses an
average. The change affects kernel selection; workspace sizing is unchanged.
The implementation is documented in
[FlashInfer PR #5560](https://github.com/flashinfer-ai/flashinfer/pull/5560),
with the corresponding report in [vLLM issue #58799](https://github.com/vllm-project/vllm/issues/58799).

On eight H20 GPUs with EP8, the 288-token uniform case fell from 1.808 ms to
0.564 ms (3.21×). The 192-token case reached 3.24×, and the 2,048-token case
reached 2.65×. The skewed cases improved by 1.99× and 1.05×; a one-token case
was effectively unchanged. All 64 rank/case output checks matched. Each
measurement used five warmups and 24 CUDA Graph/event samples; latency is the
median of each sample's slowest rank.

![EP8 FP8 MoE operator latency across routing shapes](/assets/figures/deepseek-v41-h20/moe-operator-results.svg)

The measurements used an installed-version backport. An exact-upstream build
and focused pytest were run subsequently.

## Complete serving recipe: Decode 6,459 and Prefill 15,048 tok/s

We compared two live deployments with GuideLLM 0.7.4 on separate H20 nodes.
Both used vLLM v0.30.0, TP8 and two-token DSpark speculation. The baseline
kept the original MXFP4 checkpoint and used Humming W4A8 for routed experts;
startup logs confirmed the Humming backend and indexed experts. The tuned
deployment used the converted FP8 checkpoint, `flashinfer_cutlass` MoE,
EP8 with 16 EPLB replicas, asynchronous scheduling, chunked prefill, prefix
caching, 64-token KV blocks, Engram CPU offload, and a CUDA
Graph capture limit of 768 rather than 512. These are complete serving
recipes; the comparison does not isolate one configuration change.

Each concurrency tier was run three times with 10 seconds of warmup. Decode
used unique short prompts, temperature 0.7, and 1,024 output tokens per
request; it measured for 80 seconds after warmup. Fixed-1K Prefill used
1,024 server-side input tokens, one output token, and a 50-second measurement
window after warmup. Prefix-cache hits and request errors were zero in all
runs. Throughput comes from the service's token counters during each
measurement window. Values below are the three-run mean ± sample standard
deviation; [per-run values and methods](/assets/repro/deepseek-v41-h20/README.md)
are available with the data.

| Decode concurrency | Humming W4A8 output tok/s | Tuned FP8 output tok/s | Change |
|---:|---:|---:|---:|
| 1 | 259 ± 3 | 235 ± 9 | −9.1% |
| 8 | 934 ± 19 | 1,192 ± 42 | +27.7% |
| 32 | 1,776 ± 20 | 2,826 ± 45 | +59.1% |
| 64 | 2,507 ± 49 | 4,214 ± 22 | +68.1% |
| 128 | 3,840 ± 10 | 5,769 ± 71 | +50.2% |
| 256 | 4,725 ± 66 | 6,459 ± 173 | +36.7% |

![Decode throughput by concurrency](/assets/figures/deepseek-v41-h20/decode-throughput-comparison.svg)

The tuned recipe pulled ahead from concurrency eight onward. At c256, the
time-limited runs cancelled roughly 254–256 Humming and 253–256 tuned
requests per run; neither service reported request errors.

| Fixed-1K Prefill concurrency | Humming W4A8 input tok/s | Tuned FP8 input tok/s | Change |
|---:|---:|---:|---:|
| 1 | 7,675 ± 25 | 7,123 ± 285 | −7.2% |
| 8 | 9,526 ± 34 | 13,277 ± 643 | +39.4% |
| 32 | 9,845 ± 73 | 15,060 ± 937 | +53.0% |
| 64 | 9,875 ± 63 | 15,048 ± 818 | +52.4% |

![Fixed 1K input Prefill throughput by concurrency](/assets/figures/deepseek-v41-h20/prefill-throughput-comparison.svg)

From c32 to c64, Humming stayed near 9.9k input tok/s and the tuned recipe
near 15.0k. Separate c128 and c256 Prefill reruns remained near the same
throughput plateau; they are omitted from this table to focus on the
lower-latency operating range.

## The tuned Decode profile: communication became prominent

With the dispatch fix active, a c128 Decode trace on rank 0 recorded 163.8 ms
of summed CUDA-kernel duration across ten iterations: FlashInfer MNNVL
AllReduce 65.4 ms (40.0%), DeepGEMM FP8 GEMMs 35.9 ms (21.9%), NCCL AllGather
10.4 ms (6.4%), and sparse MLA kernels 7.6 ms (4.6%). Kernel durations can
overlap across streams; these percentages are shares of summed kernel
duration. The eight-GPU run used NVLink for GPU-to-GPU communication. This is
a profile of the tuned recipe; a baseline profile was not completed.

![Tuned c128 rank-0 CUDA kernel duration mix](/assets/figures/deepseek-v41-h20/profile-c128-kernel-share.svg)

The c128 trace points to communication overlap and small expert GEMMs as the
next areas to investigate.

## Correctness checks and what remains

For a quality sanity check, we ran the same 500 fixed GSM8K questions against
both running services with temperature 0 and a 2,048-token output ceiling.
Six items were excluded from the paired score because at least one service
hit the output limit. On the remaining 494 completions, exact numeric match
was 476/494 for the Humming W4A8 baseline and 475/494 for the FP8-128 tuned
service. This is a quality sanity check for the two serving recipes. The
[score summary](/assets/repro/deepseek-v41-h20/quality-summary.json),
[scoring script](/assets/repro/deepseek-v41-h20/quality_eval.mjs),
[conversion notes](/assets/repro/deepseek-v41-h20/conversion-notes.md),
[serving results](/assets/repro/deepseek-v41-h20/serving-results.csv),
[operator results](/assets/repro/deepseek-v41-h20/operator-results.csv),
[profile samples](/assets/repro/deepseek-v41-h20/profile-c128-rank0.csv), and
[reproduction instructions](/assets/repro/deepseek-v41-h20/README.md) are
available with the benchmark data.

## Lessons from this deployment

* Fewer weight bits do not guarantee a faster kernel; validate the actual
  weight/activation path on the target GPU and routed shapes.
* For EP MoE, kernel selection should reflect per-expert work, not assume the
  global input batch is the expert GEMM's M.

On eight H20 GPUs, FP8 conversion enabled the chosen compute path; Engram
offload made the larger checkpoint deployable; and routed-row-aware dispatch
improved the measured MoE cases. Against Humming W4A8, the tuned serving
recipe delivered higher throughput at the reported multi-request Decode and
fixed-1K Prefill loads.
