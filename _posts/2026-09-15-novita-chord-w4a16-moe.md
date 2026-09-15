---
layout: post
title: "vLLM x Novita AI: Chord, Up to 2.15x Faster INT4 MoE for Kimi K2.x"
author: "Novita AI and the vLLM Team"
summary: "Novita AI has open-sourced Chord, a high-performance W4A16 MoE CUDA kernel for Kimi K2.x serving shapes, with a Humming-compatible indexed path and grouped SM90 operators."
image: /assets/figures/2026-09-15-novita-chord-w4a16-moe/cover.png
tags:
  - performance
  - quantization
  - moe
  - hardware
---

## TL;DR

[Novita AI](https://novita.ai) has open-sourced [Chord](https://github.com/novitalabs/chord), a high-performance W4A16 MoE CUDA operator for BF16 activations, INT4 weights and group-32 scales. Built for Kimi K2.x serving shapes, its indexed path exposes the Humming-compatible `humming` import root, selected with `--quantization humming` on compatible vLLM revisions. Integration of the grouped operators with vLLM's Humming backend is still a work in progress.

Measured per layer against public Humming's matching path:

- **1.11–1.20x on H200 EP8 prefill**, and **1.17–1.33x on H200 TP8** single-instance serving.
- **1.16–1.24x on H200 EP8 decode**, with the down stage reaching 1.31x.
- **1.81–2.15x on B300 EP8 decode**, against Humming's default, untuned configuration strategy.
- **1.00–1.31x and 1.16–1.35x** for the grouped H200 EP8 prefill and decode paths; the same tables measure 1.18–1.34x on EP16 and 1.13–1.30x on EP32.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/assets/figures/2026-09-15-novita-chord-w4a16-moe/benchmark-dark.svg">
  <img src="/assets/figures/2026-09-15-novita-chord-w4a16-moe/benchmark-light.svg" alt="Per-call latency versus token count for public Humming and Chord across the six measured serving scenarios; lower is better" width="100%">
</picture>
</p>

_Figure 1. Per-call latency against public Humming across the six measured scenarios, lower is better. Read each panel on its own: the B300 decode panel compares against an untuned Humming default because public Humming ships no SM100/SM103 tuning table, while every H200 panel is tuned-to-tuned. Chart from the [Chord repository](https://github.com/novitalabs/chord); full tables in [docs/performance.md](https://github.com/novitalabs/chord/blob/main/docs/performance.md)._

The idea behind these numbers is that one W4A16 MoE kernel cannot be right for every request. Routed tokens per expert varies by orders of magnitude between prefill and decode, and it is that quantity, not total token count, that decides which schedule wins. Chord picks the schedule from the shape it is actually given.

These are kernel-level measurements, not a promise of the same end-to-end gain for every workload. Full tables, shape definitions and timing methodology are in [docs/performance.md](https://github.com/novitalabs/chord/blob/main/docs/performance.md) and [docs/benchmarking.md](https://github.com/novitalabs/chord/blob/main/docs/benchmarking.md). Code and kernel tables in this post refer to [`7ca91d8`](https://github.com/novitalabs/chord/commit/7ca91d84a66baea66181af99bab6aace04d39589) (September 14, 2026).

## Two kernel families

The current `main` branch ships two independent families:

- **`indexed`** is the Humming-derived path. It consumes vLLM's `sorted_ids`/`expert_ids`/`num_tokens_padded` routing and covers H200 EP8 prefill, H200 TP8 single-instance serving, H200 EP8 decode, and B200/B300 EP8 decode.
- **`grouped_contiguous`** (prefill) and **`grouped_masked`** (decode) are a second SM90 family derived from DeepGEMM. They consume grouped routing (`m_indices` or `expert_layout`) and use a different packed weight layout.

## vLLM integration

Install the package and select the existing Humming backend:

```bash
pip install git+https://github.com/novitalabs/chord.git
# Do not co-install inclusionAI/humming: Chord intentionally owns that import name (for indexed path, grouped integration is WIP).

vllm serve <kimi-k2.x-int4-model> --quantization humming
# or select moe_backend="humming" in the vLLM configuration
```

The distribution provides both `chord` and `humming` module roots. vLLM's lazy facade resolves `humming.{dtypes,config,layer,schema,utils.weight}`; the default indexed path can use this existing integration without a Chord-specific framework patch on branches with the WNA16 group-scale support noted below. The shipped schema supports uint4, group-32, BF16 scales and the compressed-tensors pack-quantized INT4 group-32 checkpoint format used by Kimi K2.x; unsupported quantization schemes fail at load instead of silently selecting a wrong kernel.

**Grouped integration with vLLM's Humming backend is WIP.** The standalone grouped operator API is shown below.
TP8 remains the indexed `h200_tp8` profile because one TP8 weight must serve both phases.

Other deployment details:

- Profile selection recovers EP8 versus TP8 from the projection shapes already passed by the framework; no Chord-specific shard argument is required for the indexed profiles. Grouped profiles are EP-only and support EP8/EP16/EP32 on SM90.
- The indexed fast path can consume vLLM's over-allocated `moe_align_block_size` buffers without reading the routed count back to the host when trusted routing validation is disabled, so it remains CUDA Graph capturable. Grouped paths use CUDA routing tensors too, while `valid_shape_m`/`expected_m` are Python-side heuristic inputs.
- Explicitly select Humming (`moe_backend="humming"` or `--quantization humming`); vLLM's automatic WNA16 priority can choose another backend first. Keep `VLLM_HUMMING_USE_F16_ACCUM` and `VLLM_BATCH_INVARIANT` off because neither backend implements those compute options.
- Keep `VLLM_HUMMING_MOE_GEMM_TYPE` at its indexed behavior for the default integration. vLLM branches older than the generic WNA16 group-scale support in [#48918](https://github.com/vllm-project/vllm/pull/48918) may need the group-32 keys added to `_supports_quant_scheme`.

## Kernel optimizations

### Indexed kernels

The indexed family derives from public [`inclusionAI/humming`](https://github.com/inclusionAI/humming) commit [`4351af3`](https://github.com/inclusionAI/humming/commit/4351af3a8fcdce1a8dee50104ba49566af2427fb). The workload regimes below motivate different kernel profiles, selected before weights are packed:

<p align="center">
<img src="/assets/figures/2026-09-15-novita-chord-w4a16-moe/shape-regime.svg" alt="Typical prefill and decode workloads motivating different kernel profiles, with many and few routed rows per expert respectively" width="100%">
</p>

_Figure 2. Typical prefill and decode workloads. The 9–15 rows/expert label illustrates a decode test case; 80 tokens/expert is a prefill block-M heuristic threshold. Neither defines a runtime switch between prefill and decode: profiles and weight layouts are fixed at model load, while token counts tune the schedule within each profile._

#### H200 prefill and TP8

- **Batched `wait<1>` WGMMA pipelining.** One WGMMA group stays in flight while the next load and dequantization proceed, worth about 3–6% on gate/up and 1–5% on down across the published sweep. Output is bit-identical; the mechanism is described below.
- **Tokens-per-expert block-M selection.** Indexed MoE padding and register pressure are governed by routed tokens per expert (`tok_e`), not only total routed M. The H200 EP8 resolver models that quantity and keeps a separate, flatter set of windows for TP8.
- **A bounded 2-CTAs/SM window.** For the mid-size tiles where one CTA is latency-bound, a 128-register launch-bound cap raises resident warps and hides `cp.async` gather plus dequantization. The policy is applied only in the measured block-M/block-N window; outside it the original occupancy choice is retained.
- **Shape-aware stream-K gating.** The mid-K down projection disables stream-K once the ordinary M×N grid is full, avoiding split/reduction overhead. Deep-K gate/up and the TP8 projection-specific crossovers retain it where it helps.

The `tok_e` rule is deliberately simple to explain but specific to the MoE shape. Below roughly 80 routed tokens per expert, the resolver keeps the baseline block-count search; above that point it sizes `block_m` around each expert's padded rows and the register ceiling. TP8 uses flatter windows because its narrow intermediate dimension leaves fewer N tiles to fill an SM:

```python
# Conceptual form of the H200 EP8 indexed prefill heuristic.
tok_e = routed_m / num_experts
if tok_e < 80:
    block_m = argmin_total_blocks(sampled_routing)
else:
    block_m = fit_padded_expert_rows(tok_e, max_block_m=176)
```

The WGMMA mainloop also batches its asynchronous dependency management. Instead of waiting for every instruction group, it commits after a warp-K iteration and keeps one group in flight while the next shared-memory load and INT4 dequantization start:

```text
# Simplified steady state; prologue and stage management omitted.
for warp_k in K_tiles:
    load_next_packed_weights_and_scales()  # shared memory -> registers
    issue_wgmma_for_iteration(warp_k)
    commit_group()
    wait_group<1>()     # one group may remain in flight
    dequantize_next_in_alternate_buffer()  # dequant + group scale
epilogue:
    wait_group<0>()
```

Double-buffered weight registers let the next load and dequantization overlap with the outstanding WGMMA group. The accumulator is not consumed until the epilogue, and the final drain still waits for every outstanding WGMMA operation.

#### H200 and Blackwell indexed decode

At a few routed rows per expert, the WGMMA path is barrier-bound. The decode profile swaps the MMA operands so dequantized weights occupy the MMA-M operand, uses `m16n8k16`, and supports 4 CTAs/SM with block-M 8. A semi-static token-tile schedule measured 186 µs versus 216 µs for the fully dynamic schedule at 9–15 tokens/expert. Fusing subtract-then-scale dequantization into nibble extraction preserves the unfused BF16 rounding order. The same MMA instruction family is compiled for SM100/SM103; larger Blackwell decode shapes use wider non-swapped MMA tiles. No tcgen05 kernel is required for these token counts.

### Grouped SM90 kernels

The grouped backend is a different kernel family, not a second name for the indexed kernel. It specializes DeepGEMM's Hopper GEMM infrastructure for W4A16 and adapts it to Chord's JIT and launcher. Both modes use TMA, warp-specialized WGMMA and group-32 dequantization, but their routing and physical weight layouts differ:

<p align="center">
<img src="/assets/figures/2026-09-15-novita-chord-w4a16-moe/row-layouts.svg" alt="Per-expert rows in three layouts, showing where padding and unused row budget appear" width="100%">
</p>

_Figure 3. Where padding lives. Indexed leaves activations unpadded; its routing indices carry padding sentinels. Contiguous pads each expert to a 128-row boundary; masked reserves a fixed row budget per expert._

- **Contiguous prefill:** rows are concatenated by expert, padded to 128-row boundaries, and accompanied by `m_indices` (`int32`, with `-1` for padding). Inputs are `[m, K]`; the packer uses a bit-permuted INT4 buffer with `BLOCK_K=64` and transposes scales to `[G, K/32, N]` (N contiguous).
- **Masked decode:** activations have a fixed per-expert row budget (`[G*max_m, K]` or `[G, max_m, K]`) and `masked_m`/`expert_layout` carries the valid count. Its packer uses `BLOCK_K=128`; the heuristic chooses `BLOCK_M` from expected tokens per expert, gates `BLOCK_N` by wave occupancy, and tunes buffered-K stage depth.

> Grouped operator entry points (Chord API only):

```python
from chord_kernels import contiguous, masked
from chord_kernels.operator import pack_w4a16_grouped

# weight: unsigned INT4 codes [G, N, K]; scale: BF16 [G, N, K/32]
prefill_weight = pack_w4a16_grouped(weight, scale, mode="contiguous")
prefill_out = contiguous(a2, prefill_weight, m_indices)  # [m, N]

decode_weight = pack_w4a16_grouped(weight, scale, mode="masked")
decode_out = masked(a3, decode_weight, masked_m, expected_m)  # [G*max_m, N]
```

Here `expected_m` is a positive Python integer used for launch selection; `masked_m` holds the authoritative per-expert valid counts. The masked output is flat even when `a3` is three-dimensional, and consumers must ignore rows beyond each expert's valid count.

The mode is recorded in the prepared weight and checked at dispatch, so accidentally feeding a prefill-packed weight to the decode kernel fails loudly. Grouped dispatch owns the SM90 layout search and does not accept indexed `block_m` or `tuning_config` overrides. Kernel resolution and cubin loading are memoized by descriptor (and the `CHORD_W4A16_*` tuning overrides), removing the repeated host-side search measured at about 30 µs in small decode launches.

The grouped mainloop is persistent and warp-specialized: a producer warpgroup uses TMA to stage activation, packed weight and scale tiles, while consumer warpgroups execute WGMMA and write the BF16 result. The forward path sees already permuted INT4 bytes and MN-major scales, and the cached descriptor maps each `(mode, M, N, K, expert_count)` shape to its cubin without repeating the layout search on every decode call.

The grouped heuristic has a few choices that are specific to the W4A16 workload:

- **Contiguous prefill uses BM128/BK64 when the grid is large enough.** BM128 amortizes INT4 dequantization and scale promotion over more rows, while BK64 keeps each pipeline stage small enough to leave room for several stages in shared memory. A small concatenated problem falls back to BM64 so the M tiles can still fill the SMs; BM128/BK128 would consume too much shared memory and collapse the pipeline.
- **Masked decode sizes BM from K and the expected routed tail.** A masked group can spill into a second M tile, which rereads the whole K dimension. For deep-K gate/up, the heuristic therefore covers roughly `1.3 * expected_m` rows to avoid that reread. For short-K down, the extra pass is cheaper, so a leaner `ceil(1.25 * expected_m, 8)` tile leaves more room for pipeline stages.
- **Masked BN is wave-aware.** BN256 improves dequant amortization, but only helps when enough N tiles exist to keep the machine busy. The resolver keeps BN128 for under-filled waves, including the narrow EP32 gate/up case, and keeps BN128 for large BM on short-K down. Deep-K gate/up can still use BN256 when enough tiles fill the machine.
- **Buffered-K depth is latency-tuned rather than maximized.** Decode normally targets about 512 buffered K elements (`512 / BLOCK_K` stages); large masked tiles target about 768, subject to shared-memory limits. Filling all available shared memory would make barrier recycling more expensive without improving a one-block-per-SM decode launch.

These rules are why grouped does not reuse the indexed tuning table: the grouped backend selects `(BM, BN, BK, cluster, stages)` from the actual mode and shape at dispatch time. Within the H200 EP8 range quoted above, the prefill advantage narrows at 512 rows/expert because both implementations approach the same throughput ceiling; the tile and pipeline choices matter most at small and mid-sized chunks.

## Measurements

The kernel tables use `triton.testing.do_bench` and compare each Chord path with the matching public Humming backend on the same GPU. Indexed comparisons use the same shape and routing draw; grouped comparisons match the per-expert row counts. Run the two suites to check Chord's outputs against a plain-PyTorch reference and print its timing tables on supported GPUs:

```bash
python tests/test_w4a16_indexed.py
python tests/test_w4a16_grouped.py
```

The summary below adds the gate/up and down call times from the full tables. Its speedup is `Humming (gate_up + down) / Chord (gate_up + down)`; it excludes routing, activation and communication.

| Scenario | Shape point | Humming gate_up + down | Chord gate_up + down | Layer speedup |
| --- | --- | --- | --- | --- |
| H200 EP8 indexed prefill | 2048 tokens | 701.4 µs | 587.9 µs | 1.19x |
| H200 TP8 indexed mix | 8196 tokens | 2483.4 µs | 1862.3 µs | 1.33x |
| H200 EP8 indexed decode | 20 tok/GPU | 413.8 µs | 333.8 µs | 1.24x |
| B300 EP8 indexed decode | 20 tok/GPU | 493.9 µs | 229.8 µs | 2.15x |
| H200 EP8 grouped prefill | 128 rows/expert | 1204.0 µs | 917.5 µs | 1.31x |
| H200 EP8 grouped decode | 32 tokens/expert | 666.3 µs | 493.4 µs | 1.35x |

The B300 comparison is intentionally qualified: public Humming has no SM100/SM103 tuning table, so its default time is an untuned reference. H200 indexed ratios are the tuned-to-tuned comparison. Grouped rows compare against Humming's own `grouped_contiguous`/`grouped_masked` interfaces, with matched per-expert row counts and no padding advantage given to either side.

### End-to-end serving

An earlier [serving report](https://github.com/vllm-project/vllm/pull/51815#issue-5120066062) measured the indexed TP8 path on Kimi-K2.6 with 8×H200, TP8 + DCP8, FP8 KV cache and ShareGPT requests. Both providers used the same `--quantization humming` command.

| Metric | Humming | Chord | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT | 2022 ms | 1849 ms | −8.6% |
| Prefill input + output throughput | 20,716 tok/s | 22,712 tok/s | +9.6% |
| Decode output throughput, batch 8 | 483 tok/s | 503 tok/s | +4.1% |
| Decode output throughput, batch 64 | 1650 tok/s | 1740 tok/s | +5.5% |
| Decode output throughput, batch 128 | 2514 tok/s | 2715 tok/s | +8.0% |

Prefill used one output token with prefix caching disabled. Decode reused the same prompts in a second pass with a fully warm prefix cache. The report also found no accuracy regression relative to Humming on OCRBench and GSM8K.

## What's next

1. **Complete grouped integration with vLLM's Humming backend**, making the contiguous and masked operators available through the existing framework integration.
2. **Release EP8 prefill kernels for B200/B300.** We have a working implementation with promising performance in internal tests and plan to share the kernels and benchmarks in a follow-up release.

## Try Chord

Chord is available on GitHub: [novitalabs/chord](https://github.com/novitalabs/chord). The documentation covers [getting started](https://github.com/novitalabs/chord/blob/main/docs/getting_started.md), [optimizations](https://github.com/novitalabs/chord/blob/main/docs/optimizations.md), [performance](https://github.com/novitalabs/chord/blob/main/docs/performance.md), [tuning internals](https://github.com/novitalabs/chord/blob/main/docs/tuning.md) and [benchmark methodology](https://github.com/novitalabs/chord/blob/main/docs/benchmarking.md). Feedback, issues and benchmark reports from other deployments are very welcome.

## Acknowledgements

Chord's indexed path builds on [inclusionAI/Humming](https://github.com/inclusionAI/humming), while the grouped SM90 backend specializes [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)'s Hopper GEMM infrastructure for W4A16. Chord is released under Apache-2.0. The repository's [source notes](https://github.com/novitalabs/chord/blob/main/chord_kernels/operator/SOURCE.md) record the retained upstream components and notices.

We would like to thank the Novita AI team for building and open-sourcing Chord, and the vLLM maintainers and broader vLLM community for the discussions, reviews, and quantization and MoE backend infrastructure that made this integration possible.
