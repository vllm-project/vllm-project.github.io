---
layout: post
title: "Day 0 support for Mistral Large 4 on vLLM"
author: "Mistral and the vLLM Team"
summary: "Day-0 vLLM support for Mistral Large 4 with sparse MLA, cross-layer KV sharing, DSpark speculative decoding, DCP/PCP long context and NIXL P/D on NVIDIA Blackwell."
image: /assets/figures/2026-10-29-mistral-large-4/layer_pattern.png
tags:
  - model-support
  - speculative-decoding
  - disaggregated-serving
---

We're excited to announce day-0 support for **Mistral Large 4** in vLLM. Mistral Large 4 is
Mistral's new frontier mixture-of-experts native multimodal model. It combines a DeepSeek-style sparse attention
stack (MLA plus a DSA top-k indexer) with several new architecture features: cross-layer KV
sharing, NoPE layers, SSMax and LeRoPE. It ships with a built-in DSpark speculative decoding
drafter and a Pixtral vision encoder.

Supporting it took more than a model file. ML4 runs on vLLM's model runner V2 (MRV2) with breakable
CUDA graphs. Its building blocks live in a new `vllm/models/mistral/` package, and the work produced
generic improvements to decode context parallelism (DCP), prefill context parallelism (PCP),
DSpark, NIXL-based disaggregated serving and HiSparse host-memory KV offload.

## Highlights

- **Architecture:** 72-layer MoE with 128 routed experts (top-4) plus 1 shared expert and a
  hidden size of 6144. Attention is MLA (KV LoRA rank 512, RoPE dim 64) with a DSA sparse indexer
  that selects the top 2048 keys per query. 1 trillion parameters in total, 49 billion active per
  token (52 billion including embeddings and output layers).
- **Checkpoints:** FP8 (block-scaled weights) and NVFP4 (NVFP4 experts, FP8 attention/shared
  experts). Both use an FP8 latent KV cache (`fp8_ds_mla`) by default.
- **Multimodal:** a Pixtral vision encoder for image inputs, including multiple images per prompt.
- **Speculative decoding:** a built-in DSpark drafter (3 sliding-window MLA layers, depth 3)
  accepts ~2.9 tokens per step and gives about 2x output throughput on TP8.
- **Long context:** DCP gives ~7.9x more KV capacity on TP8. PCP+DCP prefill cuts time to first token (TTFT) by up to 1.6x. DSpark works under both.
- **Disaggregated serving:** NIXL prefill/decode (P/D) disaggregation with DP/EP, TP and PCP×DCP prefill
  nodes, with DSpark on both sides.
- **HiSparse:** host-memory sparse KV offload, now with support for cross-layer KV sharing.
- **AMD:** day-0 on AMD Instinct MI350 Series GPUs with ROCm, including DSpark, image inputs and expert parallelism with MXFP4 experts.

## Usage

Mistral Large 4 support ships in the `vllm/vllm-openai:mistral-large-4` Docker image, which also
includes `mistral-common` 1.13. The commands below run `vllm serve` inside it:

```bash
docker pull vllm/vllm-openai:mistral-large-4
```

**FP8 on 8 GPUs:**

```bash
vllm serve mistralai/Mistral-Large-4-1T-A52B \
  --tensor-parallel-size 8 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 3}' \
  --tool-call-parser mistral --enable-auto-tool-choice \
  --reasoning-parser mistral
```

**NVFP4 on 4 GPUs** (Blackwell; use `--tensor-parallel-size 8` on 8 GPUs):

```bash
vllm serve mistralai/Mistral-Large-4-1T-A52B-NVFP4 \
  --tensor-parallel-size 4 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 3}' \
  --tool-call-parser mistral --enable-auto-tool-choice \
  --reasoning-parser mistral
```

NVFP4 does not trade quality for speed. On multimodal, long-context, agentic coding, business
automation and cyber evals it stays within 2.1 points of FP8 on every task, and the mean difference
is −0.3 points. Token consumption is matched too: mean completion tokens stay within about 4% of FP8.
It is also faster and lighter: at TP8, NVFP4 decodes 1.08–1.19x faster per token than FP8 without
spec decoding (1.18–1.22x with DSpark) and holds ~1.5x the KV cache.

<iframe class="vllm-embed" src="/assets/interactive_pages/ml4-nvfp4-quality.html" title="Mistral Large 4 NVFP4 vs FP8 checkpoint scores and completion tokens on core evals" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 720px; border: 0; overflow: hidden;"></iframe>
*Comparison of FP8 versus NVFP4 scores and token consumption on core evaluations. Error bars show 95% confidence intervals estimated by bootstrapping. Across all domains, including multimodal reasoning (GDP.pdf, MathVision), long-context (Graphwalks 512k), coding agents (DeepSWE, SWE-bench Pro, Terminal-Bench 4), business automation (AutomationBench) and cyber (CyberGym), the NVFP4 checkpoint matches FP8 behaviour.

**Long context (DCP).** Add `--decode-context-parallel-size 8` to shard the latent KV cache across
the 8 ranks. It requires the default FP8 KV cache.

**Multimodal.** Pass `--limit-mm-per-prompt '{"image": N}'` to allow up to N images per prompt.

For multi-node, disaggregated P/D and other tested configurations, see the
[Mistral Large 4 recipe](https://recipes.vllm.ai/mistralai/Mistral-Large-4-1T-A52B).

## Implementation details

### Model architecture

| component | Mistral Large 4 |
|---|---|
| layers / hidden size / heads | 72 / 6144 / 64 |
| attention | MLA, no query LoRA, KV LoRA rank 512, RoPE dim 64 |
| sparse attention | DSA indexer, top-k 2048, 32 index heads; 44 layers own an indexer, 28 reuse the top-k of the nearest earlier indexer layer |
| positional encoding | YaRN + LeRoPE (learned per-frequency scaling); every 4th layer is NoPE |
| query scaling | SSMax (length-dependent query scale) on every attention layer and indexer |
| KV sharing | 9 layers (39, 43, …, 71) read layer 35's latent cache (1 every 4) |
| MoE | layer 0 dense; layers 1–71: 128 experts, top-4, 1 shared expert |
| output | logit soft cap 30 |
| vision | Pixtral encoder (30 layers, hidden 2048, patch 14) |

<iframe class="vllm-embed" src="/assets/interactive_pages/ml4-arch-explorer.html" title="Mistral Large 4 architecture explorer: per-layer attention, indexer, KV cache and FFN" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 800px; border: 0; overflow: hidden;"></iframe>
*The 72 decoder layers: every 4th layer is NoPE, 28 layers reuse the top-k of the closest preceding indexer layer, the NoPE layers 39–71 read layer 35's latent KV, and only layer 0 is dense.*

What it took to support each piece in vLLM:

**Sparse MLA + DSA indexer.** ML4 reuses vLLM's sparse-MLA stack (FlashMLA sparse with the
`fp8_ds_mla` latent cache, plus the sparse indexer with an FP8 indexer cache). ML4-specific
differences are layered on top: shared-indexer layers that reuse the top-k of the nearest earlier indexer layer, an
indexer with its own SSMax, and NoPE indexers on NoPE layers. Per token, the KV cache stores
47,136 bytes: 41,328 for the MLA latent and 5,808 for the indexer.

**Cross-layer KV sharing.** Nine NoPE layers read the latent KV written by layer 35 instead of
writing their own. We added `kv_sharing_target_layer_name` to `MLAAttention` and
`AttentionLayerBase`. Consumer layers skip the cache write, and the V2 runner resolves them to the
target layer's KV cache spec. Consumers take no KV memory: 63 of the 72 layers own a cache.
This is model-agnostic and is being upstreamed in
[vllm#64095](https://github.com/vllm-project/vllm/pull/64095).

**NoPE, SSMax and LeRoPE.** NoPE layers are expressed as MLA with zero RoPE dims, so they use the
same kernels and cache layout. LeRoPE is folded into the YaRN frequencies. SSMax is fused into the
MLA RoPE kernel (see fusions below).

**A Mistral model package.** Rather than adding flags to `deepseek_v2.py`, the port builds on
reusable Mistral components under `vllm/models/mistral/`: attention, indexer, SSMax, MLP, MoE,
fused ops and a per-layer spec (`common/layer_spec.py`). The ML4 model itself is
`vllm/models/mistral_large4/nvidia/model.py`, and the DSpark drafter sits next to it
(`dspark.py`). Per-layer features (RoPE/NoPE, indexer owner/sharer, KV owner/consumer) come from
a layer spec built from `params.json`, not from model-specific branches in shared code.

**Model runner V2.** ML4 runs only on MRV2. The V2 attention utilities discover KV-sharing layers
and look up their target specs. DSpark plugs into the V2 DFlash/DSpark speculator.

**Breakable CUDA graphs and fusions.** ML4 runs without `torch.compile`. It uses breakable CUDA
graphs, enabled automatically for the architecture (also when the text model sits inside the
multimodal wrapper). They gave about a 6x speedup over eager mode.
Hand-written fusions replace what the compiler would otherwise do, while enabling further optimizations:

- First round: SSMax, MLA RoPE + SSMax, the indexer Q path and the indexer K path. These cut decode
  TPOT by 8–10%.
- Decode round: FP32 router GEMM, a one-launch Triton LayerNorm for the indexer query,
  all-reduce + residual + RMSNorm in one FlashInfer kernel, and indexer K written straight into the FP8 cache. On TP8 FP8,
  c1 TPOT dropped from 10.22 to 9.00 ms (-12%) and c8 from 16.01 to 15.08 ms.

### Long context: decode and prefill context parallelism

**DCP.** MLA keeps one latent KV head, so under TP the cache is replicated on every rank. DCP shards it
by token instead. On TP8, DCP8 raises KV capacity from 2.38M to 18.88M tokens (FP8) and from 3.60M to 28.76M
tokens (NVFP4).

![Long-context capacity on 8x GB300: TP8 vs TP8 DCP8](/assets/figures/2026-10-29-mistral-large-4/long_context_capacity.png)
*Concurrent 128k / 256k / 1M-token requests whose KV cache fits on 8× GB300 with ML4 NVFP4, TP8 vs TP8 DCP8: floor(KV pool / input length) - 3.60M and 28.76M tokens, gpu-memory-utilization 0.85.*

Out of the box, DCP with sparse MLA paid a large fixed cost per decode step. A profile at 60k
context showed about 7 ms per step of DCP-only work:
- NCCL all-gathers of the LSE and of the indexer top-k candidates (44 indexer layers);
- reduce-scatters;
- a stable top-k merge;
- query gathers;
- small mask kernels and launch gaps.

We replaced these with:

- **a2a combine** for FlashMLA sparse (merged as [vllm#61285](https://github.com/vllm-project/vllm/pull/61285)):
  a symmetric-memory all-to-all LSE combine instead of all-gather + reduce-scatter. ML4 uses it
  by default.
- **Direct top-k exchange:** each rank pushes its packed (score, id) candidates into its peers'
  symmetric-memory inboxes, and a selector reads them in place. This replaces an NCCL all-gather
  and a 6-pass radix select per indexer layer.
- **Top-k compaction:** for batches of 32+ tokens, each rank compacts its valid top-k slots and
  bounds the kernel with `topk_length`, instead of scanning 2048 slots full of `-1`. In a 5k-token
  prefill profile, attention went from 165 ms to 38 ms.
- A fused kernel neutralizes empty rows (a rank that owns none of a query's top-k slots).

The DCP-specific overhead fell to 3.2 ms per step. Together with the decode fusions, on TP8 DCP8 FP8
(60k input / 1k output):

| | before | after |
|---|---:|---:|
| c1 TTFT p50 | 6.89 s | 4.73 s |
| c1 TPOT p50 | 17.82 ms | 12.50 ms |
| c64 output tok/s | 139.1 | 204.0 (+47%) |

![DCP8 before and after the collectives port](/assets/figures/2026-10-29-mistral-large-4/dcp_before_after.png)
*TTFT and TPOT p50 vs concurrency on TP8 DCP8 (60k input / 1k output, no spec), before (a2db3f0b9d) and after (bbe5926511) the DCP collectives port and decode fusions.*

These DCP optimizations ship with Mistral Large 4 support on day 0.

**When to use DCP.** DCP trades per-step latency for KV capacity. On the same 8x GB300 NVFP4
deployment and workload (no spec decoding), DCP8 decodes about 16–28% slower per token than TP8:
c1 TPOT is 11.1 vs 8.7 ms at 8k input and 11.4 vs 9.1 ms at 60k. With up to 128 concurrent
requests of 8k–60k tokens, TP8's 3.6M-token pool is never the bottleneck, so TP8 delivers more
throughput there. DCP8 pays off once KV is the limit: very long contexts, or many concurrent
long-lived sessions. In our agentic P/D runs (~100k-token sessions), a TP4 DCP4 decode node kept
TTFT at 0.7 s at 32 sessions, where a plain TP4 decode node ran out of KV and climbed to 15 s.

**PCP + DCP prefill.** For long prompts, PCP splits one prompt's tokens across ranks. PCP+DCP
now works on ML4 with CUDA graphs, including DCP > PCP and DSpark. Two fixes made this work:
- Under PCP the MLA KV-cache write first all-gathers its inputs across PCP ranks, and that gather
  depends on the batch's prefill/decode split. Captured in a piecewise graph, the split was frozen
  at capture time. Now only the gather runs as an eager break, into fixed-size buffers, and the
  cache write itself stays in the graph.
- The DCP group's ranks are ordered to match the order collectives return data in.

With graphs on TP1×PCP4×DCP4, GSM8K-200 went from 0.315 to 0.975. With DSpark it reaches 0.980.
As a P/D prefill node, PCP4×DCP4 gives the lowest TTFT we measured (see the P/D section).

### DSpark speculative decoding

ML4 ships a DSpark drafter inside the checkpoint (`mtp_dspark.*` tensors and an `mtp` block in
`params.json`). DSpark is a parallel drafter: a small non-causal transformer proposes all draft
positions of a block in one forward pass. It conditions on hidden states tapped from target
layers 1, 18, 36, 54 and 71. ML4's drafter adds:

- **an RNN Markov head:** a gated recurrent state (rank 512) runs across the draft positions, so
  each position's logits depend on the tokens drafted before it;
- **position-specific mask inputs:** each draft slot gets its own learned input embedding instead
  of one shared mask token;
- **sliding-window MLA draft layers:** 3 MLA layers with a 2048-token window and a BF16 KV cache,
  in their own KV cache group. The Triton MLA backend gained sliding-window decode for them.

<iframe class="vllm-embed" src="/assets/interactive_pages/ml4-dspark-steps.html" title="One DSpark drafting step in Mistral Large 4, step by step" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 705px; border: 0; overflow: hidden;"></iframe>
*One DSpark step: target taps feed the draft's context K/V, the [anchor, mask_1, mask_2] block runs through 3 sliding-window MLA layers in one pass, the RNN head emits 3 draft tokens in sequence, and the target verifies them.*

Enable it with `--speculative-config '{"method": "dspark", "num_speculative_tokens": 3}'`.

**Results.** TP8, 200 real prompts, greedy (temperature 0), up to 512 output tokens:

| | FP8 base TPOT | FP8 DSpark TPOT (AL) | NVFP4 base TPOT | NVFP4 DSpark TPOT (AL) |
|---|---:|---:|---:|---:|
| c1 | 8.84 ms | 3.50 ms (2.90) | 8.21 ms | 2.95 ms (2.93) |
| c8 | 15.28 ms | 7.08 ms (2.85) | 12.85 ms | 5.80 ms (2.86) |
| c32 | 29.92 ms | 11.57 ms (2.85) | 25.38 ms | 9.80 ms (2.86) |

The table uses greedy decoding. With sampling, acceptance stays close: at temperature 1.0 /
top-p 0.95, AL is ~2.8 at concurrency 8. At higher temperatures, set
`"draft_sample_method": "probabilistic"` in the speculative config: at T=1.5 it raises AL from
2.04 to 2.36–2.53 (+12–22% throughput). Both modes are lossless thanks to standard rejection
sampling.

At the same concurrency, DSpark raises output throughput 1.9–2.1x on FP8 and 1.9–2.2x on NVFP4.
It doesn't change what the model generates: every drafted token is verified against the target model,
and we checked this empirically. When the base model re-scores DSpark's outputs, they match its own
greedy choices as often as the base model's own outputs do. On GSM8K (1,319 questions), FP8 scores
0.909 without DSpark and 0.918 with it.

<iframe class="vllm-embed" src="/assets/interactive_pages/ml4-dspark-pareto.html" title="DSpark throughput vs interactivity: output tok/s per GPU vs tok/s per user, with and without DSpark" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 650px; border: 0; overflow: hidden;"></iframe>
*Output tok/s per GPU vs tok/s per user (1000 / TPOT p50), without (dashed) and with DSpark (solid, depth 3). Short: the 200-prompt TP8 runs in the table above. 8k / 32k / 60k: ML4 NVFP4 colocated on 8× GB300, TP8 and TP8 DCP8, `vllm bench serve` with real-text prompts and 1k output tokens; at 60k only the DSpark runs exist.*

**DSpark + DCP.** Before this work, sliding-window MLA drafts could not run under DCP. Now the
draft's small window cache (3 layers × ~2k tokens, about 7 MB per request) is replicated on every
DCP rank, while the target cache stays DCP-sharded. This extends the existing
support (vllm#56723) to windowed MLA drafts. On TP8 DCP8 at 60k context, c1 TPOT goes from 11.82 to 
5.22 ms (2.26x, AL 2.85), and GSM8K-200 stays at 0.965.
The cost is KV capacity: under DCP8 each 64-token draft block holds a block ID whose target
pages cover 512 tokens. At 64k context, capacity falls to about 0.69x of plain DCP8. That limit
doesn't bind at `max_num_seqs` 128 below ~99k average context.

**DSpark in P/D.** A DCP prefill node can hand off to a TP decode node with DSpark on both sides.
NIXL pulls the replicated draft window from one prefill rank and the target KV in DCP slices.
Acceptance is the same with any P/D layout: ~2.98 / 2.86 / 2.78 at 8k / 32k / 60k input.

Sliding-window MLA draft layers and DSpark under DCP are being upstreamed in
[vllm#63795](https://github.com/vllm-project/vllm/pull/63795) and
[vllm#63796](https://github.com/vllm-project/vllm/pull/63796).

### Disaggregated serving with NIXL

We benchmarked 1P1D on 4+4 GB300 GPUs (one NVL72 rack) with NIXL. Setup:
- NVFP4 with DSpark on both sides, CUDA graphs everywhere, no prefix caching;
- `vllm bench serve` with fixed-length prompts: 320 unique single-turn chat prompts per input length,
  each exactly 8,192 / 32,768 / 60,000 tokens. Each prompt is a natural-text passage (WikiText-2 and
  GSM8K problems) followed by a question, with no shared prefix between prompts, sent as token IDs;
- 1,024 output tokens per request (greedy, EOS ignored);
- a concurrency sweep and a fixed-QPS sweep.

All configurations scored 0.965–0.980 on GSM8K-200 through the proxy, with zero failed NIXL
transfers and zero failed requests.

<iframe class="vllm-embed" src="/assets/interactive_pages/ml4-pd-pareto.html" title="ML4 NVFP4 + DSpark 1P1D Pareto: output tok/s per GPU vs tok/s per user" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 640px; border: 0; overflow: hidden;"></iframe>
*Output tok/s per GPU vs tok/s per user, 1P1D on 4+4 GB300. Solid lines are concurrency sweeps; hollow/dotted markers are fixed-QPS sweeps.*

Best output tok/s per GPU (8 GPUs) at a given minimum per-user speed:

| P → D | 8k, ≥40 tok/s/user | 8k, ≥100 | 32k, ≥40 | 32k, ≥100 | 60k, ≥40 / ≥100 | c1 TTFT 8k / 32k / 60k |
|---|---:|---:|---:|---:|---:|---:|
| P DP4/EP → D DP4/EP | **718.2** | 176.7 | 202.6 | 139.2 | **108.8** | 0.49 / 1.90 / 3.65 s |
| P DP4/EP → D TP4 | 459.2 | **194.2** | **205.2** | **150.0** | 107.2 | 0.46 / 1.88 / 3.66 s |
| P PCP4×DCP4 → D TP4 | 350.6 | 192.6 | 91.5 | 91.5 | 49.8 | **0.35 / 1.37 / 2.57 s** |
| P TP4 → D TP4 | 259.2 | 185.3 | 62.8 | 62.8 | 32.3 | 0.52 / 2.07 / 4.02 s |

## AMD

Mistral Large 4 runs on AMD Instinct MI350 Series GPUs from day 0 with the ROCm build of vLLM. The sparse MLA runs on AITER's
MLA decode kernels and the DSA indexer on AITER's gfx950 indexer.

**MXFP4 experts.** On ROCm the routed experts run as OCP MXFP4 on AITER's fused MoE. There are two ways to get them:

- **AMD Quark MXFP4 checkpoint** ([Mistral-Large-4-1T-A52B-Quark-MXFP4](https://huggingface.co/amd/Mistral-Large-4-1T-A52B-Quark-MXFP4)). Quark re-quantizes only the routed
  experts, with static weight and dynamic activation scales. Attention, the shared experts, the dense feed-forward layers, the
  router, the output head, the vision encoder and the DSpark drafter keep their original precision. The checkpoint shrinks
  from about 1,061 GB to 578 GB, so it serves on four MI350 Series GPUs (288 GB each). The serving commands, evaluation results
  and the reproduction recipe are on the model card.
- **Re-quantize the FP8 checkpoint at load** (`VLLM_FP8_MOE_REQUANT=mxfp4`). The experts are converted layer by layer while
  loading, so peak memory stays low enough for TP4 as well. Expert weights drop from about 125 to 69 GiB per GPU at TP8, and
  output throughput rises 22–71% over serving the FP8 experts at 1–64 concurrent requests. `VLLM_FP8_MOE_REQUANT_ACT` sets
  the activation format the MXFP4 experts compute with: `a8w4` (FP8 activations, the default) or `a4w4` (MXFP4 activations).

Either way, one 8-GPU node serves two TP4 replicas, each with about 6× the HBM left for KV cache and activations that the FP8
experts would leave on the same four GPUs.

**Features.** TP8 and TP4, expert parallelism, image inputs through the Pixtral encoder, and DSpark speculative decoding with
the checkpoint's built-in drafter are supported on ROCm. On MI350X at TP4 with the FP8 KV cache, DSpark accepts 3.1 tokens
per step on real prompts and lowers per-token decode latency 2.7× at one request, 1.9× at 8 and 1.8× at 32 (output
throughput 2.1×, 1.6× and 1.4×). On ROCm the latent KV cache is FP8 (E4M3) by default, which gives
1.86× the KV capacity of bf16; `--kv-cache-dtype bfloat16` opts out.

**AITER.** `VLLM_ROCM_USE_AITER=1` runs vLLM's hot paths on AITER, AMD's kernel library: the fused MoE (the MXFP4 experts
on two-stage FlyDSL kernels), the block-FP8 and bf16 GEMMs, the MLA decode kernels, RMSNorm and the tensor-parallel custom
all-reduce. The ROCm image ships AITER tuning tables for this model's MoE and GEMM shapes, which AITER merges with its own at
startup; at TP4 they raise output throughput 9–16% at 1–8 concurrent requests. Two more settings help on top:
`VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1` runs the sparse MLA on AITER's Triton kernel, and
`AITER_CUSTOM_AR_MAX_SIZE=134217728` keeps the all-reduces of large prefill steps (up to 128 MiB) on AITER's custom
all-reduce. Together they raise output throughput a further 2–5% at TP4 and lower time to first token 5–7%. Both commands
below set all three.

**Specialized decode.** For latency-bound serving, `--additional-config '{"decode_backend": "latency_specialized"}'` runs
every MoE layer of a 1-, 2- or 4-request decode step as one persistent FlyDSL kernel launch. The launch fuses attention, the
indexer, the router, the MXFP4 experts and both tensor-parallel all-reduces. On MI350X at TP4 it lowers per-token decode
latency 1.5× at one request and 1.1× at four compared with vLLM's standard decode on the same GPUs. It composes with
DSpark: a single request's verify block runs on the fused kernels, for 2.9× lower per-token latency than the standard decode
without speculation. Larger batches and prefill run the standard path. It runs on the default FP8 KV cache, requires TP4 and
`VLLM_FP8_MOE_REQUANT_ACT=a4w4`, and refuses to start otherwise, giving the reason. The kernel is adapted from ROCm
ATOM's GLM-5.2 fused decode.

**Serving on MI350 Series.** Both commands enable AITER and re-quantize the FP8 experts to MXFP4 at load:

```bash
# TP8, one replica per node
VLLM_ROCM_USE_AITER=1 VLLM_FP8_MOE_REQUANT=mxfp4 VLLM_FP8_MOE_REQUANT_ACT=a8w4 \
VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1 AITER_CUSTOM_AR_MAX_SIZE=134217728 \
vllm serve mistralai/Mistral-Large-4-1T-A52B --tensor-parallel-size 8 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 3}' \
  --tool-call-parser mistral --enable-auto-tool-choice --reasoning-parser mistral

# TP4 with the specialized decode, for few concurrent users; two replicas per node (second on GPUs 4-7)
VLLM_ROCM_USE_AITER=1 VLLM_FP8_MOE_REQUANT=mxfp4 VLLM_FP8_MOE_REQUANT_ACT=a4w4 \
VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1 AITER_CUSTOM_AR_MAX_SIZE=134217728 \
vllm serve mistralai/Mistral-Large-4-1T-A52B --tensor-parallel-size 4 \
  --additional-config '{"decode_backend": "latency_specialized"}' --max-num-seqs 64 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 3}' \
  --tool-call-parser mistral --enable-auto-tool-choice --reasoning-parser mistral
```

### ROCm implementation details

**Sparse MLA on AITER.** On ROCm, ML4's sparse MLA runs on the `ROCM_AITER_MLA_SPARSE` backend. The indexer's top-2048
positions are converted to global cache slots, and each query token becomes its own entry in the slot stream. AITER's
MLA decode kernel then reads that stream in persistent mode, with work-split metadata built ahead of time to balance the
load across CUs. AITER's MLA kernels take a multiple of 16 heads: TP4 gives exactly 16 of ML4's 64 heads per GPU, while
TP8's 8 are padded to 16. `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1` switches to AITER's Triton sparse-MLA kernel instead,
which needs no head padding and no separate FP8 copy of the query. NoPE layers on the bf16 cache have no RoPE part for
AITER's kernel to read, so they run on vLLM's ragged Triton sparse-attention kernels.

**DSA indexer on AITER.** A fused Triton kernel quantizes each indexer key to FP8 with one scale and writes it into the
indexer cache, 132 bytes per token per indexer layer. Within a cache block, keys are stored in 16×16 tiles, the layout
AITER's paged reader expects. A single layout rule is shared by the writer and every reader (insert, prefill gather and
paged logits), so they cannot disagree. Prefill scores run on AITER's `fp8_mqa_logits` and decode scores on AITER's paged
MQA-logits kernel, both reading FP8 keys. The top-2048 comes from AITER's indexer top-k where it supports the shape, and
from vLLM's `top_k_per_row` otherwise. Shared-indexer layers reuse the top-k of the nearest earlier indexer layer, as on
other platforms.

**FP8 latent cache.** ROCm has no `fp8_ds_mla` layout, so when the checkpoint asks for an FP8 KV cache, ROCm uses a plain
E4M3 cache instead. All 576 latent dimensions (512 NoPE and 64 RoPE) are stored as FP8 with one per-tensor scale, 576 bytes
per token per layer, against 656 for `fp8_ds_mla` and 1,152 for bf16. The query is quantized to FP8 too, so AITER's kernel
runs FP8 × FP8. With 63 cache-owning layers plus the indexer, a token takes 42,096 bytes, against 78,384 in bf16: the 1.86×
KV capacity in the AMD section.

**MXFP4 experts at load, one layer at a time.** With `VLLM_FP8_MOE_REQUANT=mxfp4`, the checkpoint loader is unchanged: it
still loads block-FP8 expert weights with their block scales. Their parameters start on the meta device, and the layerwise
loader materializes one layer at a time. As soon as a layer's experts are loaded, each expert is dequantized to bf16 and
re-quantized to MXFP4 (E2M1 values, one E8M0 scale per 32 values), and that layer's FP8 copy is freed. Peak memory is the
MXFP4 model plus one layer of FP8 experts, which is what lets the 1T model load on four GPUs. The weight error is about
the same as quantizing the bf16 weights to MXFP4 directly. The experts then run on AITER's fused MoE: two-stage FlyDSL
kernels with FP8 activations for `a8w4`, or MXFP4 × MXFP4 for `a4w4`.

**Specialized decode: one persistent kernel per layer.** With `decode_backend=latency_specialized`, a pure decode step of
1, 2 or 4 requests (or a single request's DSpark verify block) runs each of the 71 MoE layers as one FlyDSL kernel launch.
The launch uses 256 CTAs, all resident at once on the GPU's 256 CUs. Inside it run:
- the input norm, the query and latent-KV projections with SSMax and LeRoPE;
- the whole DSA indexer: projections, key-cache write, scoring and an exact top-2048;
- sparse MLA over the latent cache, read in place in bf16 or FP8;
- the output projection, the softmax top-4 router, and the shared and routed MXFP4 experts.

The embedding, the dense layer 0 and the final norm stay on vLLM's modules.

Both tensor-parallel all-reduces run inside the kernel, over peer memory between the four GPUs, so a layer is one launch
with no host round-trip. The kernels are built for 1, 2 and 4 rows, which are vLLM's full CUDA-graph capture sizes, and are
replayed inside those graphs. Any other step runs vLLM's layers: prefill, mixed batches, wider batches. The kernel is
adapted from ROCm ATOM's GLM-5.2 fused decode, extended for ML4's layer types:
- no query LoRA;
- SSMax on the query and the index query;
- softmax routing;
- NoPE layers;
- KV-sharing layers that read layer 35's latent cache;
- index-sharing layers.

**Fail-stop.** A persistent kernel that waits on peer GPUs must never hang the server. Every in-kernel wait gives up after
10 seconds, and giving up sets a sticky fault word on every rank, which ends the other ranks' waits at once. After every
forward, a device-side assert checks the word before logits are computed. If the word is set, the step's remaining work
never runs, the worker aborts and the engine stops, so no token from a failed step reaches a client. The check costs
about 1 µs per launch.

The path is also default-deny. If a deployment uses anything the kernels don't serve, startup fails with one error listing
every reason, and all TP ranks agree on each check before anything collective runs. Unsupported features include other
TP sizes, expert or context parallelism, LoRA, KV connectors, other speculative methods, and experts not stored as `a4w4`.

## TPU

Mistral Large 4 runs on **TPU v7x from day 0** with `vllm-torchtpu`, including support for the
Pixtral vision encoder. Our validated configuration combines intra-host tensor parallelism
with cross-host data parallelism with expert parallelism across 16 TPU v7x chips.

**Serving on TPU v7x:** The following command starts a single data parallel worker on one host:

```bash
vllm serve mistralai/Mistral-Large-4-1T-A52B \
  --tensor-parallel-size 8 --data-parallel-size 4 --data-parallel-size-local 1 \
  --data-parallel-start-rank "$TPU_WORKER_ID" \
  --data-parallel-address "$HEAD_IP" --data-parallel-rpc-port 29696 \
  --enable-expert-parallel --enable-ep-weight-filter \
  --dtype bfloat16 --kv-cache-dtype fp8_e4m3
```

The TPU implementation relies on `vllm-torchtpu`'s sparse MLA and DSA indexer Pallas kernels,
extending MLA with cross-layer KV cache sharing. We store the MLA latent KV cache in FP8 and
support FP8 weight and activation quantization for dense linear projections. Routed experts use
weight-only FP4 quantization (W4A16).

## HiSparse: host-memory sparse KV offload

Sparse attention reads only 2048 keys per query, so most of a long request's KV doesn't need
to live in GPU memory. HiSparse keeps the full latent KV in pinned host memory. It keeps a
resident GPU working set plus a per-request hot buffer, and copies the selected rows from host on
a miss. Enable it with the `HiSparseConnector`:

```bash
--kv-transfer-config '{"kv_connector": "HiSparseConnector", "kv_role": "kv_both",
                       "kv_connector_extra_config": {"host_pool_gib": 440}}'
```

For ML4 we made HiSparse work with **cross-layer KV sharing**. The nine consumer layers bind to
layer 35's resident view and host copy and keep only their own hot buffer and LRU (each has its
own indexer). Nothing is duplicated, and no environment variable is needed. Compared with
materializing the shared layers, this holds 14.3% more tokens for the same host pool. We also fixed:
- reading `index_topk` from the text config of multimodal models;
- a double host mirror during CUDA graph capture.

**Capacity.** On a single GB300 node (TP4, NVFP4), the four TP ranks share one deduplicated host
pool, so `host_pool_gib` is the total. With a 440 GiB pool, HiSparse admits 8.77M KV tokens at
startup and holds up to 11.25M in steady state, vs 1.97M on the GPU alone (4.5x to 5.7x). GSM8K-200
is unchanged (0.975 both), and needle-in-a-haystack passes at ~98k (10/10) and ~248k (5/5) tokens.

![HiSparse KV capacity](/assets/figures/2026-10-29-mistral-large-4/hisparse_capacity.png)
*KV token capacity of one TP4 server on a single GB300 node (NVFP4, max_model_len 262k, prefix caching off): 1.97M tokens on the GPU alone; with HiSparse and a 440 GiB host pool shared by the four TP ranks, 8.77M admitted at startup (4.5x) and up to 11.25M in steady state (5.7x). Layers that reuse layer 35's KV are stored on the host only once.*

## What's next

- Upstreaming: MLA cross-layer KV sharing, DSpark sliding-window MLA drafts and DSpark under DCP and HiSparse KV-sharing support.
- A DCP decode node with DSpark in P/D, and a separate draft block pool so DSpark under DCP keeps
  full KV capacity.
- HiSparse with GPU landing for NIXL imports and node-shared host pools for
  multi-node TP.
- More decode fusions for low-concurrency latency.

## Acknowledgement

Mistral Large 4 support in vLLM was a collaboration led by Nicolò Lucchesi and the Mistral Inference team. We would like to extend our gratitude to the vLLM maintainers who reviewed, designed, and landed much of the underlying work: Lucas Wilkinson, Flora Feng, Matt Bonanni, and the rest of the Red Hat team; the Inferact team as a whole; Cyrus Leung as well as the entire vLLM community.

On AMD side, we would also like to thank Hongxia Yang, Felix Marty, Andy Luo, Indira Vats for Day 0 enablement and optimization on AMD Instinct GPUs, and Bowen Bao, Chuan Li, Peng Sun for general support.
