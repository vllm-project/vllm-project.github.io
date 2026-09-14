---
layout: post
title: "Kimi K3 Performance Optimizations in vLLM, 2.2–2.8× Higher Throughput"
author: "Wentao Ye, Canlin Guo, Yongye Zhu, Jiangyun Zhu, Ziming Huang, Wei Zhao, Jee Jee Li"
summary: "Kimi K3 serving optimizations across scheduling, KDA prefix caching, parallelism, memory movement, MoE, and GPU kernels."
tags:
  - models
  - performance
  - kernels
  - speculative-decoding
---

After day-0 support: scheduler, KDA prefix caching, speculative decoding, parallelism, LatentMoE, quantized GEMMs, and data movement. Tracked in [Kimi K3 Performance Optimization #50587](https://github.com/vllm-project/vllm/issues/50587).

## Performance

Measured with an 8K/1K workload with TP8, eight tokens DSpark speculation. Concurrency 1, 4, and 16. Comparison from v0.27.1 to commit `82a85dc1` (0913), tested on B300 node (CUDA 13.3).

Start the server:

```bash
vllm serve moonshotai/Kimi-K3 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser kimi_k3 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --host 0.0.0.0 \
  --port 30000 \
  --max-model-len auto \
  --no-enable-prefix-caching \
  --speculative-config '{"model":"RedHatAI/Kimi-K3-speculator.dspark","method":"dspark","num_speculative_tokens":8,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'
```

Note: Prefix caching was disabled in both runs because v0.27.1 had a known Kimi K3 prefix-caching issue fixed later.

Run the benchmark:

```bash
guidellm run \
  --backend '{"kind":"openai_http","target":"http://127.0.0.1:30000","model":"moonshotai/Kimi-K3","request_format":"/v1/chat/completions","timeout":100000,"extras":{"body":{"reasoning_effort":"max","temperature":1.0,"top_p":0.95}}}' \
  --tokenizer '{"kind":"huggingface_auto","model":"moonshotai/Kimi-K3","load_kwargs":{"trust_remote_code":true}}' \
  --data 'kind=synthetic_text,prompt_tokens=8000,output_tokens=1000' \
  --profile 'kind=concurrent,warmup=0.1,cooldown=0.1' \
  --override profile.streams '1,4,16' \
  --constraint 'kind=max_duration,seconds=150' \
  --constraint 'kind=max_errors,count=10' \
  --seed 'kind=static,value=42' \
  --metrics 'kind=generative,sample_size=20' \
  --output 'kind=json,path=guidellm-results/output_vllm_kimik3_8k1k.json'
```

| Concurrency | 0.27.1 Avg Latency (s) | 0913 main Avg Latency (s) | 0.27.1 Throughput (tok/s) | 0913 main Throughput (tok/s) | 0.27.1 Avg TTFT (ms) | 0913 main Avg TTFT (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.37 | **5.30 (−57.2%)** | 83.3 | **183.3 (+120.0%)** | 2262.9 | **376.3 (−83.4%)** |
| 4 | 23.67 | **10.50 (−55.6%)** | 166.7 | **416.7 (+150.0%)** | 2314.9 | **640.5 (−72.3%)** |
| 16 | 55.90 | **22.17 (−60.3%)** | 258.3 | **725.0 (+180.6%)** | 7601.1 | **1121.0 (−85.3%)** |

## Optimization across the stack

Hybrid KDA/MLA, Stable LatentMoE, native MXFP4, and speculative decoding. Bottlenecks across prefill, recurrent state, routing, collectives, and mixed verification batches.

### Memory layout and parallelism

Memory: latent up-projection sharding in [PR #50383](https://github.com/vllm-project/vllm/pull/50383); shared-expert sharding in [PR #50656](https://github.com/vllm-project/vllm/pull/50656); a K3-specific path in [PR #50912](https://github.com/vllm-project/vllm/pull/50912), saving 16.98 GiB per GPU in the tested configuration.

Parallelism: combined SP AllGather operations in [PR #51070](https://github.com/vllm-project/vllm/pull/51070), 1.5×–3× faster at kernel level; SP with pipeline parallelism in [PR #54347](https://github.com/vllm-project/vllm/pull/54347); overlapped low-M TP8 KDA projections in [PR #54697](https://github.com/vllm-project/vllm/pull/54697).

Distributed paths: DeepEPv2 with DeepGEMM MXFP4 in [PR #50478](https://github.com/vllm-project/vllm/pull/50478); decode context parallelism in [PR #50484](https://github.com/vllm-project/vllm/pull/50484); a shorter MLA decode concat/cache epilogue in [PR #54896](https://github.com/vllm-project/vllm/pull/54896).

### KDA and attention

Prefill: D-Spark fused KV in [PR #50585](https://github.com/vllm-project/vllm/pull/50585), 4.5×–4.6× faster; FlashKDA output in [PR #51311](https://github.com/vllm-project/vllm/pull/51311); Mamba metadata preparation in [PR #52388](https://github.com/vllm-project/vllm/pull/52388), 6.6×–7.6× faster at kernel level.

State and decode: faster SSM recovery in [PR #52993](https://github.com/vllm-project/vllm/pull/52993); DS conv-state layout in [PR #53396](https://github.com/vllm-project/vllm/pull/53396); single-token KDA PDL in [PR #53525](https://github.com/vllm-project/vllm/pull/53525)

MLA and vision: fused MoonViT Q/K complex RoPE in [PR #53168](https://github.com/vllm-project/vllm/pull/53168); grouped FP8 MLA cache insertion in [PR #55356](https://github.com/vllm-project/vllm/pull/55356), 4×–6× faster for small batches.

### MoE and GEMM

MoE: removed a MegaMoE add in [PR #51146](https://github.com/vllm-project/vllm/pull/51146); prefetched BF16 router weights for M=1 in [PR #53524](https://github.com/vllm-project/vllm/pull/53524); fused BF16 shared experts into the latent tail in [PR #53556](https://github.com/vllm-project/vllm/pull/53556).

GEMM and projections: `eh_proj` in [PR #53942](https://github.com/vllm-project/vllm/pull/53942); Hopper low-latency GEMMs in [PR #54088](https://github.com/vllm-project/vllm/pull/54088); residual skinny GEMM on SM100 in [PR #54447](https://github.com/vllm-project/vllm/pull/54447); DSV3 GEMM for inner-contiguous and row-strided tensors in [PR #54565](https://github.com/vllm-project/vllm/pull/54565); aligned NVFP4 input-projection weights in [PR #55242](https://github.com/vllm-project/vllm/pull/55242), removing an elementwise copy.

## Four end-to-end examples

### Adaptive scheduling budget

Low request counts left much of `max_num_batched_tokens` unused. [PR #51725](https://github.com/vllm-project/vllm/pull/51725): adaptive scheduled-token budget. [PR #51726](https://github.com/vllm-project/vllm/pull/51726): default limit from 8,192 to 16,384 for the high-memory GPU tier. On the reported 8K/1K workload: TTFT down 55%–65%, throughput up to 41.5%.

Let's consider `max_num_seqs=1024` and `max_num_batched_tokens=8192`, K=7

Actual Request num | Old logic scheduled tokens | Now
-- | -- | --
1 | 2048 | 8186
32 | 2048 | 8000
128 | 2048 | 7424
1024 | 2048 | 2048

The PR makes the scheduled tokens much larger when request count is small by using an adaptive strategy, so one request won't be split to multiple forward calls.

### Internal KDA prefix checkpoints

The Mamba-style prefix cache split prefill at the last cacheable block boundary, sometimes adding a full-model pass for a short suffix. [PR #52789](https://github.com/vllm-project/vllm/pull/52789): checkpoint export inside one prefill pass, TTFT down 9%–25%. [PR #53614](https://github.com/vllm-project/vllm/pull/53614): partial prefix hits and speculative decoding, including EAGLE replay-boundary alignment.

For an 8K input:

Before

```text
Model forward (7,680 tokens)
└─ each KDA layer: FlashKDA(7,680) -> export checkpoint state

Model forward (320 tokens)
└─ each KDA layer: FlashKDA(320)
```

After

```text
One model forward (8,000 tokens)
└─ each KDA layer: FlashKDA recurrence
   ├─ process the first 7,680 tokens
   ├─ export checkpoint state
   └─ continue over the remaining 320 tokens
```

This PR helps us avoid a second full-model pass through attention, MoE, routing, and TP collectives.

### Zero-copy mixed KDA batches

Mixed speculative and non-speculative batches used six `index_select` and two `index_copy_` operations per layer. [PR #56159](https://github.com/vllm-project/vllm/pull/56159): contiguous zero-copy slices and direct output writes. Throughput up 5.2%–7.7% at concurrency 4 and 16; batch size 1 flat.

```text
# Example packed batch:
Token order: [N0, N1, S0, S1, S2]
              └─ N ─┘  └──── S ────┘

Packed QKV / gate / beta
          |
          +-- index_select(non-spec) × 3 --> non-spec KDA --+
          |                                                  |
          +-- index_select(spec)     × 3 --> spec KDA -------+
                                                             |
                              index_copy_(non-spec output) ---+
                              index_copy_(spec output) -------+
                                                             |
```

After

```text
Packed QKV / gate / beta: [N0, N1 | S0, S1, S2]
                                  |
                +-----------------+-----------------+
                |                                   |
       zero-copy slice [0:2]              zero-copy slice [2:5]
                |                                   |
        non-spec KDA                         spec KDA
     out=final_output[0:2]              out=final_output[2:5]
                |                                   |
                +-----------------+-----------------+
                                  |
                    Final output already assembled
```

Six `index_select` calls and two `index_copy_` calls removed per KDA layer.

### Deferred MXFP4 finalization

[PR #53152](https://github.com/vllm-project/vllm/pull/53152): MXFP4 top-k finalization inside the latent-tail kernel; one launch and one intermediate tensor write/read removed. End-to-end latency down roughly 5%. [PR #53327](https://github.com/vllm-project/vllm/pull/53327): initialization-order fix, enabling the deferred path before weight loading.

Before

```text
MXFP4 MoE kernel
    │
    ├─ GEMM2
    │
    └─ finalize kernel
         ├─ unpermute
         ├─ apply router weights
         ├─ top-k reduction
         └─ write [M, 3584] tensor
                    │
                    ▼
latent tail reads the tensor
    └─ AllReduce + RMSNorm + Up Projection + shared expert
```

After

```text
MXFP4 MoE kernel (do_finalize=False)
    │
    └─ return:
         ├─ GEMM2 output
         ├─ router weights
         └─ permutation map
                    │
latent tail consumes the deferred outputs
    ├─ unpermute + apply router weights + top-k reduction
    └─ AllReduce + RMSNorm + Up Projection + shared expert
```

This removes one kernel launch and avoids writing and rereading the finalized intermediate tensor.

## Acknowledgments

[Bolin Sun](https://github.com/BolinSNLHM), [Duncan Moss](https://github.com/djmmoss), [Harris Nover](https://github.com/hnover-nv), [Julian Huang](https://github.com/huangzhilin-hzl), [Ming](https://github.com/mingg26), [Nick Hill](https://github.com/njhill), [Rebecca Lee](https://github.com/rebklee), [Robert Shaw](https://github.com/robertgshaw2-redhat), [Summer Yang](https://github.com/GirasoleY), [Thien Tran](https://github.com/gau-nernst), [Tyler Michael Smith](https://github.com/tlrmchlsmth), and [Xiaolong Xu](https://github.com/BabyDrangoner) for their Kimi K3 PRs. Thanks also to the reviewers, CI maintainers, benchmark owners, and hardware teams.
