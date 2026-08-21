---
layout: post
title: "Distributed Layerwise Offload: Serving 200B+ DiT Models Efficiently in vLLM-Omni"
author: "vLLM-Omni Diffusion Team"
summary: "Sharding transformer weights across DP ranks with H2D + AllGather overlap, enabling large diffusion models (up to 185GB) to run on devices with limited HBM — with zero model code changes."
image: /assets/logos/vllm-logo-text-light.png
tags:
  - performance
  - distributed
  - vllm-omni
  - cosmos3
---

## TL;DR

vLLM-Omni's Distributed Layerwise Offload enables video generation models larger than single-device HBM (e.g., Cosmos3-Super 64B / 124 GB) to run across multiple NPUs or GPUs with minimal host memory overhead. The stack includes:

- **Meta-device initialization + mmap weight loading**: Weights are loaded as mmap views pointing to shared OS page cache, eliminating O(dp_size × model_size) RSS during model creation. Cold-start peak RSS drops by 73% (178 GB → 47 GB for Cosmos3-Nano DP4).
- **Weight sharding + AllGather**: Each rank stores only 1/dp_size of the model. Full layer weights are reconstructed at runtime via AllGather, overlapped with computation on dedicated streams.
- **Fixed double-buffer scheme**: Exactly 2 layers of weights reside on each device at any time, regardless of model size — 22% HBM growth from 17B to 64B model.
- **DP multi-concurrency**: Each DP rank processes a different request in parallel, achieving 3.3× throughput vs. single-request HSDP, with near-linear scaling.
- **Platform-agnostic**: Works on both NVIDIA GPU (CUDA/NCCL) and Ascend NPU (CANN/HCCL) via vLLM-Omni's platform abstraction layer.

Tested on Ascend 910B3 with Cosmos3-Nano (33 GB) and Cosmos3-Super (124 GB): all configurations produce correct video output, with cgroup-visible host memory scaling as O(model_size + dp_size × constant) instead of O(dp_size × model_size).

## Quickstart

```bash
# 4× NPU or GPU — Cosmos3-Nano with DP=4
vllm serve /path/to/Cosmos3-Nano --omni \
    --enable-distributed-layerwise-offload \
    --data-parallel-size 4

# 2× devices — Cosmos3-Super (124 GB) with DP=2
vllm serve /path/to/Cosmos3-Super --omni \
    --enable-distributed-layerwise-offload \
    --data-parallel-size 2

# Disable AllGather (each rank loads full weights, no sharding)
vllm serve /path/to/Cosmos3-Nano --omni \
    --enable-distributed-layerwise-offload \
    --data-parallel-size 4 \
    --dlo-no-use-allgather
```

The `--dlo-use-allgather` / `--dlo-no-use-allgather` flag controls whether weights are sharded (default: sharded). When disabled, each rank loads full weights independently — useful when AllGather synchronization overhead outweighs the memory savings.

## The Problem: Large Diffusion Models vs. HBM and Host Memory

Cosmos3-Super (64B parameters, 124 GB in BF16) cannot fit on a single 64 GB HBM device. The natural approach is to shard across multiple devices, but existing solutions each have limitations:

```
┌──────────────────────────────────────────────────┐
│          Cosmos3-Super (64B, 124 GB BF16)        │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │     Single Device HBM = 64 GB             │    │
│  │     Model = 124 GB  →  DOES NOT FIT ❌    │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  Existing approaches:                            │
│                                                  │
│  HSDP:         124/4 = 31 GB/card  →  56 GB     │
│                (weights + activations = OOM)     │
│                                                  │
│  Layerwise:    Full model per rank on CPU        │
│                4 × 124 GB = 496 GB host RAM ❌   │
│                                                  │
│  Our solution:  124/4 = 31 GB/rank (sharded)     │
│                124 GB total (shared page cache)  │
│                Only 2 layers on HBM at any time  │
└──────────────────────────────────────────────────┘
```

| Approach | Device HBM | Host Memory per Rank | Limitation |
|----------|:----------:|:--------------------:|------------|
| HSDP (FSDP2) | model / N | 0 | HBM fills up: 64B → 56 GB/card (8 GB headroom) |
| Layerwise offload | 2 layers only | full model | N × model_size host RAM (4 × 124 GB = 496 GB) |
| Tensor Parallel | model / N | 0 | Activation scaling helps, but communication overhead |
| Dist. Layerwise (ours) | 2 layers only | model / N | Requires AllGather synchronization |

For multi-device deployments, the host memory bottleneck is the killer: traditional layerwise offload stores a full copy of the model on each rank's host memory. With 4 devices, that's 4 × 124 GB = 496 GB — more than most servers have.

Worse, during model loading, each rank independently calls `param.data.copy_(loaded_weight)`, creating dp_size complete private copies in RSS. Peak RSS scales as O(dp_size × model_size), reaching 2 TB for a 200B model with dp_size=4.

## Solution Overview

Distributed Layerwise Offload addresses both the HBM and host memory bottlenecks through four techniques that stack together:

| Technique | Problem It Addresses | Primary Benefit |
|-----------|---------------------|-----------------|
| Meta device + mmap | O(dp_size × model) RSS during loading | -73% cold-start peak RSS |
| Weight sharding + AllGather | N × model_size host memory | 1× model_size total (shared page cache) |
| Double-buffer prefetch | All weights on device | Only 2 layers on HBM at any time |
| DP multi-concurrency | Serial request processing | 3.3× throughput via N parallel requests |

Each technique builds on the previous one. The walkthrough below takes each in turn — in the order we implemented it — and answers three questions: Why the problem exists, Why it works, and What you gain.

## 1. Meta Device + mmap Weight Loading

**Why.** The original loading path had each rank independently call `load_model(load_device="cpu")` before `offload_backend.enable()`. This caused `param.data.copy_(loaded_weight)` to create dp_size complete private copies of the model in RSS. For Cosmos3-Nano DP4, peak RSS was 178 GB — even though the model is only 33 GB.

**Why it works.** We create the transformer on `torch.device("meta")`, which allocates zero memory — only tensor shapes and dtypes are recorded. We then load weights as mmap views using `safe_open().get_tensor()`, which returns a view into the OS page cache rather than a copy.

```python
# pipeline_cosmos3.py — create transformer on meta device
with torch.device("meta"):
    self.transformer = transformer_cls(...)

# distributed_layerwise_backend.py — load as mmap views
tensor = safe_open(file_path, framework="pt", device="cpu").get_tensor(ckpt_key)
parent._parameters[name] = Parameter(tensor)  # points to page cache, 0 RSS
```

Since all ranks mmap the same safetensors files, the OS maintains a single copy of each file page in the page cache — shared across all processes. No rank creates a private copy.

For Hugging Face repo IDs (not local paths), we resolve the snapshot path first via `download_weights_from_hf()`, matching the pattern used by vLLM's existing DiffusersPipelineLoader.

**What you gain.** Cold-start peak RSS drops from 178 GB to 47 GB for Cosmos3-Nano DP4 — a 73% reduction. The page cache (1× model_size) is shared and read-only, and can be partially reclaimed by the OS under memory pressure.

```
  Without mmap (baseline)              With mmap (optimized)
  ─────────────────────────            ─────────────────────────

  Rank 0 ┌──────────────┐             Rank 0    ┌──────────┐
         │ Full Model    │                      │  (empty)  │
         │ 33 GB (copy)  │             Rank 1    └──────────┘
  Rank 1 ┌──────────────┐                       ┌──────────┐
         │ Full Model    │                      │  (empty)  │
         │ 33 GB (copy)  │             Rank 2    └──────────┘
  Rank 2 ┌──────────────┐                       ┌──────────┐
         │ Full Model    │                      │  (empty)  │
         │ 33 GB (copy)  │             Rank 3    └──────────┘
  Rank 3 ┌──────────────┐
         │ Full Model    │             Page Cache (shared)
         │ 33 GB (copy)  │             ┌──────────────────────┐
         └──────────────┘             │  33 GB (mmap view)    │
                                      │  All ranks point here │
  Page Cache (shared)                  │  0 private copies     │
  ┌──────────────┐                    └──────────────────────┘
  │ 33 GB        │
  └──────────────┘                    ┌────────────────────────┐
                                      │ cgroup peak: 47 GB     │
  Total RSS: 4 × 33 = 132 GB          │ (was 178 GB, -73%)     │
  + page cache: 33 GB                 └────────────────────────┘
  = 178 GB ❌
```

## 2. Weight Sharding with AllGather Reconstruction

**Why.** Even with mmap loading, the layerwise offload mechanism still copies the full model into each rank's pinned CPU memory for H2D transfers. With 4 devices, that's 4 × 33 GB = 132 GB of pinned memory — and it scales linearly with device count.

**Why it works.** Instead of storing the full model, each rank stores only 1/dp_size of the weights. At runtime, the full layer weights are reconstructed via `all_gather_into_tensor` on a dedicated communication stream.

```python
# _shard_and_pin: each rank stores only its 1/dp_size shard
shard_size = (total_numel + dp_size - 1) // dp_size  # ceil division
shard = torch.zeros(shard_size, dtype=dtype, device="cpu")
# Copy only the portion within [rank * shard_size, (rank+1) * shard_size)
shard[dst_slice].copy_(mmap_view.flatten()[src_slice])
shard = shard.pin_memory()  # DMA buffer for fast H2D
```

The sharding uses ceil division with zero-padding, so all shards are equal-sized — a requirement for `all_gather_into_tensor`. After sharding, the original mmap views are replaced with zero-element placeholders, releasing the page cache references.

**What you gain.** Total pinned memory drops from dp_size × model_size to model_size (sum across all ranks). For Cosmos3-Super DP4: 4 × 124 GB → 124 GB total, 31 GB per rank.

```
  Without sharding (layerwise)         With sharding (dist_offload)
  ──────────────────────────────       ──────────────────────────────

  Rank 0: [A|B|C|D]  ← full model     Rank 0: [A]  ← 1/4 shard
  Rank 1: [A|B|C|D]  ← full model     Rank 1: [B]  ← 1/4 shard
  Rank 2: [A|B|C|D]  ← full model     Rank 2: [C]  ← 1/4 shard
  Rank 3: [A|B|C|D]  ← full model     Rank 3: [D]  ← 1/4 shard


  CPU total: 4 × 124 = 496 GB ❌      CPU total: 124 GB ✓

          AllGather reconstruction:
          ┌─────────────────────────┐
          │  Rank 0 sends [A]       │
          │  Rank 1 sends [B]       │──→ All ranks receive [A|B|C|D]
          │  Rank 2 sends [C]       │    (full layer weights)
          │  Rank 3 sends [D]       │
          └─────────────────────────┘
```

## 3. Double-Buffered Prefetch with H2D + AllGather Overlap

**Why.** Sharding solved the memory problem, but each layer still needs its full weights on-device during computation. If we load all layers at once, HBM fills up — the original problem returns. Synchronous loading (H2D → wait → AllGather → wait → compute) also wastes time: the GPU sits idle during data movement.

**Why it works.** We maintain exactly two device buffers (slots), each sized to the largest block in the model. While the compute stream executes layer N (using slot 0), background streams prepare layer N+1 into slot 1:

![DLO Double-Buffer Prefetch Pipeline](/assets/figures/2026-07-30-distributed-layerwise-offload/dlo_pipeline.gif)

*Animation: Three-stream timeline showing Compute (blue), H2D (orange), and AllGather (green) overlapped via double-buffered slots. Red dashed arrows indicate event synchronization — compute waits for AllGather to complete before switching slots.*

The two-stage preparation runs on separate streams:

1. **H2D** (`copy_stream`): load 1/dp_size shard from pinned CPU to device
2. **AllGather** (`comm_stream`): gather shards from all ranks into the full-weight buffer

Both streams are overlapped with the compute stream via event-based synchronization. After AllGather completes, parameters are re-pointed to slices of the output buffer using cached metadata.

The buffers are shared across all blocks — allocated once to the max block size, reused for every layer. This ensures HBM usage is bounded by 2 × max_block_size, independent of the total number of layers.

On Ascend NPU, `pin_memory()` allocates DMA-capable memory via `/dev/davinci_manager` (the NPU device driver). This memory resides in CPU kernel space and is not tracked by cgroup — a key finding that explains why cgroup peak is much lower than expected.

**What you gain.** HBM holds only 2 layers of weights (~2 GB for Nano, ~3 GB for Super), regardless of model size. From Nano (33 GB) to Super (124 GB), idle HBM grows only 22% (11.5 → 14.6 GB) — the model is 3.8× larger but HBM barely changes.

```
  HBM Usage: Nano vs Super (dist_offload+SP, 720p 10s)
  ──────────────────────────────────────────────────────

  Nano (33 GB)     Super (124 GB)
  ┌────────┐       ┌────────┐
  │        │       │        │  ← idle HBM: 14.6 GB
  │  23.1  │       │  28.1  │  ← peak HBM: 28.1 GB
  │  GB    │       │  GB    │     (only +22% vs Nano)
  │        │       │        │
  │        │       │        │     Model is 3.8× larger
  │        │       │        │     but HBM barely changes
  └────────┘       └────────┘
   64 GB card       64 GB card
   41 GB free       36 GB free   ← still plenty of headroom

  Compare: HSDP+SP (weights in HBM)
  Nano: 29.0 GB     Super: 56.3 GB  ← 8 GB headroom only, near OOM
```

## 4. DP Multi-Concurrency: N Requests in Parallel

**Why.** AllGather only gathers weight shards — it is completely request-independent. This means all DP ranks are synchronized at each AllGather call, but they can compute different activations (different requests) in parallel. Without exploiting this, DP ranks sit idle between AllGather calls, and throughput is limited to 1 request at a time.

**Why it works.** When `dp_concurrent` is enabled, the scheduler batches up to dp_size requests together. The executor sends all requests in a single broadcast RPC:

```
  Without DP multi-concurrency:       With DP multi-concurrency:
  ─────────────────────────────       ─────────────────────────────

  Request 1 ──→ [Worker 0]            Request 1 ─┐
  (wait...)                            Request 2 ─┤──→ [reqs_list] ──→ Broadcast RPC
  Request 2 ──→ [Worker 0]            Request 3 ─┤
  (wait...)                            Request 4 ─┘
  Request 3 ──→ [Worker 0]
  (wait...)                             ┌───────────────────────────────┐
  Request 4 ──→ [Worker 0]             │  Worker 0 (dp_rank=0): req[0] │
                                       │  Worker 1 (dp_rank=1): req[1] │
  Throughput: 1 req at a time         │  Worker 2 (dp_rank=2): req[2] │
  All ranks run same request          │  Worker 3 (dp_rank=3): req[3] │
                                       │                               │
                                       │  AllGather: weights only     │
                                       │  (request-independent)       │
                                       │  Compute: different reqs     │
                                       └───────────────────────────────┘

                                       Throughput: 4 reqs in parallel
                                       3.22 fps (3.3× vs HSDP)
```

```python
# Executor: send all requests at once
reqs_list = [nr.req for nr in new_reqs]
results = collective_rpc("execute_model", args=(reqs_list, ...),
                         unique_reply_rank=None, exec_all_ranks=True)
```

Each worker picks one request based on its DP rank (not global rank, to handle SP/TP correctly):

```python
dp_rank = get_data_parallel_rank()
req = reqs_list[dp_rank % len(reqs_list)]
```

Only the primary rank within each DP replica (SP=0, TP=0, CFG=0, PP=0) replies, tagged with `dp_rank` for result matching. The executor collects responses via round-robin polling and sorts by `dp_rank` to match results to requests.

A validation step rejects concurrent requests with different `num_inference_steps` — since AllGather is a collective, mismatched step counts would cause one rank to exit early while others hang.

**What you gain.** 4 concurrent requests achieve 3.22 fps throughput — 3.3× the HSDP single-request baseline. Scaling is near-linear (4.0×) because AllGather overhead is fixed (~150 ms/step) and amortized across 4 concurrent computations.

## Memory Model: Why It Is Not 2× Model Size

A naive analysis would expect 2× model_size in host memory: page cache (1× model) + shard buffers (1× model total). But on Ascend NPU, `pin_memory()` allocates via `/dev/davinci_manager`, placing the shard in CPU kernel DMA memory that is invisible to the cgroup memory controller.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    Physical System RAM (2 TB)                    │
  │                                                                 │
  │  ┌─────────────────────────────────┐  cgroup ✓ (cache)         │
  │  │  Page Cache (safetensors)        │  1× model_size = 31 GB    │
  │  │  Shared across all ranks         │  (mmap views, read-only)  │
  │  └─────────────────────────────────┘                           │
  │                                                                 │
  │  ┌──────────┐  ┌──────────┐  cgroup ✓ (rss)                   │
  │  │ Rank 0   │  │ Rank 1   │  ~3.5 GB × dp_size = 7 GB         │
  │  │ Framework│  │ Framework│  (Python, torch, HCCL, VAE)        │
  │  └──────────┘  └──────────┘                                    │
  │                                                                 │
  │  ════════════════════════════════════════  cgroup boundary     │
  │                                                                 │
  │  ┌─────────────────────────────────┐  cgroup ✗ (kernel DMA)    │
  │  │  /dev/davinci_manager            │  model_size = 29 GB       │
  │  │  Rank 0 shard  │  Rank 1 shard   │  (pin_memory, invisible   │
  │  │  14.5 GB       │  14.5 GB        │   to cgroup, RSS=0)       │
  │  └─────────────────────────────────┘                           │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │                    NPU HBM (64 GB/card)                         │
  │                                                                 │
  │  Framework + HCCL buffers    ~2 GB                             │
  │  Double-buffer (2 slots)    ~0.8 GB  ← max_block_size × 2     │
  │  Activations (during infer) ~7 GB                              │
  │  ─────────────────────────────────                                │
  │  Total                      ~10 GB  (55 GB headroom)           │
  │                                                                 │
  │  NOTE: Shard is NOT in HBM (it's in CPU kernel DMA above)      │
  └─────────────────────────────────────────────────────────────────┘
```

Verified with clean measurements (Cosmos3-Nano DP2, fresh cgroup):

```
cgroup usage_in_bytes = 49 GB = cache(31) + rss(18)  ← exact match, no extra
cgroup kmem           = 0 GB
davinci_manager RSS   = 0 kB  (in /proc/<pid>/smaps)
NPU HBM per card      = 10 GB  (< 14.5 GB shard → shard NOT in HBM)
Slab                  = 3.3 GB  (too small for 29 GB shard)
```

| Component | Location | Size | Tracked by cgroup? |
|-----------|----------|------|:------------------:|
| Safetensors page cache | System RAM (user space, shared) | 1× model_size | ✓ (cache) |
| Framework (Python/torch/HCCL) | System RAM (user space, per-rank) | ~3.5 GB × dp_size | ✓ (rss) |
| Shard (pinned) | CPU kernel DMA (/dev/davinci_manager) | model_size / dp_size per rank | ✗ |
| Prefetch buffers | NPU HBM | 2 × block_size per rank | ✗ |

This means cgroup-visible memory scales as O(model_size + dp_size × constant), not O(dp_size × model_size). For a 200B model with dp_size=4: ~423 GB cgroup + ~400 GB kernel DMA = ~823 GB total physical RAM (fits in 2 TB), vs. 2000 GB without mmap.

## Validation Results

All tests on Ascend 910B3 (64 GB HBM/card, 2 TB system RAM), Cosmos3-Nano (33 GB) and Cosmos3-Super (124 GB).

### Correctness

| Model | Config | Requests | HTTP | Frames | Video |
|-------|--------|:--------:|:----:|:------:|:-----:|
| Nano (33 GB) | DP2 | 2 concurrent, 35 steps | 2/2 × 200 | 29/29 | OK |
| Nano (33 GB) | DP4 | 4 concurrent, 35 steps | 4/4 × 200 | 29/29 | OK |
| Super (124 GB) | DP2 | 1 request, 5 steps | 200 | 29 | OK |
| Super (124 GB) | DP4 | 1 request, 5 steps | 200 | 29 | OK |

### Host Memory (cgroup peak)

| Model | Config | cgroup Peak | Page Cache | RSS | Per-worker HWM | vs. Baseline |
|-------|--------|:-----------:|:---------:|:---:|:--------------:|:------------:|
| Nano (33 GB) | DP4 (mmap) | 47 GB | 31 GB | 14 GB | 12.1 GB | — |
| Nano (33 GB) | DP4 (no mmap) | 178 GB | — | — | 36 GB | -73% |
| Super (124 GB) | DP2 | 157 GB | 149 GB | 7 GB | 65.2 GB | — |
| Super (124 GB) | DP4 | 172 GB | 149 GB | 14 GB | 35.5 GB | — |

### NPU HBM

| Model | Config | HBM/card (idle) | HBM/card (inference) | 64 GB Headroom |
|-------|--------|:---------------:|:--------------------:|:--------------:|
| Nano (33 GB) | DP2 | 9.9 GB | 10.4 GB | 55 GB |
| Nano (33 GB) | DP4 | 9.4 GB | 10.2 GB | 55 GB |
| Super (124 GB) | DP2 | ~15 GB | — | ~49 GB |
| Super (124 GB) | DP4 | ~10 GB | — | ~54 GB |

HBM grows only ~22% from Nano to Super, because only 2 layers of weights reside on device — the model is 3.8× larger but HBM barely changes.

### Performance

| Strategy | Per-step (ms) | Throughput (fps) | CPU/rank | HBM/card | vs. HSDP |
|----------|:------------:|:----------------:|:--------:|:--------:|:--------:|
| HSDP+SP (baseline) | 870 | 0.967 | 0 GB | 20.3 GB | — |
| dist_offload+AG (DP4, 1 req) | 1,020 | 0.806 | 3.5 GB | 12.4 GB | -17% |
| dist_offload+AG (DP4, 4 req) | 1,020 | 3.22 | 3.5 GB | 12.4 GB | 3.3× |
| dist_offload no-AG | 1,877 | 0.439 | 28.3 GB | 14.1 GB | -55% |

AllGather overhead = 150 ms/step (72 ms stream switch + 10 ms HCCL + 68 ms Python dispatch), model-size independent. With 4 concurrent requests, this fixed cost is amortized 4×.

### NVIDIA B300 GPU Results

To validate platform-agnosticism, we ran the same DLO stack on NVIDIA B300 SXM6 GPUs. All tests use Cosmos3-Super BF16 (124 GB), 4× NVIDIA B300 (physical GPUs 1,5,6,7), Python 3.12.3, PyTorch 2.11.0+cu130, CUDA 13.0, vLLM 0.25.0.

Correctness was verified via byte-identical output hashes across all strategies. For example, T2I seed 42 produced identical SHA256 `6e7d2a8c63b88391...` across DLO+AG, no-AG, DLO+USP4, legacy layerwise+USP4, and HSDP+USP4. T2V 832×480×29f seed 17 produced identical 666,029-byte output (SHA256 `c5d38f5d21ca619e...`) across all strategies.

#### 1024×1024 T2I, 50 steps

| Strategy | Concurrency | Wave latency | Throughput | Process-tree PSS | Peak HBM/card |
|----------|:-----------:|:------------:|:----------:|:----------------:|:-------------:|
| DLO+AG DP4 | 4 | 43.69s (median) | 0.0915 outputs/s | 198–202 GiB | 12.62 GiB |
| DLO no-AG DP4 | 4 | 112.96s | 0.0354 outputs/s | 532 GiB | 11.43 GiB |
| HSDP+USP4 | 1 | 15.19s | 0.0658 outputs/s | 483 GiB | 42.00 GiB |
| legacy layerwise+USP4 | 1 | 105.22s | 0.0095 outputs/s | 533 GiB | 13.99 GiB |

DLO+AG DP4 with 4 concurrent requests achieves **1.39×** the throughput of HSDP+USP4, while using only **30%** of the HBM (12.6 GiB vs 42.0 GiB).

#### 832×480 T2V, 29 frames, 35 steps

| Strategy | Outputs/wave | Wave latency | Throughput | Output SHA |
|----------|:------------:|:------------:|:----------:|:----------:|
| DLO+AG DP4 | 4 | 38.79s | 0.1033 outputs/s | c5d38f5d... |
| HSDP+USP4 | 1 | 15.38s | 0.0653 outputs/s | c5d38f5d... |
| DLO+AG+USP4 | 1 | 30.79s | 0.0326 outputs/s | c5d38f5d... |
| legacy layerwise+USP4 | 1 | 81.46s | 0.0123 outputs/s | c5d38f5d... |

#### Workload Latency and HBM (35 steps, DLO+AG DP4 vs HSDP+USP4)

| Workload | DLO+AG (4 outputs) | DLO peak HBM/card | HSDP (1 output) | HSDP peak HBM/card |
|----------|:------------------:|:-----------------:|:---------------:|:------------------:|
| 480p, 29f | 38.79s | 14.55 GiB | 15.38s | 43.77 GiB |
| 480p, ~5s (121f) | 102.58s | 15.88 GiB | 41.36s (125f) | 53.73–62.65 GiB |
| 480p, ~10s (241f) | 226.70s | 17.33 GiB | 82.47s (245f) | 53.74 GiB |
| 720p, 5s (121f) | 288.29s | 24.95 GiB | 87.47s | 52.19 GiB |
| 720p, 10s (241f) | 214.53s (DLO+AG+USP4) | 24.99 GiB | 210.05s | 53.73 GiB |

On 720p 10s (241f), DLO+AG+USP4 completed in 214.53s — within **2.13%** of HSDP's 210.05s — with byte-identical output (SHA256 `08cb679322996ea6...`), while using only **47%** of HSDP's HBM (24.99 GiB vs 53.73 GiB).

### Extrapolation to 400 GB

| Model | dp_size | cgroup Peak (est.) | Total RAM (est.) | Fits 2 TB? |
|-------|:-------:|:------------------:|:----------------:|:----------:|
| 33 GB | 4 | 47 GB | ~80 GB | ✓ |
| 124 GB | 4 | 172 GB | ~280 GB | ✓ |
| 185 GB | 4 | ~220 GB | ~420 GB | ✓ |
| 400 GB | 4 | ~423 GB | ~823 GB | ✓ |
| 400 GB | 8 | ~443 GB | ~843 GB | ✓ |

## Acknowledgements

We thank the vLLM-Omni contributors, including @hsliuustc0106 and @yuanheng-zhao for thorough code review feedback, and the Ascend NPU team for hardware support.

## References

**Source code:**

- Distributed layerwise offload backend: `distributed_layerwise_backend.py`
- OffloadConfig and strategy selection: `base.py`
- Multi-queue executor: `multiproc_executor.py`
- DP multi-concurrency worker: `diffusion_worker.py`
- Cosmos3 meta device pipeline: `pipeline_cosmos3.py`
- Unit tests: `test_distributed_layerwise_backend.py`

**RFC and PR:**

- RFC: GitHub Issue #5396
- Implementation PR: vllm-omni#5397

**Model weights:**

- Cosmos3-Nano: 33 GB safetensors (17B params, 72 blocks)
- Cosmos3-Super: 124 GB safetensors (64B params, 128 blocks)
