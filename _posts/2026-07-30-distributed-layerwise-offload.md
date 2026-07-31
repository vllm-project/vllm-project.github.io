---
layout: post
title: "Distributed Layerwise Offload: Running 185GB Models on 64GB NPU/GPU"
author: "Evan Chueng"
summary: "Sharding transformer weights across DP ranks with H2D + AllGather overlap, enabling large diffusion models (up to 185GB) to run on devices with limited HBM — with zero model code changes."
image: /assets/logos/vllm-logo-text-light.png
tags:
  - performance
  - diffusion
  - npu
---

## The Problem

Modern diffusion models far exceed the HBM capacity of a single NPU or GPU (typically 32–64 GB). Cosmos3-Super (64B / 124 GB) and larger variants (185 GB) cannot fit on a single device. Traditional solutions — Tensor Parallelism or HSDP — either require model-specific code changes or need the full model to fit across all devices' HBM combined.

**Distributed Layerwise Offload (DLO)** takes a different approach: each rank stores only `1/dp_size` of the model weights in host memory, and only 2 transformer blocks reside on each device at any time. Weights are streamed to the device via H2D copy overlapped with AllGather, keeping HBM usage constant regardless of model size.

## How It Works

### Weight Sharding + AllGather

Each transformer block's parameters are flattened by dtype, split into `dp_size` equal-sized shards, and stored in pinned CPU memory. At inference time:

1. **H2D copy**: Each rank copies its local shard to a pre-allocated GPU buffer (on a dedicated `copy_stream`)
2. **AllGather**: `all_gather_into_tensor` reconstructs the full weights (on a dedicated `comm_stream`)
3. **Re-point**: Parameters are zero-copy re-pointed to the GPU buffer via `set_tensor_storage`
4. **Compute**: The forward pass runs on the `compute_stream`, which waits for the AllGather event

Steps 1–2 for block N+1 are overlapped with step 4 for block N, achieving double-buffered prefetch.

### Double-Buffer Memory Bound

Exactly **2 transformer blocks** reside on each device at any time (current block + prefetched next block). This means HBM usage is independent of model size — only depends on the largest transformer block.

| Model | Block Size | Blocks | HBM/card (DP2) | Host RAM/rank |
|-------|:---------:|:------:|:--------------:|:------------:|
| Cosmos3-Nano (33 GB) | 368 MB | 72 | ~10 GB | ~38 GB |
| Cosmos3-Super (124 GB) | 930 MB | 128 | ~15 GB | ~157 GB |

### mmap Weight Loading

Instead of loading weights into RSS, we `mmap` safetensors files directly. The OS page cache is shared across all DP ranks, eliminating `O(dp_size × model_size)` RSS. Each rank reads only its shard from the mapped pages.

The mmap path is gated on `supports_mmap_loading()` — a shared function that checks whether the pipeline defines `_remap_ckpt_key`. This ensures the loader and the offload backend use the **same condition**:

```python
def supports_mmap_loading(pipeline: nn.Module) -> bool:
    return any(callable(getattr(type(m), "_remap_ckpt_key", None))
               for m in pipeline.modules())
```

### Slot Tracking: i%2 + Dynamic Correction

Each hook is initialized with `current_slot = i % 2`. Additionally, dynamic slot tracking via `_prefetched_slot` corrects for odd block counts, where the circular tail→head prefetch would otherwise collide with the head's read slot:

```python
if self._prev_hook is not None and self._prev_hook._prefetched_slot is not None:
    self.current_slot = self._prev_hook._prefetched_slot
```

Both mechanisms work together: `i%2` provides the initial value for the first forward pass; dynamic tracking corrects subsequent passes for any block count.

## DP Multi-Concurrency

With `--data-parallel-size N`, N concurrent requests can be processed in parallel — each rank picks `req[rank % len(reqs)]` and computes independently. AllGather only gathers weight shards, so ranks compute different requests simultaneously.

### Safety Validation

AllGather is a collective — every rank must participate at each step. To prevent deadlock:

1. **`num_inference_steps` validation**: All concurrent requests must have the same explicit step count (`None` is rejected because it may resolve differently per mode)

2. **`extra_args` whitelist**: The entire `extra_args` dict must be identical across all concurrent requests. Uses `json.dumps(ea, sort_keys=True)` to handle nested dicts/lists:

```python
extra_args_signatures: set = set()
for nr in new_reqs:
    ea = getattr(nr.req, "extra_args", None)
    if ea and isinstance(ea, dict):
        extra_args_signatures.add(json.dumps(ea, sort_keys=True))
    else:
        extra_args_signatures.add(None)
if len(extra_args_signatures) > 1:
    raise ValueError("DP multi-concurrency requires identical extra_args...")
```

### RPC Wave ID for Stale Message Prevention

Each RPC call gets a monotonically increasing `wave_id`. Workers echo it in all responses. The executor discards stale messages from failed previous waves:

```python
_MAX_STALE_DISCARDS = 16

def _validate_wave_id(self, response, expected_wave_id, deadline, method):
    discards = 0
    while discards < self._MAX_STALE_DISCARDS:
        resp_wave_id = response.get("wave_id") if isinstance(response, dict) else None
        if resp_wave_id is None or resp_wave_id == expected_wave_id:
            return response
        discards += 1
        response = self._dequeue_one_with_failure_polling(deadline, method)
    raise TimeoutError(...)
```

Additionally, all `num_responses` replies are drained before surfacing any error (`collected_errors`), preventing orphaned workers from hanging on pending AllGather operations.

## Fail-Closed Design

DLO is designed to fail closed rather than silently continue with incorrect weights:

| Scenario | Behavior |
|----------|----------|
| No safetensors found | `RuntimeError` at startup |
| Online quantization + AllGather | `ValueError` at startup (sharding breaks quantized layouts) |
| Mismatched `num_inference_steps` | `ValueError` before RPC |
| Mismatched `extra_args` | `ValueError` before RPC |
| TP > 1 | `ValueError` (DLO uses DP-based sharding) |
| HSDP + AllGather | `ValueError` (would double-shard) |

## Module Architecture

Shared utilities are extracted into focused modules:

```
offloader/
├── tensor_utils.py           # set_tensor_storage, make_offload_placeholder, is_dtensor
├── offload_plan.py           # OffloadPlan dataclass, get_offload_plan, supports_mmap_loading
├── block_discovery.py        # get_blocks_from_dit, get/set_blocks_attr_names
├── distributed_layerwise_backend.py  # Hook + Backend (1470 lines, down from 1609)
└── __init__.py               # Public API exports
```

**Zero model code changes:** `pipeline_cosmos3.py` and `transformer_cosmos3.py` are unchanged from `origin/main`. The offloader handles meta conversion, buffer save/restore, and `post_load_weights` generically.

## Performance

Tested on Ascend 910B3 (64GB HBM per card), DP2 + AllGather:

### Model Comparison

| Model | Size | Blocks | Block Size | HBM/card | Host RAM/rank |
|-------|:----:|:------:|:----------:|:--------:|:------------:|
| Cosmos3-Nano | 33 GB | 72 | 368 MB | ~10 GB | ~38 GB |
| Cosmos3-Super | 124 GB | 128 | 930 MB | ~15 GB | ~157 GB |
| Cosmos3-200B (synthetic) | 185 GB | 200 | 930 MB | ~15 GB | ~195 GB |

HBM usage is **~15 GB per card** for both 124GB and 185GB models — independent of model size. HSDP would require 3+ cards just to hold 185GB in HBM, leaving no room for activations. DLO uses only 2 cards.

### Cosmos3-Nano (33GB, 72 blocks)

| Scenario | Steps | Output Size | Latency | Tokens/Block |
|----------|:-----:|:-----------:|:-------:|:------------:|
| T2I 1024×1024 | 10 | 4.2 MB | 32s | ~1K |
| T2I 1024×1024 | 50 | 4.2 MB | 59s | ~1K |
| T2V 832×480, 29帧 (~1s) | 10 | 1.9 MB | 19s | ~2.7K |
| T2V 1280×720, 121帧 (~5s) | 10 | 17.3 MB | 159s | ~26K |

### Cosmos3-200B (185GB, 200 blocks)

| Scenario | Steps | Output Size | Latency |
|----------|:-----:|:-----------:|:-------:|
| T2I 1024×1024 | 1 | 4.2 MB | 23s |
| T2I 1024×1024 | 5 | 4.2 MB | 58s |
| T2V 832×480, 29帧 (~1s) | 5 | 1.9 MB | 61s |
| T2V 1280×720, 121帧 (~5s) | 5 | 16.9 MB | 394s |

The 200B model (185GB) runs on 2 × 64GB NPUs with DLO. **No other deployment method can achieve this** — the model is 2.9× larger than a single card's HBM, and even with 2 cards, HSDP would need 185GB / 128GB = 1.45× HBM utilization (no room for activations).

### Overlap Analysis (Cosmos3-Nano, measured)

| Metric | Value | Source |
|--------|-------|--------|
| Block size | 368 MB | DLO log |
| Shard size (DP2) | 184 MB | DLO log |
| H2D time | ~16ms | 184MB / 12GB/s |
| AllGather time | ~8ms | 368MB / 50GB/s (HCCS) |
| Prefetch total | ~24ms | H2D + AllGather |
| Per-block wall time | ~23ms | 82s DiT / (72×50) |
| Estimated compute | ~9ms | Wall - prefetch |
| MFU (T2I, 1 request) | ~40% | Compute/wall |

T2I is **prefetch-bound** (compute 9ms < prefetch 24ms). T2V 720p is **compute-bound** (compute ~120ms > prefetch 24ms, MFU ~85%).

## Getting Started

### Cosmos3-Nano with DP=2 + AllGather

```bash
vllm serve /path/to/Cosmos3-Nano \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --no-guardrails --init-timeout 3600 \
  --vae-use-tiling --vae-patch-parallel-size 1 \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 2
```

### Cosmos3-Super with DP=2 (for models > single-card HBM)

```bash
vllm serve /path/to/Cosmos3-Super \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --no-guardrails --init-timeout 3600 \
  --vae-use-tiling --vae-patch-parallel-size 1 \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 2
```

### Text-to-Image

```bash
curl -s -o output.png -X POST "http://localhost:8000/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/Cosmos3-Nano",
    "prompt": "A robot arm cleaning a plate, cinematic",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "guidance_scale": 6.0,
    "seed": 42
  }'
```

### Text-to-Video (720p, 5 seconds)

```bash
curl -s -o output.mp4 -X POST "http://localhost:8000/v1/videos/sync" \
  -H "Accept: video/mp4" \
  -F "model=/path/to/Cosmos3-Nano" \
  -F "prompt=A robot arm cleaning a plate, cinematic shot" \
  -F "negative_prompt=blurry" \
  -F "size=1280x720" \
  -F "num_frames=121" \
  -F "fps=24" \
  -F "num_inference_steps=10" \
  -F "guidance_scale=6.0" \
  -F "max_sequence_length=4096" \
  -F "flow_shift=10.0" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":false}' \
  -F "seed=17"
```

## Limitations

- `--dlo-use-allgather` (default) requires all concurrent requests to have the same `num_inference_steps` and identical `extra_args`
- Online quantization (FP8) is incompatible with AllGather mode — use `--dlo-no-use-allgather` instead
- Tensor Parallel is not supported (DLO uses DP-based sharding)
- HSDP + AllGather is rejected (would double-shard weights)

## Acknowledgments

Thanks to the review feedback from @gaohan123, @hsliuustc0106, and @yuanheng-zhao that significantly improved the safety and robustness of this feature.
