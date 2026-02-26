---
layout: post
title: "vLLM-Omni Diffusion Cache Acceleration"
author: "vLLM-Omni Team"
image: /assets/figures/2025-12-19-vllm-omni-diffusion-cache-acceleration/qwen_bear_base.png
---

# Turbocharge Your Diffusion Inference

We are thrilled to announce a major performance update for **vLLM-Omni**.

vLLM-Omni now supports various cache acceleration methods to speed up diffusion model inference with minimal quality degradation, e.g.,  **Cache-DiT** and **TeaCache**. These cache methods intelligently cache intermediate computations to avoid redundant work across diffusion timesteps.

With this update, users can now achieve **1.5x to over 2x speedups** in image generation tasks with minimal configuration and negligible quality loss.

## The Bottleneck: Redundancy in Diffusion

Diffusion models are notorious for their high computational costs. Generating a single image requires dozens of inference steps. However, adjacent steps often process very similar features.

vLLM-Omni now leverages this temporal redundancy. By intelligently caching and reusing intermediate computation results, we can skip expensive calculations in subsequent steps without retraining the model.

## Two Powerful Acceleration Backends

vLLM-Omni now supports two distinct caching backends to suit your specific needs:

### 1. Cache-DiT: Advanced Control & Maximum Performance
[Cache-DiT](https://github.com/vipshop/cache-dit) is a comprehensive library-based acceleration solution. It provides a suite of sophisticated techniques to maximize efficiency:

*   **DBCache (Dual Block Cache):** Intelligently caches Transformer block outputs based on residual differences.
*   **TaylorSeer:** Utilizes Taylor expansion-based forecasting to predict features, further reducing computational load.
*   **SCM (Step Computation Masking):** Applies adaptive masking to selectively skip computation steps.


### 2. TeaCache: Simple & Adaptive
TeaCache is implemented natively within vLLM-Omni, providing a hook-based, adaptive caching mechanism. It monitors the difference between inputs and dynamically decides when to reuse the transformer computations from the previous timestep.

## Performance Benchmarks

We benchmarked these methods on NVIDIA H200 GPUs using **Qwen-Image** (1024x1024 generation). The results are impressive:

| Model | Backend | Configuration | Time | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen-Image** | Baseline | None | 20.0s | 1.0x |
| **Qwen-Image** | **TeaCache** | `rel_l1_thresh=0.2` | 10.47s | **1.91x** ⚡ |
| **Qwen-Image** | **Cache-DiT** | DBCache + TaylorSeer | 10.8s | **1.85x** ⚡ |

<div style="display: flex; gap: 20px; justify-content: center; align-items: flex-start;">
  
  <div style="flex: 1; text-align: center;">
    <img src="/assets/figures/2025-12-19-vllm-omni-diffusion-cache-acceleration/cat.png" alt="No Cache" style="max-width: 100%; height: auto;">
    <p style="margin-top: 8px;">No Cache</p>
  </div>
  
  <div style="flex: 1; text-align: center;">
    <img src="/assets/figures/2025-12-19-vllm-omni-diffusion-cache-acceleration/cat_tea_cache.png" alt="TeaCache" style="max-width: 100%; height: auto;">
    <p style="margin-top: 8px;">TeaCache</p>
  </div>
  
  <div style="flex: 1; text-align: center;">
    <img src="/assets/figures/2025-12-19-vllm-omni-diffusion-cache-acceleration/cat_cache_dit.png" alt="Cache-DiT" style="max-width: 100%; height: auto;">
    <p style="margin-top: 8px;">Cache-DiT</p>
  </div>
</div>

### The "Edit" model
For image editing tasks, Cache-DiT shines even brighter. On **Qwen-Image-Edit**, Cache-DiT achieved a massive **2.38x speedup**, dropping generation time from 51.5s down to just 21.6s.

| Model | Backend | Configuration | Time | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen-Image-Edit** | Baseline | None | 51.5s | 1.0x |
| **Qwen-Image-Edit** | **TeaCache** | `rel_l1_thresh=0.2` | 35.0s | **1.47x** ⚡ |
| **Qwen-Image-Edit** | **Cache-DiT** | DBCache + TaylorSeer | 21.6s | **2.38x** ⚡ |

<div style="display: flex; gap: 20px; justify-content: center; align-items: flex-start;">
  
  <div style="flex: 1; text-align: center;">
    <img src="/assets/figures/2025-12-19-vllm-omni-diffusion-cache-acceleration/qwen_bear_base.png" alt="No Cache" style="max-width: 100%; height: auto;">
    <p style="margin-top: 8px;">No Cache</p>
  </div>
  
  <div style="flex: 1; text-align: center;">
    <img src="/assets/figures/2025-12-19-vllm-omni-diffusion-cache-acceleration/qwen_bear_tea_cache.png" alt="TeaCache" style="max-width: 100%; height: auto;">
    <p style="margin-top: 8px;">TeaCache</p>
  </div>
  
  <div style="flex: 1; text-align: center;">
    <img src="/assets/figures/2025-12-19-vllm-omni-diffusion-cache-acceleration/qwen_bear_cache_dit.png" alt="Cache-DiT" style="max-width: 100%; height: auto;">
    <p style="margin-top: 8px;">Cache-DiT</p>
  </div>
</div>

These caching optimization techniques show equally impressive results on heterogeneous platforms like Ascend NPU. For instance, Qwen-Image-Edit inference on Ascend NPU was accelerated using Cache-DiT from 142.38s down to 64.07s, achieving over a 2.2x speedup.

## Supported Models

| Model | TeaCache | Cache-DiT |
| :--- | :---: | :---: |
| **Qwen-Image** | ✅ | ✅ |
| **Z-Image** | ❌ | ✅ |
| **Qwen-Image-Edit** | ✅ | ✅ |

## Quick Start

Getting started with acceleration in vLLM-Omni is seamless. Simply define your `cache_backend` when initializing the `Omni` class.

### Accelerating with TeaCache
```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2} 
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50)
```

### Accelerating with Cache-DiT
```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 8,
        "enable_taylorseer": True, # Enable Taylor expansion forecasting
        "taylorseer_order": 1,
    }
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50)
```

## Learn More

Ready to speed up your diffusion pipelines? Check out our detailed documentation for advanced configurations:

*   [Cache-DiT Acceleration Guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/cache_dit_acceleration/)
*   [TeaCache Guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/teacache/)

Beyond caching, we are also actively developing optimizations in parallelization, kernel fusion, and quantization. Stay tuned for more powerful features!
