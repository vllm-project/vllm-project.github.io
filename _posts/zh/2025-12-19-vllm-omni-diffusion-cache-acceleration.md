---
layout: post
title: "vLLM-Omni 扩散缓存加速：TeaCache 与 Cache-DiT"
author: "vLLM-Omni Team"
image: /assets/figures/2025-12-19-vllm-omni-diffusion-cache-acceleration/qwen_bear_base.png
lang: zh
translation_key: vllm-omni-diffusion-cache-acceleration
tags:
  - multimodal
  - performance
  - ecosystem
---

# 让扩散推理更快

我们很高兴宣布 **vLLM-Omni** 的一次重要性能更新。

vLLM-Omni 现在支持多种缓存加速方法，可以在尽量不牺牲生成质量的前提下，加速扩散模型推理，例如 **Cache-DiT** 和 **TeaCache**。这些缓存方法会智能缓存中间计算结果，从而避免在扩散 timestep 之间重复做同样的工作。

通过这次更新，用户现在可以在图像生成任务中实现 **1.5 倍到 2 倍以上的加速**，而且配置改动非常小，画质损失也几乎可以忽略。

## 瓶颈：扩散中的冗余计算

扩散模型最出名的特点之一，就是算得非常重。生成一张图片通常需要几十个推理步骤，而相邻步骤处理的特征往往又非常相似。

vLLM-Omni 现在正是利用了这种时间冗余。通过缓存并复用中间计算结果，我们可以在后续步骤中跳过昂贵计算，而不需要重新训练模型。

## 两种强大的加速后端

vLLM-Omni 现在支持两种不同的缓存后端，可以分别满足不同的使用需求：

### 1. Cache-DiT：更强控制力与更高性能上限

[Cache-DiT](https://github.com/vipshop/cache-dit) 是一个更完整的库级加速方案，提供了一整套偏“重型”的优化手段来最大化效率：

* **DBCache（Dual Block Cache）**：根据残差差异智能缓存 Transformer block 的输出  
* **TaylorSeer**：利用基于泰勒展开的预测来推断特征，进一步减少计算量  
* **SCM（Step Computation Masking）**：通过自适应 masking，选择性跳过某些计算步骤

### 2. TeaCache：简单且自适应

TeaCache 已经原生集成在 vLLM-Omni 中，提供了一种基于 hook 的自适应缓存机制。它会监控输入之间的差异，并动态决定是否复用上一个 timestep 的 Transformer 计算结果。

## 性能基准

我们在 NVIDIA H200 GPU 上，使用 **Qwen-Image**（1024x1024 生成）对这些方法做了基准测试，结果如下：

| 模型 | 后端 | 配置 | 用时 | 加速比 |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen-Image** | 基线 | None | 20.0s | 1.0x |
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

### “Edit” 模型

在图像编辑任务上，Cache-DiT 的优势更加明显。对于 **Qwen-Image-Edit**，Cache-DiT 实现了 **2.38 倍加速**，生成时间从 51.5 秒降到仅 21.6 秒。

| 模型 | 后端 | 配置 | 用时 | 加速比 |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen-Image-Edit** | 基线 | None | 51.5s | 1.0x |
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

这些缓存优化技术在异构平台上也同样表现亮眼。例如，在昇腾 NPU 上，Qwen-Image-Edit 推理通过 Cache-DiT 从 142.38 秒降到 64.07 秒，获得了超过 **2.2 倍**加速。

## 支持的模型

| 模型 | TeaCache | Cache-DiT |
| :--- | :---: | :---: |
| **Qwen-Image** | ✅ | ✅ |
| **Z-Image** | ❌ | ✅ |
| **Qwen-Image-Edit** | ✅ | ✅ |

## 快速开始

在 vLLM-Omni 中启用缓存加速非常直接：初始化 `Omni` 类时指定 `cache_backend` 即可。

### 使用 TeaCache 加速

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2} 
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50)
```

### 使用 Cache-DiT 加速

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

## 了解更多

如果你想继续优化扩散推理流水线，可以参考以下详细文档：

* [Cache-DiT Acceleration Guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/cache_dit_acceleration/)
* [TeaCache Guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/teacache/)

除了缓存加速之外，我们还在持续推进并行化、kernel fusion 和量化方面的优化。后续还会有更多能力更新。

## 参考链接

* [Cache-DiT](https://github.com/vipshop/cache-dit)
* [TeaCache Guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/teacache/)
* [Cache-DiT Acceleration Guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/cache_dit_acceleration/)
* [英文原文：blog.vllm.ai/2025/12/19/vllm-omni-diffusion-cache-acceleration.html](https://blog.vllm.ai/2025/12/19/vllm-omni-diffusion-cache-acceleration.html)
