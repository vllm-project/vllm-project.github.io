---
layout: post
title: "vLLM Triton Attention 后端深度解析"
author: "vLLM Team at IBM Research"
image: /assets/figures/2026-03-04-vllm-triton-backend/image1.png
math: true
lang: zh
translation_key: vllm-triton-backend-deep-dive
tags:
  - performance
  - triton
  - attention
---

本文改编自 Red Hat 主办的 [vLLM Office Hours](https://www.youtube.com/watch?v=8QiM-i9ifFo&list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3&index=1) 中，IBM Research 的 Burkhard Ringlein 所做的一场深度技术分享，主题是 vLLM Triton attention 后端的技术解析。欢迎[浏览往期主题](https://www.youtube.com/playlist?list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3)，并通过[这里](https://red.ht/office-hours)加入后续的 Office Hours。

过去一年里，IBM Research、Red Hat 和 AMD 的团队共同开发并上游合并了一个基于 Triton 的 attention 后端，目标是在跨 GPU 厂商保持强可移植性的同时，达到业界领先的推理性能。这项工作的直接动因，是加速器硬件日益多样化，以及维护大量高度专用 kernel 的成本不断攀升。

本文将深入解析这项工作：为什么 [Triton](https://github.com/triton-lang/triton) 非常适合 [vLLM](https://github.com/vllm-project/vllm)，Triton attention 后端是什么、它会在什么情况下被使用，以及如何实现一个高性能的 paged attention kernel。过程中我们会涉及 kernel 级优化、并行化策略、CUDA graph 交互以及基准测试结果，最后简要展望 Helion。

## **为什么 Triton 适合 vLLM**

vLLM 的目标是在不同平台、模型和执行策略下提供尽可能好的推理性能。现实里，这意味着它必须支持多种加速器与硬件代际、丰富的模型架构，以及形态各异的工作负载，例如不同 batch size、序列长度和 attention 模式。

一种直接思路，是为每种模型和 GPU 架构都编写高度专用的 kernel。这样做确实有效，但不可持续。随着平台扩展到 NVIDIA Hopper、Blackwell、AMD MI300、Intel 乃至未来的新平台，维护成百上千个 kernel 很快就会变得不切实际。

因此，我们更倾向于编写能够自动适配底层硬件的性能可移植 kernel。Triton 后端正体现了这一思路。

Triton 是一种领域特定语言，允许开发者用 Python 编写 GPU kernel，例如矩阵乘法或 attention。这些 kernel 会被编译为适配多平台的高效 GPU 代码。Triton 的 tiled 编程模型在抽象层次上取得了平衡：它足够底层，可以表达与硬件密切相关的优化；同时又足够高层，能在很大程度上保持硬件无关性。

如图 1 所示，开发者以逻辑 tile 为单位描述计算。Triton 编译器和 autotuner 会决定这些 tile 如何映射到具体硬件。不同 GPU 上的 tile 形状和执行布局可能差异很大，但这些选择都由编译器自动完成，并常由 autotuning 引导。更多细节可参考我们的论文 [GPU Performance Portability needs Autotuning](https://arxiv.org/abs/2505.03780)。

<figure style="text-align: center;">
  <img src="/assets/figures/2026-03-04-vllm-triton-backend/image1.png" alt="Figure 1" />
  <figcaption>图 1：Triton 的 tiled 编程模型，逻辑 tile 由编译器和 autotuner 自动映射到硬件特定的执行布局。</figcaption>
</figure>

## **vLLM 中的 Triton Attention 后端**

Attention 通常是大语言模型中最关键、也最敏感的性能热点。为了控制复杂度，vLLM 引入了 attention backends 这一抽象层，将 attention 的具体实现隐藏在统一 API 后面，并与线性层、layer normalization 等更简单的组件解耦。

在这层抽象下，vLLM 支持多种 attention 后端，包括 CUDA 平台上的 FlashAttention 和 FlashInfer、基于 ROCm 的 attention 后端，以及面向 MLA 风格 attention 的专用后端（完整列表见[这里](https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends)）。[Triton attention backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py) 则完全用 Triton 实现，并作为 vLLM 的原生组件存在。

开发这个后端的原因，主要是性能可移植性和依赖管理。它在 NVIDIA、AMD 和 Intel GPU 上运行同一份源代码，只依赖 PyTorch 和 Triton，并始终作为 vLLM 的一部分可用。虽然最初由 IBM Research 和 Red Hat AI 开发，但现在已经由更广泛的社区共同维护和扩展。

## **Triton Attention 后端何时被使用**

Triton attention 后端是 ROCm 上 AMD GPU 的默认选项，也被用于 Intel XPU。当使用 float32 时，由于 FlashAttention 在这些场景下不支持 fp32，vLLM 会回退到 Triton Attention。它还支持一些需要特定特性的模型，例如 StepFun 音频模型使用的 ALiBi sqrt，以及 sink tokens 和 GPT-OSS 行为，特别是在 A100 这类 pre-Hopper 的 NVIDIA GPU 上。

此外，它还支持小 head size 模型、encoder 和 decoder attention，以及多模态 prefix attention。由于 Triton attention 后端始终存在，所以当 FlashAttention、FlashInfer 或其他依赖不可用或导入失败时，它也会充当一个可靠的 fallback backend。诸如 batch invariance 之类的特性同样可以被支持。

## **用 Triton 编写高性能、可移植的 Paged Attention Kernel**

当 Triton attention 后端的开发开始时，kernel 最初是在 vLLM 外部独立实现的，并通过大量 microbenchmark 做性能评估。Kernel API 按照 vLLM 的需求设计，但性能调优则是在端到端集成之前独立完成的。

[Microbenchmarks](https://github.com/foundation-model-stack/vllm-triton-backend) 对理解不同工作负载下的性能行为至关重要，包括 prefill 密集、decode 密集和混合型工作负载，以及不同 batch size 与上下文长度的组合。

图 2 展示了有代表性的微基准测试结果。x 轴表示 token 总数，y 轴表示延迟。不同子图区分了 prefill-only、mixed 和 decode-only 工作负载。结果表明，不同 kernel 变体在不同工作区间各有优势，没有单一配置能在所有场景中占优。

<figure style="text-align: center;">
  <img src="/assets/figures/2026-03-04-vllm-triton-backend/image2.png" alt="Figure 2" />
  <figcaption>图 2：多种 Triton paged attention kernel 变体在 prefill、decode 和混合工作负载中的微基准测试对比。</figcaption>
</figure>

Microbenchmark 与端到端 benchmark 互为补充，因为它能暴露系统级效应可能掩盖掉的 kernel 级行为。

## **回顾：Paged Attention Kernel 在做什么**

Paged attention 通过对 KV cache 进行分页，实现了一种更节省内存的 attention 方式。对于 batch 中的每个 query，kernel 会处理每个 query token。对每个 token，它会遍历 query heads 及对应的 KV heads，再遍历分页后的 KV cache，以计算 attention score 并应用 value vectors。

图 3 展示了这一结构。Query tokens 沿 x 轴排列，query heads 沿 y 轴排列，而 paged KV cache 的遍历构成最内层循环。为了简洁起见，这里省略了 causal masking 和 sliding window 等细节。

<figure style="text-align: center;">
  <img src="/assets/figures/2026-03-04-vllm-triton-backend/image3.png" alt="Figure 3" />
  <figcaption>图 3：Paged attention 的概念视图，展示 query tokens、query heads 与分页 KV cache 的遍历方式。</figcaption>
</figure>

如果想深入了解这个 kernel 的底层优化细节，推荐阅读 kernel 作者撰写的 PyTorch 博客：[https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/](https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/)

代码在这里：<https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py>

## **使用 Q Blocks 优化 tl.dot 的 Tile Size**

Attention 中的核心计算是矩阵乘法，在 Triton 中通过 `tl.dot` 实现。但想获得高性能，就需要足够大的 tile 来充分利用底层硬件，而仅仅加载分页 KV cache 并不能自动带来理想表现。

KV 侧的 tile size 会受到 KV cache 页大小的限制，因此优化重点转向 query 侧。对于 group query attention，可以通过把与单个 KV head 对应的所有 query heads 一起处理，来提高 cache reuse。为了进一步增加并行度，还可以把多个 query token 组合成一个工作项，这个工作项被称为一个 Q block。

图 4 展示了这一方法。Launch grid 覆盖 batch size 和 KV heads，而 Q blocks 决定每个 kernel instance 处理多少 query tokens 和 query heads。Autotuning 会为不同平台自动选择合适的 block size。

<figure style="text-align: center;">
  <img src="/assets/figures/2026-03-04-vllm-triton-backend/image4.png" alt="Figure 4" />
  <figcaption>图 4：Q blocks 将多个 query heads 和 query tokens 组合到一个工作项中，以提高 tl.dot 的利用率和 cache 复用。</figcaption>
</figure>

## **通过 Parallel Tiled Softmax 增加并行度**

一次处理多个 query token 对 prefill 工作负载效果很好，但对 decode 工作负载没有帮助，因为 decode 阶段通常只处理单个 query token。为了解决这个问题，可以通过 parallel tiled softmax 引入额外并行化，也就是所谓的 “3D kernel”。

这种方法会把遍历 KV cache 的工作拆分到多个 kernel instance 中。每个 instance 先计算一部分结果，随后通过归约得到最终输出。由于 Triton 不提供全局 barrier，这种归约必须通过启动第二个 kernel 来完成，因此需要在额外并行度和额外启动开销之间做权衡。系统会通过启发式规则判断这种做法何时值得启用。

## **CUDA Graphs、Launch Grids 与 GPU 执行 Waves**

CUDA graphs 可以通过记录并重放固定的执行图来减少 kernel 启动开销。但 attention kernel 带来一个挑战：它们的 launch grid 往往依赖于 batch size 和序列长度。

GPU 通过固定数量的 streaming multiprocessors（SMs）执行 kernel。当启动的线程数超过可用 SM 数量时，执行就会分 wave 进行。图 5 展示了这一行为，其中第二波执行会造成资源利用不足。

<figure style="text-align: center;">
  <img src="/assets/figures/2026-03-04-vllm-triton-backend/image5.png" alt="Figure 5" />
  <figcaption>图 5：当启动线程数超过可用 streaming multiprocessors 时的 GPU 执行 waves。在这个例子中，GPU 有 8 个 SM，而我们需要执行 12 个线程。</figcaption>
</figure>

当这些执行被捕获进 CUDA graph 后，即便后续有效工作负载规模变小，这种低效也会被原样重放。图 6 展示了固定 launch grid 如何导致额外的无效工作和更高的延迟。

<figure style="text-align: center;">
  <img src="/assets/figures/2026-03-04-vllm-triton-backend/image6.png" alt="Figure 6" />
  <figcaption>图 6：通过 CUDA graphs 重放固定 launch grids 时产生的额外浪费。</figcaption>
</figure>

## **从可变 Launch Grids 到 Persistent Kernels**

早期的 paged attention kernel 使用会随工作负载大小变化的 launch grids，如图 7 所示。它虽然灵活，但与 CUDA graphs 的配合并不理想。

<figure style="text-align: center;">
  <img src="/assets/figures/2026-03-04-vllm-triton-backend/image7.png" alt="Figure 7" />
  <figcaption>图 7：早期 paged attention kernel 使用的可变 launch grids。</figcaption>
</figure>

为了解决这个问题，我们设计了 persistent kernels（对应 PR 仍在提交中）。做法是启动一个固定数量的 kernel instance，其数量等于可用计算资源的数量。每个 instance 再通过从 GPU 显存读取 metadata，动态决定自己需要处理多少工作。这样一来，launch grid 保持恒定，CUDA graphs 也就能被高效复用。

<figure style="text-align: center;">
  <img src="/assets/figures/2026-03-04-vllm-triton-backend/image8.png" alt="Figure 8" />
  <figcaption>图 8：Persistent kernel 方案，特点是固定 launch grids 和动态工作分配。</figcaption>
</figure>

## **基准测试结果**

2025 年底的 benchmark 结果证明了这套方法的有效性。图 9 展示了在 NVIDIA H100 和 AMD MI300 上，Llama 3.1 8B、batch size 为 1、输入长度为 500 tokens 条件下的端到端延迟结果，x 轴表示输出长度。

在 H100 上，Triton attention backend 在长 decode 请求中达到了 FlashAttention 3 性能的 100.7%。在 MI300 上，相比更早的实现则获得了约 5.8× 的加速。更重要的是，这两个平台使用的是完全相同的 Triton kernel 源码。需要注意的是，Triton 中的 paged attention 实现大约只有 800 行代码，而 FlashAttention 3 大约有 70,000 行代码。

<figure style="text-align: center;">
  <img src="/assets/figures/2026-03-04-vllm-triton-backend/image9.png" alt="Figure 9" />
  <img src="/assets/figures/2026-03-04-vllm-triton-backend/image10.png" alt="Figure 9" />
  <figcaption>图 9：Triton paged attention 与 FlashAttention 3 在 NVIDIA H100 和 AMD MI300 上的端到端延迟对比。结果以最左侧 baseline 归一化。</figcaption>
</figure>

## **预览：Helion 中的 Paged Attention**

[Helion](https://github.com/pytorch/helion) 是 PyTorch 团队推出的一种新领域特定语言，可以把它理解为更高层的 Triton，或者 tiled PyTorch。作为实验，一个简化版的 paged attention kernel 已经用 Helion 实现，并获得了相当令人鼓舞的早期结果。相关工作已经发布在 [PyTorch 博客](https://pytorch.org/blog/portable-paged-attention-in-helion/) 上，代码则以一个 [draft pull request](https://github.com/vllm-project/vllm/pull/27293) 的形式存在于 vLLM 仓库中。

## **结论**

随着模型、推理优化和硬件平台持续演进，性能可移植性变得越来越重要。vLLM 中的 Triton attention backend 证明了：通过单一、可移植的 kernel 实现，也完全有可能做到业界领先的 attention 性能。

通过精心的 kernel 设计、大量 microbenchmark，以及 persistent kernels 和 CUDA graphs 等系统级优化，Triton 后端在保持跨 GPU 厂商可移植性的同时，匹配甚至超越了许多高度专用的实现。今天，它已经是 AMD 上的默认 attention 后端，同时也在 NVIDIA 和 Intel 平台上使用同一份源代码高效运行。

本文概述了 Triton Attention 后端中最重要的一些优化。如果你想查看全部细节和更多 benchmark 结果，可以参考我们的相关论文 [The Anatomy of a Triton Attention Kernel](https://arxiv.org/abs/2511.11581)。

## Acknowledgments

这项工作由 IBM Research 的 AI platform 团队完成，感谢所有参与者：Burkhard Ringlein、Jan van Lunteren、Chih-Chieh Yang、Sara Kokkila Schumacher、Thomas Parnell、Mudhakar Srivatsa、Raghu Ganti。

## 参考链接

- [vLLM Office Hours](https://www.youtube.com/watch?v=8QiM-i9ifFo&list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3&index=1)
- [往期主题](https://www.youtube.com/playlist?list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3)
- [加入 Office Hours](https://red.ht/office-hours)
- [Triton](https://github.com/triton-lang/triton)
- [vLLM](https://github.com/vllm-project/vllm)
- [GPU Performance Portability needs Autotuning](https://arxiv.org/abs/2505.03780)
- [Attention backends 完整列表](https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends)
- [Triton attention backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py)
- [Microbenchmarks](https://github.com/foundation-model-stack/vllm-triton-backend)
- [PyTorch 博客](https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/)
- [Triton unified attention 代码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py)
- [Helion](https://github.com/pytorch/helion)
- [Helion PyTorch 博客](https://pytorch.org/blog/portable-paged-attention-in-helion/)
- [Draft pull request](https://github.com/vllm-project/vllm/pull/27293)
- [The Anatomy of a Triton Attention Kernel](https://arxiv.org/abs/2511.11581)
- [英文原文：blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html)
