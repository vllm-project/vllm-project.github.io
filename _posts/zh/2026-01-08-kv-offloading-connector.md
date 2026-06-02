---
layout: post
title: "深入解析 vLLM 新推出的 KV Offloading Connector：智能内存传输助力推理吞吐量最大化"
author: "Or Ozeri, Danny Harnik (vLLM Team at IBM Research)"
image: /assets/figures/2026-01-08-kv-offloading-connector/figure2.png
lang: zh
translation_key: kv-offloading-connector
tags:
  - performance
---

本文将介绍 vLLM 0.11.0 中引入的全新 KV 缓存卸载能力。我们重点关注 KV 卸载到 CPU 内存（DRAM）的做法，以及它如何提升整体推理吞吐量。博客后半部分还会进一步深入我们为优化 Host-to-Device 与 Device-to-Host KV 传输吞吐量所做的工作。

# 动机

为 LLM 提供服务是一项计算密集型任务，而其中一个核心环节就是计算所谓的 KV 数据。生成用户提示词回复的第一步，是为该提示词计算对应的 KV 值。这个阶段在请求生命周期中通常被称为 prefill。prefill 需要为每个提示词计算 KV 值，成本很高，通常也需要 GPU 这类专用加速硬件才能快速完成。

一个提示词计算出来的 KV 值，可以被其他共享相同前缀的提示词复用，从而避免重复计算。对于很多用例来说，缓存并复用 KV 值通常会带来两个直接收益：

* **降低请求延迟**，前提是从缓存读取 KV 比重新计算更快  
* **提升单节点吞吐量**，因为 GPU 核心负担变小后，可以承载更多并发请求

此外，**即便请求之间没有任何公共前缀，KV 缓存卸载也依然有价值**。具体来说，当系统同时处理很多并发请求时，GPU 可能会没有足够空间继续存放这些请求对应的 KV 值。这时推理引擎可能会抢占某个正在运行的请求，把它的 KV 值从 GPU 显存中丢掉。之后这个请求会被重新调度执行，于是它的 KV 值就必须重新计算。如果在该请求被抢占前，先把 KV 缓存卸载到更大的存储层，例如 CPU DRAM，就可以避免这部分重算成本。

## CPU 卸载

本文重点讨论将 KV 卸载到 CPU 内存（DRAM）。这种做法之所以重要，是多方面因素共同作用的结果：

* CPU RAM 在大多数部署环境里都很常见  
* 它的容量通常比 GPU 显存大，因此能容纳更大的 KV 缓存  
* CPU RAM 与 GPU 显存之间的传输通常兼具较低延迟和较高吞吐  
  再结合上一点，这使得 CPU 卸载**非常适合高效处理请求抢占**
* CPU RAM 还是进一步卸载到外部存储时一个**很方便的中间层**
  这在外部存储延迟较高时尤其有帮助

# 全新的 Offloading Connector

## vLLM Connector API

vLLM 很早就提供了一套用于读取和写入 KV 数据的 API，并把它集成进请求生命周期中，这套 API 就叫 Connector API。从高层次看，vLLM 在处理任意请求前，都会先查询这个 API，以便从外部源导入已有 KV 数据；当新的 KV 数据计算完成后，vLLM 也会再次调用该 API，把新生成的 KV 值写回外部目标。

这套 Connector API 最初是同步的。也就是说，当 vLLM 在外部加载或存储 KV 值时，整个 vLLM 引擎会被阻塞，无法并行处理新的请求 batch。到了 vLLM 0.9.0，Connector API 扩展为支持 **异步加载与异步存储 KV 数据**。Offloading Connector 正是基于这套新的异步 API 实现了 KV 缓存卸载。

我们引入的 **Offloading Connector**，允许异步地卸载和加载 KV 数据。它暴露出一套可插拔的 backend API，因此理论上可以接入任意介质来承载卸载目标。这也让接入新卸载后端变得很简单。你基本只需要定义一个传输函数，用来实现不同介质之间的 KV 数据拷贝。

Offloading Connector 自带了一个 CPU backend，使 vLLM 原生支持把 KV 数据卸载到 CPU。下文我们将聚焦于这一 CPU 卸载方案。

## 如何使用 Offloading Connector

如果要通过 Offloading Connector 启用 CPU 卸载，只需要在 `vllm serve` 命令中添加以下 CLI 参数：

```bash
--kv_offloading_backend native --kv_offloading_size <size_in_GB>
```

这个 CLI 形式依赖 [PR #24498](https://github.com/vllm-project/vllm/pull/24498)，预计会包含在 0.14.0 中。

对于旧版本，也可以通过以下 CLI 启用 CPU 卸载：

```bash
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks": <num_cpu_blocks>}}'
```

其中 `num_cpu_blocks` 表示为 CPU KV cache 分配的 CPU block 数量。

# 通过 Offloading Connector 进行 CPU 卸载的收益

我们展示了两类不同的微基准测试。第一类测量单请求的 TTFT，重点强调单个请求的加速收益；第二类测量系统在处理多个并发请求时的吞吐量，用于展示卸载如何帮助应对更重的负载。

在第一个基准测试中，我们测量了处理单个 prefill 请求时的延迟，对比的是：从 CPU 缓存加载 KV，和直接由 GPU 重算 KV 值。

<p align="center">
<picture>
<img src="/assets/figures/2026-01-08-kv-offloading-connector/figure1.png" width="100%">
</picture><br>
<b>图 1</b>：单请求 TTFT（Llama-3.1-8B-Instruct，NVIDIA H100）。
</p>

结果表明，从 CPU 加载 KV 值可以把 TTFT 降低 **2 倍到 22 倍**，具体取决于提示词大小。本文末尾附有我们所用的精确测试设置和基准代码。

需要注意的是，KV 卸载，也就是把 KV 数据从 GPU 拷到 CPU，这部分延迟通常不是用户可感知的，因为它理论上不应该影响响应时间。原因在于卸载本身也是异步执行的，请求可以在传输完成之前就结束。因此，**使用 Offloading Connector 对 cache miss 的 TTFT 影响非常小**。

接下来，我们评估在处理多个并发请求时，CPU 卸载对系统整体吞吐量的影响。具体做法是提交一批 10,000 个彼此唯一的请求，每个请求长度为 512 token，并测量在不同 CPU 缓存命中率下所达到的吞吐量。

我们测量处理完这批请求所需的时间，不包括 CPU 缓存预热时间，然后据此推导 token/s 吞吐量。为了更纯粹地观察 CPU 缓存的效果，这里没有启用 GPU 缓存。

<p align="center">
<picture>
<img src="/assets/figures/2026-01-08-kv-offloading-connector/figure2.png" width="100%">
</picture><br>
<b>图 2</b>：并发请求吞吐量（Llama-3.1-8B-Instruct，NVIDIA H100，10000 个 512 token 的 prefill 请求）。
</p>

结果显示，吞吐量会随着 CPU KV 缓存命中率的提高而提升。我们观察到，**吞吐量最高可提升 9 倍**，而这个提示词长度下的 TTFT 只降低了 2 倍。这说明 **KV 缓存卸载最核心的收益其实是吞吐量最大化**。

## Offloading Connector 在不同 vLLM 版本中的演进

需要特别指出的是，**Offloading Connector 在 vLLM 0.12.0 中性能有了显著提升**。例如，在 Llama-3.1-8B-Instruct + NVIDIA H100 的测试中，我们看到 TTFT 最多降低 **4 倍**，吞吐量最多提升 **5 倍**。这一改进的关键细节，我们会在下文讨论 vLLM 物理块大小时展开。

更多改进预计会在即将到来的 0.14.0 中引入，尤其包括：

* 允许被抢占的请求从 CPU 加载回 KV（[PR #29870](https://github.com/vllm-project/vllm/pull/29870)）  
* 修复卸载与模型计算之间的竞态条件（[PR #31341](https://github.com/vllm-project/vllm/pull/31341)）

本文中的评估结果已经包含了这些改进。

# GPU-CPU 传输技术评估

接下来的部分，我们会对设计 CPU 卸载时的一些关键取舍做更深入的技术讨论。重点是：如何在最大化 GPU-CPU 吞吐量的同时，尽量减小对 GPU 核心和 CPU 核心的干扰，从而进一步优化整体推理吞吐量。

如前所述，为 Offloading Connector 定义 backend 时，最核心的部件就是**传输函数**。对于 CPU backend 来说，这个函数负责在 GPU 内存和 CPU 内存之间双向复制数据。目前它支持 **兼容 CUDA 的设备**，也就是 NVIDIA 和 AMD。

CPU backend 当前实现的传输函数使用的是 `cudaMemcpyAsync`，它会调用 GPU 上的 DMA（Direct Memory Access）硬件模块。这个硬件模块专门用于设备端（GPU）与主机内存之间的高吞吐数据传输。而且，使用 DMA 进行传输时，对 CPU 和 GPU 核心的开销都非常小。这一点非常重要，因为我们的传输是与模型计算异步并行进行的。

DMA 在处理物理上连续且较大的内存拷贝时，吞吐量最好。这意味着，卸载性能会明显依赖于 KV 数据在内存中的布局。KV 数据块越大，模型表现就越好。

但 DMA 究竟有多快？它和用自定义 CUDA kernel 做拷贝相比又如何？

为回答这个问题，我们实现了一个微基准 [gpu_cpu_benchmark](https://github.com/orozery/playground/tree/kv-offloading-blog-dec-2025/kvcache/gpu_cpu_benchmark)。在这个基准中，我们测试了两种 GPU 与 CPU 之间的数据拷贝方式：

* 使用 **DMA** 拷贝，也就是通过 `cudaMemcpyAsync`  
* 使用 **自定义 CUDA kernel** 拷贝，让 **GPU 核心** 通过原始指针复制 16 字节字。这种方式之所以有效，是因为它利用了 GPU 核心的大规模并行能力；但与此同时，它也会对 GPU 核心本身承担的模型计算造成更强干扰

我们的第一个测试，测量在单次 1000 个 block 传输中，不同 block 大小（从 4KB 到 16MB）的吞吐量表现：

<p align="center">
<picture>
<img src="/assets/figures/2026-01-08-kv-offloading-connector/figure3.png" width="100%">
</picture><br>
<b>图 3</b>：单次 GPU -> CPU 传输吞吐量（NVIDIA H100，单次传输 1000 个 block）。
</p>

<p align="center">
<picture>
<img src="/assets/figures/2026-01-08-kv-offloading-connector/figure4.png" width="100%">
</picture><br>
<b>图 4</b>：单次 CPU -> GPU 传输吞吐量（NVIDIA H100，单次传输 1000 个 block）。
</p>

结果验证了一个关键事实：**DMA 在大 block 上表现很好，但在小 block 上就不够理想**。对于较小 block，自定义 kernel 的吞吐量会明显更高。不过我们也观察到，自定义 kernel 的结果噪声更大，方差更高。

接下来，我们继续测试双向传输吞吐量，也就是同时发起两个并发传输，一个读、一个写。这里我们把 block size 固定为 2MB，并调整两个方向之间的传输比例。对这两种拷贝机制而言，当两个方向传输量接近时，都能达到峰值吞吐。但虽然单向传输两者都能跑到大约 50GB/s，双向传输的结果就明显不同了：

* DMA 达到 83.4 GB/s  
* 自定义 kernel 达到 68.5 GB/s

因此，在两种方法之间做选择时，问题就变成了：

* **vLLM 实际使用的有效 block size 到底有多大？**  
  这取决于服务的模型和 vLLM 配置。下一节我们会针对当前常见模型给出具体答案。  
* **这两种方法分别会怎样影响 GPU 模型计算性能？**  
  别忘了，Offloading Connector 的设计目标，是在 GPU 做模型计算的同时并行卸载或加载 KV 数据。我们后面的评估会看到，这两种方案如何影响整体吞吐量。

# 改变 vLLM 的内存布局

这一节我们将介绍我们对 vLLM 中 GPU 内存布局所做的修改，使其更适合 KV 传输，同时又不牺牲模型计算性能。

先从默认布局说起。vLLM 默认按 token block 的粒度来分配 GPU 内存，缺省是每块 16 个 token。实际物理布局会取决于使用的注意力后端，例如 FlashAttention 或 FlashInfer，以及所服务的模型。

当前最常见的是统一模型（uniform model）：模型由多层组成，每层都有自己的 KV cache，而且各层形状相同。vLLM 也支持混合模型（hybrid model），但这一类目前还没有针对 Offloading Connector 做优化。对于统一模型，vLLM 会给每一层单独分配一份 KV cache，所以一个逻辑 block 的 KV 数据会被打散成 `num_layers` 份，每层一份。此外，根据注意力后端的不同，每层 block 还可能继续被拆成两个子块，分别用于 K cache 和 V cache。

这种碎片化对模型计算性能并没有意义，但对 KV 卸载却非常糟糕，因为它在 KV cache 布局中引入了大量无意义碎片，使有效 block size 变得更小。为解决这个问题，我们最近 upstream 了一个 [KV cache 布局改动](https://github.com/vllm-project/vllm/pull/27743)，让所有层的 KV 数据能够组成一个连续的物理块。这个改动使物理块大小有效放大了 `2 * num_layers` 倍，从而让 **Offloading Connector 的吞吐量提升了一个数量级**。

下表总结了若干常见模型在旧布局（0.11.0）与新布局（0.12.0）下的物理 block size，对比时假设 vLLM 使用 16 token block：

| 模型 | 旧块大小 | 新块大小 |
| :---- | :---- | :---- |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (tensor_parallel_size=2) | 16 KB | 2 MB |
| deepseek-ai/DeepSeek-V2-Lite-Chat (GPU block size=64) | 72 KB | 1.9 MB |
| meta-llama/Llama-3.1-8B-Instruct | 32 KB | 2 MB |
| meta-llama/Llama-3.2-1B-Instruct | 16 KB | 0.5 MB |
| meta-llama/Llama-3.1-70B-Instruct | 8 KB | 1.25 MB |
| mistralai/Mistral-7B-Instruct-v0.2 | 32 KB | 2 MB |
| mistralai/Mistral-Small-24B-Instruct-2501 | 32 KB | 2.5 MB |
| Qwen/Qwen2.5-3B-Instruct | 8 KB | 0.56 MB |
| Qwen/Qwen3-0.6B | 32 KB | 1.75 MB |
| Qwen/Qwen2.5-7B-Instruct | 16 KB | 0.87 MB |
| Qwen/Qwen3-4B-Instruct-2507 | 32 KB | 2.25 MB |
| Qwen/Qwen2.5-1.5B-Instruct | 8 KB | 0.44 MB |
| Qwen/Qwen3-8B | 28 KB | 1.97 MB |
| Qwen/Qwen3-1.7B | 32 KB | 1.75 MB |
| Qwen/Qwen3-32B (tensor_parallel_size=2) | 16 KB | 2 MB |

可以看到，在新的 vLLM KV cache 布局下，物理块大小大约在 0.5MB 到 2MB，而旧布局里通常只有几 KB。结合前面 GPU-CPU 微基准结果，我们预计 **DMA 方法的性能将与自定义 kernel 相当，或者仅略逊一筹**，具体取决于模型。

# 拷贝方法的端到端评估

接下来，我们使用两个 vLLM 微基准测试，对比 Offloading Connector 的两个变体：

* 上游版本，使用基于 DMA 的传输函数  
* 一个补丁版本，使用我们在 GPU-CPU 微基准里实现的自定义 kernel

我们特意选择展示 Offloading Connector 的**最差情况模型**，也就是物理 block size 相对较小，仅 0.5MB 的场景。

<p align="center">
<picture>
<img src="/assets/figures/2026-01-08-kv-offloading-connector/figure5.png" width="100%">
</picture><br>
<b>图 5</b>：单请求 TTFT（Llama-3.2-1B-Instruct，NVIDIA H100）。
</p>

在单请求基准中，我们看到**自定义 kernel 的 TTFT 略好一些**：对于 1K 提示词，差异不到 1ms；对于 90K 的大提示词，差异最高可到 15ms。考虑到在 0.5MB block size 下 GPU-CPU 微基准的表现，这是符合预期的。对于物理 block 更大的模型，两种方案的结果基本相同。

<p align="center">
<picture>
<img src="/assets/figures/2026-01-08-kv-offloading-connector/figure6.png" width="100%">
</picture><br>
<b>图 6</b>：并发请求吞吐量（Llama-3.2-1B-Instruct，NVIDIA H100，10000 个 512 token 的 prefill 请求）。
</p>

但在并发请求测试里，我们看到**DMA 的吞吐量优于自定义 kernel**。这个优势在 0% 命中率时大约是 5.5%，在 80% 命中率时扩大到大约 15%。

背后的原因是，自定义 kernel 会与模型计算争抢 GPU 核心，因为两者都在使用 GPU cores。对 0% 命中率而言，自定义 kernel 方案的吞吐量实际上比完全不开启 CPU 卸载还差 6%。而在 100% 命中率时，由于没有模型计算与 CPU 加载并行，因此两种方案之间的差距会缩小。

需要强调的是，我们这里展示的是 **DMA 方法最不利模型** 下的结果。大多数常见模型都拥有更大的物理块大小，因此通常会更偏向 DMA。以 **Llama-3.1-8B-Instruct** 为例，DMA 在 TTFT 基本持平的前提下，吞吐量最高可比自定义 kernel 高出 **32%**。

总结来说，我们通过修改 GPU 内存布局，使得 vLLM 能够高效使用 DMA 进行 KV 传输，并最终获得更好的整体吞吐量。

# 评估环境与基准代码

为了评估 vLLM 的 CPU 卸载能力，我们使用了如下测试环境：

* 单机 Ubuntu 24.04.1 LTS 容器  
* 内核 5.14.0-427.81.1.el9_4.x86_64  
* Intel Xeon SapphireRapids 2.1Ghz（限制 8 核）  
* NVIDIA H100 80GB HBM3  
* 500GB DRAM  
* CUDA 版本 12.9  
* vLLM commit hash：`2a1776b7ac4fae7c50c694edeafc1b14270e4350`  
* Flash Attention 后端  
* 禁用 GPU prefix caching（以便单独评估 CPU 命中）  
* GPU block size 为 16 token  
* CPU block size 为 16 token  
* 禁用反/分词

我们的基准测试代码可以在[这里](https://github.com/orozery/playground/blob/kv-offloading-blog-dec-2025/kvcache/kv_offload_benchmark.py)找到。

## 下一步计划

我们会继续增强 vLLM 原生 KV 卸载能力。下一个里程碑，是让 CPU KV cache 能作为面向更慢存储卸载时的中间层。

和以往一样，我们的首要目标仍然是正确性与性能。欢迎你实际试用、分享测试结果，并告诉我们是否遇到了任何问题。

**加入讨论：** 欢迎在 [vLLM Slack](https://vllm-dev.slack.com/archives/C09AYJFFLKD) 的 `#feat-v1-cpu-offloading` 频道分享你的用例与反馈。
