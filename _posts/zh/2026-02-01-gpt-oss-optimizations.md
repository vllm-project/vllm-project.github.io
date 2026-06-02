---
layout: post
title: "GPT-OSS 在 NVIDIA Blackwell 上的性能优化：推动 Pareto 前沿"
author: "The vLLM and NVIDIA team"
image: /assets/figures/blackwell-inferencemax/gpt-oss-120b-8k-1k-nov-jan.png
lang: zh
translation_key: gpt-oss-optimizations
tags:
  - performance
  - hardware
---

**摘要：** vLLM 与 NVIDIA 团队联合为运行在 NVIDIA Blackwell GPU 上的 `gpt-oss-120b` 模型取得了显著性能突破。通过与 FlashInfer 的深度集成、基于 `torch.compile` 的新型内核融合，以及一系列推理运行时优化，我们把这个模型的性能 Pareto 前沿进一步向外推开，同时在**最大吞吐量上提升 38%**，并在**最佳交互性上提升 13%**。

本文详细介绍这条工程优化路径、关键技术突破，以及如何复现这些结果。持续更新的基准测试也可在 **[SemiAnalysis Inference MAX](https://inferencemax.semianalysis.com/)** 和 **[vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)** 上查看。

## 目录

- [引言](#introduction)
- [FlashInfer 与 torch.compile](#fi-tc)
- [运行时改进](#runtime)
- [部署配方](#recipes)
- [结果](#results)
- [下一步计划](#next-steps)
- [致谢](#acknowledgements)

---

## 引言 <a name="introduction"></a>

只针对单一指标做优化，无论是最大吞吐量，还是单批次延迟，在真实部署场景里通常都不够。不同业务有不同的延迟约束和请求并发需求。因此，真正的挑战在于优化 **Pareto 前沿**：也就是在 **每 GPU Tokens Per Second（TPS/GPU，对应 TCO，总拥有成本）** 和 **每用户 TPS（TPS/User，对应交互性）** 之间，找到最佳权衡曲线。把这条曲线向上、向右推进，意味着既能让单个用户拿到更快的生成速度，也能让更多用户共享同一套硬件。[SemiAnalysis InferenceMAX](https://inferencemax.semianalysis.com/) 正是围绕这一关键需求，系统化地测量、报告并推动现代 GPU 上的 LLM 推理性能改进。

本次优化的一个核心场景，是服务 OpenAI 的 `gpt-oss-120b` 模型。它是一个原生 4-bit 量化（MXFP4）的 Mixture-of-Experts（MoE）大语言模型，在这一体量上具备 SoTA 级别精度与很强的 agentic 能力。在最近的 SemiAnalysis InferenceMAX 展示中，vLLM 展示了其在 NVIDIA 最新 Blackwell（B200/GB200）架构上高效承载这一工作负载的能力。

这些优化的核心，在于硬件与软件的协同设计。NVIDIA B200/GB200 GPU 引入了很多关键能力，例如原生 FP4 TensorCore，以及每 GPU 192GB HBM，这些都是承载像 `gpt-oss` 这样的大型 MoE 模型所必需的。为了把这些硬件能力吃满，vLLM 与 NVIDIA 团队深度集成了 **FlashInfer**，并采用了一套非常明确的优化策略，重点围绕内核融合、通信开销削减，以及主机端与设备端的重叠执行来展开。

## FlashInfer 集成与基于 torch.compile 的融合 <a name="fi-tc"></a>

为了最大化 Blackwell TensorCore 的利用率，vLLM 将 **FlashInfer** 作为注意力、MoE 以及其他计算密集型和融合操作的主要内核后端。

**1. 关键计算内核集成：**

* **MoE 后端：** 我们为 MoE 操作启用了 `trtllm-gen`（[PR23819](https://github.com/vllm-project/vllm/pull/23819)）和 `cutlass`（[PR23696](https://github.com/vllm-project/vllm/pull/23696)）两种 FlashInfer 后端。这使得 vLLM 可以在专家路由与计算过程中选择性能最好的内核。除提供当前最优的 LLM 内核外，FlashInfer 还支持 JIT 编译、自动调优和内核缓存，这极大改善了高性能内核开发者的使用体验。  
* **FP8 KV-Cache：** 将 KV cache 以 FP8 精度存储，可以在同样的 KV cache 预算下服务更多并发请求。与此同时，部分注意力操作也可以在 FP8 精度下执行，从而降低注意力计算的计算/显存复杂度。为了在这个场景中获得最佳性能，vLLM 集成了 [FlashInfer 在 PR25674 中优化过的注意力内核](https://github.com/vllm-project/vllm/pull/25674/)。

**2. 基于 torch.compile 的图融合：** 我们优化工作的一个重要部分，是通过内核融合来降低显存访问与 kernel launch 开销。vLLM 并没有采用硬编码式的融合优化，而是基于 `torch.compile` 建立了一套[较完整的编译基础设施](https://github.com/vllm-project/vllm/tree/main/vllm/compilation)，让内核融合可以自动化完成。这种方式不仅提升了性能，也大幅降低了启用、泛化和维护此类优化的成本。

* **AR + Norm 融合：** 我们实现了 AllReduce（AR）与 RMSNorm 的融合。这一点对于张量并行（TP）部署尤其重要，因为在这类部署中，通信开销往往会成为瓶颈，详见 [PR20691](https://github.com/vllm-project/vllm/pull/20691)。  
* **Pad + Quant 与 Finalize + Slice：** 我们正在持续推出 [PR30647](https://github.com/vllm-project/vllm/pull/30647) 中的融合 pass，用于 padding/quantization 以及 finalize/slice 操作，以进一步精简 MoE 执行路径，预计可带来约 6% 的性能提升。

随着我们不断识别并开发新的融合操作，这套基础设施会继续为 vLLM 自动带来性能增益。

## 运行时改进 <a name="runtime"></a>

在 Blackwell 这样的新一代硬件上，GPU 快到 CPU（主机）经常会反过来成为瓶颈，因为它很难足够快地调度内核，持续让 GPU 保持繁忙。此外，`prepare_batch`、请求调度和采样逻辑本身也包含大量 CPU 侧处理。这种“主机开销”会表现为 kernel 执行之间的空洞，进而拉低性能和整体 GPU 利用率。

为了解决这个问题，我们在 vLLM 中实现了 **异步调度（Async Scheduling）** 和 **Stream Interval**，这两项能力能够有效压低主机侧开销。

[异步调度](https://github.com/vllm-project/vllm/pull/23569)：

* **机制：** 该调度器将 CPU 的请求调度与 GPU 的执行解耦。通过让 CPU 在 GPU 仍在处理当前 batch 的同时，提前准备下一批请求，我们等于把主机开销隐藏掉了。  
* **影响：** 对 `gpt-oss` 这类模型来说，这项优化在高吞吐和最低延迟两类场景中都很关键。在更强的 GPU 上，例如 H200、B200 和 GB200，通常可以看到约 10% 的性能收益。  
* **配置：** 在最近的 vLLM 版本中，这项能力已经默认启用。

[Stream Interval](https://github.com/vllm-project/vllm/pull/27869)：

* **机制：** 该特性通过先缓冲一部分生成 token，再统一发回客户端，降低网络响应的粒度。也就是说，引擎不再为每一个 token 单独触发一次网络调用，而是等到达到指定缓冲大小，也就是 “interval”，再批量发送。关键点在于，这套实现仍然保留了响应性：**第一个 token 依然会立即发送**，从而维持较低的 TTFT，而后续 token 才会被打包。  
* **影响：** 通过降低 HTTP/gRPC 响应分发频率，它显著减少了网络 I/O 与序列化相关的 CPU 开销。在高并发基准测试中，例如 `gpt-oss-20b` 配合 1024 并发请求，这项优化缓解了输出队列瓶颈，带来了 **57% 的端到端性能提升**，并改善了 TPOT（Time Per Output Token）。  
* **配置：** 用户可以使用 `--stream-interval <num_tokens>` 配置该行为。默认值是 `1`，也就是标准逐 token 流式；但在高吞吐部署里，把这个值增大，例如设成 `10`，通常会很有效地减少主机开销。

## 部署配方 <a name="recipes"></a>

上述大多数优化，在最新版本的 vLLM 中都已经默认启用。除此之外，如果你想在 Blackwell GPU（B200/GB200）上复现 `gpt-oss` 的优化性能，我们推荐在 vLLM 部署中使用如下配置。这些配置也可以在 [vLLM Recipes 页面](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) 中找到。

**推荐配置：**

* **图捕获：**
  * `--cuda-graph-capture-size 2048`
* **调度：**
  * `--api-server-count 20` 或 `--stream-interval 20`：这有助于把 HTTP API 服务器开销与推理引擎解耦，在高并发下稳定性能。
* **MoE 后端：**
  * 显式启用针对 FP8/FP4 MoE 优化过的 Cutlass 后端，以确保最大吞吐量：`VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`

## 结果 <a name="results"></a>

这些优化叠加之后，自 [InferenceMax](https://blog.vllm.ai/2025/10/09/blackwell-inferencemax.html) 发布以来，性能出现了明显跃升。尤其是，**最大吞吐量提升了 38%**，而**最低延迟场景提升了 13%**。

<p align="center">
<picture>
<img src="/assets/figures/blackwell-inferencemax/gpt-oss-120b-8k-1k-nov-jan.png" width="100%">
</picture><br>
</p>

这些收益并不是只对某一个用例有效，而是**覆盖整条 Pareto 曲线**，让整个 vLLM 社区都能受益。

## 下一步计划 <a name="next-steps"></a>

围绕 `gpt-oss` 的优化工作还在继续。下面是我们正在推进、用于进一步推动 Pareto 前沿的几个工程方向。完整列表也可以在 [Issue 30758](https://github.com/vllm-project/vllm/issues/30758) 中查看。

### 分离式架构

如果把 Prefill 阶段和 Decode 阶段拆到不同 GPU 上运行，我们有机会获得更高的每 GPU 吞吐量。我们目前正在持续实验这类部署方式，并寻找最优配置。

### 数据并行 + 专家并行性能

我们的预估显示，在同等延迟，也就是相同 TPS/user 下，使用 DEP2（Attention DP + MoE EP，运行在 2 张 GPU 上）有潜力实现比 TP1 和 TP2 更高的每 GPU 吞吐量。不过当前 DEP2 的表现仍然落后于 TP1/TP2，主要原因是 MoE 内核选择还不够理想。我们正在积极解决这个问题。

### 最低延迟性能

针对最低延迟场景，尤其是 TP8 并发 8 的配置，我们已经识别出几项值得继续优化的点：

* RoPE + Q + Cache 融合：FlashInfer 中已经有对应内核，vLLM 集成正在进行中  
* router gemm 与 fc_qkv/fc_o_proj gemm：我们可以使用更适合这类工作负载、并支持 PDL 的专用 tiny gemm 内核

## 致谢 <a name="acknowledgements"></a>

感谢 vLLM 社区中所有参与这项工作的工程师：

* Red Hat：Michael Goin、Alexander Matveev、Lucas Wilkinson、Luka Govedič、Wentao Ye、Ilia Markov、Matt Bonanni、Varun Sundar Rabindranath、Bill Nell、Tyler Michael Smith、Robert Shaw  
* NVIDIA：Po-Han Huang、Pavani Majety、Shu Wang、Elvis Chen、Zihao Ye、Duncan Moss、Kaixi Hou、Siyuan Fu、Benjamin Chislett、Xin Li、Vadim Gimpelson、Minseok Lee、Amir Samani、Elfie Guo、Lee Nau、Kushan Ahmadian、Grace Ho、Pen Chun Li
* vLLM：Chen Zhang、Yongye Zhu、Bowen Wang、Kaichao You、Simon Mo、Woosuk Kwon、Zhuohan Li
* Meta：Yang Chen、Xiaozhu Meng、Boyuan Feng、Lu Fang

## 参考链接

* [SemiAnalysis InferenceMAX](https://inferencemax.semianalysis.com/)
* [trtllm-gen MoE 后端 PR23819](https://github.com/vllm-project/vllm/pull/23819)
* [cutlass MoE 后端 PR23696](https://github.com/vllm-project/vllm/pull/23696)
* [FlashInfer FP8 注意力内核 PR25674](https://github.com/vllm-project/vllm/pull/25674)
* [vLLM torch.compile 基础设施](https://github.com/vllm-project/vllm/tree/main/vllm/compilation)
* [AR + Norm 融合 PR20691](https://github.com/vllm-project/vllm/pull/20691)
* [Pad + Quant 融合 PR30647](https://github.com/vllm-project/vllm/pull/30647)
* [异步调度 PR23569](https://github.com/vllm-project/vllm/pull/23569)
* [Stream Interval PR27869](https://github.com/vllm-project/vllm/pull/27869)
* [vLLM Recipes - GPT-OSS](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)
* [InferenceMax 博客](https://blog.vllm.ai/2025/10/09/blackwell-inferencemax.html)
* [后续优化 Issue 30758](https://github.com/vllm-project/vllm/issues/30758)
* [英文原文：blog.vllm.ai/2026/02/01/gpt-oss-optimizations.html](https://blog.vllm.ai/2026/02/01/gpt-oss-optimizations.html)
