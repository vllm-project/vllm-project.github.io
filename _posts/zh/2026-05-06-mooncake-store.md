---
layout: post
title: "vLLM × Mooncake：用分布式 KV Cache Pool 服务大规模 Agentic 推理"
author: "Yifan Qiao, Trong Dao Le, Ao Shen, Zhewen Li, Bowen Wang"
image: /assets/figures/2026-05-06-mooncake-store/hero_vllm_mooncake.svg
lang: zh
translation_key: mooncake-store
tags:
  - agentic
  - kv_cache
  - large-scale-serving
  - disaggregation
---

<p align="center">
<img src="/assets/figures/2026-05-06-mooncake-store/hero_vllm_mooncake.svg" width="95%">
</p>

**TL;DR：** Agentic 工作负载会反复带着上万 token 的共享前缀做 prefill，把这部分前缀复用起来，每轮真正要算的只有几千 token 的增量。vLLM 通过集成 Mooncake Store，把 KV cache 池化为跨实例的分布式资源。在真实 Codex / SWE-bench Pro trace（Kimi-2.5 NVFP4，GB200，1P1D）上，吞吐提升 **3.8 倍**，P50 TTFT 降低 **46 倍**，端到端延迟降低 **8.6 倍**；扩展到 **60 张 GB200** 仍保持 **95% 以上**的缓存命中率，接近线性扩展。

## Agentic 工作负载正在重塑 LLM 推理

随着 Claude Code、OpenClaw 等 LLM agent 的兴起，推理负载正在发生本质变化。正如黄仁勋在 GTC 2026 [keynote](https://www.nvidia.com/gtc/keynote/) 中强调的，LLM 正在从简单的对话机器人，演进为可以自主规划、推理、并采取行动以达成复杂目标的长时运行系统。

Agentic 工作负载在结构上也很不一样：它通常是长 horizon 的多轮 loop，在 *reasoning step*（处理上下文、产生中间思考）和 *action step*（发起工具调用、接收外部输出）之间反复切换。

为了量化这种行为，我们采集并分析了 Codex 与 GPT-5.4 在 SWE-bench Pro 上的真实 trace，并把数据集[开源在 HuggingFace](https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces)，方便社区共同研究 agentic serving。

图 1 总结了 Codex / SWE-bench Pro trace 的结构，并展示了一条代表性的 agentic session。

<p align="center">
<img src="/assets/figures/2026-05-06-mooncake-store/agentic_trace.svg" width="90%">
<br>
<em>图 1：Codex / SWE-bench Pro trace 的解剖。每行是一次 LLM 调用；每轮的尺寸取自 610 条 trace 的中位数。系统提示、skills/memory、历史轮次构成的前缀会被反复复用，每轮真正“新进来”的只有 tool 输出和模型 decode 部分。</em>
</p>

数据非常鲜明：到第 30 轮时，上下文长度通常已经增长到约 **80K tokens**，最长可超 **180K tokens**。但每轮新增的内容往往只有几百到几千 token，其余都是模型已经见过的前缀。整个数据集的平均 input/output token 比约为 **131:1**。

如果能把这些前缀缓存住，prefill 在这部分上几乎就是“白嫖” 每轮真正的成本只剩下增量。

在 610 条 trace、中位数 33 轮的数据集上我们观察到：

- 94.2% 缓存命中率
- 131:1 input/output token 比
- 每轮上下文平均增长约 2,242 tokens
- 单条 trace 上下文中位数从 12K 增长到 80K tokens
- 轮间间隔从中位数 5.2s 到 P99 81.4s

但本地 KV offload（CPU DRAM / 磁盘）在 agentic 场景有两个硬限制：

- **容量与淘汰。** 100K token 的上下文动辄 GB 级（例如 Kimi-2.5 FP8 KV cache 大约 3.8 GB）。一台繁忙的实例上挂了大量长会话，本地容量很快就会撑不住，触发淘汰。
- **跨实例不命中。** 为了均衡负载，路由器不一定把同一会话的下一轮送回原实例。一旦切到新实例，就得从零开始重算前缀。

**结论：** 我们不能再把推理服务看作一组孤立的 vLLM 副本。Agentic 工作负载需要实例之间共享一个分布式 KV cache 池，既扩容量、又跨实例命中。

## 用 Mooncake Store 构建分布式 KV Cache Pool

[Mooncake](https://github.com/kvcache-ai/Mooncake) 是一个开源的高性能 KV cache 传输与分布式存储库。vLLM 此前已经通过 [`MooncakeConnector`](https://docs.vllm.ai/en/stable/features/mooncake_connector_usage/) 把 Mooncake transfer engine 用在 prefill-decode（PD）解耦上，用来在 GPU 间搬运 KV cache。这次我们把集成再往前推一步，基于 Mooncake Store 构建一个分布式 KV cache 池。

图 2 展示了整体设计。

<p align="center">
<img src="/assets/figures/2026-05-06-mooncake-store/overall_design_option_C.svg" width="90%">
<br>
<em>图 2：vLLM 分布式 KV cache 池整体设计。多个 vLLM 实例内嵌 Mooncake client，共享一个集群级 Mooncake Store。Mooncake master 管理 KV block 元数据、服务发现和 client 健康；worker 通过 RDMA 在 GPU HBM 与分布式 DRAM/SSD 池之间搬运 KV block。</em>
</p>

整体上，Mooncake Store 提供一个 master server 和一组 client。Master server 全集群运行，负责管理元数据，包括 KV block hash、大小等；同时监控 client 的健康与可用性，提供服务发现和死节点清理。

Mooncake client 跑在每个 GPU 节点上，管理本地 CPU / DRAM / SSD 资源，并通过 RDMA 互联做 KV cache 传输。所有 client 共同组成一个分布式的 KV cache 池。

vLLM 这一侧的集成接入了已有的 [`KVConnector`](https://github.com/vllm-project/vllm/blob/db9a84e0cd0e17ab693467ff4a71103abd4b77bf/vllm/distributed/kv_transfer/kv_connector/v1/base.py) 接口，PD 解耦也是用的这套抽象。Connector 在 vLLM 中分成两个角色：

在 **scheduler 侧**，新请求到来时，vLLM 先 hash prompt 的 token block，向 Mooncake master 查询匹配的 KV cache block，并据此指导调度决策。

在 **worker 侧**，每个 GPU worker 内嵌一个 Mooncake client，并启动后台线程做数据搬运。GPU KV cache 内存会被注册为 RDMA buffer，使得 GPUDirect RDMA 读写可以直接通过 Mooncake client 完成，不占 SM，也不经过 CPU staging。

## 设计亮点

### GPUDirect RDMA：免 SM、零拷贝的 KV 传输

传统 GPU 到 CPU 的数据传输通常走两种路径：一种是 `cudaMemcpyAsync`，使用 GPU copy engine，但在大量小传输时吞吐不一定理想；另一种是发起专用 GPU kernel 做复制，利用 SM 传输，但可能与其他 GPU kernel 互相干扰。

我们走第三条路：使用 RDMA NIC 配合 GPUDirect RDMA，直接在 GPU HBM 和远端 CPU 内存之间搬运 KV block。这条路径不需要 staging buffer，也不消耗 SM，同时在大量小 KV block 传输上表现良好。

借助 Mooncake Transfer Engine，传输路径还能利用单节点上的多张 RNIC 做 multi-NIC pooling 和拓扑感知路径选择，让 KV 传输能更充分聚合并利用网卡带宽。

### 全异步传输

虽然 RDMA 操作本身是异步的，但准备 descriptor、发起 RDMA 读写仍然需要一定 CPU 工作量。这部分开销会随序列长度增长，因为序列越长，涉及的 KV block 越多。

为了避免阻塞主 CPU 路径，进而拖延 GPU kernel launch，所有 RDMA 操作都运行在专门的后台 I/O 线程上。从 vLLM 的视角看，整条传输路径是完全异步的。

### 通过 MultiConnector 同时启用 PD 解耦 + 分布式 KV Cache Pool

这套集成天然也可以扩展到 PD 解耦，通过 [`MultiConnector`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py) 接口实现。如图 3 所示，`MultiConnector` 是一个把多个子 connector 串起来的 wrapper，每个 connector 独立工作、互不依赖。

<p align="center">
<img src="/assets/figures/2026-05-06-mooncake-store/animation.gif" width="80%">
<br>
<em>图 3：通过 MultiConnector 把 PD 解耦与分布式 KV cache 池结合起来。</em>
</p>

**Prefill：** prefill 实例为 PD connector 准备 KV block 的同时，也通过 store connector 把它们写入分布式 KV cache 池。命中时，vLLM 会查询所有 connector，并能从 Mooncake Store connector 恢复匹配的前缀。

**Decode：** decode 实例写入分布式池的 KV block，对 prefill 实例立即可见。当前 decode 自身不从池里读取，因为 vLLM 会把每个请求同时调度到一个 prefill 实例和一个 decode 实例，由 prefill 实例从池里加载前缀 KV block，再通过 PD connector 转发给 decode。

我们正在推进多路径 KV cache 加载：从 prefill 实例和分布式池同时拉取，以进一步打满可用网络带宽。

## 性能

当前实现对应 PR 在[这里](https://github.com/vllm-project/vllm/pull/40900)。Benchmark 脚本放在 artifact 仓库[这里](https://github.com/ivanium/vllm/tree/feat/mooncake-store-int/scripts/mooncake/artifacts)。本文挑两组结果。

实验配置是：Kimi-2.5 NVFP4 模型，GB200 节点，PD 解耦。Prefill 实例使用 TP4，decode 实例使用 DP8 + EP。我们发现这是 latency / throughput 平衡最好的配置。

### 在真实 agentic trace 上的加速

我们先用前面提到的 Codex agentic trace 做一组真实场景评估：1P1D 部署，共 **12 张 GPU**。

<p align="center">
<img src="/assets/figures/2026-05-06-mooncake-store/pd_compare_mooncake_vs_nixl.png" width="90%">
<br>
<em>图 4：vLLM + Mooncake Store vs. baseline，在真实 Codex agentic trace 上（1P1D，12 张 GB200）。分布式 KV cache 池把吞吐提升 3.8 倍、P50 TTFT 降低 46 倍、E2E 延迟降低 8.6 倍，这背后是缓存命中率从 1.7% 跳到 92.2%。</em>
</p>

分布式 KV cache 池把 vLLM 吞吐提升 **3.8 倍**，P50 TTFT 与 E2E 延迟分别降低 **46 倍** 与 **8.6 倍**。这些收益的根源是缓存命中率的剧烈提升：从只能缓存系统提示的 **1.7%**，提升到几乎全前缀命中的 **92.2%**。

### 跨节点扩展

为了进一步压测跨节点 datapath，我们扩到了更多节点，并使用了一份基于 Codex 工作负载衍生出来的合成数据集来做受控扩展实验。

实验设置：

- 20K 公共 token（系统指令）
- 第一轮输入 10K token
- 后续每轮输入 2,048 token
- 每轮输出 900 token
- 共 30 轮
- 会话数随 GPU 数线性增长：75 → 150 → 225 → 300 → 375
- 参数大致对齐 Codex 工作负载，整体 output/input 比保持在约 1.3%

<p align="center">
<img src="/assets/figures/2026-05-06-mooncake-store/pd_scaling.png" width="90%">
<br>
<em>图 5：在 round-robin 路由下，从 12 张 GB200 扩展到 60 张 GB200 的吞吐扩展。所有规模下系统都保持 95% 以上的缓存命中率，并接近线性扩展。</em>
</p>

为了把 datapath 推到极限，我们特意使用 round-robin 路由。这样同一会话的不同轮次会被调度到不同节点，迫使系统跨节点拉取 KV cache。

在没有分布式 KV cache 池的方案上，这种路由模式会带来大规模 cache miss 与严重的吞吐崩塌；而 vLLM + Mooncake Store 始终保持 **95% 以上**的命中率，并从 12 GPU 扩展到 60 GPU 时接近线性扩展。

这一结果说明：分布式 KV cache 池在显著提升缓存命中率的同时，仍然能让数据通路在集群规模扩大时保持高效。

## 接下来的工作

我们正在推进以下方向：

- **分布式磁盘 offload。** 把存储层级扩展到 NVMe SSD 与分布式文件系统，进一步扩容缓存。
- **混合架构模型的 KV offload。** 支持新兴的混合 attention 模型，不同层可能需要不同的缓存策略。
- **Cache-aware routing。** 让请求路由器和 KV cache 池协同决策，优先把会话送到已经持有相关前缀的实例上，最大化本地命中，必要时再回退到分布式池。
- **数据通路进一步优化。** 在 RDMA 之外利用 NVIDIA 多节点 NVLink，做更快的多路径 KV 传输；同时探索类似 [DualPath](https://arxiv.org/abs/2602.21548) 的从 prefill 与 decode 实例同时加载 KV，进一步打满聚合带宽。

## 致谢

vLLM Mooncake Store 集成在很大程度上受到 [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) 中已有工作的启发。特别感谢 Ant Group 的 Chao Lei 贡献了初版实现，感谢 Inferact 的 Zijing Liu 贡献了 agentic trace 与分析。

同时感谢 Approaching.AI 的 Jiahao Lu、Zuoyuan Zhang、Zihan Tang、Ke Yang；Huawei 的 Pengbo Zhao、Fuqiao Duan、Tianyu Xu；Alibaba Cloud Computing 的 Tianchen Ding、Xuchun Shang、Xingrui Yi、Teng Ma；Ant Group 的 Yunxiao Ning、Dejiang Zhu、Shoujian Zheng；以及 9#AISoft 的 Feng Ren，他们都提供了宝贵的工程反馈。

也感谢 vLLM 与 Mooncake 社区的支持与建议。最后特别感谢 Inferact 团队全程的紧密协作与讨论。

## 相关链接

- [vLLM PR：vllm-project/vllm#40900](https://github.com/vllm-project/vllm/pull/40900)
- [Mooncake：github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- [开源 agentic trace 数据集：huggingface.co/datasets/Inferact/codex_swebenchpro_traces](https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces)
- [英文原文：vllm.ai/blog/mooncake-store](https://vllm.ai/blog/mooncake-store)
