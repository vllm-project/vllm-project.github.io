---
layout: post
title: "不止于能跑：vLLM 如何在 AMD ROCm 上做原生推理优化"
author: "AMD and Embedded LLM"
image: /assets/figures/2026-02-27-rocm-attention-backend/ROCm-Attention-rocm_aiter_fa.png
math: true
lang: zh
translation_key: rocm-attention-backend
tags:
  - performance
  - hardware
---

## 引言

长期以来，启用 AMD 支持意味着“移植”，也就是先让代码跑起来。**这个时代已经结束了。**

面对 AMD CDNA<sup>TM</sup> 3 架构硬件（AMD Instinct<sup>TM</sup> MI300X、Instinct MI325X、Instinct MI355X GPU）以及 DeepSeek MLA 这样的复杂模型结构，“能跑”已经远远不够。这类工作负载需要的是架构协同设计，也就是让软件调度层与底层硬件原语紧密配合。

vLLM 现在在 AMD ROCm<sup>TM</sup> 上提供了 7 个注意力后端。本文会逐一解释它们为什么存在、各自的取舍，以及在什么场景下应该选哪个。我们还给出了覆盖所有后端的透明基准测试，并展示 `ROCM_AITER_FA`（用于 MHA，Multi-Head Attention）和 AITER MLA（Multi-Head Latent Attention）后端，如何借助 AMD AITER 原语与 vLLM 的内核编排，实现 **1.2 到 4.4 倍的吞吐提升（TPS）**。

---

## 每个 Batch 都在处理混合工作负载

在生产环境的 LLM 推理中，每一步推理处理的，通常都是由不同请求类型 token 混合而成的一个 batch。业界已经充分认识到这个问题，并形成了多种路线：有的使用内部调度很复杂的统一内核，有的使用专门内核配合多路路由。AMD 的 `ROCM_AITER_FA` 采用的是显式路由这一路线，把工作负载感知优化直接提升为核心设计原则，而不是隐藏在内核实现细节里。

- **Prefill**：服务器刚收到的新 prompt。这类请求通常包含成千上万个输入 token，需要一次性完成注意力计算。此时 GPU 主要在做大规模矩阵乘法，因此 prefill 是**计算密集型**工作负载。
- **Extend**：为一个已经部分构建好 KV cache 的请求继续处理 prompt 侧 token，例如 chunked prefill、prefix cache 复用，或者多轮对话中的后续输入。因为这些新 token 既要关注已有缓存上下文，也要关注本轮新输入，所以 extend 是一种混合型工作负载。在线服务的调度器会利用这个阶段，把长 prompt 切成小段，并和其他进行中请求的 decode 交错执行，从而在延迟和吞吐之间取得更好的平衡。
- **Decode**：逐个生成输出 token。每次 decode 都要从内存中加载整段 KV cache 来生成一个 token，瓶颈主要是内存带宽，因此 decode 是**内存密集型**工作负载。

这些请求类型会随机到达，然后被合并批处理以提高整体效率。

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/continuous-batching.png"
    width="100%"
    alt="Continuous batching diagram" />
  <figcaption>5 个并发请求的在线推理示意。第 4 步中，prefill、extend 和 decode token 被合并进同一个 batch。</figcaption>
</figure>

优化上的难点在于：prefill 需要更大的 tile size 和更高的 ALU 利用率，而 decode 需要更连续的内存访问和尽量少的 cache 读取。**一个为某类工作负载调优的内核，往往会在另一类场景中留下性能空间。**

这正是 `ROCM_AITER_FA` 三路路由要解决的问题：它不强迫所有请求类型共用一个内核，而是把每种请求路由到针对该类工作负载特征优化的专用内核。

---

## 其他 MHA 后端

在深入 `ROCM_AITER_FA` 之前，先看一下其他可用的 MHA 后端。

### 统一注意力后端

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/ROCm-Attention-unified-attn.png"
    style="width: 60%; display: block; margin: 0 auto;"
    alt="Unified attention kernel flow diagram" />
  <figcaption>统一注意力后端会把所有 token 通过一个内核路径处理。</figcaption>
</figure>

这类后端会把所有 token（prefill、extend、decode）统一送入单一内核路径：

| 后端 | 内核来源 | 用途 |
| ---- | -------- | ---- |
| [TRITON_ATTN](https://github.com/vllm-project/vllm/blob/v0.14.0rc2/vllm/v1/attention/backends/triton_attn.py) | [vLLM Triton kernel](https://github.com/vllm-project/vllm/blob/v0.14.0rc2/vllm/v1/attention/ops/triton_unified_attention.py) | 默认回退后端 |
| [ROCM_AITER_UNIFIED_ATTN](https://github.com/vllm-project/vllm/blob/v0.14.0rc2/vllm/v1/attention/backends/rocm_aiter_unified_attn.py) | [AITER Triton kernel](https://github.com/ROCm/aiter/blob/v0.1.10.post3/aiter/ops/triton/_triton_kernels/attention/unified_attention.py) | AITER 的单内核路径 |

```python
def forward():
    # 阶段 1：将 Key/Value 写入 KV Cache
    reshape_and_cache_flush(new_key, new_value, ...)
    # 阶段 2：由单一内核完成所有注意力计算
    unified_attention_kernel(new_query, KV-Cache, ...)
```

### ROCM_ATTN：遗留的双路后端

[ROCM_ATTN](https://github.com/vllm-project/vllm/blob/v0.14.0rc2/vllm/v1/attention/backends/rocm_attn.py) 使用的是双路路由，不同阶段走不同内核：

- **Prefill**：Triton 内核
- **Decode**：HIP paged attention 内核（在受支持配置下）

这个后端有两个重要特征：

1. **遗留的双路架构**：prefill 使用 Triton，decode 使用 HIP paged attention。需要注意的是，HIP paged attention 只支持部分 KV head size。对于不受支持的配置，例如 Qwen3-235B，它会回退到 Triton decode 内核，导致性能显著下降。
2. **支持 Radeon GPU**：它与 `TRITON_ATTN` 一样支持 **Radeon GPU**，适合 AITER 原语不可用的消费级硬件部署场景。

---

## ROCM_AITER_FA：面向 AMD 的内核调度系统

`ROCM_AITER_FA` 不只是一个内核封装层，它本质上是一个更复杂的编排层：把请求路由到不同的专用内核上，将 vLLM 的高层管理能力与 AMD AITER 原语结合在一起。

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/ROCm-Attention-rocm_aiter_fa.png"
    style="width: 60%; display: block; margin: 0 auto;"
    alt="Flowchart diagram of ROCM_AITER_FA architecture" />
  <figcaption>ROCM_AITER_FA 会把 token 路由到三条专用路径上。</figcaption>
</figure>

### 关键创新

1. **三路路由**：系统会把请求动态分类到 Decode、Prefill、Extend 三条路径上，每条路径都配有针对性的优化内核。

   - **Prefill Path**：新序列使用 `flash_attn_varlen_func`，借助 CDNA 矩阵核心处理计算密集型工作。
   - **Extend Path**：续接序列使用分块注意力加 LSE 合并，高效支持 100K+ 长上下文。
   - **Decode Path**：单 token 生成使用 AITER 的高度优化内核，以最大化内存带宽利用率。

    <figure style="text-align: center;">
    <iframe src="/assets/figures/2026-02-27-rocm-attention-backend/iteration2_attention_backend_routing.html" width="600" height="420" style="border: 1px solid #dee2e6; border-radius: 8px;" frameborder="0"></iframe>
    <figcaption>动画演示：R1（decode token）被路由到 Decode 路径，R2（prefill token）被路由到 Prefill 路径。</figcaption>
    </figure>

2. **Batch 重排序（Model Runner）**：`ROCM_AITER_FA` 是少数会在真正处理前先重排请求顺序的后端之一。vLLM 的 Model Runner 会把请求重排为 `[decode:extend:prefill]`，以实现更连续的内存访问。每个注意力后端都可以通过设置 `reorder_batch_threshold` 来选择是否启用这个机制，而 `ROCM_AITER_FA` 将其设为 1，也就是每个混合 batch 都会在三路路由前完成重排。

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/batch_reordering.png"
    style="width: 80%; display: block; margin: 0 auto;"
    alt="Diagram showing batch reordering optimization" />
  <figcaption>Batch 重排序确保每条内核路径处理的 token 在内存中连续，从而消除冗余的 KV cache 读取。</figcaption>
</figure>

<figure style="text-align: center;">
<iframe src="/assets/figures/2026-02-27-rocm-attention-backend/iteration4_batch_reordering_extend.html" width="600" height="670" style="border: 1px solid #dee2e6; border-radius: 8px;" frameborder="0"></iframe>
<figcaption>动画演示：Batch 重排序先把请求排列成 [decode &gt; extend &gt; prefill]，然后将 R3 路由到 Extend 路径。</figcaption>
</figure>

3. **分块上下文处理**：长序列会按固定的单次迭代 token 预算进行分块处理，总量约为 32K token，并在多个 extend 请求之间切分；随后用基于 LSE 的合并保证数值稳定性。

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/chunked_context_flow.png"
    style="width: 80%; display: block; margin: 0 auto;"
    alt="Diagram showing chunked context processing workflow" />
  <figcaption>100K+ token 的长上下文会被拆成约 32K 的分块处理，并通过基于 LSE 的合并保持数值稳定。</figcaption>
</figure>

4. **面向硬件优化的 KV Cache 布局**：它采用了 AMD AITER 内核团队设计的 preshuffled KV cache 布局。

```python
k_cache: [num_blocks, num_heads, head_dim // x, block_size, x]
v_cache: [num_blocks, num_heads, block_size // x, head_dim, x]
```

这种布局让内存访问模式与 AMD CDNA 架构更匹配，使 decode 路径能够以**零布局转换开销**直接调用 AITER 的 `pa_fwd_asm` 内核，相比标准 KV cache 布局可带来 **15% 到 20% 的 decode 吞吐提升**。

### 为什么要做显式三路路由

`ROCM_AITER_FA` 做了一个很明确的架构选择：在软件层显式地对工作负载进行路由，而不是依赖一个统一内核处理所有情况。

这种显式方案带来了几项直接收益：

- **更容易调试**：每条路径都可以独立做 profiling、调优和优化。
- **更强的可移植性**：同一套路由逻辑可以直接跨 MI300X、MI325X、MI355X 复用，而不需要为每代硬件重写控制逻辑。
- **更好的可扩展性**：可以直接添加新的工作负载类型或新的内核变体，而不必重构核心架构。
- **更可预测**：执行路径是确定性的，性能分析更直接。

其中 extend 路径尤其重要。prefix caching 和多轮对话已经成为生产部署中的常见需求，给它们一条专用路径并加入分块上下文注意力，意味着这些工作负载能获得一等公民级别的优化。

### 三条路径的处理细节

**Prefill Path**：Query、Key、Value 都采用标准的 `[num_tokens, num_heads, head_dim]` 布局，这样可以与高度优化的 AITER MHA 内核直接对齐，避免额外内存拷贝。

**Extend Path**：这是最复杂的一条路径。新 token 需要与保存在 shuffled KV cache 布局里的上下文 token 做注意力计算。由于这种 shuffled 布局在长上下文场景下并不兼容 AITER 的 MHA 内核，因此这里会插入额外的 KV cache 提取算子 `cp_mha_gather_cache`，把上下文的 Key / Value 提取并转换为标准布局。为了控制内存占用，长上下文还会继续被切成多个分块：

```python
def extend_forward():
    # 阶段 1：处理新 token 的注意力
    flash_attn_varlen_func()  # 调用 AITER MHA

    # 阶段 2：循环处理上下文分块
    for chunk in context_chunks:
        cp_mha_gather_cache()      # Triton gather 内核
        flash_attn_varlen_func()   # 调用 AITER MHA
        merge_attn_states()        # 基于 LSE 的合并

    # 阶段 3：得到最终结果
    merge_attn_states()
```

每个分块都会产生一个输出和一个 LSE（log-sum-exp）。LSE 记录了 softmax 的分母，因此可以实现数值稳定的合并。注意力得分更高的分块，会自然在最终结果中占主导地位。

**Decode Path**：这条路径会直接利用 shuffled KV cache 布局。自定义的 `reshape_and_cache_flush` 算子保证 cache 始终以 shuffled 布局存储，从而让注意力后端可以零转换开销地调用 AITER 高性能的 `pa_fwd_asm` 内核。

### 交互动画：ROCM_AITER_FA 的请求流

下面这个动画展示了多个请求如何在 7 次迭代中流经 ROCM_AITER_FA 后端。你可以使用控件开始、暂停，或者跳转到某个具体迭代步骤。

<div style="width: 100vw; position: relative; left: 50%; right: 50%; margin-left: -50vw; margin-right: -50vw; overflow-x: auto; padding: 20px 0;">
<div style="display: flex; justify-content: center; min-width: 1420px;">
<iframe src="/assets/figures/2026-02-27-rocm-attention-backend/rocm_aiter_fa_flow_animated_new.html" width="1400" height="850" style="border: 1px solid #dee2e6; border-radius: 12px;" frameborder="0"></iframe>
</div>
</div>

**迭代说明：**

| 迭代 | 关键事件 |
| ---- | -------- |
| **1** | R1 进入，随后经过分词、调度队列、QKV projection，进入 **Prefill Path**，再采样 1 个 token。R2 在本轮中途到达，进入队列等待。 |
| **2** | R1 和 R2 一起被批处理。R1 进入 **Decode Path**，R2 进入 **Prefill Path**。R3 和 R4 到达并进入队列。 |
| **3** | 4 个请求被一起批处理。token 预算为 100，因此 R3 本轮调度 100 个 token，剩余 180。R3 当前输出为 0，因为 prompt token 还没全部算完。 |
| **4** | R3 转入 **Extend Path**，继续处理剩余 prompt token。此时 batch 会被重排为 [decode &gt; extend &gt; prefill]。 |
| **5** | Batch 重排序继续，顺序变为 [decode &gt; extend]。R5 完成 extend 后转入 decode。 |
| **6-7** | 所有请求都进入 **Decode Path**，持续生成 token，直到收到停止信号。 |

_这个动画展示了 ROCM_AITER_FA 如何根据请求状态，在 Prefill、Extend、Decode 三条路径之间动态切换，从而高效地对混合工作负载进行批处理。_

---

## AITER MLA 后端：为 DeepSeek 优化

DeepSeek 和 Kimi 的 MLA 架构会把 KV cache 压缩到 **576 维**，而标准 MHA 大约是 8K 维，等于把内存需求压缩了约 14 倍。这种压缩会直接改变注意力计算的性能特征，因此它需要和标准 MHA 不同的优化策略。

### 混合方案

vLLM 提供了两个基于 AITER 的 MLA 后端，它们的差异主要在 prefill 实现上：

| 后端 | Prefill 内核 | Decode 内核 |
| ---- | ------------ | ----------- |
| `TRITON_MLA` | vLLM Triton | vLLM Triton |
| `ROCM_AITER_MLA` | AITER MHA | AITER 汇编 |
| `ROCM_AITER_TRITON_MLA` | AITER Triton MHA | AITER 汇编 |

基础的 `TRITON_MLA` 后端在两个阶段都使用 vLLM 默认的 Triton 内核。AITER 后端则把 decode 内核替换为手工调优的汇编实现 `mla_decode_fwd`，这也是大部分性能提升的来源。两个 AITER 后端唯一的区别在于 prefill 路径：`ROCM_AITER_MLA` 调用 `aiter.flash_attn_varlen_func`，由 AITER MHA 自动分发到 CK 或汇编内核；`ROCM_AITER_TRITON_MLA` 则调用 `aiter.ops.triton.mha.flash_attn_varlen_func`，使用 AITER Triton MHA。

### Absorbed 与 Non-Absorbed 方案

所有 MLA 后端都采用相同的基本处理策略：

- **Prefill / Extend（Non-Absorbed）**：在未压缩表示上，使用标准 MHA 内核来计算注意力。
- **Decode（Absorbed）**：使用专门的 MLA 内核，直接在压缩后的 576 维 latent 空间上工作。

```python
def _forward_prefill():
    # 阶段 1：处理新 token 的注意力（non-absorbed）
    _run_prefill_new_tokens()

    # 阶段 2：对 extend 路径循环处理上下文分块
    for chunk in context_chunks:
        gather_and_maybe_dequant_cache()
        _run_prefill_context_chunk()
        merge_attn_states()

    # 阶段 3：最终合并
    merge_attn_states()
```

在 **decode** 阶段，模型会逐 token 生成输出。虽然压缩后的 KV cache 让单次需要加载的数据更少，但整体瓶颈依然是内存带宽，因此 decode 依然是**内存密集型**任务。AITER 的汇编内核 `mla_decode_fwd` 会尽可能榨干 HBM3 带宽，因此能显著胜过通用的 Triton decode 内核。

### 为什么汇编版 Decode 内核这么关键

两个 AITER MLA 后端（`ROCM_AITER_MLA` 和 `ROCM_AITER_TRITON_MLA`）共享**同一个汇编 decode 内核** `mla_decode_fwd`。绝大多数性能提升都来自这里：

| 阶段 | AITER MLA 后端 | vLLM TRITON_MLA 基线 |
| ---- | -------------- | -------------------- |
| **Prefill** | AITER MHA 或 Triton（取决于后端） | Triton flash attention |
| **Decode** | 汇编 `mla_decode_fwd` | Triton `decode_attention_fwd` |

整体 **1.2 到 1.6 倍加速**，主要来自这个共享的汇编 decode 内核。因为 TPOT 在 OSL=1K 时会执行 1K 次 decode 迭代，所以 decode 是绝对主导项。相比之下，两个 AITER 后端之间的 prefill 内核差异，对总性能的影响非常小。

除了纯粹的内核性能，这两个后端还继承了 FlashMLABackend 的完整能力，包括 FULL_AND_PIECEWISE CUDA graph 支持和 MTP 支持。另一个优势是它们几乎能在任意 KV cache block size 下保持一致性能，也就是你可以把每个 token 都当成 prefix cache，而不用担心细粒度缓存通常会带来的性能惩罚。

---

## 性能基准测试

**测试方法**：所有基准测试均使用 `rocm/vllm-dev:nightly_main_20260115` 运行，ROCm 版本为 7.0.0。这是 2026 年 1 月 15 日基于 [vLLM main 分支](https://github.com/vllm-project/vllm) 构建的 nightly Docker 镜像。我们先用初始请求做预热，下面汇报的结果都排除了第一次运行，以消除 JIT 编译开销。

<details markdown="1">
<summary><strong>基准测试服务端命令（点击展开）</strong></summary>

**MHA 基准测试（Qwen3-235B）：**

```bash
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1

# 选择后端：TRITON_ATTN, ROCM_ATTN, ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN
ATTN_BACKEND="ROCM_AITER_FA"

model_path=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
vllm serve $model_path \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --attention-backend ${ATTN_BACKEND} \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --async-scheduling \
    --port 1234
```

**MLA 基准测试（DeepSeek-R1）：**

```bash
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

# 选择后端：TRITON_MLA, ROCM_AITER_MLA, ROCM_AITER_TRITON_MLA
ATTN_BACKEND="ROCM_AITER_MLA"

model_path=deepseek-ai/DeepSeek-R1-0528
vllm serve $model_path \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --attention-backend ${ATTN_BACKEND} \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --async-scheduling \
    --port 1234
```

</details>

### MHA 基准结果

**模型**：[Qwen3-235B-A22B-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8)，Attention 使用 TP8，MoE 使用 EP8  
**工作负载**：ISL=10K，OSL=1K，并发请求数为 64 和 128

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/mha_tpot_comparison.png"
    style="width: 100%; display: block; margin: 0 auto;"
    alt="MHA TPOT Comparison" />
  <figcaption>与遗留的 ROCM_ATTN 相比，ROCM_AITER_FA 在 MI300X、MI325X、MI355X 上带来 2.8 到 4.6 倍更快的 TPOT。</figcaption>
</figure>

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/mha_ttft_comparison.png"
    style="width: 100%; display: block; margin: 0 auto;"
    alt="MHA TTFT Comparison" />
  <figcaption>TTFT（Time To First Token）对比显示，在 64 和 128 并发下，ROCM_AITER_FA 和 ROCM_AITER_UNIFIED 的 prefill 性能领先。</figcaption>
</figure>

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/mha_tps_comparison.png"
    style="width: 100%; display: block; margin: 0 auto;"
    alt="MHA TPS Comparison" />
  <figcaption>输出吞吐量（TPS）与 TPOT 结果一致，ROCM_AITER_FA 相比遗留 ROCM_ATTN 实现了 2.7 到 4.4 倍更高吞吐。</figcaption>
</figure>

**以 ROCM_AITER_FA 为基准的 TPS 相对倍数（64 并发请求）：**

| 硬件 | ROCM_AITER_FA | ROCM_AITER_UNIFIED_ATTN | TRITON_ATTN | ROCM_ATTN |
| ---- | ------------- | ----------------------- | ----------- | --------- |
| MI300X | **1.00x** | 1.05x | 1.30x | 3.82x |
| MI325X | **1.00x** | 1.02x | 1.19x | 4.36x |
| MI355X | **1.00x** | 0.95x | 1.08x | 3.61x |

**以 ROCM_AITER_FA 为基准的 TPS 相对倍数（128 并发请求）：**

| 硬件 | ROCM_AITER_FA | ROCM_AITER_UNIFIED_ATTN | TRITON_ATTN | ROCM_ATTN |
| ---- | ------------- | ----------------------- | ----------- | --------- |
| MI300X | **1.00x** | 1.05x | 1.36x | 2.65x |
| MI325X | **1.00x** | 1.00x | 1.28x | 3.12x |
| MI355X | **1.00x** | 1.01x | 1.23x | 2.88x |

不同 GPU 代际上的相对性能趋势基本一致。在这种请求尺寸均匀的工作负载下，`ROCM_AITER_UNIFIED_ATTN`（单内核路径）与 `ROCM_AITER_FA`（三路路由）之间差距在 5% 以内。若换成包含 prefix cache 命中的混合负载，三路路由的优势会更明显。

_注：这里 `ROCM_ATTN` 慢 2.7 到 4.4 倍，是因为 Qwen3-235B 的 KV head size 不受 HIP paged attention 支持，被迫退回 Triton decode 内核。对于 head size 受支持的模型，`ROCM_ATTN` 其实会快于 `TRITON_ATTN`。_

### MLA 基准结果

**模型**：[DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)，TP8，block_size=16  
**工作负载**：ISL=10K，OSL=1K，并发请求数为 64 和 128

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/mla_tpot_comparison.png"
    style="width: 100%; display: block; margin: 0 auto;"
    alt="MLA TPOT Comparison" />
  <figcaption>由于共享了汇编版 decode 内核，AITER MLA 后端相比 TRITON_MLA，在 MI300X、MI325X、MI355X 上带来 1.2 到 1.6 倍更快的 TPOT。</figcaption>
</figure>

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/mla_ttft_comparison.png"
    style="width: 100%; display: block; margin: 0 auto;"
    alt="MLA TTFT Comparison" />
  <figcaption>TTFT 对比显示，在 MI355X 的 128 并发场景下，ROCM_AITER_MLA 获得了最优 TTFT。</figcaption>
</figure>

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/mla_tps_comparison.png"
    style="width: 100%; display: block; margin: 0 auto;"
    alt="MLA TPS Comparison" />
  <figcaption>输出吞吐量（TPS）显示，AITER MLA 后端相比 TRITON_MLA 最高可实现 1.5 倍更高吞吐。</figcaption>
</figure>

**以 ROCM_AITER_MLA 为基准的 TPS 相对倍数（64 并发请求）：**

| 硬件 | ROCM_AITER_MLA | ROCM_AITER_TRITON_MLA | TRITON_MLA |
| ---- | -------------- | --------------------- | ---------- |
| MI300X | **1.00x** | 0.98x | 1.33x |
| MI325X | **1.00x** | 0.98x | 1.41x |
| MI355X | **1.00x** | 1.03x | 1.52x |

**以 ROCM_AITER_MLA 为基准的 TPS 相对倍数（128 并发请求）：**

| 硬件 | ROCM_AITER_MLA | ROCM_AITER_TRITON_MLA | TRITON_MLA |
| ---- | -------------- | --------------------- | ---------- |
| MI300X | **1.00x** | 0.97x | 1.24x |
| MI325X | **1.00x** | 0.97x | 1.24x |
| MI355X | **1.00x** | 1.01x | 1.35x |

两个 AITER MLA 后端的整体性能非常接近。在 gfx942（MI300X、MI325X）上，`ROCM_AITER_TRITON_MLA` 的 TPS 高 2% 到 3%。在 gfx950（MI355X）上，`ROCM_AITER_MLA` 持平甚至略优，因为它使用 AITER 的汇编 MHA prefill。`ROCM_AITER_MLA` 同时也拿到了 MI355X 上最好的 TTFT，因此更推荐使用自动选择的 `ROCM_AITER_MLA`。

_注：这些基准测试使用的都是尺寸均匀的请求。生产环境中的 prefix caching、混合上下文长度和更复杂的请求分布，会更充分地体现三路路由架构的价值。_

---

## 协作方式：vLLM + AITER

这些性能提升并不是某一个单点优化带来的，而是 vLLM 的调度编排层与 AMD AITER 原语共同作用的结果。理解这层协作，也就能明白为什么“仅仅移植”远远不够。

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/system_stack.png"
    style="width: 80%; display: block; margin: 0 auto;"
    alt="System architecture stack diagram" />
  <figcaption>完整系统栈：从用户请求，经由 vLLM 编排层，最终落到 AMD 硬件上的 AITER 原语。</figcaption>
</figure>

### 创新归因

这些性能到底来自哪里？答案是上下两层一起作用：

<figure style="text-align: center;">
  <img
    src="/assets/figures/2026-02-27-rocm-attention-backend/innovation_attribution.png"
    style="width: 100%; display: block; margin: 0 auto;"
    alt="Innovation Attribution" />
  <figcaption>vLLM 编排层负责路由与分块，AITER 提供面向硬件优化的原语。</figcaption>
</figure>

**核心洞察**在于：AITER 提供的是为 CDNA 架构量身打造的高性能注意力原语，而 vLLM 的调度编排层则进一步提供了工作负载感知路由和分块处理，把性能推到了最后一个层级。两者缺一不可。

---

## 快速开始

### 推荐用法

```bash
# 推荐：让 vLLM 自动选择优化后的后端
export VLLM_ROCM_USE_AITER=1
vllm serve <your-model> --tensor-parallel-size <tp>
```

设置 `VLLM_ROCM_USE_AITER=1` 后，vLLM 会自动选择：

- MHA 模型（Llama、Qwen、Mistral）使用 `ROCM_AITER_FA`
- MLA 模型（DeepSeek、Kimi）使用 `ROCM_AITER_MLA`

### 手动指定后端

如果你希望手动实验，也可以通过 `--attention-backend` 明确指定：

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528 \
    --tensor-parallel-size 8 \
    --attention-backend ROCM_AITER_TRITON_MLA
```

基准结果表明，这两个 AITER MLA 后端因为共享同一个汇编 decode 内核，所以整体性能差异很小。prefill 内核会随着架构略有不同，但由于 decode 才是主导项，因此总差距很有限。对大多数用户来说，自动选择的 `ROCM_AITER_MLA` 就足够合适。

### 硬件支持

| GPU | 显存 | 架构 |
| --- | ---- | ---- |
| MI300X | 192GB HBM3 | gfx942 |
| MI325X | 256GB HBM3e | gfx942 |
| MI355X | 288GB HBM3e | gfx950 |

### 完整后端参考

vLLM 在 AMD ROCm 上一共提供 7 个注意力后端，分别面向不同场景：

| 分类 | 后端 | 启用方式 | 说明 |
| :--- | :--- | :------- | :--- |
| MHA | TRITON_ATTN | `--attention-backend TRITON_ATTN` | 基线方案，支持 Radeon |
| MHA | ROCM_AITER_UNIFIED_ATTN | `--attention-backend ROCM_AITER_UNIFIED_ATTN` | AITER 统一内核 |
| MHA | ROCM_ATTN | `--attention-backend ROCM_ATTN` | 遗留双路后端，支持 Radeon |
| MHA | **ROCM_AITER_FA** | `--attention-backend ROCM_AITER_FA` + `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` | **推荐**，启用 AITER 时会自动选择 |
| MLA | TRITON_MLA | `--attention-backend TRITON_MLA` | 基线方案，支持 Radeon |
| MLA | **ROCM_AITER_MLA** | `--attention-backend ROCM_AITER_MLA` | **推荐**，启用 AITER 时会自动选择 |
| MLA | ROCM_AITER_TRITON_MLA | `--attention-backend ROCM_AITER_TRITON_MLA` | 备选的 AITER MLA 后端 |

---

## 结论

“仅仅移植”的时代已经结束了。本文系统梳理了 vLLM 在 AMD ROCm 上提供的全部 7 个注意力后端，并给出了覆盖不同后端的透明基准测试和取舍分析。

**核心结论（ISL=10K，OSL=1K 测试集）：**

- `ROCM_AITER_FA`：在 MHA 模型上，相比 `ROCM_ATTN` 实现 **2.7 到 4.4 倍更高 TPS**
- `ROCM_AITER_MLA`：在 DeepSeek MLA 上，通过汇编版 decode 内核，相比 `TRITON_MLA` 实现 **1.2 到 1.5 倍更高 TPS**
- 性能趋势在 MI300X、MI325X、MI355X 之间保持一致

**推荐做法**是直接设置 `export VLLM_ROCM_USE_AITER=1`，让 vLLM 自动选择最优后端。默认组合，也就是 MHA 使用 `ROCM_AITER_FA`、MLA 使用 `ROCM_AITER_MLA`，在我们测试过的所有工作负载中都表现很好。

这就是原生 AMD 优化应有的样子：不是“移植过来”，而是“专门构建”。三路路由架构体现了一种明确的设计思路，即在软件层显式分离工作负载，再让每条路径调用针对硬件优化过的 AITER 原语。最终得到的是一个更容易调试、能跨 GPU 代际迁移，并且真正适合生产 LLM 混合工作负载的系统。

---

## 致谢

感谢所有参与这次协作的工程师。

**AMD**：Hattie Wu、Yi Gan、Zejun Chen、Carlus Huang、Lingpeng Jin、Peng Sun，以及 AITER 团队。

**Embedded LLM**：Pin Siang Tan、Tun Jian Tan、Jun Kang Chow，以及 Embedded LLM 团队。

## 相关资源

- [AITER Library (AMD)](https://github.com/ROCm/aiter)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3-235B Model](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8)
- [DeepSeek-R1 Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)

---

## 声明

本测试由 AMD AI Framework 团队于 2026 年 1 月 29 日完成，测量的是 AMD Instinct MI300X、MI325X、MI355X 平台上的 TPS 推理性能。

**硬件配置**

- MI300X：AMD EPYC 9654 96 核处理器服务器，配备 8 张 AMD Instinct MI300X（192GB，750W）GPU，Supermicro AS-8125GS-TNMR2，NPS1（每个 socket 1 个 NUMA），2.2TiB 内存（24 条 DIMM，4800 mts，96 GiB/DIMM），BIOS 版本 3.2
- MI325X：AMD EPYC 9575F 64 核处理器服务器，配备 8 张 AMD Instinct MI325X（256GB，1000W）GPU，Supermicro AS-8125GS-TNMR2，NPS1（每个 socket 1 个 NUMA），2.2TiB 内存（24 条 DIMM，4800 mts，96 GiB/DIMM），BIOS 版本 3.2
- MI355X：AMD EPYC 9575F 64 核处理器服务器，配备 8 张 AMD Instinct MI355X（288GB，1400W）GPU，Supermicro AS-8125GS-TNMR2，NPS1（每个 socket 1 个 NUMA），2.2TiB 内存（24 条 DIMM，4800 mts，96 GiB/DIMM），BIOS 版本 3.2

**软件配置**

Ubuntu 22.04LTS，Linux 内核 5.15.0-116-generic，ROCm 7.0 软件栈，PyTorch 2.9.0a0，vLLM 0.14.0rc2（2026 年 1 月 15 日版本）。

不同服务器厂商的配置可能存在差异，因此结果也可能不同。实际性能还会受到具体配置、软件版本、vLLM 版本，以及是否使用最新驱动和优化的影响。

---
