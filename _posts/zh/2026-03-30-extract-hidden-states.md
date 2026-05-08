---
layout: post
title: "vLLM 原生 Hidden States 提取：投机解码训练数据不再绕路"
author: "Fynn Schmitt-Ulms"
image: /assets/figures/2026-03-30-extract-hidden-states/design_diagram.png
lang: zh
translation_key: extract-hidden-states
tags:
  - speculative-decoding
---

PR [#33736](https://github.com/vllm-project/vllm/pull/33736)（已包含在 `vllm>=v0.18.0` 中）为 vLLM 引入了原生的 hidden states 提取系统。它为投机解码模型的训练提供了一条更高效、更灵活的数据生成路径。本文介绍这一特性的动机、设计思路、使用方法与后续规划，以及它在 vLLM 的 [Speculators](https://github.com/vllm-project/speculators/) 库中的应用。Speculators 是一个用于创建和训练投机解码模型的工具库。

## 动机

Hidden states 是模型对 token 序列的内部中间表示。它们能揭示模型内部状态，并在投机解码中被大量使用。

### 投机解码回顾

投机解码通常将一个“验证器”（verifier）模型，也就是你真正要服务的大型 LLM，与一个较小的“草稿”（draft）模型结合使用。Draft 模型先生成草稿 token，随后 verifier 模型并行验证这些 token。这样可以显著加速解码过程，根据方法不同，常见加速比可达 2-5 倍，尤其在较低 batch size、模型性能受显存带宽限制的场景中效果明显。

研究者发现，如果把 verifier 模型内部的 hidden states 提供给 draft 模型，可以提升 draft 的对齐度和整体质量。因此，[Eagle-3](https://arxiv.org/abs/2503.01840)、[P-Eagle](https://arxiv.org/abs/2602.01469)、[DFlash](https://arxiv.org/abs/2602.06036) 等方法都被设计出来，它们都需要 verifier 多个层的 hidden states 作为输入。

由于 draft 模型以 hidden states 为输入，训练它们就需要访问大规模的 hidden states 与 verifier 输出数据集。大多数投机解码库（例如 Speculators）通常通过以下两种方式来解决这个问题：

1. 使用 `transformers` 生成 hidden states。这种方法能用，但有两个主要缺点：
   A. vLLM 的性能优化能力，例如大模型支持、分布式支持等，都用不上。
   B. 它会引入一整类潜在 bug，这些 bug 来自 `transformers` 与 vLLM 生成的 hidden states 之间的细微差异。
2. 对 vLLM 做大量修改和 patch。这通常需要手工组装 vLLM 核心组件并直接调用内部 API。随着 vLLM 内部实现演进，维护成本会非常高。同时，这也意味着许多 vLLM 特性，例如 prefix caching、自动 batching、异步 server 等，都需要被禁用。这就是过去版本的 Speculators（`<0.5.0`）处理 hidden states 生成的方式。

这两种方案各有明显不足。随着投机解码越来越普及，需要一个更高效、性能更好的解决方案。

## 设计考量

将 hidden states 提取直接集成到 vLLM 中时，需要同时满足多项要求。

首先，系统必须能以高性能的方式返回 hidden states。模型的 hidden states 体积可能非常大。以 `Qwen3-8B` 为例，其 `hidden_size` 为 4096，提取出的 hidden states 形状是 `[seq_len, num_layers_to_extract, 4096]`。如果序列长度是 8k tokens、提取 4 层、使用 FP16，那么数据量会达到 268 MB。因此，把 hidden states 序列化后直接放在请求响应体中返回，是不现实的。

而且 hidden states 占用空间如此之大，即便只是临时保存在显存中，也不是一件简单的事。系统必须为所有并发请求统一预分配和管理内存，同时处理 chunked prefill、请求抢占等情况，以避免 OOM。

由于这一功能只在用户确实需要从 vLLM 中提取 hidden states 时才会用到，而绝大多数部署场景并不需要它，因此它不能给 vLLM 的“热路径”带来任何新的开销，无论是运行时开销还是认知复杂度。实践中，这意味着必须限制改动范围，并尽可能复用已有能力。

最后，最终用户对 hidden states 的使用、存储和传输方式可能完全不同。例如，在“离线”投机模型训练中，hidden states 会先针对整份数据集生成并落盘缓存，再开始训练；而“在线”训练则是在训练过程中实时生成 hidden states，并要求它们高效地传输给训练进程，最好无需先写盘。为了支持这些不同场景，hidden states 提取系统必须足够灵活且可扩展。

## 设计洞察

在上述需求基础上，几个关键设计洞察共同促成了 hidden states 提取系统的实现：

1. vLLM 已经支持使用 Eagle-3 及类似方法的投机解码模型运行推理，而这些模型会把 verifier 模型的 hidden states 作为输入。因此，vLLM 内部已经存在一条把 hidden states 从 verifier 模型传给 draft 模型的通路。
2. vLLM 已经有一个可扩展的 [KV Connector API](https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/)，用于高效提取 KV cache 数据，并支持 Prefill/Decode Disaggregation 等特性。这个 API 的现有实现已经支持通过 Nixl 传输 KV cache、写入磁盘、存入共享内存等方式，同时也支持异步传输，并保证在传输完成前不会释放相关 KV cache block。
3. Hidden states 与其输入 token 序列之间的映射关系，与 KV cache 数据的映射方式是相同的。换句话说，每个 token 都对应一个 hidden state 值，且这个值只在其前缀序列上下文中有效。
4. vLLM 支持为投机 draft 模型单独配置 KV cache 大小。

把这些思路组合起来（见图 1），我们就可以这样提取 hidden states：

1. 创建一个 dummy draft 模型，利用 vLLM 中现有的 Eagle-3 数据通路来接收 verifier 的 hidden states。
2. 这个 dummy 模型有一个 dummy attention 层和它自己的 KV cache。它不会真的执行 attention，而是直接把输入的 hidden states 写入自己的 KV cache。
3. 随后，一个自定义 KV Connector 就可以把这个 dummy draft 模型 KV cache 中的数据保存到磁盘，或者以其他方式传输走。此时，这份 KV cache 实际上存储的就是我们要的 hidden states。

这样一来，所有设计要求都得到了满足：通过已有 Eagle-3 通路把 hidden states 送入 draft 模型；通过 KV Connector API 提供高性能、可扩展的提取方式；并且可以通过不同 connector 适配不同的下游使用场景。

由于 draft 模型把 hidden states 存在 dummy attention 层中，vLLM 知道应该为这些数据分配显存。同时，vLLM 还会像管理 KV cache 一样，用同样的分页内存系统来管理 hidden states，因此也自然继承了 prefix caching、chunked prefill、高效 batching 等能力。

<p align="center">
<img src="/assets/figures/2026-03-30-extract-hidden-states/design_diagram.png" width="100%">
<br>
<em>图 1：Hidden states 提取系统的设计示意图。</em>
</p>

## 使用方法与限制

[examples/offline_inference/extract_hidden_states.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/extract_hidden_states.py) 展示了如何通过 Python API 提取 hidden states。该系统同样可以与 vLLM server 配合使用，启动命令如下：

```bash
vllm serve Qwen/Qwen3-8B --speculative_config '{
    "method": "extract_hidden_states",
    "num_speculative_tokens": 1,
    "draft_model_config": {
        "hf_config": {
            "eagle_aux_hidden_state_layer_ids": [3, 18, 33, 36]
        }
    }
}' --kv_transfer_config '{
    "kv_connector": "ExampleHiddenStatesConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "shared_storage_path": "/tmp/hidden_states"
    }
}'
```

这个命令会配置系统的两个主要组件。

`--speculative_config` 会告诉 vLLM 使用一个假的 `"extract_hidden_states"` 投机方法，以此完成 dummy draft 模型的搭建；同时它也支持指定要提取哪些层的 hidden states。

第二个组件 `--kv_transfer_config` 则负责配置自定义 KV Connector。这个 connector 专门用于从 draft 模型层中提取 hidden states。写作本文时，系统里只有 `"ExampleHiddenStatesConnector"` 这一种实现，它是一个简单的写盘版本，但更高性能的 connector 很快就会加入。需要注意，这两个组件必须同时使用，整个系统才能按预期工作。

当 vLLM 启动后，任何发往 server 的请求都会返回一个 `"kv_transfer_params"` 字典，其中包含 `"hidden_states_path"`。这个路径指向一个保存了 hidden states 和 token ids 的 safetensors 文件。保存目录可以通过上面配置中的 `"shared_storage_path"` 字段指定。

```python
# `/tmp/hidden_states/{req_id}.safetensors`
{
    "token_ids": [prompt_seq_len],
    "hidden_states": [prompt_seq_len, num_hidden_layers, hidden_size]
}
```

注意事项：

- 支持 `--tensor-parallel-size` 和 `--data-parallel-size`，因此可用于单节点多 GPU 部署。
- 系统只会保存 prompt token 及其 hidden states。因此我们建议调用 `v1/completions` 接口时，把 `max_tokens` 设为 `1`。

## 后续工作

- **集成到 vLLM 的 [Speculators](https://github.com/vllm-project/speculators) 项目中。** Speculators 库旨在高效训练投机解码算法。最近合并的 [speculators PR #353](https://github.com/vllm-project/speculators/pull/353) 已经更新为使用 vLLM 原生的 hidden states 提取系统，并支持 draft 模型的在线训练。该功能将包含在 `speculators v0.5.0` 中。
- **性能优化。** 当前 hidden states 专用 KV Connector，也就是 `"ExampleHiddenStatesConnector"` 的初始实现仍未优化，其中包含阻塞式 hidden states 写入。相关团队正在积极推进异步写入支持。
- **设备间直传 connector。** ExampleHiddenStatesConnector 直接把 hidden states 写到磁盘，再由训练进程读取使用。这个方式足够简单，也很适合作为测试实现，但在更大规模训练负载下扩展性有限。后续会开发更高级的 hidden states connector，支持在设备之间直接传输 hidden states，包括多节点环境。

## 参考链接

- [Speculators](https://github.com/vllm-project/speculators/)
- [Eagle-3](https://arxiv.org/abs/2503.01840)
- [P-Eagle](https://arxiv.org/abs/2602.01469)
- [DFlash](https://arxiv.org/abs/2602.06036)
- [KV Connector API](https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/)
- [extract_hidden_states.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/extract_hidden_states.py)
- [speculators PR #353](https://github.com/vllm-project/speculators/pull/353)
- [英文原文：vllm.ai/blog/extract-hidden-states](https://vllm.ai/blog/extract-hidden-states)
