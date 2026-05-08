---
layout: post
title: "vLLM 流式输入与 Realtime API"
author: "Meta, Mistral AI as well as the vLLM team"
image: /assets/figures/2026-01-31-streaming-realtime/streaming-realtime.png
math: true
lang: zh
translation_key: streaming-realtime
tags:
  - multimodal
---

传统的大语言模型推理通常遵循一个简单前提：用户先提交完整 prompt，请求被模型完整处理后，再返回响应。这个范式非常适合文本聊天机器人和批处理型任务，但在处理实时应用时就不够用了，比如流式音频或视频。

vLLM 最近在引擎中新增了 **流式输入（streamable inputs）** 支持，并在此基础上构建了一个 **Realtime WebSocket API**，在服务端新增了 `/v1/realtime` 端点。

本文将解释为什么需要实时推理，并介绍 vLLM 中解锁这类能力的两个新特性：**流式输入支持** 与 **Realtime WebSocket API**。

_注意_：如果你想直接了解如何在 vLLM 中使用流式输入或 Realtime API，可以先参考以下资料：
- [流式输入](https://github.com/vllm-project/vllm/tree/main/tests/v1/streaming_input)
- [Realtime WebSocket API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/?h=realtime+api#realtime-api)

# 为什么需要实时推理

## vLLM 的传统批处理范式

传统 LLM 推理默认完整 prompt 在一开始就已经可用。用户通过完整请求，例如 [`ChatCompletionRequest`](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)，提交全部输入，等待模型完整处理后再接收响应。虽然 vLLM 早已支持**输出流式**，即 token 生成时逐步发送，但**输入侧**一直是固定的：必须先把整个请求提供完整，推理才能开始。

这种方式对于大多数应用都足够。文本聊天、文档摘要、代码生成都能自然适配。但越来越多的应用场景，无法等到完整输入全部到达后再开始处理。

## 流式输入为什么重要

可以想象一个通过语音控制电脑或手机的语音助手。用户不需要键盘、鼠标或触摸屏，所有操作都由语音完成。麦克风录到的音频会持续流式发送给 LLM，由 LLM 充当语音助手模型。

这时，LLM 需要持续处理音频流，并实时生成动作。

对于这种应用，延迟，或者更准确地说 [TTFT（Time-To-First-Token）](https://www.emergentmind.com/topics/time-to-first-token-ttft)，至关重要。用户不会接受为了打开一个应用或者在搜索框里输入几个字而等上一两秒。一个更自然、更接近人类对话方式的语音助手，应该能够一边“听”，一边“说”，也就是 LLM 需要同时处理音频流和生成动作。

一个自然的问题是：能不能用非流式 LLM 通过“切块处理”近似实现流式行为？理论上可以。音频可以先被缓冲成若干片段，每个片段独立处理，再把结果拼接起来。但实际操作中，这会带来不少限制。要做到亚秒级 TTFT，就需要非常准确的 chunk detection，也就是精确判断何时切分音频流，同时又不能丢掉关键信息。切分不好，要么会导致 TTFT 变高，要么因为破坏了原本连续的时间上下文而降低模型效果。按块处理还无法实现真正的双向交互，因为每个块都必须先完整处理，之后才能生成响应，听和说不能并行，只能形成“轮流说话”的交互模式，而不是像人类那样连续、重叠的交流。

这个问题在很多领域里都会出现：

- **语音助手** 需要亚秒级响应，才能让交互显得自然
- **实时转录服务** 需要在语音被识别的同时显示文字
- **机器人和具身 AI** 需要处理持续不断的传感器流，例如摄像头、麦克风、激光雷达，并以极低延迟生成控制动作，从而安全地与物理世界交互

对于这类应用来说，传统的批处理范式带来的延迟是不可接受的。我们需要的是一种能够**增量处理输入**、并在所有输入都到齐之前就开始生成输出的基础设施。

_注意_：即便对于传统应用，也就是那种必须先读完整个输入才能生成第一个输出 token 的任务，只要输入可以边到边传，仍然可能受益。默认情况下，vLLM 会启用 [chunked prefill](https://docs.vllm.ai/en/stable/cli/serve/?h=max+num+b#-enable-chunked-prefill-no-enable-chunked-prefill)，因此长度为 $N$ 的输入会被拆成 $N \div M$ 次前向传播来处理，其中 $M$ 是 [`max_num_batched_tokens`](https://docs.vllm.ai/en/stable/cli/serve/?h=max+num+b#-max-num-batched-tokens)。当 $N \div M > 1$ 时，如果输入一可用就立刻流式送入，那么第一次 prefill 前向传播就可以更早被调度，从而降低整体 TTFT。

## 流式推理的前提条件

并不是所有模型都能支持真正的流式推理。要做到这一点，需要满足两个关键条件：正确的注意力模式，以及面向增量处理的训练方式。

### 注意力模式

注意力机制决定了模型能否增量处理输入，还是必须等完整序列到齐后再统一处理。

- 因果注意力（单向 mask）要求每个位置 $t$ 只能关注位置 $j \le t$ 的 token。由于未来 token 被排除在外，因此时刻 $t$ 的输出在 token $t$ 到达后就已经是最终结果。这意味着真正的流式成为可能：每个新 token 都可以立即处理，之前的输出也不需要回滚或修订。

- 双向注意力（完整 mask）允许每个位置同时关注过去和未来的 token。因此，位置 $t$ 的输出会依赖一些可能尚未到达的 token。只要完整输入序列还没有全部给定，模型就无法为任何位置计算稳定输出，因为未来 token 可能改变对早先 token 的解释。

因此，双向注意力天然要求先看到完整输入序列，才能输出结果，这让它不适合流式或在线处理。

而对于长时间运行甚至无限长的流式场景，标准因果注意力本身也还不够。如果每个 token 都要关注全部历史上下文，计算量和显存占用会无限增长，这在实践中不可行。因此通常需要截断历史上下文。

一种常见的架构做法是滑动窗口注意力（sliding-window attention），让每个 token 只关注最近的固定窗口内的 token。这样可以把计算和内存占用控制在有界范围内，同时继续支持流式处理。所以，带滑动窗口的因果注意力，通常是现代流式模型的首选架构。

### 面向流式输入的训练

不过，光有一个“理论上可流式”的架构还不够，模型还必须经过专门训练，才能真正支持**流式输入**。

设 $X = (x_0, x_1, \ldots, x_T)$ 为输入序列，$Y = (y_0, y_1, \ldots, y_{T'})$ 为输出序列。在流式应用中，模型应当在时间步 $t$ 以尽可能小的延迟生成输入 $x_t$ 对应的输出 $y_t$。一个直观例子是：$y_t$ 可以理解为在时间 $t$ 流入模型的音频帧 $x_t$ 的转录结果。

标准的 next-token 训练目标，通常会让下一个 token 的分布依赖于**整个输入序列**：

$$
P(y_i \mid y_{i-1}, \ldots, y_0, x_T, x_{T-1}, \ldots, x_0).
$$

这种形式不适合流式场景，因为生成 $y_i$ 需要完整输入序列 $X$，而实时场景中完整输入往往还没到齐。

流式模型则必须能够只依赖历史输入，以及可选的一小段未来上下文，来预测 $y_i$：

$$
P(y_i \mid y_{i-1}, \ldots, y_0, x_{i+\delta}, \ldots, x_i, \ldots, x_0),
$$

其中 $\delta$ 是 lookahead 参数，应尽可能小。理论上它可以是零；但在实践中，通常需要一个很小的延迟，才能获得可接受的效果。

因此，训练流式模型需要满足两点：
- **i)** 对齐输入与输出序列，使得 $T' = T$，并确保每个 $y_i$ 都是输入 $x_i$ 对应的正确输出
- **ii)** 使用一种架构，使得模型在处理新输入 $x_{i+1}$ 时，不需要重新完整处理旧输入 $x_i, \ldots, x_0$

一种比较直观的架构，由 [Delayed Streams Modeling](https://arxiv.org/pdf/2509.08753) 首先提出，并被 [Voxtral-Realtime](https://mistral.ai/news/voxtral) 采用。它会把输入嵌入，例如语音嵌入，以及输出嵌入，例如文本嵌入，做 sum-pooling，合并成单一嵌入序列。然后模型预测：

$$
P(y_i \mid y'_{i-1}, \ldots, y'_0),
$$

其中

$$
y'_k = y_k + x_{k+\delta}.
$$

这一区别对部署非常关键：你不能简单拿一个任意的因果模型，就期待它在流式场景里表现良好。想要做到完全可流式，模型必须使用上述输入输出对齐方式和架构约束做过显式训练，以确保条件 **i)** 和 **ii)** 都成立。

## 为什么模型架构会直接影响推理部署

vLLM 可以部署任意模型，但真正的流式能力要求模型在架构上具备因果性。像 [Voxtral](https://mistral.ai/news/voxtral) 这样的模型，从设计之初就是为了流式场景打造的，它使用支持增量处理的因果注意力机制。

同样重要的是，推理服务基础设施本身也必须支持增量输入。即便模型本身具备流式能力，只要服务端仍然要求必须先收到完整 prompt 才开始推理，延迟优势就会完全丢失。

这正是 vLLM 在已有输出流式能力之外，进一步支持输入流式的原因。

### 关于流式架构的延伸阅读

- [Transformer Transducer](https://arxiv.org/abs/2002.02562) 是训练可流式语音识别系统中最著名、也是最成功的方法之一
- [Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling](https://arxiv.org/abs/2509.08753) 由 Kyutai 团队撰写，对上面这类流式架构做了非常好的深入说明
- [Streaming Simultaneous Speech Translation with Augmented Memory Transformer](https://arxiv.org/abs/2011.00033) 讨论了流式语音翻译。翻译不像语音识别那样“单调”，所以高性能流式会更难做
- [Voxtral-Realtime](https://mistral.ai/news/voxtral) 是一个大规模预训练并开源的流式模型，在效果上可与大多数离线语音识别模型竞争

# vLLM 中的流式输入支持

通过 [PR #28973](https://github.com/vllm-project/vllm/pull/28973)，vLLM 现在已经支持流式输入推理。这正是上文所说的增量处理能力：输入随着时间持续到达，输出也随之持续生成。

## StreamingInput 接口

核心抽象是 `StreamingInput` 这个数据类：

```python
from dataclasses import dataclass
from vllm.inputs import PromptType
from vllm.sampling_params import SamplingParams

@dataclass
class StreamingInput:
    prompt: PromptType
    sampling_params: SamplingParams | None = None
```

现在你不再只能向 `AsyncLLM.generate()` 传一个固定 prompt，而是可以传入一个 `AsyncGenerator`，它会随着时间不断 yield 出 `StreamingInput` 对象。每个 `StreamingInput` 表示接下来要追加到累计 prompt 中的新输入块。示例如下：

```python
import asyncio
from vllm.inputs.data import StreamingInput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams

async def streaming_input_example():
    async_llm = AsyncLLM.from_engine_args(...)

    # 输入队列可以在单独的异步任务中消费新输入
    input_queue = asyncio.Queue[list[int]]()

    async def input_generator():
        # 遇到空列表时结束 => 输入流结束
        while new_tokens := input_queue.get():
            yield StreamingInput(prompt=new_tokens)

    output_generator = async_llm.generate(
        prompt=input_generator(),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
    )

    # 消费输出
    async for output in output_generator:
        # ...

asyncio.run(streaming_input_example())
```

你可以在上一个输入对应的输出完成之后，再发送下一个输入，但这不是必须的，输入块会在内部排队。输入流的终止通过退出这个异步生成器，或者显式调用 `aclose` 来表示。只有在**所有已收到输入都被处理完**，并且输入生成器自身也已经结束之后，返回的输出生成器才会真正完成。

## 工作机制

在内部，vLLM 会把每个输入块都当作一个带累计 prompt 的独立请求来处理。当新输入块到达时，引擎会：

1. 用 `max_tokens - 1` 个已生成的 `output_tokens` 加上新到达的 `prompt_token_ids`，来扩展 `prompt_token_ids`
2. 复用所有已经缓存好的 KV 值
3. 基于当前累计 prompt 和指定的 `max_tokens` 生成输出 token
4. 当新输入到达时，可选地丢弃部分输出

这意味着：在两个输入块之间生成的输出 token，随着更多上下文到来，有可能被修订。最终输出代表的是在完整输入下得到的结果。

在实现层面，vLLM 通过一种 *sticky session* 机制来支持流式输入。第一个输入块会创建一个 **anchor request**，它会贯穿整个会话持续存在。之后带有相同内部 request ID 的输入块，会按顺序入队并依次处理。

### Anchor Request 模式

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAMING SESSION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   用户的 AsyncGenerator                调度器                               │
│   ═══════════════════                 ═════════                             │
│                                                                             │
│   ┌──────────────┐                                                          │
│   │   块 1       │ ──────────────►   添加 ANCHOR REQUEST                    │
│   │   [A, B, C]  │                   ┌────────────────────────────────┐     │
│   └──────────────┘                   │  Request (id="session_1")      │     │
│                                      │  ├── resumable: true           │     │
│                                      │  ├── max_tokens: 2             │     │
│                                      │  ├── streaming_queue: deque()  │     │
│                                      │  ├── status: RUNNING           │     │
│                                      │  └── prompt_token_ids: [A,B,C] │     │
│                                      └────────────────────────────────┘     │
│                                               │                             │
│                                               ▼                             │
│   ┌──────────────┐                   ┌────────────────┐                     │
│   │   块 2       │                   │    ENGINE      │  生成中...          │
│   │   [D, E]     │ ─────┐            │  Processing    │  ──► 输出: [X, Y]  │
│   └──────────────┘      │            └────────────────┘                     │
│                         │                                                   │
│                         ▼             Anchor 正忙？先排队！                  │
│   ┌──────────────┐      │            ┌────────────────────────────────┐     │
│   │   块 3       │      └────────►   │  streaming_queue:              │     │
│   │   [F, G]     │ ─────────────►    │  ┌───────┐ ┌───────┐           │     │
│   └──────────────┘                   │  │[D, E] │→│[F, G] │→ ...      │     │
│                                      │  └───────┘ └───────┘           │     │
│                                      └────────────────────────────────┘     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                     当锚定请求完成当前块                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   引擎信号：当前块处理完成 (stopped = True)                                  │
│          │                                                                  │
│          ▼                                                                  │
│   ┌────────────────────────────────────────────────────────────────┐        │
│   │  _handle_stopped_request() 从队列弹出第一项                    │        │
│   │                                                                │        │
│   │  streaming_queue: [[D,E], [F,G]]  ──►  [[F,G]]                │        │
│   │                      ▲                                        │        │
│   │                      │                                        │        │
│   │                    弹出!                                       │        │
│   └──────────────────────┬─────────────────────────────────────────┘        │
│                          │                                                  │
│                          ▼                                                  │
│   ┌────────────────────────────────────────────────────────────────┐        │
│   │  _update_request_as_session(anchor, update=[D, E])             │        │
│   │                                                                │        │
│   │  更新前:                         更新后:                        │        │
│   │  ┌───────────────────────┐      ┌───────────────────────────┐  │        │
│   │  │ prompt_token_ids:     │      │ prompt_token_ids:         │  │        │
│   │  │   [A, B, C]           │      │   [A, B, C, X, D, E]      │  │        │
│   │  │ _output_token_ids:    │ ──►  │ _output_token_ids:        │  │        │
│   │  │   [X, Y]              │      │   []                      │  │        │
│   │  │ _all_token_ids:       │      │ _all_token_ids:           │  │        │
│   │  │   [A, B, C, X, Y]     │      │   [A, B, C, X, D, E]      │  │        │
│   │  │ num_computed_tokens: 4│      │ num_computed_tokens: 4    │  │        │
│   │  │ status: RUNNING       │      │ status: WAITING           │  │        │
│   │  └───────────────────────┘      └───────────────────────────┘  │        │
│   │                                                                │        │
│   │  注意：Y 会被丢弃（最后一个采样 token，尚无 KV cache）         │        │
│   │        只有 X 被保留（num_computed_tokens = 4，因此是 [A,B,C,X]）│      │
│   └────────────────────────────────────────────────────────────────┘        │
│                          │                                                  │
│                          ▼                                                  │
│   ┌────────────────────────────────────────────────────────────────┐        │
│   │  锚定请求回到等待队列 → 再次调度 → ENGINE                      │        │
│   └────────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**为什么最后一个 token（Y）会在下一轮 `prompt_token_ids` 中被丢弃？**

当接收到一个可恢复请求时，我们真正想要的是：为所有 `prompt_token_ids`，以及 `max_tokens - 1` 个已生成 token，计算好 KV cache。这里用户通过 `max_tokens` 表示，前一个请求里 `max_tokens - 1` 个生成 token 是最终可复用的。

而 `output_token_ids` 里的最后一个 token，只是最近一次前向传播额外得到的结果，它本身还没有对应的 KV cache 状态。用户当然可以自己决定如何使用它，但由于它没有对应的 KV cache，所以在更新后的 anchor request 的 `prompt_token_ids` 中，它会被丢弃。

丢弃它本质上几乎是“免费”的：我们并没有使任何已有缓存失效，反而如果想保留它，还必须重新计算。

正如 [Realtime API 的实现](https://github.com/vllm-project/vllm/blob/a2443de5fa4a0605607f6c3d9219022c7f6ac480/vllm/entrypoints/openai/realtime/connection.py#L209) 所做的那样，在大多数应用里，`max_tokens` 应该设置为 `1`。这样每个可恢复请求只会为 `prompt_token_ids` 计算 KV cache，而 `output_token_ids` 中生成的那个单独 token，则由用户自己决定是否、以及如何利用。以 [Voxtral Realtime](https://mistral.ai/news/voxtral) 为例，这个单 token 会与新到达的音频块一起组成下一次可恢复请求的输入。

_注意事项_：有些模型会生成特殊的停止 token，而模型后续继续生成时又必须重新看到这些 token。在这种情况下，调度逻辑需要额外多留出 `+1` 个 token，用于先重算这个停止 token，再去处理新的输入块。

## 示例流程

为了便于说明，可以把多个 `max_tokens` 不同的可恢复请求，作为流式输入依次送入。在这种情况下，生成逻辑会像下面这样运作：

```text
输入块: ([A1, B1, C1], max_tokens=1), ([A2, B2], max_tokens=2), ([A3], max_tokens=2)

1. 第一个块 [A1, B1, C1] 到达
   -> 模型生成 [D1]

2. 第二个块 [A2, B2] 到达
   -> 累积 prompt: [A1, B1, C1, A2, B2]（D1 被丢弃）
   -> 模型生成 [C2, D2, E2]

3. 第三个块 [A3] 到达
   -> 累积 prompt: [A1, B1, C1, A2, B2, C2, D2, A3]（E2 被丢弃）
   -> 模型生成 [C3, D3]

输出流: D1, C2, D2, E2, C3, D3
```

# 基于 WebSocket 的 Realtime API

流式输入提供了核心能力，但生产环境里的实时应用还需要一个方便的双向通信接口。 [PR #33187](https://github.com/vllm-project/vllm/pull/33187) 引入了一个基于 WebSocket 的 Realtime API，其设计受到 [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) 的启发。

## 架构

Realtime API 提供了一个 WebSocket 端点，用于实现客户端与 vLLM 服务之间的双向流式通信。客户端发送音频数据，服务端返回转录文本和模型输出。

整体架构包含：

1. **WebSocket Client**：从麦克风采集音频，并将音频块发送到服务端
2. **Realtime Handler**：接收 WebSocket 消息，并将其转换为 `StreamingInput`
3. **AsyncLLM**：处理流式输入并生成输出
4. **Response Stream**：通过 WebSocket 将生成 token 发送回客户端

## 服务端启动方式

启动支持 Realtime API 的 vLLM 服务：

```bash
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --enforce-eager
```

服务端会在 `ws://localhost:8000/v1/realtime` 暴露 WebSocket 端点。

## 客户端示例

下面是一个基础客户端示例，它会流式发送音频文件，并接收转录结果：

```python
import asyncio
import base64
import json
import librosa
import numpy as np
import websockets

def load_audio_as_pcm16(audio_path: str) -> bytes:
    """加载音频文件并转换为 PCM16 @ 16kHz。"""
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    return (audio * 32767).astype(np.int16).tobytes()

async def stream_audio_file(audio_path: str, server_url: str = "ws://localhost:8000/v1/realtime"):
    async with websockets.connect(server_url) as ws:
        response = json.loads(await ws.recv())

        # 加载并转换音频为 PCM16
        pcm_audio = load_audio_as_pcm16(audio_path)

        # 校验模型
        await ws.send(json.dumps({"type": "session.update", "model": model}))

        # 标记音频流开始
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # 以 4KB 块流式发送音频
        for i in range(0, len(pcm_audio), 4096):
            chunk = pcm_audio[i:i + 4096]
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode()
            }))

        # 标记音频流结束
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        # 接收转录
        async for message in ws:
            data = json.loads(message)
            if data["type"] == "transcription.delta":
                print(data["delta"], end="", flush=True)
            elif data["type"] == "transcription.done":
                break

asyncio.run(stream_audio_file("audio.wav"))
```

这个示例展示了实时音频流的核心流程：

- **加载并转换音频**：将音频文件加载后转换成 16kHz 的 PCM16 格式，这是 Realtime API 期望的输入格式
- **建立 WebSocket 连接**：连接到服务端的 `/v1/realtime` 端点，并发送 `session.update` 消息来确认模型
- **按块流式发送音频**：音频会以 4KB 的块，通过 `input_audio_buffer.append` 消息发送，并用 `input_audio_buffer.commit` 标记流的开始与结束
- **增量接收转录结果**：服务端通过 `transcription.delta` 消息持续返回局部转录，直到收到 `transcription.done`
- **关于“实时”行为**：为简化示例，这里是先把全部音频发完，再监听转录结果。但 WebSocket 协议本身支持完全异步的双向通信，也就是说发送音频块和接收转录可以同时进行。在真正的生产级实时服务中，第一块音频一到，转录就会开始，发送与接收并发进行，从而实现真正的低延迟语音识别

## 消息类型

Realtime API 使用基于消息的协议。核心消息类型包括：

**客户端到服务端：**
- `session.create`：初始化新会话
- `input_audio_buffer.append`：发送音频数据
- `input_audio_buffer.commit`：标记音频输入结束
- `response.create`：请求模型响应

**服务端到客户端：**
- `session.created`：确认会话初始化成功
- `response.text.delta`：增量文本输出
- `response.audio.delta`：增量音频输出，用于 TTS 模型
- `response.done`：响应完成
- `error`：错误消息

## 示例脚本

vLLM 仓库中已经提供了可直接使用的客户端示例：

- [examples/online_serving/openai_realtime_client.py](https://docs.vllm.ai/en/latest/examples/online_serving/openai_realtime_client/?h=realtime#openai-realtime-client)：基础 WebSocket 客户端
- [examples/online_serving/openai_realtime_microphone_client.py](https://docs.vllm.ai/en/latest/examples/online_serving/openai_realtime_microphone_client/#openai-realtime-microphone-client)：麦克风集成示例

这些示例展示了如何从系统麦克风捕获音频，并实时流式发送给 vLLM。

## 性能考量

与直接发送多个独立请求相比，使用专门的 `AsyncGenerator` 会话接口有一个重要优势：该会话的 KV cache 会被原样保留下来。这比依赖 vLLM 自动前缀缓存更合适，因为：

- 可以确保在等待下一个输入块时，对应 cache block 不会被驱逐
- 前缀缓存工作在 block 级别，通常是 16 个 token，因此如果只依赖它，每次有新输入时，少量已有 token 还是会被重复计算

但这也意味着，需要更谨慎地管理会话生命周期。因为只要一个流式会话保持打开，它占用的内存就不能被其他请求使用，这可能会影响整体容量和吞吐量。目前 vLLM 还不会抢占那些“空闲中”的流式输入会话，这一行为将在未来版本中进一步改进。

## 未来方向

我们对 vLLM 中流式输入支持的潜力非常期待。随着越来越多模型提供方开源与这种输入流式设计兼容的完整流式模型权重，我们预计实时应用生态会显著扩大。

由于流式输入在 LLM 推理服务里仍然是一项相对新的能力，我们也预期会继续调整和扩展这套实现，以覆盖尽可能多的架构和应用场景。这包括：

- 与不同音频、视频编码器做更紧密的集成
- 针对不同延迟目标优化 anchor request 模式
- 扩展对多模态流式场景的支持

## 参与贡献

欢迎你试用 vLLM 的输入流式能力和 Realtime API。你的反馈对这些特性的完善非常重要。你可以在 [vLLM GitHub 仓库](https://github.com/vllm-project/vllm) 分享使用体验、报告问题，或者提出改进建议。

也欢迎你继续参与 vLLM 实时能力的后续建设。

## 致谢

流式输入支持与 Realtime API 的实现，得益于多个团队的协作：

**Meta：** Joshua Deng、Jiatong Zhou、Zhuohan Li、Yu Luo、Jeremy Teboul

**Mistral AI：** Patrick von Platen、Andy Lo

**vLLM 团队：** Nick Hill、Roger Wang、Cyrus Leung、Nicolò Lucchesi、Woosuk Kwon

我们也感谢 vLLM 中其他流式输入实现者：Tao He（阿里巴巴通义千问）、Edward Wibowo（布朗大学）、Deepti Raghavan（布朗大学）以及 Luis Gaspar Schroeder（加州大学伯克利分校）。

## 参考链接

- [流式输入测试与示例](https://github.com/vllm-project/vllm/tree/main/tests/v1/streaming_input)
- [Realtime WebSocket API 文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/?h=realtime+api#realtime-api)
- [OpenAI ChatCompletionRequest API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [vLLM chunked prefill 文档](https://docs.vllm.ai/en/stable/cli/serve/?h=max+num+b#-enable-chunked-prefill-no-enable-chunked-prefill)
- [max_num_batched_tokens 参数](https://docs.vllm.ai/en/stable/cli/serve/?h=max+num+b#-max-num-batched-tokens)
- [Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling](https://arxiv.org/abs/2509.08753)
- [Voxtral-Realtime](https://mistral.ai/news/voxtral)
- [Transformer Transducer](https://arxiv.org/abs/2002.02562)
- [Streaming Simultaneous Speech Translation with Augmented Memory Transformer](https://arxiv.org/abs/2011.00033)
- [PR #28973：流式输入支持](https://github.com/vllm-project/vllm/pull/28973)
- [Realtime API 连接实现](https://github.com/vllm-project/vllm/blob/a2443de5fa4a0605607f6c3d9219022c7f6ac480/vllm/entrypoints/openai/realtime/connection.py#L209)
- [PR #33187：Realtime WebSocket API](https://github.com/vllm-project/vllm/pull/33187)
- [OpenAI Realtime API 指南](https://platform.openai.com/docs/guides/realtime)
- [基础 WebSocket 客户端示例](https://docs.vllm.ai/en/latest/examples/online_serving/openai_realtime_client/?h=realtime#openai-realtime-client)
- [麦克风集成客户端示例](https://docs.vllm.ai/en/latest/examples/online_serving/openai_realtime_microphone_client/#openai-realtime-microphone-client)
- [vLLM GitHub 仓库](https://github.com/vllm-project/vllm)
- [英文原文：blog.vllm.ai/2026/01/31/streaming-realtime.html](https://blog.vllm.ai/2026/01/31/streaming-realtime.html)
