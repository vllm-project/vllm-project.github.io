---
layout: post
title: "Streaming Requests & Realtime API in vLLM"
author: "TODO"
image: /assets/figures/2026-01-31-streaming-realtime/architecture.png
math: true
---

Large language model inference has traditionally operated on a simple premise: the user submits a complete prompt (request), the model processes it, and returns a response (either streaming or at once). This paradigm works well for text-based chatbots and batch processing workloads, but it falls short when dealing with realtime applications, such as streaming audio or video. 

vLLM has recently added support for **streamable inputs** to its engine as well as a **Realtime WebSocket API** building on top of it, exposing a new `/v1/realtime` endpoint in the server.

In this post, we motivate the need for realtime inference and introduce the two new features in vLLM that unlock these capabilities: **streaming input support** and the **Realtime WebSocket API**.


# Why Realtime is Needed

## The Traditional Batch Paradigm in vLLM

Traditional LLM inference assumes that complete prompts are available upfront. The user submits their full request *e.g.* via a `ChatCompletionRequest`, waits for the model to process it entirely, and then receives the complete response. While vLLM has long supported *output* streaming—emitting tokens as they are generated—the *input* side was always fixed: you had to provide the entire request before inference could begin.

This approach is sufficient for most applications. Text-based chatbots, document summarization, and code generation all fit naturally into this model. But a growing class of applications cannot wait for complete input before processing begins.

## The Importance of Streaming

Consider a voice assistant to control a computer or phone. Instead of using a keyboard and mouse or touchpad, all actions are controlled by voice. Speech is recorded by a microphone and sent as a stream of audio to your LLM which acts as the voice assistant model. The LLM needs to continuously process the audio stream and generate actions in realtime. For such applications latency matters—a user does not want to wait more than a second to open an application, type text into a search bar, etc. For the most natural, human-like voice assistant, it needs to be able to listen and speak at the same time, i.e., the LLM needs to be able to process the audio stream and generate actions simultaneously.

A natural question is whether streaming behavior can be approximated using non-streaming LLMs by processing the input in chunks. In principle, audio can be buffered into segments, each segment processed independently, and the resulting outputs concatenated. In practice, this approach introduces several limitations. Achieving sub-second latency requires highly performant chunk detection, i.e., accurately determining when to segment the audio stream such that no relevant information is lost. Poor segmentation can introduce additional latency or degrade model performance by fragmenting meaningful temporal context. Chunk-based processing also precludes true bidirectional interaction: each chunk must be fully processed before a response can be generated, preventing listening and speaking from occurring concurrently. This results in a turn-based interaction model rather than the continuous, overlapping communication characteristic of human conversation.

This problem appears across many domains:

- **Voice assistants** require sub-second response times to feel natural as described above
- **Live transcription services** need to display text as speech is recognized
- **Robotics and embodied AI** need to process continuous sensor streams (cameras, microphones, LIDAR) and generate control actions with minimal delay to interact safely with the physical world

For these applications, the traditional batch paradigm introduces unacceptable delays. We need infrastructure that can process input incrementally and begin generating output before all input has arrived.

## Architectural Requirements for Streaming

Not all models can support true streaming inference. Two key requirements must be met: the right attention pattern and training for incremental processing.

### Attention Patterns

The attention mechanism determines whether a model can process input incrementally or must wait for the entire sequence.

- Causal attention (uni-directional mask) restricts each position t to attend only to tokens at positions ≤ t. Because future tokens are excluded, the model’s output token at time t is final once token t arrives. This makes true streaming possible: each new token can be processed immediately, and earlier outputs never need to be revised.

- Bidirectional attention (full mask) allows every position to attend to both past and future tokens. As a result, the model's output token at position t is conditions on tokens that may not have arrived yet. Until the full input sequence is known, the model cannot compute a stable output for any position, because future tokens could change how earlier tokens are interpreted.

For this reason, bidirectional attention inherently requires access to the complete input sequence before producing outputs, which makes it incompatible with streaming or online processing.

For long-running or infinite streaming, standard causal attention is not enough. If each token attends to the entire past, computation and memory grow without bound, which is impractical. In practice, past context must be truncated. 
A common architectural solution is sliding-window attention, where each token attends only to a fixed-size window of recent tokens, keeping compute and memory bounded while supporting streaming. Hence, causal attention with a sliding window is often the architecture of choice for modern streaming models.

### Training for Streaming inputs

However, having a fully streamable architecture is not sufficient on its own: the model must also be trained to support **true streaming input**.

Let $X = (x_0, x_1, \ldots, x_T)$ denote the input sequence and $Y = (y_0, y_1, \ldots, y_{T'})$ the output sequence. In streaming applications, the model should generate the output $y_t$ corresponding to input $x_t$ at time step $t$, with as little latency as possible. Concretely, one can think of $y_t$ as the transcription of an audio frame $x_t$ that is streamed into the model at time $t$.

The standard next-token training objective typically conditions the distribution of the next token on the *entire* input sequence:

$$
P(y_i \mid y_{i-1}, \ldots, y_0, x_T, x_{T-1}, \ldots, x_0).
$$

This formulation is unsuitable for streaming, because generating $y_i$ requires processing the full input sequence $X$, which is not available in real time.

Instead, a streaming model must be able to predict $y_i$ using only past inputs and, optionally, a small amount of future context:

$$
P(y_i \mid y_{i-1}, \ldots, y_0, x_{i+\delta}, \ldots, x_i, \ldots, x_0),
$$

where $\delta$ is a lookahead parameter that should be as small as possible. In theory, $\delta$ could be set to zero; in practice, a small delay is usually necessary to achieve reasonable performance.

As a result, training a streaming model requires:
- **i)** aligning input and output sequences such that $T' = T$, and each $y_i$ is the correct output corresponding to $x_i$;
- **ii)** using an architecture that can process new inputs $x_{i+1}$ while previous inputs $x_i, \ldots, x_0$ have already been processed.

An intuitive architecture, used in [Moshi](https://arxiv.org/abs/2410.00037) and [Voxtral-Realtime](TODO), sum-pools input embeddings (e.g., speech embeddings) and output embeddings (e.g., text embeddings) into a single sequence of embeddings. The model then predicts

$$
P(y_i \mid y'_{i-1}, \ldots, y'_0),
$$

where

$$
y'_k = \sum_(y_k, x_{k+\delta}).
$$

This distinction is important for deployment: one cannot simply take an arbitrary causal model and expect it to perform well in a streaming setting. To be fully streamable, the model must be explicitly trained with the above alignment and architectural constraints, ensuring that both conditions **i)** and **ii)** are satisfied.

## Why Model Architecture Matters for Serving

vLLM can serve any model, but true streaming requires architecturally-causal models. Models like [Voxtral](https://mistral.ai/news/voxtral) are designed from the ground up for streaming, using causal attention mechanisms that support incremental processing.

Equally important, the serving infrastructure must support incremental input. Even with a streaming-capable model, if the server requires the complete prompt before beginning inference, you lose the latency benefits. This is why vLLM now supports streaming input alongside its existing output streaming capabilities.

### Further Reading

....

# Streaming Input Support in vLLM

TODO(Needs rewriting / restructuring)

With [PR #28973](https://github.com/vllm-project/vllm/pull/28973), vLLM now supports streaming input for inference. This enables the incremental processing described above, where input arrives over time and output is generated continuously.

## The StreamingInput Interface

The core abstraction is the `StreamingInput` dataclass:

```python
from dataclasses import dataclass
from vllm.inputs import PromptType
from vllm.sampling_params import SamplingParams

@dataclass
class StreamingInput:
    prompt: PromptType
    sampling_params: SamplingParams | None = None
```

Rather than passing a fixed prompt to `AsyncLLM.generate()`, you can now pass an `AsyncGenerator` that yields `StreamingInput` objects over time. Each `StreamingInput` contains the cumulative prompt up to that point.

## How It Works

Internally, vLLM handles streaming input by treating each chunk as a separate request with a cumulative prompt. As new chunks arrive, the engine:

1. Extends the prompt with the new content
2. Reuses cached KV values for the prefix (via prefix caching)
3. Generates output tokens based on the current cumulative prompt
4. Optionally discards speculative output when new input arrives

This design means that output tokens generated between input chunks may be revised as more context becomes available. The final output reflects the complete input.

Internally, vLLM implements streaming input through a *sticky session* mechanism. The first input chunk creates an **anchor request** that persists throughout the session. Subsequent chunks with the same internal request ID are queued and processed in order.

### The Anchor Request Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAMING SESSION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User's AsyncGenerator              Scheduler                              │
│   ═══════════════════               ═════════                               │
│                                                                             │
│   ┌──────────────┐                                                          │
│   │   Chunk 1    │ ──────────────►  Add ANCHOR REQUEST                      │
│   │   [A, B, C]  │                  ┌────────────────────────────────┐      │
│   └──────────────┘                  │  Request (id="session_1")      │      │
│                                     │  ├── resumable: true           │      │
│                                     │  ├── streaming_queue: deque()  │      │
│                                     │  ├── status: RUNNING           │      │
│                                     │  └── prompt_token_ids: [A,B,C] │      │
│                                     └────────────────────────────────┘      │
│                                              │                              │
│                                              ▼                              │
│   ┌──────────────┐                  ┌────────────────┐                      │
│   │   Chunk 2    │                  │    ENGINE      │  Generating...       │
│   │   [D, E]     │ ─────┐           │  Processing    │  ──► Output: [X, Y]  │
│   └──────────────┘      │           └────────────────┘                      │
│                         │                                                   │
│                         ▼           Anchor busy? Queue it!                  │
│   ┌──────────────┐      │           ┌────────────────────────────────┐      │
│   │   Chunk 3    │      └────────►  │  streaming_queue:              │      │
│   │   [F, G]     │ ─────────────►   │  ┌───────┐ ┌───────┐           │      │
│   └──────────────┘                  │  │[D, E] │→│[F, G] │→ ...      │      │
│                                     │  └───────┘ └───────┘           │      │
│                                     └────────────────────────────────┘      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                     WHEN ANCHOR FINISHES CURRENT CHUNK                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Engine signals: chunk complete (stopped = True)                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌────────────────────────────────────────────────────────────────┐        │
│   │  _handle_stopped_request() pops first item from queue          │        │
│   │                                                                │        │
│   │  streaming_queue: [[D,E], [F,G]]  ──►  [[F,G]]                 │        │
│   │                      ▲                                         │        │
│   │                      │                                         │        │
│   │                    pop!                                        │        │
│   └──────────────────────┬─────────────────────────────────────────┘        │
│                          │                                                  │
│                          ▼                                                  │
│   ┌────────────────────────────────────────────────────────────────┐        │
│   │  _update_request_as_session(anchor, update=[D, E])             │        │
│   │                                                                │        │
│   │  BEFORE:                       AFTER:                          │        │
│   │  ┌───────────────────────┐     ┌───────────────────────────┐   │        │
│   │  │ prompt_token_ids:     │     │ prompt_token_ids:         │   │        │
│   │  │   [A, B, C]           │     │   [A, B, C, X, D, E]      │   │        │
│   │  │ _output_token_ids:    │ ──► │ _output_token_ids:        │   │        │
│   │  │   [X, Y]              │     │   []                      │   │        │
│   │  │ _all_token_ids:       │     │ _all_token_ids:           │   │        │
│   │  │   [A, B, C, X, Y]     │     │   [A, B, C, X, D, E]      │   │        │
│   │  │ num_computed_tokens: 4│     │ num_computed_tokens: 4    │   │        │
│   │  │ status: RUNNING       │     │ status: WAITING           │   │        │
│   │  └───────────────────────┘     └───────────────────────────┘   │        │
│   │                                                                │        │
│   │  Note: Y is DISCARDED (last sampled token, not yet computed)   │        │
│   │        Only X is kept (num_computed_tokens = 4, so [A,B,C,X])  │        │
│   └────────────────────────────────────────────────────────────────┘        │
│                          │                                                  │
│                          ▼                                                  │
│   ┌────────────────────────────────────────────────────────────────┐        │
│   │  Anchor returns to waiting queue → scheduled again → ENGINE    │        │
│   └────────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why is the last token (Y) discarded?**

In streaming models, the final sampled token for each chunk can be typically considered as a special token that signals "I'm done processing this input chunk and waiting for more." When the next input chunk arrives, this speculative token becomes meaningless—it was a placeholder indicating "waiting for input" that is now superseded by actual new input. The model will generate a fresh continuation based on the new context, so the speculative token is discarded rather than kept in the prompt. Note, This behavior is model-specific behavior and can be changed if needed.

## Example Flow

Consider a voice assistant receiving speech incrementally:

```
Input chunks: [A1, B1, C1], [A2, B2], [A3, B3]

1. First chunk [A1, B1, C1] arrives
   -> Model generates [D1]

2. Second chunk [A2, B2] arrives
   -> Cumulative prompt: [A1, B1, C1, A2, B2]
   -> Model generates [C2, D2, E2] (D1 discarded as speculative)

3. Third chunk [A3, B3] arrives
   -> Cumulative prompt: [A1, B1, C1, A2, B2, C2, D2, A3, B3]
   -> Model generates [C3, D3]

Output stream: D1, C2, D2, E2, C3, D3
```

The key insight is that early output tokens provide immediate feedback to the user, even though they may be revised as more context arrives. This dramatically reduces perceived latency.

## Code Example

...

## KV Cache Efficiency

...

# Realtime API with WebSockets

While streaming input support provides the core capability, production applications need a convenient API for real-time communication. [PR #33187](https://github.com/vllm-project/vllm/pull/33187) introduces a WebSocket-based Realtime API inspired by [OpenAI's Realtime API](https://platform.openai.com/docs/guides/realtime).

## Architecture

The Realtime API provides a WebSocket endpoint that enables bidirectional streaming between clients and the vLLM server. Clients send audio data, and the server responds with transcribed text and model outputs.

The architecture consists of:

1. **WebSocket Client**: Captures audio from microphone, sends chunks to server
2. **Realtime Handler**: Receives WebSocket messages, converts to StreamingInput
3. **AsyncLLM**: Processes streaming input, generates output
4. **Response Stream**: Sends generated tokens back through WebSocket

## Server Setup

Starting a vLLM server with Realtime API support:

```bash
vllm serve mistralai/Voxtral-Mini-3B-Realtime-2602 \
    --gpu_memory_utilization 0.8 \
    --max_model_len 32768
```

The server exposes a WebSocket endpoint at `ws://localhost:8000/v1/realtime`.

## Client Example

Here's a basic client that sends audio and receives responses:

...

## Message Types

The Realtime API uses a message-based protocol. Key message types include:

**Client to Server:**
- `session.create`: Initialize a new session
- `input_audio_buffer.append`: Send audio data
- `input_audio_buffer.commit`: Signal end of audio input
- `response.create`: Request model response

**Server to Client:**
- `session.created`: Session initialization confirmed
- `response.text.delta`: Incremental text output
- `response.audio.delta`: Incremental audio output (for TTS models)
- `response.done`: Response complete
- `error`: Error occurred

## Example Scripts

The vLLM repository includes ready-to-use example clients:

- `examples/online_serving/openai_realtime_client.py`: Basic WebSocket client
- `examples/online_serving/openai_realtime_microphone_client.py`: Microphone integration

These examples demonstrate how to capture audio from system microphone and stream it to vLLM in real time.

## Performance Considerations

...

## Current Limitations

...

## Future Directions

...

## Get Involved

...

We welcome feedback and contributions as we continue to develop vLLM's real-time capabilities.
