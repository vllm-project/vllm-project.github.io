---
layout: post
title: "Streaming Requests & Realtime API in vLLM"
author: "Meta, Mistral AI as well as the vLLM team"
math: true
---

Large language model inference has traditionally operated on a simple premise: the user submits a complete prompt (request), the model processes it, and returns a response (either streaming or at once). This paradigm works well for text-based chatbots and batch processing workloads, but it falls short when dealing with realtime applications, such as streaming audio or video. 

vLLM has recently added support for **streamable inputs** to its engine as well as a **Realtime WebSocket API** building on top of it, exposing a new `/v1/realtime` endpoint in the server.

In this post, we motivate the need for realtime inference and introduce the two new features in vLLM that unlock these capabilities: **streaming input support** and the **Realtime WebSocket API**.

_Note_: If you wish to know how to use the new streaming input or realtime API with vLLM, please refer to the following references:
- [Streaming input](https://github.com/vllm-project/vllm/tree/main/tests/v1/streaming_input)
- [Realtime WebSocket API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/?h=realtime+api#realtime-api)

# Why Realtime is Needed

## The Traditional Batch Paradigm in vLLM

Traditional LLM inference assumes that complete prompts are available upfront. The user submits their full request *e.g.* via a [`ChatCompletionRequest`](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), waits for the model to process it entirely, and then receives the complete response. While vLLM has long supported *output* streaming—emitting tokens as they are generated—the *input* side was always fixed: you had to provide the entire request before inference could begin.

This approach is sufficient for most applications. Text-based chatbots, document summarization, and code generation all fit naturally into this model. But a growing class of applications cannot wait for complete input before processing begins.

## The Importance of Streaming

Consider a voice assistant to control a computer or phone. Instead of using a keyboard and mouse or touchpad, all actions are controlled by voice. Speech is recorded by a microphone and sent as a stream of audio to your LLM which acts as the voice assistant model. 
The LLM needs to continuously process the audio stream and generate actions in realtime. 
For such applications latency, or more precisely [Time-To-First-Token (TTFT)](https://www.emergentmind.com/topics/time-to-first-token-ttft), matters—a user does not want to wait more than a second to open an application, type text into a search bar, etc. For the most natural, human-like voice assistant, it needs to be able to listen and speak at the same time, i.e., the LLM needs to be able to process the audio stream and generate actions simultaneously.

A natural question is whether streaming behavior can be approximated using non-streaming LLMs by processing the input in chunks. In principle, audio can be buffered into segments, each segment processed independently, and the resulting outputs concatenated. In practice, this approach introduces several limitations. Achieving sub-second TTFT requires highly performant chunk detection, i.e., accurately determining when to segment the audio stream such that no relevant information is lost. Poor segmentation can lead to an increased TTFT or degrade model performance by fragmenting meaningful temporal context. Chunk-based processing also precludes true bidirectional interaction: each chunk must be fully processed before a response can be generated, preventing listening and speaking from occurring concurrently. This results in a turn-based interaction model rather than the continuous, overlapping communication characteristic of human conversation.

This problem appears across many domains:

- **Voice assistants** require sub-second response times to feel natural as described above
- **Live transcription services** need to display text as speech is recognized
- **Robotics and embodied AI** need to process continuous sensor streams (cameras, microphones, LIDAR) and generate control actions with minimal delay to interact safely with the physical world

For these applications, the traditional batch paradigm introduces unacceptable delays. Infrastructure is needed that can process input incrementally and begin generating output before all input has arrived.

_Note_: Even for traditional applications, where the full input needs to be read in order to generate the first output token, the ability to stream input as it becomes available can still be beneficial.
By default, vLLM makes use of [chunked prefill](https://docs.vllm.ai/en/stable/cli/serve/?h=max+num+b#-enable-chunked-prefill-no-enable-chunked-prefill) and therefore processes an input of $N$ tokens in $N \div M$ forward passes with $M$ being [`max_num_batched_tokens`](https://docs.vllm.ai/en/stable/cli/serve/?h=max+num+b#-max-num-batched-tokens). In case $N \div M > 1$ streaming the input as it becomes available reduces the overall TTFT because the first prefill forward pass can be scheduled earlier.

## Requirements for Streaming

Not all models can support true streaming inference. Two key requirements must be met: the right attention pattern and training for incremental processing.

### Attention Patterns

The attention mechanism determines whether a model can process input incrementally or must wait for the entire sequence.

- Causal attention (uni-directional mask) restricts each position $t$ to attend only to tokens at positions $j$ with $j \le t$. Because future tokens are excluded, the model’s output token at time $t$ is final once token $t$ arrives. This makes true streaming possible: each new token can be processed immediately, and earlier outputs never need to be revised.

- Bidirectional attention (full mask) allows every position to attend to both past and future tokens. As a result, the model's output token at position $t$ is conditions on tokens that may not have arrived yet. Until the full input sequence is known, the model cannot compute a stable output for any position, because future tokens could change how earlier tokens are interpreted.

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

An intuitive architecture, as pioneered by [Delayed Streams Modeling](https://arxiv.org/pdf/2509.08753) and picked up by [Voxtral-Realtime](TODO), sum-pools input embeddings (e.g., speech embeddings) and output embeddings (e.g., text embeddings) into a single sequence of embeddings. The model then predicts

$$
P(y_i \mid y'_{i-1}, \ldots, y'_0),
$$

where

$$
y'_k = y_k + x_{k+\delta}.
$$

This distinction is important for deployment: one cannot simply take an arbitrary causal model and expect it to perform well in a streaming setting. To be fully streamable, the model must be explicitly trained with the above alignment and architectural constraints, ensuring that both conditions **i)** and **ii)** are satisfied.

## Why Model Architecture Matters for Serving

vLLM can serve any model, but true streaming requires architecturally-causal models. Models like [Voxtral](https://mistral.ai/news/voxtral) are designed from the ground up for streaming, using causal attention mechanisms that support incremental processing.

Equally important, the serving infrastructure must support incremental input. Even with a streaming-capable model, if the server requires the complete prompt before beginning inference, you lose the latency benefits. 
This is why vLLM now supports streaming input alongside its existing output streaming capabilities.

### Further Reading on Streaming architecture

- [Transformer Transducer](https://arxiv.org/abs/2002.02562) is a well-known and one of the most successful modeling approaches for training streamable speech recognition systems.
- [Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling](https://arxiv.org/abs/2509.08753) by the Kyutai folks is a great read on further diving into streaming architectures as explained above.
- [Streaming Simultaneous Speech Translation with Augmented Memory Transformer](https://arxiv.org/abs/2011.00033) on streaming speech translation. Translation is not as "monotonic" as speech, which makes the problem of performant streaming more difficult.
- [Voxtral-Realtime](https://mistral.ai/news/voxtral) is a massively pretrained and open-sourced streaming model competitive with most offline speech recognition models.

# Streaming Input Support in vLLM

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

Rather than passing a fixed prompt to `AsyncLLM.generate()`, you can now pass an `AsyncGenerator` that yields `StreamingInput` objects over time. Each `StreamingInput` contains the next input chunk to be appended to a cumulative prompt. Here is an example of how it can be used:

```python
import asyncio
from vllm.inputs.data import StreamingInput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams

async def streaming_input_example():
    async_llm = AsyncLLM.from_engine_args(...)
    
    # Input queue can consume inputs in separate async task
    input_queue = asyncio.Queue[list[int]]()
    
    async def input_generator():
        # Loop until empty list encountered => input finished
        while new_tokens := input_queue.get():
            yield StreamingInput(prompt=new_tokens)

    output_generator = async_llm.generate(
        prompt=input_generator(),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
    )
    
    # Consume outputs
    async for output in output_generator:
        # ...

asyncio.run(streaming_input_example())
```

You can wait until the output corresponding to the last input has completed before sending the next input, but it's not required (input chunks are queued internally). Termination of the input stream is indicated by exiting from the async input generator or closing it using the `aclose` function. The returned outputs generator won't complete until all received inputs have been processed _and_ the input generator has completed. 

## How It Works

Internally, vLLM handles streaming input by treating each chunk as a separate request with a cumulative prompt. As new chunks arrive, the engine:

1. Extends `prompt_token_ids` with `max_tokens - 1` generated `output_tokens` as well as new incoming `prompt_token_ids`.
2. Reuses all cached KV values
3. Generates output tokens based on the current cumulative prompt and the indicated `max_tokens`
4. Optionally discards output when new input arrives

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
│                                     │  ├── max_tokens: 2             │      │
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

**Why is the last token (Y) discarded for the next `prompt_token_ids`?**

When receiving a resumable request, what we really care about is computing the KV cache for all `prompt_token_ids` as well as `max_tokens - 1` generated tokens. Note that the user indicates with `max_tokens` that `max_tokens - 1` generated tokens of the first request are final and shall be re-used. The final token of the `output_token_ids` tensor is simply the result of the most recent forward pass, which does not yet have corresponding KV cache states. It is up to the user to decide what do to with it, but as it does not have any corresponding KV cache states, it will 
be discarded for the `prompt_token_ids` of the updated anchor request.
Discarding it is essentially "free": we're not invalidating any cached state, and it would need to be recomputed anyway if we kept it.

As is done in the [Realtime API](https://github.com/vllm-project/vllm/blob/a2443de5fa4a0605607f6c3d9219022c7f6ac480/vllm/entrypoints/openai/realtime/connection.py#L209), in most application, `max_tokens` shall be set to `1` so that each resumable request computes the KV cache states only for the `prompt_token_ids` and it is up to the user to decide how to make use of the generated single token in `output_token_ids`. As an example, for [Voxtral Realtime](https://mistral.ai/news/voxtral), the generated single token in `output_token_ids` will be combined with the incoming new audio chunk to form the next resumable request.

*Caveat:* Some models emit special stop tokens that the model requires to properly continue generation. In such cases, the scheduling logic needs to accommodate +1 token to recompute the stop token before processing the new input chunk.


## Example Flow

For illustrative purposes, multiple resumable requests with different `max_tokens` can be streamed as inputs.
In such a case, the generation logic would function as follows:

```
Input chunks: ([A1, B1, C1], max_tokens=1), ([A2, B2], max_tokens=2), ([A3], max_tokens=2)

1. First chunk [A1, B1, C1] arrives
   -> Model generates [D1]

2. Second chunk [A2, B2] arrives
   -> Cumulative prompt: [A1, B1, C1, A2, B2] (D1 discarded)
   -> Model generates [C2, D2, E2]

3. Third chunk [A3] arrives
   -> Cumulative prompt: [A1, B1, C1, A2, B2, C2, D2, A3] (E2 discarded)
   -> Model generates [C3, D3]

Output stream: D1, C2, D2, E2, C3, D3
```

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
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --enforce-eager
```

The server exposes a WebSocket endpoint at `ws://localhost:8000/v1/realtime`.

## Client Example

Here's a basic client that streams an audio file and receives transcription:

```python
import asyncio
import base64
import json
import librosa
import numpy as np
import websockets

def load_audio_as_pcm16(audio_path: str) -> bytes:
    """Load audio file and convert to PCM16 @ 16kHz."""
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    return (audio * 32767).astype(np.int16).tobytes()

async def stream_audio_file(audio_path: str, server_url: str = "ws://localhost:8000/v1/realtime"):
    async with websockets.connect(server_url) as ws:
        response = json.loads(await ws.recv())

        # Load and convert audio to PCM16
        pcm_audio = load_audio_as_pcm16(audio_path)

        # Validate model
        await ws.send(json.dumps({"type": "session.update", "model": model}))

        # Signal start of audio stream
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Stream audio in 4KB chunks
        for i in range(0, len(pcm_audio), 4096):
            chunk = pcm_audio[i:i + 4096]
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode()
            }))

        # Signal end of audio stream
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        # Receive transcription
        async for message in ws:
            data = json.loads(message)
            if data["type"] == "transcription.delta":
                print(data["delta"], end="", flush=True)
            elif data["type"] == "transcription.done":
                break

asyncio.run(stream_audio_file("audio.wav"))
```

This example demonstrates the core workflow for realtime audio streaming:

- **Load and convert audio**: The audio file is loaded and converted to PCM16 format at 16kHz, which is the expected input format for the realtime API
- **Establish WebSocket connection**: Connect to the server's `/v1/realtime` endpoint and send a `session.update` message to validate the model
- **Stream audio in chunks**: The audio is sent in 4KB chunks using `input_audio_buffer.append` messages, with `input_audio_buffer.commit` signals to mark the start and end of the stream
- **Receive transcription incrementally**: The server responds with `transcription.delta` messages containing partial transcriptions, which are printed in real-time until `transcription.done` is received
- **Note on realtime behavior**: While this example sends all audio before listening for transcriptions (for simplicity), the WebSocket protocol enables fully asynchronous communication—audio chunks can be sent and transcriptions received simultaneously. In a production realtime service, transcription would begin immediately as the first audio chunk arrives, with both sending and receiving happening concurrently for true low-latency speech recognition

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

- [examples/online_serving/openai_realtime_client.py](https://docs.vllm.ai/en/latest/examples/online_serving/openai_realtime_client/?h=realtime#openai-realtime-client): Basic WebSocket client
- [examples/online_serving/openai_realtime_microphone_client.py](https://docs.vllm.ai/en/latest/examples/online_serving/openai_realtime_microphone_client/#openai-realtime-microphone-client): Microphone integration

These examples demonstrate how to capture audio from system microphone and stream it to vLLM in real time.


## Performance Considerations

An advantage of using the dedicated `AsyncGenerator`-based session interface over just sending separate requests is that the KV cache for the session is preserved as-is. This is preferable to relying on vLLM's automatic prefix caching because:
- It ensures that the corresponding cache blocks won't be evicted while waiting for the next input chunk
- Prefix caching works at a block-level (typically 16 tokens), meaning that a small number of existing tokens would otherwise be re-computed for each new input

However, this also means that additional care must be taken to avoid holding sessions open as they will be blocking the corresponding memory from being used by other requests, potentially harming overall capacity/throughput. Currently, vLLM will not preempt "idle" streaming input sessions - this behaviour will be improved in a future update.

## Future Directions

We are excited about the potential for streaming input support in vLLM. As more model providers open-source fully streamable model weights that are compatible with our input streaming design, we expect the ecosystem of realtime applications to grow significantly.

Since streaming input is still a novel capability in LLM serving, we anticipate adapting and extending our implementation to support a maximum number of different architectures and use cases. This includes exploring tighter integration with various audio and video encoders, optimizing the anchor request pattern for different latency requirements, and expanding support for multi-modal streaming scenarios.

## Get Involved

We encourage you to try out vLLM's input streaming functionality and Realtime API. Your feedback is invaluable in helping us improve these features. Please share your experiences, report issues, or suggest enhancements on the [vLLM GitHub repository](https://github.com/vllm-project/vllm).

We welcome feedback and contributions as we continue to develop vLLM's real-time capabilities.

## Acknowledgements

Streaming input support and the Realtime API were made possible through collaborative efforts across multiple teams:

**Meta:** Joshua Deng, Jiatong Zhou, Zhuohan Li, Yu Luo, Jeremy Teboul

**Mistral AI:** Patrick von Platen, Andy Lo

**vLLM Team:** Nick Hill, Roger Wang, Cyrus Leung, Nicolò Lucchesi, Woosuk Kwon

We would also like to acknowledge other implementations of streaming input in vLLM: Tao He (Alibaba Qwen), Edward Wibowo (Brown University), Deepti Raghavan (Brown University), and Luis Gaspar Schroeder (UC Berkeley).
