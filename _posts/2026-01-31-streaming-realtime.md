---
layout: post
title: "Streaming Requests & Realtime API in vLLM"
author: "vLLM Team"
image: /assets/figures/2026-01-31-streaming-realtime/architecture.png
---

Large language model inference has traditionally operated on a simple premise: the user submits a complete prompt (request), the model processes it, and returns a response (either streaming or at once). This paradigm works well for text-based chatbots and batch processing workloads, but it falls short when dealing with realtime applications, such as streaming audio or video. 

vLLM has recently added support for **streamable inputs** to its engine as well as a **Realtime WebSocket API** building on top of it.

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

Causal attention (triangular mask) restricts each position t to attend only to tokens at positions ≤ t. Because future tokens are explicitly excluded, the model’s output at time t is final once token t arrives. This makes true streaming possible: each new token can be processed immediately, and earlier outputs never need to be revised.

Bidirectional attention (full mask) allows every position to attend to both past and future tokens. As a result, the representation at position t is defined in terms of tokens that may not have arrived yet. Until the full input sequence is known, the model cannot compute a stable output for any position, because future tokens could change how earlier tokens are interpreted.

For this reason, bidirectional attention inherently requires access to the complete input sequence before producing outputs, which makes it incompatible with streaming or online processing.

Causal attention is therefore a necessary condition for streaming, but on its own it is not sufficient.

### Training for Token-by-Token Processing

A model with causal attention can still fail at streaming if it wasn't trained to process input incrementally. Consider a speech recognition model trained on complete utterances: given a full audio clip, it produces a transcript. Even if this model uses causal attention internally, it expects the entire input before generating meaningful output—it has learned to "wait" for context that won't arrive in a streaming setting.

True streaming models like [Moshi]( ) or [Voxtral Realtime](Link) are trained differently. They learn to produce useful output after each input token, refining their predictions as more context arrives. This requires training data and objectives that reward incremental accuracy, not just final-output quality.

The distinction matters for deployment: you cannot simply take any causal model and expect it to stream well. The model must be designed and trained for the streaming use case from the start.

## Why Model Architecture Matters for Serving

vLLM can serve any model, but true streaming requires architecturally-causal models. Models like [Voxtral](https://mistral.ai/news/voxtral) are designed from the ground up for streaming, using causal attention mechanisms that support incremental processing.

Equally important, the serving infrastructure must support incremental input. Even with a streaming-capable model, if the server requires the complete prompt before beginning inference, you lose the latency benefits. This is why vLLM now supports streaming input alongside its existing output streaming capabilities.

### Further Reading

For more background on streaming transformer architectures:

- [Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss](https://arxiv.org/abs/2002.02562) - foundational work on transformer streaming
- [Adaptive Non-causal Transformers for Streaming](https://arxiv.org/abs/2305.04159) - ANCAT architecture
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) - state-of-the-art streaming ASR
- [SpeechBrain Conformer Streaming Tutorial](https://speechbrain.readthedocs.io/en/v1.0.3/tutorials/nn/conformer-streaming-asr.html)

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

Rather than passing a fixed prompt to `AsyncLLM.generate()`, you now pass an `AsyncGenerator` that yields `StreamingInput` objects over time. Each `StreamingInput` contains the cumulative prompt up to that point.

## How It Works

Internally, vLLM handles streaming input by treating each chunk as a separate request with a cumulative prompt. As new chunks arrive, the engine:

1. Extends the prompt with the new content
2. Reuses cached KV values for the prefix (via prefix caching)
3. Generates output tokens based on the current cumulative prompt
4. Optionally discards speculative output when new input arrives

This design means that output tokens generated between input chunks may be revised as more context becomes available. The final output reflects the complete input.

<p align="center">
<picture>
<img src="/assets/figures/2026-01-31-streaming-realtime/streaming-input-flow.png" width="90%">
</picture><br>
<b>Figure 3</b>: StreamingInput objects yield incrementally as ASR chunks arrive. vLLM accumulates context and reuses KV cache across chunks for efficient processing.
</p>

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

Here is a complete example of using streaming input:

```python
import asyncio
from vllm import AsyncLLM
from vllm.inputs import StreamingInput

async def input_generator():
    """Simulate incremental input arriving over time."""
    yield StreamingInput(prompt="The weather in")
    await asyncio.sleep(0.1)  # Simulate ASR latency

    yield StreamingInput(prompt="The weather in Paris today is")
    await asyncio.sleep(0.1)

    yield StreamingInput(prompt="The weather in Paris today is sunny. What should I wear?")

async def main():
    llm = AsyncLLM(model="mistralai/Voxtral-Mini-3B-Realtime-2602")

    async for output in llm.generate(input_generator()):
        # Output arrives incrementally as input chunks are processed
        print(output.outputs[0].text, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

In practice, the `input_generator()` would be connected to an ASR system that yields text as speech is recognized.

## KV Cache Efficiency

A critical optimization is that the KV cache computed for earlier chunks is reused when processing subsequent chunks. When the second chunk arrives, vLLM doesn't recompute attention for the prefix—it retrieves the cached KV values and only computes attention for the new tokens.

This is essential for performance. Without KV cache reuse, each new chunk would require reprocessing the entire cumulative prompt, making streaming input prohibitively expensive.

The prefix caching mechanism works seamlessly with streaming input:

1. When the first chunk arrives, vLLM computes KV values and caches them
2. When subsequent chunks arrive, vLLM matches the prefix against the cache
3. Only the new tokens require fresh computation
4. Output generation continues with minimal redundant work

This means that even for long conversations, the marginal cost of processing each new chunk remains proportional to the chunk size, not the cumulative context length. For a typical ASR system producing 10-20 tokens per chunk, this represents a substantial efficiency gain over recomputing from scratch.

# Realtime API with WebSockets

While streaming input support provides the core capability, production applications need a convenient API for real-time communication. [PR #33187](https://github.com/vllm-project/vllm/pull/33187) introduces a WebSocket-based Realtime API inspired by [OpenAI's Realtime API](https://platform.openai.com/docs/guides/realtime).

## Architecture

The Realtime API provides a WebSocket endpoint that enables bidirectional streaming between clients and the vLLM server. Clients send audio data, and the server responds with transcribed text and model outputs.

<p align="center">
<picture>
<img src="/assets/figures/2026-01-31-streaming-realtime/realtime-architecture.png" width="90%">
</picture><br>
<b>Figure 4</b>: The Realtime API bridges WebSocket messages to vLLM's streaming input interface, enabling bidirectional audio-to-text pipelines.
</p>

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

```python
import asyncio
import websockets
import json
import base64

async def realtime_client(audio_stream):
    """
    Connect to vLLM Realtime API and process audio.

    Args:
        audio_stream: Iterator yielding audio chunks (bytes)
    """
    uri = "ws://localhost:8000/v1/realtime"

    async with websockets.connect(uri) as ws:
        # Create session
        await ws.send(json.dumps({
            "type": "session.create",
            "model": "mistralai/Voxtral-Mini-3B-Realtime-2602"
        }))

        # Wait for session confirmation
        response = await ws.recv()
        session = json.loads(response)
        print(f"Session created: {session['session_id']}")

        # Send audio chunks
        for audio_chunk in audio_stream:
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_chunk).decode()
            }))

            # Check for responses (non-blocking)
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=0.01)
                data = json.loads(response)
                if data["type"] == "response.text.delta":
                    print(data["delta"], end="", flush=True)
            except asyncio.TimeoutError:
                pass  # No response yet, continue sending audio

        # Signal end of input
        await ws.send(json.dumps({
            "type": "input_audio_buffer.commit"
        }))

        # Receive remaining responses
        async for message in ws:
            data = json.loads(message)
            if data["type"] == "response.text.delta":
                print(data["delta"], end="", flush=True)
            elif data["type"] == "response.done":
                break
```

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

# End-to-End Example: Voice Assistant

Combining streaming input and the Realtime API, we can build a complete voice assistant that responds to speech in real time.

```python
import asyncio
import json
import base64
import websockets
import pyaudio

class VoiceAssistant:
    """Real-time voice assistant using vLLM."""

    def __init__(self, server_url="ws://localhost:8000/v1/realtime"):
        self.server_url = server_url
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk_size = 1024

    async def run(self):
        """Start the voice assistant."""
        async with websockets.connect(self.server_url) as ws:
            # Create session
            await ws.send(json.dumps({
                "type": "session.create",
                "model": "mistralai/Voxtral-Mini-3B-Realtime-2602"
            }))

            # Run send and receive concurrently
            await asyncio.gather(
                self._send_audio(ws),
                self._receive_responses(ws)
            )

    async def _send_audio(self, ws):
        """Capture and send audio from microphone."""
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        print("Listening... (Ctrl+C to stop)")

        try:
            while True:
                # Read audio chunk
                chunk = stream.read(self.chunk_size, exception_on_overflow=False)

                # Send to server
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode()
                }))

                # Small delay to prevent overwhelming the connection
                await asyncio.sleep(0.01)

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    async def _receive_responses(self, ws):
        """Receive and display responses from the server."""
        async for message in ws:
            data = json.loads(message)

            if data["type"] == "response.text.delta":
                # Print text as it arrives
                print(data["delta"], end="", flush=True)

            elif data["type"] == "response.audio.delta":
                # Handle audio output (for TTS-capable models)
                audio_data = base64.b64decode(data["audio"])
                # Play audio_data through speakers...

            elif data["type"] == "error":
                print(f"\nError: {data['message']}")

async def main():
    assistant = VoiceAssistant()
    await assistant.run()

if __name__ == "__main__":
    asyncio.run(main())
```

This example demonstrates the power of combining vLLM's streaming capabilities with the Realtime API. The user speaks naturally, and responses begin appearing before they finish speaking.

## Performance Considerations

When deploying real-time applications, consider:

**Latency**: The primary goal is minimizing time-to-first-token. Streaming input reduces this by beginning processing before input is complete. In our testing, streaming pipelines can begin generating responses within 100-200ms of the first audio arriving, compared to seconds of delay when waiting for complete input.

**Memory**: Long-running sessions accumulate context. Configure `max_model_len` appropriately and consider implementing context windowing for very long sessions. For voice assistants, a 32K context window typically provides ample room for extended conversations.

**Concurrency**: WebSocket connections are persistent. Plan capacity based on expected concurrent sessions rather than requests per second. Each active session maintains state on the server, so monitor memory usage as concurrent connections scale.

**Audio Format**: The examples use 16kHz mono PCM audio. Ensure your audio capture matches the expected format for your model. Voxtral expects 16kHz sample rate with 16-bit signed integer samples.

**Chunk Size**: The audio chunk size affects latency and network efficiency. Smaller chunks (512-1024 samples) provide lower latency but higher overhead. Larger chunks (2048-4096 samples) are more efficient but add buffering latency. A chunk size of 1024 samples at 16kHz corresponds to 64ms of audio, which is a reasonable balance.

**Error Handling**: Real-time systems must handle network interruptions gracefully. Implement reconnection logic and consider buffering audio locally during brief disconnections to avoid losing user input.

# Conclusion

Streaming input and the Realtime API represent a significant expansion of vLLM's capabilities. Together, they enable a new class of low-latency applications that were previously impractical with batch-oriented inference.

## Summary

- **Streaming Input**: `AsyncLLM.generate()` now accepts an `AsyncGenerator` of `StreamingInput` objects, enabling incremental prompt processing
- **Realtime WebSocket API**: Production-ready bidirectional streaming interface for audio and text
- **KV Cache Efficiency**: Prefix caching ensures incremental input doesn't require recomputation

## Current Limitations

The initial release focuses on speech-to-text with streaming-capable models like Voxtral. Some features are not yet supported:

- Pooling models
- `n > 1` (multiple completions)
- `output_kind == FINAL_ONLY`
- Stop strings with streaming input

## Future Directions

We are actively developing:

- Expanded model support beyond Voxtral
- Additional modalities (video streaming, multimodal output)
- Integration with [vLLM-Omni](https://github.com/vllm-project/vllm-omni) for full omni-modal pipelines

## Get Involved

- [PR #28973 - Streaming Input Support](https://github.com/vllm-project/vllm/pull/28973)
- [PR #33187 - Realtime API](https://github.com/vllm-project/vllm/pull/33187)
- [vLLM Documentation](https://docs.vllm.ai)
- Join **#feat-realtime-api** on [slack.vllm.ai](https://slack.vllm.ai)

We welcome feedback and contributions as we continue to develop vLLM's real-time capabilities.
