---
layout: post
title: "Full-Duplex Conversations with vLLM-Omni"
author: "natureofnature"
summary: "How vLLM-Omni serves live voice and video conversations with a unified duplex session runtime, model-specific execution, streaming audio optimizations, and interaction-aware evaluation."
image: /assets/figures/2026-10-12-vllm-omni-unified-duplex/unified-duplex-architecture.svg
social_image: /assets/figures/2026-10-12-vllm-omni-unified-duplex/unified-duplex-architecture.svg
read_time_minutes: 13
tags:
  - vllm-omni
  - multimodal
  - realtime
---

Imagine asking a voice assistant to explain what is on your screen. Halfway through its answer, you point to a different object and say, “Actually, I meant this one.” A natural interaction requires more than fast speech synthesis: the system must keep receiving your voice and camera frames, preserve the conversation state, and let new information affect what happens next.

This is the serving problem behind full-duplex interaction. The input does not end when generation starts, and the conversation does not end when one response finishes.

vLLM-Omni's unified duplex framework brings these interactions into a common session runtime. It serves native duplex models such as **PersonaPlex** and **MiniCPM-o 4.5**, as well as models such as **AURA** and **Qwen3-Omni** that execute one turn at a time within an ongoing duplex session. The framework shares connection handling, session control, and stage execution while preserving the behavior each model actually supports.

## TL;DR

- **One session framework, different execution modes.** Native duplex models incorporate continuing input into live model execution. Turn-based models start each response with a new request built from the current turn and retained history.
- **Clear ownership from connection to computation.** The API Server handles communication; EngineClient connects applications to session handles; the Duplex Orchestrator coordinates session state and work; model stages perform inference.
- **Model behavior stays model-specific.** Plugins adapt input preparation, capabilities, and output interpretation. Model-driven speech, local playback stop, and server cancellation remain distinct operations.
- **Optimization targets the entire audio path.** Batching ready sessions, reusing execution graphs, reducing repeated codec work, and streaming stage outputs all matter. Evaluation must also measure audio continuity and interaction quality.

## From streaming replies to continuous interaction

Chat interfaces made request–response interaction familiar: submit an input and receive an answer. Voice interfaces added streaming audio, but many still followed the same pattern—finish speaking, submit the utterance, then play the response.

Streaming output improves the wait for the first sound. Native full duplex changes what can happen *during* that sound. The model continues processing incoming audio, and where supported video, while producing its own output. New input can affect subsequent speech, including whether to keep speaking or yield the floor. Already played audio cannot be taken back; the opportunity is to adapt what comes next.

The interface, interaction mode, and model execution describe different layers:

<iframe class="vllm-embed" src="/assets/interactive_pages/vllm-omni-duplex/interfaces.html" title="Interfaces, interaction modes and model execution: request-response and duplex session paths" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 1000px; border: 0; overflow: hidden;"></iframe>

*Figure 1. Follow an interface through its interaction and execution modes, then see example model profiles. Select a branch to highlight its path. A duplex session supports both native duplex and turn-based execution. [Open the interactive diagram](/assets/interactive_pages/vllm-omni-duplex/interfaces.html) or [view the static version](/assets/figures/2026-10-12-vllm-omni-unified-duplex/interfaces-interaction-execution.svg).*

A bidirectional connection is therefore useful for both execution modes. Qwen3-Omni with Server VAD can receive microphone audio during playback, stop a response, and start another after a new turn is committed. That provides an interruptible voice experience, while its model execution remains turn-based. PersonaPlex and MiniCPM-o 4.5 additionally support model-native processing of continuing input.

The public duplex surface is **`/v1/realtime?duplex=1`**, with **`/v1/duplex`** as an alias. Its message protocol builds on OpenAI Realtime event names and adds vLLM-Omni extensions for capabilities such as video input, session recovery, and playback reporting. Support for individual operations is advertised by the model's session capabilities; sharing an endpoint does not make every feature universal. See the [Realtime and Duplex API guide](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/docs/serving/realtime_duplex_api.md).

## A session runtime with clear responsibilities

A live conversation has several activities in flight: input arrives, inference produces output, the client plays audio, and the user may interrupt or disconnect. The unified framework gives these activities a common session identity and explicit owners.

<iframe class="vllm-embed" src="/assets/interactive_pages/vllm-omni-duplex/architecture.html" title="Inside a live duplex session: communication, session coordination, and model execution" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 960px; border: 0; overflow: hidden;"></iframe>

*Figure 2. Step through the input and output paths, or click a component to explore its role. Ordinary session output returns through a bounded per-session buffer. [Open the interactive diagram](/assets/interactive_pages/vllm-omni-duplex/architecture.html) or [view the static version](/assets/figures/2026-10-12-vllm-omni-unified-duplex/unified-duplex-architecture.svg).*

| Component | Responsibility |
|---|---|
| **API Server** | Read and validate WebSocket messages, translate them into commands, render outgoing events, and manage connection attachment and replay information. |
| **EngineClient** | Expose session handles to the application, coordinate open/close/resume operations, submit commands, and deliver each session's events to its consumer. |
| **Duplex Orchestrator** | Admit sessions, order session work, coordinate responses and playback state, and submit model work through the stage machinery. |
| **Model stages** | Encode inputs, run autoregressive or other generation steps, preserve model computation state where supported, and produce text, codec tokens, or PCM audio. |

Inside the orchestrator, **SessionManager** manages admission, session-to-runner mappings, input pressure, leases, and cleanup. Each session has a **SessionRunner** that processes an ordered mailbox of commands and stage outputs. **DuplexEngineSession** holds the session's turns, responses, epochs, request resources, and playback progress.

The **model plugin** supplies the model-specific rules: which capabilities are available, how to buffer and construct inputs, how to configure generation, and how to interpret outputs. For example, a MiniCPM `LISTEN` decision must be handled as a session decision rather than sent onward as text to synthesize.

These responsibilities also clarify the process boundary. In the local serving arrangement, the API side and the orchestrator run in the same frontend process, with the orchestrator on its own thread and event loop. Stage engines and GPU workers perform model computation in their execution processes. A logical box in the architecture is not automatically a separate process or GPU.

The common output path is deliberately short: a session emits an event into its bounded output buffer, its handle consumes the event, and the API sends it to the client. This avoids routing every audio chunk through a global event queue. Per-session output limits bound the audio backlog. If a slow consumer fills that buffer, the response fails with a backpressure error. The [framework design](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/docs/design/fullduplex.md) describes the lifecycle and ordering rules behind this separation.

## Four model paths and what they retain

The shared session runtime does not force every model into the same inference pipeline. These four paths illustrate both the differences in computation and the differences in conversation state.

<iframe class="vllm-embed" src="/assets/interactive_pages/vllm-omni-duplex/models.html" title="Speech execution and state: compare PersonaPlex, MiniCPM-o 4.5, AURA, and Qwen3-Omni" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 1000px; border: 0; overflow: hidden;"></iframe>

*Figure 3. Switch models and step through their encoders, autoregressive modules, waveform decoders, and retained state. Brackets group modules into serving stages: neural modules and autoregressive loops do not map one-to-one to stages. [Open the interactive diagram](/assets/interactive_pages/vllm-omni-duplex/models.html) or [compare all four models in the static version](/assets/figures/2026-10-12-vllm-omni-unified-duplex/model-stages-and-state.svg).*

### PersonaPlex: frame-by-frame speech interaction

PersonaPlex processes 24 kHz PCM audio in **80 ms model frames**. Client packets do not have to match that size: input buffering assembles complete frames for the model.

Stage 0 combines Mimi audio encoding with a Temporal Transformer and a Depformer. The Temporal Transformer advances across frames. The Depformer predicts the audio codebooks within a frame. Both are autoregressive, but they operate along different dimensions and are tightly coupled, so they belong to the same Talker stage. Stage 1 uses Mimi to decode the generated tokens into PCM.

Temporal context and streaming codec state carry forward across frames. The Depformer's short-lived KV state is rebuilt for each frame. This allows the model to keep listening as it produces speech without requiring a client commit or Server VAD to mark every turn. **80 ms describes the input processing cadence, not an end-to-end latency guarantee.** See the [PersonaPlex serving guide](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/examples/online_serving/personaplex/README.md).

### MiniCPM-o 4.5: audio and video with model-driven speech decisions

MiniCPM-o 4.5 accumulates PCM audio and associated video frames into input chunks, typically around one second. Its first stage combines audio and visual encoding with language-model execution. A `LISTEN` result keeps the conversation listening; a `SPEAK` result forwards text and model features to the Talker.

The Talker predicts audio tokens. Code2Wav then uses conditional flow matching, implemented with a DiT, followed by a HiFT vocoder to produce PCM. Continuing input reuses the supported LLM and audio-processing state instead of reconstructing every turn from scratch. Context remains bounded by the model and serving configuration; KV reuse is not unlimited conversation memory. The [MiniCPM pipeline](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py) separates understanding, speech-token generation, and waveform synthesis into three stages.

### AURA and Qwen3-Omni: new requests from retained history

AURA follows a four-stage path: **ASR → Thinker → Talker → Code2Wav**. ASR converts user audio to text, while video goes to the visual Thinker. The Thinker-side session history retains transcripts, video clips, and assistant text for later turns.

Qwen3-Omni instead feeds audio features from its AuT encoder and visual features into the Thinker, followed by Talker and Code2Wav. Its duplex adapter retains multimodal conversation content and prepares subsequent requests from that history.

Both maintain a live session. Each response starts a new inference request built from the current turn and retained history. Committing input and requesting a response can be separate operations. This differs from appending input to a native duplex model's continuing computation. Prefix reuse, when applicable, is a separate optimization and does not change that distinction. The implementations are described by the [AURA pipeline](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/vllm_omni/model_executor/models/aura_omni/pipeline.py) and [Qwen3-Omni duplex adapter](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/vllm_omni/model_executor/models/qwen3_omni/duplex/plugin.py).

## Model decisions, playback stop, and server cancellation

“Interruptible” can describe several different behaviors. Applications need to distinguish them because they affect different parts of the conversation.

| Mechanism | What changes | Example |
|---|---|---|
| **Model-driven speech** | The model responds to continuing input and determines its subsequent speech or silence. | PersonaPlex; MiniCPM-o `LISTEN` / `SPEAK`. |
| **Local playback stop** | The client stops playing queued audio. Inference can continue unless separately cancelled. | Holding the talk button in the AURA demo. |
| **Server cancellation** | The serving runtime cancels the active generation. The client must also handle queued audio appropriately. | Qwen3-Omni with Server VAD and `interrupt_response` enabled. |

Acoustic VAD answers whether speech is present and uses a silence policy to help decide when to commit a turn. It can also trigger cancellation when a user begins speaking. It does not, by itself, establish that the user's meaning is complete or that an utterance is intended to interrupt.

Model-driven overlap also does not imply arbitrary state rewind. In the current PersonaPlex integration, explicit cancellation resets conversation context, and a dropped connection requires a new session. MiniCPM-o 4.5, AURA, and Qwen3-Omni expose session resume, subject to the existing session remaining alive and retained events being available for replay. Resume reattaches a connection; it is not recovery from a destroyed engine or restarted frontend. These differences are part of the [advertised session capabilities](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/docs/serving/realtime_duplex_api.md), rather than assumptions a client must infer from the model name.

## Optimizing the live audio path

Full duplex repeatedly performs small pieces of work under a timing budget. A short audio frame may need encoding, model execution, token prediction, waveform decoding, and delivery before the next frames create a backlog. Optimizing only the largest Transformer misses much of that path.

### Batch ready sessions together

Several active conversations produce similar work at nearby times. Processing ready frames together amortizes per-call overhead and gives the GPU larger operations. PersonaPlex includes batched input encoding and per-frame code prediction while retaining separate session state. Batching improves utilization, but an interactive scheduler must also avoid waiting too long to assemble a batch.

### Reuse prepared execution and reduce small operations

CUDA Graphs reduce repeated CPU kernel-launch work for supported, stable execution shapes. Model-specific graph capture is especially useful for repeated audio-frame operations. Kernel fusion addresses a related cost: several small operations can otherwise require separate launches and intermediate memory traffic.

These techniques apply at different points in different models. PersonaPlex has model-local Mimi frame graphs, while MiniCPM's Code2Wav path contains graph execution and fused DiT work. Capture policy, batch shapes, precision, and memory budgets still need to match the stage and hardware. The [PersonaPlex deploy profile](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/vllm_omni/deploy/personaplex.yaml) and [MiniCPM execution profiles](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/docs/design/minicpm_o45_mrv2_performance.md) expose those choices.

### Keep codec state and process new audio

A streaming codec should carry the context needed for the next chunk instead of repeatedly decoding the entire conversation prefix. PersonaPlex's Mimi implementation preserves convolution and attention state across frames. This reduces repeated work while maintaining the context required at audio boundaries. Closing a session also releases its codec state so another conversation can use the capacity. See the [streaming Mimi implementation](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/vllm_omni/model_executor/models/personaplex/personaplex_mimi.py).

### Start downstream work as soon as useful input is available

Speech pipelines can overlap work when a stage forwards a usable chunk instead of waiting for the whole response. The appropriate boundary is model-dependent: it may be a completed model segment, a set of codec tokens, or one codec frame. Transfer cadence and the decoder's context window are separate decisions; making a network packet smaller does not automatically make the decoder faster or preserve audio quality.

This is also a central lesson of [Engineering TTS Inference in vLLM-Omni](https://vllm-project.github.io/2026/06/23/vllm-omni-tts.html): first-audio latency, decoder continuity, and stage throughput need to be optimized together.

## Measure first audio, continuity, and interaction quality

For a live conversation, a fast first response is only part of the experience. Speech must arrive continuously, and the assistant must respond at the right time as new input arrives. Three complementary views capture these requirements:

| View | Measure | Why it matters |
|---|---|---|
| **First response** | Text TTFT; audio TTFP; conversational response timing | The first packet's arrival and the decision to answer are different events. Define the starting point for each measurement. |
| **Sustained audio** | Generation RTF, chunk-arrival intervals, playback underruns | Good first-audio latency does not guarantee continuous playback. |
| **Conversation under load** | Session count and duration, memory use, cleanup, task quality | A brief single-session demo does not establish long-session or concurrent serving capacity. |

For generation measurements, **RTF = generation time / generated audio duration**, so lower is better and a value below one indicates faster-than-real-time generation for that measured interval. Output audio duration divided by elapsed time is an audio-throughput measure; it is the reciprocal only when both measurements use the same scope and aggregation.

A paced duplex benchmark has a different clock. If it includes the time spent sending a real-time input stream, its client-observed RTF measures the session's delivery behavior, not just inference speed. Record that timing window explicitly, and distinguish warm runs from cold starts. The [PersonaPlex load-test guide](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/docs/design/fullduplex-personaplex.md) documents these metrics and fixed-window acceptance checks.

vLLM-Omni also includes [**OmniInteract**](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/vllm_omni/benchmarks/omniinteract.py) and **Omni-DuplexEval** integrations to evaluate what the assistant understands and how it interacts with streamed inputs. These complement serving metrics: a system can emit audio quickly yet answer at the wrong time or miss an event in the video. Model adapters and task coverage determine which evaluations apply; the [Omni-DuplexEval CI guide](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/tests/e2e/accuracy/omni_duplex_eval/README.md) provides a concrete MiniCPM-o 4.5 evaluation path.

## Try a native duplex session

With vLLM-Omni installed, a supported GPU, and access to the gated PersonaPlex checkpoint, start the server from the repository checkout:

```bash
python -m vllm_omni.entrypoints.cli.main serve \
  /path/to/personaplex-7b-v1 \
  --omni \
  --deploy-config vllm_omni/deploy/personaplex.yaml
```

The shipped profile enables duplex serving. Connect through `ws://127.0.0.1:8000/v1/realtime?duplex=1`, select a bundled voice and persona, and continuously stream audio. The Python client preset prepares the expected 24 kHz PCM session configuration; the model supplies the speech behavior without client commits. Follow the [PersonaPlex quickstart](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/examples/online_serving/personaplex/README.md) for the client code and validation command.

## Next steps for the framework

The common session runtime provides a foundation for broader model support and richer interactions. The next areas of work are:

- **Broader unified tool calling.** [Qwen3-Omni](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/vllm_omni/entrypoints/openai/realtime/connection.py) has tool-call parsing on its separate turn-mode Realtime service (`/v1/realtime?duplex=0`), while [Nemotron VoiceChat](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/vllm_omni/model_executor/models/nemotron_voicechat/duplex/plugin.py) supports calls and results in unified native duplex. Extend unified support across models while handling interruptions, timeouts, and duplicate or late results without applying an old result to a new turn.
- **Semantic turn completion.** Add meaning-aware endpointing alongside acoustic speech detection, and integrate it with commit and response policies. A pause in “Book a flight to… Beijing tomorrow” should not always end the turn.
- **JoyAI and Gander integration.** Extend the unified duplex framework with model adapters for JoyAI and [Gander](https://github.com/Omni-Interaction-Gander/Omni-Interaction-Agent). For Gander, connect continuous audio-visual interaction with asynchronous agent work, so users can ask follow-up questions or redirect a task while it runs.
- **Longer conversations and broader scale.** Continue work on bounded context, predictable memory use, multi-session execution, and routing beyond one API worker, building on the existing stage machinery.
- **Broader interaction evaluation.** Expand model and task coverage, including interruptions, long audio/video sessions, and regressions in both quality and responsiveness.

Full-duplex serving makes the conversation itself the unit of coordination. vLLM-Omni brings that coordination together with model-specific streaming execution and stage-level optimization, so applications can build toward assistants that keep listening, respond naturally, and adapt as the interaction unfolds.

The work is developed in the open. Join the discussion in [vLLM-Omni](https://github.com/vllm-project/vllm-omni), try a session with your workload, and contribute model adapters, evaluation cases, or performance improvements.

## References

- [Full-Duplex Serving Roadmap and PR Status — RFC #6494](https://github.com/vllm-project/vllm-omni/issues/6494): the roadmap and development history behind the duplex serving work.
- [Design of Unified Full-duplex Framework — RFC #7181](https://github.com/vllm-project/vllm-omni/issues/7181): the motivation for separating communication, session coordination, and model execution.
- [Full-duplex framework design](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/docs/design/fullduplex.md): session ownership, command ordering, cancellation, and stage coordination in the implementation discussed here.
- [Realtime and Duplex API guide](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/docs/serving/realtime_duplex_api.md): endpoints, messages, model capabilities, playback feedback, and session recovery.

- [PersonaPlex serving guide](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/examples/online_serving/personaplex/README.md) and [native-duplex design and load testing](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/docs/design/fullduplex-personaplex.md): streaming execution, state lifetime, deployment, and measurement.
- [MiniCPM-o 4.5 pipeline](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py) and [execution and performance design](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/docs/design/minicpm_o45_mrv2_performance.md): model stages, retained state, and implementation-specific optimization.
- [AURA integration — PR #7633](https://github.com/vllm-project/vllm-omni/pull/7633) and [AURA pipeline](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/vllm_omni/model_executor/models/aura_omni/pipeline.py): the ASR–Thinker–Talker–Code2Wav path and its duplex integration.
- [Qwen3-Omni duplex adapter](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/vllm_omni/model_executor/models/qwen3_omni/duplex/plugin.py): turn-based execution and input/history preparation within a duplex session.
- [OmniInteract implementation](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/vllm_omni/benchmarks/omniinteract.py) and [Omni-DuplexEval integration guide](https://github.com/vllm-project/vllm-omni/blob/c10115052ef691e4d0270f1ddf5bce5bc4d140f8/tests/e2e/accuracy/omni_duplex_eval/README.md): evaluation of streamed multimodal inputs and conversational behavior.

- [Silero VAD](https://github.com/snakers4/silero-vad): acoustic speech-activity detection.
- [Pipecat Smart Turn](https://github.com/pipecat-ai/smart-turn) and [LiveKit turn detection](https://docs.livekit.io/agents/logic/turns/turn-detector/): turn-completion detection and its combination with acoustic VAD.
- [Gander](https://github.com/Omni-Interaction-Gander/Omni-Interaction-Agent) and [vLLM-Omni community roadmap #8264](https://github.com/vllm-project/vllm-omni/issues/8264): continuous multimodal interaction, asynchronous agent work, and proposed integration priorities.

- [vLLM Office Hours #57 — vLLM-Omni Project Update & Demos](https://docs.google.com/presentation/d/1L84FWrGAI-fns6TaR0Q8R6I9CvbrWQvbVbGl_KtXhy8/edit): the September 2026 presentation, including the v0.28 duplex update on slide 19.
- [Streaming Requests & Realtime API in vLLM](https://vllm-project.github.io/2026/01/31/streaming-realtime.html): incremental input processing and the Realtime transport foundation.
- [Engineering TTS Inference in vLLM-Omni](https://vllm-project.github.io/2026/06/23/vllm-omni-tts.html): staged speech generation, batching, CUDA Graphs, and model-specific kernels.
- [Experience and Lessons Learned from Serving Multi-Stage Qwen3-Omni in vLLM-Omni](https://vllm-project.github.io/2026/07/01/qwen3-omni-optimization.html): stage-level optimization methods and performance measurement.
- [Announcing vLLM-Omni](https://vllm-project.github.io/2025/11/30/vllm-omni.html): the project's original multimodal and multi-stage serving architecture.

<script>
(function () {
  window.addEventListener('message', function (event) {
    var data = event.data;
    if (event.origin !== window.location.origin || !data || data.type !== 'vllm-embed-resize' || !Number.isFinite(data.height)) return;
    document.querySelectorAll('iframe.vllm-embed').forEach(function (frame) {
      if (frame.contentWindow === event.source) {
        frame.style.height = Math.max(400, Math.min(2400, Math.ceil(data.height))) + 'px';
      }
    });
  });
})();
</script>
