---
layout: post
title: "Encoder Disaggregation in vLLM: Scaling Multimodal Serving with EPD"
summary: "Recent advances in vLLM encoder disaggregation: metadata-only multimodal requests, dedicated encoder execution, remote embedding transfer, cross-encoder caching, and lower-overhead EPD routing."
image: /assets/figures/2026-10-18-encoder-disaggregation-epd/01-epd-architecture.png
social_image: /assets/figures/2026-10-18-encoder-disaggregation-epd/01-epd-architecture.png
tags:
  - multimodal
  - disaggregation
  - performance
---

Multimodal inference adds an encoding stage before language-model execution. Images, video frames, and audio must be processed into embeddings before the language model can use them. The demand on this stage varies with media size, modality, and request mix, and can change independently of prefill and decode demand.

Encoder / Prefill / Decode (EPD) disaggregation separates these stages into independently deployed vLLM instances. Over the past two months, the vLLM community has expanded EPD across model execution, remote embedding transfer, shared caching, multimodal input handling, and request routing.

This post explains the engineering behind that progress, with a focus on four changes:

- **Dedicated encoder execution** with a separate Model Runner V2 path, encoder-only CUDA Graph support, and reduced checkpoint-loading I/O.
- **Metadata-only requests** that let downstream instances consume remote embeddings without repeating media preprocessing.
- **Remote transfer and shared output reuse** through NIXL and Mooncake integrations.
- **Lower orchestration overhead** in the Python EPD proxy and EPD support in vLLM Router.

The [EPD tracker](https://github.com/vllm-project/vllm/issues/52409) links the ongoing work. Features described as merged are available on the corresponding repository's main branch; availability in an installed release depends on its version.

## Why Separate the Encoder?

A multimodal request typically passes through three stages:

1. **Encoder (E):** preprocess media and produce multimodal embeddings.
2. **Prefill (P):** process the prompt and multimodal embeddings, building the state needed for generation.
3. **Decode (D):** generate output tokens using that state.

EPD supports two deployment patterns: **E+PD**, with a separate encoder and a combined prefill/decode instance, and **E/P/D**, with all three stages deployed separately.

![Encoder, prefill, and decode deployment patterns](/assets/figures/2026-10-18-encoder-disaggregation-epd/01-epd-architecture.png)

*Figure 1. E+PD separates encoding from language-model execution. E/P/D additionally separates prefill and decode. EC connectors deliver encoder embeddings; KV connectors transfer the state between prefill and decode.*

An image-heavy workload may need more encoder capacity without a corresponding increase in decode capacity. A workload with longer responses may require the opposite. Separating the stages allows operators to adjust their instance counts and execution settings independently, while moving encoder computation off the GPUs serving the language model.

The benefit depends on the workload. Remote delivery adds communication and coordination to the request path, so deployment choices should account for media processing, network latency, cache reuse, and the resource balance across stages.

## Metadata-Only Requests: Avoiding Duplicate Preprocessing

Transferring embeddings solves only part of the handoff. The consumer also needs enough information to reconstruct the multimodal input layout: placeholder lengths, image grids, feature lengths, and identifiers linking those positions to the transferred embeddings.

A straightforward EPD implementation sends the original media to the encoder, transfers the resulting embeddings through an EC connector, and forwards the original request to P/PD. This can work correctly, but the consumer may still fetch, decode, resize, normalize, or patchify the media to recover its input layout. The encoder has already performed that work.

**A metadata-only request replaces the downstream media item with a reference containing the structural information already produced on the encoder side.** The consumer reconstructs placeholders from that metadata and receives the actual embeddings through the EC connector.

“Metadata-only” applies to the multimodal input item in the downstream request. The text prompt and generation settings remain present. Embeddings are still required; their payload travels through the separate EC data path.

![Raw-media forwarding compared with metadata-only requests](/assets/figures/2026-10-18-encoder-disaggregation-epd/02-control-and-data.png)

*Figure 2. Both paths can reuse remote embeddings. Metadata-only forwarding additionally avoids downstream media preprocessing performed solely to reconstruct the input layout. The router/proxy handles request orchestration; EC connectors deliver embedding tensors.*

For example, Qwen-family image inputs require grid information such as `image_grid_thw` to construct the appropriate multimodal layout. The encoder can publish the grid it already computed, allowing P to use it directly. The earlier image-path work in [#50390](https://github.com/vllm-project/vllm/pull/50390) established this approach.

| Comparison | Raw media forwarded to P | Metadata-only |
| --- | --- | --- |
| Downstream media item | Image URL, base64 data, or another raw-media input | Input identity and model-required structural metadata |
| Input-layout reconstruction | Media processing, or reuse of cached preprocessing results | Placeholder construction from encoder-published metadata |
| Embedding source | EC connector; an available remote embedding avoids another encoder forward | EC connector |
| Main overhead or requirement | Potentially duplicated fetching, decoding, and preprocessing | A metadata contract consistent with the transferred embedding and input layout |

Without metadata-only forwarding, the encoder neural network does not necessarily run twice: the remote embedding may already satisfy that dependency. The remaining duplication is often in media processing. Its cost depends on input size, preprocessing-cache hits, and how media is delivered. Forwarding a URL, for example, may cause the consumer to fetch the media again rather than sending its bytes through the proxy.

### JSON Metadata and Audio Support

Metadata-only forwarding and metadata serialization address different costs. [#56090](https://github.com/vllm-project/vllm/pull/56090) allows model-declared metadata fields to be represented as numeric JSON arrays, avoiding PyTorch tensor serialization and base64 conversion for small structural fields. This also gives the router and other frontends a language-neutral representation.

The audio path follows the same principle. [#57887](https://github.com/vllm-project/vllm/pull/57887) lets the encoder publish audio token counts and required feature lengths. The consumer uses them to rebuild placeholders without repeating audio preprocessing.

The implementation covers the relevant Qwen2-Audio, AudioFlamingo3, Ultravox, Qwen2.5-Omni, and Qwen3-Omni paths for pure-audio Chat Completions requests. It does not introduce realtime or combined audiovisual input support.

![Audio metadata-only throughput and latency results](/assets/figures/2026-10-18-encoder-disaggregation-epd/04-audio-results.png)

*Figure 3. Results reported in #57887 on MMAU test-mini: 1,000 samples over two passes, Qwen3-Omni-30B-A3B-Instruct, BF16, TP=1, Model Runner V2, Mooncake TCP, 1E+1PD, concurrency 64, and CUDA Graphs enabled.*

In this evaluation, metadata-only forwarding increased throughput from **62.21 to 73.13 requests/s** and reduced mean latency from **980 to 853 ms**. Accuracy changed from 75.90% to 75.65%; that difference remains to be investigated. The comparison measures two audio input paths within EPD, rather than EPD against colocated serving.

Video processing has also progressed. [#54242](https://github.com/vllm-project/vllm/pull/54242) adds video embedding inputs, while [#53675](https://github.com/vllm-project/vllm/pull/53675) enables GPU NVDEC media processing on encoder-only instances. In the latter PR's 30-frame MMVU workload, end-to-end throughput improved by approximately 94–115% across three GB200 nodes. Each run used Qwen3.5-35B-A3B, a 1E+3PD topology, eager execution, and concurrency 64. The gains include both GPU preprocessing and hardware video decoding; as the encoder became faster, the PD tier became the bottleneck.

## A Dedicated Execution Path for Encoder Instances

An encoder-only instance runs no language model, maintains no language-model KV cache, and produces no sampled tokens. Its execution lifecycle should reflect those differences.

[#53176](https://github.com/vllm-project/vllm/pull/53176) consolidates Model Runner V2's encoder-only behavior in `MMEncoderModelRunner`. This removes encoder-specific branches from the shared runner and addresses a synchronization issue exposed by the different output lifecycle.

In ordinary generation, returning sampled outputs includes a device wait associated with copying results to the CPU. An encoder-only step can instead construct an empty runner output on the host. Without an equivalent synchronization point, the host can advance far enough to recycle a pooled host buffer while a GPU kernel is still using it, causing an illegal memory access. The dedicated runner places the required synchronization in the encoder-only path.

Related changes make encoder execution more consistent:

- **Configuration normalization:** producer-only EC instances automatically enter encoder-only mode. [#53656](https://github.com/vllm-project/vllm/pull/53656)
- **Correct asynchronous execution:** the engine uses `is_mm_encoder_only` to determine whether a step needs the subsequent sampling path. This fixes standalone `--mm-encoder-only` instances incorrectly entering `sample_tokens()` when no EC connector is configured. [#58490](https://github.com/vllm-project/vllm/pull/58490)
- **Reduced startup I/O:** encoder-only loading skips pure language-model checkpoint shards when the weight layout can be classified safely. This reduces unnecessary checkpoint reads, in addition to the existing savings from not constructing the language model. [#58086](https://github.com/vllm-project/vllm/pull/58086)
- **Encoder CUDA Graph capture:** the encoder-only V2 runner supports opt-in ViT graph capture using the existing budget selection, replay, and eager fallback mechanisms. [#56922](https://github.com/vllm-project/vllm/pull/56922)

Encoder graph validation remains sensitive to the model, shapes, and attention backend. The CUDA Graph PR documents backend-specific NaN observations and comparison tests; enabling graph capture should be accompanied by correctness checks for the chosen configuration.

## Remote Embedding Transfer and Failure Isolation

`ECExampleConnector` provides a shared-filesystem path for validating EPD. Cross-node deployments need network transport, destination-buffer management, and clear ownership of in-flight outputs.

Two remote-transfer paths have landed:

| Backend | Capability | Implementation |
| --- | --- | --- |
| NIXL + CPU EC | Shares CPU-offloaded encoder outputs between instances using NIXL transfers between registered CPU mmap regions | [#47941](https://github.com/vllm-project/vllm/pull/47941) |
| Mooncake EC | Delivers encoder outputs through Mooncake TransferEngine, with TCP and RDMA configuration paths | [#41567](https://github.com/vllm-project/vllm/pull/41567), [usage guide](https://docs.vllm.ai/en/latest/features/mooncake_ec_connector_usage/) |

`ECMooncakeConnector` transfers encoder outputs. The separate `MooncakeConnector` serves the corresponding KV-transfer path between P and D.

### Transfer Shapes Must Match Encoder Outputs

Transfer buffers may need to be allocated before an item is encoded. Their size cannot be inferred reliably from a single language-model hidden-size field: different modalities and model architectures can produce different encoder output widths.

For example, the Qwen3-Omni checkpoint tested in [#57904](https://github.com/vllm-project/vllm/pull/57904) produces visual outputs of width 8192, including DeepStack information, and audio outputs of width 2048. That change measures output widths per modality during producer initialization and carries the actual shape through the transfer protocol. It also updates the NIXL EC protocol version, requiring producers and consumers to be upgraded together.

### A Failed Transfer Should Fail the Affected Request

Metadata-only requests make failure handling especially important. A consumer with the original media may be able to fall back to local encoding. A consumer holding only metadata lacks that input.

[#55290](https://github.com/vllm-project/vllm/pull/55290) handles this distinction: when remote encoding cannot arrive and local recovery is unavailable, it reports the affected request as unavailable instead of taking an unsafe fallback path that can terminate EngineCore. Resource-lifecycle fixes, including releasing shared memory after NIXL setup failures in [#59875](https://github.com/vllm-project/vllm/pull/59875), further improve failure handling.

## Reusing Outputs Across Encoder Instances

Instance-local caching leaves a reuse gap. An image encoded by E0 may later be routed to E1 after load balancing, scaling, or recovery. If E1 has no cached output, it can repeat an encoder forward that another instance has already completed.

[#56242](https://github.com/vllm-project/vllm/pull/56242) adds cross-encoder output reuse with Mooncake Store. On a local miss, the encoder checks the shared Store. A hit supplies the embedding without another forward; a miss follows normal encoding and asynchronously publishes the output. Delivery to the selected P instance continues through Mooncake P2P.

![Cross-encoder embedding reuse with Mooncake Store](/assets/figures/2026-10-18-encoder-disaggregation-epd/03-cross-encoder-cache.png)

*Figure 4. Mooncake Store provides shared output reuse across encoders. ECMooncakeConnector retains the P2P delivery path to the selected consumer.*

Shared reuse validates input identity, model identity, and the output tensor contract. Initial support includes images, Model Runner V2, encoder TP=1, and no dynamic LoRA. Publication is asynchronous, so concurrent cold requests can still duplicate computation. See the [cross-encoder reuse guide](https://docs.vllm.ai/en/latest/features/cross_encoder_cache/) for configuration and supported combinations.

A controlled reassignment experiment in the PR used six H20 GPUs, InternVL3_5-38B-HF, a 2E+4PD topology, and same-node Mooncake TCP. After cache population, images were reassigned between encoders to create local misses. Across 128 measured requests containing 512 images, shared reuse reduced repeated image encodings from **512 to zero**. TTFT p50 decreased from **1.209 to 0.819 seconds**, and p90 from **1.656 to 0.891 seconds**, with essentially unchanged TPOT.

This experiment used one run per configuration and a controlled full-reassignment workload; the performance experiment was not repeated after the final refactor. Production savings depend on reuse frequency and the additional Store memory and I/O costs.

Preprocessing caches, encoder-output caches, and prefix KV caches serve different stages. Reusing one does not imply that the next stage's result is available.

## Reducing Proxy Overhead on the Critical Path

EPD also depends on the CPU work that connects its stages. The proxy distributes media, collects encoder metadata, rewrites consumer requests, and forwards responses. Multi-image requests may generate multiple encoder subrequests, multiplying orchestration work.

As model execution becomes faster—or a cache hit removes encoding entirely—request parsing, serialization, and forwarding can account for a larger share of latency. Under concurrency, that CPU work can accumulate into queueing and limit how quickly requests reach otherwise available GPU workers. **Independent scaling of the compute stages requires an orchestration path that can keep pace.**

Metadata-only requests reduce the amount of media carried downstream. JSON metadata in [#56090](https://github.com/vllm-project/vllm/pull/56090) removes tensor serialization and base64 conversion for small structural fields. [#56657](https://github.com/vllm-project/vllm/pull/56657) then reduces work inside the Python EPD proxy itself:

- Use `msgspec` for request and E/P metadata JSON encoding and decoding.
- Forward final non-streaming response bytes without parsing and re-encoding the JSON body.
- Forward streaming response bytes directly, removing per-chunk text decoding and using `iter_any()` to receive available data.

Routing and metadata inspection still require structured processing. Final responses that are ready for the client can pass through with fewer conversions. Eliminating this repeated work matters across both high request rates and long streaming responses, even when the GPU execution path is unchanged.

### EPD Orchestration in vLLM Router

The Python example proxy supports launcher-managed dynamic registration through [#54176](https://github.com/vllm-project/vllm/pull/54176). Registered roles form either E+PD or E/P/D topologies, with health probes and recovery handling. Instance startup and autoscaling remain the external launcher's responsibility.

[vLLM Router #266](https://github.com/vllm-project/router/pull/266) brings EPD orchestration to the native router through `--epd-config`. The router selects P/PD before encoding, distributes media to encoder workers, and forwards JSON metadata references. EC backends handle embedding delivery, while separate P/D deployments reuse the existing KV handoff.

The initial integration is experimental and supports the native CLI, static worker configuration, and DP=1. The [router EPD guide](https://github.com/vllm-project/router/blob/main/docs/epd.md) describes its current scope and backend constraints.

## Getting Started

Start with E+PD to validate the encoder-to-consumer handoff before adding a separate decode tier. The [Disaggregated Encoder guide](https://docs.vllm.ai/en/latest/features/disagg_encoder/) links the example deployments and integration tests.

With a compatible vLLM checkout, Mooncake installed, and two available GPUs, the following integration script runs a baseline and compares outputs with a disaggregated deployment:

```bash
GPU_E=0 GPU_PD=1 MOONCAKE_EC_PROTOCOL=tcp \
  bash tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh
```

Use the [Mooncake EC guide](https://docs.vllm.ai/en/latest/features/mooncake_ec_connector_usage/) for transport dependencies and consumer control-address configuration, or the [CPU EC guide](https://docs.vllm.ai/en/latest/features/ec_cpu_connector/) for CPU-offloading and NIXL-based sharing. Choose the E/PD resource split using the target media mix, cache behavior, and latency requirements.

## What's Next

Recent work has expanded EPD beyond its core execution and transport paths. MiMo-V2-Omni now declares the embedding fields needed for image EPD in [#58938](https://github.com/vllm-project/vllm/pull/58938), while XPU support includes EC integration and an encoder-only producer runner in [#49532](https://github.com/vllm-project/vllm/pull/49532) and [#59320](https://github.com/vllm-project/vllm/pull/59320).

Further work includes observability, additional transport and cache backends, frontend integration, and clearer upstream error propagation. Operators need to distinguish time spent computing embeddings, loading cached outputs, and transferring them remotely.

Another opportunity is coordinating encoder work with prefill-side reuse. In an E-first pipeline, encoding can begin before P discovers that its KV cache already covers a multimodal occurrence, or that a reusable embedding is available locally. [RFC #55639](https://github.com/vllm-project/vllm/issues/55639) explores how P's demand could suppress unnecessary upstream work. The design must balance saved computation and delivery against additional lookup, coordination, and cold-request latency.
