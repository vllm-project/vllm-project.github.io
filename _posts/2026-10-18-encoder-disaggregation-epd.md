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

DeepSeek's [inference deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md) describes an E/P/D architecture with a separately deployed vision encoder. vLLM has supported encoder disaggregation since [our December 2025 architecture post](https://vllm.ai/blog/2025-12-15-vllm-epd), and continues to improve that support. DeepSeek V4.1 Flash can use vLLM's raw-image EPD path; the experiment below evaluates a metadata-only extension to that path.

This post explains the engineering behind that progress, with a focus on five changes:

- **Metadata-only requests** that let downstream instances consume remote embeddings without repeating media preprocessing.
- **Dedicated encoder execution** with a separate Model Runner V2 path, encoder-only CUDA Graph support, and reduced checkpoint-loading I/O.
- **Remote embedding transfer** through NIXL and Mooncake integrations.
- **Cross-encoder output reuse** through Mooncake Store.
- **Lower orchestration overhead** in the Python EPD proxy and EPD support in vLLM Router.

The [EPD tracker](https://github.com/vllm-project/vllm/issues/52409) links the ongoing work. The core vLLM capabilities described below, including MiMo and XPU support, are included in vLLM v0.32.0. The DeepSeek metadata-only experiment instead uses main plus [#62802](https://github.com/vllm-project/vllm/pull/62802). Router EPD requires router main with [#266](https://github.com/vllm-project/router/pull/266).

## Why Separate the Encoder?

A multimodal request typically passes through three stages:

1. **Encoder (E):** preprocess media and produce multimodal embeddings.
2. **Prefill (P):** process the prompt and multimodal embeddings, building the state needed for generation.
3. **Decode (D):** generate output tokens using that state.

EPD supports two deployment patterns: **E+PD**, with a separate encoder and a combined prefill/decode instance, and **E/P/D**, with all three stages deployed separately.

![Encoder, prefill, and decode deployment patterns](/assets/figures/2026-10-18-encoder-disaggregation-epd/01-epd-architecture.png)

*Figure 1. E+PD separates encoding from language-model execution. E/P/D additionally separates prefill and decode. EC connectors deliver encoder embeddings; KV connectors transfer the state between prefill and decode.*

An image-heavy workload may need more encoder capacity without a corresponding increase in decode capacity. A workload with longer responses may require the opposite. Separating the stages allows operators to adjust their instance counts and execution settings independently, while moving encoder computation off the GPUs serving the language model.

## Metadata-Only Requests: Avoiding Duplicate Preprocessing

Transferring embeddings solves only part of the handoff. The consumer also needs enough information to reconstruct the multimodal input layout: placeholder lengths, image grids, feature lengths, and identifiers linking those positions to the transferred embeddings.

A straightforward EPD implementation sends the original media to the encoder, transfers the resulting embeddings through an EC connector, and forwards the original request to P/PD. This can work correctly, but the consumer may still fetch, decode, resize, normalize, or patchify the media to recover its input layout. The encoder has already performed that work.

**A metadata-only request replaces the downstream media item with a reference containing the structural information already produced on the encoder side.** The consumer reconstructs placeholders from that metadata and receives the actual embeddings through the EC connector.

Only the multimodal item changes: the text prompt and generation settings remain, and the embeddings still travel through the EC data path.

![Raw-media forwarding compared with metadata-only requests](/assets/figures/2026-10-18-encoder-disaggregation-epd/02-control-and-data.png)

*Figure 2. Both paths can reuse remote embeddings. Metadata-only forwarding additionally avoids downstream media preprocessing performed solely to reconstruct the input layout. The router/proxy handles request orchestration; EC connectors deliver embedding tensors.*

For example, Qwen-family image inputs require grid information such as `image_grid_thw` to construct the appropriate multimodal layout. The encoder can publish the grid it already computed, allowing P to use it directly.

| Comparison | Raw media forwarded to P | Metadata-only |
| --- | --- | --- |
| Downstream media item | Image URL, base64 data, or another raw-media input | Input identity and model-required structural metadata |
| Input-layout reconstruction | Media processing, or reuse of cached preprocessing results | Placeholder construction from encoder-published metadata |
| Embedding source | EC connector; an available remote embedding avoids another encoder forward | EC connector |
| Main overhead or requirement | Potentially duplicated fetching, decoding, and preprocessing | A metadata contract consistent with the transferred embedding and input layout |

Without metadata-only forwarding, the encoder does not necessarily run twice, since the remote embedding already covers it. The duplication is in media processing, and forwarding a URL can also make the consumer fetch the media again.

### Metadata Across Images, Video, and Audio

The Python EPD path supports metadata-only forwarding for images, video, and audio in supported models. All three use the same handoff: the consumer reconstructs the input layout from encoder-published metadata while receiving the embeddings through an EC connector. The required fields depend on the model and modality:

- **Images:** image grids such as `image_grid_thw` let the consumer construct the visual placeholder range without processing the pixels again.
- **Video:** the frontend accepts `video_embeds` references with the metadata needed to reconstruct the video layout. For Qwen3-VL, this includes `video_grid_thw` and the encoder-published timestamps, which affect the prompt's token layout.
- **Audio:** audio token counts and required feature lengths let the consumer rebuild placeholders without repeating audio preprocessing.

Metadata serialization is a separate optimization. Model-declared metadata fields can now be represented as numeric JSON arrays, avoiding PyTorch tensor serialization and base64 conversion for small structural fields. This also gives the router and other frontends a language-neutral representation.

The audio evaluation below illustrates the benefit of this handoff. The audio change covers the relevant Qwen2-Audio, AudioFlamingo3, Ultravox, Qwen2.5-Omni, and Qwen3-Omni paths for pure-audio Chat Completions requests; realtime and combined audiovisual inputs are outside its scope.

![Audio metadata-only throughput and latency results](/assets/figures/2026-10-18-encoder-disaggregation-epd/03-audio-results.png)

*Figure 3. Audio metadata-only evaluation on MMAU test-mini: 1,000 samples over two passes, Qwen3-Omni-30B-A3B-Instruct, BF16, TP=1, Model Runner V2, Mooncake TCP, 1E+1PD, concurrency 64, and CUDA Graphs enabled.*

In this [audio evaluation](https://github.com/vllm-project/vllm/pull/57887), metadata-only forwarding increased throughput from **62.21 to 73.13 requests/s** and reduced mean latency from **980 to 853 ms**. Accuracy changed from 75.90% to 75.65%; that difference remains to be investigated. The comparison measures two audio input paths within EPD, rather than EPD against colocated serving.

### GPU Video Processing

Beyond avoiding duplicate work on the consumer, GPU NVDEC accelerates the encoder's own video media processing. In a [30-frame MMVU evaluation](https://github.com/vllm-project/vllm/pull/53675), end-to-end throughput improved by approximately 94–115% over three runs, each on a separate 4-GPU GB200 node. Each run used Qwen3.5-35B-A3B, a 1E+3PD topology with a shared-storage EC connector, eager execution, and concurrency 64. The gains include both GPU preprocessing and hardware video decoding; as the encoder became faster, the PD tier became the bottleneck.

## A Dedicated Execution Path for Encoder Instances

An encoder-only instance runs no language model, maintains no language-model KV cache, and produces no sampled tokens. Its execution lifecycle should reflect those differences.

Model Runner V2 now consolidates encoder-only execution in a dedicated `MMEncoderModelRunner`. This removes encoder-specific branches from the shared runner and addresses a synchronization issue exposed by the different output lifecycle.

In ordinary generation, returning sampled outputs includes a device wait associated with copying results to the CPU. An encoder-only step can instead construct an empty runner output on the host. Without an equivalent synchronization point, the host can advance far enough to recycle a pooled host buffer while a GPU kernel is still using it, causing an illegal memory access. The dedicated runner places the required synchronization in the encoder-only path.

Related changes make encoder execution more consistent:

- **Configuration normalization:** producer-only EC instances automatically enter encoder-only mode.
- **Correct asynchronous execution:** the engine uses `is_mm_encoder_only` to determine whether a step needs the subsequent sampling path. This fixes standalone `--mm-encoder-only` instances incorrectly entering `sample_tokens()` when no EC connector is configured.
- **Reduced startup I/O:** encoder-only loading skips pure language-model checkpoint shards when the weight layout can be classified safely. On a pruned Kimi-K3 checkpoint, the encoder read 2 of 96 shards and weight loading dropped from 683 s to 0.31 s.
- **Encoder CUDA Graph capture:** the encoder-only V2 runner supports opt-in ViT graph capture using the existing budget selection, replay, and eager fallback mechanisms.

## Remote Embedding Transfer and Failure Isolation

`ECExampleConnector` provides a shared-filesystem path for validating EPD. Cross-node deployments need network transport, destination-buffer management, and clear ownership of in-flight outputs.

Two remote-transfer paths have landed:

| Backend | Capability | Documentation |
| --- | --- | --- |
| NIXL + CPU EC | Shares CPU-offloaded encoder outputs between instances using NIXL transfers between registered CPU mmap regions | [CPU EC guide](https://docs.vllm.ai/en/latest/features/ec_cpu_connector/) |
| Mooncake EC | Delivers encoder outputs through Mooncake TransferEngine, with TCP and RDMA configuration paths | [Mooncake EC guide](https://docs.vllm.ai/en/latest/features/mooncake_ec_connector_usage/) |

`ECMooncakeConnector` transfers encoder outputs. The separate `MooncakeConnector` serves the corresponding KV-transfer path between P and D.

### Transfer Shapes Must Match Encoder Outputs

Transfer buffers may need to be allocated before an item is encoded. Their size cannot be inferred reliably from a single language-model hidden-size field: different modalities and model architectures can produce different encoder output widths.

For example, the tested Qwen3-Omni checkpoint produces visual outputs of width 8192, including DeepStack information, and audio outputs of width 2048. EC producers now measure output widths per modality during initialization and carry the actual shape through the transfer protocol. The NIXL EC protocol version was also updated, requiring producers and consumers to be upgraded together.

### A Failed Transfer Should Fail the Affected Request

Metadata-only requests make failure handling especially important. A consumer with the original media may be able to fall back to local encoding. A consumer holding only metadata lacks that input.

The consumer now handles this distinction explicitly: when remote encoding cannot arrive and local recovery is unavailable, it reports the affected request as unavailable instead of taking an unsafe fallback path that can terminate EngineCore. Resource-lifecycle fixes, including releasing shared memory after NIXL setup failures, further improve failure handling.

## Reusing Outputs Across Encoder Instances

Instance-local caching leaves a reuse gap. An image encoded by E0 may later be routed to E1 after load balancing, scaling, or recovery. If E1 has no cached output, it can repeat an encoder forward that another instance has already completed.

Mooncake Store enables cross-encoder output reuse. On a local miss, the encoder checks the shared Store. A hit supplies the embedding without another forward; a miss follows normal encoding and asynchronously publishes the output. Delivery to the selected P instance continues through Mooncake P2P.

![Cross-encoder embedding reuse with Mooncake Store](/assets/figures/2026-10-18-encoder-disaggregation-epd/04-cross-encoder-cache.png)

*Figure 4. Mooncake Store provides shared output reuse across encoders. ECMooncakeConnector retains the P2P delivery path to the selected consumer.*

Shared reuse validates input identity, model identity, and the output tensor contract. Initial support includes images, Model Runner V2, encoder TP=1, and no dynamic LoRA. Publication is asynchronous, so concurrent cold requests can still duplicate computation. See the [cross-encoder reuse guide](https://docs.vllm.ai/en/latest/features/cross_encoder_cache/) for configuration and supported combinations.

A [controlled reassignment experiment](https://github.com/vllm-project/vllm/pull/56242) used six H20 GPUs, InternVL3_5-38B-HF, a 2E+4PD topology, and same-node Mooncake TCP. After cache population, images were reassigned between encoders to create local misses. Across 128 measured requests containing 512 images, shared reuse reduced repeated image encodings from **512 to zero**. TTFT p50 decreased from **1.209 to 0.819 seconds**, and p90 from **1.656 to 0.891 seconds**, with essentially unchanged TPOT.

This was a single run of a workload in which every image was reassigned, so it shows the best case for shared reuse. Production savings depend on how often outputs are reused, against the Store's memory and I/O costs.

## Reducing Proxy Overhead on the Critical Path

EPD also depends on the CPU work that connects its stages. The proxy distributes media, collects encoder metadata, rewrites consumer requests, and forwards responses. Multi-image requests may generate multiple encoder subrequests, multiplying orchestration work.

As model execution becomes faster—or a cache hit removes encoding entirely—request parsing, serialization, and forwarding can account for a larger share of latency. Under concurrency, that CPU work can accumulate into queueing and limit how quickly requests reach otherwise available GPU workers. **Independent scaling of the compute stages requires an orchestration path that can keep pace.**

Metadata-only requests reduce the amount of media carried downstream. JSON metadata removes tensor serialization and base64 conversion for small structural fields. The Python EPD proxy also reduces encoding and forwarding overhead:

- Use `msgspec` for request and E/P metadata JSON encoding and decoding.
- Forward final non-streaming response bytes without parsing and re-encoding the JSON body.
- Forward streaming response bytes directly, removing per-chunk text decoding and using `iter_any()` to receive available data.

Routing and metadata inspection still parse JSON, but final responses now pass through without conversion, which matters at high request rates and for long streaming responses.

### EPD Orchestration in vLLM Router

The Python example proxy supports launcher-managed dynamic registration. Registered roles form either E+PD or E/P/D topologies, with health probes and recovery handling. Instance startup and autoscaling remain the external launcher's responsibility.

vLLM Router now supports EPD orchestration through `--epd-config`. The router selects P/PD before encoding, distributes media to encoder workers, and forwards JSON metadata references. EC backends handle embedding delivery, while separate P/D deployments reuse the existing KV handoff.

The initial integration is experimental and supports the native CLI, static worker configuration, and DP=1. Use router main with [#266](https://github.com/vllm-project/router/pull/266). The [router EPD guide](https://github.com/vllm-project/router/blob/main/docs/epd.md) describes its current scope and backend constraints.

## DeepSeek V4.1 Flash: EPD in Practice

We compared PD serving with EPD on GB200 using DeepSeek V4.1 Flash and 512 MP-DocVQA validation questions, each containing four to six original document pages. Requests averaged 4,905 input tokens and 5.2 output tokens. Each concurrency level ran three times, with caches and CUDA Graphs enabled.

The PD baseline used four GPUs for P and four for D. EPD added four single-GPU encoder instances while keeping the same P/D configuration: TP=1, DP=4, expert parallelism, and GPU-resident Engram tables sharded across DP ranks (DEP4). EPD used Mooncake GPU RDMA for embeddings; both deployments used NIXL for KV transfer. Both ran main at `5314c58281` plus the [DeepSeek metadata-only extension](https://github.com/vllm-project/vllm/pull/62802).

![DeepSeek V4.1 Flash throughput per GPU versus P90 time to first token, comparing PD and EPD](/assets/figures/2026-10-18-encoder-disaggregation-epd/05-deepseek-v41-epd.png)

*Figure 5. Higher throughput and lower P90 TTFT are better. C denotes client concurrency. Throughput counts input plus output tokens, divided by elapsed time and all deployed GPUs: eight for PD, twelve for EPD, including all four encoder GPUs. Points pool three rounds; P90 pools all 1,536 requests per concurrency level. [Plot data](/assets/figures/2026-10-18-encoder-disaggregation-epd/05-deepseek-v41-epd.csv).*

At concurrency **64–128**, EPD delivered **6.6–9.5% higher input-plus-output throughput per GPU** than the eight-GPU PD baseline, counting all **12 EPD GPUs**, including the encoders. P90 TTFT was **44–47% lower**. At concurrency 16, however, EPD had lower per-GPU throughput; at 32 it was approximately break-even. The benefit depends on load and resource allocation, rather than separation alone.

## Getting Started

Start with E+PD to validate the encoder-to-consumer handoff before adding a separate decode tier. The [Disaggregated Encoder guide](https://docs.vllm.ai/en/latest/features/disagg_encoder/) links the example deployments and integration tests.

With a compatible vLLM checkout, Mooncake installed, and two available GPUs, the following integration script runs a baseline and compares outputs with a disaggregated deployment:

```bash
GPU_E=0 GPU_PD=1 MOONCAKE_EC_PROTOCOL=tcp \
  bash tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh
```

Use the [Mooncake EC guide](https://docs.vllm.ai/en/latest/features/mooncake_ec_connector_usage/) for transport dependencies and consumer control-address configuration, or the [CPU EC guide](https://docs.vllm.ai/en/latest/features/ec_cpu_connector/) for CPU-offloading and NIXL-based sharing. Choose the E/PD resource split using the target media mix, cache behavior, and latency requirements.

## What's Next

Recent work has expanded EPD beyond its core execution and transport paths. MiMo-V2-Omni now declares the embedding fields needed for image EPD, while XPU support includes EC integration and an encoder-only producer runner.

Further work includes observability, additional transport and cache backends, frontend integration, and clearer upstream error propagation. Operators need to distinguish time spent computing embeddings, loading cached outputs, and transferring them remotely.

Another opportunity is coordinating encoder work with prefill-side reuse. In an E-first pipeline, encoding can begin before P discovers that its KV cache already covers a multimodal occurrence, or that a reusable embedding is available locally. An ongoing [design discussion](https://github.com/vllm-project/vllm/issues/55639) explores how P's demand could suppress unnecessary upstream work. The design must balance saved computation and delivery against additional lookup, coordination, and cold-request latency.

## Related Resources

- [Encoder Disaggregation for Scalable Multimodal Model Serving](https://vllm.ai/blog/2025-12-15-vllm-epd): the original architecture and motivation.
- [Disaggregated Encoder documentation](https://docs.vllm.ai/en/latest/features/disagg_encoder/): deployment examples and configuration.
- [EPD tracker](https://github.com/vllm-project/vllm/issues/52409): ongoing implementation work.

<details markdown="1">
<summary>Implementation references</summary>

- Multimodal inputs: [image metadata](https://github.com/vllm-project/vllm/pull/50390), [video references and timestamps](https://github.com/vllm-project/vllm/pull/54242), [audio metadata](https://github.com/vllm-project/vllm/pull/57887), and [JSON metadata](https://github.com/vllm-project/vllm/pull/56090).
- Encoder execution: [dedicated runner](https://github.com/vllm-project/vllm/pull/53176), [configuration normalization](https://github.com/vllm-project/vllm/pull/53656), [asynchronous execution](https://github.com/vllm-project/vllm/pull/58490), [checkpoint loading](https://github.com/vllm-project/vllm/pull/58086), and [CUDA Graph capture and backend validation](https://github.com/vllm-project/vllm/pull/56922).
- Remote delivery: [NIXL + CPU EC](https://github.com/vllm-project/vllm/pull/47941), [Mooncake EC](https://github.com/vllm-project/vllm/pull/41567), [measured transfer shapes](https://github.com/vllm-project/vllm/pull/57904), [request failure isolation](https://github.com/vllm-project/vllm/pull/55290), and [shared-memory cleanup](https://github.com/vllm-project/vllm/pull/59875).
- Caching and media processing: [cross-encoder reuse](https://github.com/vllm-project/vllm/pull/56242) and [GPU video processing](https://github.com/vllm-project/vllm/pull/53675).
- Orchestration: [Python proxy overhead reduction](https://github.com/vllm-project/vllm/pull/56657), [dynamic registration](https://github.com/vllm-project/vllm/pull/54176), and [vLLM Router integration](https://github.com/vllm-project/router/pull/266).
- Model and device support: [MiMo-V2-Omni](https://github.com/vllm-project/vllm/pull/58938), [XPU EC integration](https://github.com/vllm-project/vllm/pull/49532), and [XPU encoder-only execution](https://github.com/vllm-project/vllm/pull/59320).

</details>
