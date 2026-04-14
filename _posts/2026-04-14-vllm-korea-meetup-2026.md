---
layout: post
title: "vLLM Korea Meetup 2026 Wrap-Up"
author: "vLLM Team"
image: /assets/figures/vllm-korea-meetup-2026/banner.jpg
tags:
  - community
---

<p align="center">
<picture>
<img src="/assets/figures/vllm-korea-meetup-2026/banner.jpg" width="100%">
</picture><br>
</p>

Hosted by the vLLM KR Community, with support from Rebellions, SqueezeBits, Red Hat APAC, and PyTorch Korea, the vLLM Korea Meetup 2026 was held in Seoul on April 2nd.

This meetup proved to be much more than a standard tech event. Not only did it see strong turnout on the day, but the post-event survey recorded an impressive ~75% response rate — a testament to the active engagement of the attendees. Results reflected high overall satisfaction, confirming that the meetup delivered both in-depth practical content and a genuine community experience.

Field engineers from a wide range of companies and research institutions gathered to share real-world deployment stories and infrastructure strategies for running LLMs in production. As AI moves beyond the research phase and into full-scale services, handling inference workloads efficiently has become a central challenge. Against this backdrop, vLLM is rapidly establishing itself as the foundational infrastructure for high-performance LLM serving, seeing adoption across environments from cloud to enterprise.

## Intro: The Expansion and Standardization of the vLLM Ecosystem

<p align="center">
<picture>
<img src="/assets/figures/vllm-korea-meetup-2026/networking.jpg" width="100%">
</picture><br>
</p>

The meetup opened with Dr. Hongseok Kim from Rebellions and Li Ming from Red Hat APAC sharing the latest vLLM project updates and community news.

Dr. Kim introduced the operational structure the vLLM KR community has built over the six months since its inaugural meetup — a Steering Group-centered governance model supported by regular meetups and hands-on workshops. On the technical side, he highlighted vLLM's complete architectural migration from v0 to v1, which simplifies the codebase and strengthens modularity. Internal structural changes — including async scheduling and Model Runner improvements — have been accompanied by rapid feature expansion: a streaming API, semantic router, and vLLM-Omni.

<p align="center">
<picture>
<img src="/assets/figures/vllm-korea-meetup-2026/intro_liming.jpg" width="100%">
</picture><br>
</p>

Li Ming introduced vllm-playground, designed to lower vLLM's notoriously high barrier to entry (140+ configuration parameters). The GUI-based tool shortens time-to-first-run, supports CPU and macOS environments, and includes performance visualization — making it significantly easier for teams to experiment with and adopt vLLM.

The message from this session was unambiguous: LLM serving is no longer just a question of which framework to pick. It has grown into an infrastructure challenge — one that needs to run efficiently across vastly different environments.

## Integrating AI Accelerators with vLLM

Dr. Kim also covered the integration roadmap between vLLM and AI accelerator hardware. Rebellions, an AI semiconductor company, is developing the vllm-rbln plugin to bring its proprietary NPUs into the vLLM ecosystem. Core features like paged attention and continuous batching are already implemented and supported in the NPU environment. More advanced capabilities — including speculative decoding, distributed KV cache, and prefill/decode disaggregation — are currently in development, with next-generation NPUs like the Rebel100TM opening the door to large-scale inference cluster deployments.

This approach reflects a broader industry shift: rather than hardware-specific, siloed optimizations, AI inference infrastructure is being restructured around vLLM as the common layer connecting diverse accelerators.

## vLLM Production Stack: Present and Future

<p align="center">
<picture>
<img src="/assets/figures/vllm-korea-meetup-2026/intro_hongseok.jpg" width="100%">
</picture><br>
</p>

In the third session, Taesoo Kim, CTO of SqueezeBits, presented on the vLLM production stack — covering what it currently offers in real operating environments, how it has evolved, and where it is headed.

The central theme: vLLM is growing well beyond simply serving models. It is steadily acquiring the operational features and scalability that production environments genuinely require.

## Two Tracks: Open Source and Business

From the midpoint onward, the meetup split into two parallel tracks to accommodate a wider range of real-world perspectives. Attendees chose between Track 1: vLLM with Open Source and Track 2: vLLM in Business, each featuring two sessions.

<p align="center">
<picture>
<img src="/assets/figures/vllm-korea-meetup-2026/production_stack.jpg" width="100%">
</picture><br>
</p>

### Track 1 — Session 1: Memory and Cache as the Core of LLM Serving Optimization

Juho Lee from XCENA — a memory-centric computing startup building CXL 3.0-based intelligent memory semiconductors for large-scale data processing — presented on the vLLM production stack and KV cache optimization strategies.

He framed LLM serving fundamentally as a "cluster efficiency problem", arguing that how KV cache is stored and reused is what simultaneously determines both performance and cost. His talk introduced KV cache tiering and routing via LMCache to reduce dependence on AI accelerator memory, and explored CXL memory as a large-capacity cache expansion tier — forming a new layer in the memory hierarchy.

The implication: LLM infrastructure optimization is moving beyond compute into optimizing data movement and memory architecture itself.

### Track 1 — Session 2: From Open-Source Model to Production Service

Inseo Song from Upstage — the AI startup behind the Solar LLM — shared the engineering journey of taking an open-source model and deploying it as a reliable production service. The talk placed heavy emphasis on the engineering complexity that emerges after training is complete.

He walked through the design of Chat Templates to meet diverse requirements — OpenAI-compatible APIs, multi-turn conversation, reasoning, function calling, and structured outputs — and the creation of structures capable of parsing state at the token level. He also explained how parsers and logits processors are used during vLLM integration to exercise fine-grained control over generation behavior.

The takeaway was: "serving stably" is a far more complex problem than simply "building a good model."

### Track 2 — Session 1: LLM Operational Strategy in the Enterprise

Sungsu Kim from Samsung Electronics presented under the title "Protecting Sensitive Data with vLLM." The session opened with a direct argument: in enterprise deployments, security is the single most critical factor.

He shared a case study on eliminating data leakage risk in environments where external SaaS models are off the table — achieved by building a private LLM API on internal GPU infrastructure and routing all requests through an air-gapped closed network. The system now serves over 4,000 employees through interfaces including OpenWebUI, OpenAI-compatible APIs, Dify, and Claude Code. He also covered how task-separated RAG-based agents with access control structures keep sensitive data protected, all while minimizing custom development by leaning on open-source tooling.

The session made a clear case: technical performance is only part of the equation. Security architecture and operational design matter just as much.

### Track 2 — Session 2: Serving Architecture for the Multimodal Era

The final session featured Jaeeun Gil from NAVER Cloud, presenting on serving the HyperCLOVA Omni model. Omni-modal models — handling text, images, and audio together — combine an autoregressive architecture with a diffusion-based decoder, making them structurally heterogeneous and difficult to serve efficiently with conventional approaches.

The proposed solution was a disaggregated serving architecture: encoder, LLM, and decoder separated into independent stages and optimized individually. Analysis identified the vision decoder as the primary latency bottleneck, accounting for the majority of end-to-end latency. Through sequence parallelism and kernel optimization, the team achieved performance improvements of over 3x.

The presentation illustrated how LLM serving is evolving from single-model execution into a complex, multi-component pipeline optimization problem.

## Closing Thoughts: LLM Infrastructure Reshaping Around vLLM

<p align="center">
<picture>
<img src="/assets/figures/vllm-korea-meetup-2026/closing.jpg" width="100%">
</picture><br>
</p>

A consistent theme ran through every session: LLM serving is no longer about running a specific model quickly. It has evolved into an infrastructure problem — efficiently operating diverse models, heterogeneous hardware, and complex pipelines at scale. Beyond the technical content, the meetup was a chance to witness the energy and depth of the community firsthand.

vLLM sits at the center of a fast-moving shift, with hardware vendors, cloud providers, AI service companies, and end users all building strategies around it. The technology and community centered on vLLM will continue to expand — and the practical, field-driven case studies shared within it will only grow richer.
