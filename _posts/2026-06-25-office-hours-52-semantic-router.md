---
layout: post
title: "vLLM Semantic Router and the DeepLearning.AI vLLM course: a recap of Office Hours #52"
author: "Sawyer Bowerman"
tags:
  - community
  - office-hours
  - semantic-router
  - routing
  - inference
---

At vLLM Office Hours #52, the Red Hat AI team covered two topics: a lightning talk on the new [Fast & Efficient LLM Inference with vLLM](https://www.deeplearning.ai/short-courses/fast-and-efficient-llm-inference-with-vllm/) course on DeepLearning.AI, and a deep dive into the vLLM Semantic Router with Christopher Nuland. We also covered project updates including the vLLM v0.23 release, prefill/decode disaggregation on AMD, new model support, and agentic coding tool integrations.

Here is a summary of the key points from the session.

---

## vLLM project updates

Before the main topics, Michael Goin shared what has been happening across the vLLM ecosystem. Highlights include:

- **Prefill/decode disaggregation on AMD** using Ray and vLLM on MI325X GPUs delivered up to 2.7x more throughput and 67% lower compute costs. On Qwen3-235B, P/D disaggregation served 2.2x more QPS under SLA compared to aggregated serving. On DeepSeek-V3, it served 1.8x more QPS. An important caveat: choosing the wrong prefill-to-decode ratio can make performance worse than aggregated, so tuning the ratio matters. Read the [full results from Anyscale](https://anyscale.com/blog/ray-vllm-prefill-decode-disaggregation-amd-mi325x-67-percent-savings).
- **Codex and Claude Code integration** landed in vLLM. By implementing the OpenAI Responses API and the Anthropic Messages API, vLLM servers can now act as drop-in backends for agentic coding tools. Point Codex or Claude Code at a vLLM endpoint running GLM 5.2, Kimi K2.7 Code, MiniMax M3, or any compatible open model and start coding. See the [Codex integration docs](https://docs.vllm.ai/en/latest/serving/integrations/codex).
- **Day-0 model support** shipped for three major releases: [MiniMax M3](https://vllm.ai/blog/minimax-m3-vllm) with Sparse Attention for 1M-token context, [Poolside Laguna-M.1](https://recipes.vllm.ai/poolside/Laguna-M.1) (225B total, 23B activated MoE coding model, 74.6% SWE-bench Verified), and [GLM 5.2](https://recipes.vllm.ai/zai-org/GLM-5.2) with IndexShare for lower FLOPs and higher MTP acceptance length.
- **GLM 5 performance sprint** is underway at [github.com/vllm-project/vllm/issues/46654](https://github.com/vllm-project/vllm/issues/46654). Active optimizations include replacing MOE all-reduce with reduce-scatter for 3.1-3.2x end-to-end throughput improvement, block-FP8 fused MoE for low-batch decode, vectorized fp32 `moe_sum`, virtual-batch PCP for MLA, and masked MHA for top-k sparse attention. Join the sprint at `#sprint-glm52` on Slack.
- **RL at 1T scale** was demonstrated using [prime-rl](https://primeintellect.ai/blog/rl-at-1t-scale) with vLLM and llm-d. The setup ran GLM-5 across 16 inference and 12 trainer nodes with 131k context. The llm-d router selected the best data-parallel rank for each request, with a shared KV-cache store (Mooncake) pooled across all nodes. Router replay kept trainer-inference KL divergence low and stable throughout training.
- **vLLM v0.23** landed with 408 commits from 200 contributors, including 63 new contributors. On the models and engine side: DeepSeek-V4 matured across backends, Model Runner V2 became the default for Llama and Mistral dense models, Gemma 4 Unified shipped encoder-free with raw 16kHz waveform input, and MRv2 pipeline-parallel bubble elimination reached up to 3.17x tok/s on long-prefill workloads. New model support includes Step-3.7-Flash, Cosmos3 Reasoner, Granite Speech Plus, and JetBrains Mellum v2. On the hardware and API side: AMD RDNA3 received a native W4A16 kernel (2.5-4.2x faster than Triton in bf16), NVIDIA CUTLASS FP8 scaled-mm bypass added +20% on Hopper, multi-tier KV cache offloading shipped with S3/MinIO support, the Anthropic Messages API landed, the Rust frontend added streaming `generate` and dynamic LoRA endpoints, and EPLB became async by default.

## Lightning topic: Fast & Efficient LLM Inference with vLLM on DeepLearning.AI

Cedric Clyburn presented a lightning talk on the new [Fast & Efficient LLM Inference with vLLM](https://www.deeplearning.ai/short-courses/fast-and-efficient-llm-inference-with-vllm/) course, a free 1-hour-38-minute intermediate course built in partnership with Red Hat and Andrew Ng's DeepLearning.AI.

The course walks through the full optimize, deploy, and benchmark workflow for serving open models with vLLM. The syllabus covers seven sections: an introduction to inference fundamentals, why efficient deployment matters, inference and memory mechanics, LLM optimization techniques, hands-on model compression with LLM Compressor, serving with vLLM across two parts, and benchmarking and evaluation with GuideLLM.

The hands-on labs include quantizing a Qwen model with LLM Compressor using a GPTQ recipe, monitoring vLLM's Prometheus-compatible `/metrics` endpoint live under concurrent traffic, and benchmarking a deployment under simulated load with GuideLLM. The course covers practical topics like GPU memory budgeting for KV cache, the difference between weight-only and weight-and-activation quantization, and how to read serving metrics to identify bottlenecks.

As Andrew Ng noted: *"Deploying open-source LLMs efficiently, for many users, with low latency and reasonable cost, is challenging. This course shows you how."*

The course is available now at [deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm](https://www.deeplearning.ai/short-courses/fast-and-efficient-llm-inference-with-vllm/). Read the [announcement blog post](https://vllm.ai/blog/deeplearning-ai-vllm-course) for more details.

## vLLM Semantic Router: intelligent routing for multi-model inference

Christopher Nuland presented a deep dive into the [vLLM Semantic Router](https://github.com/vllm-project/semantic-router), an open-source, signal-driven routing layer for Mixture-of-Models deployments. The project addresses a practical problem: as organizations deploy multiple models with different capabilities, cost profiles, and latency characteristics, choosing the right model for each incoming request becomes a system-level decision.

### What the Semantic Router does

The Semantic Router sits in front of model endpoints, analyzes each request before generation begins, and routes it to the most appropriate backend model. Simple queries go to smaller, faster models to save compute. Complex requests that require deep reasoning go to larger, more capable models. The router also handles safety, caching, and privacy-sensitive content detection inline.

The system is written in Rust using Hugging Face's [Candle](https://github.com/huggingface/candle) framework, which keeps inference latency low and concurrency high. It operates as an Envoy External Processor, routing OpenAI API-compatible requests, and integrates directly with [llm-d](https://github.com/llm-d/llm-d) and the [vLLM Production Stack](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/semantic-router-integration.html).

### Signal-Decision framework

The routing architecture follows a Signal-Decision framework. Signals are observations extracted from each incoming request: domain classification, keyword matching, embedding similarity, fact-checking signals, user feedback, and preference signals. The decision engine composes routing policies from these signals using Boolean logic, allowing operators to define rules like "route to the reasoning model if complexity is high AND domain is code, otherwise route to the fast model."

A key optimization is LoRA-based multi-task classification. Rather than running separate models for each classification task, the router shares base model computation across all tasks using Low-Rank Adaptation, which reduces latency while supporting domain classification, jailbreak detection (via HaluGate), PII detection, and semantic caching in a single pass.

### Routing accuracy and fine-tuning

Nuland also discussed improving routing accuracy through fine-tuning. The pretrained embedding model used for routing understands semantic similarity but not task complexity, which led to a 20% misrouting rate in initial testing. By expanding 48 seed anchors to over 1,000 examples through a synthetic data generation pipeline and training with contrastive learning (`BatchAllTripletLoss`), routing accuracy improved from 80.39% to 98.53%.

In benchmarks on MMLU-Pro with Qwen3 30B, the Semantic Router delivered a 10.2% accuracy improvement on complex tasks (by routing them to the right model), a 47.1% latency reduction, and a 48.5% reduction in token usage compared to sending everything to the largest model.

### Current status

The project has shipped two major releases: [v0.1 Iris](https://vllm.ai/blog/2026-01-05-vllm-sr-iris) in January 2026 and v0.2 Athena in March 2026. The v0.3 Themis release focuses on production stability at scale. The project is open to contributions at [github.com/vllm-project/semantic-router](https://github.com/vllm-project/semantic-router).

## Watch the full session

The slide deck from Office Hours #52 is [available here](https://docs.google.com/presentation/d/1hqDA046cxyMbV_x8UX472xK9x3WsIn8c9513kulec_Y/edit).
