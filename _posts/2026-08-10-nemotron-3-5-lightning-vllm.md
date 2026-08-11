---
layout: post
title: "Announcing Day-0 Support for NVIDIA Nemotron 3.5 Lightning on vLLM"
author: "NVIDIA Nemotron Team and vLLM Team"
summary: "How vLLM serves NVIDIA Nemotron 3.5 Lightning with OpenAI-compatible APIs, speculative decoding, and BF16/NVFP4 checkpoints across NVIDIA GPUs and edge systems."
tags:
  - model-support
---

We are excited to announce Day-0 support for NVIDIA Nemotron 3.5 Lightning on vLLM.

Nemotron 3.5 Lightning is a customizable open model for always-on agents, from personal assistants running locally to high-volume agentic tasks in the datacenter and in the cloud. It excels at coding, tool use, instruction following, and multi-turn intelligence and comes in a compact hybrid mixture-of-experts (MoE) architecture with 30 billion total parameters and only 3 billion active parameters at a time.

The model was distilled from NVIDIA Nemotron 3 Ultra and developed with the Nemotron Coalition. Modern agent platforms increasingly divide work across multiple models. A frontier model can take responsibility for difficult planning and orchestration, while a smaller model handles frequent, well-scoped steps. Nemotron 3.5 Lightning is built for that second role without giving up the capabilities required by real agent workflows.

It addresses two practical requirements for always-on agents:

* **Fast execution at scale:** Agent systems often spend most of their time completing small but numerous steps. Nemotron 3.5 Lightning combines a hybrid MoE design, with 3B of 30B parameters active per token, and multi-token prediction to reduce compute and accelerate generation. These optimizations deliver up to 4x higher throughput than similarly sized open models.
* **Adaptable agent intelligence:** Production agents need to understand organization-specific terminology, follow policies, use tools correctly, and maintain context over multiple turns. Nemotron 3.5 Lightning is trained for popular agent harnesses and can be post-trained, making it suitable for specialized tasks in applications such as financial and risk automation, cybersecurity investigation, telecommunications operations, retail experiences, and local personal assistants.

With vLLM, developers can expose the model through an OpenAI-compatible API and connect it to existing agent frameworks, local applications, and enterprise automation systems.

# TL;DR: About Nemotron 3.5 Lightning

* **Architecture:** Hybrid mixture-of-experts architecture
* **Model size:** 30B total parameters, 3B active parameters
* **Context length:** Up to 1 million tokens
* **Modalities:** Text input and text output
* **Speculative decoding:** Multi-token prediction, DFlash, and DSpark
* **Reasoning:** Reasoning can be enabled or disabled for each request, with support for a configurable reasoning-token budget
* **Training:** Distilled from NVIDIA Nemotron 3 Ultra and trained for popular agent harnesses
* **Customization:** Open model trained with open datasets, with support for post-training on specialized workflows
* **Availability at launch:** BF16 and NVFP4
* **Deployment targets:** NVIDIA DGX Spark, DGX Station, RTX PRO, RTX, NVIDIA Jetson, H100, H200, A100, L40S, B200/GB200, and B300/GB300
* **Get started:**
  * Download the model weights from Hugging Face: [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) and [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4).
  * Run Nemotron 3.5 Lightning with vLLM using the getting-started [cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/vllm_cookbook.ipynb).

# Run High-Throughput Inference with vLLM

Nemotron 3.5 Lightning is intended to run across a wide range of NVIDIA platforms. vLLM provides the serving layer needed to bring the model into production workflows, including continuous batching, prefix caching, speculative decoding, and an OpenAI-compatible API.

The BF16 checkpoint offers a straightforward baseline for deployment. NVFP4 is also available at launch for environments that can take advantage of lower-precision inference.

## Install vLLM

```bash
docker pull vllm/vllm-openai:v0.27.1

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --network=host \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.27.1
```

## Serve the Model

This command assumes a 1 x H100 setup.

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --async-scheduling \
  --mamba-backend flashinfer \
  --moe-backend humming \
  --linear-backend humming \
  --mamba-ssu-algorithm horizontal \
  --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --reasoning-parser nemotron_v3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --host 0.0.0.0 \
  --port 8000
```

Once the server is running, applications can send prompts through an OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="null",
)

response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Briefly explain: what is vLLM?"},
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=1024,
)

choice = response.choices[0]
print("Reasoning:", choice.message.reasoning)
print("Content:", choice.message.content)
```

# Accelerate Long-Running Agentic Workflows with Speculative Decoding

Nemotron 3.5 Lightning supports three speculative decoding techniques: Multi-Token Prediction (MTP), DFlash, and DSpark. These accelerate token generation while preserving the target model's output quality.

MTP uses lightweight, model-integrated prediction heads to propose several future tokens. DFlash uses a diffusion-based drafter to generate an entire candidate block in parallel. DSpark adds confidence-aware, semi-autoregressive drafting to balance speed with token-acceptance quality. Together, they let teams choose the best latency, throughput, and deployment trade-off for their inference workload.

Nemotron 3.5 Lightning is architecturally identical to Nemotron 3 apart from the weights and the speculative decoding stack, so most of the performance work landed in the runtimes themselves. Here's what we contributed upstream to vLLM:

* **DSpark integration:** We wired DSpark, a hybrid speculator that blends autoregressive and diffusion-style drafting, into vLLM and the Nemotron model definition, giving you three speculators to choose from alongside MTP and DFlash.
* **Quantized DSpark draft head:** Quantizing the draft head to W4A16 cuts its memory footprint and per-step latency without hurting acceptance rate, which matters most on memory-constrained parts like DGX Spark.
* **Removal of syncs and async scheduling:** We eliminated host-device syncs in the draft-and-verify loop and enabled async scheduling, so the next batch is prepared while the current one is still executing.
* **MoE and linear backend for W4A16:** We replaced vLLM's default Marlin backend with a Hopper-optimized Humming backend, using W4A16 GEMM kernels for Nemotron's non-gated ReLU<sup>2</sup> MoE, worth roughly 20% throughput, and extended the same recipe to the dense linear layers.
* **ReplaySSM integration for Mamba2:** We integrated ReplaySSM for the Mamba2 state-space layers to reduce per-step overhead in the recurrent path of the hybrid architecture.

For low-latency serving, use DSpark across H100, H200, and DGX Spark. For maximum throughput today, we recommend running without speculative decoding.

## Multi-Token Prediction

Nemotron 3.5 Lightning includes built-in multi-token prediction heads. During decoding, these heads propose future tokens and the target model verifies them, reducing the number of sequential generation steps required for longer responses.

The vLLM launch configuration can enable the model's MTP path through speculative decoding:

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.method mtp \
  --speculative_config.num_speculative_tokens 3 \
  --speculative_config.moe_backend flashinfer_cutlass \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

## DFlash

DFlash takes a different approach. It uses a dedicated diffusion draft model to propose a linear block of tokens, which the target model verifies in parallel. DFlash requires a compatible draft checkpoint and is configured separately from MTP.

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --speculative_config.num_speculative_tokens 3 \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

DFlash draft checkpoint: [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash).

## DSpark

DSpark is a hybrid speculator that combines autoregressive and parallel diffusion-style drafting, sitting between MTP's fully autoregressive approach and DFlash's fully diffusion-based one, and delivers the best performance of the three on DGX Spark.

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --speculative_config.num_speculative_tokens 3 \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

DSpark draft checkpoint: [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark).

# Local Deployment on NVIDIA DGX Spark

If you are running locally on DGX Spark, the following should provide a starting configuration for single-user local development:

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --compilation_config.cudagraph_capture_sizes '[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 1024, 2048, 4096, 8192]' \
  --speculative_config.num_speculative_tokens 3 \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.method dspark \
  --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
```

<p align="center">
<picture>
<img src="/assets/figures/2026-nemotron-3-5-lightning/figure1-dgx-spark-pareto.png" width="100%" alt="Pareto chart comparing inference performance of Nemotron 3.5 Lightning using various speculative decoding techniques on NVIDIA DGX Spark.">
</picture>
</p>

Figure 1: Pareto chart comparing inference performance of Nemotron 3.5 Lightning using various speculative decoding techniques on NVIDIA DGX Spark. Config - Prefix - 32K, and then 10 rounds of 2k input and 10k output.

Alt text: Image of Pareto chart showcasing inference performance of Nemotron 3.5 Lightning using various speculative decoding techniques on NVIDIA DGX Spark. Config - Prefix - 32K, and then 10 rounds of 2k input and 10k output.

## Deploy on NVIDIA H100

If you are running on the NVIDIA H100, the following should provide a starting configuration for single-user local development:

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend humming \
  --linear-backend humming \
  --max-num-seqs 256 \
  --trust-remote-code \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --async-scheduling \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align \
  --mamba-ssu-algorithm horizontal \
  --reasoning-parser nemotron_v3
```

<p align="center">
<picture>
<img src="/assets/figures/2026-nemotron-3-5-lightning/figure2-h100-pareto.png" width="100%" alt="Pareto chart comparing inference performance of Nemotron 3.5 Lightning using various speculative decoding techniques on NVIDIA H100 GPUs.">
</picture>
</p>

Figure 2: Pareto chart comparing inference performance of Nemotron 3.5 Lightning using various speculative decoding techniques on NVIDIA H100 GPUs. Config - Prefix - 32K, and then 10 rounds of 2k input and 10k output.

Alt text: Image of Pareto chart showcasing inference performance of Nemotron 3.5 Lightning using various speculative decoding techniques on NVIDIA H100 GPUs. Config - Prefix - 32K, and then 10 rounds of 2k input and 10k output.

# Local Deployment on NVIDIA Jetson

If you are running locally on NVIDIA Jetson, the following should provide a starting configuration for single-user local development:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --reasoning-parser nemotron_v3 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align
```

# Leading Accuracy and Efficiency for Specialized Agent Tasks

Nemotron 3.5 Lightning is designed to make specialized agents both capable and economical to run. Its hybrid MoE architecture activates only 3B of 30B parameters per token, while multi-token prediction reduces the sequential work needed during generation. Together, these features enable up to 4x higher throughput than similarly sized open models.

Nemotron 3.5 Lightning offers leading accuracy for agentic tasks. By distilling capabilities from Nemotron 3 Ultra and training across popular agent harnesses, Nemotron 3.5 Lightning brings strong performance to agent productivity, coding, tool use, instruction following, and long-context reasoning benchmarks.

As shown in Figure 3, higher inference throughput and token efficiency places Nemotron 3.5 Lightning on the efficiency frontier, helping always-on agents finish high-volume work faster.

<p align="center">
<picture>
<img src="/assets/figures/2026-nemotron-3-5-lightning/figure3-efficiency-frontier.png" width="100%" alt="Line chart comparing PinchBench accuracy with time to complete 10,000 tasks.">
</picture>
</p>

Figure 3: Nemotron 3.5 Lightning leads the efficiency frontier by completing agentic tasks up to 30% faster at comparable accuracies.

Alt text: Line chart comparing PinchBench accuracy with time to complete 10,000 tasks. Nemotron 3.5 Lightning reaches similar accuracy as Qwen3.6 35B 30% faster.

# Summary

NVIDIA Nemotron 3.5 Lightning brings customizable agent intelligence to local systems, the edge, datacenters, and the cloud. It combines a 30B-parameter hybrid MoE architecture with 3B active parameters, a context window of up to 1 million tokens, controllable reasoning, and speculative generation through MTP or DFlash.

With Day-0 support in vLLM, developers can serve the model through an OpenAI-compatible stack and integrate it into local assistants, agent harnesses, and specialized enterprise workflows.

Ready to build faster, more efficient agent systems with Nemotron 3.5 Lightning?

* Download the model weights from Hugging Face: [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) and [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4).
* Run Nemotron 3.5 Lightning with vLLM using the getting-started [cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/vllm_cookbook.ipynb).

*Stay up to date on [NVIDIA Nemotron](https://developer.nvidia.com/nemotron) by subscribing to NVIDIA news and following NVIDIA AI on [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all), [X](https://x.com/NVIDIAAIDev), [YouTube](https://www.youtube.com/@NVIDIADeveloper), and the [Nemotron channel](https://discord.com/channels/1019361803752456192/1407781691698708682) on [Discord](https://discord.com/invite/nvidiadeveloper).*

# Acknowledgement

NVIDIA: Nirmal Kumar Juluru, Anusha Pant, Amir Klein, Faradawn Yang, Nave Assaf, Ryan Stewart, Alex Steiner, Bita Rouhani

# FAQs

## What is new compared with the Nemotron 3 Nano?

Nemotron 3 Nano established an efficient hybrid Mamba-Transformer MoE design with 30B total parameters, 3B active parameters, a 1M-token context window, and controllable reasoning. Nemotron 3.5 Lightning builds on that foundation in four important ways:

* **Frontier-model distillation:** Nemotron 3.5 Lightning is distilled from Nemotron 3 Ultra, transferring capabilities from NVIDIA's frontier agentic model into a much smaller deployment footprint.
* **Agent-harness optimization:** Nemotron 3.5 Lightning is trained for popular agent harnesses and multi-turn workflows, with an emphasis on coding, tool use, instruction following, and specialized task completion.
* **Speculative decoding:** Nemotron 3.5 Lightning supports multi-token prediction (MTP), DFlash, and DSpark to accelerate generation by drafting and verifying multiple tokens in parallel.

The result is a model designed to complete more agent tasks more accurately in less time.
