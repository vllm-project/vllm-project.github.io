---
layout: post
title: "Announcing Day 0 Support for NVIDIA Nemotron 3.5 Super VL on vLLM"
author: "NVIDIA Nemotron Team"
summary: "Run NVIDIA Nemotron 3.5 Super VL on vLLM for multimodal agent workflows with image, document, and video understanding, OpenAI-compatible serving, and multi-token prediction."
image: /assets/figures/2026-nemotron-3-5-super-vl/figure1-b200-task-completion.png
tags:
  - model-support
---

Always-on agents increasingly need to interpret visual information, reason across it, use tools, and act. These workflows require a model that combines multimodal understanding with high-accuracy execution.

NVIDIA Nemotron 3.5 Super VL is an open multimodal model for long-running agents that need to see, reason, and act. It combines text, image, document, and video understanding with agentic reasoning and tool use in one model.

Built on a 120-billion-parameter mixture-of-experts (MoE) architecture with 12 billion active parameters, Nemotron 3.5 Super VL balances accuracy and efficiency for specialized, always-on agent tasks. Organizations can post-train it with NVIDIA NeMo using their data, tools, policies, and workflows.

Within a system of models, Nemotron 3.5 Super VL is designed as the default high-accuracy agentic execution model with native document, image, and video understanding. It complements NVIDIA Nemotron 3.5 Lightning for high-volume text execution and NVIDIA Nemotron 3 Ultra for the hardest text-only reasoning.

Nemotron 3.5 Super VL addresses two practical requirements for multimodal, always-on agents:

* **Native visual understanding with agentic execution:** The model combines high-accuracy reasoning and tool use with document, image, and video understanding, preserving task-relevant context across specialized workflows.
* **Efficient, controllable deployment:** A hybrid Mamba-Transformer architecture, multi-token prediction, and NVFP4 KV cache quantization are designed to improve generation efficiency and create headroom for more concurrent long-context sessions.

## TL;DR: About NVIDIA Nemotron 3.5 Super VL

* **Architecture:** 120B total parameters with 12B active parameters in a mixture-of-experts design
* **Modalities:** Text, images, documents, and video as input; text as output
* **Optimization:** C-RADIOv4 vision encoder, hybrid Mamba-Transformer architecture, multi-token prediction, and NVFP4 KV cache quantization
* **Model role:** High-accuracy agentic execution with native visual understanding for systems of models
* **Customization:** Post-train with NVIDIA NeMo using organization-specific data, tools, policies, and workflows
* **Deployment:** NVIDIA NIM for optimized production serving across cloud and data center environments
* **Deployment targets:** 1x B10, B200, B300, RTX PRO 6000, H200, and 2x H100
* **Availability:**
  * Download the model weights from Hugging Face: BF16 and NVFP4
  * Run Nemotron 3.5 Super VL with vLLM using the getting-started [cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/NVIDIA-Nemotron-3.5-Super-VL/vllm_cookbook.ipynb)

## Run High-Throughput Inference with vLLM

Nemotron 3.5 Super VL is intended to run across a wide range of NVIDIA platforms. vLLM provides the serving layer needed to bring the model into production workflows, including continuous batching, prefix caching, speculative decoding, and an OpenAI-compatible API.

The BF16 checkpoint offers a straightforward baseline for deployment. NVFP4 is also available at launch for environments that can take advantage of lower-precision inference.

### Install vLLM

```sh
docker pull vllm/vllm-openai:v0.31.0

docker run --rm -it --gpus all --ipc=host --network=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.31.0

```

### Serve the Model

The command below should be replaced with the validated BF16 launch configuration for the target hardware.
This command assumes a 1 x H200 setup.

```sh
export MODEL=nvidia/Nemotron-3.5-Super-VL-120B-A12B-GA-candidate-MTP-boosted-20261001

vllm serve $MODEL \
  --served-model-name nemotron-3.5-super \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --max-model-len 114688 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 32 \
  --limit-mm-per-prompt '{"video": 1, "image": 1}' \
  --video-pruning-rate 0.5 \
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
  --speculative_config.method mtp \
  --speculative_config.num_speculative_tokens 3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --host 0.0.0.0 \
  --port 8000

```

Once the server is running, applications can send prompts through an OpenAI-compatible client:

```py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="null")

response = client.chat.completions.create(
    model="nemotron-3.5-super",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what happens in this video."},
                {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
            ],
        }
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=4096,
)

print(response.choices[0].message.reasoning)  # thinking trace
print(response.choices[0].message.content)    # final answer

```

## Accelerate Long-Running Agentic Workflows with Speculative Decoding

Nemotron 3.5 Super VL supports Multi-Token Prediction (MTP) to accelerate token generation while preserving the target model’s output quality.

Nemotron 3.5 Super VL includes built-in multi-token prediction (MTP) heads. During decoding, these heads propose future tokens and the target model verifies them, reducing the number of sequential generation steps required for longer responses.

The vLLM launch configuration can enable the model's MTP path through speculative decoding:

```sh
vllm serve $MODEL \
  --served-model-name nemotron-3.5-super \
  --trust-remote-code \
  --speculative_config.method mtp \
  --speculative_config.num_speculative_tokens 3

```

## Built for Specialized, Always-On Agent Workflows

Organizations can adapt Nemotron 3.5 Super VL to their terminology, data, policies, tools, and workflows with NVIDIA NeMo. This enables a general multimodal model to become a specialized execution engine for domain-specific agent tasks while preserving control over model behavior and data handling.

### Example Use Cases

* **Customer service:** Resolve multi-step requests by interpreting customer-provided documents and images, applying policy, and using service tools.
* **Document intelligence:** Interpret complex documents - including OCR, tables, charts, and page layouts - to extract information and generate summaries or reports.
* **Enterprise productivity:** Complete multi-step tasks by interpreting visual information, reasoning across it, and using tools.
* **Video understanding:** Extract key information from video to support search, content review, advertising, retail, and incident analysis.
* **Web research:** Search across websites, connect information from multiple sources, and verify results.

## Accuracy and Efficiency for Multimodal Agent Tasks

Nemotron 3.5 Super VL is designed to deliver high accuracy for specialized multimodal agent tasks while using only a subset of its total parameters for each token. The hybrid Mamba-Transformer architecture and multi-token prediction help accelerate output generation, while NVFP4 KV cache quantization reduces KV cache memory footprint.

In multi-turn agent workflows, retaining more cached prefixes can reduce KV cache eviction and recomputation. That can create additional headroom for concurrent long-context sessions and help production systems serve more users while maintaining responsive interaction.

As shown in Figure 1, Nemotron 3.5 Super VL performs tasks 4x faster compared to other leading open models of similar size.

![Coding accuracy versus time to complete 10,000 tasks on B200, comparing Nemotron 3.5 Super VL with other open models.](/assets/figures/2026-nemotron-3-5-super-vl/figure1-b200-task-completion.png)

Figure 1. Nemotron 3.5 Super VL takes advantage of higher throughput and token efficiency to deliver 4x faster task completion.

This model was also optimized to run locally on DGX Spark, based on the independent evaluation by EXO Labs (Figure 2) Nemotron 3.5 Super is on the Pareto efficiency curve, providing the balance of delivering leading intelligence for local tasks while completing those tasks in a reasonable amount of time.

![Accuracy versus task completion speed for open models on DGX Spark, showing the Nemotron 3.5 Super VL Pareto frontier.](/assets/figures/2026-nemotron-3-5-super-vl/figure2-dgx-spark-pareto.png)

Figure 2: Nemotron 3.5 Super VL delivers the balance of accuracy and speed to run agentic tasks on DGX Spark.

### Get Started

Ready to build agents that can see, reason, and act?

* Download the model weights from Hugging Face: BF16 and NVFP4
* Run Nemotron 3.5 Super VL with vLLM using the getting-started [cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/NVIDIA-Nemotron-3.5-Super-VL/vllm_cookbook.ipynb)

## Acknowledgements

Nirmal Kumar Juluru, Anusha Pant, Faradawn Yang, ...
