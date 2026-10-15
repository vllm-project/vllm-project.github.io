---
layout: post
title: "Announcing Day 0 Support for NVIDIA Nemotron 3.5 Super VL on vLLM"
author: "NVIDIA Nemotron Team"
summary: "Run NVIDIA Nemotron 3.5 Super VL on vLLM for multimodal agent workflows with image, document, and video understanding, OpenAI-compatible serving, and multi-token prediction."
image: /assets/figures/2026-nemotron-3-5-super-vl/figure1-agentic-throughput.png
tags:
  - model-support
---

Always-on agents increasingly need to interpret visual information, reason across it, use tools, and act. These workflows require a model that combines multimodal understanding with high-accuracy execution.

NVIDIA Nemotron 3.5 Super VL is an open, customizable multimodal model built for long-running agents, enabling developers to maximize accuracy and efficiency for specialized workflows.

It combines reasoning, instruction following, and tool use with text, image, document, and video understanding in one model, reducing the need to deploy separate text and vision models and helping maintain task-relevant context across workflow steps.

Built on a 120-billion-parameter mixture-of-experts (MoE) architecture with 12 billion active parameters, Nemotron 3.5 Super VL balances accuracy and efficiency for specialized, always-on agent tasks. Organizations can post-train it with NVIDIA NeMo using their data, tools, policies, and workflows.

Within a system of models, Nemotron 3.5 Super VL is designed as the default high-accuracy agentic execution model with native document, image, and video understanding. It complements NVIDIA Nemotron 3.5 Lightning for high-volume text execution and NVIDIA Nemotron 3 Ultra for the hardest text-only reasoning.

Nemotron 3.5 Super VL addresses two practical requirements for multimodal, always-on agents:

* **Native visual understanding with agentic execution:** The model combines high-accuracy reasoning and tool use with document, image, and video understanding, preserving task-relevant context across specialized workflows.
* **Efficient, controllable deployment:** A hybrid Mamba-Transformer architecture, multi-token prediction, and NVFP4 quantization are designed to improve generation efficiency and create headroom for more concurrent long-context sessions.

## TL;DR: About NVIDIA Nemotron 3.5 Super VL

* **Architecture:** 120B total parameters with 12B active parameters in a mixture-of-experts design
* **Modalities:** Text, images, documents, and video as input; text as output
* **Optimization:** C-RADIOv4 vision encoder, hybrid Mamba-Transformer architecture, multi-token prediction, and NVFP4 quantization
* **Model role:** High-accuracy agentic execution with native visual understanding for systems of models
* **Customization:** Post-train with NVIDIA NeMo using organization-specific data, tools, policies, and workflows
* **Deployment:** NVIDIA NIM for optimized production serving across cloud and data center environments
* **Deployment targets:** 1x GB10, B200, B300, RTX PRO 6000, H200, and 2x H100
* **Availability:**
  * Download the model weights from Hugging Face: [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16) and [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
  * Run Nemotron 3.5 Super VL with vLLM using the getting-started [cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Super-VL/vllm_cookbook.ipynb)

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

The following NVFP4 launch configuration is an example for a 1× GB200 setup. Validate it on the target hardware before use.

```sh
vllm serve nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --served-model-name nemotron-3.5-super \
  --trust-remote-code \
  --max-model-len 114688 \
  --reasoning-parser nemotron_v3 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 5
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
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/clip.mp4"}
                },
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

For example, MTP can also be enabled with three speculative tokens:

```sh
vllm serve nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --served-model-name nemotron-3.5-super \
  --trust-remote-code \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
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

Nemotron 3.5 Super VL is designed to deliver high accuracy for specialized multimodal agent tasks while using only a subset of its total parameters for each token. The hybrid Mamba-Transformer architecture and multi-token prediction along with NVFP4 quantization help accelerate output generation.

As shown in Figure 1, on a single GPU, this model delivers up to 5x faster throughput for agentic tasks compared to other [open models](https://www.nvidia.com/en-us/glossary/open-models/) in its size category.

![Nemotron 3.5 Super VL reaches over 600 output tokens per second at 100,000 input tokens, compared with other open models.](/assets/figures/2026-nemotron-3-5-super-vl/figure1-agentic-throughput.png)

Figure 1. Nemotron 3.5 Super VL delivers over 600 TPS on Artificial Analysis Intelligence Index agentic tasks on Baseten endpoint.

As shown in Figure 2, Nemotron 3.5 Super VL performs tasks 4x faster compared to other leading open models of similar size.

![Chart comparing SWE-bench Pro accuracy and hours to complete 10,000 tasks on four B200 GPUs; Nemotron 3.5 Super VL has the shortest completion time.](/assets/figures/2026-nemotron-3-5-super-vl/figure2-b200-task-completion.png)

Figure 2. Nemotron 3.5 Super VL takes advantage of higher throughput and token efficiency to deliver 4x faster task completion.

For context, Figure 3 shows a DGX Spark comparison for the related Nemotron 3.5 Super model. The chart does not measure Nemotron 3.5 Super VL.

![Nemotron 3.5 Super, a different model from Super VL, scores 80 on the local.ai intelligence index and takes 6.5 minutes per task on DGX Spark.](/assets/figures/2026-nemotron-3-5-super-vl/figure3-dgx-spark-pareto.png)

Figure 3. Nemotron 3.5 Super on the DGX Spark efficiency frontier.

### Get Started

Ready to build agents that can see, reason, and act?

* Download the model weights from Hugging Face: [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16) and [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
* Run Nemotron 3.5 Super VL with vLLM using the getting-started [cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Super-VL/vllm_cookbook.ipynb)

### Acknowledgements

NVIDIA: Nirmal Kumar Juluru, Anusha Pant, Faradawn Yang, Pen Chung Li, Alex Steiner, Ran Zilberstein, Nave Assaf, Amir Klein
