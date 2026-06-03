---
layout: post
title: "Announcing Day-0 Support for NVIDIA Nemotron 3 Ultra on vLLM"
author: "NVIDIA Nemotron Team"
image: /assets/figures/2026-nemotron-3-ultra/hero.png
tags:
  - model-support
---

<p align="center">
<picture>
<img src="/assets/figures/2026-nemotron-3-ultra/hero.png" width="100%">
</picture>
</p>

We are excited to announce Day-0 Support for the newly released NVIDIA Nemotron 3 Ultra on vLLM.

[Nemotron 3 Ultra](https://blogs.nvidia.com/blog/nvidia-gtc-taipei-computex-2026-news/#nemotron-3-ultra), part of the Nemotron family of open models, is built for frontier-class reasoning in long-running autonomous agent workflows. It is designed for complex orchestration, coding, deep research, enterprise automation, and other tasks where agents must plan, call tools, recover from errors, and reason across extended context.

Modern agentic systems are increasingly persistent. They do not simply answer a single prompt; they search, write code, run tests, inspect failures, coordinate tools, evaluate evidence, and continue working across long task horizons. These workflows demand models that can sustain reasoning depth while keeping inference fast enough for practical deployment.

Nemotron 3 Ultra addresses two major requirements for advanced agentic AI:

**Fast Task Completion:**

Long-running agents need more than raw model intelligence. They need throughput that lets them complete more reasoning steps within the same time budget. Nemotron 3 Ultra combines a hybrid Transformer-Mamba MoE architecture, multi-token prediction, and NVIDIA-optimized inference precision to deliver high throughput for demanding agent workloads.

**Advanced Agentic Reasoning:**

Agent workflows often require architectural planning, multi-step debugging, source evaluation, regulatory review, or design verification. Nemotron 3 Ultra is post-trained for reasoning, tool use, and instruction following across agentic environments, helping agents make progress through complex tasks without sacrificing accuracy.

Using this model, agentic systems can complete hard reasoning workflows faster while maintaining strong performance across coding, tool calling, research synthesis, and enterprise automation.

vLLM was a key part of the Nemotron 3 Ultra training workflow, powering high-throughput multi-node inference for rollouts and model evaluation throughout training. Within NeMo RL, vLLM serves as a generation backend for reinforcement learning rollouts, enabling efficient sampling, scalable inference, and integration with NeMo Gym for multi-step and multi-turn training environments.

Nemotron team also used vLLM as part of the evaluation loop that helped us track progress, validate improvements, and understand whether each stage of training was moving the model in the right direction.

# TL;DR: About Nemotron 3 Ultra

* **Architecture:** Mixture of Experts with Hybrid Transformer-Mamba Architecture  
  * Model size: 550B total parameters, 55B active parameters  
  * Context length: Up to 1M tokens  
  * Modalities: Text input, text output  
* **Efficiency:** High-throughput inference with NVFP4 and BF16 support. NVFP4 checkpoint works on Blackwell GPUs.  
* **Reasoning:** Optimized for long-running autonomous agents, tool calling, coding, deep research, and orchestration  
* **Training:** Post-trained with multi-environment reinforcement learning for robust reasoning and agentic behavior  
* **Deployment:** Open weights, open data, and open recipes for customization and deployment across infrastructure  
* **Supported GPUs:**  
  * BF16: 8x GB200/B200/GB300/B300, 16x H100, 8x H200,   
  * NVFP4: 4x GB200/B200/GB300/B300, 8x H100  
* **Get Started**  
  * Download model weights from Hugging Face - [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16), [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)  
  * Run inference with vLLM using the getting started [cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/vllm_cookbook.ipynb)  
  * Read the [Nemotron 3 Ultra technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) for architecture, training, and benchmark details

# Run Optimized Agentic Inference with vLLM

Nemotron 3 Ultra is designed for high-throughput agentic inference across BF16 and NVFP4 precision modes. With vLLM, developers can serve the model through an OpenAI-compatible API and integrate it into existing agent frameworks, coding systems, research pipelines, and enterprise automation workflows.

For an easier setup with vLLM, refer to the Nemotron 3 Ultra getting started [cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/vllm_cookbook.ipynb) or use the NVIDIA Brev [launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-3EPQRUP8Sl27sxp1fMvXt3Lor8T) for NVFP4.

## Install vLLM

```bash
docker pull vllm/vllm-openai:v0.22.0

docker run --rm -it --gpus all --ipc=host --network=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.22.0
```

## Serve the model

The command below is configured for a 8x B200 setup. If your hardware differs, adjust the parallelism flags and related settings for your environment.

```bash
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

vllm serve nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
  --served-model-name nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 16 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 32768 \
  --enable-flashinfer-autotune \
  --async-scheduling \
  --speculative_config.method mtp \
  --speculative_config.num_speculative_tokens 5 \
  --mamba-backend triton \
  --mamba-ssm-cache-dtype float32 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder

```

Once the server is running, send prompts using an OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me 3 bullet points about vLLM"},
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=1024,
)

msg = resp.choices[0].message
print("Reasoning:", getattr(msg, "reasoning", None))
print("Content:", msg.content)
```

For NVFP4 deployment guidance, refer to the [Nemotron 3 Ultra vLLM cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/vllm_cookbook.ipynb).

# High-Throughput Reasoning for Long-Running Agents

Nemotron 3 Ultra is optimized for agentic systems that need sustained reasoning over many steps. 

As shown in Figure 1, Figure 2 and Figure 3, Nemotron 3 Ultra leads on accuracy on agent productivity, instruction following, and long context tasks and provides leading throughput, saving 30% on costs compared to other leading open models. 

<p align="center">
<picture>
<img src="/assets/figures/2026-nemotron-3-ultra/figure1.svg" width="100%">
</picture>
<br>
Figure 1: Nemotron 3 Ultra leads among open models on agentic benchmarks for agent productivity, coding, and instruction following.
</p>

<p align="center">
<picture>
<img src="/assets/figures/2026-nemotron-3-ultra/figure2.svg" width="100%">
</picture>
<br>
Figure 2: Nemotron 3 Ultra is in the most attractive quadrant with leading accuracy and leading throughput among open models. Config - vLLM with 10k/2k ISL/OSL, BS 1.
</p>
<p align="center">
<picture>
<img src="/assets/figures/2026-nemotron-3-ultra/figure3.svg" width="100%">
</picture>
<br>
Figure 3: Nemotron 3 Ultra saves up to 30% in costs and leads on the cost efficiency frontier.
</p>

To mitigate the typical efficiency-accuracy tradeoffs for high-capacity reasoning models, the Nemotron models introduce profound architectural innovations:

* **Post-Trained for Agent Harness:** Nemotron models are post-trained using the NVIDIA [NeMo RL](https://github.com/nvidia-nemo/rl) and [Gym](https://github.com/NVIDIA-NeMo/gym) across many agent harnesses. They are optimized for agent leading open harnesses, not just single-turn chat and specifically optimized to work inside workflows where agents plan, call tools, read observations, delegate to sub-agents, validate outputs, and recover from errors across many turns.  
* **Hybrid Mamba-Transformer:** Mamba layers improve sequence efficiency for long-context workloads, while Transformer layers preserve precise recall when agents need to retrieve specific facts from large context windows.  
* **Latent MoE:** Latent MoE supports more efficient expert routing, helping the model handle workflows that span reasoning, code generation, tool calls, and domain-specific logic.  
* **Multi-Token Prediction (MTP):** MTP helps reduce generation time by predicting multiple future tokens in a single forward pass, improving throughput for long outputs and multi-turn workflows.  
* **NVFP4 precision:** The same NVFP4 checkpoint runs on NVIDIA Hopper and Blackwell GPUs, so developers can seamlessly use one checkpoint across both architectures thanks to specialized NVFP4 quantization kernels.

# Summary

NVIDIA Nemotron 3 Ultra is an open frontier reasoning model for long-running autonomous agents. It combines high-throughput inference, long-context reasoning, tool-use capability, and open deployment flexibility for developers and enterprises building advanced agentic AI systems.

Ready to build faster, more capable agent workflows?

* Download model weights from Hugging Face - [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16), [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)  
* Run Nemotron 3 Ultra with vLLM using the [cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/vllm_cookbook.ipynb)  
* Read the [Nemotron 3 Ultra technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)

*Stay up to date on [NVIDIA Nemotron](https://developer.nvidia.com/nemotron) by subscribing to NVIDIA news and following NVIDIA AI on [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all), [X](https://x.com/NVIDIAAIDev), [YouTube](https://www.youtube.com/@NVIDIADeveloper)*, *and the [Nemotron channel](https://discord.com/channels/1019361803752456192/1407781691698708682) on [Discord](https://discord.com/invite/nvidiadeveloper).*

# Acknowledgement

Thanks to everyone who contributed to bringing the NVIDIA Nemotron 3 Ultra to vLLM.

NVIDIA: Nirmal Kumar Juluru, Anusha Pant, Alex Steiner, Tomer Asida, Daniel Afrimi, Shaun Kotek, Roi Koren, Daniel Serebrenik, Amir Klein, Omer Ullman Argov, Netanel Haber, Amit Zuker, Shahar Mor, Tomer Bar Natan  
vLLM team and community: Michael Goin, Kaichao You, Yongye Zhu, Roger Wang, Simon Mo, Woosuk Kwon, Yasong Wang, Nick Hill, Zachary Xi
