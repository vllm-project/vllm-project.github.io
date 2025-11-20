---
layout: post
title: "Docker Model Runner Integrates vLLM for High-Throughput Inferencing"
author: "Docker Team"
image: /assets/figures/docker-model-runner/docker-model-page.png
---

## Expanding Docker Model Runner's Capabilities

Today, we're excited to announce that Docker Model Runner now integrates the vLLM inference engine and safetensors models, unlocking high-throughput AI inference with the same Docker tooling you already use.

When we first introduced Docker Model Runner, our goal was to make it simple for developers to run and experiment with large language models (LLMs) using Docker. We designed it to integrate multiple inference engines from day one, starting with llama.cpp, to make it easy to get models running anywhere.

Now, we're taking the next step in that journey. With vLLM integration, you can scale AI workloads from low-end to high-end Nvidia hardware, without ever leaving your Docker workflow.

![docker_vllm_models](/assets/figures/docker-model-runner/docker-model-page.png)

## Why vLLM?

vLLM is a high-throughput, open-source inference engine built to serve large language models efficiently at scale. It's used across the industry for deploying production-grade LLMs thanks to its focus on throughput, latency, and memory efficiency.

Here's what makes vLLM stand out:

- **Optimized performance**: Uses PagedAttention, an advanced attention algorithm that minimizes memory overhead and maximizes GPU utilization.
- **Scalable serving**: Handles batch requests and streaming outputs natively, perfect for interactive and high-traffic AI services.
- **Model flexibility**: Works seamlessly with popular open-weight models like GPT-OSS, Qwen3, Mistral, Llama 3, and others in the safetensors format.

By bringing vLLM to Docker Model Runner, we're bridging the gap between fast local experimentation and robust production inference.

## How vLLM Works

Running vLLM models with Docker Model Runner is as simple as installing the backend and running your model, no special setup required.

Install Docker Model Runner with vLLM backend:

```bash
docker model install-runner --backend vllm --gpu cuda
```

Once the installation finishes, you're ready to start using it right away:

```bash
docker model run ai/smollm2-vllm "Can you read me?"
```

```
Sure, I am ready to read you.
```

Or access it via API:

```bash
curl --location 'http://localhost:12434/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
  "model": "ai/smollm2-vllm",
  "messages": [
    {
      "role": "user",
      "content": "Can you read me?"
    }
  ]
}'
```

Note that there's no reference to vLLM in the HTTP request or CLI command.

That's because Docker Model Runner automatically routes the request to the correct inference engine based on the model you're using, ensuring a seamless experience whether you're using llama.cpp or vLLM.

## Why Multiple Inference Engines?

Until now, developers had to choose between simplicity and performance. You could either run models easily (using simplified portable tools like Docker Model Runner with llama.cpp) or achieve maximum throughput (with frameworks like vLLM).

Docker Model Runner now gives you both.

You can:

- Prototype locally with llama.cpp.
- Scale to production with vLLM.

Use the same consistent Docker commands, CI/CD workflows, and deployment environments throughout.

This flexibility makes Docker Model Runner a first in the industry — no other tool lets you switch between multiple inference engines within a single, portable, containerized workflow.

By unifying these engines under one interface, Docker is making AI truly portable, from laptops to clusters, and everything in between.

## Safetensors (vLLM) vs. GGUF (llama.cpp): Choosing the Right Format

With the addition of vLLM, Docker Model Runner is now compatible with the two most dominant open-source model formats: Safetensors and GGUF. While Model Runner abstracts the complexity of setting up the engines, understanding the difference between these formats helps in choosing the right tool for your infrastructure.

- **GGUF (GPT-Generated Unified Format)**: The native format for llama.cpp, GGUF is designed for high portability and quantization. It is excellent for running models on commodity hardware where memory bandwidth is limited. It packages the model architecture and weights into a single file.
- **Safetensors**: The native format for vLLM and the modern standard for high-end inference, safetensors is built for high-throughput performance.

Docker Model Runner intelligently routes your request: if you pull a GGUF model, it utilizes llama.cpp; if you pull a safetensors model, it leverages the power of vLLM. With Docker Model Runner, both can be pushed and pulled as OCI images to any OCI registry.

## vLLM-compatible models on Docker Hub

vLLM models are in safetensors format. Some early safetensors models available on Docker Hub:

- [ai/smollm2-vllm](https://hub.docker.com/r/ai/smollm2-vllm)
- [ai/qwen3-vllm](https://hub.docker.com/r/ai/qwen3-vllm)
- [ai/gemma3-vllm](https://hub.docker.com/r/ai/gemma3-vllm)
- [ai/gpt-oss-vllm](https://hub.docker.com/r/ai/gpt-oss-vllm)

## Available Now: x86_64 with Nvidia

Our initial release is optimized for and available on systems running the x86_64 architecture with Nvidia GPUs. Our team has dedicated its efforts to creating a rock-solid experience on this platform, and we're confident you'll feel the difference.

## What's Next?

This launch is just the beginning. Our vLLM roadmap is focused on two key areas: expanding platform access and continuous performance tuning.

- **WSL2/Docker Desktop compatibility**: We know that a seamless "inner loop" is critical for developers. We are actively working to bring the vLLM backend to Windows via WSL2. This will allow you to build, test, and prototype high-throughput AI applications Docker Desktop with the same workflow you use in Linux environments, starting with Nvidia Windows machines.
- **DGX Spark compatibility**: We are optimizing Docker Model Runner for different kinds of hardware. We are working to add compatibility for Nvidia DGX systems.
- **Performance Optimization**: We're also actively tracking areas for improvement. While vLLM offers incredible throughput, we recognize that its startup time is currently slower than llama.cpp's. This is a key area we are looking to optimize in future enhancements to improve the "time-to-first-token" for rapid development cycles.

Thank you for your support and patience as we grow.

## How You Can Get Involved

The strength of Docker Model Runner lies in its community, and there's always room to grow. We need your help to make this project the best it can be. To get involved, you can:

- **Star the repository**: Show your support and help us gain visibility by starring the [Docker Model Runner repo](https://github.com/docker/model-runner).
- **Contribute your ideas**: Have an idea for a new feature or a bug fix? Create an issue to discuss it. Or fork the repository, make your changes, and submit a pull request. We're excited to see what ideas you have!
- **Spread the word**: Tell your friends, colleagues, and anyone else who might be interested in running AI models with Docker.

We're incredibly excited about this new chapter for Docker Model Runner, and we can't wait to see what we can build together. Let's get to work!
