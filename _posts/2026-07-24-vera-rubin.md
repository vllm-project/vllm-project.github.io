---
layout: post
title: "vLLM Runs on NVIDIA Vera Rubin Hardware"
author: "Tyler Michael Smith, Erwan Gallen, Lucas Wilkinson, and Michael Goin"
summary: "vLLM now runs end-to-end on pre-release NVIDIA Vera Rubin hardware."
tags:
  - hardware-support
---

vLLM now runs end-to-end on pre-release NVIDIA Vera Rubin hardware. This follows PyTorch support for the `sm_107` compute capability in [PR #190654](https://github.com/pytorch/pytorch/pull/190654), vLLM support in [PR #49387](https://github.com/vllm-project/vllm/pull/49387), and a preview release of the [CUDA 13.4 developer toolkit](https://docs.nvidia.com/cuda/developer-preview/13.4/cuda-toolkit-release-notes/index.html).

Here is our first request to a model served on next-generation NVIDIA silicon:

```console
[root@vera-rubin ~]# curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"google/gemma-4-31B-it","messages":[{"role":"user","content":"Hello from Vera Rubin!"}],"max_tokens":64}'
```

```json
{
  "id": "chatcmpl-a04459851a0ca396",
  "object": "chat.completion",
  "created": 1784841946,
  "model": "google/gemma-4-31B-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! It is an honor to be greeted by the spirit (or a fan!) of one of the most influential astronomers in history.\n\nIf you are channeling the great **Vera Rubin**, then we have a lot to talk about—from the rotation curves of spiral galaxies to the invisible mystery of **dark matter**."
      },
      "finish_reason": "length"
    }
  ],
  "system_fingerprint": "vllm-0.1.dev257+g7bc831b73-8da9db8c",
  "usage": {
    "prompt_tokens": 18,
    "total_tokens": 82,
    "completion_tokens": 64
  }
}
```

We also verified serving with `poolside/Laguna-XS-2.1`:

```console
[root@vera-rubin ~]# curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"poolside/Laguna-XS-2.1","messages":[{"role":"user","content":"Hello from Vera Rubin!"}],"max_tokens":64}'
```

```json
{
  "id": "chatcmpl-8b9e79937c94c192",
  "object": "chat.completion",
  "created": 1784853360,
  "model": "poolside/Laguna-XS-2.1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello, Vera Rubin! It's a pleasure to connect with someone associated with such groundbreaking work in astrophysics. Your contributions to our understanding of galaxy rotation and dark matter have been pivotal. How can I assist you today? Whether it's discussing your research, exploring cosmic mysteries, or something else entirely, I'm here to"
      },
      "finish_reason": "length"
    }
  ],
  "system_fingerprint": "vllm-0.1.dev257+g7bc831b73-e257006e",
  "usage": {
    "prompt_tokens": 46,
    "total_tokens": 110,
    "completion_tokens": 64
  }
}
```

## Why Rubin matters for inference

Vera Rubin is a step change over current-generation NVIDIA hardware, and vLLM is ready for it. [Rubin delivers up to 22 TB/s of HBM4 bandwidth](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/), a 2.8x uplift over Blackwell that directly accelerates the decode phase where most agentic workloads spend their time. Tensor Core throughput doubles per clock, and exponential operations run up to 4x faster at BF16, which keeps softmax from bottlenecking long-context attention.

## What's next

Today's milestone is functional bring-up. In the coming months, we will bring Rubin's new capabilities into vLLM:

- LUT-based GEMMs for improved weight compression
- Locality-aware compute for better memory subsystem utilization
- Adaptive sparsity instructions for long-context sparse attention
- Reduced whitespace between kernel launches through PDL improvements such as tile-level triggering

## Try it and contribute

A [tracking issue for Rubin and Vera Rubin support](https://github.com/vllm-project/vllm/issues/49735) will be updated as we discover and work through issues on the new hardware. Join the discussion on [GitHub](https://github.com/vllm-project/vllm) or bring questions to [vLLM Office Hours](https://red.ht/office-hours).

Thanks to NVIDIA for access to the pre-release hardware and drivers, the PyTorch team for landing `sm_107` support, and the broader Red Hat Enterprise Linux team for delivering [RHEL for NVIDIA](https://access.redhat.com/support/policy/updates/RHEL_for_NVIDIA) with support for NVIDIA's Vera Rubin systems, making it possible to run vLLM on RHEL immediately.
