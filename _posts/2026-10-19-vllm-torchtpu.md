---
layout: post
title: "Introducing vLLM TorchTPU"
author: "Inferact and Google Teams"
summary: "vllm-torchtpu is a new vLLM backend for Google Cloud TPUs built on TorchTPU. It runs vLLM's existing model definitions with Pallas kernels for attention, MoE, and quantization, and supports multi-host parallelism, KV offloading, and P/D disaggregation."
image: /assets/figures/2026-10-19-vllm-torchtpu/vllm-torchtpu.png
tags:
  - hardware
---


<p align="center">
<picture>
<img src="/assets/figures/2026-10-19-vllm-torchtpu/vllm-torchtpu.png" width="80%">
</picture><br>
</p>

Today we're releasing [vLLM TorchTPU](https://github.com/vllm-project/vllm-torchtpu) (`vllm-torchtpu`), a new backend for running vLLM on Google Cloud TPUs. It's built on Google's [TorchTPU](https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/) stack and runs the model definitions that already live in vLLM, so most models that work in vLLM today can be served on TPU without changes to the model code.

This work comes out of the [engineering partnership between Inferact and Google](https://inferact.ai/news/google-tpu-partnership) we announced last month. Our goals for the project are to:

1. Make it easy to bring new models to TPU by reusing vLLM's existing model code.
2. Push TPU inference performance in open source, with Pallas kernels for attention, Mixture-of-Experts, and quantized matmul.
3. Keep the vLLM experience the same: the same CLI, OpenAI-compatible server, scheduler, and KV cache manager you use on other hardware.

### How It Works

`vllm-torchtpu` is an out-of-tree vLLM platform plugin. Once installed, vLLM picks it up automatically through the `vllm.platform_plugins` entry point.


Models run through vLLM's standard model runner and are compiled with `torch.compile(backend="tpu")`, which lowers to XLA. Each TPU chip gets its own worker process, the same way vLLM runs on GPUs, and communication between chips goes through `torch.distributed`. This means the scheduler, workers, KV cache manager, and API server are all unmodified upstream vLLM. The TPU-specific work sits underneath them, mostly in a set of JAX/Pallas kernels that are registered as custom ops and called from inside the compiled graph.

#### Getting Started

1. Install

    ```shell
    pip install vllm-torchtpu
    ```

    This requires Python 3.12 on a Cloud TPU VM, and installs a matching vLLM, `torch`, `torch_tpu`, and `libtpu` alongside the plugin. The [installation guide](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/getting_started/installation.md) also covers `uv`, source builds, and [Docker](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/getting_started/docker.md).

2. Serve a model

    ```shell
    vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --kv-cache-dtype fp8 \
      --max-model-len 16384 \
      --async-scheduling
    ```

    The first startup compiles TPU graphs for the configured shapes, which can take a few minutes. Set `VLLM_XLA_CACHE_PATH` to reuse compiled artifacts across restarts (see [Compilation & Warmup Cache](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/user_guide/compilation_cache.md)).

Speculative decoding, prefix caching, multi-LoRA, structured outputs, and multimodal inputs are all enabled with the usual vLLM flags. See the [quickstart](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/getting_started/quickstart.md) for more examples, including offline batch inference, multimodal models, and multi-LoRA.

### Kernels

Much of the performance work in `vllm-torchtpu` lives in its Pallas kernel library:

* Ragged Paged Attention v3, the default attention backend for MHA/GQA models, with `bf16` and `fp8` KV cache.
* Dense and sparse Multi-head Latent Attention (MLA) for DeepSeek, Kimi, and GLM, including SparseCore top-k selection for DeepSeek-V4's sparse attention.
* Gated DeltaNet (Qwen3.5) and Kimi Delta Attention (Kimi-Linear, Kimi K3) kernels for chunked prefill and recurrent decode. Linear-attention state and the KV cache share a single block pool.
* Expert-parallel fused MoE, with token routing offloaded to the TPU SparseCores and hierarchical EP dispatch on dual-core TPU v7x.
* FP8, INT8, MXFP4, and NVFP4 quantized matmuls. Weights are repacked once at load time so the kernels can use the MXU's low-precision paths directly.

Kernels can also be written for a single model. For Moonshot AI's Kimi K3, we added a Pallas [megakernel](https://inferact.ai/blog/tpu-megakernels) that runs an entire decode step as one fused kernel.

### Scaling Out

We've focused heavily on the workloads that put the most pressure on serving systems today, especially multi-turn agentic and multimodal traffic, where contexts keep growing and tool outputs can be long.

#### Parallelism

All six of vLLM's parallelism dimensions are supported, on both single-host and multi-host TPU slices, and they can be combined:

| Dimension | Flag |
| :--- | :--- |
| Tensor Parallelism (TP) | `--tensor-parallel-size` |
| Data Parallelism (DP) | `--data-parallel-size` |
| Pipeline Parallelism (PP) | `--pipeline-parallel-size` |
| Expert Parallelism (EP) | `--enable-expert-parallel` |
| Prefill Context Parallelism (PCP) | `--prefill-context-parallel-size` |
| Decode Context Parallelism (DCP) | `--decode-context-parallel-size` |

For large MoE models, we usually get the best throughput from data-parallel attention with expert parallelism (for example, `DP=8, TP=1` with EP on a single v7x host). This keeps attention local to each chip and spreads the experts across the whole slice. For long contexts, PCP and DCP split prefill and the KV cache across chips. The [parallelism guide](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/user_guide/parallelism.md) has more detail and example configurations.

#### KV Cache Offloading and Disaggregated Serving

`vllm-torchtpu` uses [TPU Sync](https://github.com/google/tpu-sync), an open-source C++ library that can read TPU device buffers directly, outside the Python runtime. Through vLLM's KV connector interface, this gives us:

* Zero-copy KV cache offloading to host DRAM, so the prefix cache for long multi-turn sessions isn't limited by HBM.
* Disaggregated prefill/decode serving across TPU hosts without adding transfer latency. Prefill and decode can use different layouts, for example prefill on `PCP=4, TP=2` and decode on `DP=8, TP=1`, and the KV cache is resharded during transfer.

See the [disaggregated serving guide](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/user_guide/disaggregated_serving.md) to set it up.

#### Speculative Decoding and Prefix Caching

Speculative decoding supports MTP, EAGLE-3, DFlash, DSpark, n-gram, and suffix decoding ([guide](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/user_guide/speculative_decoding.md)). Automatic prefix caching also works for hybrid models like Qwen3.5 and Kimi K3, where linear-attention state is cached along with the KV cache.

### Models, Features, and What's Next

`vllm-torchtpu` can serve most models supported by vLLM, with TPU-specific kernel, quantization, and parallelism work for the Qwen, DeepSeek, Kimi, GLM, Llama, GPT-OSS, and Gemma families. TPU v7x (Ironwood) is the recommended hardware and is tested in CI on every pull request; v6e, v5e, and v5p are experimental.

The [Supported Models & Features](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/user_guide/supported_models.md) page has the full list of validated checkpoints and features, with how to enable each one. The [nightly support matrices](https://github.com/vllm-project/vllm-torchtpu/tree/main/support_matrices/nightly) track TPU v7x correctness and benchmark results from CI.

#### What's Next?

We want new open models from the major families (Qwen, DeepSeek, GLM, Kimi, Gemma, GPT-OSS) to run on TPU on the day they're released. Some of what's coming, from the [public roadmap](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/roadmap.md):

* An automated CI pipeline that onboards, benchmarks, and publishes support matrices for new open-weight models within 24 hours of release.
* Multi-host xP-yD disaggregation with [llm-d](https://llm-d.ai).
* KV cache offloading with TPU Sync to host RAM and Google Cloud Managed Lustre, for context windows of 1M+ tokens.
* Prefix-, cache-, and session-aware routing, and llm-d TPU memory-tier routing.
* PyTorch interfaces for [Tokamax](https://github.com/openxla/tokamax) kernels, shared with other projects in the TorchTPU ecosystem.
* Diffusion model serving through `vllm-omni`.

[Contributions are welcome!](https://github.com/vllm-project/vllm-torchtpu/blob/main/CONTRIBUTING.md)

### TPU Capacity for Contributors

We're also opening a Community Contribution TPU Capacity Pool. If you're building on `vllm-torchtpu`, you can apply for free TPU time, and core vLLM maintainers at Inferact will help with design and PR reviews.

[Apply for contributor TPU access here.](https://docs.google.com/forms/d/1Jm0uUXvcwYJRrvRojfS2RzMLjaqlt2a7lNyALFZs-nQ/viewform)

## Try It Out

* [Quickstart](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/getting_started/quickstart.md)
* [Cloud TPU Setup](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/getting_started/tpu_setup.md)
* [TPU v7x (Ironwood) Serving Recipes](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/deployment/v7x_recipes.md)
* [vLLM TorchTPU Recipes on Ironwood](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/vLLM-TorchTPU)
* [Kubernetes Deployment](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/deployment/kubernetes.md)
* [Contributing Guide](https://github.com/vllm-project/vllm-torchtpu/blob/main/CONTRIBUTING.md)

If you have questions or want to get involved, find us in `#sig-tpu` on the [vLLM Developer Slack](https://slack.vllm.ai).

## Acknowledgment

<!-- TODO: fill in acknowledgments for the Inferact, Google TorchTPU, and vLLM community contributors. -->

Thanks to the vLLM community, the Google TorchTPU team, and everyone who contributed to `tpu-inference`, whose work this builds on.
