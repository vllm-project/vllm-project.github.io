---
layout: post
title: "Notes on vLLM v.s. DeepSpeed"
author: "vLLM Team"
---

---
**TL;DR:**
- vLLM is as fast as DeepSpeed in common scenarios and faster than Deepspeed when outputs are long.
- DeepSpeed only outperforms vLLM in long prompt, short output use cases due to its Dynamic SplitFuse optimization. This optimization is on vLLM’s roadmap.
- vLLM’s mission is to build the fastest and easiest-to-use open-source LLM inference and serving engine. It is Apache 2.0 and community-owned with broad model and optimization support.

---

Recently, the DeepSpeed team published [a blog](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen) claiming 2x throughput improvement over vLLM by utilizing the Dynamic Splitfuse technique. We are happy to see the technology advancements from the open-source community. In this blog, we clarify the workloads that benefit from the Dynamic SplitFuse enhancement, which are quite narrow. For most workloads, vLLM is on par with or faster than DeepSpeed MII.

In this post, we will discuss the difference between the two systems, share our benchmarks, and discuss future steps.

### Performance Benchmark

In terms of performance optimization, we believe there are 2 key differences between vLLM and DeepSpeed:
DeepSpeed uses a conservative/suboptimal memory allocation scheme, which wastes memory when output lengths are large.
DeepSpeed uses Dynamic SplitFuse scheduling which gives speedup only when prompt lengths are much greater than output lengths.

Consequently, DeepSpeed wins when the workload is consistently long prompt and short output. In other cases, vLLM wins.

#### Scenario 1: Long Prompt Length, Short Output
In this scenario, we expect DeepSpeed to perform well due to Dynamic SplitFuse. However, the benefit we observe is not as significant as 2x.

<p align="center">
<picture>
<img src="/assets/figures/notes-vllm-vs-deepspeed/s1.png" width="50%">
</picture>
</p>

#### Scenario 2: All other cases
In this scenario, we observe vLLM perform better or on par with DeepSpeed.

<p align="center">
<picture>
<img src="/assets/figures/notes-vllm-vs-deepspeed/s2.png" width="50%">
</picture>
</p>


### vLLM’s Future: A True Community Project
We are committed to making vLLM the best open-source project incorporating the community’s best models, optimizations, and hardware. Coming out of UC Berkeley Sky Computing Lab, we are building vLLM truly in open source with the Apache 2.0 license.

The vLLM team prioritizes collaborations and we strive to keep the codebase with high quality code and easy to contribute. We are actively working on system performance; as well as new features like LoRA, Speculative Decoding, and better Quantization Support. Additionally, we are collaborating with hardware vendors like AMD, AWS Inferenetia, and Intel Habana to bring LLM to the broadest community.

Specifically for the Dynamic SplitFuse optimization, we are actively investigating the proper integration. If you have any questions and suggestions, please feel free to contact us on [GitHub](https://github.com/vllm-project/vllm). We also published the benchmark code [here](https://github.com/vllm-project/vllm/pull/1649).

### Appendix: Feature Comparison
DeepSpeed currently supports only basic functionalities. For example, it only supports 3 types of models and does not support popular features like stop strings and parallel sampling (beam search). We do expect the DeepSpeed open source are eager to catch up and we welcome the creative innovation in the market!

|                            |                   vLLM                  |                    DeepSpeed                    |
|----------------------------|:---------------------------------------:|:-----------------------------------------------:|
| Runtime                    | Python/PyTorch                          | Python/PyTorch                                  |
| Model implementation       | HuggingFace Transformers                | Custom implementation + converter for HF models |
| Server frontend            | Simple FastAPI server for demo purposes | Custom gRPC-based server                        |
| Scheduling                 | Continuous batching                     | Dynamic SplitFuse                               |
| Attention kernel           | PagedAttention & FlashAttention         | PagedAttention & FlashAttention                 |
| Custom kernels (for LLaMA) | Attention, RoPE, RMS, SILU              | Attention, RoPE, RMS, SILU, Embedding           |
| KV Cache allocation        | Near-optimal                            | Suboptimal/conservative                         |
| Supported models           | 16 different architectures              | LLaMA, Mistral, OPT                             |
| Sampling methods           | Random, parallel, beam search           | Random                                          |
| Stop criterion             | Stop strings, stop tokens, EOS          | EOS                                             |
