---
layout: post
title: "Distributed Inference with vLLM"
author: "vLLM Team"
image: /assets/logos/vllm-logo-only-light.png
noindex: true
---

### Motivation

Serving large models often leads to memory bottlenecks, such as the dreaded **CUDA out of memory** error. To tackle this, there are two main solutions:

1. **Reduce Precision** – Utilizing FP8 and lower-bit quantization methods can reduce memory usage. However, this approach may impact accuracy and scalability, and is not sufficient by itself as models grow beyond hundreds of billions of parameters.  
2. **Distributed Inference** – Spreading model computations across multiple GPUs or nodes enables scalability and efficiency. This is where distributed architectures like tensor parallelism and pipeline parallelism come into play.

### vLLM Architecture and Large Language Model Inference Challenges

LLM inference poses unique challenges compared to training:

* Unlike training, which focuses purely on throughput with known static shapes, inference requires low latency and dynamic workload handling.  
* Inference workloads must efficiently manage KV caches, speculative decoding, and prefill-to-decode transitions.  
* Large models often **exceed single-GPU capacity**, requiring advanced **parallelization strategies**.

To address these issues, vLLM provides:

* **Tensor parallelism** to shard each model layer across multiple GPUs within a node.  
* **Pipeline parallelism** to distribute contiguous sections of model layers across multiple nodes.  
* **Optimized communication kernels and control plane architecture** to minimize CPU overhead and maximize GPU utilization.

## GPU Parallelism Techniques in vLLM

### Tensor Parallelism

#### Problem: Model Exceeds Single GPU Capacity

As models grow, a single GPU cannot accommodate them, necessitating multi-GPU strategies. Tensor parallelism **shards model weights across GPUs**, allowing concurrent computation for lower latency and enhanced scalability.

This approach, originally developed for training in [Megatron-LM (Shoeybi et al., 2019\)](https://arxiv.org/abs/1909.08053), has been adapted and optimized in vLLM for inference workloads.

<figure>
  <img src="/assets/figures/distributed-inference/tp_strategies.png" />
</figure>

Tensor Parallelism relies on two primary techniques:

1. Column Parallelism: Splitting weight matrices along columns and concatenating results after computation.  
2. Row Parallelism: Splitting matrices along rows, summing partial results post-computation.

<figure>
  <img src="/assets/figures/distributed-inference/column_row_parallel.png" />
</figure>

As a specific example, let’s break down how this parallelism works for the MLP (multi-layer perceptron) layers in Llama models:

* Column parallelism applies to up-projection operations.  
* Element-wise activation functions (e.g., SILU) operate on sharded outputs.  
* Row parallelism is used in down-projection, with an **all-reduce** operation to aggregate final results.

Tensor parallelism ensures that inference computations are distributed across multiple GPUs, maximizing the memory bandwidth and compute available. When used, we can achieve latency improvements from effectively multiplying memory bandwidth. This occurs because sharding model weights allows multiple GPUs to access memory in parallel, reducing bottlenecks that a single GPU might encounter. 

<figure>
  <img src="/assets/figures/distributed-inference/tensor_parallelism.png" />
<figcaption>
Source: <a href="https://sebastianraschka.com/blog/2023/pytorch-memory-optimization.html" target="_blank">Sebastian Raschka, 2023</a>.
</figcaption>
</figure>

However, it requires **high-bandwidth interconnects** between each GPU, like NVLink or InfiniBand, to minimize overhead from the increased communication costs.

### Pipeline Parallelism

#### Problem: Model Exceeds Multi-GPU Capacity

For extremely large models (e.g., DeepSeek R1, Llama 3.1 405B), a single node may not suffice. Pipeline parallelism **shards models across nodes**, each handling specific contiguous model layers.

#### How It Works

* Each GPU loads and processes a distinct set of layers.  
* **Send/Receive Operations:** Intermediate activations are transmitted between GPUs as computation progresses.

**This results in lower communication overhead** compared to tensor parallelism since data transfer occurs once per pipeline stage.

Pipeline Parallelism reduces memory constraints across GPUs but does not inherently decrease inference latency as tensor parallelism does. To mitigate throughput inefficiencies, vLLM incorporates **advanced pipeline scheduling**, ensuring that all GPUs remain active by optimizing micro-batch execution.

### Combining Tensor Parallelism and Pipeline Parallelism

As a general rule of thumb, think of the applications of parallelism like this:

* Use **pipeline parallelism across nodes** and **tensor parallelism within nodes** when interconnects are slow.  
* If interconnects are efficient (e.g., NVLink, InfiniBand), **tensor parallelism can extend across nodes**.  
* Combining both techniques intelligently **reduces unnecessary communication overhead** and maximizes GPU utilization.

#### Performance Scaling and Memory Effects

While the basic principles of parallelization suggest linear scaling, in practice, the **performance improvements can be super-linear** due to memory effects. With either Tensor Parallelism or Pipeline Parallelism, throughput improvements can arise in non-obvious ways due to the memory available for KV Cache increasing super-linearly.  

<figure>
  <img src="/assets/figures/distributed-inference/kv_cache_effects.png" />
</figure>

This super-linear scaling effect occurs because larger caches allow for larger batch sizes for processing more requests in parallel and better memory locality, resulting in improved GPU utilization beyond what might be expected from simply adding more compute resources. In the above graph you can see between TP=1 and TP=2, we are able to increase the amount of KV Cache blocks by 13.9x which allows us to observe **3.9x more token throughput** \- much more than the linear 2x we would expect from using 2 GPUs instead of 1\.

### Further Reading

For readers interested in diving deeper into the techniques and systems that influenced vLLM's design:

* [Megatron-LM (Shoeybi et al., 2019\)](https://arxiv.org/abs/1909.08053) introduces the foundational techniques for model parallelism in large language models  
* [Orca (Yu et al., 2022\)](https://www.usenix.org/conference/osdi22/presentation/yu) presents an alternative approach to distributed serving using iteration-level scheduling  
* [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) and [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) provide complementary perspectives on optimizing transformer inference

### Conclusion

Serving large models efficiently requires a combination of **Tensor Parallelism**, **Pipeline Parallelism**, and **performance optimizations** like **Chunked Prefill**. vLLM enables scalable inference by leveraging these techniques while ensuring adaptability across different hardware accelerators. As we continue to enhance vLLM, staying informed about new developments such as **expert parallelism for Mixture of Experts (MoE)** and **expanded quantization support** will be crucial for optimizing AI workloads.

##### Come to the Bi-weekly Office Hours to learn more about LLM inference optimizations and vLLM\!

### Acknowledgement

Sangbin Cho (xAI) for the origination of some of the figures.
