---
layout: post
title: "vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP"
author: "vLLM Team"
image: /assets/figures/2025-12-17-large-scale-serving/prefill_throughput.png
---

# Introduction

In v0.11.0, the last code from vLLM V0 engine was removed, marking the complete migration to the improved [V1 engine](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) architecture. This achievement would not have been possible without vLLM’s community of 1,969 contributors, authoring over 950 commits in the past month (as of 12/18/25).

These efforts have been validated by vLLM’s inclusion in the SemiAnalysis open source InferenceMax performance [benchmarks](https://inferencemax.semianalysis.com/). In addition, vLLM is proud to be trusted in production by teams at Meta, LinkedIn, Red Hat, Mistral, and HuggingFace.

DeepSeek-style disaggregated serving and sparse mixture-of-experts (MoE) model deployments remain state-of-the-art for high-performance LLM inference. This article outlines the key optimizations the vLLM team has built to push throughput even further, including:

* Async scheduling  
* Dual-batch overlap  
* Disaggregated serving  
* CUDA graph mode `FULL_AND_PIECEWISE`  
* DeepGEMM enabled by default  
* DeepEP kernels integration  
* Expert parallel load balancing  
* SiLU kernel for DeepSeek-R1

For further reference, we recommend these excellent writeups by the llm-d, PyTorch, Dynamo, and Anyscale teams on [large scale serving](https://llm-d.ai/blog/llm-d-v0.3-expanded-hardware-faster-perf-and-igw-ga), [disaggregated serving](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/), [distributed inference](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/#boosting_inference_performance_on_nvidia_gb200_nvl72_by_30x), and [wide-EP](https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm) using vLLM.

# Results

Recent [community benchmarks](https://llm-d.ai/blog/llm-d-v0.3-expanded-hardware-faster-perf-and-igw-ga#wide-ep-performance) on a Coreweave H200 cluster connected using Infiniband with ConnectX-7 NICs now show a sustained throughput of 2.2k tokens/s per H200 GPU in production-like, multi-node deployments.

This marks a significant increase over earlier benchmarks, which showed ~1.5k tokens/s per GPU. This gain is a direct result of ongoing optimization work, including kernel improvements (silu-mul-quant fusion, Cutlass QKV kernels, TP attention bug fixes) and the implementation of Dual Batch Overlap (DBO) for decode.

This performance allows operators to realize immediate benefits by consolidating workloads and reducing the number of replicas needed for a target QPS, ultimately lowering token-per-dollar cost.

<p align="center">
<img src="/assets/figures/2025-12-17-large-scale-serving/prefill_throughput.png" width="100%">
<br>
<em>Prefill Results</em>
</p>

<p align="center">
<img src="/assets/figures/2025-12-17-large-scale-serving/decode_throughput.png" width="100%">
<br>
<em>Decode Results</em>
</p>

# Key Components

## Wide-EP

Deploying frontier models like the DeepSeek-V3 model family for large scale serving requires two major considerations:

- Sparse expert activation: in DeepSeek-R1, only 37B of the model’s 671B total parameters are active with each forward pass  
- KV cache management: tensor parallel deployment is not optimal for DeepSeek’s multi-head latent attention (MLA) attention architecture, since latent projections are duplicated across shards

Expert parallelism (EP) is a deployment pattern that leverages these characteristics to maximize effective KV cache, and is supported in vLLM via the `--enable-expert-parallel` flag. In this pattern, a single set of experts are shared across ranks in the deployment. During a forward pass, tokens are routed between ranks to be processed by the appropriate expert.

<p align="center">
<img src="/assets/figures/2025-12-17-large-scale-serving/wide_ep.gif" width="100%">
<br>
<em>Wide-EP token routing</em>
</p>

Wide-EP combines EP with data parallelism (DP). Data parallel deployments can be launched with either the `mp` or `ray` data parallel backends, offering simpler setup within a Ray cluster. The benefit over tensor parallelism is shown in the following figure, which shows memory usage per GPU for DeepSeek-V3 using tensor parallel and expert parallel sharding strategies. 

The TP strategy shows 34GB free device memory per H200, but for MLA models, each rank must duplicate latent attention projections. In a DP deployment, attention layers are duplicated so that latent projections are independent across ranks, increasing effective batch size across the deployment.

<p align="center">
<img src="/assets/figures/2025-12-17-large-scale-serving/kv_cache.png" width="100%">
</p>

Increasing the expert parallelism degree increases synchronization overhead between ranks. To address this, vLLM has integrated support for the [DeepEP](https://github.com/deepseek-ai/DeepEP) high throughput and low latency all-to-all kernels. In addition, vLLM supports Perplexity [MoE kernels](https://github.com/perplexityai/pplx-kernels) and a NCCL-based AllGather-ReduceScatter all-to-all. See the vLLM MoE [kernel docs](https://docs.vllm.ai/en/latest/design/moe_kernel_features/) for information on the all-to-all backends available in vLLM.

<div style="width: 100vw; position: relative; left: 50%; right: 50%; margin-left: -50vw; margin-right: -50vw;">
<p align="center">
<img src="/assets/figures/2025-12-17-large-scale-serving/a2a_backends.png" style="max-width: 60%; height: auto;">
<br>
<em>vLLM <a href="https://docs.vllm.ai/en/latest/design/moe_kernel_features/#fused-moe-modular-all2all-backends">all-to-all backends</a></em>
</p>
</div>

## Dual-batch Overlap (DBO)

vLLM has integrated support for DeepSeek’s [microbatching strategy](https://github.com/deepseek-ai/profile-data) as dual batch overlap (DBO), available via `--enable-dbo` flag from the command line. This strategy overlaps compute and collective communication to increase GPU utilization. In particular, vLLM implements this as follows:

1. A collective `all_reduce` across ranks to agree microbatching will be beneficial, with minimum threshold adjustable via `--dbo-decode-token-threshold`
2. The main thread creates microbatch worker threads, which complete CUDA graph capture  
3. vLLM’s modular MoE all-to-all kernel base class coordinates microbatch worker launches, yielding control while waiting for GPU work to complete

Below is a profiling trace from a DeepSeek decode workload **without** DBO. The “MoE Dispatch/Combine” section shows the outsize duration spent in collective communication, despite the small compute load.

<p align="center">
<img src="/assets/figures/2025-12-17-large-scale-serving/dbo_before.png" width="100%">
<br>
<em>Before DBO</em>
</p>

The following trace shows the same workload **with** DBO. The first microbatch worker thread initiates and completes MoE dispatch, then immediately yields to the second microbatch worker thread. Next, the second thread completes its own dispatch, yielding back to the first thread once it completes. Finally, the first worker completes its combine before yielding back to the second microbatch worker.

This results in higher GPU utilization in deployments where communication overhead is high, as is the case in deployments with high expert parallelism degree.

<p align="center">
<img src="/assets/figures/2025-12-17-large-scale-serving/dbo_after.png" width="100%">
<br>
<em>After DBO</em>
</p>

## Expert Parallel Load Balancing (EPLB)

MoE expert layers are optimized for balanced load across experts at train time, but at inference time, real workloads may cause imbalanced token routing. See NVIDIA’s [experimental results](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/#experimental_results) on MoE expert routing for statistics on the difference in expert load balance between workloads.

In a wide-EP setup, this means some EP ranks could stay idle, while others process large batches of tokens. To alleviate this, vLLM implements the hierarchical and global load balancing policies from DeepSeek's [expert parallel load balancer](https://github.com/deepseek-ai/EPLB) (EPLB). EPLB is controlled by the `--enable-eplb` CLI flag, with configurable window size, rebalance interval, redundant experts, and logging options.

<p align="center">
<img src="/assets/figures/2025-12-17-large-scale-serving/eplb.gif" width="100%">
<br>
<em>EPLB in action</em>
</p>

To implement EPLB, each MoE forward pass records per-token load, and a sliding window aggregates these statistics across EP ranks. When the rebalance interval is reached, the load balancer computes a new logical-to-physical expert mapping and orchestrates a weight shuffle so the new placement takes effect without restarting the model.

## Disaggregated Serving

The disaggregated prefill/decode serving pattern, described by Hao AI Lab in the 2024 DistServe [paper](https://hao-ai-lab.github.io/blogs/distserve-retro/), is especially useful for expert parallel deployments.

<p align="center">
<img src="/assets/figures/2025-12-17-large-scale-serving/disaggregated_serving.gif" width="100%">
<br>
<em>P/D disaggregation in action</em>
</p>

Since experts are distributed across ranks, a request's tokens starting on one rank may require processing by an expert on any other rank in the EP group. This requires synchronization between MoE layers (and dummy passes if a rank goes unused) so that layer combine collectives are ready to receive tokens at the appropriate time.

This means a single compute-bound prefill request can delay the forward pass of the entire EP group, amplifying the benefit of disaggregated serving. In addition, DeepSeek deployments can be configured to exclusively use the DeepEP kernel suited to their workload (high throughput vs. low latency).

# Deployment Paths

## llm-d

llm-d is a Kubernetes-native distributed inference serving stack providing well-lit paths for anyone to serve large generative AI models at scale. llm-d helps you achieve the fastest "time to state-of-the-art (SOTA) performance" for key OSS models across most hardware accelerators and infrastructure providers. For more details, check out llm-d's Wide EP [well lit path](https://github.com/llm-d/llm-d/tree/main/guides/wide-ep-lws) to replicate the results in this post.

<p align="center">
<img src="/assets/figures/2025-12-17-large-scale-serving/llm-d.png" width="100%">
</p>

## Dynamo

Dynamo is designed for high throughput and low latency production deployments of LLMs. Features such as KV aware routing, KV Block Manager for cache offloading, and Planner for dynamic load matching enable you to hit tighter SLAs while scaling across more GPUs. vLLM and wide-EP serving is natively supported in Dynamo with all of these features. For more details check out [Dynamo](https://docs.nvidia.com/dynamo/latest/index.html) and the [example recipe](https://github.com/ai-dynamo/dynamo/pull/4463/files#diff-363ddf6952864a610a1047f6b99c52461d6de9a4e198f89eb49d34f009a4d22b) to replicate the performance in this blog post.

<p align="center">
<img src="/assets/figures/2025-12-17-large-scale-serving/dynamo.png" width="100%">
</p>

## Ray Serve LLM

Building on Ray Serve primitives, Ray Serve LLM provides first-class serving patterns for [prefill/decode disaggregation](https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/prefill-decode.html), [data parallel attention](https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/data-parallel.html) and [prefix cache-affinity request routing](https://docs.ray.io/en/latest/serve/llm/architecture/routing-policies.html), focusing on modularity and ease of deployment on Ray clusters (including KubeRay on  Kubernetes). A key differentiator is its seamless integration with the broader Ray ecosystem, including data processing and reinforcement learning (RL). 

The framework integrates with NIXL and LMCache connectors for efficient KV transfer, and leverages Ray's distributed computing primitives to enable independent autoscaling of each phase based on load characteristics. Together, the solution provides a flexible and programmable layer for inference workloads that can be easily extended and composed to implement diverse serving patterns.

<p align="center">
<img src="/assets/figures/2025-12-17-large-scale-serving/ray_serve_llm.png" width="100%">
</p>

# Roadmap

vLLM is continuously in improvement, with the following efforts currently in progress:

* Elastic expert parallelism  
* Long context serving  
* KV cache transfer via CPU  
* Full determinism and batch invariance  
* Large MoE optimizations, e.g. op fusion for DeepSeek-R1 and gpt-oss models  
* Improve FlashInfer integration for latest kernels, e.g. SwapAB  
* Support independent TP sizes in disaggregated serving deployments  
* GB200 Optimizations for large scale serving

For the most up-to-date reference, see [roadmap.vllm.ai](http://roadmap.vllm.ai).

# Summary

* vLLM has fully migrated to the V1 engine, which demonstrates high throughput for DeepSeek-style MoE deployments and achieving 2.2k tok/s/H200 with wide-EP.  
* Wide-EP maximizes KV cache efficiency for MLA architectures, while dual-batch overlap and EPLB reduce communication bottlenecks and load imbalance.  
* Disaggregated prefill/decode further optimizes prefill and decode deployments for MoE workloads, with deployment options such as llm-d, Dynamo, and Ray Serve LLM.
