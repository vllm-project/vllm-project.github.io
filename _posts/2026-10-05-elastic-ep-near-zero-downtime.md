---
layout: post
title: "Achieving near-zero-downtime scaling in Elastic Expert Parallelism"
author: "Itay Alroy (NVIDIA)"
summary: "How asynchronous preparation and NIXL EP enabled near-zero-downtime Elastic EP scaling in vLLM."
image: /assets/figures/2026-10-05-elastic-ep-near-zero-downtime/figure00-pytorch-slo.png
tags:
  - large-scale-serving
  - elastic-ep
  - expert-parallelism
  - moe
  - nixl
  - cuda-graphs
---

Traditionally, changing the number of GPUs in a vLLM service required adding or removing entire model replicas or restarting the engine with a different GPU count. As models grow larger, each replica needs more GPUs, so scaling by replica adds or removes GPUs in increasingly large increments. Restarting vLLM interrupted serving for **197 seconds** (see our [benchmark](#performance-on-deepseek-v4-pro)).

In our [first Elastic EP post](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism), we showed how vLLM could add or remove GPUs from a running MoE engine without restarting it. While we avoided a restart, serving still stopped throughout reconfiguration as we initialized new workers, prepared the expanded communication groups, loaded model weights, and repeated kernel warmup and CUDA graph capture. In our DeepSeek V4 Pro benchmark, this pause lasted **100 seconds when scaling from 16 to 20 GPUs** and **56 seconds when scaling back to 16 GPUs** (full details in the [performance section](#performance-on-deepseek-v4-pro)).

In this post, we describe how asynchronous preparation and [NIXL EP](https://github.com/ai-dynamo/nixl/blob/main/examples/device/ep/README.md) enabled Elastic EP reconfiguration with near-zero downtime. In the same benchmark, we reduced downtime to **0.82 seconds when scaling from 16 to 20 GPUs** and **1.96 seconds when scaling back to 16 GPUs**, allowing us to scale up and down with almost no impact on serving performance.

## Overlapping topology preparation with serving

We reorganized vLLM's reconfiguration flow into two phases: background preparation and a short commit. Each existing worker prepares its part of the new topology on a separate thread while continuing to serve requests using its current communication groups and CUDA graphs. In parallel, joining workers initialize, receive model weights, and prepare their communication and GPU execution. During the commit phase, vLLM briefly pauses its schedulers, activates the prepared topology, and resumes serving. These changes were introduced in [asynchronous preparation](https://github.com/vllm-project/vllm/pull/47288) and [prepare/commit optimizations](https://github.com/vllm-project/vllm/pull/51885).

Preparation covers four main tasks:

1. **Initialize joining workers.** Start their processes and allocate storage for the model and execution.
2. **Prepare communication.** Set up the target communication groups and expand the EP communicator.
3. **Transfer model weights.** Transfer the required weights from existing workers to joining workers over NVLink or RDMA.
4. **Prepare execution.** Complete the required compilation, kernel warmup, tuning setup, and CUDA graph construction on joining workers.

After the commit phase, asynchronous Expert Parallel Load Balancing (EPLB) allows new workers to begin handling attention while existing workers handle expert computation. EPLB then redistributes experts across the expanded group in the background.

The main challenge is preparing communication and GPU execution without involving or disrupting the workers that are serving requests.

## Expand EP communication in place

Before NIXL EP, the EP communication backends available in vLLM used fixed membership: each instance was created for a specific set of ranks. Adding or removing ranks requires a new instance. Even when that instance can be created in the background, two problems remain.

First, the old and new instances need separate communication buffers and other resources, including memory registrations, IPC handles, and RDMA Queue Pairs (QPs). Keeping both instances alive increases communication memory usage and requires repeating setup rather than reusing existing resources.

Second, switching EP communicators changes the buffers and communication state used by the EP communication kernels. This invalidates previously captured CUDA graphs, which must be recaptured before serving can resume. That requires repeating warmup and graph construction for many batch shapes across all retained workers. In our DeepSeek V4 Pro benchmark, even with asynchronous preparation, performing warmup and graph recapture during the commit phase resulted in serving pauses of **58 seconds during scale-up** and **44 seconds during scale-down**.

To solve these problems, we designed **NIXL EP**, an EP communication library tailor-made for Elastic EP. It adds or removes ranks in place, reusing the same NIXL agent, buffers, and connections. New peers can be connected while existing workers continue serving, and only joining peers need new connections and registrations. NIXL EP also keeps captured EP dispatch and combine entries valid across reconfiguration, allowing retained workers to reuse their existing CUDA graphs.

During preparation, vLLM's background thread connects joining peers without activating them. The commit phase then activates the prepared peers without replacing the communicator.

![The same NIXL EP communicator progresses from the current active group to connected inactive joining peers, then activates the target group at commit.](/assets/figures/2026-10-05-elastic-ep-near-zero-downtime/figure02-nixl-ep-lifecycle.png)

*Figure 1. Connecting and activating new EP peers. Green ranks are active; gray ranks are connected but masked. During the commit phase, new peers are activated without replacing the agent or buffers.*

vLLM still creates new DP coordination groups and EPLB groups for the target topology, **but their communication runs outside the captured model forward pass**, so switching to them does not invalidate CUDA graphs.

## Preserve CUDA graphs on existing workers

A CUDA graph records GPU operations so that vLLM can replay a forward pass without repeatedly launching each operation from the CPU. A captured kernel node includes its launch configuration and argument values, including pointers to input and output tensors. Memory contents can change between replays, but the recorded pointers must remain valid. These requirements are part of the [CUDA graph execution contract](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).

vLLM [captures separate graphs for different batch shapes and execution modes](https://docs.vllm.ai/en/latest/design/cuda_graphs/) and runs warmup forward passes before each capture.

Preserving captured CUDA graphs requires more than keeping the communicator object alive. NIXL EP also keeps kernel launch configurations, graph-visible communication arguments, and input and output tensor addresses stable across reconfiguration. Its kernels read active group membership from device state that can be updated in place.

We also changed vLLM to keep the captured arguments, launch configurations, and input and output tensor addresses of the surrounding compute kernels stable across reconfiguration:

- We reuse retained workers' fused-MoE execution objects instead of rebuilding them for a different active EP size.
- We keep the addresses and layouts of graph-visible workspaces and dispatch outputs stable across reconfiguration so that captured expert kernels can continue using them. Dispatch outputs also serve as inputs to expert computation. If their addresses or layouts changed with EP size, captured expert kernels would still use the old pointers and offsets.
- We preserve the backing storage of expert-routing tensors and update mappings in place. This allows EPLB to change the mapping contents without moving the tensors used by captured kernels.

Together, these changes allow retained workers to [reuse their existing CUDA graphs](https://github.com/vllm-project/vllm/pull/54985) after both scale-up and scale-down.

## Prepare joining workers independently

Existing workers have graphs to preserve. Joining workers have none.

Joining workers normally prepare by running model forward passes with EP dispatch and combine, as well as exchanging batch information with other workers. These steps require other workers to participate, but existing workers are already serving real requests on the current topology.

NIXL EP lets each joining worker temporarily mask every remote EP peer during warmup and graph preparation. Only that worker's local rank is active for EP communication. We also modified vLLM so that these preparation passes do not wait for DP batch information from other workers. This lets joining workers run their dummy batches and capture CUDA graphs without requiring existing workers to participate.

![Existing workers serve with their current peers active and joining peers masked. Each joining worker keeps only itself active for local warmup and graph capture, masking all remote peers, including other joining workers.](/assets/figures/2026-10-05-elastic-ep-near-zero-downtime/figure04-independent-capture.png)

*Figure 2. Green ranks are active for EP communication; gray ranks are masked. Existing workers serve together while each joining worker masks all remote EP peers for preparation.*

After capture, NIXL EP can reactivate peers without invalidating the graphs. Joining workers reuse their locally captured graphs when they begin serving as part of the expanded group.

## Performance on DeepSeek V4 Pro

### Scale-up and scale-down timings

We benchmarked DeepSeek V4 Pro on GB200 GPUs, scaling from 16 to 20 GPUs and back. The table shows which stages run while serving is paused and which run in the background while existing workers continue serving.

| Stage | A: Blocking | B: Async preparation | C: Async preparation + graph reuse |
| --- | --- | --- | --- |
| Initialize joining workers | Serving paused | Background | Background |
| Prepare target communication | Serving paused | Background | Background |
| Transfer model weights | Serving paused | Background | Background |
| Warmup and graph capture on joining workers | Serving paused | Serving paused | Background |
| Warmup and graph capture on existing workers | Serving paused | Serving paused | Not needed (existing graphs stay valid) |
| Activate the target topology | Serving paused | Serving paused | Serving paused |

Configuration C is supported **only by NIXL EP**.

- **Downtime:** the observed serving interruption, when in-flight requests stop progressing and new requests are not admitted.
- **Total time:** the duration of the full scaling operation, including preparation.

The two durations are measured separately.

| Metric | A: Blocking | B: Async preparation | C: Async preparation + graph reuse |
| --- | ---: | ---: | ---: |
| 16 to 20 GPUs: downtime | 99.62 s | 58.04 s | **0.82 s** |
| 20 to 16 GPUs: downtime | 56.25 s | 44.07 s | **1.96 s** |
| 16 to 20 GPUs: total time | 99.72 s | 97.27 s | 92.34 s |
| 20 to 16 GPUs: total time | 56.21 s | 51.46 s | 9.32 s |

All configurations passed numerical correctness checks, including checks on every joining worker.

### Maintaining the SLO while adding GPUs

We served DeepSeek V4 Pro on 16 GPUs, then gradually increased the incoming request rate from about 8 to 16 requests per second. We replayed the same 11,391-request scenario against three deployments: one stayed at 16 GPUs, one restarted with 20 GPUs, and one scaled to 20 GPUs using Elastic EP with NIXL EP. The service-level objective (SLO) was a rolling p95 time to first token below 5 seconds.

![Elastic EP stays below the SLO throughout the scenario. During preparation, its serving performance matches the static 16-GPU deployment. After scaling, it matches the recovered 20-GPU deployment's performance.](/assets/figures/2026-10-05-elastic-ep-near-zero-downtime/figure00-pytorch-slo.png)

*Figure 3. End-to-end serving under rising request load. Curves show 30-second rolling p95 outcomes, with failed requests assigned a 30-second penalty rather than a measured TTFT.*

- Elastic EP completed all 11,391 requests and stayed below the SLO throughout the scenario.
- During preparation, Elastic EP matched the static 16-GPU deployment's serving performance, with no visible degradation from background preparation.
- After scaling, Elastic EP matched the recovered 20-GPU deployment's performance, with no visible post-scale degradation.

## Padding in expert-kernel inputs

To keep CUDA graphs stable across reconfiguration, we size dispatch outputs (the inputs to expert kernels) for the maximum EP capacity. Expert inputs are padded even in a static deployment, but reserving maximum capacity increases the padding. Backends that do not efficiently skip padded token slots incur more overhead as a result.

In a separate capacity benchmark, we kept 16 workers active and compared a maximum EP capacity of 64 against 16. These measurements use DeepSeek-V3 on 16 GB200 GPUs, with backend-specific weight formats.

| Expert backend | Decode throughput change | Prefill throughput change |
| --- | ---: | ---: |
| Batched DeepGEMM | 0.08% higher | 0.97% lower |
| FlashInfer CuteDSL | 1.21% higher | 0.89% lower |
| Batched Triton | 18.31% lower | 19.21% lower |
| Batched CUTLASS FP8 | 77.49% lower | 63.90% lower |

Throughput with DeepGEMM and CuteDSL is effectively unaffected in this comparison. Triton and CUTLASS still process parts of the padded buffers, so their padding overhead grows with the configured maximum EP capacity.

We also plan to reduce NIXL EP's extra memory cost with [CUDA virtual memory management (VMM)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html): reserve stable virtual addresses, but back only the regions needed by the active group with physical HBM. This would let buffers grow without changing the addresses recorded in CUDA graphs or allocating physical HBM for the full maximum capacity up front.

## Try Elastic EP

Background preparation, CUDA graph reuse, and independent graph capture on joining workers are all available today in the latest [vLLM](https://pypi.org/project/vllm/) and [NIXL](https://pypi.org/project/nixl/) releases. Install them with pip and start using Elastic EP.

For a complete usage example, see the [`cuda_graphs_heavy_nixl_ep` case in vLLM's Elastic EP test](https://github.com/vllm-project/vllm/blob/v0.30.0/tests/distributed/test_elastic_ep.py#L224). The example includes the server configuration and scaling API calls. It scales from two GPUs to four and back while sending requests, checking serving during preparation and model accuracy around each resize.

## What comes next

### Restore warm workers with Dynamo Snapshot

We have effectively solved the long serving interruption. The next challenge is getting new capacity ready sooner. We plan to integrate [Dynamo Snapshot](https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/) so joining workers can restore initialized process and CUDA state instead of repeating compilation, kernel warmup, graph construction, and other startup work.

Communication for the target topology must be prepared separately because it is not restored from the snapshot. Restoring large model weights from checkpoint storage can dominate restore time, so we plan to keep Elastic EP's GPU-to-GPU weight transfer from existing workers instead.

### Achieve state-of-the-art EP communication with NIXL EP

We aim for state-of-the-art dispatch and combine performance with NIXL EP while maintaining full support for Elastic EP and fault tolerance. An upcoming NIXL EP performance update is the next step toward that goal (see the [NIXL EP roadmap RFC](TODO_NIXL_EP_ROADMAP_RFC_URL)).

### Connect Elastic EP to Dynamo and llm-d orchestration

With the engine-level capability in place, the next step is to use Elastic EP from orchestrator autoscalers. Work toward this integration is already underway across the vLLM, Dynamo, and llm-d communities.

These efforts include [vLLM's external-load-balancer work](https://github.com/vllm-project/vllm/pull/43202) and [Dynamo's Elastic EP proposal](https://github.com/ai-dynamo/dynamo/issues/13121).

If you run inference at scale, we encourage you to try Elastic EP and share your workloads and operational requirements. If you build inference infrastructure, we welcome contributions to vLLM Elastic EP and NIXL EP. Let's get Elastic EP running in production.

## Acknowledgements

- Tyler Michael Smith and Sage Moore from Red Hat for their invaluable feedback and thorough reviews throughout this work.
- Linoy Geva (NVIDIA) for her work on serving performance during preparation.
- The entire NIXL team for their efforts on NIXL EP.
- Tzu-Ling Kan, Julien Mancuso, and the entire Dynamo team for their efforts to support Elastic EP in Dynamo.
