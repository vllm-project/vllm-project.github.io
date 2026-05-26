---
layout: post
title: "Native RL APIs in vLLM"
author: "Aaron Hao, Sumanth Hegde, Kourosh Hakhamaneshi, and the vLLM team"
image: /assets/figures/2026-05-12-native-rl-apis/weight_transfer_nccl.svg
social_image: /assets/figures/2026-05-12-native-rl-apis/weight_transfer_nccl.svg
read_time_minutes: 12
tags:
  - reinforcement-learning
  - async-rl
---

As post-training workloads continue to scale, we've seen widespread adoption of vLLM as the inference engine of choice. However, two issues repeatedly arise:

1. Weight syncing between training and inference is implemented in an ad-hoc fashion and duplicated across frameworks.
2. Asynchronous RL setups become fragile at scale, especially in P/D and DPEP deployments.

In this post, we introduce two improvements in vLLM:

1. Native weight syncing APIs that provide a standard interface for RL frameworks.
2. Improved support for asynchronous RL, including a new pause mode and fixes for deadlocks in DPEP setups.

## Native Weight Syncing APIs in vLLM

### Background

In online RL setups, vLLM model weights must be synced periodically to make sure that the rollouts generated are from the latest or a recent version of the model weights, so as to provide more useful feedback.

<p align="center">
<img src="/assets/figures/2026-05-12-native-rl-apis/rl_system_overview.png" width="80%">
<br>
<em>Figure 1: RL system overview.</em>
</p>

Traditionally, this weight loading has been handled by each RL framework separately, typically by extending vLLM workers with custom logic for receiving and loading weights. While this works, it leads to a few issues:

* **Added complexity**: Framework authors have to implement and maintain custom worker extensions, and it would be good for native support for popular transport strategies.
* **Duplicated effort**: Most RL frameworks end up having very similar implementations (e.g., packed tensor transfer, RPC endpoints).
* **Version locking**: Frameworks typically have ad-hoc ways of dealing with pre/post-processing of received weights to enable vLLM workers to load them, which can lead to version locked implementations.

### New APIs in vLLM

We introduce native weight syncing APIs in vLLM to standardize this. The weight transfer APIs consist of four phases with a pluggable backend:

1. **Initialization** (`init_weight_transfer_engine`): Establishes the communication channel between the trainer and inference workers. Called once before the training loop begins.
2. **Start weight update** (`start_weight_update`): Start a weight update. Called after each training step (or batch of steps). Prepares the vLLM workers to receive weights.
3. **Update weights** (`update_weights`): Update all or a subset of weights from the trainer to the inference engine. Can be invoked multiple times for chunked weight transfers.
4. **Finish weight update** (`finish_weight_update`): Finish the current weight update. Runs any necessary post-processing (e.g., quantization).

Corresponding APIs are implemented at the API server and the engine level.

Currently, we support the following backends:

1. **NCCL**: Uses NCCL broadcast operations for weight transfer between training and inference workers on separate GPUs.
2. **IPC**: Uses CUDA IPC for same-device weight transfer via shared memory handles.

Both backends support an optimized packed implementation to minimize serialization overhead.

The core transport logic is implemented with a pluggable `WeightTransferEngine` abstraction to separate weight transport from the worker implementation, allowing users to easily bring in their own implementations. The core idea is that **initialization** and **update weights** phases are typically customized by RL framework developers and include *transport* logic, while start and finish are control messages, and involve transport-agnostic pre/postprocessing in vLLM.

### Example

For example, here is how the different operations would look for weight transfer via NCCL with FP8 quantization on vLLM:

<p align="center">
<img src="/assets/figures/2026-05-12-native-rl-apis/weight_transfer_nccl.svg" width="80%">
<br>
<em>Figure 2: Weight transfer via NCCL with FP8 quantization on vLLM.</em>
</p>

The new APIs can be used as follows:

**1. Configure the engine for weight transfer**

```py
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(
    model="my-model",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
)
```

**2. Initialize communication state**: Initialize communication state between trainer and inference engine. The trainer's rank 0 process and all inference workers join a shared NCCL process group.

```py
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest

# Initialization for inference
llm.init_weight_transfer_engine(
    WeightTransferInitRequest(      # <--- initialization parameters
        init_info=dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,          # <--- offset accounts for trainer rank 0
            world_size=world_size,  # <--- trainer + all inference workers
        )
    )
)

# Initialization for training
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
)

group = NCCLWeightTransferEngine.trainer_init(
    dict(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
    )
)
```

**3. Send weights from the trainer**: Start the weight transfer on the trainer. `WeightTransferEngine` implements a `trainer_send_weights` method that takes in an iterable list of parameters and initializes the transfer for all or a subset of parameters. Users can also implement their own send functionality. Here, we can also leverage packed tensor broadcasting for higher throughput transfer by batching multiple small tensors into a larger buffer.

```py
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)

trainer_args = NCCLTrainerSendWeightsArgs(
    group=group,
    packed=True,  # use packed broadcasting for efficiency
)

# send weights from an `AutoModelForCausalLM` instance
NCCLWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

**4. Receive weights in the inference engine**

```py
from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

# executed asynchronously while trainer sends weights
llm.start_weight_update()
llm.update_weights(
    WeightTransferUpdateRequest(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=True,
        )
    )
)
llm.finish_weight_update()
```

### Customizing Weight Transfer

One of the primary goals for the weight transfer APIs is to enable RL frameworks to implement custom weight transfer strategies with vLLM. With the new APIs, users would implement and register a custom `WeightTransferEngine`:

```python
from dataclasses import dataclass
from typing import Iterator, Callable, Any
from torch import Tensor
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)


# define custom dataclasses for initialization and update weights metadata
@dataclass
class MyInitInfo(WeightTransferInitInfo):
    """Custom initialization info."""
    ...


@dataclass
class MyUpdateInfo(WeightTransferUpdateInfo):
    """Custom update info."""
    ...

# custom weight transfer engine
class MyWeightTransferEngine(WeightTransferEngine):
    init_info_cls = MyInitInfo
    update_info_cls = MyUpdateInfo

    def init_transfer_engine(self, init_info: MyInitInfo):
        ...

    def receive_weights(
        self,
        update_info: MyUpdateInfo,
        load_weights: Callable[[list[tuple[str, Tensor]]], None],
    ):
        ...

    @classmethod
    def trainer_send_weights(
        cls,
        iterator: Iterator[tuple[str, Tensor]],
        trainer_args: dict[str, Any] | Any,
    ):
        ...

# finally, register the weight transfer engine
from vllm.distributed.weight_transfer import WeightTransferEngineFactory
WeightTransferEngineFactory.register_engine("my_weight_transfer", MyWeightTransferEngine)
```

Note that the `trainer_send_weights` method is optional to use. It encodes send logic used on the trainer and users are not required to structure their send logic in this way.

The above simple API can enable many advanced use-cases. As a prototype, we demonstrate how sharded weight transfer in the style of [Etha](https://github.com/cmriat/Etha) can be implemented [here](https://github.com/hao-aaron/vllm/blob/89c951b3296578c60cbb82e05ca3d1734364ba8c/examples/rl/sharded_reloading/README.md).

## Improved Pause/Resume Support for Asynchronous RL

In asynchronous RL, weights are updated while inference requests are still in flight. Typically, weight syncing involves three operations in async RL: pausing generation, transferring updated weights, and then resuming generation. Users choose how to deal with in-flight requests (e.g., abort all running requests, or resume generation from previously generated tokens), as well as keeping or discarding the KV cache.

<p align="center">
<img src="/assets/figures/2026-05-12-native-rl-apis/async_rl.svg" width="80%">
<br>
<em>Figure 3: Asynchronous RL system diagram, inspired by <a href="https://arxiv.org/pdf/2505.24298v3">AReaL</a>. Training and generation overlap, with training utilizing 4 samples for each step. After a training step finishes, all the engines are paused, weights are updated, the KV cache is discarded, and then the engines are resumed. KV cache is recomputed on resumption and generation progresses as before.</em>
</p>

### Keep Mode for Pause/Resume

To safely update weights while the inference engine is running, vLLM provides `pause_generation` and `resume_generation` methods. The same functionality is available in the HTTP servers as `POST /pause` and `POST /resume` APIs. Previously, `AsyncLLMEngine.pause_generation` supported two modes:

* abort all requests
* wait for requests to finish

We add a third option: **keep mode**. The different modes are compared in the table below:

| Mode | Explanation | Client-side impact | Asynchronous RL possible? |
| -------- | -------- | -------- |  -------- |
| `abort` | Aborts all ongoing requests | Client must handle retries |  Yes |
| `wait` | Wait for all ongoing requests | Client need not retry | No, generation needs to finish before weight update |
| `keep` | Pause ongoing requests | Client need not retry | Yes |


Keep mode can be used as follows:

```py
# pause - preserve ongoing requests
await engine.pause_generation(mode="keep")
# update weights here
# resume
await engine.resume_generation()
```

In keep mode:

* Ongoing requests are paused but not discarded.
* The scheduler is stopped, but state is preserved.

### Fixing Deadlocks in DPEP Setups

Large scale asynchronous RL requires careful coordination for in-flight weight updates in DPEP deployments. In vLLM, a `DPCoordinator` ensures that generation is carefully coordinated across vLLM ranks to prevent deadlocks. More specifically, each DP rank executes a forward pass while there are active requests scheduled in any of the DP ranks.

<p align="center">
<img src="/assets/figures/2026-05-12-native-rl-apis/dp_generate.svg" width="80%">
<br>
<em>Figure 4: DP-coordinated generation across vLLM ranks.</em>
</p>

Previously, asynchronous RL in DP deployments with vLLM often led to deadlocks, primarily because some engines would have received a pause signal while others would be actively handling requests and waiting for all engines to join. One of the reasons this occurred was because the pause state was tracked in the `AsyncLLM` object, while DP coordination messages were exchanged between `EngineCore` processes and the `DPCoordinator`. To illustrate, for a DP world size of 2, one could have a deadlock scenario as follows:

1. API Server DP Rank 0 receives a generation request and forwards it to `EngineCore`. API Server issues a `FIRST_REQ` message to the `DPCoordinator` to start a new wave. The request is forwarded to DP Rank 0 `EngineCore` which begins a new scheduler step.
2. The Controller issues a `/pause` request to both the engines. Pause state is set in the `AsyncLLM` object, and new requests are not forwarded to `EngineCore`. API servers on all DP ranks return right after.
3. The Trainer issues weight update requests. Meanwhile, DP Rank 0 `EngineCore` has entered the forward pass and is waiting for other DP ranks to join. (For simplicity, we ignore the `start_weight_update` and `finish_weight_update` requests here).
4. The weight update request reaches DP Rank 1 `EngineCore` and the replica enters an NCCL broadcast collective waiting for other ranks. The weight update request is queued on DP Rank 0 `EngineCore`.
5. `DPCoordinator` sends a `START_DP_WAVE` message to DP Rank 1 `EngineCore` but the message is queued.
6. Different ranks are in different collectives and deadlock.

The same scenario is represented here:

<p align="center">
<img src="/assets/figures/2026-05-12-native-rl-apis/vllm_deadlock.svg" width="80%">
<br>
<em>Figure 5: A deadlock scenario possible in DPEP deployments in vLLM.</em>
</p>

We address this with two changes:

**1. Move pause logic into `EngineCore`.** Instead of tracking pause state at the `AsyncLLM` entrypoint layer, it is now handled directly in the scheduler. This reduces race conditions between pause and generation requests.

**2. Two-phase pause/resume.**

* **Phase 1 (local pause)**: Each engine pauses scheduling but continues stepping by respecting any inbound `START_DP_WAVE` requests, so it can still participate in required forward passes.
* **Phase 2 (global pause)**: Currently, all the ranks perform a global all-reduce every 32 steps to check if there are pending requests in any DP rank. In the same all-reduce phase, we also check if all the engines are in the "local pause" state. If all the ranks agree, they stop together.

This ensures:

* No rank gets stuck waiting.
* `START_DP_WAVE` is respected even if an engine receives a pause request.
* All workers transition consistently.

Thus, the same scenario as before is handled gracefully:

1. API Server DP Rank 0 receives a generation request and forwards it to `EngineCore`. API server issues a `FIRST_REQ` message to the `DPCoordinator` to start a new wave. The request is forwarded to DP Rank 0 `EngineCore` which begins a new scheduler step.
2. The Controller issues a `/pause` request to both the engines. The pause request is forwarded to `EngineCore` for both the ranks. The pause request is queued in DP Rank 0 `EngineCore` until the step is complete. Note that the API server doesn't yet return.
3. DP Rank 0 `EngineCore` starts executing a forward pass, with workers waiting on an all-to-all collective.
4. DP Rank 1 `EngineCore` receives the pause request, enters "local pause" state.
5. `DPCoordinator` sends a `START_DP_WAVE` message to DP Rank 1 `EngineCore`.
6. DP Rank 1 `EngineCore` starts executing a forward pass. Forward pass completes since both DP ranks joined.
7. DP Rank 0 `EngineCore` processes the pause request, enters "local pause" state.
8. DP Rank 0 and DP Rank 1 `EngineCore` participate in a periodic all-reduce, realize both engines are in "local pause" state, and enter "global pause" state.
9. API servers return on the `/pause` call.
10. Trainer issues weight update requests. API servers forward the weight update request to the `EngineCore` processes. (For simplicity, we ignore the `start_weight_update` and `finish_weight_update` requests here).
11. All vLLM workers enter the NCCL broadcast collective. Trainer starts NCCL broadcast.
12. Weight update finishes successfully.

<p align="center">
<img src="/assets/figures/2026-05-12-native-rl-apis/vllm_no_deadlock.svg" width="80%">
<br>
<em>Figure 6: Deadlock-free pause/resume in DPEP deployments with the two-phase protocol.</em>
</p>

## Validation

### Demonstrating the New RL APIs

We demonstrate usage of the new RL APIs in [SkyRL](https://github.com/NovaSky-AI/SkyRL).

In SkyRL, the trainer interacts with inference engines over HTTP. For weight syncing, SkyRL uses the native weight syncing APIs, as well as the native `/pause` and `/resume` APIs for asynchronous RL. The integration with the native RL APIs is detailed in the [docs](https://docs.skyrl.ai/docs/getting-started/inference_architecture), and we demonstrate asynchronous training for Qwen3-1.7B on the original DAPO recipe ([example](https://github.com/NovaSky-AI/SkyRL/blob/dec7137d9c57db59458a677de09add0b24413f26/examples/train/algorithms/dapo/run_dapo_qwen3_1.7b_aime_fully_async_onestep.sh)).

<p align="center">
<img src="/assets/figures/2026-05-12-native-rl-apis/skyrl_validation.svg" width="80%">
<br>
<em>Figure 7: Asynchronous training of Qwen3-1.7B on the DAPO recipe in SkyRL using the native RL APIs.</em>
</p>

### Validation at Scale: Fully Async RL in a Wide-EP Setup

The Prime-RL team has validated the RL APIs against a deployment of `zai-org/GLM-5.1-FP8` with inference running in a P/D disaggregated setup across 16 8xH200 nodes - 2 replicas of 4P+4D both with DPEP32 for both prefill and decode. All instances were also configured with CPU KV cache offloading with a capacity of 1TB per node. The routing across engines was enabled using `vllm-router` which provides cache-aware sticky routing. The Trainer ran a BF16 model equivalent (`zai-org/GLM-5.1`) on another 16 8xH200 nodes, on a custom math environment with [IcePop](https://arxiv.org/abs/2510.18855) as the algorithm of choice. This deployment has proven stable over 100+ steps while training, with growing evaluation performance, an upward RL curve, stable KL mismatch and weight updates progressing normally.

<p align="center">
<img src="/assets/figures/2026-05-12-native-rl-apis/prime_rl.svg" width="95%">
<br>
<em>Figure 8: Prime-RL validation of fully async RL with <code>zai-org/GLM-5.1-FP8</code> across 16 8xH200 nodes.</em>
</p>

## Conclusion

We've seen growing interest in the vLLM RL community for building on top of the new RL APIs. Some ongoing work from the vLLM RL community includes integrating a new [K8s-native weight transfer engine](https://github.com/vllm-project/vllm/pull/40828) as well as supporting [sharding-aware, RDMA-native weight transfer](https://github.com/vllm-project/vllm/issues/40822) in a generic way. New development is tracked in the [vLLM RL Roadmap](http://github.com/vllm-project/vllm/issues/41733).

Read more about the RL tooling in vLLM in the docs:

* [Weight transfer](https://docs.vllm.ai/en/latest/training/weight_transfer/)
* [Async RL](https://docs.vllm.ai/en/latest/training/async_rl/)

Try out the new APIs on vLLM here: [https://github.com/vllm-project/vllm/tree/main/examples/rl](https://github.com/vllm-project/vllm/tree/main/examples/rl).

## Acknowledgements

Thanks to the following groups and individuals who made this possible:

* **Prime-RL** team (especially [Matej Sirovatka](https://github.com/S1ro1)) and [**Junjie Zhang**](https://github.com/junjzhang) for helping to validate and debug the RL APIs with large-scale runs.
* **NemoRL** team for providing an optimized packed tensor implementation.
* [Robert Shaw](https://github.com/robertgshaw2-redhat) for organizing RL-related efforts and [Kyle Sayers](https://github.com/kylesayrs) for making weight reloading possible through layerwise reloading.
