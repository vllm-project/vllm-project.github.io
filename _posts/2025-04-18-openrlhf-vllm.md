---
layout: post
title: "Accelerating RLHF with vLLM Ray Executor (OpenRLHF)"
author: "The OpenRLHF Team"
image: /assets/figures/openrlhf-vllm/ray.png
thumbnail-img: /assets/figures/openrlhf-vllm/ray.png
share-img: /assets/figures/openrlhf-vllm/ray.png
---

As the demand for training reasoning large language models (LLMs) grows, Reinforcement Learning from Human Feedback (RLHF) has become a pivotal technique. However, traditional RLHF training pipelines, especially those involving Proximal Policy Optimization (PPO), often face significant computational bottlenecks, with long chain-of-thought generation consuming up to 90% of the total training time.

## Design Philosophy

To address these challenges, OpenRLHF is designed as a user-friendly, high-performance framework for Reinforcement Learning from Human Feedback (RLHF), integrating key technologies such as Ray, vLLM, ZeRO-3, and AutoTP:

**Ray** serves as the backbone for distributed programming within OpenRLHF. Its robust scheduling and orchestration capabilities make it ideal for managing the complex data flows and computations inherent in RLHF training, including the distribution of reward models across multiple nodes. 

**vLLM with Ray Executor and AutoTP** is central to accelerating inference within OpenRLHF. It naturally supports Ray Executors and integrates with Hugging Face Transformers, enabling efficient weight updates through AutoTP. This combination ensures high-throughput, memory-efficient serving of large language models.

**ZeRO-3 with HuggingFace Transformers**, a memory optimization strategy from DeepSpeed, enables OpenRLHF to train large-scale models without the need for complex frameworks like Megatron. This allows for seamless integration with HuggingFace Transformers, facilitating straightforward loading and fine-tuning of pre-trained models. 

By combining Ray, vLLM, ZeRO-3, and HuggingFace Transformers, OpenRLHF offers a leading and simple solution for accelerating RLHF training. This architecture has influenced other frameworks, such as veRL, which adopt a similar paradigm for efficient and scalable RLHF training.

<img align="center" src="/assets/figures/openrlhf-vllm/ray.png" alt="Ray and vLLM in OpenRLHF" width="90%" height="90%">

​As illustrated in the figure, OpenRLHF utilizes [Ray's placement group API](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html) to flexibly schedule various RLHF components, including the vLLM engine, Actor, Critic, Reference, and Reward models. While these models are depicted separately, they can be co-located within shared placement groups to optimize resource utilization. For instance, all modules can share the same GPU group in a Hybrid Engine configuration, or specific components like the Actor and Critic can be assigned to the same GPU group. Weight synchronization between the Actor and the vLLM engine is achieved through high-performance communication mechanisms such as NVIDIA's NCCL or CUDA IPC memory copying, particularly in Hybrid Engine setups. 

##  Implementing RLHF Acceleration with vLLM Ray Executor

vLLM provides examples demonstrating how to accelerate RLHF training using Ray. By defining a custom `WorkerExtension` class, users can implement logic for weight synchronization between training and inference components. The `VLLM_RAY_PER_WORKER_GPUS` environment variable facilitates the allocation of GPU resources per worker, enabling configurations like hybrid engines where multiple components share the same GPU group.

[An example](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf_colocate.html) setup involves initializing Ray with a specified number of GPUs, creating a placement group for resource allocation, and defining training actors and inference engines. The training actors handle model initialization and weight updates, while the inference engines serve the models using vLLM. Weight synchronization between these components is achieved through inter-process communication mechanisms like CUDA IPC or NCCL, ensuring consistency across the training pipeline.


```python
import os
import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM
from transformers import AutoModelForCausalLM

class ColocateWorkerExtension:
    """
    Extension class for vLLM workers to handle weight synchronization.
    This class ensures compatibility with both vLLM V0 and V1.
    """
    def report_device_id(self) -> str:
        """Report the unique device ID for this worker"""
        from vllm.platforms import current_platform
        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

    def update_weights_from_ipc_handles(self, ipc_handles):
        """Update model weights using IPC handles"""
        handles = ipc_handles[self.device_uuid]
        device_id = self.device.index
        weights = []
        for name, handle in handles.items():
            func, args = handle
            list_args = list(args)
            list_args[6] = device_id  # Update device ID for current process
            tensor = func(*list_args)
            weights.append((name, tensor))
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()

    def check_weights_changed(self):
        """Verify if weights have been updated"""
        return all(torch.allclose(p, torch.zeros_like(p)) 
                  for p in self.model_runner.model.parameters())

class MyLLM(LLM):
    """
    Custom LLM class to handle GPU resource allocation and bundle indices.
    This ensures proper GPU utilization and placement group management.
    """
    def __init__(self, *args, bundle_indices: list, **kwargs):
        # Prevent Ray from manipulating CUDA_VISIBLE_DEVICES at the top level
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # Configure GPU utilization per worker
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.4"
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        super().__init__(*args, **kwargs)

class TrainingActor:
    """
    Actor class for model training.
    Handles model initialization and weight synchronization.
    """
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        self.model.to("cuda:0")
        # Initialize weights to zero for demonstration
        for p in self.model.parameters():
            p.data.zero_()
        torch.cuda.synchronize()
        from vllm.platforms import current_platform
        self.device_uuid = current_platform.get_device_uuid(0)

    def report_device_id(self) -> str:
        return self.device_uuid

    def get_weight_ipc_handles(self):
        """Get IPC handles for model weights"""
        from torch.multiprocessing.reductions import reduce_tensor
        return {self.device_uuid: {name: reduce_tensor(p.detach()) 
                                 for name, p in self.model.named_parameters()}}

# Initialize Ray with 4 GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
ray.init()

# Create placement group for GPU allocation
pg = placement_group([{"GPU": 1, "CPU": 0}] * 4)
ray.get(pg.ready())

# Create training actors
training_actors = []
for i in range(4):
    actor = ray.remote(
        num_gpus=0.4,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=i
        )
    )(TrainingActor).remote()
    training_actors.append(actor)

# Create inference engines
inference_engines = []
for bundle_indices in [[0, 1], [2, 3]]:
    llm = ray.remote(
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg
        )
    )(MyLLM).remote(
        model="facebook/opt-125m",
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.4,
        worker_extension_cls="__main__.ColocateWorkerExtension",
        bundle_indices=bundle_indices
    )
    inference_engines.append(llm)

# Collect device IDs for verification
training_device_ids = [ray.get(actor.report_device_id.remote()) 
                      for actor in training_actors]
inference_device_ids = [ray.get(llm.collective_rpc.remote("report_device_id", args=tuple()))
                       for llm in inference_engines]

# Verify device placement
assert training_device_ids[:2] == inference_device_ids[0]
assert training_device_ids[2:] == inference_device_ids[1]

# Synchronize weights
ipc_handles = {}
for actor in training_actors:
    ipc_handles.update(ray.get(actor.get_weight_ipc_handles.remote()))

for llm in inference_engines:
    ray.get(llm.collective_rpc.remote("update_weights_from_ipc_handles", 
                                     args=(ipc_handles,)))
    assert ray.get(llm.collective_rpc.remote("check_weights_changed", args=tuple()))
```