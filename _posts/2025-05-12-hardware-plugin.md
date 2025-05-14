---
layout: post
title: "Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU"
author: "The Ascend Team on vLLM"
image: /assets/logos/vllm-logo-only-light.png
---

Since December 2024, through the joint efforts of the vLLM community and the Ascend team on vLLM, we have completed the [Hardware Pluggable RFC](https://github.com/vllm-project/vllm/issues/11162). This proposal allows hardware integration into vLLM in a decoupled manner, enabling rapid and modular support for different hardware platforms.

---

## Why vLLM Hardware Plugin?

Currently, vLLM already supports multiple backends. However, as the number of vLLM backends continues to grow, several challenges have emerged:

- **Increased Code Complexity**: Each hardware backend has its own `Executor`, `Worker`, `Runner`, and `Attention` components. This has increased the complexity of the vLLM codebase, with non-generic backend-specific code scattered throughout the project.
- **High Maintenance Costs**: The cost of maintaining backends is high, not only for the backend developers but also for the vLLM community. The scarcity of community contributor resources makes efficiently adding new features difficult when backend maintainers are not present. 
- **Lack of Extensibility**: While vLLM follows a well-structured layered design by implementing backends through `Executor`, `Worker`, `Runner`, and `Attention`, supporting new hardware often requires invasive modifications or patching rather than dynamic registration. This makes adding new backends cumbersome.

Recognizing the need for a flexible and modular approach to integrating hardware backends, we proposed hardware plugins as a feasible solution:

- **Decoupled Codebase**: The hardware backend plugin code remains independent, making the vLLM core code cleaner.
- **Reduced Maintenance Burden**: vLLM developers can focus on generic features without being overwhelmed by the differences caused by backend-specific implementations.
- **Faster Integration & More Independent**: New backends can be integrated quickly with less work to do and evolve independently.

---

## What is the vLLM Hardware Plugin?

Before introducing the vLLM Hardware Plugin, let's first look at two prerequisite RFCs:

- [[RFC] vLLM Plugin System](https://github.com/vllm-project/vllm/issues/7131): This RFC introduces a plugin-based approach to support various customization requirements, allowing users to define custom models, executors, schedulers, etc.
- [[RFC] Make vLLM Device-Agnostic for Diverse Hardware Support](https://github.com/vllm-project/vllm/issues/9268) and ([vllm-project/vllm#6080](https://github.com/vllm-project/vllm/pull/6080)): This RFC introduces the **platform** submodule, which centralizes hardware-related implementations to reduce conditional logic in the main codebase and lays the foundation for modularization.

Based on these RFCs, we proposed [[RFC] Hardware Pluggable](https://github.com/vllm-project/vllm/issues/11162), which integrates the `Platform` module into vLLM as a plugin. Additionally, we refactored `Executor`, `Worker`, `ModelRunner`, `AttentionBackend`, and `Communicator` to support hardware plugins more flexibly.

Currently, the vLLM community has successfully implemented the Platform module introduced in the RFC. The functionality is validated through the [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) and [vllm-project/vllm-spyre](https://github.com/vllm-project/vllm-spyre) projects. Using this plugin mechanism, we successfully integrated vLLM with the Ascend NPU and IBM Spyre backends.

---

## How to Integrate a New Backend via vLLM Hardware Plugin Mechanism

This section will dive into integrating a new backend via the hardware plugin in both developer and user perspective.

### Developer Perspective

To integrate a new backend into vLLM using the hardware plugin, follow these steps:

#### Step 1: Create a New Project and Initialize the Platform

Start by creating a Python project for the new backend and adding a `platform.py` file. Then, import the `Platform` class from `vllm.platforms` and implement the required attributes and methods.

You can refer to the [`platform.py`](https://github.com/vllm-project/vllm-ascend/blob/72a43a61d8d2193dddbfcc60578fd642008225a5/vllm_ascend/platform.py#L52) in vLLM Ascend project for an example.

#### Step 2: Implement Custom Worker, Model Runner, Attention Backend, and Communicator Modules

Depending on the new backend's requirements, implement the following modules:

```python
from vllm.worker.worker_base import WorkerBase
from vllm.worker.model_runner_base import ModelRunnerBase
from vllm.attention.backends.abstract import AttentionBackend
from vllm.distributed.device_communicators.base_communicator import CommunicatorBase
```

Each of these classes has a corresponding base class in vLLM. Again, you can refer to [vLLM Ascend's implementation](https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend) for an example.

#### Step 3: Register the Plugin

Register the plugin in `setup.py` using the entrypoint mechanism of python:

```python
setup(
    entry_points={'vllm.platform_plugins': ["{your_platform_name} = {code_path}:{register_function}"]}
)
```

- `{your_platform_name}`: The name of the new backend (can be arbitrary).  
- `{code_path}`: The path to the main Python module.  
- `{register_function}`: The register function, which returns the path of `Platform` class defined in step 1.

Refer to [`setup.py`](https://github.com/vllm-project/vllm-ascend/blob/72a43a61d8d2193dddbfcc60578fd642008225a5/setup.py#L102) in vLLM Ascend for a practical example.

---

### User Perspective

Users only need to install vllm and your plugin before running, taking [vllm-ascend](https://github.com/vllm-project/vllm-ascend) as an example:

```bash
pip install vllm vllm-ascend
```

On startup, you will observe the following logs, which means the backend plugin is working properly:

```bash
INFO 02-06 15:49:01 __init__.py:30] Available plugins for group vllm.platform_plugins:
INFO 02-06 15:49:01 __init__.py:32] name=ascend, value=vllm_ascend:register
… …
INFO 02-06 15:49:01 __init__.py:44] plugin ascend loaded.
INFO 02-06 15:49:01 __init__.py:181] Platform plugin ascend is activated
```

---

## What's Next?

Moving forward, we will continue collaborating with developers in the vLLM community to enhance the following aspects:

1. Continuous enhancements to the V1 Engine and VLMs.
2. Expanding plugin support for more modules and features, such as scheduler, graph mode and custom operators.
3. Better user experience and higher performance.
4. Maintenance and enhancement of a stable plugin architecture for appropriate hardware platforms

We encourage everyone to try out this new feature! If you have any questions, join the [vLLM Slack](https://slack.vllm.ai) and participate in the **#sig-extensible-hardware** channel for discussions. 🚀


## Acknowledgements

This flexible hardware backend plugin mechanism would not have been possible without the efforts of many vLLM contributors. Thus we are deeply grateful to the vLLM maintainers, including [Kaichao You](https://github.com/youkaichao), [Simon Mo](https://github.com/simon-mo), [Cyrus Leung](https://github.com/DarkLight1337), [Robert Shaw](https://github.com/robertgshaw2-redhat), [Michael Goin](https://github.com/mgoin) and [Jie Li](https://github.com/jeejeelee) for related refactor, deep discussion and quick review, [Xiyuan Wang](https://github.com/wangxiyuan), [Shanshan Shen](https://github.com/shen-shanshan), [Chenguang Li](https://github.com/noemotiovon) and [Mengqing Cao](https://github.com/MengqingCao) from the Ascend team on vLLM for mechanism design and implementation, [Joe Runde](https://github.com/joerunde) and [Yannick Schnider](https://github.com/yannicks1) from the Spyre team on vLLM for pluggable scheduler design and implementation, and other contributors, including [yancong](https://github.com/ice-tong) for extendable quantization method design and implementation, [Aviv Keshet](https://github.com/akeshet) for extendable `SamplingParams`.
