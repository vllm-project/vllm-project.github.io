---  
layout: post
title: "Building Clean, Maintainable vLLM Modifications Using the Plugin System"
author: "Dhruvil Bhatt (AWS SageMaker)"
image: /assets/logos/vllm-logo-text-light.png
---
> [!NOTE]
> Originally posted on this [Medium article](https://medium.com/@dhruvilbhattlm10/building-clean-maintainable-vllm-modifications-using-the-plugin-system-e80df0f62861).


<p align="center">
<picture>
<img src="/assets/figures/2025-11-20-vllm-plugin-system/vllm-plugin-system-arch.png" width="100%">
</picture><br>
<em>Source: <a href="https://github.com/vllm-project/vllm-ascend" target="_blank">https://github.com/vllm-project/vllm-ascend</a></em>
</p>


---

<p align="center">
<strong><em>Avoiding forks, avoiding monkey patches, and keeping sanity intact</em></strong>
</p>


## Overview

Large Language Model inference has been evolving rapidly, and [vLLM](https://github.com/vllm-project/vllm/) has emerged as one of the most powerful engines for high-throughput, low-latency model serving. It provides continuous batching, efficient scheduling, paged attention, and a production-ready API layer - making it an ideal choice for serving everything from small language models to massive frontier systems.

But as with any fast-moving system, there comes a point where teams or individuals may want to modify vLLM's internal behavior. Maybe you want to experiment with custom scheduling logic, alter KV-cache handling, inject proprietary optimizations, or patch a part of the model-execution flow.

And that is where the real challenge begins.

---

## The Problem: "I need to modify vLLM… what now?"

If the change is simple, or if it benefits the general community, the answer is straightforward:

### Option A - Upstream your contribution to vLLM

This is always the cleanest approach. Your change lives in open-source, receives community review, and remains tied to the evolution of vLLM.

However, reality isn't always that accommodating. Many modifications are:
- **Proprietary**
- **Domain-specific**
- **Too experimental**
- **Not generalizable enough** for upstream acceptance
- Or **blocked by internal timelines** that don't align with open-source review cycles

When upstreaming is not possible, you must find another path.

---

### Option B - Maintain your own vLLM fork

This is usually the first instinct:
> "Let's just fork vLLM and add our changes there."

While it works for tiny, slow-moving projects, **vLLM is not one of those**.

vLLM is an extremely active repository, releasing newer versions as frequently as **two weeks apart** and merging **hundreds of PRs every week**.

Maintaining a long-running fork means:
- ❌ Constantly rebasing or merging upstream changes
- ❌ Resolving conflicts on rapidly changing areas
- ❌ Reapplying your patches manually
- ❌ Performing heavy compatibility testing
- ❌ Managing internal developer workflows around a custom vLLM artifact

Before long, the fork becomes a **full-time responsibility**.

For many teams, that operational load is simply unsustainable.

---

### Option C - Use monkey patching

Another route is building a small Python package that applies monkey patches on top of vanilla vLLM at build-time.

At first glance, this seems appealing:
- ✅ No fork
- ✅ No divergence from vanilla vLLM
- ✅ Patches applied dynamically
- ✅ Small code footprint

…but the reality is far from ideal.

Monkey patching typically requires replacing entire classes or modules, even if you only want to change ten lines. This means:

- ❌ **You copy large chunks of vLLM's source** - Even the parts you don't modify
- ❌ **Every vLLM upgrade breaks your patch** - Because you replaced full files, not just the individual lines of interest
- ❌ **Debugging becomes painful** - Is the bug in your patch? In unchanged vanilla code? Or because monkey patching rewired behavior unexpectedly?
- ❌ **Operational complexity grows over time** - Every vLLM release forces you to diff and re-sync your copied files - exactly the same problem as maintaining a fork, just disguised inside your Python package
- ❌ Monkey-patching certain modules (like the `Scheduler`) often **does not work** because they run inside `EngineCore` in a separate process. This can lead to process-synchronization issues, where `EngineCore` continues invoking the stale implementation of the module you intended to modify.


Monkey patching solves the surface-level problem, but introduces long-term maintenance challenges that can become unmanageable.

---

## A Cleaner Alternative: Leverage the vLLM Plugin System

To overcome the limitations of both forks and monkey patches, I explored vLLM's evolving [general_plugin architecture](https://docs.vllm.ai/en/stable/design/plugin_system.html), which allows developers to inject targeted modifications into the engine without altering upstream code.

This architecture enables:
- ✅ Structured, modular patches
- ✅ Runtime activation
- ✅ Surgical-level code overrides
- ✅ Compatibility safeguards
- ✅ No full-file duplication
- ✅ No monkey-patching gymnastics
- ✅ No need for a maintained fork

This presents a middle ground between "upstream everything" and "replace entire files."

---

> **Note:** vLLM provides *four* plugin groups/mechanisms — platform plugins, engine plugins, model plugins, and **general plugins**. This article specifically focuses on the **general plugin system**, which is loaded in all vLLM processes and is therefore ideal for the clean vLLM modification approach described in this article. For more details on the different plugin groups, see the vLLM docs: [Types of Supported Plugins](https://docs.vllm.ai/en/latest/design/plugin_system/#types-of-supported-plugins).

---

## Building a Clean Extensions Framework Using vLLM Plugins

Using the plugin system, I created a small extensions package that acts as a container for all custom modifications. Instead of replacing entire modules or forking the whole repository, each patch:

- Contains **just the exact code snippet or class** that needs to change
- Can be **enabled or disabled at runtime**
- Can specify **minimum supported vLLM versions**
- Can remain **dormant unless a specific model config requests it**

Because plugins are applied at runtime, we maintain a **single, unified container image** for serving multiple models while selectively enabling different patches per model.

This approach is inspired by plugin-based designs like [ArcticInference](https://github.com/snowflakedb/ArcticInference), where patches are injected cleanly and selectively at runtime.

---

## Implementation: Creating Your First vLLM Plugin Package

Let's walk through building a plugin-based extension system using vLLM's `general_plugins` entry point.

### Project Structure

```
vllm_custom_patches/
├── setup.py
├── vllm_custom_patches/
│   ├── __init__.py
│   ├── core.py              # Base patching infrastructure
│   └── patches/
│       ├── __init__.py
│       └── priority_scheduler.py
└── README.md
```

---

### Core Patching Infrastructure

The foundation is a clean patching mechanism that allows surgical modifications:

```python
# vllm_custom_patches/core.py
import logging
from types import MethodType, ModuleType
from typing import Type, Union
from packaging import version
import vllm

logger = logging.getLogger(__name__)

PatchTarget = Union[Type, ModuleType]

class VLLMPatch:
    """
    Base class for creating clean, surgical patches to vLLM classes.
    
    Usage:
        class MyPatch(VLLMPatch[TargetClass]):
            def new_method(self):
                return "patched behavior"
        
        MyPatch.apply()
    """
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_patch_target'):
            raise TypeError(
                f"{cls.__name__} must be defined as VLLMPatch[Target]"
            )
    
    @classmethod
    def __class_getitem__(cls, target: PatchTarget) -> Type:
        if not isinstance(target, (type, ModuleType)):
            raise TypeError(f"Can only patch classes or modules, not {type(target)}")
        
        return type(
            f"{cls.__name__}[{target.__name__}]",
            (cls,),
            {'_patch_target': target}
        )
    
    @classmethod
    def apply(cls):
        """Apply this patch to the target class/module."""
        if cls is VLLMPatch:
            raise TypeError("Cannot apply base VLLMPatch class directly")
        
        target = cls._patch_target
        
        # Track which patches have been applied
        if not hasattr(target, '_applied_patches'):
            target._applied_patches = {}
        
        for name, attr in cls.__dict__.items():
            if name.startswith('_') or name in ('apply',):
                continue
            
            if name in target._applied_patches:
                existing = target._applied_patches[name]
                raise ValueError(
                    f"{target.__name__}.{name} already patched by {existing}"
                )
            
            target._applied_patches[name] = cls.__name__
            
            # Handle classmethods
            if isinstance(attr, MethodType):
                attr = MethodType(attr.__func__, target)
            
            setattr(target, name, attr)
            action = "replaced" if hasattr(target, name) else "added"
            logger.info(f"✓ {cls.__name__} {action} {target.__name__}.{name}")

def min_vllm_version(version_str: str):
    """
    Decorator to specify minimum vLLM version required for a patch.
    
    Usage:
        @min_vllm_version("0.9.1")
        class MyPatch(VLLMPatch[SomeClass]):
            pass
    """
    def decorator(cls):
        original_apply = cls.apply
        
        @classmethod
        def checked_apply(cls):
            current = version.parse(vllm.__version__)
            minimum = version.parse(version_str)
            
            if current < minimum:
                logger.warning(
                    f"Skipping {cls.__name__}: requires vLLM >= {version_str}, "
                    f"but found {vllm.__version__}"
                )
                return
            
            original_apply()
        
        cls.apply = checked_apply
        cls._min_version = version_str
        return cls
    
    return decorator
```

---

### Example Patch: Priority-Based Scheduling

Now let's create a concrete patch that adds priority scheduling to vLLM:

```python
# vllm_custom_patches/patches/priority_scheduler.py
import logging
from vllm.core.scheduler import Scheduler
from vllm_custom_patches.core import VLLMPatch, min_vllm_version

logger = logging.getLogger(__name__)

@min_vllm_version("0.9.1")
class PrioritySchedulerPatch(VLLMPatch[Scheduler]):
    """
    Adds priority-based scheduling to vLLM's scheduler.
    
    Requests can include a 'priority' field in their metadata.
    Higher priority requests are scheduled first.
    
    Compatible with vLLM 0.9.1+
    """
    
    def schedule_with_priority(self):
        """
        Enhanced scheduling that respects request priority.
        
        This method can be called instead of the standard schedule()
        to enable priority-aware scheduling.
        """
        # Get the standard scheduler output
        output = self._schedule()
        
        # Sort by priority if metadata contains priority field
        if hasattr(output, 'scheduled_seq_groups'):
            output.scheduled_seq_groups.sort(
                key=lambda seq: getattr(seq, 'priority', 0),
                reverse=True
            )
            
            logger.debug(
                f"Scheduled {len(output.scheduled_seq_groups)} sequences "
                f"with priority ordering"
            )
        
        return output
```

---

### Plugin Entry Point and Registry

The plugin system ties everything together:

```python
# vllm_custom_patches/__init__.py
import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class PatchManager:
    """Manages registration and application of vLLM patches."""
    
    def __init__(self):
        self.available_patches: Dict[str, type] = {}
        self.applied_patches: List[str] = []
    
    def register(self, name: str, patch_class: type):
        """Register a patch for later application."""
        self.available_patches[name] = patch_class
        logger.info(f"Registered patch: {name}")
    
    def apply_patch(self, name: str) -> bool:
        """Apply a single patch by name."""
        if name not in self.available_patches:
            logger.error(f"Unknown patch: {name}")
            return False
        
        try:
            self.available_patches[name].apply()
            self.applied_patches.append(name)
            return True
        except Exception as e:
            logger.error(f"Failed to apply {name}: {e}")
            return False
    
    def apply_from_env(self):
        """
        Apply patches specified in VLLM_CUSTOM_PATCHES environment variable.
        
        Format: VLLM_CUSTOM_PATCHES="PatchOne,PatchTwo"
        """
        env_patches = os.environ.get('VLLM_CUSTOM_PATCHES', '').strip()
        
        if not env_patches:
            logger.info("No custom patches specified (VLLM_CUSTOM_PATCHES not set)")
            return
        
        patch_names = [p.strip() for p in env_patches.split(',') if p.strip()]
        logger.info(f"Applying patches: {patch_names}")
        
        for name in patch_names:
            self.apply_patch(name)
        
        logger.info(f"Successfully applied: {self.applied_patches}")

# Global manager instance
manager = PatchManager()

def register_patches():
    """
    Main entry point called by vLLM's plugin system.
    This function is invoked automatically when vLLM starts.
    """
    logger.info("=" * 60)
    logger.info("Initializing vLLM Custom Patches Plugin")
    logger.info("=" * 60)
    
    # Import and register all available patches
    from vllm_custom_patches.patches.priority_scheduler import PrioritySchedulerPatch
    
    manager.register('PriorityScheduler', PrioritySchedulerPatch)
    
    # Apply patches based on environment configuration
    manager.apply_from_env()
    
    logger.info("=" * 60)
```

---

### Setup Configuration

The `setup.py` file registers the plugin with vLLM:

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name='vllm-custom-patches',
    version='0.1.0',
    description='Clean vLLM modifications via the plugin system',
    packages=find_packages(),
    install_requires=[
        'vllm>=0.9.1',
        'packaging>=20.0',
    ],
    # Register with vLLM's plugin system
    entry_points={
        'vllm.general_plugins': [
            'custom_patches = vllm_custom_patches:register_patches'
        ]
    },
    python_requires='>=3.11',
)
```

---

## Usage Examples

### Installation

```bash
# Install the plugin package
pip install -e .
```

### Running with Different Configurations

```bash
# Vanilla vLLM (no patches)
VLLM_CUSTOM_PATCHES="" python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2

# With priority scheduling patch
VLLM_CUSTOM_PATCHES="PriorityScheduler" python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct
```

### Docker Integration

```dockerfile
# Dockerfile
FROM vllm/vllm-openai:latest

COPY . /workspace/vllm-custom-patches/
RUN pip install -e /workspace/vllm-custom-patches/

ENV VLLM_CUSTOM_PATCHES=""

CMD python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --host 0.0.0.0 \
    --port 8000
```

```bash
# Run with patches
docker run \
    -e MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct \
    -e VLLM_CUSTOM_PATCHES="PriorityScheduler" \
    -p 8000:8000 \
    vllm-with-patches

# Run vanilla vLLM
docker run \
    -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2 \
    -e VLLM_CUSTOM_PATCHES="" \
    -p 8000:8000 \
    vllm-with-patches
```

---

## How It Works: The vLLM Plugin Lifecycle

Understanding when and how patches are applied is crucial. Here's the complete lifecycle:

### Automatic Plugin Loading by vLLM

**Critical insight:** vLLM's architecture involves multiple processes, especially when using distributed inference with tensor parallelism, pipeline parallelism, or other parallelism techniques. To ensure consistency, vLLM automatically calls `load_general_plugins()` in **every single process** it creates before that process starts any actual work.

This means:
- ✅ Your patches are loaded in the **main process**
- ✅ Your patches are loaded in **all worker processes**
- ✅ Your patches are loaded in **GPU workers, CPU workers, and any auxiliary processes**
- ✅ Loading happens **before model initialization**, before scheduler creation, and before any inference begins

### The Complete Startup Sequence

When vLLM starts up, here's what happens in each process:

1. **Process Creation:** vLLM spawns a new process (main, worker, etc.)
2. **Plugin System Activation:** vLLM internally calls `load_general_plugins()` before any other vLLM work
3. **Entry Point Discovery:** Python's entry point system finds all registered `vllm.general_plugins`
4. **Plugin Function Execution:** Our `register_patches()` function is called
5. **Patch Registration:** Available patches are registered with the manager
6. **Environment Check:** The `VLLM_CUSTOM_PATCHES` variable is read
7. **Selective Application:** Only specified patches are applied via `VLLMPatch.apply()`
8. **Version Validation:** Each patch checks vLLM version compatibility via `@min_vllm_version`
9. **Surgical Modification:** Specific methods are added/replaced on target classes
10. **Normal vLLM Startup:** Only now does vLLM proceed with model loading, scheduler initialization, etc.

This guarantees that your patches are always active **before vLLM does anything**, ensuring consistent behavior across all processes and preventing race conditions.

---

## Benefits of the Plugin-Based Extensions Approach

### 1. Extremely small, surgical patch definitions
No duplicated files. No redundant code. Just the modifications. The `VLLMPatch` system lets you add a single method without copying the entire class.

### 2. Supports multiple models on the same vLLM build
Different models can enable different patches via the `VLLM_CUSTOM_PATCHES` environment variable.

### 3. Version-aware safety checks
Each patch can declare its minimum required version:
```python
@min_vllm_version("0.9.1")
class MyPatch(VLLMPatch[TargetClass]):
    pass
```
This prevents unexpected behavior during upgrades.

### 4. No more forking, syncing, or rebasing
Upgrading vLLM is as simple as `pip install --upgrade vllm` and testing your patches.

### 5. Eliminates monkey patching's complexity
Clean, trackable modifications without the silent breakages of traditional monkey patching.

### 6. Officially supported by vLLM
Uses vLLM's official `general_plugins` entry point system, meaning it's a supported extension mechanism.

---

## Why This Pattern Matters

As inference engines evolve at high velocity, teams often find themselves forced to choose between:
- Modifying internal behavior
- **OR** staying compatible with upstream releases

The plugin-based extensions model **removes that trade-off**. It lets you innovate rapidly while staying in sync with the rapidly growing vLLM ecosystem.

This approach keeps the operational overhead minimal while maintaining long-term flexibility - something both small teams and large platform groups will appreciate.

---

## Final Thoughts

If you're experimenting with or deploying vLLM and find yourself needing custom behavior, consider leveraging the general plugin system before committing to a fork or monkey-patch strategy.

It strikes the right balance between **control**, **maintainability**, and **sanity** - and it keeps your codebase clean, modular, and future-proof.

### Key takeaways:
- ✅ Use `VLLMPatch[TargetClass]` for surgical, class-level modifications
- ✅ Register via `vllm.general_plugins` entry point in `setup.py`
- ✅ Control patches with `VLLM_CUSTOM_PATCHES` environment variable. 
    - Note: `VLLM_CUSTOM_PATCHES` is **not** an official vLLM environment variable — it’s just an example used in this article. You can choose any env var name in your own plugin package.
- ✅ Version-guard patches with `@min_vllm_version` decorator
- ✅ One Docker image, multiple configurations

This pattern has proven effective in production environments and scales from experimental prototypes to multi-model production deployments.

---

### Contact Me

If you're interested in plugin-based architectures for inference systems or want to explore how to structure runtime patching in a clean way, feel free to reach out. Always happy to chat about scalable LLM deployment and design patterns😊 You can reach me at:

- **LinkedIn:** [https://www.linkedin.com/in/dhruvil-bhatt-uci/](https://www.linkedin.com/in/dhruvil-bhatt-uci/)
- **Website** - [https://www.dhruvilbhatt.com/](https://www.dhruvilbhatt.com/)
- **Email:** dhruvilbhattlm10@gmail.com
