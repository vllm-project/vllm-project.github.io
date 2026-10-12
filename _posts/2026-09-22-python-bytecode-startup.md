---
layout: post
title: "Reducing vLLM startup time with one uv flag"
author: "Nils Matteson"
summary: "A pinned official nightly tested with 8B and 32B models on a Xeon/H100 server, plus a separate three-model rig study. Models are local; image pull is excluded."
image: /assets/figures/2026-09-22-python-bytecode-startup/bytecode-server.png
social_image: /assets/figures/2026-09-22-python-bytecode-startup/bytecode-server.png
tags:
  - performance
  - deployment
---

Shipped Python bytecode saved 17.6 seconds with Qwen3-8B and 17.8 seconds with Qwen3-32B in median paired startup time on a Xeon/H100 server. I used four adjacent pairs per model, timing process launch to the first correct response with weights already local. The build change is one uv setting:

```dockerfile
ENV UV_COMPILE_BYTECODE=1
```

This is uv's `--compile-bytecode` flag in environment-variable form. It compiles Python source when packages are installed, so new containers can use the bytecode from their first launch.

For the server test, I used one pinned official nightly and removed or restored only its source-backed package `.pyc` files. The control starts without package bytecode; normal cache writes remain enabled in both arms. The interpreter, installed software and paths stayed the same. This checks the benefit of the shipped caches rather than comparing two release images.

| Model | Median paired startup saving | Median paired reduction |
|---|---:|---:|
| Qwen3-8B | 17.60 s | 17.6% |
| Qwen3-32B | 17.75 s | 13.9% |

![All four paired process-launch-to-first-correct-response observations for each model, comparing package bytecode absent with the original shipped caches.](/assets/figures/2026-09-22-python-bytecode-startup/bytecode-server.png)

*Four adjacent pairs per model, two in each order, on one Xeon Platinum 8480+ / H100 SXM 80GB server. BF16, TP1, maximum context 2048. Affinity selected eight physical cores; allocation was nonexclusive. Models local; fresh application/compiler caches. Python/native files were made resident in both arms before timing. No image pull or model download is included.*

These are per-model paired medians, not a fleet-wide estimate. The [server methods and all eight pairs](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#server-follow-up-october-9-2026), [JSON](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/server-measurements.json) and [16-observation CSV](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/server-observations.csv) retain the exact starting state and checks. The same-family models test size; they do not establish architecture diversity or general serving equivalence.

I also ran a separate matched-image study on my i9-13900HX / RTX 4090 Laptop: six balanced pairs each for Qwen3-0.6B, Qwen3-1.7B and SmolLM2-1.7B. All 18 pairs favored precompilation, with median paired savings of 12.8–13.5 seconds. Its [all-observation figure](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/bytecode-multimodel.png), [methods](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#three-model-launch-only-follow-up-october-9-2026) and [data](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/multi-model-measurements.json) remain separate. Hardware, runtime and clock differ, so a change in saving cannot be assigned to the CPU alone.

In the earlier Qwen3-0.6B experiment, the clock also included pulling and unpacking the image from a local loopback registry into an empty Docker store. Its three paired savings were 15.36, 11.21 and 16.10 seconds. The precompiled image added 153 MB compressed, or 1.8%; that transfer tradeoff remains relevant on a slow first pull. Those [original image-delivery methods](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#image-delivery) and data remain separate from the launch-only follow-up.

For official-image users, [PR #55422](https://github.com/vllm-project/vllm/pull/55422) is included in this [CUDA nightly](https://hub.docker.com/r/vllm/vllm-openai/tags?name=nightly-8cbd5d03006c33185f402249ff2b448efd594986), whose build setting I verified from the published image metadata:

```text
vllm/vllm-openai:nightly-8cbd5d03006c33185f402249ff2b448efd594986
```

As of October 9, the latest stable release, [v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0), does not include the change. The server follow-up uses this pinned nightly. Its actual linux/amd64 runtime had valid caches for all 31,528 package source files checked; the earlier matched-image studies use their retained images. The [supplement](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#official-image-availability-october-9-2026) records the image digests and verification scope.

## Why fresh containers compile Python again

CPython compiles source into code objects before executing it. A valid `.pyc` file stores that compiled code so Python can load it directly next time. Module initialization still runs. These are Python imports; `torch.compile` and GPU kernels have separate compilation and cache paths.

pip compiles bytecode during installation by default; [uv skips it](https://docs.astral.sh/uv/pip/compatibility/#bytecode-compilation). Python can write missing caches while a container runs, but those files live in its writable layer. The next fresh container starts from the image and has the same compilation work waiting for it. [uv's Docker guide](https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode) recommends enabling compilation for production images.

In the earlier matched-image study, I built two images from the same vLLM development commit, with and without the setting, using vLLM's CUDA Dockerfile for the test GPU's architecture. Their Python interpreter, installed package versions, source files and native libraries matched.

To check that Python used the caches, I ran `import vllm.entrypoints.cli.serve` in a fresh process. This loads the Python code and dependencies behind `vllm serve`, without loading a model or serving requests. I'll call this the **serving-CLI import**:

| Matched Docker images | Default build | Precompiled build |
|---|---:|---:|
| Installed package source files | 26,734 | 26,734 |
| Sources with valid bytecode | 407 | 26,734 |
| Source compilations during serving-CLI import | 5,257 | 0 |

Running the precompiled image with `-X pycache_prefix` pointing at an empty directory brought source compilation back. The packages and source hadn't changed; the loader could no longer find the shipped bytecode.

## Compiler time explains most of the import saving

In a separately instrumented serving-CLI import, Python spent 6.99 seconds compiling source. Valid bytecode shortened that import by 7.19 seconds. Compiler time accounts for most of this difference, leaving about 0.20 seconds unexplained. This diagnostic used one run per condition; it explains the import saving, not the full server-startup saving.

For repeated measurements without instrumentation, I used a released vLLM 0.28.0 installation outside Docker. The serving-CLI import took a median 12.27 seconds without precompiled package bytecode and 5.02 seconds with it. The other import roots improved too:

![Fresh-process import measurements for torch, transformers, vLLM and the serving CLI. Each root is measured independently, with and without bytecode.](/assets/figures/2026-09-22-python-bytecode-startup/import-times.png)

*Three fresh processes per condition, with the measured Python and native-library files resident in memory and no compiler instrumentation. Dots are observations; ticks are medians. Each import includes its dependencies, so rows are not additive.*

In this vLLM release, `import vllm` already loads Dynamo, Inductor and SymPy; a plain `import torch` leaves them unloaded. Most of the source belongs to dependencies, but vLLM's import path decides when it loads.

To isolate the compiler, I traced the files that the serving-CLI import compiled: 5,118 files containing 78.6 MB of source. This installation differs from the Docker build above, which compiled 5,257 files. I read the corpus into memory before timing Python's compiler directly:

```python
# Read and verify the sources before starting the clock.
for filename, source_bytes in preloaded_sources:
    compile(source_bytes, filename, "exec", dont_inherit=True, optimize=0)
```

Compiling and discarding those code objects took 6.30 seconds median across three passes, with 6.30 seconds of current-thread CPU time. No file reads, module execution or bytecode writes occurred inside the timed loop.

## The saving remains with source files already in RAM

Precompiled bytecode still saved 9–12 seconds per server launch when the same 8.85 GB of installed Python and shared-library files were already resident in memory in both conditions. Keeping these files in the OS page cache did not remove the saving.

A faster second launch would not establish the cause: the first can warm filesystem pages and write `.pyc` files. Repeating an import in the same interpreter can also reuse `sys.modules`.

For this test, I launched `vllm serve` directly from a Linux Python installation, outside Docker. I verified residency before each fresh launch, then timed launch to the first correct HTTP response. No image pull was involved. One condition began with precompiled package `.pyc` files; the other began with none in the tested package installation.

Launch to first correct response took 54.7–60.6 seconds without precompiled package caches and 44.6–48.4 seconds with them. The three paired savings were 10.13, 9.35 and 12.23 seconds. All six launches returned correct responses.

Normal bytecode writes and parent/child cache reuse stayed enabled, so this measures more than the compiler alone. It also uses a different vLLM build, Python build, CPU limit and timing boundary from the Docker experiment. The results are not additive. These are one-host observations; CPU frequency and thermal state were not fixed.

## Using it in an image

For your own image, set `UV_COMPILE_BYTECODE=1` before the uv commands that install runtime packages, then rebuild.

The main trap is an existing base image. `uv pip install` compiles files it installs or reinstalls; enabling the flag before adding one package does not necessarily prepare everything inherited from `FROM`. Compile that existing environment explicitly with the runtime Python's [compileall](https://docs.python.org/3.12/library/compileall.html) if needed. Check the final image, and keep both source and bytecode when copying packages between stages. Adding `-O` or changing the runtime cache path can change which files Python looks for. The [methods supplement](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#final-image-diagnostic) includes the cache-coverage command and compatibility caveats.

In the earlier matched-image test, the extra 153 MB can matter on a slow image pull. Nodes with the layers already cached avoid that transfer when starting another container. Build-time overhead was not measured separately. In the earlier isolated installation, the cached serving-CLI import still took about five seconds; module initialization remains work to investigate.

[Methods and measurements](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md) give the exact environments, run order, controls and limitations, with [JSON](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/supplementary-measurements.json) and [CSV](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/measurements.csv) data. Thanks to Simon Mo for pushing me to control for the page cache and time the compiler on its own.
