---
layout: post
title: "Reducing vLLM startup time with one uv flag"
author: "Nils Matteson"
summary: "One uv build flag saved 17.6–17.8 seconds per startup for Qwen3-8B and Qwen3-32B on an H100 server, with images and models already local."
image: /assets/figures/2026-09-22-python-bytecode-startup/bytecode-server.png
social_image: /assets/figures/2026-09-22-python-bytecode-startup/bytecode-server.png
tags:
  - performance
  - deployment
---

Building Python bytecode into the image saved **17.6 seconds for Qwen3-8B and 17.8 seconds for Qwen3-32B** in median paired startup time on a Xeon/H100 server. The change is one uv setting, enabled before installing runtime packages:

```dockerfile
ENV UV_COMPILE_BYTECODE=1
```

This tells uv to compile Python source during package installation, so fresh containers can load the prepared `.pyc` files instead of compiling that source at startup.

For official-image users, [PR #55422](https://github.com/vllm-project/vllm/pull/55422) is already included in this [CUDA nightly](https://hub.docker.com/r/vllm/vllm-openai/tags?name=nightly-8cbd5d03006c33185f402249ff2b448efd594986):

```text
vllm/vllm-openai:nightly-8cbd5d03006c33185f402249ff2b448efd594986
```

The change will be supported in official release images starting with **v0.32.0**.

## Startup savings on the H100

I timed process launch to the first correct response, with the image and model files already local. Four adjacent pairs per model compared the nightly's shipped package bytecode with those caches removed. All eight pairs improved. **Image pull and model download are excluded.**

![All four paired process-launch-to-first-correct-response observations for each model, comparing package bytecode absent with the original shipped caches.](/assets/figures/2026-09-22-python-bytecode-startup/bytecode-server.png)

*Four pairs per model, two in each order, on one Xeon Platinum 8480+ / H100 SXM 80GB server. Software files were resident in RAM; application/compiler caches started fresh. Lines connect paired starts.*

The interpreter, installed software and paths stayed the same; only package `.pyc` availability changed. Normal cache writes remained enabled in both conditions. The [methods](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#server-follow-up-october-9-2026), [JSON](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/server-measurements.json) and [all 16 observations](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/server-observations.csv) record the configuration and output checks. These are paired medians on one deployment, not a fleet-wide estimate.

The absolute saving was similar for the two model sizes, while the percentage reduction was larger for 8B: 17.6% versus 13.9%. That is consistent with avoiding Python import work shared by both models. The time saved will depend on the runtime and host.

## Other startup measurements

On my i9-13900HX / RTX 4090 Laptop, a separate matched-image study covered Qwen3-0.6B, Qwen3-1.7B and SmolLM2-1.7B with six balanced pairs each. All 18 pairs favored precompilation, with median paired savings of 12.8–13.5 seconds. The [figure](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/bytecode-multimodel.png) shows every pair, and the [methods](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#three-model-launch-only-follow-up-october-9-2026) link the data. Hardware, runtime and clock differ from the H100 study, so I keep the results separate.

An earlier Qwen3-0.6B test also included pulling and unpacking matched images from a local loopback registry into an empty Docker store. It saved a median paired 15.4 seconds across three pairs. The precompiled image added 153 MB compressed, or 1.8%, which can matter on a slow first pull. The [image-delivery measurements](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#image-delivery) retain that separate timing boundary and image-size tradeoff.

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

In one separately instrumented serving-CLI comparison, valid bytecode cut import time from **12.20 to 5.01 seconds**, a saving of 7.19 seconds. Without the caches, Python spent 6.99 seconds inside 5,118 source-compiler calls; with them, it made zero such calls. Compilation accounts for most of that import gap. This one-run-per-condition diagnostic does not partition the full server-startup saving.

A separate, repeated study measured imports without compiler instrumentation in a released vLLM 0.28.0 installation outside Docker. Every import root improved:

![Fresh-process import measurements for torch, transformers, vLLM and the serving CLI. Each root is measured independently, with and without bytecode.](/assets/figures/2026-09-22-python-bytecode-startup/import-times.png)

*Three fresh processes per condition, with the measured Python and native-library files resident in memory and no compiler instrumentation. These are separate observations from the diagnostic above. Dots are observations; ticks are medians. Each import includes its dependencies, so rows are not additive.*

In this vLLM release, `import vllm` already loads Dynamo, Inductor and SymPy; a plain `import torch` leaves them unloaded. Most of the source belongs to dependencies, but vLLM's import path decides when it loads.

For a separate compiler-only test, I used the traced serving-CLI corpus: 5,118 files containing 78.6 MB of source. This installation differs from the Docker build above, which compiled 5,257 files. I read the corpus into memory before timing Python's compiler directly:

```python
# Read and verify the sources before starting the clock.
for filename, source_bytes in preloaded_sources:
    compile(source_bytes, filename, "exec", dont_inherit=True, optimize=0)
```

Compiling and discarding those code objects took 6.30 seconds median across three fresh-process passes, with 6.30 seconds of current-thread CPU time. No file reads, module execution or bytecode writes occurred inside the timed loop. This is the standalone compiler median; the 6.99 seconds above was measured during an actual import with per-call timing hooks.

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
