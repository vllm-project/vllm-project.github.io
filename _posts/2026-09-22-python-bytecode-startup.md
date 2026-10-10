---
layout: post
title: "Saving 15 seconds of vLLM startup with one uv flag"
author: "Nils Matteson"
summary: "Precompiling Python bytecode saved about 15 seconds in a three-pair vLLM startup test, including a local image pull. One build flag, a 1.8% larger image."
image: /assets/figures/2026-09-22-python-bytecode-startup/bytecode-startup.png
social_image: /assets/figures/2026-09-22-python-bytecode-startup/bytecode-startup.png
tags:
  - performance
  - deployment
---

One build setting cut about 15 seconds from my vLLM CUDA test image's pull-to-first-response time, including the pull of its slightly larger image from a local registry:

```dockerfile
ENV UV_COMPILE_BYTECODE=1
```

This is uv's `--compile-bytecode` flag in environment-variable form. It compiles Python source when packages are installed, so new containers can use the bytecode from their first launch. The question was how much this would save once image delivery and the rest of vLLM startup were included.

![Three matched image-delivery pairs compare the default build with precompiled Python bytecode. Paired savings are 15.4, 11.2 and 16.1 seconds.](/assets/figures/2026-09-22-python-bytecode-startup/bytecode-startup.png)

*Three pairs from empty Docker stores. Qwen3-0.6B weights were already local; both images used gzip and an uncapped loopback registry. Intel i9-13900HX, RTX 4090 Laptop GPU, four-CPU server quota. Host file-cache state was not controlled. Gray is the default build; orange has precompiled bytecode. Savings use unrounded timings.*

The paired savings were 15.36, 11.21 and 16.10 seconds, a median of 15.36. The precompiled image grew by 153 MB compressed, or 1.8%. Pull time varied in both directions, but container start to first response improved in every pair, by 11.9–17.3 seconds. All six launches returned the expected tokens.

## Why fresh containers compile Python again

CPython compiles source into code objects before executing it. A valid `.pyc` file stores that compiled code so Python can load it directly next time. Module initialization still runs. These are Python imports; `torch.compile` and GPU kernels have separate compilation and cache paths.

pip compiles bytecode during installation by default; [uv skips it](https://docs.astral.sh/uv/pip/compatibility/#bytecode-compilation). Python can write missing caches while a container runs, but those files live in its writable layer. The next fresh container starts from the image and has the same compilation work waiting for it. [uv's Docker guide](https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode) recommends enabling compilation for production images.

I built two images from the same vLLM development commit, with and without the setting, using vLLM's CUDA Dockerfile for the test GPU's architecture. Their Python interpreter, installed package versions, source files and native libraries matched. A fresh serving-CLI import showed that the built caches were actually being used:

| Matched Docker images | Default build | Precompiled build |
|---|---:|---:|
| Installed package source files | 26,734 | 26,734 |
| Sources with valid bytecode | 407 | 26,734 |
| Source compilations during serving-CLI import | 5,257 | 0 |

Running the precompiled image with `-X pycache_prefix` pointing at an empty directory brought source compilation back. The packages and source hadn't changed; the loader could no longer find the shipped bytecode.

## How much of that is compiler time?

Outside Docker, in a released vLLM 0.28.0 installation, importing the serving CLI took a median 12.27 seconds without bytecode and 5.02 seconds with it. The other import roots improved too:

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

A separate instrumented import, one run per condition, spent 6.99 seconds inside the 5,118 compiler calls. Valid bytecode shortened that import by 7.19 seconds. The compiler spans account for most of that difference, with about 0.20 seconds left unexplained. This is an explanation of the import saving, not a decomposition of the full server startup.

## Could this just be the page cache?

A faster second launch would not establish the cause: the first can warm filesystem pages and write `.pyc` files. Repeating an import in the same interpreter can also reuse `sys.modules`.

For a separate native serving test, I made the same 8.85 GB of installed Python and shared-library files resident in memory before every fresh server launch, and verified their residency. One condition started with precompiled package bytecode; the other started without it.

Launch to first correct response took 54.7–60.6 seconds without starting bytecode and 44.6–48.4 seconds with it. The three paired savings were 10.13, 9.35 and 12.23 seconds. All six launches returned correct responses. Keeping the measured source pages in RAM did not remove the saving.

Normal bytecode writes and parent/child cache reuse stayed enabled, so this measures more than the compiler alone. It also uses a different vLLM build, Python build, CPU limit and timing boundary from the Docker experiment. The results are not additive. These are one-host observations; CPU frequency and thermal state were not fixed.

## Using it in an image

Set `UV_COMPILE_BYTECODE=1` before the uv commands that install runtime packages, then rebuild. [PR #55422](https://github.com/vllm-project/vllm/pull/55422) has merged this setting into vLLM's Docker builds.

The main trap is an existing base image. `uv pip install` compiles files it installs or reinstalls; enabling the flag before adding one package does not necessarily prepare everything inherited from `FROM`. Compile that existing environment explicitly with the runtime Python's [compileall](https://docs.python.org/3.12/library/compileall.html) if needed. Check the final image, and keep both source and bytecode when copying packages between stages. Adding `-O` or changing the runtime cache path can change which files Python looks for. The [methods supplement](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#final-image-diagnostic) includes the cache-coverage command and compatibility caveats.

The extra 153 MB can matter on a slow image pull. Nodes with the layers already cached avoid that transfer when starting another container. Build-time overhead was not measured separately. Even with bytecode, the serving-CLI import still took about five seconds here; module initialization remains work to investigate.

[Methods and measurements](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md) give the exact environments, run order, controls and limitations, with [JSON](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/supplementary-measurements.json) and [CSV](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/measurements.csv) data. Thanks to Simon Mo for pushing me to control for the page cache and time the compiler on its own.
