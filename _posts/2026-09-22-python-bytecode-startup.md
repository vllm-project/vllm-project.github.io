---
layout: post
title: "Save 15s vLLM startup time with one uv flag"
author: "Nils Matteson"
summary: "Precompiling Python bytecode saved 15 seconds from image pull to first response in a matched vLLM experiment. How one uv setting works, how we measured it, and the image-size tradeoff."
image: /assets/figures/2026-09-22-python-bytecode-startup/image-delivery-pairs.png
tags:
  - performance
  - deployment
---

One Dockerfile setting got our vLLM test image from pull to first correct response **about 15 seconds sooner**, even after accounting for a larger image:

```dockerfile
ENV UV_COMPILE_BYTECODE=1
```

Put it before the uv commands that install your Python packages, then rebuild. uv will compile Python source during installation, so new containers can use the bytecode from their first launch. In a multistage build, make sure those files reach the final image. [PR #55422](https://github.com/vllm-project/vllm/pull/55422) applies this setting across eight vLLM Dockerfiles.

![Three paired Docker delivery measurements. The precompiled image reaches its first correct response sooner in every pair.](/assets/figures/2026-09-22-python-bytecode-startup/image-delivery-pairs.png)

*Pull/unpack to first correct response, three pairs from empty Docker stores. Qwen3-0.6B was already local; the registry used uncapped loopback. Both images used gzip. RTX 4090 Laptop GPU, four-CPU server quota. Median paired saving: 15.36 seconds. [Full setup](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#image-delivery).*

The image grew by **153 MB compressed, or 1.8%**, and still finished sooner in all three pairs. The tradeoff depends on the network, but the work being avoided is the same: fresh containers were compiling thousands of unchanged Python files.

## Wait, we're compiling Python on every fresh container?

CPython compiles source into code objects before executing it. A [valid `.pyc` file](https://docs.python.org/3.12/reference/import.html#cached-bytecode-invalidation) stores that compiled code so Python can load it directly next time. It still executes the module's top-level code. These are Python imports, before the later `torch.compile` and GPU kernel compilation stages.

Python can write missing caches as the container runs. The next fresh container starts from the image, though, and the previous container's writable layer doesn't come along with it. If the image ships mostly source, each new container has the same work waiting for it.

uv skips bytecode compilation by default to keep installation fast. That saves work during the build, but a serving image can be started many times. [uv's Docker guide](https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode) already recommends enabling compilation for production images. The setting was there; we wanted to know how much it mattered for vLLM.

We built two images from the same vLLM source, with and without the setting. We checked that their Python interpreter, installed package versions, source files and native payloads matched. Then we traced a serving-CLI import in a fresh container from each:

| In the matched images | Default build | Precompiled build |
|---|---:|---:|
| Installed package source files | 26,734 | 26,734 |
| Sources with valid bytecode | 407 | 26,734 |
| Source compilations during serving-CLI import | 5,257 | 0 |

To check that Python was actually using the shipped caches, we pointed the precompiled image at an empty alternative cache directory. Source compilation came back. The packages and source hadn't changed; we had only made the bytecode unavailable to the loader.

That explains *what* the setting avoids. But five thousand small compilations taking seconds still deserved a closer look.

## Just call the compiler

We traced the serving CLI separately in a vLLM 0.28.0 environment outside Docker. It compiled **5,118 files containing 78.6 MB of source**, mostly from dependencies. We read that entire corpus into memory before timing Python's compiler directly:

```python
# Read and verify the sources before starting the clock.
for filename, source_bytes in preloaded_sources:
    compile(source_bytes, filename, "exec", dont_inherit=True, optimize=0)
```

Compiling and discarding the code objects took **6.30 seconds median across three passes**. CPU time was also 6.30 seconds. No file reads, module execution or bytecode writes occurred inside that timed loop.

That's about **1.23 milliseconds per file** on average. A single compilation is small; the import tree contains thousands of them.

We also timed compiler calls inside a separate serving-CLI import:

| Diagnostic measurement | No initial bytecode | Valid bytecode |
|---|---:|---:|
| Import elapsed time | 12.20 s | 5.01 s |
| Source compiler calls | 5,118 | 0 |
| Time inside compiler calls | 6.99 s | 0 s |

The import got **7.19 seconds** faster, with **6.99 seconds** spent in compiler calls in the uncached run. Those calls disappear with valid bytecode. That accounts for most of the import difference, with about 0.20 seconds left over.

These diagnostic imports were instrumented, with one run per condition. They account for most of the import gap; they don't explain every second saved during full server startup. The [methods](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#direct-source-compilation) record the Python builds and timing boundaries for each experiment.

## Could this just be the page cache?

A faster second launch wouldn't establish this on its own. The first launch can warm the OS file cache and write new `.pyc` files. Importing twice in one interpreter can also reuse the module through `sys.modules`.

We ran a separate serving comparison outside Docker to control those effects. Before every fresh server launch, we made the same **8.85 GB of Python and native-library files resident in memory** and checked their residency. One condition started with bytecode; the other started without it.

The three paired savings to first correct response were **10.13, 9.35 and 12.23 seconds**. All six launches returned correct responses. Having the measured source pages in memory didn't remove the cost of compiling them.

This ten-second result starts at server launch. The opening Docker result also includes image pull and unpack; the two savings aren't additive. Normal bytecode writes and child-process reuse remained enabled in the serving test, so its saving also includes cache I/O and multiprocessing effects. It isn't ten seconds of pure compiler CPU time.

## Where the imports come from

The serving CLI pulls in a much larger import tree than a bare `import torch`. In fresh-process measurements, its median import time fell from **12.27 to 5.02 seconds** with bytecode. Here is the comparison across four entry points:

![Individual timings and medians for four independent Python import roots, with and without bytecode.](/assets/figures/2026-09-22-python-bytecode-startup/import-times.png)

*Same vLLM 0.28.0 environment, three fresh-process observations per condition, with source/native-library pages resident and no compiler instrumentation. Points show observations; markers show medians. Timing covers the import alone. Each root includes its dependencies, so the rows are not additive. The serving CLI root is `vllm.entrypoints.cli.serve`.*

`import vllm` already pulls in Dynamo, Inductor and SymPy in this environment; a plain `import torch` leaves those unloaded. Bare Transformers also uses lazy loading, so its import time doesn't cover everything a serving process will eventually need.

This distinction matters for the next optimization. Most of the source belongs to dependencies, but vLLM's import path controls when much of it loads. Bytecode makes those imports cheaper. Separating CLI declarations from runtime setup can keep commands such as help and version from needing that import tree at all.

## What changes for a real deployment?

The matched images' gzip-compressed runtime layers grew from **8.512 GB to 8.665 GB**. A slower registry connection makes those extra 153 MB more expensive; at sufficiently low bandwidth, transfer time could outweigh the startup saving. Our loopback test doesn't settle that tradeoff for your network. Image-build overhead wasn't measured in isolation either.

The implementation uses CPython's existing cache mechanism. There are still a few build details worth checking:

- **Check the final image.** Caches left in a builder stage won't help the running container. The setting applies to uv operations, and its coverage depends on the command and uv version. Check the resulting runtime environment. [uv's install reference](https://docs.astral.sh/uv/reference/cli/#uv-pip-install--compile-bytecode) describes the behavior.
- **Use the runtime's Python and retain source.** Build caches with the interpreter the image will use. Keeping source lets Python handle normal cache invalidation and fallback.
- **Check special build requirements.** Reproducible image bytes and QEMU-emulated builds have known bytecode-related caveats. The [compatibility notes](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#compatibility-and-build-caveats) cover the upstream reports and portability check.

Even with bytecode, the serving-CLI import still took about five seconds here. There is more work to do. But the Python source shipped in an image is already known at build time. Shipping its compiled form lets every new replica reuse that preparation.

## Measurements and methods

The [methods supplement](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md) contains the exact environments, run order, image-identity checks and limitations. [JSON](/assets/figures/2026-09-22-python-bytecode-startup/supplementary-measurements.json) contains the numerical data for all experiments; [CSV](/assets/figures/2026-09-22-python-bytecode-startup/measurements.csv) contains the native serving and import observations. The [PR discussion](https://github.com/vllm-project/vllm/pull/55422#issuecomment-5608794542) records the page-cache question and controlled follow-up. The full harnesses and raw logs have not yet been published.

Thanks to Simon Mo for review and for pushing on filesystem-cache attribution and independent source-compilation measurements.
