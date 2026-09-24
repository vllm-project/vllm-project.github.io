---
layout: post
title: "Save 15s vLLM startup time with one uv flag"
author: "Nils Matteson"
summary: "Precompiling Python bytecode with one uv setting cut about 15 s (median of three pairs) from image pull to first vLLM response. How it works and what it costs."
image: /assets/figures/2026-09-22-python-bytecode-startup/image-delivery-pairs.png
tags:
  - performance
  - deployment
---

One build setting cut **about 15 seconds** from our vLLM test image's pull-to-first-response time, including the pull of its slightly larger image from a local registry:

```dockerfile
ENV UV_COMPILE_BYTECODE=1
```

This is the environment form of uv's `--compile-bytecode` flag. Put it before the uv commands that install your Python packages, then rebuild. uv compiles the Python source it installs, so new containers can use the bytecode from their first launch. In a multistage build, make sure those files reach the final image. [PR #55422](https://github.com/vllm-project/vllm/pull/55422) adds this line to eight vLLM Dockerfiles. As of September 22, it is approved and awaiting merge. Check your image's bytecode coverage before assuming it includes this change.

![Six stacked bars show pull/unpack and post-pull time for three matched pairs. Precompiled bytecode saves 15.4, 11.2 and 16.1 seconds.](/assets/figures/2026-09-22-python-bytecode-startup/image-delivery-pairs.png)

*Three pairs from empty Docker stores. Gray shows pull/unpack; blue shows everything after pull through the first correct response. Paired savings: 15.36, 11.21 and 16.10 seconds (median 15.36). Qwen3-0.6B was already local; the registry used uncapped loopback. Both images used gzip. Intel Core i9-13900HX, RTX 4090 Laptop GPU, four-CPU server quota. [Full setup](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#image-delivery).*

The image grew by **153 MB compressed, or 1.8%**. On the local registry that was hard to see: the precompiled image pulled 3.4 seconds faster in one pair and 2.5 and 1.2 seconds slower in the other two. From container start to first response, it was 11.9 to 17.3 seconds faster in every pair. Fresh containers were compiling thousands of unchanged Python files before they could serve a request.

## Does your image need it?

Probably, if it installs packages with uv. pip compiles bytecode by default; uv skips it to keep installs fast ([uv's pip compatibility notes](https://docs.astral.sh/uv/pip/compatibility/#bytecode-compilation)), and vLLM's Dockerfiles install with uv. The official `vllm/vllm-openai` image we counted on September 3 (vLLM 0.28.0) had **27,753 Python source files and 408 `.pyc` files**.

The main trap is an existing base image. With compilation enabled, `uv pip install` compiles the files it installs or reinstalls; sync operations process the environment's existing packages too ([uv reference](https://docs.astral.sh/uv/reference/cli/#uv-pip-install--compile-bytecode)). Adding the setting before installing one extra package does not necessarily precompile everything inherited from `FROM`. Compile the existing environment explicitly with the runtime Python's [compileall](https://docs.python.org/3.12/library/compileall.html) if needed. In our matched build, all 26,734 installed source files had valid bytecode.

Check the **final image**, using the Python interpreter, optimization settings and cache location your service will use. For its default Python settings:

```sh
docker run --rm -i --read-only --network none --entrypoint python3 your-image -B - <<'PY'
import importlib.util
from pathlib import Path
import site

sources = {p for d in site.getsitepackages() for p in Path(d).rglob("*.py")}
missing = sum(
    not Path(importlib.util.cache_from_source(str(p))).is_file()
    for p in sources
)
print(f"{missing} of {len(sources)} installed .py files have no .pyc")
PY
```

This checks for missing cache files; it does not validate existing ones. `-B` prevents the check from writing new caches. Keep the source files for normal invalidation and fallback. Caches left in a builder stage won't help the final image, and changing `PYTHONPYCACHEPREFIX` or adding `-O` at runtime can change which cache Python looks for.

The extra **153 MB compressed** is about **418 MB of added bytecode** before compression. For scale, transferring 153 MB takes about 1.2 seconds at 1 Gbit/s or 12 seconds at 100 Mbit/s, ignoring protocol and unpacking costs. Those are arithmetic examples, not measured pull times or a measured break-even. The added transfer is paid when a node fetches the new layers; later containers can reuse them. Build-time overhead wasn't measured separately.

For emulated or reproducible builds, there are two upstream caveats: compilation has hung in affected QEMU setups, and equivalent `.pyc` contents can have different serialized bytes and image digests. The [compatibility notes](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#compatibility-and-build-caveats) link the reports.

## Wait, we're compiling Python on every fresh container?

CPython compiles source into code objects before executing it. A [valid `.pyc` file](https://docs.python.org/3.12/reference/import.html#cached-bytecode-invalidation) stores that compiled code so Python can load it directly next time. It still executes the module's top-level code. These are Python imports, which happen before the [`torch.compile`](https://vllm.ai/blog/2025-08-20-torch-compile) and GPU kernel compilation stages; those keep their own caches, which this setting doesn't touch.

Python can write missing caches as the container runs, when it has somewhere writable to put them. The next fresh container starts from the image, though, and the previous container's writable layer doesn't come along with it. If the image ships mostly source, each new container has the same work waiting for it.

uv's own [Docker guide](https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode) already recommends this for production images. We wanted to know how much it mattered for vLLM.

We built two images from the same vLLM development commit with vLLM's CUDA Dockerfile, compiled only for our test GPU's architecture, with and without the setting. We checked that their Python interpreter, installed package versions, source files and native payloads matched. Then we traced a serving-CLI import in a fresh container from each:

| Matched Docker images | Default build | Precompiled build |
|---|---:|---:|
| Installed package source files | 26,734 | 26,734 |
| Sources with valid bytecode | 407 | 26,734 |
| Source compilations during serving-CLI import | 5,257 | 0 |

To check that Python was actually using the shipped caches, we pointed the precompiled image at an empty alternative cache directory. Source compilation came back. The packages and source hadn't changed; we had only made the bytecode unavailable to the loader.

## How much of that is compiler time?

Outside Docker, in a released vLLM 0.28.0 installation, a fresh process took a median **12.27 seconds** to import the serving CLI without bytecode and **5.02 seconds** with it. Every entry point we timed got faster:

![Import times for torch, transformers, vllm and the vLLM serving CLI, three runs each with and without bytecode. Medians fall from 2.13 to 0.80 s, 1.29 to 0.68 s, 6.71 to 2.11 s and 12.27 to 5.02 s.](/assets/figures/2026-09-22-python-bytecode-startup/import-times.png)

*vLLM 0.28.0 on the same machine, three fresh-process observations per condition, with source and native-library pages resident and no compiler instrumentation. Dots are individual runs; the vertical tick is the median. The clock covers the import alone. Each root includes its dependencies, so the rows are not additive. The serving CLI root is `vllm.entrypoints.cli.serve`.*

In this release, `import vllm` already loads Dynamo, Inductor and SymPy; a plain `import torch` leaves them unloaded. Most of the source belongs to dependencies, but vLLM's import path decides when it loads.

To isolate the compiler, we traced which files the serving-CLI import compiled: **5,118 files containing 78.6 MB of source** in this installation. (The test images came from a different commit with a larger package set, which is why they compiled 5,257.) We read that whole corpus into memory before timing Python's compiler directly:

```python
# Read and verify the sources before starting the clock.
for filename, source_bytes in preloaded_sources:
    compile(source_bytes, filename, "exec", dont_inherit=True, optimize=0)
```

Compiling and discarding the code objects took **6.30 seconds median across three passes**, with 6.30 seconds of current-thread CPU time. No file reads, module execution or bytecode writes occurred inside the timed loop. That's about **1.23 milliseconds per file**.

In a separate instrumented import, one run per condition, the uncached serving CLI spent **6.99 seconds** inside those 5,118 compiler calls, and the import finished **7.19 seconds** sooner with valid bytecode. That accounts for most of this import difference, with about 0.20 seconds left over. The earlier [PR discussion](https://github.com/vllm-project/vllm/pull/55422#issuecomment-5608794542) quotes 8.20 seconds from a September 9 trace under a different CPython build; these figures come from September 15. The [methods](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md#instrumented-import-diagnostic) give each experiment's environment and timing boundaries.

## Could this just be the page cache?

A faster second launch wouldn't establish this on its own. The first launch can warm the OS file cache and write new `.pyc` files. Importing twice in one interpreter can also reuse the module through `sys.modules`.

We ran a separate serving comparison outside Docker to control those effects. Before every fresh server launch, we made the same **8.85 GB of Python and native-library files resident in memory** and checked their residency. One condition started with precompiled installed-package bytecode; the other started without it.

Launch to first correct response took 54.7 to 60.6 seconds without starting bytecode and 44.6 to 48.4 seconds with it. The three paired savings were **10.13, 9.35 and 12.23 seconds**. All six launches returned correct responses. Having the measured source pages in memory didn't remove the bytecode saving.

This test starts at server launch. The Docker study uses a different vLLM build, Python build, CPU limit and timing boundary. We haven't isolated their contributions to the different savings, and the two results aren't additive. Bytecode writes and child-process reuse stayed enabled here, so this saving includes cache I/O and multiprocessing effects, not just compiler time.

Even with bytecode, the serving-CLI import still takes about five seconds here, so there is more to cut. Recompiling the same unchanged files in every new container is the easy part to remove: the source is fixed when the image is built, so compile it there, once.

## Measurements and methods

The [methods supplement](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/METHODS.md) contains the exact environments, run order, image-identity checks and limitations. [JSON](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/supplementary-measurements.json) contains the numerical data for all experiments; [CSV](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2026-09-22-python-bytecode-startup/measurements.csv) contains the native serving and import observations. The PR discussion records [the page-cache question](https://github.com/vllm-project/vllm/pull/55422#issuecomment-5605837756) and [the controlled follow-up](https://github.com/vllm-project/vllm/pull/55422#issuecomment-5608794542). The full harnesses and raw logs have not yet been published.

Thanks to Simon Mo for review, and for pushing us to control for the page cache and to time the compiler on its own.
