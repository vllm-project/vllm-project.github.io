---
layout: post
title: "Installing and Developing vLLM with Ease"
author: "vLLM Team"
image: /assets/logos/vllm-logo-only-light.png
---

The field of LLM inference is advancing at an unprecedented pace. With new models and features emerging weekly, the traditional software release pipeline often struggles to keep up. At vLLM, we aim to provide more than just a software package. We’re building a system—a trusted, trackable, and participatory ecosystem for LLM inference. This blog post highlights how vLLM enables users to install and develop with ease while staying at the forefront of innovation.

## TL;DR:

* Flexible and fast installation options from stable releases to nightly builds.  
* Streamlined development workflow for both Python and C++/CUDA developers.  
* Robust version tracking capabilities for production deployments.

## Seamless Installation of vLLM Versions

### Install Released Versions

We periodically release stable versions of vLLM to the [Python Package Index](https://pypi.org/project/vllm/), ensuring users can easily install them using standard Python package managers. For example:

```sh
pip install vllm
```

For those who prefer a faster package manager, [**uv**](https://github.com/astral-sh/uv) has been gaining traction in the vLLM community. After setting up a Python environment with uv, installing vLLM is straightforward:

```sh
uv pip install vllm
```

Refer to the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html?device=cuda#create-a-new-python-environment) for more details on setting up [**uv**](https://github.com/astral-sh/uv). Using a simple server-grade setup (Intel 8th Gen CPU), we observe that [**uv**](https://github.com/astral-sh/uv) is 200x faster than pip:

```sh
# with cached packages, clean virtual environment
$ time pip install vllm
...
pip install vllm 59.09s user 3.82s system 83% cpu 1:15.68 total

# with cached packages, clean virtual environment
$ time uv pip install vllm
...
uv pip install vllm 0.17s user 0.57s system 193% cpu 0.383 total
```

### Install the Latest vLLM from the Main Branch

To meet the community’s need for cutting-edge features and models, we provide nightly wheels for every commit on the main branch.

**Using pip**:

```sh
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

Adding `--pre` ensures pip includes pre-released versions in its search.

**Using uv**:

```sh
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
```

## Development Made Simple

We understand that an active, engaged developer community is the backbone of innovation. That’s why vLLM offers smooth workflows for developers, regardless of whether they’re modifying Python code or working with kernels.

### Python Developers

For Python developers who need to tweak and test vLLM’s Python code, there’s no need to compile kernels. This setup enables you to start development quickly.

```sh
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install -e .
```

The `VLLM_USE_PRECOMPILED=1` flag instructs the installer to use pre-compiled CUDA kernels instead of building them from source, significantly reducing installation time. This is perfect for developers focusing on Python-level features like API improvements, model support, or integration work.

This lightweight process runs efficiently, even on a laptop. Refer to our [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html?device=cuda#build-wheel-from-source) for more advanced usage.

### C++/Kernel Developers

For advanced contributors working with C++ code or CUDA kernels, we incorporate a compilation cache to minimize build time and streamline kernel development. Please check our [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html?device=cuda#build-wheel-from-source) for more details.

## Track Changes with Ease

The fast-evolving nature of LLM inference means interfaces and behaviors are still stabilizing. vLLM has been integrated into many workflows, including [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [veRL](https://github.com/volcengine/verl), [open_instruct](https://github.com/allenai/open-instruct), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), etc. We collaborate with these projects to stabilize interfaces and behaviors for LLM inference. To facilitate the process, we provide powerful tools for these advanced users to track changes across versions.

### Installing a Specific Commit

To simplify tracking and testing, we provide wheels for every commit in the main branch. Users can easily install any specific commit, which can be particularly useful to bisect and track the changes.

We recommend using [**uv**](https://github.com/astral-sh/uv) to install a specific commit:

```sh
# use full commit hash from the main branch
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
```

In [**uv**](https://github.com/astral-sh/uv), packages in `--extra-index-url` have [higher priority than the default index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes), which makes it possible to install a developing version prior to the latest public release (at the time of writing, it is v0.6.6.post1).

In contrast, pip combines packages from `--extra-index-url` and the default index, choosing only the latest version, which makes it difficult to install a developing version prior to the released version. Therefore, for pip users, it requires specifying a placeholder wheel name to install a specific commit:

```sh
# use full commit hash from the main branch
export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd
pip install https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

## Conclusion

At vLLM, our commitment extends beyond delivering high-performance software. We’re building a system that empowers trust, enables transparent tracking of changes, and invites active participation. Together, we can shape the future of AI, pushing the boundaries of innovation while making it accessible to all.

For collaboration requests or inquiries, reach out at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu). Join our growing community on [GitHub](https://github.com/vllm-project/vllm) or connect with us on the [vLLM Slack](https://slack.vllm.ai/). Together, let’s drive AI innovation forward.

## Acknowledgments

We extend our gratitude to the [uv community](https://docs.astral.sh/uv/) — particularly [Charlie Marsh](https://github.com/charliermarsh) — for creating a fast, innovative package manager. Special thanks to [Kevin Luu](https://github.com/khluu) (Anyscale), [Daniele Trifirò](https://github.com/dtrifiro) (Red Hat), and [Michael Goin](https://github.com/mgoin) (Neural Magic) for their invaluable contributions to streamlining workflows. [Kaichao You](https://github.com/youkaichao) and [Simon Mo](https://github.com/simon-mo) from the UC Berkeley team lead these efforts.  
