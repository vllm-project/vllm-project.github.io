---
layout: post
title: "Speculators v0.5.0: DFlash Support and Online Training"
author: "Fynn Schmitt-Ulms, Helen Zhao, Rahul Tuli and Dipika Sikka (Red Hat AI Model Optimization Team)"
image: /assets/figures/2026-05-28-speculators-v050/gemma4-dflash-acceptance-rates.png
tags:
  - speculative-decoding
  - ecosystem
---

The [v0.5.0 release](https://github.com/vllm-project/speculators/releases/tag/v0.5.0) brings significant architectural improvements to speculative decoding model training, introducing DFlash algorithm support, fully unified online training capabilities, and a complete migration to vLLM's native hidden states extraction system. This release represents a major step forward in both training flexibility and production readiness for speculative decoding workflows.

Key features include:

* DFlash algorithm support - Single-pass draft token generation with block diffusion
* Gemma 4 DFlash results
* vLLM-native online and offline training with unified hidden states extraction
* Updated documentation and examples outlining key workflows

## DFlash Algorithm Support

v0.5.0 introduces training support for the DFlash speculative decoding algorithm, a fundamentally different approach to draft token generation compared to the autoregressive Eagle 3 models. While Eagle 3 generates draft tokens autoregressively through multiple forward passes, DFlash employs block diffusion to generate all draft tokens in a single forward pass.

The single-pass nature of DFlash can dramatically reduce the overhead of speculative decoding, particularly for longer draft sequences. The drafter produces a block of tokens of length B for each prefix. This block structure is entirely accomplished using the attention mask. Another key difference from Eagle3 is that DFlash uses a non-causal attention pattern where queries within a block can attend to all other tokens within the same block.

During training, multiple predicted blocks are trained on in parallel. A straightforward approach would be to start a prediction block after every possible point in the sequence. For a long sequence, however, this causes the attention mask to grow extremely large, making training impractical in both memory usage and compute cost. To avoid this, we do not start blocks everywhere. Instead, we randomly choose a smaller set of “anchor” positions from locations that actually contribute to the training loss. Predicted blocks are only attached to these anchors. This keeps the number of predicted blocks fixed regardless of sequence length, allowing training to scale to much longer contexts while keeping the attention mask manageable.

## Training a DFlash Speculator

Training a DFlash model follows a similar online workflow to Eagle 3\. A complete tutorial is provided [here](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_dflash_online/).

The key difference from Eagle 3 is the speculator-specific parameters in the training command, as shown below:

```bash
torchrun --standalone --nproc_per_node 2 scripts/train.py \
    --verifier-name-or-path "Qwen/Qwen3-8B" \
    --vllm-endpoint "http://localhost:8000/v1" \
    --speculator-type dflash \
    --draft-vocab-size 8192 \
    --block-size 8 \
    --max-anchors 3072 \
    --num-layers 5 \
    --target-layer-ids "2 18 33" \
    --epochs 5 --lr 1e-4
```

DFlash-specific parameters include:

```bash
--block-size # Number of tokens generated per diffusion block
--max-anchors # Maximum anchor points for speculation during training
--speculator-type # Must specify dflash
```

## Gemma 4 DFlash Speculator

Using the DFlash algorithmic support, a [Gemma 4 31B DFlash speculator](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dflash) was trained and acceptance rates were evaluated across diverse task types. The results demonstrate strong performance particularly on reasoning and code generation tasks:
<p align="center">
<img src="/assets/figures/2026-05-28-speculators-v050/gemma4-dflash-acceptance-rates.png">
<br>
<em>Figure 1: Gemma 4 DFlash acceptance rates across diverse task types.</em>
</p>

Gemma 4 DFlash achieves better inter-token latency than both Eagle 3 and a standalone FP8 quantized verifier. Combining DFlash with an FP8 quantized verifier yields even greater gains, as shown below:

<p align="center">
<img src="/assets/figures/2026-05-28-speculators-v050/gemma4-dflash-latency.png">
<br>
<em>Figure 2: Gemma 4 DFlash inter-token latency comparison.</em>
</p>

## Serving DFlash models in vLLM

DFlash models integrate seamlessly with vLLM's speculative decoding infrastructure, as of PR [\#38300](https://github.com/vllm-project/vllm/pull/38300), which is included in `vllm>=0.20.0`.

Similar to the Eagle 3 models, DFlash models contain a `speculators_config` in their config.json which contain details on the target model, speculative tokens, the name of the speculative algorithm, etc. With this config, models can be served using a basic `vllm serve` command, as shown below.

```shell
vllm serve -tp 2 RedHatAI/gemma-4-31B-it-speculator.dflash
```

## Unified Online and Offline Training Support

v0.5.0 adds native support for both online and offline training modes through [vLLM's hidden states extraction system](https://vllm.ai/blog/extract-hidden-states) (introduced in vLLM v0.18.0). Previous versions of Speculators extracted hidden states using lower-level utilities from vLLM, requiring vLLM to be a direct python dependency. This approach tightly coupled the training pipeline to vLLM's internal APIs, which often changes between vLLM version updates and required manual synchronization with upstream changes. This integration removes the previous custom data generation pipeline and eliminates vLLM as a direct python dependency.

Both training modes now use the same vLLM-based extraction path:

* Online training: Extract hidden states on-the-fly during training
* Offline training: Pre-generate and cache hidden states to disk, then train

By leveraging vLLM's native hidden states extraction, Speculators inherits all of vLLM's inference optimizations including efficient memory management, batching strategies, and hardware acceleration support. Training now communicates with a running vLLM server via its standard REST API, decoupling the training infrastructure from vLLM's internal implementation details. This architectural shift provides better version stability and makes it easier for teams to update vLLM independently of the Speculators training framework.

What happens during online training:

1. vLLM server initializes with the base model (and some special configuration)
2. Training prompts are sent to vLLM for inference
3. Hidden states are extracted and temporarily written to disk (or ram disk)
4. The training process loads the extracted hidden states and deletes the file
5. Speculator model trains on extracted states

[This tutorial](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_eagle3_online/) provides more information on the online training workflow.

Offline data generation has also been updated to use the same hidden states extraction system and data format as online. New scripts have been developed to saturate the running vLLM server with requests and write them to disk. The two approaches are so tightly coupled that you can even run a combination of them. For example, you can partially generate hidden states offline and then run training and load the existing hidden states, while generating any that are missing. You can also run an online training job that does not clear the files after generating them, allowing you to generate once on the first epoch and then load the files on subsequent ones.

[This tutorial](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_eagle3_offline/) goes into more detail on the offline training workflow.

## Added Comprehensive Documentation

Another feature worth highlighting is the updated [documentation site](https://docs.vllm.ai/projects/speculators/en/latest/). We’ve added concise introductions to the speculative decoding algorithms supported by Speculators, along with detailed tutorial walkthroughs for training speculator models. For developers, we also introduced a guide covering how to add new speculative decoding algorithms to the Speculators library, as well as a comprehensive API reference.

