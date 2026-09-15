---
layout: post
title: "How we trained the fastest DSpark for Kimi-K3"
author: "Helen Zhao, Fynn Schmitt-Ulms, Yuchen Fama, and Antonio Javier Fernandez Dominguez"
summary: "How Speculators and Mooncake enabled multi-node DSpark training for Kimi K3."
image: /assets/figures/2026-09-15-kimi-k3-dspark/throughput-vs-interactivity.png
social_image: /assets/figures/2026-09-15-kimi-k3-dspark/throughput-vs-interactivity.png
tags:
  - speculators
  - speculative-decoding
  - performance
  - distributed
---

In June, DeepSeek shipped [DSpark](https://arxiv.org/abs/2607.05147), an extension to the DFlash block-level speculative decoding algorithm. The new algorithm promised stronger intertoken coherence and therefore better acceptance lengths. But the real question for the open-source community is always the same: can you train it, package it, and deploy it without a PhD student babysitting the checkpoint?

Thanks to the vLLM Project's [**Speculators** training library](https://github.com/vllm-project/speculators), the answer is a resounding yes! With Speculators it’s easy to train, package, and deploy DSpark draft models in a standard, Hugging Face-compatible format that vLLM loads directly. The implementation has already been validated on [**Qwen3.6-35B-A3B**](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-speculator.dspark), [**Gemma-4-31B-it**](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dspark), [**GLM-5.2**](https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark), and more. This blog post explores how we extended our training library to support Kimi K3, a 2.8T-parameter frontier model. Our new DSpark speculator boosts single-stream interactivity from ~110 to ~435 tok/s/user on math reasoning, while delivering up to **~3.5× higher output throughput** at matched interactivity under concurrent load.

![Kimi K3 output throughput versus interactivity with and without the DSpark speculator](/assets/figures/2026-09-15-kimi-k3-dspark/throughput-vs-interactivity.png)

*Figure 1. The Kimi K3 DSpark speculator raises both single-stream interactivity and aggregate output throughput on math reasoning workloads.*

## What Is DSpark?

Large language models generate text one token per forward pass. Speculative decoding accelerates this process by having a lightweight drafter propose several tokens, which the full target model verifies together.

[EAGLE-3](https://arxiv.org/abs/2503.01840) is a strong baseline, but it still drafts autoregressively: seven proposed tokens require seven sequential drafting steps. [DFlash](https://arxiv.org/abs/2602.06036) instead predicts an entire block in one non-causal backbone pass, reporting up to 2.5× greater acceleration than EAGLE-3.

The tradeoff is that parallel positions cannot condition on one another. For a prompt “Thank you!” the model might independently favor responses of both “Of” and “No” at position one and both “course” and “problem” at the next, producing mismatches such as “Of problem.” Because verification stops at the first rejected token, one mistake also invalidates the remaining suffix—a problem the [DSpark paper](https://arxiv.org/abs/2607.05147) calls *suffix decay*.

DSpark preserves DFlash’s parallel backbone while adding two lightweight components:

- The **Markov logit-bias head** samples tokens sequentially and adjusts each position’s logits using the previously selected token. Its low-rank transition matrix restores important local dependencies without another transformer pass.
- The **confidence head** estimates the probability that each token will be accepted. A hardware-aware scheduler uses these estimates to verify longer prefixes under light load and trim unlikely suffixes when the system is busy.

DSpark therefore keeps the main advantage of parallel drafting by inheriting the single backbone pass, while recovering some of the coherence of autoregressive generation. Across Qwen3 target models, it reports 16–18% longer accepted sequences than DFlash and 27–31% longer sequences than EAGLE-3. In DeepSeek-V4 production serving, it improved per-user generation speed by 60–85% over the previous MTP-1 baseline at matched throughput.

![DSpark architecture combining parallel drafting, sequential correction, confidence scoring, and hardware-aware verification](/assets/figures/2026-09-15-kimi-k3-dspark/dspark-architecture.png)

*Figure 2. DSpark combines a parallel block with sequential correction and hardware-aware prefix scheduling before target-model verification.*

## Performance During Inference

DSpark’s semi-autoregressive design improves inference only when the additional sequential work remains much cheaper than another draft-model forward pass. The released [Kimi K3 DSpark speculator](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark) uses a five-layer, five-billion-parameter draft model and proposes eight tokens per decoding step.

Across nine evaluation domains, it achieves a macro-average acceptance length of 4.11 tokens per verification round. Performance is strongest on structured tasks: 6.42 tokens for mathematical reasoning, 4.96 for HumanEval, and 4.65 for translation. The speedup is especially significant at low request counts.

This model works especially well for long-context prompts. On a challenging domain-specific dataset like LongBench-v2, our Kimi K3 DSpark reached up to 5.31 output tokens per decode iteration on a 378K-token prompt. Even across the broader workload, the top 10% of requests achieved at least 3.76 tokens per iteration, demonstrating that deep speculative runs remain possible at genuinely long context lengths.

It scales effectively as more requests arrive simultaneously. Increasing concurrency from 1 to 16 raises aggregate output throughput from 177 to 683 tokens per second.

Crucially, response startup remains fast under load. Despite serving 16 times as many concurrent requests, median time to first token increases by only 100 milliseconds, from 379 to 479 milliseconds.

Deployment is easy. Follow the official vLLM [recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3) for your hardware and use case:

```shell
docker run --gpus all \
  --privileged --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e GLOO_SOCKET_IFNAME=$IFACE_NAME \
  -e NCCL_SOCKET_IFNAME=$IFACE_NAME \
  -e VLLM_ALLREDUCE_USE_FLASHINFER=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_RUST_FRONTEND=1 \
  vllm/vllm-openai:latest moonshotai/Kimi-K3 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr $HEAD_IP \
  --load-format fastsafetensors \
  --no-enable-flashinfer-autotune \
  --max-model-len 1048576 \
  --kv-cache-dtype fp8 \
  --attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"TOKENSPEED_MLA"}' \
  --enable-prefix-caching \
  --attention-backend TOKENSPEED_MLA \
  --prefix-match-unit 128 \
  --reasoning-parser kimi_k3 \
  --language-model-only \
  --speculative-config '{"model":"RedHatAI/Kimi-K3-speculator.dspark", "num_speculative_tokens":8, "method":"dspark", "draft_sample_method":"probabilistic", "rejection_sample_method":"block"}'
```

## Hardware Setup

This model was trained on a GB300 rack generously provided by Verda. Verda is a European AI cloud that owns the stack end to end, from data centers to managed services, and runs on 100% renewable energy. Verda was among the first providers in Europe to deploy GB300 NVL72 racks. Its in-house AI Lab also conducts inference research on the same racks.

The GB300 rack keeps the NVIDIA reference design and runs bare metal, without virtualization, using Ubuntu 24.04.4 LTS on NVIDIA's 64K-page kernel 6.14.

Verda focuses its optimization efforts on the system-level configuration choices that matter most, giving researchers access to a frontier-class environment. The system uses NVIDIA’s 610.57.04 open-kernel GPU driver (R610). CUDA 13.4.0 Developer Preview is installed alongside CUDA 13.1 and 12.9, while NCCL 2.31.2 is available system-wide.

CUDA 13.4.0 was selected for its new Blackwell programming features and because it is the [first toolkit to include Rubin support](https://github.com/pytorch/pytorch/pull/190654) (`sm_107`). This helps ensure that code written and profiled on GB300 today can be built for the next generation of hardware without unexpected compatibility issues. Hardware-counter-based GPU profiling is also available without root access, allowing every researcher to profile both training and inference workloads.

We have been collaborating with Verda to help open-source training and inference infrastructure scale to the latest GPU architectures, with a current focus on rack-scale systems spanning GB300 through VR200. We are grateful to see the results of this collaboration taking shape.

## Extracting Hidden States at Scale

Part of what makes speculative decoding draft models so powerful despite their small size is that they often take hidden states from the target model as inputs to inform their predictions. This greatly improves their working context and makes it possible for the drafters to closely align their predictions with the target models.

The caveat is that training these drafters requires a dataset of hidden-state inputs and target-model log-probability outputs. Thankfully, vLLM has a hidden-state extraction system that makes it possible to get internal target hidden states on demand for dataset samples. This system uses a dummy draft model, which reuses vLLM’s draft-model plumbing to receive target hidden states and inserts them into the KV cache of a dummy attention layer. From there, a class implementing the `KVConnector` interface can retrieve and transfer the hidden states out of vLLM. Currently vLLM ships with an `ExampleHiddenStatesConnector` that does just that, writing the hidden states asynchronously to disk.

This system works well for single-node vLLM-and-training configurations. For example, half the node’s GPUs can be devoted to training, while the other half serve the target model in vLLM and extract hidden states on demand. This system has been well supported in both Speculators and vLLM for months. However, with a 2.8T-parameter model like Kimi K3, even state-of-the-art accelerators start to run into VRAM limits despite 4-bit quantization for model weights. We need a system that scales beyond single-node training and enables disaggregated training and hidden-state extraction.

![Mooncake hidden-state transfer between Speculators training and vLLM inference processes](/assets/figures/2026-09-15-kimi-k3-dspark/mooncake-hidden-state-transfer.png)

*Figure 3. The Mooncake connector separates the control path from the hidden-state data path, using RDMA or TCP for transfers between vLLM and the Speculators dataloader.*

With these requirements in mind, we built the `MooncakeHiddenStatesConnector`, which uses the Mooncake transfer engine as its backend for streaming hidden states between processes and across nodes. The new system uses a master Mooncake proxy process to manage communication with vLLM and training instances that register themselves as clients. Once set up, training processes can send requests to the vLLM frontend and receive a Mooncake store key in response. That key is then provided to the Mooncake master, which brokers a transfer from the vLLM engine to the Speculators dataloader. All of this happens automatically, managed by the Mooncake server, and—depending on configuration—uses high-speed RDMA transfers or regular TCP to send data between processes on the same or different nodes.

## Training Kimi K3 DSpark

![Twelve-node Kimi K3 DSpark training topology with disaggregated vLLM inference and Mooncake transfers](/assets/figures/2026-09-15-kimi-k3-dspark/training-topology.png)

*Figure 4. Each four-node set dedicates one four-GPU node to training and two four-GPU nodes to vLLM inference, with Mooncake transferring hidden states.*

With the new `MooncakeHiddenStatesConnector` in hand, we now have a system capable of scaling to large-model, multi-node training. Even with Kimi K3 quantized to 4 bits, the model still requires at least two GB300 nodes (four GPUs each) to serve. We experimented with different training and vLLM configurations and found that sets of three nodes—two for inference and one for training—offered the best throughput. Another advantage of Speculators with the Mooncake connector is how easy it is to independently scale either component up or down for the best performance.

## Conclusion

[Speculators](https://github.com/vllm-project/speculators) is an open-source library for building, training, evaluating, and sharing speculative decoding models that integrate directly with inference engines such as vLLM. Whether you are training a new draft model, exploring algorithms such as DFlash, DSpark, or DFlash2, or improving production inference performance, we welcome your ideas and contributions. Join the [vLLM Community Slack](https://slack.vllm.ai/) and find us in `#speculators` and `#feat-spec-decode` to ask questions, share results, discuss new algorithms, and collaborate with the community. Contributions to the code, documentation, examples, model support, and evaluation tooling are all welcome.
