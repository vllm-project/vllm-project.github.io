---
layout: post
title: "DiffusionGemma: The First Diffusion LLM (dLLM) Natively Supported in vLLM"
author: "The vLLM Team and Google Team"
image: /assets/figures/2026-06-10-diffusion-gemma/ar-vs-diffusion.svg
summary: "DiffusionGemma is the first diffusion language model (dLLM) supported in vLLM. We integrated it using model runner v2's ModelState abstraction and reused vLLM's speculative decoding to cleanly demonstrate the flexibility of model runner v2 and how future dLLMs may be supported."
read_time_minutes: 6
tags:
  - model
  - ecosystem
  - inference
---
# DiffusionGemma in vLLM

Google’s DiffusionGemma is a 26B-parameter discrete diffusion language model built on the Gemma4 backbone, and the first dLLM supported in vLLM. Integrating DiffusionGemma into vLLM required supporting a fundamentally different decoding pattern. dLLMs do not fit cleanly into the standard autoregressive serving path: they require bidirectional attention, iterative refinement, block-based generation, and custom sampling behavior at each denoising step.

We integrated DiffusionGemma into vLLM using [model runner v2's](https://vllm.ai/blog/2026-03-24-mrv2) new ModelState abstraction, which allows models to define their custom input preparation and provides hooks for managing per-request model-specific state. The result matches the accuracy of the Hugging Face reference implementation while enabling efficient batched serving.

Unlike standard autoregressive transformers, which generate text one token at a time from left to right, diffusion language models generate tokens by iteratively denoising a fixed-length canvas. This allows the model to refine multiple tokens in parallel across several denoising steps, effectively trading memory bandwidth pressure for additional compute — a particularly attractive tradeoff at low batch sizes, where spare compute is plentiful and memory bandwidth is the bottleneck. Generating many tokens per forward pass can translate into very low latency responses. DiffusionGemma specifically denoises a canvas of 256 tokens at a time.

<figure style="text-align: center;">
  <img src="/assets/figures/2026-06-10-diffusion-gemma/ar-vs-diffusion.svg" alt="Autoregressive vs. block diffusion" />
  <figcaption>Autoregressive vs. block diffusion decoding.</figcaption>
</figure>

## DiffusionGemma Architecture and Sampling Loop

DiffusionGemma is built on a standard Gemma4 backbone, but runs it in two modes that share the same weights — one set of layers, used two ways:

- **Encoder mode** uses *causal* attention and writes to the KV cache. It runs twice per block: once to prefill the prompt, and once to "commit" a finished block.  
- **Decoder mode** uses *bidirectional* attention and only reads the KV cache. This is the denoising mode — every position in the canvas can attend to every other position, which is what lets the model refine the whole block at once.

Because the encoder uses ordinary causal attention and the committed KV is written exactly as it would be for an autoregressive model, vLLM's automatic prefix caching works out of the box: shared prompt prefixes are reused across requests with no diffusion-specific changes.

The loop for a single 256-token block works as follows. After the prompt is prefilled (encoder), the canvas is initialized to random tokens and its state is then set to denoising. Each denoising step runs the backbone in decoder mode over the full canvas, samples a candidate token at every position, and decides which positions to keep. Once the block stops changing, the state is set back to encoding and a final encoder pass commits it — writing its KV and emitting the 256 tokens — and the next block starts from a fresh random canvas.

<figure style="text-align: center;">
  <img src="/assets/figures/2026-06-10-diffusion-gemma/sampling-loop-horizontal.svg" alt="DiffusionGemma block sampling loop" />
  <figcaption>DiffusionGemma's per-block sampling loop.</figcaption>
</figure>

Within a block all 256 positions denoise in parallel; across blocks, generation is still left-to-right, since each new block conditions on all previously committed tokens.

### Entropy-bound denoising

Every denoise step re-samples *all* canvas positions, but only the positions the model is confident about are kept; the rest are discarded and replaced with fresh random tokens for the next step. Confidence is measured by the entropy of each position's predicted distribution — low entropy means the model has largely made up its mind.

DiffusionGemma uses an **entropy-bound** rule to decide how many positions to accept: it walks positions from most confident to least, accepting tokens until their accumulated entropy exceeds a fixed budget. Early on the model is unsure about almost everything, so only a few positions lock in. As those anchors propagate context to their neighbors, the distributions sharpen, more positions fall under the budget, and the block snaps into focus over a handful of steps.

<figure style="text-align: center;">
  <img src="/assets/figures/2026-06-10-diffusion-gemma/denoising-grid.svg" alt="Block denoising in context" />
  <figcaption>Entropy-bound denoising over several steps.</figcaption>
</figure>

A canvas is considered **converged** once its best-guess (argmax) prediction stops changing for a couple of consecutive steps **and** its mean per-token entropy falls below a confidence threshold — or it hits a hard denoising-step limit. At that point the committed tokens are that clean argmax prediction, not the noisy sampled canvas carried between steps.

### Self-conditioning

To make the denoising loop more stable and converge faster, DiffusionGemma uses **self-conditioning**: between steps, the model is conditioned on its *own previous prediction*. Instead of feeding back hard tokens, it feeds back the full softmax distribution from the previous step, converts it into a probability-weighted average of token embeddings, and adds it — through a small gated MLP — onto the canvas embeddings before the next pass.

<figure style="text-align: center;">
  <img src="/assets/figures/2026-06-10-diffusion-gemma/self-conditioning.svg" alt="Self-conditioning" />
  <figcaption>Self-conditioning feedback path.</figcaption>
</figure>

This gives each step a memory of what the model believed last time, so even positions that were renoised to random tokens carry forward information from the previous step rather than having to start from scratch. Self-conditioning is active only in decoder/denoise mode — on the encoder prefill and commit passes the feedback is zeroed, so those passes see plain token embeddings.

## Implementation in vLLM

### Reusing the Speculative Decoding Data Path

vLLM's engine already has a very mature and stable speculative decoding path. Inspired by [RFC \#36155](https://github.com/vllm-project/vllm/issues/36155), we reuse this path to implement DiffusionGemma. Reusing the speculative decoding path for diffusion LLMs in vLLM is a natural fit since on each step the current canvas can be viewed as a large set of draft tokens that will be either fully rejected or fully accepted. This leads to very minimal changes to core vLLM components like the scheduler and model runner. The notable exception is that with speculative decode we always sample one extra token (typically referred to as the bonus token in speculative decoding literature), support for sampling 0 tokens was added and is controlled by the ModelState.

Concretely, diffusion plugs into the existing stack as follows — the scheduler, model runner, and Gemma4 backbone are reused unchanged, and only the ModelState and sampler are diffusion-specific:

<figure style="text-align: center;">
  <img src="/assets/figures/2026-06-10-diffusion-gemma/stack.svg" alt="How DiffusionGemma plugs into vLLM's speculative-decoding stack" />
  <figcaption>DiffusionGemma in vLLM's software abstractions.</figcaption>
</figure>

### The ModelState Interface

Before ModelState, adding a non-autoregressive model to V1 would have required forking the model runner and threading diffusion-specific state through input preparation, attention metadata, and sampling. ModelState avoids this by defining a set of hooks that the runner calls at each stage of the forward loop:

| Hook | DiffusionGemma Uses It To... |
| :---- | :---- |
| `prepare_inputs()` | Embed canvas tokens and apply self-conditioning |
| `prepare_attn()` | Set per-request causal (encoder) vs. bidirectional (denoise) attention |
| `custom_sampler()` | Replace the default sampler with `DiffusionSampler` |
| `add_request()` / `remove_request()` | Initialize and tear down per-request diffusion state (e.g. the canvas and self-conditioning probs) |

Models self-register their ModelState by defining `get_model_state_cls()` on the model class. The model runner stays generic. At each step, it calls `prepare_attn(...)` to build metadata, merges `prepare_inputs(...)` into the forward kwargs, and delegates sampling to whatever sampler `custom_sampler()->DiffusionSampler` installed.

This means adding a new block diffusion model requires implementing a ModelState and a one-line registration on the model class and no changes to the runner, scheduler, or any shared infrastructure. We believe this can act as a blueprint for cleanly adding diffusion language models to vLLM in the future.

### Putting It Together: DiffusionGemmaModelState and DiffusionSampler

`DiffusionGemmaModelState` is the ModelState implementation for `DiffusionGemma`. It holds the per-request state (mostly related to the diffusion loop): a phase flag for whether the request is committing or denoising, the current `canvas`, a history used for convergence checks, self-conditioning probabilities, and more. This state lives in pre-allocated GPU tensors and is updated in place. `DiffusionGemmaModelState.prepare_inputs()` embeds the canvas tokens and applies self-conditioning: it takes the softmax distribution from the previous denoise step (from the internal per-request state), computes a probability-weighted average of the token embeddings, and feeds that through a gated MLP so the model can see its own previous prediction. `prepare_attn()` builds the attention metadata, using the phase flag to decide whether attention should be causal (commit phase / encoder) or bidirectional (denoise phase / decoder). Since a single batch can hold a mix of prefill, denoise, and commit requests, and the per-request causal flag is set asynchronously on the GPU, we had to make some attention-kernel modifications that we discuss in a later section.

`DiffusionSampler` takes the place of vLLM's usual `(Sampler, RejectionSampler)` pair and is responsible for initializing and resetting the canvas and per-request diffusion state during phase changes. The per-step work is a single `@torch.compile`d function, `_compiled_sample_step`, vectorized over all in-flight decode requests, covering three cases:

- **Prefill**: initialize the canvas to random tokens and return `num_sampled = 0`.  
- **Denoise**: temperature-scale the logits, draw a candidate token at each canvas position with the Gumbel-max trick (`argmax(logits/T + gumbel_noise)`), accept the most confident positions up to the entropy bound, and renoise the rest to random tokens. The step also records the argmax canvas and checks for convergence: the argmax canvas has been stable for the configured number of steps and mean entropy is below threshold, or the step cap is reached.  
- **Commit**: emit the clean `argmax_canvas` (`num_sampled = 256`), reinitialize the canvas for the next block, and reset the per-request state.

During denoise the sampler reports `num_sampled = 0` and `num_rejected = query_len`, so the KV cache position does not move; only a commit advances it. Marking every canvas position as rejected tells the scheduler to keep the sequence where it is and reschedule the same block on the next step, which keeps the whole denoising loop inside the existing speculative-decoding accounting without any scheduler changes.

### Dynamic Per-sequence Causal Attention

As described above, DiffusionGemma operates in two modes: an **encoder** mode that uses causal attention and a **decoder** mode that uses bidirectional attention. Until now, causality was a single batch-wide property – every request in a forward pass shared the same mask type. Typical decoder models use only causal attention, whereas encoder-decoder models such as Whisper use only bidirectional attention in their encoder layers. For DiffusionGemma, however, requests alternate between these modes as the prompt is prefilled and then canvases are iteratively denoised and accepted. To minimize latency, vLLM mixes requests at different stages in the batch during each forward pass. Therefore, we have implemented **dynamic per-sequence causal attention**, which adapts the attention mask to each request’s causality. This situation is depicted below: here, we show a batch with three requests, each at a different stage.

- Request 0 is a prefill of length 6, so it uses causal attention (“encoder” pass), where entries above the diagonal are masked off – each query token only attends to keys from tokens up to and including itself. We also note that attention is computed in tiles (shaped 2x2 in this example, though these are much larger and have hardware-dependent tuning in practice), and tiles containing only masked entries are skipped entirely, saving both compute and the memory bandwidth of loading their K/V tiles from HBM.  
- Request 1 has already completed its prefill of length 6, and is now generating new tokens in a decoder mode. Within the canvas of size 4, all queries attend to all keys in the canvas using bidirectional attention. They also attend to all keys in the context. No entries are masked off and no blocks are skipped.  
- Finally, request 2 has completed its denoising steps, and its canvas is ready to be accepted. We run the encoder pass one last time, using causal attention and filling the KV cache with the entries from the newly accepted tokens. Again, all queries also attend to the cached keys.

<figure style="text-align: center;">
  <img src="/assets/figures/2026-06-10-diffusion-gemma/per_seq_causal_attention.svg" alt="Dynamic per-sequence causal attention" />
  <figcaption>Dynamic per-sequence causal attention.</figcaption>
</figure>

We support this dynamic causal attention in two attention backends: Triton Attention (`TRITON_ATTN`) and FlashAttention 4 (`FLASH_ATTN`). In both of these backends, the single boolean argument `causal` is replaced by a tensor indicating the causality of each request. The mask is updated appropriately, and the tiling behavior is preserved.

### Sliding window attention

Finally, some layers of DiffusionGemma use sliding window attention. For tokens in the canvas, sliding window attention must also become symmetric: for a window size `W`, instead of attending only to itself and the `W` tokens before it, a canvas token also attends to the `W` tokens after it, for a total window size of `2*W + 1`. We depict this below:

<figure style="text-align: center;">
  <img src="/assets/figures/2026-06-10-diffusion-gemma/per_seq_sliding_window.svg" alt="Per-sequence sliding window attention" />
  <figcaption>Dynamic causal sliding-window attention.</figcaption>
</figure>

As before, the same three requests are shown on a sliding-window layer with `W=2`. Requests 0 and 2 (prefill and acceptance) keep the one-sided causal window — each query attends to itself and the `W` keys before it, narrowing attention to a band along the diagonal — while the denoising canvas of Request 1 uses the symmetric window, attending to the `W` keys on either side and thus only to the context tokens that fall within it.

Supporting this in both backends required only modifying the window's right-hand bound for bidirectional requests: a causal request keeps a left-only window, while a bidirectional request uses a symmetric window of `W` on each side.

## Quantized Checkpoint Support

Quantized checkpoints of the DiffusionGemma model were created using [LLM Compressor](https://github.com/vllm-project/llm-compressor) and saved in the [compressed-tensors](https://github.com/vllm-project/compressed-tensors) format. These include an FP8 model with quantized weights and fully dynamic activations, as well as an NVFP4 model with both weights and activations quantized to the NVFP4 format.

The quantized checkpoints can be found on the RedHatAI hub:

1. [https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4)   
2. [https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic)

To validate the accuracy of the models, preliminary evaluations were performed both with and without thinking enabled, on the AIME 2025, GPQA Diamond, and GSM8k benchmarks using vLLM. See model cards for evaluations and recovery scores.

## Results

### Low-Latency Performance

DiffusionGemma’s architecture enables extremely low-latency inference, making it well suited for interactive applications. To evaluate the performance of our implementation in this setting, we benchmarked vLLM at batch size 1 on a single H100 using the built-in `vllm bench serve`. For consistency in the benchmark, we employ a fixed count of 16 denoising steps for each canvas, with no confidence-based committing (`confidence_threshold=0.0`). This number was selected because it is representative of typical workloads from popular evaluation datasets in our experimentation. We also disable prefix caching, though cache hits are negligible for random data. At FP8 quantization, we observe that vLLM achieves **over 1700 tokens per second** of output token generation in this configuration. Our complete benchmarking command is:

(UPDATE MODEL NAME)

```bash
vllm serve gg-hf-st/test-checkpoint-26B-RC1 \
  --max-num-seqs 4 \
  --max_model_len 8192 \
  --diffusion-config '{"canvas_length": 256, "max_denoising_steps": 16}' \
  --hf_overrides '{"diffusion_sampler": "entropy_bound", "diffusion_entropy_bound": 0.1, "diffusion_confidence_threshold": 0.0}' \
  --trust-remote-code \
  --quantization "fp8" \
  --no-enable-prefix-caching \
  --attention-backend FLASH_ATTN
```

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8000 \
  --model "gg-hf-st/test-checkpoint-26B-RC1" \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 512 \
  --ignore-eos \
  --num-prompts 500 \
  --max-concurrency 1
```

Since our output length is 512 and our canvas size is 256, the model will output 2 canvases for each request. The ITL measures the time between the completion of the first and second canvas. Our benchmark yields:

```
============ Serving Benchmark Result ============
Successful requests:                     500
Failed requests:                         0
Maximum request concurrency:             1
Benchmark duration (s):                  285.92
Total input tokens:                      2048000
Total generated tokens:                  256000
Request throughput (req/s):              1.75
Output token throughput (tok/s):         895.36
Peak output token throughput (tok/s):    5.00
Peak concurrent requests:                3.00
Total token throughput (tok/s):          8058.25
---------------Time to First Token----------------
Mean TTFT (ms):                          425.87
Median TTFT (ms):                        321.66
P99 TTFT (ms):                           603.35
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.29
Median TPOT (ms):                        0.40
P99 TPOT (ms):                           0.59
---------------Inter-token Latency----------------
Mean ITL (ms):                           145.74
Median ITL (ms):                         203.88
P99 ITL (ms):                            299.14
----------------Diffusion Decoding----------------
Committed throughput (tok/s):            895.36
Denoising steps per canvas:              15.98
Committed per denoising step:            16.02
Committed tokens:                        256000
Denoising steps:                         15978
Canvas positions evaluated:              4346368
==================================================
```

With each 256-token canvas being generated in an average of 145.74ms, this yields a decode speed of **1756.55 tok/s**.

## Acknowledgements

Thanks to everyone who contributed to bringing DiffusionGemma to vLLM. This was a close collaboration between Google DeepMind, the vLLM team, and Red Hat.

- **Google DeepMind:** Martin Kukla, João Gante, Luciano Martins
- **vLLM:** Lucas Wilkinson (RedHat), Matthew Bonanni (RedHat), Nicolò Lucchesi (RedHat), Dipika Sikka (RedHat), Doug Smith (RedHat), Edward Arthur Quarm Jnr (RedHat), Alon Kellner (RedHat), Nick Hill (Inferact)
- **NVIDIA:** Dimitrios Bariamis, Alec Kohlhoff, Porras Huang, Eugene Rakhmatulin

