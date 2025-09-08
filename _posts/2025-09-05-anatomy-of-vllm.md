---
layout: post
title: "Inside vLLM: Anatomy of a High-Throughput LLM Inference System"
author: "Aleksa Gordic"
image: /assets/logos/vllm-logo-text-light.png
---

> [!NOTE]
> Originally posted on [Aleksa Gordic's website](https://www.aleksagordic.com/blog/vllm).

### From paged attention, continuous batching, prefix caching, specdec, etc. to multi-GPU, multi-node dynamic serving at scale

In this post, I'll gradually introduce all of the core system components and advanced features that make up a modern high-throughput LLM inference system. In particular I'll be doing a breakdown of how vLLM [[1]](#ref-1) works.

This post is the first in a series. It starts broad and then layers in detail (following an inverse-pyramid approach) so you can form an accurate high-level mental model of the complete system without drowning in minutiae.

Later posts will dive into specific subsystems.

This post is structured into five parts:

1. [LLM engine & engine core](#llm-engine--engine-core): fundamentals of vLLM (scheduling, paged attention, continuous batching, etc.)
2. [Advanced features](#advanced-features--extending-the-core-engine-logic): chunked prefill, prefix caching, guided & speculative decoding, disaggregated P/D
3. [Scaling up](#from-uniprocexecutor-to-multiprocexecutor): from single-GPU to multi-GPU execution
4. [Serving layer](#distributed-system-serving-vllm): distributed / concurrent web scaffolding
5. [Benchmarks and auto-tuning](#benchmarks-and-auto-tuning---latency-vs-throughput): measuring latency and throughput

> [!NOTE]
> * Analysis is based on [commit 42172ad](https://github.com/vllm-project/vllm/tree/42172ad) (August 9th, 2025).
> * Target audience: anyone curious about how state-of-the-art LLM engines work, as well as those interested in contributing to vLLM, SGLang, etc.
> * I'll focus on the [V1 engine](https://docs.vllm.ai/en/latest/usage/v1_guide.html). I also explored V0 (now [deprecated](https://github.com/vllm-project/vllm/issues/18571)), which was valuable for understanding how the project evolved, and many concepts still carry over.
> * The first section on LLM Engine / Engine Core might be a bit overwhelming/dry - but the rest of the blog has plenty examples and visuals. :)

## LLM Engine & Engine Core

The LLM engine is the fundamental building block of vLLM. On its own, it already enables high-throughput inference - but only in an offline setting. You can't serve it to customers over the web yet.

We'll use the following offline inference snippet as our running example (adapted from [basic.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/basic.py)).

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```

> [!NOTE]
> Environment vars:
> * VLLM_USE_V1="1" # we're using engine V1
> * VLLM_ENABLE_V1_MULTIPROCESSING="0" # we're running in a single process

This configuration is:

* offline (no web/distributed system scaffolding)
* synchronous (all execution happens in a single blocking process)
* single-GPU (no data/model/pipeline/expert parallelism; DP/TP/PP/EP = 1)
* using standard transformer [[2]](#ref-2) (supporting hybrid models like Jamba requires a more complex hybrid KV-cache memory allocator)

From here, we'll gradually build up to an online, async, multi-GPU, multi-node inference system - but still serving a standard transformer.

In this example we do two things, we:

1. Instantiate an engine
3. Call <code>generate</code> on it to sample from the given prompts

Let's start analyzing the constructor.

### LLM Engine constructor
The main components of the engine are:

* vLLM config (contains all of the knobs for configuring model, cache, parallelism, etc.)
* processor (turns raw inputs → <code>EngineCoreRequests</code> via validation, tokenization, and processing)
* engine core client (in our running example we're using <code>InprocClient</code> which is basically == <code>EngineCore</code>; we'll gradually build up to <code>DPLBAsyncMPClient</code> which allows serving at scale)
* output processor (converts raw <code>EngineCoreOutputs</code> → <code>RequestOutput</code> that the user sees)
> [!NOTE]
> With the V0 engine being deprecated, class names and details may shift. I'll emphasize the core ideas rather than exact signatures. I'll abstract away some but not all of those details.

Engine core itself is made up of several sub components:

* Model Executor (drives forward passes on the model, we're currently dealing with <code>UniProcExecutor</code> which has a single <code>Worker</code> process on a single GPU). We'll gradually build up to <code>MultiProcExecutor</code> which supports multiple GPUs
* Structured Output Manager (used for guided decoding - we'll cover this later)
* Scheduler (decides which requests go into the next engine step) - it further contains:
  <ol type="a">
  <li>policy setting - it can be either <b>FCFS</b> (first come first served) or <b>priority</b> (higher priority requests are served first)</li>
  <li><code>waiting</code> and <code>running</code> queues</li>
  <li>KV cache manager - the heart of paged attention <a href="#ref-3">[3]</a></li>

The KV-cache manager maintains a <code>free_block_queue</code> - a pool of available KV-cache blocks (often on the order of hundreds of thousands, depending on VRAM size and block size). During paged attention, the blocks serve as the indexing structure that map tokens to their computed KV cache blocks.

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/engine_constructor.png" width="100%">
</picture><br>
<b>Figure 1</b>: Core components described in this section and their relationships
</p>

> [!NOTE]
> Block size for a standard transformer layer (non-MLA [[4]](#ref-4)) is computed as follows:
> 2 (key/value) * <code>block_size</code> (default=16) * <code>num_kv_heads</code> * <code>head_size</code> * <code>dtype_num_bytes</code> (e.g. 2 for bf16)

During model executor construction, a <code>Worker</code> object is created, and three key procedures are executed. (Later, with <code>MultiProcExecutor</code>, these same procedures run independently on each worker process across different GPUs.)

1. Init device:
* Assign a CUDA device (e.g. "cuda:0") to the worker and check that the model dtype is supported (e.g. bf16)
* Verify enough VRAM is available, given the requested <code>gpu_memory_utilization</code> (e.g. 0.8 → 80% of total VRAM)
* Set up distributed settings (DP / TP / PP / EP, etc.)
* Instantiate a <code>model_runner</code> (holds the sampler, KV cache, and forward-pass buffers such as <code>input_ids</code>, <code>positions</code>, etc.)
* Instantiate an <code>InputBatch</code> object (holds CPU-side forward-pass buffers, block tables for KV-cache indexing, sampling metadata, etc.)

2. Load model:
* Instantiate the model architecture
* Load the model weights
* Call model.eval() (PyTorch's inference mode)
* Optional: call torch.compile() on the model

3. Initialize KV cache
* Get per-layer KV-cache spec. Historically this was always <code>FullAttentionSpec</code> (homogeneous transformer), but with hybrid models (sliding window, Transformer/SSM like Jamba) it became more complex (see Jenga [[5]](#ref-5))
* Run a dummy/profiling forward pass and take a GPU memory snapshot to compute how many KV cache blocks fit in available VRAM
* Allocate, reshape and bind KV cache tensors to attention layers
* Prepare attention metadata (e.g. set the backend to FlashAttention) later consumed by kernels during the fwd pass
* Unless <code>--enforce-eager</code> is provided, for each of warmup batch sizes do a dummy run and capture CUDA graphs. CUDA graphs record the whole sequence of GPU work into a DAG. Later during fwd pass we launch/replay pre-baked graphs and cut on kernel launch overhead and thus improve latency.

I've abstracted away many low-level details here — but these are the core pieces I'll introduce now, since I'll reference them repeatedly in the following sections.

Now that we have the engine initialized let's proceed to the <code>generate</code> function.

### Generate function

The first step is to validate and feed requests into the engine. For each prompt we:

1. Create a unique request ID and capture its arrival time
2. Call an input preprocessor that tokenizes the prompt and returns a dictionary containing <code>prompt</code>, <code>prompt_token_ids</code>, and a <code>type</code> (text, tokens, embeds, etc.)
3. Pack this info into an <code>EngineCoreRequest</code>, adding priority, sampling params, and other metadata
4. Pass the request into the engine core, which wraps it in a <code>Request</code> object and sets its status to <code>WAITING</code>. This request is then added to the scheduler's <code>waiting</code> queue (append if FCFS, or heap-push if priority)

At this point the engine has been fed and execution can begin. In the synchronous engine example, these initial prompts are the only ones we'll process — there's no mechanism to inject new requests mid-run. In contrast, the asynchronous engine supports this (aka <b>continuous batching</b> [[6]](#ref-6)): after each step, both new and old requests are considered.

> [!NOTE]
> Because the forward pass flattens the batch into a single sequence and custom kernels handle it efficiently, continuous batching is fundamentally supported even in the synchronous engine.

Next, as long as there are requests to process, the engine repeatedly calls its <code>step()</code> function. Each step has three stages:

1. Schedule: select which requests to run in this step (decode, and/or (chunked) prefill)
2. Forward pass: run the model and sample tokens
3. Postprocess: append sampled token IDs to each <code>Request</code>, detokenize, and check stop conditions. If a request is finished, clean up (e.g. return its KV-cache blocks to <code>free_block_queue</code>) and return the output early

> [!NOTE]
> Stop conditions are:
> * The request exceeds its length limit (<code>max_model_length</code> or its own <code>max_tokens</code>)
> * The sampled token is the EOS ID (unless <code>ignore_eos</code> is enabled -> useful for benchmarking when we want to force a generation of a certain number of out tokens)
> * The sampled token matches any of the <code>stop_token_ids</code> specified in the sampling parameters
> * Stop strings are present in the output - we truncate the output until the first stop string appearance and abort the request in the engine (note that <code>stop_token_ids</code> will be present in the output but stop strings will not).

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/engine_loop.png" width="100%">
</picture><br>
<b>Figure 2</b>: Engine loop
</p>

> [!NOTE]
> In streaming mode, we would send intermediate tokens as they are generated, but we'll ignore that for now.

Next, we'll examine scheduling in more detail.

### Scheduler

There are two main types of workloads an inference engine handles:

1. <b>Prefill</b> requests — a forward pass over all prompt tokens. These are usually <b>compute-bound</b> (threshold depends on hardware and prompt length). At the end, we sample a single token from the probability distribution of the final token's position.
2. <b>Decode</b> requests — a forward pass over just the most recent token. All earlier KV vectors are already cached. These are <b>memory-bandwidth-bound</b>, since we still need to load all LLM weights (and KV caches) just to compute one token.

> [!NOTE]
> In the [benchmarking section](#benchmarks-and-auto-tuning---latency-vs-throughput) we'll analyze the so-called roofline model of GPU perf. That will go into more detail behind prefill/decode perf profiles.

The V1 scheduler can mix both types of requests in the same step, thanks to smarter design choices. In contrast, the V0 engine could only process either prefill or decode at once.

The scheduler prioritizes decode requests — i.e. those already in the <code>running</code> queue. For each such request it:

1. Computes the number of new tokens to generate (not always 1, due to speculative decoding and async scheduling — more on that later).
2. Calls the KV-cache manager's <code>allocate_slots</code> function (details below).
3. Updates the token budget by subtracting the number of tokens from step 1.

After that, it processes prefill requests from the <code>waiting</code> queue, it:

1. Retrieves the number of computed blocks (returns 0 if prefix caching is disabled — we'll cover that later).
2. Calls the KV-cache manager's <code>allocate_slots</code> function.
3. Pops the request from waiting and moves it to running, setting its status to <code>RUNNING</code>.
4. Updates the token budget.

Let's now look at what <code>allocate_slots</code> does, it:

1. <b>Computes number of blocks</b> — determines how many new KV-cache blocks (<code>n</code>) must be allocated. Each block stores 16 tokens by default. For example, if a prefill request has 17 new tokens, we need <code>ceil(17/16) = 2</code> blocks.
2. <b>Checks availability</b> — if there aren't enough blocks in the manager's pool, exit early. Depending on whether it's a decode or prefill request, the engine may attempt recompute preemption (swap preemption was supported in V0) by evicting low-priority requests (calling <code>kv_cache_manager.free</code> which returns KV blocks to block pool), or it might skip scheduling and continue execution.
3. <b>Allocates blocks</b> — via the KV-cache manager's coordinator, fetches the first <code>n</code> blocks from the block pool (the <code>free_block_queue</code> doubly linked list mentioned earlier). Stores to <code>req_to_blocks</code>, the dictionary mapping each <code>request_id</code> to its list of KV-cache blocks.

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/kv_cache_blocks.png" width="100%">
</picture><br>
<b>Figure 3</b>: list of KV cache blocks
</p>

We're finally ready to do a forward pass!

### Run forward pass

We call model executor's <code>execute_model</code>, which delegates to the <code>Worker</code>, which in turn delegates to the model runner.

Here are the main steps:

1. <b>Update states</b> — prune finished requests from <code>input_batch</code>; update misc fwd pass related metadata (e.g., KV cache blocks per request that will be used to index into paged KV cache memory).
2. <b>Prepare inputs</b> — copy buffers from CPU→GPU; compute positions; build <code>slot_mapping</code> (more on that in example); construct attention metadata.
3. <b>Forward pass</b> — run the model with custom paged attn kernels. All sequences are flattened and concatenated into one long "super sequence". Position indices and attention masks ensure each sequence only attends to its own tokens, which enables continuous batching without right-padding.
4. <b>Gather last-token states</b> — extract hidden states for each sequence's final position and compute logits.
5. <b>Sample</b> — sample tokens from computed logits as dictated by the sampling config (greedy, temperature, top-p, top-k, etc.).

Forward-pass step itself has two execution modes:

1. <b>Eager mode</b> — run the standard PyTorch forward pass when eager execution is enabled.
2. <b>"Captured" mode</b> — execute/replay a pre-captured CUDA Graph when eager is not enforced (remember we captured these during engine construction in the initialize KV cache procedure).

Here is a concrete example that should make continuous batching and paged attention clear:

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/fwd_pass.png" width="100%">
</picture><br>
<b>Figure 4</b>: Forward pass: continuous batching and paged attention
</p>

## Advanced Features — extending the core engine logic

With the basic engine flow in place, we can now look at the advanced features.

We've already discussed preemption, paged attention, and continuous batching.

Next, we'll dive into:

1. Chunked prefill
2. Prefix caching
3. Guided decoding (through grammar-constrained finite-state machines)
4. Speculative decoding
5. Disaggregated P/D (prefill/decoding)

### Chunked prefill

Chunked prefill is a technique for handling long prompts by splitting their prefill step into smaller chunks. Without it, we could end up with a single very long request monopolizing one engine step disallowing other prefill requests to run. That would postpone all other requests and increase their latency.

For example, let each chunk contain <code>n</code> (=8) tokens, labeled with lowercase letters separated by "-". A long prompt <code>P</code> could look like <code>x-y-z</code>, where <code>z</code> is an incomplete chunk (e.g. 2 toks). Executing the full prefill for <code>P</code> would then take ≥ 3 engine steps (> can happen if it's not scheduled for execution in one of the steps), and only in the last chunked prefill step would we sample one new token.

Here is that same example visually:

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/chunked_pt1.png" width="100%">
</picture><br>
<b>Figure 5</b>: Chunked prefill
</p>

Implementation is straightforward: cap the number of new tokens per step. If the requested number exceeds <code>long_prefill_token_threshold</code>, reset it to exactly that value. The underlying indexing logic (described earlier) takes care of the rest.

In vLLM V1, you enable chunked prefill by setting <code>long_prefill_token_threshold</code> to a positive integer. (Technically, it can happen irrespective of this, if the prompt length exceeds the token budget we truncate it and run a chunked prefill.)

### Prefix Caching

To explain how prefix caching works, let's take the original code example and tweak it a bit:

```python
from vllm import LLM, SamplingParams

long_prefix = "<a piece of text that is encoded into more than block_size tokens>"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(long_prefix + prompts[0], sampling_params)
    outputs = llm.generate(long_prefix + prompts[1], sampling_params)

if __name__ == "__main__":
    main()
```

Prefix caching avoids recomputing tokens that multiple prompts share at the beginning - hence <b>prefix</b>.

The crucial piece is the <code>long_prefix</code>: it's defined as any prefix longer than a KV-cache block (16 tokens by default). To simplify our example let's say <code>long_prefix</code> has exactly length <code>n x block_size</code> (where <code>n ≥ 1</code>).

> [!NOTE]
> i.e. it perfectly aligns with block boundary - otherwise we'd have to recompute <code>long_prefix_len % block_size</code> tokens as we can't cache incomplete blocks.

Without prefix caching, each time we process a new request with the same <code>long_prefix</code>, we'd recompute all <code>n x block_size</code> tokens.

With prefix caching, those tokens are computed once (their KVs stored in KV cache paged memory) and then reused, so only the new prompt tokens need processing. This speeds up prefill requests (though it doesn't help with decode).

How does this work in vLLM?

During the first <code>generate</code> call, in the scheduling stage, inside <code>kv_cache_manager.get_computed_blocks</code>, the engine invokes <code>hash_request_tokens</code>:

1. This function splits the <code>long_prefix + prompts[0]</code> into 16-token chunks.
2. For each complete chunk, it computes a hash (using either the built-in hash or SHA-256, which is slower but has fewer collisions). The hash combines the previous block's hash, the current tokens, and optional metadata.
> [!NOTE]
> optional metadata includes: MM hash, LoRA ID, cache salt (injected into hash of the first block ensures only requests with this cache salt can reuse blocks).
3. Each result is stored as a <code>BlockHash</code> object containing both the hash and its token IDs. We return a list of block hashes.

The list is stored in <code>self.req_to_block_hashes[request_id]</code>.

Next, the engine calls <code>find_longest_cache_hit</code> to check if any of these hashes already exist in <code>cached_block_hash_to_block</code>. On the first request, no hits are found.

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/prefix_pt1.png" width="100%">
</picture><br>
<b>Figure 6</b>: Prefix caching - hash function
</p>

Then we call <code>allocate_slots</code> which calls <code>coordinator.cache_blocks</code>, which associates the new <code>BlockHash</code> entries with allocated KV blocks and records them in <code>cached_block_hash_to_block</code>.

Afterwards, the forward pass will populate KVs in paged KV cache memory corresponding to KV cache blocks that we allocated above.

> [!NOTE]
> After many engine steps it'll allocate more KV cache blocks but it doesn't matter for our example because the prefix has diverged immediately after <code>long_prefix</code>.

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/prefix_pt2.png" width="100%">
</picture><br>
<b>Figure 7</b>: Prefix caching - populate KVs in paged memory
</p>

On a second <code>generate</code> call with the same prefix, steps 1-3 repeat, but now <code>find_longest_cache_hit</code> finds matches for all <code>n</code> blocks (via linear search). The engine can reuse those KV blocks directly.

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/prefix_pt3.png" width="100%">
</picture><br>
<b>Figure 8</b>: Prefix caching - reuse KVs
</p>

If the original request were still alive, the reference count for those blocks would increment (e.g. to 2). In this example, the first request has already completed, so the blocks were freed back to the pool and their reference counts set back to 0. Because we were able to retrieve them from <code>cached_block_hash_to_block</code> we know they're valid (the logic of the KV cache manager is setup in such a way), so we just remove them from <code>free_block_queue</code> again.

> [!NOTE] Advanced note:
> KV-cache blocks become invalid only when they're about to be reallocated from the <code>free_block_queue</code> (which pops from the left) and we discover the block still has an associated hash and is present in <code>cached_block_hash_to_block</code>. At that moment, we clear the block's hash and remove its entry from <code>cached_block_hash_to_block</code>, ensuring it can't be reused via prefix caching (at least not for that old prefix).

And that's the gist of prefix caching: don't recompute prefixes you've already seen — just reuse their KV cache!

If you understood this example you also understood how paged attention works.

Prefix caching is enabled by default. To disable it: <code>enable_prefix_caching = False</code>.

### Guided Decoding (FSM)

Guided decoding is a technique where, at each decoding step, the logits are constrained by a grammar-based finite state machine. This ensures that only tokens allowed by the grammar can be sampled.

It's a powerful setup: you can enforce anything from regular grammars (Chomsky type-3, e.g. arbitrary regex patterns) all the way up to context-free grammars (type-2, which cover most programming languages).

To make this less abstract, let's start with the simplest possible example, building on our earlier code:

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

prompts = [
    "This sucks",
    "The weather is beautiful",
]

guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```

In the toy example I gave (assume character-level tokenization): at prefill, the FSM masks logits so only "P" or "N" are viable. If "P" is sampled, the FSM moves to the "Positive" branch; next step only "o" is allowed, and so on.

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/fsm.png" width="100%">
</picture><br>
<b>Figure 9</b>: Toy example FSM
</p>

How this works in vLLM:

1. At LLM engine construction, a <code>StructuredOutputManager</code> is created; it has access to the tokenizer and maintains a <code>_grammar_bitmask</code> tensor.
2. When adding a request, its status is set to <code>WAITING_FOR_FSM</code> and <code>grammar_init</code> selects the backend compiler (e.g., <code>xgrammar</code> [[7]](#ref-7); note that backends are 3rd party code).
3. The grammar for this request is compiled asynchronously.
4. During scheduling, if the async compile has completed, the status switches to <code>WAITING</code> and <code>request_id</code> is added to <code>structured_output_request_ids</code>; otherwise it's placed in <code>skipped_waiting_requests</code> to retry on next engine step.
5. After the scheduling loop (still inside scheduling), if there are FSM requests, the <code>StructuredOutputManager</code> asks the backend to prepare/update <code>_grammar_bitmask</code>.
6. After the forward pass produces logits, xgr_torch_compile's function expands the bitmask to vocab size (32x expansion ratio because we use 32 bit integers) and masks disallowed logits to –∞.
7. After sampling the next token, the request's FSM is advanced via <code>accept_tokens</code>. Visually we move to the next state on the FSM diagram.

Step 6 deserves further clarification.

If <code>vocab_size = 32</code>, <code>_grammar_bitmask</code> is a single integer; its binary representation encodes which tokens are allowed ("1") vs disallowed ("0"). For example, "101…001" expands to a length-32 array <code>[1, 0, 1, ..., 0, 0, 1]</code>; positions with 0 get logits set to –∞. For larger vocabularies, multiple 32-bit words are used and expanded/concatenated accordingly. The backend (e.g., <code>xgrammar</code>) is responsible for producing these bit patterns using the current FSM state.

> [!NOTE]
> Most of the complexity here is hidden in the 3rd party libs like xgrammar.

Here is an even simpler example with vocab_size = 8 and 8-bit integers (for those of you who like my visuals):

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/fsm2.png" width="100%">
</picture><br>
<b>Figure 10</b>: Toy example
</p>

You can enable this in vLLM by passing in a desired <code>guided_decoding</code> config.

### Speculative Decoding

In autoregressive generation, each new token requires a forward pass of the large LM. This is expensive — every step reloads and applies all model weights just to compute a single token! (assuming batch size == 1, in general it's <code>B</code>)

Speculative decoding [[8]](#ref-8) speeds this up by introducing a smaller draft LM. The draft proposes <code>k</code> tokens cheaply. But we don't ultimately want to sample from the smaller model — it's only there to guess candidate continuations. The large model still decides what's valid.

Here are the steps:

1. <b>Draft</b>: run the small model on the current context and propose <code>k</code> tokens
2. <b>Verify</b>: run the large model once on context + <code>k</code> draft tokens. This produces probabilities for those <code>k</code> positions plus one extra (so we get <code>k+1</code> candidates)
3. <b>Accept/reject</b>: going from left to right over the <code>k</code> draft tokens:
  <ul>
      <li>If the large model's probability for the draft token ≥ the draft's probability, accept it</li>
      <li>Otherwise, accept it with probability <code>p_large(token)/p_draft(token)</code></li>
      <li>Stop at the first rejection, or accept all <code>k</code> draft tokens</li>
      <ul>
        <li>If all <code>k</code> draft tokens are accepted, also sample the extra <code>(k+1)</code>-th token "for free" from the large model (we already computed that distribution)</li>
        <li>If there was a rejection create a new rebalanced distribution at that position (<code>p_large - p_draft</code>, clamp min at 0, normalize to sum to 1) and sample the last token from it</li>
      </ul>
  </ul>


<b>Why this works</b>: Although we use the small model to propose candidates, the accept/reject rule guarantees that in expectation the sequence is distributed exactly as if we had sampled token by token from the large model. This means speculative decoding is statistically equivalent to standard autoregressive decoding — but potentially much faster, since a single large-model pass can yield up to <code>k+1</code> tokens.

> [!NOTE]
> I recommend looking at [gpt-fast](https://github.com/meta-pytorch/gpt-fast) for a simple implementation, and the [original paper](https://arxiv.org/abs/2302.01318) for the math details and the proof of equivalence to sampling from the full model.

vLLM V1 does not support the LLM draft model method, instead it implements faster—but less accurate—proposal schemes: n-gram, EAGLE [[9]](#ref-9), and Medusa [[10]](#ref-10).

One-liners on each:

* <b>n-gram</b>: take the last <code>prompt_lookup_max</code> tokens; find a prior match in the sequence; if found, propose the <code>k</code> tokens that followed that match; otherwise decrement the window and retry down to <code>prompt_lookup_min</code>

> [!NOTE]
> The current implementation returns <code>k</code> tokens after the first match. It feels more natural to introduce a recency bias and reverse the search direction? (i.e. last match)

* <b>Eagle</b>: perform "model surgery" on the large LM—keep embeddings and LM head, replace the transformer stack with a lightweight MLP; fine-tune that as a cheap draft

* <b>Medusa</b>: train auxiliary linear heads on top (embeddings before LM head) of the large model to predict the next <code>k</code> tokens in parallel; use these heads to propose tokens more efficiently than running a separate small LM

Here's how to invoke speculative decoding in vLLM using <code>ngram</code> as the draft method:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

speculative_config={
    "method": "ngram",
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 3,
    "num_speculative_tokens": 3,
}

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", speculative_config=speculative_config)

    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```

How does this work in vLLM?

<b>Setup (during engine construction):</b>


1. Init device: create a <code>drafter</code> (draft model, e.g., <code>NgramProposer</code>) and a <code>rejection_sampler</code> (parts of it are written in Triton).
2. Load model: load draft model weights (no-op for n-gram).

<b>After that in the <code>generate</code> function</b> (assume we get a brand new request):

1. Run the regular prefill step with the large model.
2. After the forward pass and standard sampling, call <code>propose_draft_token_ids(k)</code> to sample <code>k</code> draft tokens from the draft model.
3. Store these in <code>request.spec_token_ids</code> (update the request metadata).
4. On the next engine step, when the request is in the running queue, add <code>len(request.spec_token_ids)</code> to the "new tokens" count so <code>allocate_slots</code> reserves sufficient KV blocks for the fwd pass.
5. Copy <code>spec_token_ids</code> into <code>input_batch.token_ids_cpu</code> to form (context + draft) tokens.
6. Compute metadata via <code>_calc_spec_decode_metadata</code> (this copies over tokens from <code>input_batch.token_ids_cpu</code>, prepares logits, etc.), then run a large-model forward pass over the draft tokens.
7. Instead of regular sampling from logits, use the <code>rejection_sampler</code> to accept/reject left-to-right and produce <code>output_token_ids</code>.
8. Repeat steps 2-7 until a stop condition is met.

The best way to internalize this is to fire up your debugger and step through the codebase, but this section hopefully gives you a taste for it. This as well:

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/specdec_pt1.png" width="100%">
</picture>
</p>

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/specdec_pt2.png" width="100%">
</picture><br>
<b>Figure 11</b>: Speculative decoding
</p>

### Disaggregated P/D

I've already previously hinted at the motivation behind disaggregated P/D (prefill/decode).

Prefill and decode have very different performance profiles (compute-bound vs. memory-bandwidth-bound), so separating their execution is a sensible design. It gives tighter control over latency — both <code>TFTT</code> (time-to-first-token) and <code>ITL</code> (inter-token latency) — more on this in the [benchmarking](#benchmarks-and-auto-tuning---latency-vs-throughput) section.

In practice, we run <code>N</code> vLLM prefill instances and <code>M</code> vLLM decode instances, autoscaling them based on the live request mix. Prefill workers write KV to a dedicated KV-cache service; decode workers read from it. This isolates long, bursty prefill from steady, latency-sensitive decode.

How does this work in vLLM?

For clarity, the example below relies on <code>SharedStorageConnector</code>, a debugging connector implementation used to illustrate the mechanics.

> [!NOTE]
> Connector is vLLM's abstraction for handling the exchange of KVs between instances. Connector interface is not yet stable, there are some near-term improvements planned which will involve changes, some potentially breaking.

We launch 2 vLLM instances (GPU 0 for prefill and GPU 1 for decode), and then transfer the KV cache between them:

```python
import os
import time
from multiprocessing import Event, Process
import multiprocessing as mp

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

def run_prefill(prefill_done):
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

  ktc=KVTransferConfig(
      kv_connector="SharedStorageConnector",
      kv_role="kv_both",
      kv_connector_extra_config={"shared_storage_path": "local_storage"},
  )

  llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", kv_transfer_config=ktc)
  llm.generate(prompts, sampling_params)

  prefill_done.set()  # notify decode instance that KV cache is ready

  # To keep the prefill node running in case the decode node is not done;
  # otherwise, the script might exit prematurely, causing incomplete decoding.
  try:
      while True:
          time.sleep(1)
  except KeyboardInterrupt:
      print("Script stopped by user.")

def run_decode(prefill_done):
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"

  sampling_params = SamplingParams(temperature=0, top_p=0.95)

  ktc=KVTransferConfig(
      kv_connector="SharedStorageConnector",
      kv_role="kv_both",
      kv_connector_extra_config={"shared_storage_path": "local_storage"},
  )

  llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", kv_transfer_config=ktc)

  prefill_done.wait()  # block waiting for KV cache from prefill instance

  # Internally it'll first fetch KV cache before starting the decoding loop
  outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
  prefill_done = Event()
  prefill_process = Process(target=run_prefill, args=(prefill_done,))
  decode_process = Process(target=run_decode, args=(prefill_done,))

  prefill_process.start()
  decode_process.start()

  decode_process.join()
  prefill_process.terminate()
```

> [!NOTE]
> I've also experimented with <code>LMCache</code> [[11]](#ref-11), the fastest production-ready connector (uses NVIDIA's NIXL as the backend), but it's still at the bleeding edge and I ran into some bugs. Since much of its complexity lives in an external repo, <code>SharedStorageConnector</code> is a better choice for explanation.

These are the steps in vLLM:

1. <b>Instantiation</b> — During engine construction, connectors are created in two places:
* Inside the worker's init device procedure (under init worker distributed environment function), with role "worker".
* Inside the scheduler constructor, with role "scheduler".
2. <b>Cache lookup</b> — When the scheduler processes prefill requests from the <code>waiting</code> queue (after local prefix-cache checks), it calls connector's <code>get_num_new_matched_tokens</code>. This checks for externally cached tokens in the KV-cache server. Prefill always sees 0 here; decode may have a cache hit. The result is added to the local count before calling <code>allocate_slots</code>.
3. <b>State update</b> — The scheduler then calls <code>connector.update_state_after_alloc</code>, which records requests that had a cache (no-op for prefill).
4. <code>Build metadata object</code> — At the end of scheduling, the scheduler calls <code>meta = connector.build_connector_meta</code>:
* Prefill adds all requests with <code>is_store=True</code> (to upload KV).
* Decode adds requests with <code>is_store=False</code> (to fetch KV).
5. <b>Context manager</b> — Before the forward pass, the engine enters a KV-connector context manager:
* On enter: <code>kv_connector.start_load_kv</code> is called. For decode, this loads KV from the external server and injects it into paged memory. For prefill, it's a no-op.
* On exit: <code>kv_connector.wait_for_save</code> is called. For prefill, this blocks until KV is uploaded to the external server. For decode, it's a no-op.

Here is a visual example:

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/pd.png" width="100%">
</picture><br>
<b>Figure 12</b>: disaggregated P/D
</p>

> [!NOTE] Additional notes:
> * For <code>SharedStorageConnector</code> "external server" is just a local file system.
> * Depending on configuration, KV transfers can also be done layer-by-layer (before/after each attention layer).
> * Decode loads external KV only once, on the first step of its requests; afterwards it computes/stores locally.

## From UniprocExecutor to MultiProcExecutor

With the core techniques in place, we can now talk about scaling up.

Suppose your model weights no longer fit into a single GPU's VRAM.

The first option is to shard the model across multiple GPUs on the same node using tensor parallelism (e.g., <code>TP=8</code>). If the model still doesn't fit, the next step is pipeline parallelism across nodes.

> [!NOTE] Notes:
> * Intranode bandwidth is significantly higher than internode, which is why tensor parallelism (TP) is generally preferred over pipeline parallelism (PP). (It is also true that PP communicates less data than TP.)
> * I'm not covering expert parallelism (EP) since we're focusing on standard transformers rather than MoE, nor sequence parallelism, as TP and PP are the most commonly used in practice.

At this stage, we need multiple GPU processes (workers) and an orchestration layer to coordinate them. That's exactly what <code>MultiProcExecutor</code> provides.

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/multiprocexecutor.png" width="100%">
</picture><br>
<b>Figure 13</b>: MultiProcExecutor in a TP=8 setting (driver worker being rank 0)
</p>

How this works in vLLM:

1. <code>MultiProcExecutor</code> initializes an <code>rpc_broadcast_mq</code> message queue (implemented with shared memory under the hood).
2. The constructor loops over <code>world_size</code> (e.g. <code>TP=8 ⇒ world_size=8</code>) and spawns a daemon process for each rank via <code>WorkerProc.make_worker_process</code>.
3. For each worker, the parent first creates a reader and writer pipe.
4. The new process runs <code>WorkerProc.worker_main</code>, which instantiates a worker (going through the same "init device", "load model", etc. as in <code>UniprocExecutor</code>).
5. Each worker determines whether it is the driver (rank 0 in the TP group) or a regular worker. Every worker sets up two queues:
* <code>rpc_broadcast_mq</code> (shared with the parent) for receiving work.
* <code>worker_response_mq</code> for sending responses back.
6. During initialization, each child sends its <code>worker_response_mq</code> handle to the parent via the pipe. Once all are received, the parent unblocks — this completes coordination.
7. Workers then enter a busy loop, blocking on <code>rpc_broadcast_mq.dequeue</code>. When a work item arrives, they execute it (just like in <code>UniprocExecutor</code>, but now with TP/PP-specific partitioned work). Results are sent back through <code>worker_response_mq.enqueue</code>.
8. At runtime, when a request arrives, <code>MultiProcExecutor</code> enqueues it into <code>rpc_broadcast_mq</code> (non-blocking) for all children workers. It then waits on the designated output rank's <code>worker_response_mq.dequeue</code> to collect the final result.

From the engine's perspective, nothing has changed — all of this multiprocessing complexity is abstracted away through a call to model executor's <code>execute_model</code>.

* In the <code>UniProcExecutor</code> case: execute_model directly leads to calling execute_model on the worker
* In the <code>MultiProcExecutor</code> case: execute_model indirectly leads to calling execute_model on each worker through <code>rpc_broadcast_mq</code>

At this point, we can run models that are as large as resources allow using the same engine interface.

The next step is to scale out: enable data parallelism (<code>DP > 1</code>) replicating the model across nodes, add a lightweight DP coordination layer, introduce load balancing across replicas, and place one or more API servers in front to handle incoming traffic.

## Distributed system serving vLLM

There are many ways to set up serving infrastructure, but to stay concrete, here's one example: suppose we have two H100 nodes and want to run four vLLM engines across them.

If the model requires <code>TP=4</code>, we can configure the nodes like this.

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/server_setup.png" width="100%">
</picture><br>
<b>Figure 14</b>: server configuration with 2 8xH100 nodes (1 headless, 1 api server)
</p>

On the first node, run the engine in headless mode (no API server) with the following arguments:

```shell
vllm serve <model-name>
  --tensor-parallel-size 4
  --data-parallel-size 4
  --data-parallel-size-local 2
  --data-parallel-start-rank 0
  --data-parallel-address <master-ip>
  --data-parallel-rpc-port 13345
  --headless
```

and run that same command on the other node with few tweaks:

* no <code>--headless</code>
* modify DP start rank

```shell
vllm serve <model-name>
  --tensor-parallel-size 4
  --data-parallel-size 4
  --data-parallel-size-local 2
  --data-parallel-start-rank 2
  --data-parallel-address <master-ip>
  --data-parallel-rpc-port 13345
```

> [!NOTE]
> This assumes networking is configured so all nodes can reach the specified IP and port.

How does this work in VLLM?

### On the headless server node

On the headless node, a <code>CoreEngineProcManager</code> launches 2 processes (per <code>--data-parallel-size-local</code>) each running <code>EngineCoreProc.run_engine_core</code>. Each of these functions creates a <code>DPEngineCoreProc</code> (the engine core) and then enters its busy loop.

<code>DPEngineCoreProc</code> initializes its parent <code>EngineCoreProc</code> (child of <code>EngineCore</code>), which:

1. Creates an <code>input_queue</code> and <code>output_queue</code> (<code>queue.Queue</code>).
2. Performs an initial handshake with the frontend on the other node using a <code>DEALER</code> ZMQ socket (async messaging lib), and receives coordination address info.
3. Initializes DP group (e.g. using NCCL backend).
4. Initializes the <code>EngineCore</code> with <code>MultiProcExecutor</code> (<code>TP=4</code> on 4 GPUs as described earlier).
5. Creates a <code>ready_event</code> (<code>threading.Event</code>).
6. Starts an input deamon thread (<code>threading.Thread</code>) running <code>process_input_sockets(…, ready_event)</code>. Similarly starts an output thread.
7. Still in the main thread, waits on <code>ready_event</code> until all input threads across all 4 processes (spanning the 2 nodes) have completed the coordination handshake finally executing <code>ready_event.set()</code>.
8. Once unblocked, sends a "ready" message to the frontend with metadata (e.g., <code>num_gpu_blocks</code> available in paged KV cache memory).
9. The main, input, and output threads then enter their respective busy loops.

TL;DR: We end up with 4 child processes (one per DP replica), each running a main, input, and output thread. They complete a coordination handshake with the DP coordinator and frontend, then all three threads per process run in steady-state busy loops.

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/dpenginecoreproc.png" width="100%">
</picture><br>
<b>Figure 15</b>: distributed system with 4 DP replicas running 4 DPEngineCoreProc
</p>

<b>Current steady state</b>:

* <b>Input thread</b> — blocks on the input socket until a request is routed from the API server; upon receipt, it decodes the payload, enqueues a work item via <code>input_queue.put_nowait(...)</code>, and returns to blocking on the socket.
* <b>Main thread</b> — wakes on <code>input_queue.get(...)</code>, feeds the request to the engine; <code>MultiProcExecutor</code> runs the forward pass and enqueues results to <code>output_queue</code>.
* <b>Output thread</b> — wakes on <code>output_queue.get(...)</code>, sends the result back to the API server, then resumes blocking.

<b>Additional mechanics</b>:

* <b>DP wave counter</b> — the system tracks "waves"; when all engines become idle they quiesce, and the counter increments when new work arrives (useful for coordination/metrics).
* <b>Control messages</b> — the API server can send more than just inference requests (e.g., aborts and utility/control RPCs).
* <b>Dummy steps for lockstep</b> — if any DP replica has work, all replicas execute a forward step; replicas without requests perform a dummy step to participate in required synchronization points (avoids blocking the active replica).

> [!NOTE]
> Lockstep clarification: this is actually only required for MoE models where the expert layers form an EP or TP group while attention layers are still DP. It's currently always done with DP - this is just because there's limited use for "built-in" non-MoE DP since you could just run multiple independent vLLMs and load-balance between them in a normal way.

Now for the second part, what happens on the API server node?

### On the API server node

We instantiate an <code>AsyncLLM</code> object (an asyncio wrapper around the LLM engine). Internally this creates a <code>DPLBAsyncMPClient</code> (data-parallel, load-balancing, asynchronous, multiprocessing client).

Inside the parent class of <code>MPClient</code>, the <code>launch_core_engines</code> function runs and:

1. Creates the ZMQ addresses used for the startup handshake (as seen on the headless node).
2. Spawns a <code>DPCoordinator</code> process.
3. Creates a <code>CoreEngineProcManager</code> (same as on the headless node).

Inside <code>AsyncMPClient</code> (child of <code>MPClient</code>), we:

1. Create an <code>outputs_queue</code> (<code>asyncio.Queue</code>).
2. We create an asyncio task <code>process_outputs_socket</code> which communicates (through the output socket) with output threads of all 4 <code>DPEngineCoreProc</code> and writes into <code>outputs_queue</code>.
3. Subsequently one more asyncio task <code>output_handler</code> from <code>AsyncLLM</code> reads from this queue and finally sends out information to the <code>create_completion</code> function.

Inside <code>DPAsyncMPClient</code> we create an asyncio task <code>run_engine_stats_update_task</code> which communicates with DP coordinator.

The DP coordinator mediates between the frontend (API server) and backend (engine cores). It:

* Periodically sends load-balancing info (queue sizes, waiting/running requests) to the frontend's <code>run_engine_stats_update_task</code>.
* Handles <code>SCALE_ELASTIC_EP</code> commands from the frontend by dynamically changing the number of engines (only works with Ray backend).
* Sends <code>START_DP_WAVE</code> events to the backend (when triggered by frontend) and reports wave-state updates back.

To recap, the frontend (<code>AsyncLLM</code>) runs several asyncio tasks (remember: concurrent, not parallel):

* A class of tasks handles input requests through the <code>generate</code> path (each new client request spawns a new asyncio task).
* Two tasks (<code>process_outputs_socket</code>, <code>output_handler</code>) process output messages from the underlying engines.
* One task (<code>run_engine_stats_update_task</code>) maintains communication with the DP coordinator: sending wave triggers, polling LB state, and handling dynamic scaling requests.

Finally, the main server process creates a FastAPI app and mounts endpoints such as <code>OpenAIServingCompletion</code> and <code>OpenAIServingChat</code>, which expose <code>/completion</code>, <code>/chat/completion</code>, and others. The stack is then served via Uvicorn.

So, putting it all together, here's the full request lifecycle!

You send from your terminal:

```curl
curl -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.7
}'
```

What happens next:

1. The request hits <code>OpenAIServingCompletion</code>'s <code>create_completion</code> route on the API server.
2. The function tokenizes the prompt asynchronously, and prepares metadata (request ID, sampling params, timestamp, etc.).
3. It then calls <code>AsyncLLM.generate</code>, which follows the same flow as the synchronous engine, eventually invoking <code>DPAsyncMPClient.add_request_async</code>.
4. This in turn calls <code>get_core_engine_for_request</code>, which does load balancing across engines based on the DP coordinator's state (picking the one that has minimal score / lowest load: <code>score = len(waiting) * 4 + len(running)</code>).
5. The <code>ADD</code> request is sent to the chosen engine's <code>input_socket</code>.
6. At that engine:
* Input thread — unblocks, decodes data from the input socket, and places a work item on the <code>input_queue</code> for the main thread.
* Main thread — unblocks on <code>input_queue</code>, adds the request to the engine, and repeatedly calls <code>engine_core.step()</code>, enqueueing intermediate results to <code>output_queue</code> until a stop condition is met.

> [!NOTE]
> Reminder: <code>step()</code> calls the scheduler, model executor (which in turn can be <code>MultiProcExecutor</code>!), etc. We have already seen this!

* Output thread — unblocks on <code>output_queue</code> and sends results back through the output socket.

7. Those results trigger the <code>AsyncLLM</code> output asyncio tasks (<code>process_outputs_socket</code> and <code>output_handler</code>), which propagate tokens back to FastAPI's <code>create_completion</code> route.
8. FastAPI attaches metadata (finish reason, logprobs, usage info, etc.) and returns a <code>JSONResponse</code> via Uvicorn to your terminal!

And just like that, your completion came back — the whole distributed machinery hidden behind a simple <code>curl</code> command! :) So much fun!!!

> [!NOTE] Additional notes:
> * When adding more API servers, load balancing is handled at the OS/socket level. From the application's perspective, nothing significant changes — the complexity is hidden.
> * With Ray as a DP backend, you can expose a URL endpoint (<code>/scale_elastic_ep</code>) that enables automatic scaling of the number of engine replicas up or down.

## Benchmarks and auto-tuning - latency vs throughput

So far we've been analyzing the "gas particles" — the internals of how requests flow through the engine/system. Now it's time to zoom out and look at the system as a whole, and ask: how do we measure the performance of an inference system?

At the highest level there are two competing metrics:

1. <b>Latency</b> — the time from when a request is submitted until tokens are returned
2. <b>Throughput</b> — the number of tokens/requests per second the system can generate/process

<b>Latency</b> matters most for interactive applications, where users are waiting on responses.

<b>Throughput</b> matters in offline workloads like synthetic data generation for pre/post-training runs, data cleaning/processing, and in general - any type of offline batch inference jobs.

Before explaining why latency and throughput compete, let's define a few common inference metrics:

<table className="border border-gray-300 border-collapse w-full my-4">
  <thead>
    <tr className="bg-gray-100">
      <th className="border border-gray-300 px-4 py-2 text-left font-semibold">Metric</th>
      <th className="border border-gray-300 px-4 py-2 text-left font-semibold">Definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td className="border border-gray-300 px-4 py-2"><code>TTFT</code><br/>(time to first token)</td>
      <td className="border border-gray-300 px-4 py-2">Time from request submission until the first output token is received</td>
    </tr>
    <tr>
      <td className="border border-gray-300 px-4 py-2"><code>ITL</code><br/>(inter-token latency)</td>
      <td className="border border-gray-300 px-4 py-2">Time between two consecutive tokens (e.g., from token i-1 to token i)</td>
    </tr>
    <tr>
      <td className="border border-gray-300 px-4 py-2"><code>TPOT</code><br/>(time per output token)</td>
      <td className="border border-gray-300 px-4 py-2">The average ITL across all output tokens in a request</td>
    </tr>
    <tr>
      <td className="border border-gray-300 px-4 py-2"><code>Latency / E2E</code><br/>(end-to-end latency)</td>
      <td className="border border-gray-300 px-4 py-2">Total time to process a request, i.e. TTFT + sum of all ITLs, or equivalently the time between submitting request and receiving the last output token</td>
    </tr>
    <tr>
      <td className="border border-gray-300 px-4 py-2"><code>Throughput</code></td>
      <td className="border border-gray-300 px-4 py-2">Total tokens processed per second (input, output, or both), or alternatively requests per second</td>
    </tr>
    <tr>
      <td className="border border-gray-300 px-4 py-2"><code>Goodput</code></td>
      <td className="border border-gray-300 px-4 py-2">Throughput that meets service-level objectives (SLOs) such as max TTFT, TPOT, or e2e latency. For example, only tokens from requests meeting those SLOs are counted</td>
    </tr>
  </tbody>
</table>

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/latency_diagram.png" width="100%">
</picture><br>
<b>Figure 16</b>: ttft, itl, e2e latency
</p>

Here is a simplified model explaining the competing nature of these 2 metrics.

> [!NOTE] Assumption:
> weight i/o and not KV cache i/o dominates; i.e. we're dealing with short sequences.

The tradeoff becomes clear when looking at how batch size <code>B</code> affects a single decode step. As <code>B ↓</code> toward 1, ITL drops: there's less work per step and the token isn't "competing" with others. As <code>B ↑</code> toward infinity, ITL rises because we do more FLOPs per step—but throughput improves (until we hit peak perf) because weight I/O is amortized across more tokens.

A roofline model helps with understanding here: below a saturation batch <code>B_sat</code>, the step time is dominated by HBM bandwidth (streaming weights layer-by-layer into on-chip memory), so step latency is nearly flat—computing 1 vs 10 tokens can take a similar time. Beyond <code>B_sat</code>, the kernels become compute-bound and step time grows roughly with <code>B</code>; each extra token adds to ITL.

<p align="center">
<picture>
<img src="/assets/figures/2025-vllm-anatomy/roofline.png" width="100%">
</picture><br>
<b>Figure 17</b>: roofline perf model
</p>

> [!NOTE] Note:
> For a more rigorous treatment, we have to account for kernel auto-tuning: as <code>B</code> grows, the runtime may switch to more efficient kernels for that shape, changing the achieved performance <code>P_kernel</code>. Step latency is <code>t = FLOPs_step / P_kernel</code>, where <code>FLOPs_step</code> is the work in the step. You can see that as <code>P_kernel</code> hits <code>P_peak</code> more compute per step will directly lead to an increase in latency.

### How to benchmark in vLLM

vLLM provides a <code>vllm bench {serve,latency,throughput}</code> CLI that wraps vllm / benchmarks / {server,latency,throughput}.py.

Here is what the scripts do:

* <b>latency</b> — uses a short input (default 32 tokens) and samples 128 output tokens with a small batch (default 8). It runs several iterations and reports e2e latency for the batch.
* <b>throughput</b> — submits a fixed set of prompts (default: 1000 ShareGPT samples) all at once (aka as <code>QPS=Inf</code> mode), and reports input/output/total tokens and requests per second across the run.
* <b>serve</b> — Launches a vLLM server and simulates a real-world workload by sampling request inter-arrival times from a Poisson (or more generally, Gamma) distribution. It sends requests over a time window, measures all the metrics we’ve discussed, and can optionally enforce a server-side max concurrency (via a semaphore, e.g. limiting the server to 64 concurrent requests).

Here is an example of how you can run the latency script:

```shell
vllm bench latency
  --model <model-name>
  --input-tokens 32
  --output-tokens 128
  --batch-size 8
```

> [!NOTE]
> Benchmark configs used in CI live under <code>.buildkite/nightly-benchmarks/tests</code>.


There is also an auto-tune script that drives the serve benchmark to find argument settings that meet target SLOs (e.g., "maximize throughput while keeping p99 e2e < 500 ms"), returning a suggested config.

## Epilogue

We began with the basic engine core (<code>UniprocExecutor</code>), added advanced features like speculative decoding and prefix caching, scaled up to <code>MultiProcExecutor</code> (with <code>TP/PP > 1</code>), and finally scaled out, wrapped everything in the asynchronous engine and distributed serving stack—closing with how to measure system performance.

vLLM also includes specialized handling that I've skipped. E.g.:

* <b>Diverse hardware backends</b>: TPUs, AWS Neuron (Trainium/Inferentia), etc.
* <b>Architectures/techniques</b>: <code>MLA</code>, <code>MoE</code>, encoder-decoder (e.g., Whisper), pooling/embedding models, <code>EPLB</code>, <code>m-RoPE</code>, <code>LoRA</code>, <code>ALiBi</code>, attention-free variants, sliding-window attention, multimodal LMs, and state-space models (e.g., Mamba/Mamba-2, Jamba)
* <b>TP/PP/SP</b>
* <b>Hybrid KV-cache logic</b> (Jenga), more complex sampling methods like beam sampling, and more
* <b>Experimental</b>: async scheduling

The nice thing is that most of these are orthogonal to the main flow described above—you can almost treat them like "plugins" (in practice there's some coupling, of course).

I love understanding systems. Having said that, the resolution definitely suffered at this altitude. In the next posts I'll zoom in on specific subsystems and get into the nitty-gritty details.

> [!NOTE] Get in touch:
> If you spot any errors in the post, please DM me - feel free to drop me a message on [X](https://x.com/gordic_aleksa) or [LinkedIn](https://www.linkedin.com/in/aleksagordic/) or via [anon feedback](https://docs.google.com/forms/d/1z1fEirrN2xtGxAsJvptpM7yV4ByT5SF25S-XiMPrXNA).

### Acknowledgments

A huge thank you to [Hyperstack](https://www.hyperstack.cloud/) for providing me with H100s for my experiments over the past year!

Thanks to [Nick Hill](https://www.linkedin.com/in/nickhillprofile/) (core vLLM contributor, RedHat), [Mark Saroufim](https://x.com/marksaroufim) (PyTorch), [Kyle Krannen](https://www.linkedin.com/in/kyle-kranen/) (NVIDIA, Dynamo), and [Ashish Vaswani](https://www.linkedin.com/in/ashish-vaswani-99892181/) for reading pre-release version of this blog post and providing feedback!

References
1. <a id="ref-1"> </a> vLLM <a href="https://github.com/vllm-project/vllm">https://github.com/vllm-project/vllm </a>
2. <a id="ref-2"> </a> "Attention Is All You Need" <a href="https://arxiv.org/abs/1706.03762">https://arxiv.org/abs/1706.03762</a>
3. <a id="ref-3"> </a> "Efficient Memory Management for Large Language Model Serving with PagedAttention" <a href="https://arxiv.org/abs/2309.06180">https://arxiv.org/abs/2309.06180</a>
4. <a id="ref-4"> </a> "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" <a href="https://arxiv.org/abs/2405.04434">https://arxiv.org/abs/2405.04434</a>
5. <a id="ref-5"> </a> "Jenga: Effective Memory Management for Serving LLM with Heterogeneity" <a href="https://arxiv.org/abs/2503.18292">https://arxiv.org/abs/2503.18292</a>
6. <a id="ref-6"> </a> "Orca: A Distributed Serving System for Transformer-Based Generative Models" <a href="https://www.usenix.org/conference/osdi22/presentation/yu">https://www.usenix.org/conference/osdi22/presentation/yu</a>
7. <a id="ref-7"> </a> "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" <a href="https://arxiv.org/abs/2411.15100">https://arxiv.org/abs/2411.15100</a>
8. <a id="ref-8"> </a> "Accelerating Large Language Model Decoding with Speculative Sampling" <a href="https://arxiv.org/abs/2302.01318">https://arxiv.org/abs/2302.01318</a>
9. <a id="ref-9"> </a> "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" <a href="https://arxiv.org/abs/2401.15077">https://arxiv.org/abs/2401.15077</a>
10. <a id="ref-10"> </a> "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" <a href="https://arxiv.org/abs/2401.10774">https://arxiv.org/abs/2401.10774</a>
11. <a id="ref-11"> </a> LMCache <a href="https://github.com/LMCache/LMCache">https://github.com/LMCache/LMCache</a>