---
layout: post
title: "Taking vLLM Apart: A Practical Guide to Disaggregated Serving"
author: "Martin Hickey (IBM Research)"
summary: "What disaggregated serving actually buys you, how to run it end to end in vLLM today with prefill/decode plus the new GPU-less frontend and the things we're still working on."
image: /assets/figures/2026-09-20-disaggregated-serving-guide/pipeline.svg
social_image: /assets/figures/2026-09-20-disaggregated-serving-guide/pipeline.png
tags:
  - disaggregation
---

**TL;DR:** vLLM can now be split across two dimensions. Prefill and decode run on separate instances with the KV cache transferred through a connector. Tokenization, detokenization, tool-call parsing and reasoning parsing run on a CPU-only frontend via `/render` and `/derender`. This leaves the GPU focused on token generation while everything else runs on lower cost CPUs. For chat and agent workflows, prefill can reuse conversation state from decode instead of recomputing it. In this post, we will show how to run each piece on vLLM v0.30.0 or later, how they fit together, who's running it in production and what's still missing.

---

## One Server Doing Three Unrelated Jobs

Start a plain `vllm serve` and you get one process handling three workloads that have nothing in common.

Prefill chews through the whole prompt in parallel. Big GEMMs, compute bound, cost scales with input length and it sets your time to first token (TTFT). Decode emits one token at a time, dragging model weights out of HBM on every step. It's memory bandwidth bound and it sets your inter-token latency (ITL). Put them on the same GPU and they fight. One long prompt lands mid-batch and dozens of decode streams stutter while it clears. That's the ITL spike everyone sees the moment concurrency goes up.

The third job however is quieter and easier to miss. Chat templating, tokenization, detokenization, reasoning parsing, tool call parsing. All pure CPU work running on a box you're renting for its accelerators.

Disaggregation is the obvious response: stop making one process do all of it. If you want the vLLM engine level picture before you go further, [Inside vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) covers how the scheduler mixes prefill and decode today and has a short section on P/D.

## The Dimensions

vLLM offers several points where work can be split and they can be combined.

**Prefill and decode** run as two instances. What passes between them is the KV cache: the attention keys and values for every prompt token which decode needs before it can emit anything. It's big. Llama-3.1-70B in BF16 stores 320 KiB per token, so a 10k-token prompt hands decode about 3 GB. That's roughly 65 ms at a 400 Gb/s line rate before any overhead and all of it lands on TTFT. A [KV connector](https://docs.vllm.ai/en/latest/features/disagg_prefill/) moves it, usually over RDMA. There are more than a dozen connectors upstream now, including NIXL, LMCache, Mooncake, FlexKV and AMD's MoRI-IO, plus a `MultiConnector` that chains them.

**The frontend** comes off the GPU box entirely. `/render` turns an OpenAI request into token IDs, the engine runs token-in / token-out and `/derender` turns the output token IDs back into a proper OpenAI response with `content`, `reasoning` and `tool_calls` split out. That last leg only landed recently and it's what closes the loop.

**Beyond P/D**, there's [encoder disaggregation](https://vllm.ai/blog/2025-12-15-vllm-epd) for multimodal and the [AFD plugin](https://vllm.ai/blog/2026-07-23-vllm-afd-plugin) for splitting attention from FFN in MoE models. Both apply the same disaggregation principle but at different points in the model pipeline. P/D itself now also covers [hybrid SSM models](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) with Mamba state transfer included.

<p align="center">
<picture>
<img src="/assets/figures/2026-09-20-disaggregated-serving-guide/pipeline.svg" width="95%" alt="Collocated vLLM serving versus a four-tier disaggregated pipeline">
</picture>
<br>
<em>Figure 1. One process doing everything, versus the same pipeline cut into four tiers.</em>
</p>

## What It Buys You and What It Costs

The pitch isn't raw throughput. The [disaggregated prefill docs](https://docs.vllm.ai/en/latest/features/disagg_prefill/) say it outright: **disaggregated prefill does not improve throughput.** What it buys you is control and control is what you're actually selling when you sign an SLA.

**You can tune TTFT and ITL independently.** Different parallelism on each tier, sized for the phase it's running. Prefill can be TP-heavy, decode can be sized for batch. Neither change drags the other one with it.

**Tail latency gets boring.** A decode instance that only ever runs decode batches produces stable per-token latency no matter what's arriving at the front door. Chunked prefill gets you partway there but only if you guess the chunk size right and that guess moves with the traffic.

This was measured on one box with two NVIDIA L40S GPUs (48 GB, PCIe, no NVLink): Qwen2.5-7B-Instruct, ~8k-token prompts, 256 output tokens, 100 Poisson-arrival requests per rate, prefix caching off. Collocated is one `vllm serve --data-parallel-size 2`, so both setups get the same two GPUs. P/D is one prefiller and one decoder over NIXL behind the example proxy.

<p align="center">
<picture>
<img src="/assets/figures/2026-09-20-disaggregated-serving-guide/itl-tail.svg" width="95%" alt="p99 inter-token latency: collocated 23, 169 and 182 ms versus P/D 25, 29 and 30 ms at 0.2, 0.4 and 0.6 req/s">
</picture>
<br>
<em>Figure 2. Median ITL is 21–24 ms in both setups. At 0.4 req/s, collocated p99 jumps to 169 ms while P/D holds at 29 ms.</em>
</p>

Same median, about six times the tail. That's 8k-token prefills landing on a GPU that's also decoding and stalling every stream on it until they finish. P/D keeps them off the decode GPU entirely.

**Every first token pays for the transfer.** On this box it paid a lot. At 0.2 req/s, where nothing should queue, P/D median TTFT is 2.2 s against 0.7 s collocated. Almost all of that extra 1.5 s is the KV transfer: each 8k prompt hands decode about 470 MB of KV cache (56 KiB per token for Qwen2.5-7B) and each pull takes about 1.3 s. The GPUs can't do peer-to-peer copies (`nvidia-smi topo -p2p r` reports `NS`), so every block detours through host memory.

<p align="center">
<picture>
<img src="/assets/figures/2026-09-20-disaggregated-serving-guide/transfer-cost.svg" width="95%" alt="Output throughput and goodput versus offered load. P/D throughput flattens at about 182 tok/s; its goodput never exceeds 0.04 req/s, while collocated goodput peaks at 0.43 req/s">
</picture>
<br>
<em>Figure 3. P/D throughput flattens at about 182 tok/s (≈0.7 req/s), and its goodput stays near zero. Goodput counts requests that meet TTFT under 2 s and TPOT under 30 ms.</em>
</p>

That slow transfer caps throughput too and it sinks goodput. Goodput is the request rate you can sustain while requests still meet both latency targets which is what an SLA actually cares about. P/D fails on TTFT at every rate, even though its decode speed passes TPOT easily. Collocated fails the other way: prefill interference pushes more and more requests past 30 ms per token, so its goodput peaks at 0.43 req/s and falls to zero by 2 req/s.

This is less a P/D trade-off than a slow wire. With a transfer in the tens of milliseconds, P/D's TTFT would sit close to collocated's and its flat tail would start winning goodput wherever collocated's falls apart. So check the transfer before you benchmark anything else. On one box, `nvidia-smi topo -p2p r` should say `OK` between your prefill and decode GPUs. Then send a few long prompts one at a time and read decode's `KV Transfer metrics` line. If `Avg xfer time` is in the hundreds of milliseconds, fix that first.

**Goodput goes up when the transfer is fast.** AMD's single-node [MoRI-IO benchmark](https://vllm.ai/blog/2026-04-07-moriio-kv-connector) ran Qwen3-235B-A22B-FP8 at 8 req/s on one 8-GPU MI300X node. 73 of 100 requests met both a 1 s TTFT and a 50 ms ITL target, against 30 of 100 for collocated serving. That's about 2.4× the goodput, with no extra hardware, just a different arrangement of it. At cluster scale, [llm-d's P/D guide](https://github.com/llm-d/llm-d/tree/main/guides/pd-disaggregation) reports about 59% lower mean end-to-end latency and 67% lower P95 for gpt-oss-120b on 16 H200s, compared with the same GPUs run as aggregated replicas.

**The CPU tier is cheap.** Once tokenization and parsing move off the GPU box, you scale them against CPU load instead of buying accelerator time to run a tokenizer. Long prompts, multimodal preprocessing and reasoning or tool parsing are where the render tier does real work and none of it needs a GPU. On the same box, templating and tokenizing a 9k-token chat prompt for Qwen2.5-7B cost about 15 ms of CPU. One render server with default settings topped out at 73 req/s using just over one core. At the 0.4 req/s where collocated's tail fell apart, rendering is under 1% of one core.

**The catch:** you now operate three or four services instead of one and the KV transfer is a new failure mode. Collocated is still the right answer for plenty of deployments.

| Your situation | Recommendation |
| -------------- | -------------- |
| ITL p99 misses your SLO under production load | Disaggregate. This is the main use case. |
| Long prompts at high concurrency | Disaggregate, if your KV transfer is fast. Prefill interference is worst here. |
| Chat or agent loops over a growing context | Disaggregate, with bidirectional transfer (below). |
| Templating, tokenizing or parsing shows up in your GPU nodes' CPU profile | Split off the render tier. |
| TTFT is the binding constraint | Stay collocated or measure first. The transfer lands on every first token. |
| Your KV transfer is slow (check decode's `KV Transfer metrics`) | Fix it or stay collocated. It lands on TTFT and caps throughput. |
| Low, bursty or latency-insensitive traffic | Stay collocated. |

## Running Prefill/Decode

Everything from here on assumes vLLM v0.30.0 or later. The examples use Qwen3-0.6B because it loads fast. That's fine for checking the wiring but it's too small to show a P/D benefit, so benchmark with a larger model (see [Where to start](#where-to-start)).

Three processes: prefiller, decoder, proxy.

```bash
# Prefiller on GPU 0
CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-0.6B --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'

# Decoder on GPU 1
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve Qwen/Qwen3-0.6B --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'

# Proxy
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192 \
  --prefiller-hosts localhost --prefiller-ports 8100 \
  --decoder-hosts localhost --decoder-ports 8200
```

The proxy is what ties them together. For each request it calls prefill first, with `max_tokens=1` and `kv_transfer_params: {"do_remote_decode": true}`. Prefill computes the KV cache, holds the blocks and returns `kv_transfer_params` that point at them. The proxy then forwards the original request to decode with those params attached and decode pulls the blocks over NIXL before it generates.

Point your client at 8192 and it looks like any other OpenAI endpoint. The proxy in `tests/` is an example that's fine for development but not for production. Check out llm-d and Dynamo for production level capability.

Three settings worth knowing early. `VLLM_NIXL_SIDE_CHANNEL_PORT` must be unique per worker on a host. `kv_lease_duration` (set in `kv_connector_extra_config`, default 30s) controls how long the prefiller holds blocks waiting for the decoder to collect them. Under load, that's the timeout you'll be tuning. And `kv_load_failure_policy` decides what happens when a transfer fails: `fail`, the default, errors the request, while `recompute` has decode recompute the missing KV itself. Slower, but the request survives.

One easy win if you're running P/D on `/v1/chat/completions`: the prefill stage already tokenized the prompt, so the decode stage doesn't need to do it again. Ask for `return_token_ids` on prefill, then hand the IDs to decode through `kv_transfer_params`. This is the same two step call the proxy makes and the token IDs are the only new part:

```python
from openai import OpenAI

MODEL = "Qwen/Qwen3-0.6B"
messages = [{"role": "user", "content": "What is 17 * 23?"}]

prefill_client = OpenAI(base_url="http://localhost:8100/v1", api_key="EMPTY")
decode_client = OpenAI(base_url="http://localhost:8200/v1", api_key="EMPTY")

prefill = prefill_client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1,
    extra_body={"return_token_ids": True, "kv_transfer_params": {"do_remote_decode": True}})

# prefill's kv_transfer_params point decode at its blocks; add the token IDs on top
decode = decode_client.chat.completions.create(model=MODEL, messages=messages, stream=True,
    extra_body={"kv_transfer_params": {**prefill.kv_transfer_params,
                                       "prompt_token_ids": prefill.prompt_token_ids}})
```

Once it works, sizing it is the hard part. The [Qwen3.8-2.4T PD post](https://vllm.ai/blog/2026-09-21-qwen38-pd-serving) walks through picking prefill and decode topologies from KV cache capacity up.

## Multi-Turn: Stop Recomputing the Conversation

Standard P/D moves the cache one way. Prefill computes and decode reads. Fine for one-shot requests but wasteful for chat. On turn two of chat, decode still holds KV for everything it just generated and prefill has never computed those tokens. Prefill's own prefix cache usually still covers the turn-one prompt, so what it recomputes is the previous answer. It recomputes the whole conversation if its cache was evicted under load or the turn lands on a different prefill instance. The longer the answers, the more that costs. Agent loops make it worse because every tool call is another turn over a transcript that keeps growing.

Bidirectional transfer flips the arrow on a cache hit. Prefill pulls the blocks it doesn't already have back from decode over RDMA and computes only the new tokens.

<p align="center">
<picture>
<img src="/assets/figures/2026-09-20-disaggregated-serving-guide/multiturn.svg" width="95%" alt="Turn 1 cache miss versus turn 2 cache hit with bidirectional KV transfer">
</picture>
<br>
<em>Figure 4. On a cache hit, prefill reads the conversation back from decode instead of recomputing it.</em>
</p>

Switch it on with `bidirectional_kv_xfer` on **both** instances:

```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer",
  "kv_connector_extra_config":{"bidirectional_kv_xfer":true}}'
```

Something has to remember the conversation and vLLM doesn't. That's the proxy's job: it caches the `kv_transfer_params` decode returns at the end of each turn and attaches them to the next request, keyed by a `conversation_id` the client sends.

```bash
python examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py \
  --host 0.0.0.0 --port 8000 \
  --prefiller-host <P_IP> --prefiller-port 8100 \
  --decoder-host <D_IP> --decoder-port 8200
```

`conversation_id` is a non-standard field that identifies the conversation. The proxy consumes it and never forwards it to the engine. Leave it out and nothing links the turns, so every request is a full recompute.

Two defaults worth knowing. `kv_recompute_threshold` (64 tokens) is the point below which prefill recomputes locally rather than pulling because the transfer isn't worth the round trip. `decoder_kv_blocks_ttl` (480s) is how long decode holds blocks for reuse and unlike the prefiller's lease it isn't renewed by heartbeats. So a conversation that goes quiet for longer pays full price on its next turn.

Benchmarking this has one trap. `benchmarks/multi_turn/benchmark_serving_multi_turn.py` needs `--send-conversation-id` which is off by default so the benchmark stays compatible with frontends that reject unknown fields. Forget it and every turn is a miss and you measure exactly the thing you were trying to avoid.

The gotcha is reasoning models. Decode's blocks cover every token it generated, thinking traces included. If the next turn's prompt drops those traces, it's missing tokens from the middle of what decode produced. Clients can do that and so can chat templates. Qwen3's template drops `<think>` blocks from earlier assistant turns on its own, however the client sends the history. Block alignment assumes prefill's prompt is a prefix of decode's sequence, so the pull hands over cache computed for the wrong positions and you get wrong output, not just slow output. Nothing in vLLM catches that mismatch today. The NIXL docs leave it to the router and the tracking issue ([#43094](https://github.com/vllm-project/vllm/issues/43094)) was closed as stale without a fix. Check your model's chat template before you turn this on. If it or your clients strip thinking traces, either make your router detect the mismatch or keep bidirectional transfer off for that model. The feature is also CUDA-only with device buffer KV for now. Host-buffer support for XPU and similar is still to come.

## Running the GPU-Less Frontend

Two servers: a render tier with no GPU and an engine that speaks tokens only. The parsers go on the render server since that's where derender runs them.

```bash
vllm launch render Qwen/Qwen3-0.6B --port 8100 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes
vllm serve Qwen/Qwen3-0.6B --tokens-only --port 8200
```

Both of those modes always expose their scale-out endpoints. On a regular `vllm serve` they're off by default. If you want `/render`, `/derender` or `/inference/v1/generate` on a server that also takes normal OpenAI traffic, add `--enable-scale-out`.

Then it's three hops — render, generate, derender:

```python
import httpx

MODEL = "Qwen/Qwen3-0.6B"
RENDER = "http://localhost:8100"  # vllm launch render
ENGINE = "http://localhost:8200"  # vllm serve --tokens-only

chat_request = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "max_tokens": 2048,
}

with httpx.Client(timeout=60.0) as client:
    # 1. request -> token IDs (no GPU)
    generate_request = client.post(f"{RENDER}/v1/chat/completions/render", json=chat_request).json()

    # 2. token IDs -> token IDs (GPU)
    generate_response = client.post(f"{ENGINE}/inference/v1/generate", json=generate_request).json()

    # 3. token IDs -> ChatCompletionResponse (no GPU)
    response = client.post(f"{RENDER}/v1/chat/completions/derender", json={
        "model": MODEL,
        "generate_response": generate_response,
        "prompt_tokens": len(generate_request["token_ids"]),
        "chat_request": chat_request,
    }).json()

message = response["choices"][0]["message"]
print(message["reasoning"], message["content"], sep="\n---\n")
```

Pass `chat_request` back into the derender step. The parsers need the original context (tools, `tool_choice`, `include_reasoning`) to produce the same `content` / `reasoning` / `tool_calls` split a normal `vllm serve` would. On a model with a parser configured, leaving it out gets you a 400 rather than a silent fallback that leaks `<tool_call>` markup into `content`.

Both endpoints also take `stream: true`. Streaming derender is stateless which means each call returns `{chunk, stream_state}` and the client echoes the state back on the next one. No sessions on the render tier.

## Putting Both Splits Together

Figure 1 shows both splits at once. That's four tiers but only three servers since one `vllm launch render` server handles both `/render` and `/derender`. Nothing upstream drives the four hops for you yet but the pieces combine because `/inference/v1/generate` accepts `kv_transfer_params` just like `/v1/chat/completions` does.

```bash
# Render and derender, no GPU
vllm launch render Qwen/Qwen3-0.6B --port 8000 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes

# Prefill on GPU 0
CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-0.6B --tokens-only --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# Decode on GPU 1
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve Qwen/Qwen3-0.6B --tokens-only --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

Your code then plays the proxy's part between render and derender:

```python
import httpx

MODEL = "Qwen/Qwen3-0.6B"
RENDER, PREFILL, DECODE = "http://localhost:8000", "http://localhost:8100", "http://localhost:8200"

chat_request = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "max_tokens": 2048,
}

with httpx.Client(timeout=60.0) as client:
    # 1. Render: request -> token IDs (no GPU)
    generate_request = client.post(f"{RENDER}/v1/chat/completions/render", json=chat_request).json()

    # 2. Prefill: compute the KV cache, generate one token, hold the blocks
    prefill_response = client.post(f"{PREFILL}/inference/v1/generate", json={
        **generate_request,
        "sampling_params": {**generate_request["sampling_params"], "max_tokens": 1},
        "kv_transfer_params": {"do_remote_decode": True},
    }).json()

    # 3. Decode: pull prefill's blocks over NIXL, then generate
    generate_response = client.post(f"{DECODE}/inference/v1/generate", json={
        **generate_request,
        "kv_transfer_params": prefill_response["kv_transfer_params"],
    }).json()

    # 4. Derender: token IDs -> ChatCompletionResponse (no GPU)
    response = client.post(f"{RENDER}/v1/chat/completions/derender", json={
        "model": MODEL,
        "generate_response": generate_response,
        "prompt_tokens": len(generate_request["token_ids"]),
        "chat_request": chat_request,
    }).json()

message = response["choices"][0]["message"]
print(message["reasoning"], message["content"], sep="\n---\n")
```

Step 2 is the same trick the proxy uses: a one-token budget plus `do_remote_decode`, so prefill computes the KV cache and holds on to the blocks. Step 3 hands decode the `kv_transfer_params` prefill returned which point at those blocks. The example proxy only handles `/v1/completions` and `/v1/chat/completions`, so in production this orchestration is yours to write. It's also the job of llm-d and Dynamo which the renderer docs name as its intended callers.

## Who's Actually Running This

[llm-d](https://llm-d.ai/) uses vLLM's native remote prefill/decode and NIXL KV-transfer capabilities. llm-d's router and EPP scheduler choose the prefill/decode endpoints and orchestrate the KV-cache handoff between them. [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) runs vLLM on Kubernetes in aggregated or disaggregated mode with its own router and planner. [KServe](https://kserve.github.io/website/) exposes it through `LLMInferenceService`, built on llm-d.

The [vLLM production stack](https://github.com/vllm-project/production-stack) ships disaggregated prefill via Helm and [AIBrix](https://github.com/vllm-project/aibrix) covers the control plane side. On the connector side, Moonshot's [Mooncake](https://vllm.ai/blog/2026-05-06-mooncake-store), LMCache and AMD's MoRI-IO are all upstream. And because decode sits behind a connector, it doesn't even have to be vLLM: [TileRT](https://vllm.ai/blog/2026-07-14-vllm-tilert-pd) pairs stock vLLM prefill with its own latency optimized decode engine.

## What's Still Left To Do?

Substantial progress has been made but a few things are worth knowing about before you start.

Streaming derender with a parser is expensive. Reasoning and tool parsers hold state that can't be serialized, so each chunk rebuilds a fresh parser and replays the token history through it. Transport is O(n) per chunk, replay is O(n) parse calls per chunk and `parse_delta` itself rescans accumulated text for parsers like Hermes and DeepSeek-R1. That's O(n³) character work over a long generation. A benchmark on multi-thousand-token reasoning output measured roughly 9× the in-process CPU at matched load and +15% on E2E p50. A caching layer is scheduled ([#57571](https://github.com/vllm-project/vllm/issues/57571)). Also, replay runs on the renderer's executor, `renderer_num_workers` defaults to **1** and the tier saturates under concurrent parsed streams. Size it for your concurrent parsed stream load.

Long prompts dominate the wire. The parser path resends `prompt_token_ids` in full on every chunk. A 100k-token prompt with 1k tokens of output means the prompt is ~99% of each request body. Plain detokenization without a parser doesn't have this problem.

Some small correctness gaps are being worked on. Batch derender doesn't always return `finish_reason: "tool_calls"` when it parses tool calls (fix pending in [#47931](https://github.com/vllm-project/vllm/pull/47931)) and it mints its own tool call IDs instead of keeping the parser's. Logprobs come back from the tokenizer free engine as `"token_id:N"` placeholder strings which works but isn't ideal. [#57574](https://github.com/vllm-project/vllm/issues/57574) proposes returning real integer token IDs. 

There is one open design piece of work in progress. [#56851](https://github.com/vllm-project/vllm/issues/56851) asks whether `/inference/v1/generate` should just return text or derendered output directly, skipping the third hop for callers who don't need it. 

The umbrella RFCs are [#42729](https://github.com/vllm-project/vllm/issues/42729) (detokenization batch), [#47161](https://github.com/vllm-project/vllm/issues/47161) (detokenization streaming), [#22817](https://github.com/vllm-project/vllm/issues/22817) (tokens-in, tokens-out) and [#34407](https://github.com/vllm-project/vllm/issues/34407) (disaggregated frontend). Please weigh in on the RFCs and let us know what you think.

## Where To Start

If you've got two GPUs, run the NIXL example above with a 7B-class model. Qwen3-0.6B prefills so fast that it barely disturbs decode, so there's nothing for P/D to fix. Check the transfer first, as discussed above. Then run the same benchmark against the proxy on 8192 and against one collocated `vllm serve --data-parallel-size 2` on the same two GPUs, and compare p99 ITL, TTFT and goodput. Step `--request-rate` up until the collocated server's p99 ITL falls apart. With 8k prompts on the test box that was 0.4 req/s. Start every server with `--no-enable-prefix-caching` if you reuse a `--seed` across rates. Otherwise later rates replay prompts the server has cached and the cache hits hide exactly the prefill interference you're trying to measure.

```bash
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --port 8192 \
  --dataset-name random --random-input-len 8192 --random-output-len 256 --ignore-eos \
  --request-rate 0.4 --num-prompts 100 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 \
  --goodput ttft:2000 tpot:30
```

Serving chat or agents? Turn on `bidirectional_kv_xfer` and measure TTFT on turn five, not turn one. Point the multi-turn benchmark at the multi-turn proxy and don't forget the flag:

```bash
python benchmarks/multi_turn/benchmark_serving_multi_turn.py \
  --model Qwen/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --input-file benchmarks/multi_turn/generate_multi_turn.json \
  --num-clients 2 --max-active-conversations 6 \
  --send-conversation-id
```

If you're already on Kubernetes, start from llm-d or Dynamo rather than building a proxy yourself. And if you're running a reasoning or tool-calling model through streaming derender, benchmark the render tier before you size it — that's the part most likely to surprise you.

Docs: [disaggregated prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/), [NixlConnector](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/), [renderer](https://docs.vllm.ai/en/latest/serving/online_serving/renderer/), [derenderer](https://docs.vllm.ai/en/latest/serving/online_serving/derenderer/).

## Acknowledgements

Thanks to @aoshen02, Bongwoo Bak, Chauncey, Guan-Ming Chiu, Hyunkyun Moon, Konstantin Dunas, Maroon Ayoub, Nick Hill, Nicolò Lucchesi, Nithin Chalapathi, Robert Shaw, Sagi Ahrac, Seiji Eicher, @snadampal, Will Eaton, Yuqi Wang, and Yuge Zhang, as well as others in the vLLM community for helping deliver disaggregated serving.
