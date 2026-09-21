---
layout: post
title: "Push-Based KV Transfer for Disaggregated Inference in vLLM"
author: "Sunita Nadampalli (AWS), Nicolò Lucchesi (Mistral AI), and the vLLM Team"
summary: "A push-based NIXL connector for vLLM disaggregated serving: the prefill instance writes KV directly into decode memory over RDMA, taking the router relay, decode-side allocation, and transfer setup off the TTFT critical path."
image: /assets/figures/2026-09-21-nixl-push-connector/ttft-push-vs-pull.png
social_image: /assets/figures/2026-09-21-nixl-push-connector/ttft-push-vs-pull.png
tags:
  - disaggregation
  - kv_cache
  - nixl
  - performance
---

Disaggregated prefill(P)/decode(D) serving splits the two phases of large language model (LLM) inference across distinct engine instances: a *prefill* instance computes the KV cache for the prompt, and a *decode* instance consumes that cache to generate tokens. This lets each phase scale and be tuned independently, but it adds a new step to the critical path: the KV cache must move from the prefill instance to the decode instance before generation can begin.

One of the ways vLLM performs this transfer is through the NIXL connector, which moves KV blocks over RDMA-capable interconnects such as InfiniBand or AWS Elastic Fabric Adapter (AWS EFA, the low latency RDMA network protocol used across all servers in AWS). The default mode for the NIXL connector is **pull-based**: only after prefill finishes does the decode instance allocate its blocks, configure NIXL, and issue an RDMA READ to pull the KV, and it learns the prefill instance's block IDs through a router round trip first. Because these steps run serially before the first token, they land directly on the time-to-first-token (TTFT) critical path, making TTFT sensitive to router and scheduler congestion as shown in the following timeline diagram.

<figure>
  <img src="/assets/figures/2026-09-21-nixl-push-connector/pull-mode-timeline.png" alt="Pull-mode timeline: router relay, decode-side allocation, and NIXL READ run serially on the path to first token." style="width: 100%;">
  <figcaption><em>Pull mode is fully serial: the router relay, the decode-side allocation, and the NIXL READ all sit between the end of prefill and the first token.</em></figcaption>
</figure>

This post introduces a **push-based** NIXL connector that takes each of these off the critical path. The decode instance registers its target blocks up front, overlapping with prefill, so there are no post-prefill allocations or router relay to wait on. The prefill instance then writes the KV directly into the decode instance's memory as soon as its blocks are ready, replacing the decode-initiated READ. Because AWS EFA supports GPUDirect RDMA, the KV blocks move directly between GPU memories without staging through host memory. A dedicated background thread handles the transfer setup — descriptor preparation and submission — so it runs off the model's forward-pass loop instead of in sync with it. Together these deliver up to **3x better TTFT** and up to **30% better time-per-output-token (TPOT)** (see the [Performance data](#performance-data) section). We built the feature on AWS Trainium instance clusters with AWS EFA and upstreamed it to vLLM as an accelerator-agnostic feature. The results here are from an AWS p5en (GPU) cluster with EFA, showing the gains carry over to GPUs.

The feature ([#35264](https://github.com/vllm-project/vllm/pull/35264)) is implemented in `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` and interoperates with vLLM's hybrid KV cache manager for sliding-window models. It does not modify the default pull-mode connector. The two are mutually exclusive and are selected at server launch by connector name. Available with `vllm>=v0.24.0`.

## Push-based KV transfer

### Architecture overview

Push-based transfer improves TTFT and decode throughput for two reasons:

1. **It removes the serialization of the pull model** by overlapping producer and consumer work. The decode instance registers its target blocks in advance, while the producer is still computing KV. The transfer then begins as soon as the producer's blocks are ready, rather than after a consumer-side identifier-discovery round trip, removing that round trip from the path to first token.
2. **It moves the transfer-setup cost off the decode instance.** That setup — preparing the NIXL descriptors and submitting the transfer — runs on the prefill side and on a background thread, so the decode instance does not spend those cycles during its latency-critical token-generation phase.

The net effect is lower TTFT and higher output-token throughput on the decode instance.

The push connector coordinates three actors: the decode instance, which registers its target blocks up front; the prefill instance, which matches each request and issues the transfer; and a dedicated background thread that carries the KV write off the engine's forward-pass loop. The diagram below shows how these pieces fit together.

<figure>
  <img src="/assets/figures/2026-09-21-nixl-push-connector/push-architecture.png" alt="Push architecture: the router issues the prefill and decode legs simultaneously, D registers its blocks with P, and P pushes KV to D with a NIXL WRITE." style="width: 100%;">
  <figcaption><em>The router issues both legs at once. D allocates and registers its target blocks with P while P is still computing KV; P finishes prefill and pushes the blocks with a NIXL WRITE plus a completion notification.</em></figcaption>
</figure>

### Threading model

On the prefill instance, the push-path transfer setup — connection setup, descriptor preparation, and transfer submission — runs on a dedicated background thread, one per worker, kept off the main engine loop. This is the key design point: that setup would otherwise run synchronously with respect to the engine loop. The main thread runs the model forward pass and drives the engine's execution loop, so it sits on the latency-critical path. This transfer setup is non-trivial per-request work: it maps the request's KV block IDs to NIXL memory descriptors — merging contiguous blocks into fewer, larger regions to keep the descriptor count down — and then prepares and submits the WRITE. Doing it inline would steal cycles from token generation and inflate per-step latency. Offloading the transfer setup to the background thread frees the engine loop for model execution while the transfer proceeds concurrently.

The diagram below traces a single request through the push path — registration from the decode instance, rendezvous on the prefill instance, and the background-thread NIXL WRITE — and shows which steps stay off the main engine loop.

<figure>
  <img src="/assets/figures/2026-09-21-nixl-push-connector/push-request-flow.png" alt="Sequence diagram of one request through the push path, across the client, router, D scheduler, D worker, D writer, P writer, P worker, and P scheduler." style="width: 100%;">
  <figcaption><em>One request end to end. The dedicated writer threads (D Writer, P Writer) carry registration and the NIXL WRITE, so the main worker threads on both sides stay on model execution.</em></figcaption>
</figure>

### Implementation details

The protocol has three logical steps.

**1. Registration (decode side).** When the decode instance allocates local blocks for a request that will be prefilled remotely, it prepares a registration describing where the KV should be written: its own identity and the block locations it has just reserved. Within the same engine step, the worker transmits this registration to the prefill instance as a NIXL notification. Because registration happens at allocation time, the destination is known before prefill completes.

**2. Rendezvous (prefill side).** The registration from the decoder and the completion of prefill on the producer can arrive in either order, and the connector handles both. If the registration arrives first, the producer holds it until its own prefill finishes. If prefill finishes first, the producer holds the request's blocks until the registration arrives. Whichever event happens second triggers the transfer. Requests are paired by identifier, and the matching accounts for reissued requests, so preempted or retried requests are still associated correctly.

**3. Transfer (prefill side).** Once a request is matched, the producer establishes a connection to the decode instance (if one does not already exist) and issues a NIXL WRITE that copies the KV blocks directly into the decoder's registered memory, addressing each destination tensor-parallel rank. The producer and consumer may use different internal block layouts. Each side maps block locations to its own physical layout at transfer time, so the two need not agree on a common block size. Each WRITE carries a completion notification that NIXL delivers to the decode instance once the data has landed. The decoder counts these notifications, one per producer rank that writes into it, and treats the KV as received only after all expected notifications have arrived, at which point the request proceeds to decode.

### Timing comparison: pull vs push

In pull mode the stages run one after another: the router relays the prefill instance's `kv_transfer_params` to the decode instance, which then allocates blocks, configures NIXL, and issues a NIXL READ. Only then does decoding begin. In push mode the decode instance registers its blocks up front, overlapping the prefill computation, and the prefill instance writes the KV as soon as it is ready, so the relay, allocation, and NIXL configuration no longer sit between prefill and the first token.

The following diagram contrasts the pull and push timelines from prompt arrival to the first token.

<figure>
  <img src="/assets/figures/2026-09-21-nixl-push-connector/timing-pull-vs-push.png" alt="Timing comparison of pull and push modes on the path to first token." style="width: 100%;">
  <figcaption><em>Pull mode (top) versus push mode (bottom). In push mode D's alloc and register overlap P's compute, and the router is off the critical path.</em></figcaption>
</figure>

- **Pull:** TTFT = prefill compute (t0 -> t1) + router relay (t1 -> t2) + decode-side alloc and NIXL READ (t2 -> t3) + decode time.
- **Push:** the decode instance's alloc and register overlap with the prefill instance's compute. Once the prefill instance finishes, it pushes immediately, so TTFT = prefill compute (t0 -> t1) + NIXL WRITE (t1 -> t2) + decode time. The router relay and the decode-side alloc are off the critical path.

This comparison centers on TTFT, but the two modes trade off in other ways. Pull mode defers decode-instance selection until prefill finishes, so the router can steer each request to a decode instance that is not already congested — useful for balancing load and preserving quality of service. Push mode commits to a decode instance earlier; in return, under equal load it pins the KV blocks on the prefill instance for a shorter window, because the transfer starts as soon as those blocks are ready rather than after a decode-side round trip.

## Performance data

This chart compares time-to-first-token (TTFT) for push and pull KV transfer on Qwen3-32B, run on an AWS p5en cluster with EFA. We sweep input lengths 1k–16k and request rates 1–8 QPS. At low load and short prompts, the two modes perform almost identically. As QPS and input length grow, pull mode degrades sharply, reaching about 1,744 ms at 8 QPS and 16k. Push mode stays far lower at roughly 540 ms, around a 3x reduction. The gap opens because pull mode posts NIXL descriptors inline on the decode worker, synchronously within its engine loop, so that work stalls token generation. Push mode moves it to a background thread, off the critical path.

<figure>
  <img src="/assets/figures/2026-09-21-nixl-push-connector/ttft-push-vs-pull.png" alt="TTFT versus input length at 1, 2, 4, and 8 QPS for push and pull KV transfer on Qwen3-32B." style="width: 100%;">
  <figcaption><em>TTFT versus input prompt length at 1/2/4/8 QPS, Qwen3-32B on an AWS p5en cluster with EFA, 128 output tokens.</em></figcaption>
</figure>

This chart shows TPOT under the same conditions. The pattern mirrors TTFT. At light load, push and pull are close, around 5.6–5.8 ms. As load and input length rise, pull climbs to about 10.4 ms at 4–8 QPS and 16k, while push holds near 7 ms. The reason is that push offloads descriptor preparation and transfer submission from the decode worker. Token generation is never starved, so per-token latency stays flat even at high concurrency and long contexts.

<figure>
  <img src="/assets/figures/2026-09-21-nixl-push-connector/tpot-push-vs-pull.png" alt="TPOT versus input length at 1, 2, 4, and 8 QPS for push and pull KV transfer on Qwen3-32B." style="width: 100%;">
  <figcaption><em>TPOT versus input prompt length at 1/2/4/8 QPS, same deployment and workload.</em></figcaption>
</figure>

## Conclusion

Although push mode improves performance for most use cases, it isn't the right fit for every workload. Some cases still depend on pull mode: for example, bidirectional KV transfer, where the prefill node reads KV from the decode node, or setups where the prefill node reads KV from external storage. Pull mode also defers decode-instance selection until prefill completes, so it can route each request to a decode instance that is not congested and better preserve quality of service; under equal load, by contrast, push mode keeps the prefill node's blocks pinned for a shorter time. Our goal is therefore to support both push-based and pull-based KV connectors and let users choose the one that suits their deployment. As a possible next step, we are exploring bringing the background NIXL threading model to the pull-mode connector, moving NIXL setup off the critical path so that pull-mode workloads could benefit from the same optimizations; we have opened a pull request ([#45211](https://github.com/vllm-project/vllm/pull/45211)) to explore this.

## Appendix: Reproducing our results

The topology is two p5en GPU nodes (prefill on `:8100`, decode on `:8200`) with EFA enabled, plus a third smaller node that runs the proxy and the benchmark client. Three scripts ship with this post: [`serve_pd.sh`](/assets/repro/2026-09-21-nixl-push-connector/serve_pd.sh) launches either leg in either mode, [`run_sweep.sh`](/assets/repro/2026-09-21-nixl-push-connector/run_sweep.sh) drives the sweep, and [`cleanup.sh`](/assets/repro/2026-09-21-nixl-push-connector/cleanup.sh) tears everything down. [`plot_push_vs_pull.py`](/assets/repro/2026-09-21-nixl-push-connector/plot_push_vs_pull.py) redraws the two figures above from the result JSONs.

### 1. Provision the nodes

Install the EFA userspace stack on both GPU nodes and confirm the provider is present:

```bash
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
tar -xf aws-efa-installer-latest.tar.gz && cd aws-efa-installer
sudo ./efa_installer.sh -y
fi_info -p efa            # verify the EFA provider is present
```

The two GPU nodes must reach each other on `8100`/`8200` and on the NIXL side channel (`5600`).

### 2. Install vLLM and NIXL

On both GPU nodes:

```bash
python3.12 -m venv ~/vllm_env && source ~/vllm_env/bin/activate
pip install --upgrade pip
pip install vllm==0.29.0      # any build that includes the NIXL push connector
pip install nixl==1.4.1       # NIXL with the libfabric/EFA backend
```

On the proxy node (no GPU runtime needed):

```bash
pip install vllm aiohttp fastapi uvicorn pandas datasets
```

### 3. Download the model

```bash
export HF_HOME=/shared/huggingface     # a large shared volume
hf download Qwen/Qwen3-32B
```

### 4. Launch the P/D pair

Run `serve_pd.sh` once per GPU node. The only difference between the two modes is `kv_connector`: `NixlConnector` for pull, `NixlPushConnector` for push. Ports, engine IDs, roles, and ranks stay the same.

```bash
# on the prefill node
PREFILL_IP=<PREFILL_IP> DECODE_IP=<DECODE_IP> ROLE=prefill MODE=pull ./serve_pd.sh
# on the decode node
PREFILL_IP=<PREFILL_IP> DECODE_IP=<DECODE_IP> ROLE=decode  MODE=pull ./serve_pd.sh
```

Which expands to, on the prefill side:

```bash
vllm serve Qwen/Qwen3-32B --served-model-name Qwen --dtype bfloat16 \
  --tensor-parallel-size 8 --max-model-len 32768 --max-num-seqs 8 --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_buffer_device":"cuda",
    "kv_role":"kv_producer","engine_id":"prefill-engine-001","kv_rank":0,
    "kv_parallel_size":2,"kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'
```

The decode server adds `--no-enable-prefix-caching`, which forces every request to fetch full KV from prefill — exactly what we are measuring. With caching on, overlapping prompts could hit the decode node's local cache and skip the P->D transfer, giving misleading numbers. Wait for both `/health` endpoints to return 200 before benchmarking.

### 5. Run the sweep

The benchmark script lives in [#57567](https://github.com/vllm-project/vllm/pull/57567); it picks the proxy that matches the mode (`disagg_proxy_demo.py` for pull, `disagg_proxy_pushconnector_demo.py` for push) and drives it with `vllm bench serve`.

```bash
git clone https://github.com/vllm-project/vllm.git ~/vllm && cd ~/vllm
git fetch origin pull/57567/head:pd-bench && git checkout pd-bench
export VLLM_SRC=~/vllm

VLLM_SRC=~/vllm PREFILL_IP=<PREFILL_IP> DECODE_IP=<DECODE_IP> \
  MODE=pull ./run_sweep.sh
```

Then stop both servers, relaunch step 4 with `MODE=push` on each node, and rerun the sweep with `MODE=push`. Push mode additionally needs the prefill instance's NIXL coordinates (`PREFILL_ENGINE_ID`, `PREFILL_KV_HOST`, `PREFILL_SIDE_CHANNEL_PORT`, `PREFILL_TP_SIZE`, `PREFILL_PP_SIZE`) so the proxy can tell the decode node where to register; `run_sweep.sh` fills these in from the defaults used above. Keep `QPS_LIST`, `INPUT_LENS`, and `OUTPUT_LENS` identical across the two runs, and make sure `PREFILL_TP_SIZE` matches the prefill server's actual tensor-parallel size.

`results/` holds `<mode>_qps<q>_in<in>_out<out>_iter<i>.json` (plus `.log`) with per-run TTFT, ITL, throughput, and end-to-end latency. To redraw the figures:

```bash
python plot_push_vs_pull.py --results-dir ./results --metric ttft
python plot_push_vs_pull.py --results-dir ./results --metric tpot
```

### Troubleshooting

- **Server crashes right after the first request** — the mode doesn't match the connector. Pull needs `NixlConnector`; push needs `NixlPushConnector`, on both P and D.
- **`Backend 404: model ... does not exist`** — `SERVED_MODEL_NAME` doesn't match the servers' `--served-model-name` (`Qwen`).
- **Proxy readiness times out** — confirm the proxy for the chosen mode exists under `$VLLM_SRC/examples/disaggregated/disaggregated_serving/`, and that both P and D `/health` return 200.
- **NIXL handshake or transfer errors** — verify EFA (`fi_info -p efa`), that the side-channel port (`5600`) is open between P and D, and that `PREFILL_TP_SIZE` matches the prefill server's tensor-parallel size.

### Cleanup

```bash
./cleanup.sh proxy      # on the proxy / benchmark driver node
./cleanup.sh server     # on each GPU node
```

Then follow the AWS EC2 user guide to terminate the instances.

## Related resources

- [vLLM disaggregated serving guide](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)
- [AWS EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)

## Acknowledgements

We thank the vLLM community for their continued support in reviewing and merging this feature.
