---
layout: post
title: "Push-Based KV Transfer to Improve TTFT and TPOT for Disaggregated Inference in vLLM"
author: "Sunita Nadampalli (AWS), Nicolò Lucchesi (Mistral)"
summary: "A push-based NIXL connector for vLLM disaggregated serving: the prefill instance writes KV directly into decode memory over RDMA, taking the router relay, decode-side allocation, and transfer setup off the TTFT critical path."
image: /assets/figures/2026-10-12-nixl-push-connector/ttft-push-vs-pull.png
social_image: /assets/figures/2026-10-12-nixl-push-connector/ttft-push-vs-pull.png
tags:
  - disaggregation
  - kv_cache
  - nixl
  - performance
---

[vLLM](https://github.com/vllm-project/vllm) is an open-source inference and serving engine for large language models (LLMs). LLM inference has two distinct phases: prefill, which processes the input prompt and builds the key-value (KV) cache, and decode, which uses that cache to generate output one token at a time. These phases have different compute, memory, and scaling characteristics. [Prefill-decode (P-D) disaggregation](https://docs.vllm.ai/en/latest/features/disagg_prefill/) runs them on separate engine instances, allowing each phase to scale and be optimized independently. This can improve resource utilization, throughput, and latency under load. However, it also requires transferring the KV cache from the prefill instance to the decode instance before token generation can begin, making efficient KV transfer critical to time-to-first-token (TTFT).

One of the ways vLLM performs this transfer is through the NVIDIA Inference Xfer Library (NIXL) [connector](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/), which moves KV blocks over RDMA-capable interconnects such as AWS Elastic Fabric Adapter ([AWS EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html), the low latency RDMA network used across servers in AWS). On [supported instance types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types), AWS EFA supports GPUDirect RDMA and Neuron Direct RDMA, enabling NIXL to transfer KV blocks directly between GPU or Trainium device memory without staging through host memory. The default mode for NIXL connector is **pull-based**: After the prefill instance finishes computing the prompt KV cache, the router relays the source metadata, including the prefill block IDs, to the decode instance. The decode instance then allocates destination blocks, prepares the NIXL transfer, and initiates NIXL READ to pull the KV blocks from the prefill instance. Because these coordination and setup steps occur sequentially after prefill and before decoding can begin, they extend the TTFT critical path, as shown in the following timeline.

<figure>
  <img src="/assets/figures/2026-10-12-nixl-push-connector/pull-mode-timeline.png" alt="Pull-mode timeline: router relay, decode-side allocation, and NIXL READ run serially on the path to first token." style="width: 100%;">
</figure>

This post introduces a **push-based** NIXL connector that moves post-prefill transfer coordination and setup off the critical path. For customers running disaggregated prefill/decode inference with vLLM, this can improve application responsiveness by reducing TTFT and TPOT, particularly at high request rates.

In our benchmark on a 1P1D deployment using Amazon EC2 P5en instances connected with EFA, push mode delivered up to 3x better TTFT and up to 30% better time-per-output-token (TPOT) compared to pull mode (see the [Performance data](#performance-data) section) for Qwen3-32B model; results depend on workload and configuration.

We built and validated the feature on AWS Trainium instance clusters connected with EFA, and then we contributed it to vLLM as an accelerator-agnostic feature. The results here are from an AWS p5en (GPU) cluster with EFA, showing the gains carry over to GPUs. This post explains the push-based architecture, transfer flow, threading model, performance results, and how to reproduce the benchmark.

## Push-based KV transfer

### Architecture overview

Push-based transfer improves TTFT and decode throughput for two reasons:

1. **It removes the serialization of the pull model** by overlapping producer and consumer work. The decode instance registers its target blocks in advance while the producer is still computing KV. The transfer then begins as soon as the producer's blocks are ready, rather than after a consumer-side identifier-discovery round trip, removing that round trip from the path to first token.
2. **It moves the transfer-setup time off the decode instance.** That setup — preparing the NIXL descriptors and submitting the transfer — runs on the prefill side and on a background thread, so the decode instance does not spend those cycles during its latency-critical token-generation phase.

The net effect is lower TTFT and higher output-token throughput on the decode instance.

The push connector coordinates three actors: the decode instance, which registers its target blocks up front; the prefill instance, which matches each request and issues the transfer; and a dedicated background thread that carries the KV write off the engine's forward-pass loop. The following diagram shows how these pieces fit together.

<figure>
  <img src="/assets/figures/2026-10-12-nixl-push-connector/push-architecture.png" alt="Push-based KV transfer architecture overview: the decode instance registers its blocks with the prefill instance, which pushes KV to the decode instance with a NIXL WRITE." style="width: 100%;">
</figure>

### Implementation details

The protocol has three logical steps.

**1. Registration (decode side).** When the decode instance allocates local blocks for a request requiring remote prefilling, it prepares a registration describing where the prefill instance should write the KV: its own identity and the block locations it has just reserved. Within the same engine step, the worker transmits this registration to the prefill instance as a NIXL notification. Because registration happens at allocation time, the connector knows the destination blocks before the prefill instance completes prefill.

**2. Rendezvous (prefill side).** The registration from the decoder and the completion of prefill on the producer can arrive in either order, and the connector handles both. If the registration arrives first, the producer holds it until its own prefill finishes. If prefill finishes first, the producer holds the request's blocks until the registration arrives. Whichever event happens second triggers the transfer. The connector pairs requests by identifier, and the matching accounts for reissued requests, correctly associating preempted or retried requests.

**3. Transfer (prefill side).** Once a request is matched, the producer establishes a connection to the decode instance (if one does not already exist) and issues an NIXL WRITE that copies the KV blocks directly into the decoder's registered memory, addressing each destination tensor-parallel rank. The producer and consumer may use different internal block layouts. Each side maps block locations to its own physical layout at transfer time, allowing different block sizes on each side. Each WRITE carries a completion notification that NIXL delivers to the decode instance once the data arrives. The decoder counts these notifications, one per producer rank that writes into it, and treats the KV as received only after all expected notifications have arrived, at which point the request proceeds to decode.

### Threading model

The push connector separates model execution from KV-transfer preparation by using a dedicated background transfer thread on each prefill worker. At the start of a request, the decode main loop allocates its destination KV blocks and registers them with the prefill instance while the prefill main loop computes the prompt KV cache. After both the destination registration and prompt KV are ready, the prefill main loop hands the KV blocks to the background thread and quickly returns to model execution.

The background thread establishes or reuses the NIXL connection, maps the request's KV block IDs to memory descriptors, merges contiguous blocks into larger transfer regions, and submits the NIXL WRITE to the decode instance. When NIXL completes the transfer, it sends a completion notification to the decode instance, which then begins token generation. Keeping descriptor preparation and transfer submission off the latency-critical main loops prevents this per-request work from delaying model execution. The following diagram shows these main-loop and background-thread handoffs for a single request.

<figure>
  <img src="/assets/figures/2026-10-12-nixl-push-connector/push-request-flow.png" alt="Sequence diagram of one request through the push path, showing the main-loop and background-thread handoffs on the prefill and decode instances." style="width: 100%;">
</figure>

### Timing comparison: Pull vs push

The following diagram summarizes the pull and push transfer flows described in the preceding sections and highlights their effect on TTFT. Although both modes use the same NIXL and EFA/RDMA data path, pull performs the metadata relay and decode-side setup after prefill, while push overlaps decode-side setup with prefill. As a result, push removes these operations from the post-prefill critical path, allowing decoding to begin sooner.

<figure>
  <img src="/assets/figures/2026-10-12-nixl-push-connector/timing-pull-vs-push.png" alt="Timing comparison of pull and push modes on the path to first token." style="width: 100%;">
</figure>

The two modes also involve a routing tradeoff. Pull mode allows the router to select a decode instance after prefill completes, which provides greater flexibility for load-aware routing and quality-of-service management. Push mode selects the decode instance earlier so that destination setup can overlap with prefill. In return, under comparable load, the prefill instance can release its KV blocks sooner because the transfer starts as soon as both the prompt KV and destination registration are ready.

## Performance data

For evaluation, we used Qwen3-32B model on a 1P1D deployment using Amazon EC2 P5en instances connected with EFA. To isolate P→D transfer performance, we configured the benchmark to disable prefix caching with `--no-enable-prefix-caching`. This configuration makes the decode node retrieve the full KV cache from the prefill node for every request. Otherwise, overlapping prompts could reuse decode node's local cache and bypass the transfer, which would produce misleading comparison results.

The following chart compares TTFT for push and pull KV transfer. We sweep input lengths 1k–16k and request rates 1–8 QPS. At low load and short prompts, the two modes perform almost identically. As QPS and input length grow, pull mode degrades sharply, reaching about 1,744 ms at 8 QPS and 16k. Push mode stays far lower at roughly 540 ms, around 3x reduction. The gap opens because pull mode posts NIXL descriptors inline on the decode worker, synchronously within its engine loop, so that work stalls token generation. Push mode moves it to a background thread, off the critical path.

<figure>
  <img src="/assets/figures/2026-10-12-nixl-push-connector/ttft-push-vs-pull.png" alt="Four-panel TTFT comparison of push and pull KV transfer for Qwen3-32B on Amazon EC2 P5en instances with EFA. Test configuration: 1P1D topology, 128 output tokens, and prefix caching disabled on the Decode node." style="width: 100%;">
</figure>

This chart shows TPOT under the same conditions. The pattern mirrors TTFT. At light load, push and pull are close, around 5.6–5.8 ms. As load and input length rise, pull climbs to about 10.4 ms at 4–8 QPS and 16k, while push holds near 7 ms. The reason is that push offloads descriptor preparation and transfer submission from the decode worker. Token generation is never starved, so per-token latency stays flat even at high concurrency and long contexts.

<figure>
  <img src="/assets/figures/2026-10-12-nixl-push-connector/tpot-push-vs-pull.png" alt="Four-panel TPOT comparison of push and pull KV transfer for Qwen3-32B on Amazon EC2 P5en instances with EFA. Test configuration: 1P1D topology, 128 output tokens, and prefix caching disabled on the Decode node." style="width: 100%;">
</figure>

## How to run benchmark

To reproduce our 1P1D benchmark configuration, one prefill node and one decode node, using Amazon EC2 P5en instances connected with EFA, refer to the detailed instructions in the [Reproducing our results](#appendix-reproducing-our-results) guide. The guide covers environment setup, launching the vLLM prefill and decode servers, running the benchmark, and collecting results. The feature can run on other supported configurations, but performance results will vary depending on the hardware, model, workload, and network configuration.

## Conclusion

In this post, we showed how the push-based KV connector overlaps prefill computation with decode-side block allocation and registration, removing the post-prefill router round trip and decode-side transfer setup from the critical path. This reduces TTFT and TPOT, particularly at high request rates. In our Qwen3-32B benchmark on a 1P1D deployment using Amazon EC2 P5en instances connected with EFA, push mode provided up to a 3x improvement in TTFT and 30% improvement in TPOT compared with pull mode. Results vary by workload and configuration. To use push mode, configure both the prefill and decode servers to use `NixlPushConnector` instead of `NixlConnector`.

Choose the push-based connector when early decode node selection is acceptable and lower TTFT is the priority. The pull-based connector remains the better fit for workloads that require bidirectional KV transfer or defer decode-instance selection until after prefill for load-aware routing. vLLM supports both push- and pull-based KV connectors, allowing users to select the approach that best fits their deployment.

As a future improvement, we are exploring bringing the background NIXL threading model to the pull-mode connector, moving NIXL setup off the critical path so that pull-mode workloads could benefit from the same optimizations; we have opened a pull request ([#45211](https://github.com/vllm-project/vllm/pull/45211)) to explore this.

## Related resources

- [vLLM disaggregated serving guide](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)
- [AWS EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [Push-mode NIXL KV connector](https://docs.vllm.ai/en/latest/design/nixl_kv_push_connector/)

## Acknowledgements

We thank the vLLM community for their support in maintaining this feature.

## Appendix: Reproducing our results

The topology is two p5en GPU nodes (prefill on `:8100`, decode on `:8200`) with EFA enabled, plus a third smaller node that runs the proxy and the benchmark client. Three scripts ship with this post: [`serve_pd.sh`](/assets/repro/2026-10-12-nixl-push-connector/serve_pd.sh) launches either leg in either mode, [`run_sweep.sh`](/assets/repro/2026-10-12-nixl-push-connector/run_sweep.sh) drives the sweep, and [`cleanup.sh`](/assets/repro/2026-10-12-nixl-push-connector/cleanup.sh) tears everything down. [`plot_push_vs_pull.py`](/assets/repro/2026-10-12-nixl-push-connector/plot_push_vs_pull.py) redraws the two figures above from the result JSONs.

### Test environment

Both GPU nodes were configured identically:

| | |
|---|---|
| Instance type | `p5en.48xlarge` (8x NVIDIA H200), one for prefill and one for decode |
| Region / AZ | `ap-south-1` (Mumbai), `ap-south-1b` (AZ ID `aps1-az3`), both nodes in the same AZ |
| Placement | No cluster placement group; both instances launched into a targeted On-Demand Capacity Reservation, default tenancy |
| AMI | Public AWS ParallelCluster 3.14.2 image for Ubuntu 24.04 x86_64, `aws-parallelcluster-3.14.2-ubuntu-2404-lts-hvm-x86_64-202602121713` (`ami-0133f195033a4baae` in `ap-south-1`; AMI IDs differ by region). It ships kernel `6.14.0-1018-aws`, NVIDIA driver 570, and CUDA 12.8, so upgrade the driver, CUDA toolkit, and EFA installer to the versions below. |
| OS / kernel | Ubuntu 24.04.4 LTS, `6.14.0-1018-aws` |
| NVIDIA driver / CUDA | Driver `580.126.16` (CUDA 13.0 driver API); CUDA toolkit 13.0 |
| EFA interfaces | 16 per instance, one per network card (cards 0-15, all interface type `efa`), 200 Gb/s each for 3.2 Tb/s total. 8 interfaces sit on each NUMA node, and each GPU shares a PCIe switch with 2 of them. |
| EFA software | EFA installer `1.50.0`, EFA kernel module `3.3.0g`, libfabric `2.6.0amzn1.0`, rdma-core `64.0amzn0` |

**EFA device selection by NIXL.** The LIBFABRIC backend opens all 16 EFA devices as rails with the `efa` provider. It then registers each GPU's KV cache only on the 2 rails that share that GPU's PCIe switch, so every transfer stays PCIe-local. No NIXL or libfabric device overrides were set; this is the default selection. To confirm it on your own nodes, launch the servers with `NIXL_LOG_LEVEL=DEBUG` and check the registration lines:

```bash
grep -E "Created rail [0-9]+ \(device=|mapped to rail" server.log
```

### 1. Provision the nodes

Install the EFA userspace stack on both GPU nodes and confirm the provider is present. We used installer `1.50.0`:

```bash
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz
tar -xf aws-efa-installer-1.50.0.tar.gz && cd aws-efa-installer
sudo ./efa_installer.sh -y
fi_info --version         # libfabric 2.6.0amzn1.0
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
