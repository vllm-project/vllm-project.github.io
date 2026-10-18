---
layout: post
title: "Improve TTFT for Multi-Turn Agentic Workloads with Bi-Directional KV Transfer in vLLM"
author: "Sunita Nadampalli (AWS)"
summary: "Bi-directional KV transfer lets vLLM prefill nodes reuse previously computed KV from decode nodes on later conversation turns, cutting redundant prefill recompute and reducing TTFT by up to 3.3x on long multi-turn prompts."
image: /assets/figures/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload/ttft-glm-5-fp8.png
social_image: /assets/figures/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload/ttft-glm-5-fp8.png
tags:
  - disaggregation
  - kv_cache
  - nixl
  - performance
---

Multi-turn agentic workloads build on accumulated context as an agent reasons, invokes tools, and incorporates earlier results. With each new turn, the client resends the conversation history, including system instructions, prior prompts, tool results, and model responses, causing the growing context to be processed repeatedly. In [Prefill-decode (P-D) disaggregation](https://docs.vllm.ai/en/latest/features/disagg_prefill/), each request is split across specialized nodes: the prefill node processes the prompt and builds its KV cache, then the decode node retrieves those KV blocks and generates the response.

Conventional disaggregated serving, however, only transfers KV cache in one direction: from prefill to decode node. This one-directional flow creates two inefficiencies for multi-turn workloads. First, after generating a response, the decode node holds KV cache entries for both the prompt and the newly generated tokens. When processing the next turn, the prefill node cannot retrieve those entries, so it must recompute the KV projections for the previous response tokens. Second, the decode node maintains the session state while the prefill nodes do not, increasing the probability of cache eviction for active sessions on the prefill nodes. On a full cache miss, the prefill node must recompute KV projections for the entire conversation history, including system instructions, prior user requests, tool results, and model-generated responses. As the conversation grows, this redundant work consumes more accelerator capacity and increases time to first token (TTFT).

For customers building conversational assistants and agentic applications, longer interaction histories can increase latency and consume accelerator capacity through repeated prefill computation. In this post, we show how **bi-directional KV cache transfer** enables prefill and decode nodes to reuse previously computed context across turns, improving accelerator utilization and helping applications sustain responsive TTFT as conversations grow.

In our multi-turn benchmarks on a 1P1D deployment using Amazon EC2 P5en instances connected with AWS Elastic Fabric Adapter ([AWS EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html), the low latency RDMA network used across servers in AWS), bi-directional KV transfer reduced TTFT by up to 1.6x for Qwen3-32B and up to 3.3x for zai-org/GLM-5-FP8 (see the [Performance data](#performance-data) section); results depend on workload and configuration.

We originally built and validated the feature on AWS Trainium instance clusters connected with EFA and contributed it to vLLM as an accelerator-agnostic capability. The results presented here are from an Amazon EC2 P5en GPU cluster with EFA, showing that the benefits also carry over to GPUs. This post explains the bi-directional transfer architecture, router metadata flow, KV-block retention and recompute controls, performance results, and how to reproduce the benchmark.

## Bi-directional KV transfer

Bi-directional KV transfer is a mechanism that allows a decode node to return previously computed KV to a prefill node on subsequent turns of a conversation, avoiding recomputation of shared context on the prefill node, controlled by a configurable KV recompute threshold. This mitigates both computational-redundancy scenarios described earlier, and realizing this requires the KV to survive past the turn that produced it and to travel in the reverse direction, coordinated by a KV-cache-aware router.

### NIXL KV connector enhancements for bi-directional transfer

We extended the NVIDIA Inference Xfer Library (NIXL) [KV connector](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/) to let prefill nodes consume remote KV blocks, similar to how decode nodes already consume KV blocks that the prefill node provides. This is not a trivial change, because the two node types are architecturally different: prefill nodes compute KV representations for new prompt tokens and can provide complete prompt KV blocks. Decode nodes lack the capability (or efficiency) to generate complete KV representations and can only serve previously cached KV blocks. As a result, prefill and decode nodes still carry their own roles and logic for computing the metadata of remote and local KV blocks. The actual KV transfer mechanism (a NIXL READ), however, is common to both the `P->D` and `D->P` directions.

The following diagram first shows the existing flow in blue: the router sends the request to prefill node, prefill node computes the prompt KV and returns its block metadata, and the router forwards the request and metadata to decode node for the standard P→D NIXL READ. The numbered red path illustrates the bi-directional KV transfer additions: (1) decode node retains eligible KV blocks after generation, (2) returns their metadata in the streaming response, and (3) the router caches that metadata for the next prefill request. Prefill node then (4) checks the remotely reusable KV after it applies local cache matches, (5) queues an asynchronous receive when the reusable token count meets the threshold, and (6) performs a D→P NIXL READ before computing the remaining suffix.

<figure>
  <img src="/assets/figures/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload/kv-connector-enhancement-blocks.png" alt="Architecture diagram of bidirectional KV transfer. Existing behavior is shown in blue: the client sends a request to the KV-cache-aware router, the router sends it to the prefill node, receives prefill metadata, and forwards the request and metadata to the decode node, which reads the newly computed prompt KV with the existing P-to-D NIXL READ. New behavior is shown as six numbered red steps: (1) the decode node retains completed prompt and response KV until decoder_kv_blocks_ttl, (2) returns decode metadata in the terminal response chunk, (3) the router caches it by conversation_id and attaches it to the next prefill request, (4) the prefill node checks remote reuse after its local prefix-cache lookup, (5) queues an asynchronous remote receive when the reusable remainder meets kv_recompute_threshold, and (6) loads the retained KV from the decode node with a new D-to-P NIXL READ and computes the remaining suffix locally. The router carries metadata only; KV tensors move peer-to-peer." style="width: 100%;">
</figure>

### KV-cache-aware router

vLLM includes connector-side support for bi-directional KV transfer, while coordinating KV reuse across conversation turns requires a KV-cache-aware router. vLLM GitHub provides an [example proxy](https://github.com/vllm-project/vllm/blob/main/examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py) that implements the flow used in this post which can be adapted for your environment.

The KV-cache-aware router is a lightweight, stateful control-plane component between the client, prefill, and decode nodes. It maintains an in-memory mapping from the client-provided `conversation_id` to decode's `kv_transfer_params`. The router stores only block-location metadata, not KV tensors, and obtains that metadata from the response stream rather than separately querying decode node.

The following diagram shows how the router manages KV metadata across conversation turns. On the first turn, the cache lookup misses and the request follows the standard P→D flow; the router then saves decode's metadata from the response stream. On later turns, the router retrieves that metadata using the `conversation_id`, enabling prefill to reuse KV from decode, and refreshes the metadata after generation for the next turn.

<figure>
  <img src="/assets/figures/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload/router-conversation-kv-flow.png" alt="Two stacked sequence diagrams focused on the KV-cache-aware router. On turn one, the router records a cache miss, sends the request to Prefill, receives Prefill metadata, sends Decode the request and Prefill metadata, saves Decode metadata from the stream, and returns the response. On a later turn, the router pops saved Decode metadata, sends it to Prefill, receives updated Prefill metadata, sends Decode the request and Prefill metadata, refreshes the Decode metadata entry, and returns the response. Purple arrows show the Decode-to-Prefill and Prefill-to-Decode KV reads; KV tensors do not pass through the router." style="width: 100%;">
</figure>

### Block retention TTL

Normally a decode instance frees a request's blocks as soon as generation finishes. Under bi-directional mode it holds them for a configurable lifetime (`decoder_kv_blocks_ttl`, default 480 seconds) and publishes their location and expiry, so the conversation's KV, both prompt prefix and generated response, stays available for the next turn. For a turn that arrives before the TTL expires, prefill node reuses the KV; one that arrives later falls back to normal prefill. Because this TTL is fixed and not renewed, tune it to the expected time between turns to balance KV reuse against accelerator-memory usage. A natural next improvement is to extend the heartbeat-renewed lease to the decode side, so a decode instance reclaims blocks as soon as a conversation goes idle rather than at the end of a fixed timer.

### KV recompute threshold

KV reuse is not always faster than local recomputation. For a short reusable prefix, interconnect and coordination overhead can exceed the time needed to recompute the tokens. The `kv_recompute_threshold` setting, which defaults to 64 tokens, controls this decision. Prefill node recomputes when the token count to compute is below the threshold and pulls KV from decode node when it is at or above the threshold. Deployments can tune this value based on model computation time, KV size, and interconnect performance.

## Performance data

For evaluation, we used Qwen3-32B and GLM-5-FP8 models on a 1P1D deployment using Amazon EC2 P5en instances connected with EFA. To emulate scenarios in which the prefill node has evicted its local cache or the router selects a different prefill node that does not have the conversation's KV blocks, we disabled prefix caching on prefill node with `--no-enable-prefix-caching`.

Across both the models, turning on bi-directional KV transfer reduces prefill computation on multi-turn conversations, and the benefit typically grows with prompt length. With the feature OFF, the prefill instance recomputes the entire grown context each turn, so TTFT rises steeply: Qwen3-32B climbs from 158 ms at 2k to 778 ms at 24k, and GLM-5-FP8 from 500 ms at 2k to 2,334 ms at 20k. With it ON, the decode instance's retained KV is reused instead of recomputed, keeping TTFT far flatter: Qwen3-32B rises only from 122 ms to 460 ms over the same range (about a 1.6x reduction at the longest prompt), while GLM-5-FP8 rises from 358 ms to 702 ms and runs up to 3.3x below the recompute path at long prompts. The prefill-duration breakdown explains why.

Bi-directional transfer adds a small KV-transfer time (about 30 ms to 130 ms for Qwen3-32B; about 29 ms to 110 ms for GLM-5-FP8), but the prefill compute it removes far outweighs that KV-transfer time. As a result, the total prefill duration with transfer ON (Qwen3-32B: 128 ms to 414 ms; GLM-5-FP8: 316 ms to 541 ms) stays well below the recompute-only path (Qwen3-32B: 153 ms to 744 ms; GLM-5-FP8: 500 ms to 2,331 ms). The larger GLM-5-FP8 model shows a bigger absolute and relative gap, since its higher per-token recompute time makes KV reuse pay off even more as models and contexts scale.

### Qwen3-32B

The chart below plots TTFT against prompt length for Qwen3-32B, comparing bi-directional KV transfer ON versus OFF.

<figure>
  <img src="/assets/figures/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload/ttft-qwen3-32b.png" alt="TTFT versus input prompt length for Qwen3-32B with bi-directional KV transfer ON versus OFF." style="width: 100%;">
</figure>

The next chart breaks the prefill duration for the same runs into its recompute and KV-transfer components.

<figure>
  <img src="/assets/figures/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload/prefill-breakdown-qwen3-32b.png" alt="Prefill duration for Qwen3-32B split into prefill compute and KV transfer, ON versus OFF at each prompt length." style="width: 100%;">
</figure>

Together the charts show TTFT staying nearly flat with the feature ON while the OFF path climbs steeply as context grows, because the small KV-transfer time is far outweighed by the recompute it removes.

### zai-org/GLM-5-FP8

The same two views follow for GLM-5-FP8. The first plots TTFT against prompt length with the feature ON versus OFF.

<figure>
  <img src="/assets/figures/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload/ttft-glm-5-fp8.png" alt="Line plot for zai-org/GLM-5-FP8 on Amazon EC2 P5en instances with AWS EFA, comparing time to first token with bi-directional KV transfer enabled and disabled across input prompt lengths." style="width: 100%;">
</figure>

The second breaks the prefill duration into its recompute and KV-transfer components.

<figure>
  <img src="/assets/figures/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload/prefill-breakdown-glm-5-fp8.png" alt="Grouped bar plot for zai-org/GLM-5-FP8 on Amazon EC2 P5en instances with AWS EFA, comparing prefill duration with bi-directional KV transfer enabled and disabled and showing the KV-transfer component." style="width: 100%;">
</figure>

GLM-5-FP8 shows a much larger gap than Qwen3-32B, up to about 3.3x lower TTFT at long prompts, because its higher per-token recompute time makes KV reuse pay off even more as context length grows.

## How to run benchmark

To reproduce our 1P1D benchmark configuration, one prefill node and one decode node, using Amazon EC2 P5en instances with EFA, follow the step-by-step instructions in the [Reproducing our results](#appendix-reproducing-our-results) guide below. The guide covers environment setup, launching the vLLM prefill and decode servers, running the benchmark, and collecting results. The feature can run on other supported configurations, but performance results will vary depending on the hardware, model, workload, and network configuration.

## Conclusion

In this post, we showed how bi-directional KV cache transfer can reduce TTFT as an agent's conversation history grows. In P-D disaggregated deployments, the feature allows prefill node to reuse KV blocks retained by decode node across turns, reducing repeated computation. Configurable block-retention and recompute thresholds determine when reuse occurs. On a 1P1D deployment using Amazon EC2 P5en instances connected with EFA, at the longest prompt lengths tested, we observed TTFT reductions of 1.6x for Qwen3-32B and 3.3x for zai-org/GLM-5-FP8; results depend on the workload and deployment configuration. To implement this in your environment, use vLLM v0.21.0 or later with `NixlConnector`, enable `bidirectional_kv_xfer` on both prefill and decode servers, use a KV-cache-aware router, and follow the [reproduction guide](#appendix-reproducing-our-results).

## Related resources

- [AWS EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [Prefill-Decode (P-D) disaggregated serving](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [Benchmark KV Cache Offloading with Multi-Turn Conversations](https://github.com/vllm-project/vllm/tree/main/benchmarks/multi_turn)

## Acknowledgments

We thank Nicolò Lucchesi (Mistral), Mark McLoughlin (Red Hat), and the vLLM community for their support in maintaining this feature.

## Appendix: Reproducing our results

The topology is two AWS p5en instances (prefill on `:8100`, decode on `:8200`) with EFA enabled, plus a third small node running the proxy and the benchmark client on `:8000`.

Three scripts ship with this post: [`serve_pd_multiturn.sh`](/assets/repro/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload/serve_pd_multiturn.sh) launches either leg with the feature on or off, [`run_multiturn_sweep.sh`](/assets/repro/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload/run_multiturn_sweep.sh) drives the prompt-length sweep, and [`cleanup.sh`](/assets/repro/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload/cleanup.sh) tears the deployment down.

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
fi_info -p efa -t FI_EP_RDM            # verify the EFA provider is present on both GPU nodes
```

### 2. Install vLLM and NIXL on both GPU nodes

```bash
python3 -m venv ~/venv-pd && source ~/venv-pd/bin/activate
pip install --upgrade pip && pip install vllm==0.29.0 nixl==1.4.1
```

Apply vLLM PR [#40186](https://github.com/vllm-project/vllm/pull/40186) once on both nodes. It attaches `kv_transfer_params` to *streaming* responses, which is what the multi-turn proxy needs to link turns: the proxy always queries the decode node with `stream=True`, so without this patch the decode node never returns the block metadata and every turn lands as a cache miss.

```bash
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
curl -sL https://github.com/vllm-project/vllm/pull/40186.diff -o /tmp/40186.diff
patch -p1 -d "$SITE" --fuzz=3 < /tmp/40186.diff
```

### 3. Download the model

```bash
export HF_HOME=/fsx/huggingface        # a large shared volume
hf download Qwen/Qwen3-32B
```

### 4. Launch the P/D pair

Run `serve_pd_multiturn.sh` once per GPU node. `BIDIR` is the only thing that changes between the two legs of the A/B, and it must match on both nodes:

```bash
# on the prefill node
P_IP=<P_IP> D_IP=<D_IP> ROLE=prefill BIDIR=false ./serve_pd_multiturn.sh
# on the decode node
P_IP=<P_IP> D_IP=<D_IP> ROLE=decode  BIDIR=false ./serve_pd_multiturn.sh
```

Which expands to, on the prefill side:

```bash
export FI_PROVIDER=efa
export VLLM_NIXL_SIDE_CHANNEL_HOST=<P_IP> VLLM_NIXL_SIDE_CHANNEL_PORT=5600
vllm serve Qwen/Qwen3-32B --served-model-name Qwen --dtype bfloat16 \
  --tensor-parallel-size 8 --max-model-len 32768 --max-num-seqs 8 \
  --no-enable-prefix-caching --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_buffer_device":"cuda",
    "kv_role":"kv_producer","engine_id":"prefill-engine-001","kv_rank":0,
    "kv_parallel_size":2,"kv_connector_extra_config":{"backends":["LIBFABRIC"],
    "kv_recompute_threshold":64,"decoder_kv_blocks_ttl":480,
    "bidirectional_kv_xfer":false}}'
```

`--no-enable-prefix-caching` is set on the prefill node only — that is the cache-evicted scenario the benchmark emulates. Leave prefix caching enabled on the decode node. Wait for both `/health` endpoints to return 200.

### 5. Start the multi-turn proxy

```bash
python3 -m venv ~/venv-proxy && source ~/venv-proxy/bin/activate
pip install fastapi uvicorn httpx pandas numpy aiohttp transformers tqdm
sudo dnf install -y jq || sudo apt-get install -y jq
git clone --depth 1 --branch v0.29.0 https://github.com/vllm-project/vllm.git ~/vllm_src
export VLLM_SRC=~/vllm_src

python3 $VLLM_SRC/examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py \
  --port 8000 --prefiller-host <P_IP> --prefiller-ports 8100 \
  --decoder-hosts <D_IP> --decoder-ports 8200 \
  2>&1 | tee ~/proxy_$(date +%Y%m%d_%H%M%S).log
```

The proxy takes hosts and ports, not URLs. Reuse is driven by the client sending `conversation_id`, so there is no flag to enable it here.

### 6. Run the OFF leg, then the ON leg

```bash
VLLM_SRC=~/vllm_src LEG=off ./run_multiturn_sweep.sh
```

For the ON leg, stop both servers and the proxy, relaunch step 4 with `BIDIR=true` on each node, restart the proxy (its conversation cache is in-process), then:

```bash
VLLM_SRC=~/vllm_src LEG=on ./run_multiturn_sweep.sh
```

Repeat both legs for GLM-5-FP8 with `MODEL=zai-org/GLM-5-FP8 SERVED_NAME=GLM`.

### 7. Collect the numbers

Three quantities, one per source:

- **TTFT**, from the benchmark's per-request stats JSON written by `--stats-json-output`. Each record carries `ttft_ms` alongside `input_num_tokens` and `conversation_id`. This is the y-axis of the TTFT charts.
- **Prefill duration**, from the proxy log, which reports the prefill leg of each request as `Prefill done in <N>ms`. This is the total height of each bar in the prefill-duration breakdown.
- **KV transfer time and volume**, from the prefill node's log, which reports `KV Transfer metrics:` lines with average and P90 transfer time, bytes per transfer, and throughput. This is the KV-xfer component of each ON bar.

NIXL telemetry is recorded by the side that initiates the transfer, so the D->P readback appears in the prefill node's log while the ordinary P->D pull appears in the decode node's log.

Average each quantity per prompt length, then plot: TTFT against prompt length as two lines (ON and OFF), and prefill duration as paired stacked bars per prompt length, with the ON bar split into prefill compute and KV transfer. The difference between the two views is the point — the transfer cost the bars expose is what the TTFT lines convert into a saving.

### Troubleshooting

- **The proxy always logs `cache MISS` in an ON run** — missing `--send-conversation-id`, PR #40186 not applied on both nodes, or `BIDIR=false` left on the decode node, in which case it returns no `kv_transfer_params` at all.
- **`Declining expired remote read` on the prefill node** — the decode node's blocks lapsed before the next turn arrived. Raise `decoder_kv_blocks_ttl` (default 480 s) and keep the proxy's cache TTL below it.
- **`fi_info -p efa` is empty or falls back to sockets** — the EFA provider is missing. Reinstall `aws-efa-installer` and confirm the instances were launched with EFA-enabled interfaces; without it NIXL either fails to initialize or quietly runs over TCP, which makes the transfer numbers meaningless.

### Cleanup

```bash
./cleanup.sh proxy      # on the proxy / benchmark driver node
./cleanup.sh server     # on each GPU node
```

Then terminate the instances and verify none remain running.
