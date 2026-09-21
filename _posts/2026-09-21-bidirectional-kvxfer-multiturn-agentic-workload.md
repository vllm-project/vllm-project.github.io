---
layout: post
title: "Bi-Directional KV Transfer for Multi-Turn Agentic Workloads in vLLM"
author: "Sunita Nadampalli (AWS) and the vLLM Team"
summary: "Bi-directional KV transfer lets vLLM prefill nodes reclaim previously computed KV from decode nodes on later conversation turns, cutting redundant prefill recompute and reducing TTFT by up to 3x on long multi-turn prompts."
image: /assets/figures/2026-09-21-bidirectional-kvxfer-multiturn-agentic-workload/ttft-glm52.png
social_image: /assets/figures/2026-09-21-bidirectional-kvxfer-multiturn-agentic-workload/ttft-glm52.png
tags:
  - disaggregation
  - kv_cache
  - nixl
  - performance
---

Multi-turn agentic workloads (or multi-turn conversations) are sequential dialogue exchanges between users and large language model (LLM) systems that maintain contextual coherence across interaction cycles. Implementing them requires persistent conversation-history management, where each subsequent user query is concatenated with the relevant historical context: system prompts, prior user queries, and model-generated responses.

In non-disaggregated deployments, the LLM inference server keeps sessions resident and reuses their KV cache across turns, which improves performance. Prefill-decode (P-D) disaggregated architectures, however, present a fundamental challenge on two fronts. First, the KV cache for model-generated responses from previous turns resides exclusively on the decode nodes and is inaccessible to the prefill nodes, so the current architecture forces prefill nodes to recompute KV projections for all response tokens on every turn, resulting in a substantial waste of AI-accelerator compute. Second, in a P-D disaggregated multi-turn system, the decode node maintains the session state while the prefill nodes do not, which raises the probability of cache eviction for active sessions on the prefill nodes. On a cache miss, the prefill node must then recompute KV projections for the entire conversational context (system prompts, historical user queries, and prior model responses).

In this post we cover how bi-directional KV cache transfer between prefill and decode nodes optimizes KV cache utilization in P-D disaggregated deployments and cuts redundant prefill recomputation on multi-turn conversations, reducing time to first token by up to ~3x on long multi-turn prompts (see the [Performance data](#performance-data) section). We built the feature on AWS Trainium instance clusters with AWS Elastic Fabric Adapter (EFA, the low latency RDMA network protocol used across all servers in AWS) and upstreamed it to vLLM as an accelerator-agnostic feature. The results here are from an AWS p5en (GPU) cluster with EFA, showing the gains carry over to GPUs.

**Bi-directional KV transfer ([#32553](https://github.com/vllm-project/vllm/pull/32553))**: a mechanism that allows a decode instance to return previously computed KV to a prefill instance on subsequent turns of a conversation, avoiding recomputation of shared context on the prefill node, governed by a cost-based recompute threshold. The feature is implemented in `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` and interoperates with vLLM's hybrid KV cache manager for sliding-window models. It does not modify the existing standard P->D transfer path. These are purely additive extensions that activate only when bi-directional KV transfer is enabled and the multi-turn reusable token count exceeds the user-set threshold. Available with `vllm>=v0.21.0`.

## Bi-directional KV transfer

Bi-directional KV cache transfer between prefill and decode nodes mitigates both computational-redundancy scenarios described earlier: nodes load KV cache from each other, which removes unnecessary recomputation. Realizing this requires the KV to survive past the turn that produced it and to travel in the reverse direction, coordinated by a KV-cache-aware router.

### NIXL KV connector enhancement for remote block handling

We extend the NIXL (NVIDIA Inference Xfer Library) KV connector to let prefill nodes consume remote KV blocks, similar to how decode nodes already consume the prefill node's shared KV blocks. This is not a trivial change, because the two node types are architecturally different: prefill nodes compute KV representations for new prompt tokens and can provide complete prompt KV blocks. Decode nodes lack the capability (or efficiency) to generate complete KV representations and can only serve previously cached KV blocks. As a result, prefill and decode nodes still carry their own roles and logic for computing the metadata of remote and local KV blocks. The actual KV transfer mechanism (a NIXL READ), however, is common to both the `P->D` and `D->P` directions.

<figure>
  <img src="/assets/figures/2026-09-21-bidirectional-kvxfer-multiturn-agentic-workload/kv-connector-remote-blocks.png" alt="KV connector components on the prefill and decode nodes, with KV loads in both directions and a KV-cache-aware router above them." style="width: 100%;">
  <figcaption><em>The prefill node gains a path to consume remote KV blocks, mirroring how the decode node already consumes the prefill node's shared KV blocks. Each node keeps its own logic for computing the metadata of local and remote blocks, while both the P->D and D->P transfers use the same NIXL READ mechanism. Each direction loads only the blocks that are not already present locally.</em></figcaption>
</figure>

### KV cache aware router

The KV-cache-aware router in a vLLM deployment is a lightweight, stateful component between the client, the prefill (P) node, and the decode (D) node. It maintains an in-memory cache of `kv_transfer_params` keyed by `conversation_id`, which the client carries across turns of the same conversation session. On the first turn of a conversation (cache miss), the router sends the request to P with empty `remote_block_ids`, so P computes the full KV cache from scratch. On subsequent turns (cache hit), the router looks up the cached block IDs from the previous turn and passes them to P. P then pulls the existing KV cache directly from D through remote direct memory access (RDMA) and computes only the new tokens. This keeps the router's role minimal: it never queries D for cache state, never holds KV data itself, and adds no extra hops to the critical path. The only state it tracks is a mapping from `conversation_id` to D's block metadata, which it captures for free from the token stream already flowing through it.

<figure>
  <img src="/assets/figures/2026-09-21-bidirectional-kvxfer-multiturn-agentic-workload/router-conversation-kv-flow.png" alt="Call sequence for turn 1 (cache miss, full prefill) and turn 2 onward (cache hit, prefill reads decode's KV over RDMA)." style="width: 100%;">
  <figcaption><em>The call sequence for both paths. On the first turn (cache miss) the router forwards the request to the prefill node with empty remote block IDs and the prefill node computes the full KV cache. On later turns (cache hit) the router hands the prefill node the block IDs it cached from the previous turn, and the prefill node pulls the retained KV from the decode node over RDMA and computes only the new tokens.</em></figcaption>
</figure>

### Block retention TTL

Normally a decode instance frees a request's blocks as soon as generation finishes. Under bi-directional mode it instead retains them for a configurable lifetime (`decoder_kv_blocks_ttl`, default 480 seconds) and publishes their location and expiry, so the conversation's KV, both the prompt prefix and the generated response, remains available for the next turn to claim. When that turn arrives, its prefill instance reads the retained blocks back from the decode instance (a decode-to-prefill transfer) and reuses them instead of recomputing the grown context. The retention window bounds how long the KV is held: a follow-up turn that arrives within it reuses the KV, while one that arrives later falls back to normal prefill.

### Cost-based recompute threshold

Reuse is not always the cheaper option. The reverse transfer incurs interconnect and coordination overhead, so for a short reusable prefix it can cost less to recompute the tokens than to move their KV across instances. The connector makes this trade-off explicit through a recompute threshold (`kv_recompute_threshold`, default 64 tokens). When the number of reusable tokens held on the decode instance is below the threshold, the prefill instance recomputes locally. When it is above, the prefill node pulls the KV from the decode instance. This keeps the transfer path active only when it is expected to pay off, and lets deployments tune the crossover point to their interconnect bandwidth and model compute cost.

## Performance data

Across both Qwen3-32B and GLM5.2, turning on bi-directional KV transfer sharply reduces prefill cost on multi-turn conversations, and the benefit grows with prompt length. With the feature OFF, the prefill instance recomputes the entire grown context each turn, so TTFT rises steeply: Qwen3-32B climbs from 158 ms at 2k to 778 ms at 24k, and GLM5.2 from 500 ms at 2k to 2,334 ms at 20k. With it ON, the decode instance's retained KV is reused instead of recomputed, keeping TTFT far flatter: Qwen3-32B rises only from 122 ms to 460 ms over the same range (about a 1.7x reduction at the longest prompt), while GLM5.2 rises from 358 ms to 702 ms and runs up to ~3x below the recompute path at long prompts.

The prefill-duration breakdown explains why. Bi-directional transfer adds a small KV-transfer cost (about 30 ms at 2k rising to ~130 ms for Qwen3-32B; about 29 ms to 110 ms for GLM5.2), but the prefill compute it removes far outweighs that cost. As a result, the total prefill duration with transfer ON (Qwen3-32B 128 ms to 414 ms; GLM5.2 316 ms to 541 ms) stays well below the recompute-only path (Qwen3-32B 153 ms to 744 ms; GLM5.2 500 ms to 2,331 ms). The larger GLM5.2 model shows a bigger absolute and relative gap, since its higher per-token recompute cost makes KV reuse pay off even more as models and contexts scale.

All runs use 10-turn conversations with 150-token user turns and 300-token responses, and prefix caching disabled on the prefill node to emulate cache-evicted prefill nodes.

### Qwen3-32B

<figure>
  <img src="/assets/figures/2026-09-21-bidirectional-kvxfer-multiturn-agentic-workload/ttft-qwen3-32b.png" alt="TTFT versus input prompt length for Qwen3-32B with bi-directional KV transfer ON versus OFF." style="width: 100%;">
  <figcaption><em>TTFT versus prompt length for Qwen3-32B, bi-directional KV transfer ON versus OFF.</em></figcaption>
</figure>

<figure>
  <img src="/assets/figures/2026-09-21-bidirectional-kvxfer-multiturn-agentic-workload/prefill-breakdown-qwen3-32b.png" alt="Prefill duration for Qwen3-32B split into prefill compute and KV transfer, ON versus OFF at each prompt length." style="width: 100%;">
  <figcaption><em>The same runs, with prefill duration split into its compute and KV-transfer components. The ON total is lower at every length despite a non-zero KV transfer.</em></figcaption>
</figure>

Together the charts show TTFT staying nearly flat with the feature ON while the OFF path climbs steeply as context grows, because the small KV-transfer cost is far outweighed by the recompute it removes.

### GLM5.2

<figure>
  <img src="/assets/figures/2026-09-21-bidirectional-kvxfer-multiturn-agentic-workload/ttft-glm52.png" alt="TTFT versus input prompt length for GLM5.2 with bi-directional KV transfer ON versus OFF." style="width: 100%;">
  <figcaption><em>TTFT versus prompt length for GLM5.2, bi-directional KV transfer ON versus OFF.</em></figcaption>
</figure>

<figure>
  <img src="/assets/figures/2026-09-21-bidirectional-kvxfer-multiturn-agentic-workload/prefill-breakdown-glm52.png" alt="Prefill duration for GLM5.2 split into prefill compute and KV transfer, ON versus OFF at each prompt length." style="width: 100%;">
  <figcaption><em>GLM5.2 prefill duration split into compute and KV transfer. At 20k the ON path costs 541 ms against 2,331 ms for recompute.</em></figcaption>
</figure>

GLM5.2 shows a much larger gap than Qwen3-32B — up to about 3x lower TTFT at long prompts — because its higher per-token recompute cost makes KV reuse pay off even more as context length grows.

## Conclusion

Bi-directional KV cache transfer removes redundant prefill recomputation in P-D disaggregated deployments by letting prefill nodes reclaim previously computed KV from decode nodes on later turns, governed by a block-retention window and a cost-based recompute threshold. On multi-turn workloads this keeps TTFT under SLA limits for a longer context length compared to without the feature. In our multi-turn benchmarks, this cut time to first token by up to about 1.7x for Qwen3-32B and about 3x for GLM5.2 at long prompts.

## Appendix: Reproducing our results

The topology is two AWS p5en instances (prefill on `:8100`, decode on `:8200`) with EFA enabled, plus a third small node running the proxy and the benchmark client on `:8000`.

Three scripts ship with this post: [`serve_pd_multiturn.sh`](/assets/repro/2026-09-21-bidirectional-kvxfer-multiturn-agentic-workload/serve_pd_multiturn.sh) launches either leg with the feature on or off, [`run_multiturn_sweep.sh`](/assets/repro/2026-09-21-bidirectional-kvxfer-multiturn-agentic-workload/run_multiturn_sweep.sh) drives the prompt-length sweep, and [`cleanup.sh`](/assets/repro/2026-09-21-bidirectional-kvxfer-multiturn-agentic-workload/cleanup.sh) tears the deployment down.

### 1. Provision the nodes

Install the EFA userspace stack on both GPU nodes:

```bash
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
tar -xf aws-efa-installer-latest.tar.gz && cd aws-efa-installer
sudo ./efa_installer.sh -y
fi_info -p efa            # verify the EFA provider is present on both GPU nodes
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
  2>&1 | tee ~/logs/proxy_$(date +%Y%m%d_%H%M%S).log
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

Repeat both legs for GLM5.2 with `MODEL=zai-org/GLM-5-FP8 SERVED_NAME=GLM`.

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

## Related resources

- [AWS EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [Benchmark KV cache offloading with multi-turn conversations](https://github.com/vllm-project/vllm/tree/main/benchmarks/multi_turn)
- [Bi-directional KV transfer pull request (#32553)](https://github.com/vllm-project/vllm/pull/32553)

## Acknowledgments

We thank Nicolò Lucchesi (Mistral AI), Mark McLoughlin (Red Hat), and the vLLM community for their continued support in reviewing and merging this feature.
