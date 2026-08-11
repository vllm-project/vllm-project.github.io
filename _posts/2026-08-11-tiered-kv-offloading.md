---
layout: post
title: "Tiered KV Cache Offloading in vLLM"
author: "Or Ozeri, Danny Harnik, Ronen Schaffer, Itay Etelis, Varun Sundar Rabindranath"
summary: "A host-centric framework for scaling KV cache across host memory, filesystems, object stores, and remote peers — reducing recomputation and increasing serving capacity."
image: /assets/figures/tiered-kv-offloading/architecture.svg
---

Long-context models and multi-turn conversations generate massive KV caches.
When accelerator memory (e.g., GPU HBM) fills up, previously computed KV data is evicted.
On the next request that needs it, vLLM must recompute it from scratch.

**Tiered KV cache offloading preserves evicted KV data** across host memory, storage, and remote peers.
Instead of recomputing, vLLM reloads the data from a lower tier — saving compute, reducing latency, and increasing the effective serving capacity of the cluster.

With secondary tiers, KV data also becomes **shareable across nodes** — enabling horizontal scaling of the cache, warm-starting new instances from shared storage, and transferring KV data between peers for disaggregated serving or load balancing.

The framework has been available in vLLM since v0.22.

---

## The Host-Centric Design

The core design principle: **all KV data flows through host memory (CPU DRAM)**.

When offloading, KV data moves from accelerator to host first.
From the host, it propagates to secondary tiers — filesystem, object storage, or remote peers.
When reloading, the flow reverses: a secondary tier promotes data into host memory, then it is loaded to the accelerator.

<p align="center">
<img src="/assets/figures/tiered-kv-offloading/architecture.svg" alt="Tiered KV Offloading Architecture" width="85%">
</p>

<figcaption style="text-align:center; font-style:italic; margin-bottom:1.5em;">
All KV data flows through the host primary tier. Secondary tiers extend capacity beyond what host DRAM can hold. On offload, chunks cascade to all tiers. On reload, the first tier that holds the chunk serves it.
</figcaption>

This design yields several key advantages:

### Fast accelerator release, just-in-time allocation

Copying from accelerator to host is a fast local PCIe transfer.
**Accelerator memory is freed as soon as this copy completes** — before any secondary tier transfer starts.
Storage writes, network sends, and remote RDMA all proceed from the host copy without touching accelerator memory again.
**On reload, accelerator memory is allocated only once the data is ready in the host** — not reserved in advance while waiting for tier transfers.
Together, these create a just-in-time allocation pattern: accelerator memory is held only while actively needed.

<p align="center">
<img src="/assets/figures/tiered-kv-offloading/offload-flow.svg" alt="Offload flow timeline showing early accelerator release" width="90%">
</p>

<figcaption style="text-align:center; font-style:italic; margin-bottom:1.5em;">
Accelerator memory is freed at t2 — as soon as the host copy completes. Secondary tier writes continue asynchronously from the host copy.
</figcaption>

### Consolidated I/O

In a multi-accelerator setup (e.g., `tensor_parallel_size=8`), each device holds a shard of the KV cache.
The framework **consolidates all shards into a single shared host memory region**.

<p align="center">
<img src="/assets/figures/tiered-kv-offloading/consolidated-io.svg" alt="Multiple accelerators consolidated into one host region" width="80%">
</p>

<figcaption style="text-align:center; font-style:italic; margin-bottom:1.5em;">
Multiple accelerator shards fan into one shared host region. Secondary tiers see fewer, larger I/O operations — improving storage and network throughput.
</figcaption>

### Canonical memory layout

The host region uses a canonical memory layout: a uniform, block-indexed representation where locating any KV chunk is a simple offset calculation.
This layout is the same regardless of the accelerator type, attention backend (FlashAttention, FlashInfer, Triton), or parallelism configuration (TP).
Because the layout is configuration-independent, **nodes with different setups share KV data directly** — no remapping or format conversion needed.
A TP=2 node and a TP=4 node produce identical host-side chunks for the same KV data.

### Simple secondary tiers

Routing all data through host memory makes secondary tiers **easy to build and operate**.
They are a single process per vLLM instance, they transfer data using standard CPU-based libraries (POSIX I/O, S3 SDKs, RDMA verbs), and they never touch accelerator memory or APIs.
No need to coordinate across multiple processes or understand accelerator-specific memory layouts.

---

## How Offloading and Reloading Work

The unit of operation is a **chunk** — a fixed-size piece of KV data covering a group of tokens.
By default, a chunk maps to a single accelerator block.
A configurable `blocks_per_chunk` parameter allows larger chunks, yielding larger I/Os to the host and secondary tiers.

### Offload path

New KV chunks move from accelerator to host via async DMA.
**Accelerator memory is freed immediately** — before any secondary tier transfer begins.
The tiering manager then cascades chunks to **all** configured secondary tiers simultaneously, reading from the host copy.

The host primary tier is a **proper LRU/ARC cache**, not a staging buffer.
Chunks remain in host memory and serve future hits directly.
Only when host capacity is exhausted are the least-recently-used chunks evicted — and even then, they survive in whichever secondary tier received them.

### Reload path

The scheduler checks the host cache first — if the chunk is there, it is an immediate hit.
On host miss, secondary tiers are queried in configured order; the first tier that holds the chunk serves it.
The tier promotes the chunk back into host memory asynchronously; during this time, the scheduler receives a `RETRY` and re-checks on the next cycle.

Different chunks within the same request can be served by different tiers — e.g., one chunk from the filesystem, another from a remote peer.

---

## Secondary Tiers

### Filesystem

Stores each KV chunk as a file on local or networked storage.
Uses content-addressed naming — identical token sequences map to the same key, so matching inputs share cached data automatically.

When multiple vLLM instances share the same storage mount point (e.g., network-attached storage, or multiple instances on the same node), they **share KV data automatically** with no additional configuration.

Highlights:
- **Non-blocking lookups**
- **Atomic writes**
- **Separate read/write thread pools**

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
    --kv-transfer-config '{
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": 107374182400,
            "secondary_tiers": [{"type": "fs", "root_dir": "/mnt/kv-cache"}]
        }
    }'
```

### Object Storage

Stores KV chunks in S3-compatible object stores via NIXL.
Same content-addressed scheme as the filesystem tier.
Provides a cost-effective networked storage option — typically cheaper per GB than high-performance file storage, while still enabling shared access across instances.

```bash
--kv-transfer-config '{
    "kv_connector_extra_config": {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": 107374182400,
        "secondary_tiers": [{
            "type": "obj",
            "bucket": "my-kv-cache",
            "endpoint_override": "http://minio:9000"
        }]
    }
}'
```

### Peer-to-Peer (P2P)

Enables **cross-instance KV cache sharing** over the network.
Uses ZMQ for coordination and RDMA (via NIXL) for bulk data transfer.
All transfers are **host-to-host** — no accelerator memory involved on either side.

P2P transfers are triggered via `kv_transfer_params` request headers, typically managed by an external orchestrator such as [llm-d](https://github.com/llm-d/llm-d).

```bash
--kv-transfer-config '{
    "kv_connector_extra_config": {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": 107374182400,
        "secondary_tiers": [{"type": "p2p", "host": "10.0.0.1", "port": 5710}]
    }
}'
```

Two key use-cases:

#### Prefill/Decode disaggregation

The prefill instance computes KV chunks and makes them available in its host tier.
The decode instance pulls them from the prefiller's host memory via RDMA.

A key advantage over GPU-based P/D approaches: **consolidated I/O turns many small per-GPU transfers into fewer, larger RDMA operations** — dramatically improving network throughput.
Additionally, with *chunked prefill*, each completed prefill-chunk becomes immediately available for transfer — computation and data movement overlap, reducing time-to-first-token.

#### Load balancing

Transfer KV chunks from an overloaded vLLM instance to one with available capacity.
Any node can pull chunks from any peer.

---

## Hybrid Model Support

The framework integrates with vLLM's hybrid memory allocator.
Models that combine different layer types — full attention, sliding window, MLA, Mamba — are handled transparently.

The canonical layout normalizes all KV formats into a uniform byte-buffer representation.
**Each chunk has a fixed byte size on the host**, regardless of which layer types it contains.
Different layer types pack different numbers of tokens into the same chunk — for example, Mamba state layers cover many more tokens per chunk than full-attention layers, so they are offloaded less frequently.

This means:
- **Sliding window layers** reload only the tokens within their window, not the full history
- **State-space layers** (Mamba) offload and reload their state alongside attention KV

The framework supports state-of-the-art hybrid architectures including DeepSeek V4 and GLM 5.2.

---

## Observability

The framework exposes Prometheus metrics via vLLM's standard `/metrics` endpoint:

- **Host cache utilization** — current fill ratio of the primary tier
- **Transfer throughput** — bytes and time for accelerator ↔ host transfers
- **Per-tier latencies** — how long lookups and data transfers take for each tier
- **Per-tier hit rates** — which tiers are serving your workload

Secondary tiers can **define custom metrics** (counters, histograms, gauges) that are automatically registered and exposed — no framework changes needed.

---

## KV Events

As chunks move between tiers, the framework emits structured **KV events** reporting which chunks were stored or evicted, from which tier, and with what locality (local vs. remote).
Secondary tiers can emit their own events.

These events enable external orchestration systems to make intelligent routing decisions.
Projects such as llm-d and [Dynamo](https://github.com/ai-dynamo/dynamo) consume KV events to **route requests to the instance most likely to have a cache hit** — achieving significantly higher throughput and lower latency compared to cache-unaware scheduling.
Additionally, llm-d uses these events to orchestrate P2P KV transfers between peers.

---

## Adding a New Secondary Tier

The secondary tier interface is minimal — four core methods:

```python
class SecondaryTierManager(ABC):

    def lookup(self, key, req_context) -> LookupResult:
        """Does this tier have a chunk? Returns HIT, MISS, or RETRY."""

    def submit_store(self, job_metadata: JobMetadata) -> None:
        """Start async store from host to this tier."""

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """Start async load from this tier to host."""

    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Poll completed transfers."""
```

Each tier receives a **direct memoryview** into the shared host region at construction time.
When `submit_store()` is called, the tier reads KV data directly from this region.
When `submit_load()` is called, the tier writes into it.
**No intermediate copies or serialization needed** — the tier operates directly on the primary tier's memory.

Each secondary tier also manages its own eviction policy independently.

A complete in-memory reference implementation is available at [`vllm/v1/kv_offload/tiering/example/`](https://github.com/vllm-project/vllm/tree/main/vllm/v1/kv_offload/tiering/example).
Out-of-tree secondary tiers are supported — specify a `module_path` in the tier config and vLLM loads your custom `SecondaryTierManager` implementation without any code changes to vLLM itself.

---

## Performance — Scaling to More Users

The main benefit of KV cache offloading: avoiding costly repeated prefills by reloading KV data from a cheaper tier.

With few concurrent conversations, all caching methods achieve high throughput — accelerator memory holds everything.
As the conversation pool grows, capacity limits appear at each tier:

- **Up to ~64 conversations** — HBM holds the working set; all caching methods perform well.
- **64–128 conversations** — HBM fills up; throughput drops dramatically without offloading. CPU offloading maintains performance.
- **Beyond 128 conversations** — CPU cache also fills up. Storage offloading continues serving a high cache hit ratio, more than doubling throughput compared to the alternatives.

<p align="center">
<img src="/assets/figures/tiered-kv-offloading/performance.svg" alt="Requests per second vs. conversation pool size" width="90%">
</p>

Storage has higher latency than CPU memory, so it does not reach peak throughput.
But at scale, the choice is between a storage-backed cache hit and a full recompute — storage wins decisively.

**Benchmark setup:**
- Model: Qwen/Qwen3.6-35B-A3B on a single NVIDIA H100
- Storage tier: filesystem backend on local NVMe
- Workload: multi-turn conversations, 12K-token initial prompts + 2K tokens per round, 8 rounds
- Max request concurrency: 64
- Measures prefiller throughput only (prefill-decode disaggregated)

---

## Acknowledgements

We would like to thank Liran Schour, Chang Guo, Srinivas Krovvidi, Rotem Shavitt, Effi Ofer, and Omer Paz for their contributions to the design and implementation of the tiered KV cache offloading framework, and all other community members who contributed code, reviews, and feedback.
