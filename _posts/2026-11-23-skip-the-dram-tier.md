---
layout: post
title: 'Skip the DRAM Tier: MoRI UMBP Runs the Agentic KV Cache on NVMe'
author: AMD and Embedded LLM
date: '2026-11-23'
summary: 'On AMD Instinct™ MI355X GPUs, a MoRI UMBP KV pool on NVMe matched a DRAM pool of the same size within 1% for Kimi-K3 on vLLM, and drives that hold a day of agent KV cost less than DRAM that holds 31 minutes.'
description: 'On AMD Instinct™ MI355X GPUs, a MoRI UMBP KV pool on NVMe matched a DRAM pool of the same size within 1% for Kimi-K3 on vLLM, and drives that hold a day of agent KV cost less than DRAM that holds 31 minutes.'
image: /assets/figures/2026-11-23-skip-the-dram-tier/hero.png
social_image: /assets/figures/2026-11-23-skip-the-dram-tier/hero.png
tags:
- performance
- rocm
- amd
- prefix caching
---

Most of the traffic in agentic serving is multi-turn sessions that resend long prefixes. In the [AgentX](https://inferencex.semianalysis.com/agentx) benchmark replaying real Claude Code sessions, the median turn is 142K input tokens, and over 96% of those tokens are a prefix the server has already seen [[3]](#ref-3).

A KV cache hit means the prefill node only has to process the small delta. A miss forces a full recomputation of the prefix, which burns GPU time and delays the first token. In our 42 runs of Kimi-K3 on vLLM, raising the KV cache hit rate by 1% of prompt tokens, for example from 95% to 96%, was worth about 10% more throughput (Figure 3).

A KV cache hit needs the session’s prefix to still be in memory when its next turn arrives. With dozens of sessions open at once, memory runs out. GPU memory goes first, and host DRAM is the usual next tier. But DRAM fills too: Kimi-K3 wrote 2.3 to 4.6 TB of KV to the pool an hour in our runs, so a 1.8 TB DRAM pool, which lists at about \$76,000, holds only the last 23 to 47 minutes of it.

That led us to a simple question: why not skip the DRAM tier and put vLLM’s KV pool straight on NVMe? We tried it on AMD Instinct™ MI355X GPUs with UMBP, the KV cache pool in AMD’s open-source MoRI library. At the same size, NVMe matched DRAM’s throughput within 1% on Kimi-K3, in two sets of runs on different machines (Figure 1). On a second workload with longer prompts, NVMe came within 1% of a DRAM pool of the same size. The NVMe pool keeps a small DRAM write buffer in front of its drives, and cutting it from 64 GB to 4 MiB cost no throughput. And NVMe’s cost follows the KV it writes, not the KV it holds, so drives rated for Kimi-K3’s writes cost less than that DRAM pool and hold a full day of KV.

Operators of large GPU clusters see the same problem:

> “Agentic workloads keep sessions open for minutes or hours, and their KV cache quickly outgrows GPU memory and host DRAM. Results like these, with NVMe keeping up with DRAM at the same pool size, change how we think about sizing KV cache capacity in AI clusters. We’re glad to see this work done in the open, in vLLM and MoRI.”
>
> — Dennis Kennetz, Principal Infrastructure Engineer, Oracle Cloud Infrastructure

DeepSeek already keeps agent KV caches on SSD: the deployment notes for its DeepInfer engine describe three days of retention for agent traffic, on about 14.4 TB of SSD per GPU [[4]](#ref-4). This work references DeepInfer for the idea of keeping the KV cache on SSD and for two of its techniques, layer-wise loading and bounded replay, which we test on vLLM below.

![Grouped bars of Kimi-K3 throughput on vLLM with 1.8 TB pools on the AgentX traces. At 48 sessions: DRAM pool 12,240 tok/s per GPU, NVMe pool 12,275 (+0.3%). At 70 sessions: DRAM 13,380, NVMe 13,492 (+0.8%), and NVMe with the DRAM write buffer cut to 4 MiB 13,847 (+3.5%). At 96 sessions: DRAM 13,313, NVMe 13,284 (−0.2%). KV cache hit rates, DRAM / NVMe: 96.62% / 96.67%, 95.57% / 95.65% (95.79% with the 4 MiB buffer) and 94.50% / 94.48%.](/assets/figures/2026-11-23-skip-the-dram-tier/fig1-same-size.png)

**Figure 1.** *Kimi-K3 on vLLM: throughput with a 1.8 TB DRAM pool and with a 1.8 TB NVMe pool at three loads. The NVMe pool keeps a small DRAM write buffer in front of its drives; the third bar at 70 sessions cuts that buffer to 4 MiB.*

## TL;DR

- **Same size, same throughput.** On vLLM, a 1.8 TB NVMe KV pool matched a 1.8 TB DRAM pool: Kimi-K3’s throughput stayed within 1% at 48, 70 and 96 concurrent sessions, in two sets of runs on different machines. On longer prompts, NVMe came within 1% of a DRAM pool of the same size at 70 sessions.
- **Almost no DRAM.** Cutting the DRAM write buffer in front of the drives from 64 GB to 4 MiB cost no throughput, and the long-prompt comparison ran with 4 MiB. A 16 GiB staging buffer for reads stays.
- **Why.** Throughput follows the KV cache hit rate, and same-size pools that evict the same way keep the same KV. vLLM loads KV asynchronously, so on a busy engine NVMe’s slower load finished while the request waited for prefill.
- **What NVMe buys.** Retention. DRAM costs per TB held; NVMe costs per TB written. At list prices, drives rated for Kimi-K3’s 83 TB of writes a day hold a day of KV for about \$59,000; a \$76,000 DRAM pool holds 31 minutes.
- **Limitation and roadmap.** Every block of KV the GPU computes is written to the drives, wearing them out, so we will spread those writes over more SSDs with a distributed version of the pool.

## How a KV pool works on vLLM

vLLM keeps the KV of recent prompts in GPU memory, in its prefix cache. When GPU memory fills, vLLM drops the least recently used blocks, and a session that returns after that must recompute its prefix. A KV pool keeps that KV somewhere bigger, in host DRAM or on local NVMe. The prefill engine saves each block of KV it computes to the pool. When a later turn’s prefix is no longer in GPU memory, vLLM loads it back from the pool instead of recomputing it (Figure 2). The KV cache hit rate in this post counts both places: it is the share of prompt tokens whose KV came from GPU memory or the pool.

We used UMBP, the KV cache pool in [MoRI](https://github.com/ROCm/mori), AMD’s open-source library for GPU communication over RDMA. vLLM already uses MoRI on AMD GPUs to send KV from prefill nodes to decode nodes, and UMBP also powers Kimi’s agentic serving on AMD Instinct™ GPUs, co-designed by AMD and Moonshot AI [[3]](#ref-3). UMBP, short for unified memory and bandwidth pool, stores KV blocks by key and decides where each block lives and what to evict when it is full. In our tests it runs as its own server process beside the vLLM prefill engine, which reaches it through a KV connector, `UMBPStoreConnector`.

We compared two pools of the same size, 1.8 TB:

- **DRAM pool.** UMBP keeps every block in host memory and evicts the least recently used KV when it is full.
- **NVMe pool.** New KV lands in a small DRAM write buffer, 64 GB in most runs, and moves on to two local NVMe drives as the buffer fills. A hit on a block still in the buffer loads from DRAM; a hit on the drives is read into a 16 GiB staging buffer and copied to GPU memory from there.

The drives are local NVMe SSDs: two Micron 7450 PRO drives of 3.84 TB per node, rated for about one drive write per day, each formatted ext4, with no RAID. UMBP keeps KV in segment files on those file systems and reads them with direct I/O through io_uring, so the Linux page cache never serves a read from DRAM. There is no NVMe-oF and no shared or network file system; when one pool spans two nodes, UMBP moves blocks between them over RDMA itself.

Many of the runs in Figures 3 and 5 put a larger DRAM tier in front of the drives; we call that a DRAM + NVMe pool.

![Two data paths on the prefill node. DRAM pool: the vLLM prefill engine saves KV to, and loads it from, a UMBP pool server that keeps 1.8 TB in host memory and evicts the least recently used KV. NVMe pool: the engine saves KV to a DRAM write buffer of 64 GB, or 4 MiB, and loads recent blocks from it; the buffer writes blocks on to 1.8 TB of NVMe on two drives, in 256 MiB segment files; a hit on a block already on the drives is read with direct I/O into a 16 GiB staging buffer, and the engine loads it from there.](/assets/figures/2026-11-23-skip-the-dram-tier/fig2-datapath.png)

**Figure 2.** *Where the KV pool sits, with a DRAM pool and with an NVMe pool.*

Most of our tests use Kimi-K3, a hybrid model. Of its 93 layers, 69 use linear attention (KDA), which keeps a fixed-size running state instead of per-token KV; the other 24 use MLA, which keeps compressed KV for every token. So a Kimi-K3 pool holds KV blocks and checkpoints of the KDA state, from which a later turn can resume mid-prompt. A few tests use DeepSeek-V4-Pro, and one uses DeepSeek-V4.1-Flash.

We served Kimi-K3 and DeepSeek-V4-Pro as 1P1D on 16 MI355X GPUs: TP8 prefill + TP8 decode. The prefill node sends each request’s KV to the decode node over RDMA, and in most tests it also hosts the pool. AIPerf, the load generator behind AgentX, replayed two public agent workloads on their recorded timing:

- **AgentX traces:** 393 Claude Code sessions that SemiAnalysis recorded for AgentX, the agentic benchmark in its InferenceX suite.
- **The kvlab 40-day corpus:** about 3,200 sessions of WEKA’s own agent traffic, recorded from 5 September to 14 October 2026 and published with [kvlab](https://github.com/weka/kvlab), WEKA’s open-source agent replay benchmark. Its prompts are longer: the requests our runs replayed had a median of 165K tokens, against AgentX’s 142K.

Load is the number of sessions replayed at once. The appendix gives the full setup.

## At the same size, NVMe matched DRAM

NVMe stayed within 1% of DRAM at 48, 70 and 96 sessions (Figure 1). Each comparison replayed the AgentX traces for an hour, once with each pool, back to back on the same machines with the same image and settings, and both pools filled and evicted. A repeat of all three loads on another pair of machines put NVMe ahead by 0.4%, 0.5% and 0.9%; repeats of one configuration on one pair of machines vary by 0.3% to 0.45%. Throughput counts input and output tokens per second per GPU over all 16 GPUs, cached prompt tokens included, as AgentX reports it.

The KV cache hit rates differed by at most 0.08% of prompt tokens. Median time to first token (TTFT) stayed within 0.3 s at 48 and 70 sessions and within 0.8 s at 96, where requests waited about 24 s.

These runs leaned on GPU memory: it served 96%, 90% and 79% of prompt tokens at the three loads, so the pool served 1%, 5% and 16%. The drives took 2.3 to 3.5 TB of writes an hour, 17, 3.8 and 1.5 times what they read; reads peaked at 2.3 TB an hour, about 0.64 GB/s.

### The DRAM in front of the drives can shrink to 4 MiB

The 64 GB write buffer serves hits on recent blocks from DRAM, so it might have been doing the work. To check, we cut it to 4 MiB, the smallest UMBP’s tier policy accepts, and kept 1.8 TB on the drives:

| **Kimi-K3, AgentX, 70 sessions**          | **tok/s per GPU** | **KV cache hit rate** | **TTFT p50 / p90** |
|-------------------------------------------|-------------------|-----------------------|--------------------|
| 1.8 TB DRAM pool                          | 13,380            | 95.57%                | 5.19 / 23.40 s     |
| 1.8 TB NVMe pool, 64 GB DRAM write buffer | 13,492 (+0.8%)    | 95.65%                | 4.89 / 21.26 s     |
| 1.8 TB NVMe pool, 4 MiB DRAM write buffer | 13,847 (+3.5%)    | 95.79%                | 5.53 / 19.18 s     |

With 4 MiB, the NVMe pool ran 3.5% above the DRAM pool. Its KV cache hit rate was 0.22% of prompt tokens higher, more than the 0.08% that separated the other comparisons; its median TTFT was 0.34 s higher and its TTFT p90 the lowest of the three. It is a single run, so we claim only that the smaller buffer cost no throughput. The 16 GiB staging buffer stays: reads from the drives land there before the GPU copies them.

### Longer prompts

A KV load grows with the prompt, so we also ran the kvlab corpus. At 70 sessions, a 1.8 TB NVMe pool with a 4 MiB DRAM write buffer served 8,110 tok/s per GPU. A 1.8 TB DRAM pool served 8,162, so NVMe was 0.6% behind. The NVMe pool kept a KV cache hit rate 0.29% of prompt tokens higher, 91.89% against 91.60%. Its median TTFT was 88.8 s against 92.6 s, and its TTFT p90 was 156.9 s against 149.1 s. Each is a single run, on a cluster where one configuration varies by about ±2.5% from run to run.

| **Kimi-K3, kvlab corpus, 70 sessions**    | **tok/s per GPU** | **KV cache hit rate** | **TTFT p50 / p90** |
|-------------------------------------------|-------------------|-----------------------|--------------------|
| 1.8 TB DRAM pool                          | 8,162             | 91.60%                | 92.6 / 149.1 s     |
| 1.8 TB NVMe pool, 4 MiB DRAM write buffer | 8,110 (−0.6%)     | 91.89%                | 88.8 / 156.9 s     |

DeepSeek-V4-Pro on the kvlab corpus came within 1.2% of DRAM in all three of its comparisons, two of them with pools that never filled (appendix).

## Why NVMe keeps up

A read from NVMe is slower than a read from DRAM. Three facts kept that from reaching throughput.

**Throughput follows the KV cache hit rate, and same-size pools keep the same KV.** Every prompt token that misses goes back to the prefill node as work. At a 96% KV cache hit rate, the prefill node computes 4% of prompt tokens; at 95%, it computes 5%, which is 25% more prefill work for the same traffic. In 42 runs of Kimi-K3 at 70 sessions on the AgentX traces, with pools in DRAM, on NVMe and in both, at different sizes and settings, the KV cache hit rate alone explained 96% of the variance in throughput, or 92% without the one run with a 0.9 TB pool (Figure 3). Each extra 1% of prompt tokens served from the KV cache was worth about 1,280 tok/s per GPU.

![Scatter of 42 Kimi-K3 runs at 70 sessions: throughput per GPU against KV cache hit rate. The runs rise along one line with R² 0.96, from 9,851 tok/s per GPU at a 92.56% KV cache hit rate to about 14,200 near 96%. DRAM pools, DRAM pools on both nodes, DRAM + NVMe pools and NVMe pools all sit near the line; the NVMe pool with 4 GiB segment files sits furthest below it.](/assets/figures/2026-11-23-skip-the-dram-tier/fig3-hit-rate.png)

**Figure 3.** *Kimi-K3 on the AgentX traces: throughput against the KV cache hit rate, one dot per run.*

What a pool keeps depends on its capacity and on what it evicts, not on its medium. Both pools ran the same eviction rules: the coldest KV goes first, and checkpoints age faster than KV. So at the same size they keep nearly the same KV: in the six Kimi-K3 comparisons on the AgentX traces, their KV cache hit rates differed by at most 0.08% of prompt tokens. That took one fix to how the NVMe pool frees space, the first lesson below.

**A hit is worth far more than its load.** In a single-request test in MoRI’s guide, with a 128K-token prefix, the first token came 0.45 s after a DRAM hit, 0.84 s after an NVMe hit and 7.7 s after a recompute (Kimi-K2.6 under SGLang, eight drives, six of them on a second node) [[1]](#ref-1). With no queue at all, NVMe adds about 0.4 s to a hit; a miss adds about 7 s.

**The load finishes while the request waits for prefill.** vLLM’s KV connector API lets a connector load KV asynchronously: the scheduler holds the request back until its KV arrives and keeps scheduling the others. UMBP’s connector also looks KV up asynchronously. On a busy engine a request waits in the prefill queue anyway, and the load fits inside that wait (Figure 4). On DeepSeek-V4-Pro at 64 sessions, a load took 0.80 s from DRAM and 0.85 s from NVMe, and finished about 2 s before the request’s prefill started.

<iframe class="vllm-embed" src="/assets/interactive_pages/skip-dram-async-load.html" title="Async loading hides NVMe's slower read: the KV copy into GPU memory runs while the GPU prefills the request ahead" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 913px; border: 0; overflow: hidden;"></iframe>

**Figure 4.** *The KV load hides inside the wait for a prefill slot.*

vLLM’s connector API gave us that overlap; under SGLang, our NVMe pool runs needed an extra patch, not upstream, to get it.

DeepInfer’s layer-wise loading takes the other route: it copies KV into GPU memory one layer at a time during the forward pass. We built it into vLLM for Kimi-K3 with an NVMe pool. On an idle engine it served a single pool hit 14% to 55% faster (0.91 s against 1.32 s in one test), but under load it matched the asynchronous load’s mean TTFT and prefill throughput at every session count we ran (appendix). The loading request starts sooner, and the rest of the batch waits by as much.

## What NVMe buys: retention

What NVMe buys is retention: a larger pool keeps each session’s KV longer, so a session that pauses between turns still finds its prefix.

**When sessions outgrow a tier, its throughput collapses.** Figure 5 follows Kimi-K3 on the kvlab corpus as sessions grow. With GPU memory alone, the KV cache hit rate fell from 90% at 32 sessions to 41% at 48, and throughput fell 80%. A 1.0 TB DRAM pool started falling after 32 sessions and lost 81% of its throughput by 70. A 4.6 TB NVMe pool, spread over the drives of both nodes, kept its KV cache hit rate between 92.7% and 95.8% at every load it ran. At 70 sessions it served 8,813 tok/s per GPU (5.1 times the 1.0 TB DRAM pool) and kept the KV cache hit rate at 93% against DRAM’s 52%. The gap comes from capacity, not the medium: in another run, a 1.8 TB DRAM pool still held a 91% KV cache hit rate at 70 sessions (appendix). A pool with 1.0 TB of DRAM in front of 1.8 TB of NVMe stayed within 3% of the 4.6 TB NVMe pool at every load they shared.

![Throughput per GPU against concurrent sessions on the kvlab corpus. GPU memory only: 6,261, 8,255, 7,210, 1,439 and at most 1,062 tok/s per GPU at 8, 16, 32, 48 and 70 sessions, KV cache hit rate falling to 41% and about 6%. With a 1.0 TB DRAM pool: 6,681, 9,274, 9,077, 5,783 and 1,725, KV cache hit rate 88% at 48 and 52% at 70. With 1.0 TB of DRAM plus 1.8 TB of NVMe: 6,643, 9,520, 11,064, 9,345 and 9,074. NVMe pool, 4.6 TB: 6,608, 9,466, 10,928 and 8,813 at 8, 16, 32 and 70, KV cache hit rate 93% at 70.](/assets/figures/2026-11-23-skip-the-dram-tier/fig5-load-sweep.png)

**Figure 5.** *Kimi-K3 on the kvlab corpus: throughput against concurrent sessions, for each set of memory tiers.*

<iframe class="vllm-embed" src="/assets/interactive_pages/skip-dram-agents-at-scale.html" title="At 70 agents, only the NVMe pool keeps up: GPU memory, a 1.0 TB DRAM pool and a 4.6 TB NVMe pool side by side" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 938px; border: 0; overflow: hidden;"></iframe>

The fall is steep because each miss causes more misses: a recompute holds up the prefill node, the queue grows, sessions wait longer between turns, and their KV must survive longer in the pool. At 70 sessions the DRAM pool’s median TTFT reached 743 s.

**A pool keeps KV for its capacity divided by its write rate.** At 2.3 to 4.6 TB written an hour, a 1.8 TB pool keeps 23 to 47 minutes of Kimi-K3’s KV: 36 minutes at 70 sessions on the AgentX traces, and 31 at 96. Whether that is enough depends on how long sessions pause between turns. We replayed the recorded timing of the AgentX traces through a model of a least-recently-used pool (Figure 6). A pool that keeps KV for 36 minutes misses about 1% more of prompt tokens than one that never evicts. At one hour the extra miss rate is 0.7%; at three hours, 0.3%. The model predicted what a pool gained when we grew it from 1.8 TB to 3.0 TB: 0.34% of prompt tokens, against 0.32% measured.

![Log-log curve from the AgentX traces: the extra share of prompt tokens that miss, compared with a pool that never evicts, against how long the pool keeps KV after its last use. About 14% at 1 minute, 4% at 5 minutes, 1.0% at 36 minutes, 0.7% at 1 hour, 0.27% at 3.2 hours and 0.03% at 24 hours.](/assets/figures/2026-11-23-skip-the-dram-tier/fig6-retention.png)

**Figure 6.** *Extra prompt tokens that miss against how long the pool keeps KV: a model of a least-recently-used pool, run on the recorded timing of the AgentX traces.*

## What it costs: DRAM per TB held, NVMe per TB written

[DatacenterDisk](https://datacenterdisk.com/tools/ai-memory-capex-calculator/) lists about \$42 per GB for server DDR5 and \$666 per TB for enterprise SSD (median listings, 13 November 2026). A DRAM pool costs its purchase price, about \$42,000 for every TB it holds. An NVMe pool wears out by writing: a drive rated for one full write a day takes 1,825 writes in five years, so each TB written uses about \$0.36 of drive (\$666 ÷ 1,825).

**Size NVMe for its writes.** Every block the prefill engine computes goes to the pool, whether or not a later turn reads it back. At 96 sessions on the AgentX traces, Kimi-K3 wrote 3.5 TB an hour, or 83 TB a day. Our 1.8 TB test pool sat on two 3.84 TB drives rated for about one drive write per day. At 96 sessions the pair took about 11 drive writes a day, enough to use up its rated life in about five and a half months. We will spread these writes over more SSDs with a distributed version of the pool. Drives rated for one write a day must add up to 83 TB, and at that size they hold a day of KV. So up to a day of retention, a bigger NVMe pool costs nothing extra, while a bigger DRAM pool costs \$42,000 per TB (Figure 7).

An NVMe pool’s five-year cost is whichever is larger, the price of its drives or the drive life its writes use up, plus about \$3,400 for the DRAM it keeps: the 64 GB write buffer and 16 GiB of staging. That is why a 1.8 TB NVMe pool, replaced as it wears out, and an 83 TB pool that keeps a day cost the same, about \$59,000. A 1.8 TB DRAM pool costs about \$76,000, and a DRAM pool that keeps a day about \$3.5 million. Below about 24 minutes of retention, 1.4 TB at this write rate, DRAM is the cheaper pool.

![Log-log chart of five-year cost against how long the pool keeps KV, for Kimi-K3 at 96 sessions on the AgentX traces, writing 3.5 TB an hour. The DRAM line rises from about $30,000 at 12 minutes to $3.5 million for a day. The NVMe line stays flat at about $59,000 up to a day, then rises with capacity. Marked: 1.8 TB of DRAM keeps 31 minutes for $76,000; 1.8 TB of NVMe keeps 31 minutes for $59,000; 83 TB of NVMe keeps a day for $59,000.](/assets/figures/2026-11-23-skip-the-dram-tier/fig7-cost.png)

**Figure 7.** *Five-year cost against retention for Kimi-K3 at 96 sessions on the AgentX traces. Our arithmetic at list prices.*

<iframe class="vllm-embed" src="/assets/interactive_pages/skip-dram-retention-cost.html" title="A pool keeps KV for its capacity divided by its write rate: retention and five-year cost" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 825px; border: 0; overflow: hidden;"></iframe>

What does a day of retention buy? By the model in Figure 6, a day instead of 31 minutes removes about 1% of prompt tokens from the misses on the AgentX traces; by the fit in Figure 3, that is worth roughly 10% of throughput. The estimate chains two fitted models: we checked the retention model only between 36 minutes and one hour, and the throughput fit comes from 70-session runs.

Fewer writes need fewer drives. More than half of Kimi-K3’s writes are checkpoints, and lesson 3 below shows how far dropping old ones cut them.

These costs are arithmetic at list prices, not measurements: they assume the load runs around the clock for five years and leave out power, rack space and the servers that hold the drives.

## What it took: three lessons

On NVMe, one setting matters. With UMBP’s default segment files, an NVMe pool ran 12.8% behind DRAM on Kimi-K3 at 70 sessions on the AgentX traces; smaller files closed that gap. Two other changes raise the KV cache hit rate whatever the medium; we measured them on DRAM and DRAM + NVMe pools. Each was measured on its own setup, so the gains do not add up.

**1. Free NVMe space in small segment files.** On NVMe, UMBP only appends: it writes each block at the end of a segment file. Evicting a block frees no space by itself; the space comes back only when UMBP deletes the whole file. To make room, UMBP picks the files with the fewest live blocks and the least recent use, and evicts whatever is still live in them [[1]](#ref-1). With 4 GiB files, the default, each freed file threw away KV that sessions still used along with the stale KV beside it. Moving to 256 MiB files, the only setting we changed, raised Kimi-K3’s KV cache hit rate at 70 sessions from 94.45% to 95.65%, cut TTFT p90 from 63.0 s to 21.3 s and raised throughput 15.6%, from 11,672 to 13,492 tok/s per GPU. That moved NVMe from 12.8% below DRAM to 0.8% above. Smaller files cost more open files.

<iframe class="vllm-embed" src="/assets/interactive_pages/skip-dram-segment-files.html" title="Large segment files fragment the pool, so freeing space throws away live KV" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 818px; border: 0; overflow: hidden;"></iframe>

**2. Tell the pool which KV the GPU still serves.** GPU memory serves the hottest prefixes itself, so the pool never sees them read. To the pool they look unused, and it evicts them first; when GPU memory later drops one, the pool’s copy is already gone. With touch, vLLM’s connector reports each GPU prefix hit to the pool, which marks those blocks as recently used ([MoRI #779](https://github.com/ROCm/mori/pull/779) and [charxwu/vllm#19](https://github.com/charxwu/vllm/pull/19)). On Kimi-K3 at 70 sessions with DRAM + NVMe pools, touch added 2.1% and 3.9% to throughput in two tests and cut TTFT p99 from 70–77 s to 35–43 s. Both pools in Figure 1 used touch, in a version that also refreshes the copies on the drives ([MoRI #882](https://github.com/ROCm/mori/pull/882)).

<iframe class="vllm-embed" src="/assets/interactive_pages/skip-dram-touch.html" title="Touch tells the pool which KV the GPU still uses: without and with touch side by side" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 858px; border: 0; overflow: hidden;"></iframe>

**3. Checkpoint hybrid models more often.** A turn can reuse a prefix only if GPU memory or the pool holds the model’s whole state at that point, and Kimi-K3’s KDA state exists only where vLLM kept a checkpoint. By default vLLM keeps one at the end of each prompt, so a turn that shares only part of an earlier prompt finds no checkpoint where the two diverge, and recomputes. Adding a checkpoint every 7,168 tokens (`--prefix-cache-retention-interval 7168`) raised the KV cache hit rate from 91.4% to 95.6% and throughput by 33%, from 5,113 to 6,791 tok/s per GPU (kvlab corpus, 8 sessions, DRAM pool).

<iframe class="vllm-embed" src="/assets/interactive_pages/skip-dram-checkpoints.html" title="A hybrid model resumes only at a checkpoint: checkpoints at prompt ends against every 7,168 tokens" loading="lazy" scrolling="no" style="display: block; width: 100%; height: 890px; border: 0; overflow: hidden;"></iframe>

Checkpoints cost writes: even at vLLM’s default spacing, they were more than half of Kimi-K3’s NVMe writes on the AgentX traces. Once a later turn has resumed from a checkpoint and moved well past it, the connector can mark it as replaced, and UMBP drops it when it leaves DRAM instead of moving it to NVMe. With 1.8 TB of DRAM in front of the drives, that cut NVMe writes by 53% at 70 sessions, from 2.26 to 1.05 TB an hour. The KV cache hit rate stayed the same, but throughput fell 1.0% and TTFT p99 rose from 38 s to 62 s. Behind a small write buffer, fewer checkpoints are replaced before they reach NVMe. Figure 1’s runs dropped none.

Sliding-window layers have the same problem. DeepSeek-V4.1-Flash’s window is only 128 tokens, so DeepInfer’s bounded replay keeps no window checkpoints and rebuilds the window on a hit from the last 128 tokens. That gave 2.2× the throughput of a window checkpoint every 2K tokens (DRAM pool, 288 sessions, kvlab corpus). The appendix has the DeepSeek-V4-Pro result.

## Size a pool for your workload

1.  **Measure the write rate.** Run your peak load and read how many TB the pool writes to its drives per hour; UMBP serves its counters on `/metrics`, from the pool server or, for a pool spread over several nodes, from its master.
2.  **Pick the retention.** Capacity is the write rate times how long KV must survive. To price a retention time on your traffic, take each block of prompt that a later request reuses and measure the time since its last use: a pool that keeps KV for a time T misses every block whose gap is longer than T. That is how we built Figure 6.
3.  **Check the drives’ wear.** If the pool fills its drives, drive writes per day are 24 divided by the retention in hours. A pool that keeps a day writes each drive once a day; a smaller one needs drives with a higher rating, or fewer writes.

## Limits

- **Heavy reads.** In the same-size runs the pool served at most 16% of prompt tokens. We have not compared NVMe with DRAM where the pool serves most of them. DeepSeek-V4-Pro’s pool does beyond 64 sessions, serving 64% to 86% of prompt tokens with DRAM pool reads peaking near 27 GB/s, and our DeepSeek-V4-Pro comparisons stop at 64.
- **Light traffic.** With no queue to hide in, an NVMe load adds its extra time to TTFT: about 0.4 s for a 128K-token prefix in MoRI’s test.
- **Single runs.** Most results, including the 4 MiB write buffer and the long-prompt comparison, are single runs.
- **Scale.** Our largest NVMe pool was 4.6 TB, across two nodes; the pool that keeps a day is arithmetic. Every run used UMBP on MI355X GPUs, and DeepSeek-V4.1-Flash has not run on NVMe.
- **Upstream code.** Some changes in our runs are still on their way upstream. Several are already open pull requests, and we are working to land the rest in vLLM and MoRI; the appendix lists each one and its status.
- **Models with more KV per token.** Kimi-K3 and DeepSeek-V4-Pro store little KV per token. Kimi-K3 keeps about 14 KB of KV per token, 576 bytes in each of its 24 MLA layers with FP8 KV, and DeepSeek-V4-Pro about 5.5 KB with FP8 KV or 9.9 KB with BF16, checkpoints excluded. All runs here used FP8 KV for both models. A dense model without MLA stores many times more and would need that much more drive bandwidth.

## Run it on vLLM

`UMBPStoreConnector` is in vLLM main; it landed in a ten-part series of pull requests that ended on 13 November, starting with [#61789](https://github.com/vllm-project/vllm/pull/61789). The prefill engine adds it to its KV transfer configuration next to `MoRIIOConnector`, which sends KV to the decode node; the decode engine keeps only the MoRIIO consumer. The relevant flags from our Kimi-K3 runs:

```bash
vllm serve moonshotai/Kimi-K3 --tensor-parallel-size 8 \
  --enable-prefix-caching --prefix-cache-retention-interval 7168 \
  --kv-transfer-config '{"kv_connector": "MultiConnector", "kv_role": "kv_both",
    "kv_connector_extra_config": {"connectors": [
      {"kv_connector": "MoRIIOConnector", "kv_role": "kv_producer",
       "kv_connector_extra_config": {"backend": "rdma"}},
      {"kv_connector": "UMBPStoreConnector", "kv_role": "kv_both",
       "kv_connector_extra_config": {"mode": "standalone", "endpoint": "/run/umbp/pool.sock",
         "load_async": true, "lookup_async": true, "enable_partial_hash_hits": true}}]}}'
```

Our kvlab runs set `--prefix-cache-retention-interval 7168`; the AgentX runs in Figure 1 kept vLLM’s default. `enable_partial_hash_hits` lets a hit end at any hash block, which can be smaller than a KV cache block.

The UMBP pool server runs on the same node. Its tier policy (`policy.json`) names the drives and their capacities and puts the 64 GB write buffer in front of them. It moves blocks to the drives when the buffer evicts them (`"offload_trigger": "on_evict"`), and never copies them back (`"promote_trigger": "never"`), so a hit on the drives is always read from the drives. The server command:

```bash
UMBP_BACKEND_POLICY=policy.json UMBP_DRAM_CAPACITY=64000000000 UMBP_DRAM_USE_HUGEPAGES=1 \
UMBP_SSD_SEGMENT_SIZE=268435456 UMBP_SSD_VERIFY_CRC=0 UMBP_SSD_DURABILITY=relaxed \
  umbp_standalone_server unix:///run/umbp/pool.sock
```

`UMBP_SSD_SEGMENT_SIZE=268435456` sets the 256 MiB segment files. A tier policy honors the three `UMBP_SSD_*` settings only from MoRI \#967, merged on 17 November; with older builds the drives silently keep 4 GiB segment files, checksums and synced writes. `UMBP_SSD_VERIFY_CRC=0` skips checksum verification, as MoRI’s guide advises when comparing with DRAM, which keeps no checksums; a production pool may want it on, and the guide measures verification at 51% to 67% of a read’s time. For the 4 MiB write buffer, set `UMBP_DRAM_CAPACITY=4194304` and the DRAM tier’s capacity in the policy to `4194304B`.

## Acknowledgements

Special thanks to Pin Siang Tan, Tian Di, Ma Ning, Wu Yutong, Wu Yanglong, Zhang Figo, Xiaorui Wu, Bill He, Gilbert Lei, Badr Sowid, Maral Bahari, Tun Jian Tan, HongXia Yang, Sun Peng. We also thank DeepSeek for publishing DeepInfer and its deployment notes, which shaped much of this work; SemiAnalysis for AgentX and InferenceX; and WEKA for kvlab and its 40-day corpus.

## Appendix: how we tested

- **Hardware and topology.** 1P1D on 16 AMD Instinct™ MI355X GPUs: TP8 prefill + TP8 decode, one node each. Kimi-K3 also uses DCP8 (decode context parallelism) on both nodes. The prefill node sends each request’s KV to the decode node over RDMA with MoRIIO, vLLM’s MoRI connector. In every Kimi-K3 run, decode uses DSpark speculative decoding with 4 draft tokens and a synthetic acceptance length of 3.36, as the AgentX CI recipe does. Each node has two Micron 7450 PRO NVMe drives (MTFDKCC3T8TFR, 3.84 TB each, rated for about 1 drive write per day), formatted ext4 with no RAID; the host has two AMD EPYC™ 9575F 64-core CPUs and 3 TB of DRAM. The nodes run Linux 6.8, and UMBP accesses the drives with direct I/O through io_uring. All runs used FP8 KV cache for both models.
- **Workloads.** The AgentX traces hold 393 sessions, 98,827 requests and 21.6 billion prompt tokens, published as hashes of 64-token blocks so that prefix matches survive without the content [[6]](#ref-6). kvlab’s 40-day corpus holds 3,190 sessions and 370,658 requests [[7]](#ref-7). Converting it to AIPerf’s `weka_trace` format kept 3,164 sessions and 325,755 requests with 91.6 billion input tokens, dropping 44,903 requests with no input. The requests our runs replayed had a median prompt of 165K tokens and a 90th percentile of 306K. AIPerf replays both workloads on their recorded timing with a fixed number of sessions running at once. It loads the AgentX traces by their built-in name, `--public-dataset semianalysis_cc_traces_weka_062126`; the kvlab runs used the file `replay-40d-oct14-337a03810637.jsonl` from `wekaio/replay-40d-oct14-337a03810637`.
- **Runs.** AgentX runs measure one hour after 37 to 46 minutes of warmup. kvlab runs measure one hour after a warmup that reaches about 65 minutes at 70 sessions, except every 8-session run and the 16-session rows without the checkpoint changes, which measure 30 minutes; DeepSeek-V4-Pro’s window-checkpoint test measured 20 minutes. Both pools use the same tuning. A DRAM tier starts evicting at 97% full and stops at 93%, an NVMe tier at 95% and 90%. NVMe runs with checksums off and relaxed durability. An eviction class ages Kimi-K3’s checkpoints twice as fast as KV; the kvlab runs with the checkpoint changes listed below age them eight times as fast, keep them off NVMe and hold them for five minutes after a lookup. In each AgentX comparison only the pool differs: the same image and settings, run back to back on the same machines. The first set of AgentX comparisons used a different pair of machines for each load, and the repeat used one pair for all three. The 4 MiB row shares its DRAM run with the first 70-session comparison, and its policy split the 1.8 TB a little differently between the two drives (374 and 1,426 GB, against 485 and 1,313 GB). Figure 3’s 42 runs cover 26 configurations on 11 pairs of machines; the later repeat is not among them. The kvlab runs used a shared cluster, where one configuration varies by about ±2.5% from run to run. Both runs of the 70-session comparison with a 4 MiB write buffer used the same machines and build.
- **Software.** Figure 5’s runs and the last four rows below used vLLM `0.33.0rc1.dev1065+g083fbac02`. The last four rows ran without the checkpoint changes listed below, on MoRI builds one fix apart between their DRAM and NVMe runs, and their 8-session DRAM run had 2.0 TB of DRAM on each node. The DeepSeek-V4-Pro comparison at about 2 TB a node ran on a different build from the other DeepSeek-V4-Pro rows.
- **Changes on their way upstream.** The connector side of touch, a wait for asynchronous lookups and the restore of hybrid-model prefixes are in [charxwu/vllm#19](https://github.com/charxwu/vllm/pull/19) (open). UMBP’s eviction classes, which age Kimi-K3’s checkpoints faster than KV and can drop replaced ones, and `UMBP_TRANSITION_WORKERS`, which sets how many workers move blocks from DRAM to the drives, will follow in pull requests. So will the checkpoint changes in the kvlab runs, which add checkpoints every 896 tokens over each prompt’s last 14,336 (UMBP deletes them instead of writing them to NVMe), a hold on checkpoints after a lookup and a KDA kernel that writes several checkpoints in one pass. Every AgentX image carries [vllm#59966](https://github.com/vllm-project/vllm/pull/59966) (open; decode speed only), and the DeepSeek-V4-Pro runs need [\#63002](https://github.com/vllm-project/vllm/pull/63002) (open).
- **Other checkpoint results.** Window checkpoints in the pool raised DeepSeek-V4-Pro’s KV cache hit rate from 78.4% to 89.9% and its throughput 76% over vLLM’s default, from 5,298 to 9,326 tok/s per GPU (kvlab corpus, 64 sessions, 20-minute measurements). Bounded replay on DeepSeek-V4.1-Flash went from 53,011 to 115,834 tok/s per GPU and cut the pool’s bytes per prompt token from 1,626 to 396.

Each row is one comparison: a run with a DRAM pool and a run with an NVMe pool.

| **Workload and pool**                                                  | **Load**    | **DRAM, tok/s/GPU** | **NVMe, tok/s/GPU** | **NVMe against DRAM** | **KV cache hit rate, DRAM / NVMe** | **TTFT p50, DRAM / NVMe** | **TTFT p90, DRAM / NVMe** |
|------------------------------------------------------------------------|-------------|---------------------|---------------------|-----------------------|------------------------------------|---------------------------|---------------------------|
| **Same size, both pools full and evicting**                            |             |                     |                     |                       |                                    |                           |                           |
| Kimi-K3, AgentX, 1.8 TB each                                           | 48 sessions | 12,240              | 12,275              | +0.3%                 | 96.62% / 96.67%                    | 1.57 / 1.57 s             | 7.92 / 7.15 s             |
|                                                                        | 70 sessions | 13,380              | 13,492              | +0.8%                 | 95.57% / 95.65%                    | 5.19 / 4.89 s             | 23.40 / 21.26 s           |
|                                                                        | 96 sessions | 13,313              | 13,284              | −0.2%                 | 94.50% / 94.48%                    | 24.95 / 24.39 s           | 62.57 / 56.33 s           |
| Kimi-K3, AgentX, 1.8 TB each, repeat on another pair of machines       | 48 sessions | 12,387              | 12,440              | +0.4%                 | 96.63% / 96.68%                    | 1.61 / 1.63 s             | 7.63 / 7.38 s             |
|                                                                        | 70 sessions | 13,639              | 13,710              | +0.5%                 | 95.61% / 95.69%                    | 4.73 / 4.80 s             | 22.23 / 20.83 s           |
|                                                                        | 96 sessions | 13,182              | 13,304              | +0.9%                 | 94.44% / 94.47%                    | 23.63 / 24.38 s           | 73.11 / 55.89 s           |
| Kimi-K3, AgentX, 1.8 TB each, 4 MiB DRAM write buffer                  | 70 sessions | 13,380              | 13,847              | +3.5%                 | 95.57% / 95.79%                    | 5.19 / 5.53 s             | 23.40 / 19.18 s           |
| Kimi-K3, kvlab, 1.8 TB each, 4 MiB DRAM write buffer                   | 70 sessions | 8,162               | 8,110               | −0.6%                 | 91.60% / 91.89%                    | 92.6 / 88.8 s             | 149.1 / 156.9 s           |
| DeepSeek-V4-Pro, kvlab, about 2 TB a node each       | 32 sessions | 15,459              | 15,474              | +0.1%                 | within 0.1%                        | –                         | 24.03 / 23.43 s           |
| **Pools never filled: tests loading, not capacity**                    |             |                     |                     |                       |                                    |                           |                           |
| DeepSeek-V4-Pro, kvlab, 4 TB DRAM / 2.8 TB NVMe                        | 32 sessions | 17,067              | 17,235              | +1.0%                 | within 0.1%                        | –                         | 14.80 / 14.17 s           |
|                                                                        | 64 sessions | 20,262              | 20,502              | +1.2%                 | within 0.1%                        | –                         | 54.96 / 52.71 s           |
| **Without the checkpoint changes, superseded at 70 sessions by the kvlab row above** |             |                     |                     |                       |                                    |                           |                           |
| Kimi-K3, kvlab, 1.8 TB each                                            | 8 sessions  | 6,791               | 6,570               | −3.2%                 | 95.57% / 95.49%                    | 1.59 / 1.74 s             | 9.70 / 11.17 s            |
|                                                                        | 16 sessions | 9,493               | 9,377               | −1.2%                 | 94.20% / 94.25%                    | 3.04 / 3.36 s             | 17.05 / 17.37 s           |
|                                                                        | 32 sessions | 10,424              | 9,873               | −5.3%                 | 93.58% / 93.64%                    | 27.07 / 28.50 s           | 49.57 / 56.39 s           |
|                                                                        | 70 sessions | 7,677               | 6,639               | −13.5%                | 91.10% / 90.49%                    | 96.80 / 116.06 s          | 165.03 / 211.27 s         |

Layer-wise loading against vLLM’s asynchronous load, Kimi-K3 with an NVMe pool. The rows with less GPU cache shrink GPU memory so that more requests load from the pool.

| **Kimi-K3, NVMe pool**            | **Requests that load from the pool** | **Mean TTFT, async / layer-wise** | **Prefill tok/s, async / layer-wise** |
|-----------------------------------|--------------------------------------|-----------------------------------|---------------------------------------|
| 16 sessions                       | 2.5%                                 | 7.6 / 7.2 s                       | 7,791 / 7,764                         |
| 32 sessions                       | 5.3%                                 | 32.7 / 32.5 s                     | 9,935 / 9,986                         |
| 70 sessions                       | 13%                                  | 128.1 / 130.3 s                   | 10,699 / 10,583                       |
| 16 sessions, 8.2 GiB of GPU cache | 42%                                  | 7.26 / 7.20 s                     | 7,661 / 7,601                         |
| 16 sessions, 5.7 GiB of GPU cache | 65%                                  | 7.54 / 7.30 s                     | 7,637 / 7,508                         |
| 8 sessions, 4.5 GiB of GPU cache  | 48%                                  | 3.75 / 3.87 s                     | not recorded                          |

## Sources

1. <span id="ref-1"></span>MoRI UMBP: [overview](https://github.com/ROCm/mori/blob/main/docs/umbp-overview.md), [pure-SSD mode guide](https://github.com/ROCm/mori/blob/main/src/umbp/doc/pure-ssd-mode.md) and [runtime settings](https://github.com/ROCm/mori/blob/main/src/umbp/doc/runtime-env-vars.md); pull requests [\#779](https://github.com/ROCm/mori/pull/779), [\#882](https://github.com/ROCm/mori/pull/882), [\#901](https://github.com/ROCm/mori/pull/901), [\#902](https://github.com/ROCm/mori/pull/902) and [\#967](https://github.com/ROCm/mori/pull/967).
2. <span id="ref-2"></span>vLLM: the UMBP connector series starting at [\#61789](https://github.com/vllm-project/vllm/pull/61789); Kimi-K3 on ROCm™, [\#58743](https://github.com/vllm-project/vllm/pull/58743) and [\#59966](https://github.com/vllm-project/vllm/pull/59966); DeepSeek-V4-Pro in standalone mode, [\#63002](https://github.com/vllm-project/vllm/pull/63002); connector pull requests [charxwu/vllm#7](https://github.com/charxwu/vllm/pull/7) and [\#19](https://github.com/charxwu/vllm/pull/19); [Kimi K3 on MI355X](https://vllm.ai/blog/2026-10-16-kimi-k3-mi355x).
3. <span id="ref-3"></span>AMD, [MoRI UMBP Empowers AMD Instinct MI355X GPU to Demonstrate Token-per-Dollar TCO Leadership on the Public AgentX Leaderboard](https://www.amd.com/en/developer/resources/technical-articles/2026/mori-umbp-empowers-amd-instinct-gpus.html), October 2026; and the July 2026 [technical article](https://www.amd.com/en/developer/resources/technical-articles/2026/rebuilding-agentic-ai-for-amd-gpu.html) on Kimi’s agentic serving with MoRI and UMBP.
4. <span id="ref-4"></span>DeepSeek, [DeepSeek V4.1 Flash Inference Deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md): three-day KV retention, 14.4 TB of SSD per GPU, layer-wise loading and bounded replay.
5. <span id="ref-5"></span>SGLang: the KV-load overlap patch our SGLang runs needed, not upstream: the top four commits of the [umbp-ci48-cp-prefetch](https://github.com/TianDi101/sglang/tree/umbp-ci48-cp-prefetch) branch of TianDi101/sglang, at commit `031f5ce3`, on upstream commit `1e86143bf`, built on the `lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261007` image.
6. <span id="ref-6"></span>SemiAnalysis, [AgentX v1.0 methodology](https://inferencex.semianalysis.com/agentx), part of InferenceX.
7. <span id="ref-7"></span>WEKA, [kvlab](https://github.com/weka/kvlab) and [Meet kvlab](https://www.weka.io/article/meet-weka-kvlab): the replay benchmark and its 40-day corpus.
8. <span id="ref-8"></span>Workloads: [SemiAnalysis aiperf](https://github.com/SemiAnalysisAI/aiperf) `89b21867` with a backport of ai-dynamo/aiperf#1348; AgentX traces [semianalysisai/cc-traces-weka-062126](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) (SemiAnalysis’s Claude Code traces; despite the name, not the kvlab corpus); the kvlab 40-day corpus [wekaio/replay-40d-oct14-337a03810637](https://huggingface.co/datasets/wekaio/replay-40d-oct14-337a03810637), converted to 3,164 aiperf trace files.
9. <span id="ref-9"></span>DatacenterDisk, [AI Data Center Memory CAPEX Calculator](https://datacenterdisk.com/tools/ai-memory-capex-calculator/), median listing prices on 13 November 2026 (\$42.15 per GB of DDR5 RDIMM, \$666.04 per TB of enterprise SSD).
10. <span id="ref-10"></span>Other systems: [vLLM tiered KV offloading](https://vllm.ai/blog/2026-09-10-tiered-kv-offloading), [SGLang HiCache](https://www.lmsys.org/blog/2025-09-10-sglang-hicache), [Mooncake](https://kvcache.ai/blog/scaling-kv-cache-beyond-memory/).

<script>
(function () {
  // Resize embedded interactive figures to their content height (they post it when framed).
  window.addEventListener('message', function (event) {
    var d = event.data;
    if (!d || d.type !== 'vllm-embed-resize' || typeof d.height !== 'number') return;
    var frames = document.querySelectorAll('iframe.vllm-embed');
    for (var i = 0; i < frames.length; i++) {
      if (frames[i].contentWindow === event.source) {
        frames[i].style.height = Math.ceil(d.height) + 'px';
      }
    }
  });
})();
</script>
