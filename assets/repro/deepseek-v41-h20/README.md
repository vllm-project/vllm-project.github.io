# DeepSeek V4.1 Flash H20 reproduction artifacts

This bundle separates the isolated EP8 MoE operator ablation from complete
serving-recipe measurements. It includes the FP8 conversion notes, a
machine-readable serving summary, and a paired GSM8K quality sanity check for
the original MXFP4 checkpoint with Humming W4A8 and the converted FP8-128 service.

## Measured environment

Both serving Pods used eight NVIDIA H20 GPUs on separate nodes (driver
595.58.03), the same `docker.m.daocloud.io/vllm/vllm-openai:v0.30.0` image
(`sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`),
vLLM 0.30.0, PyTorch 2.13.0+cu130, CUDA 13.0 (PyTorch runtime), and installed
FlashInfer 0.6.18.post1. The installed package did not expose a source commit
in its package metadata. The related [FlashInfer PR #5560](https://github.com/flashinfer-ai/flashinfer/pull/5560)
had head commit `f0880d6cb67a3bd9a9d57d72d1f956e14dd0ae47` when this record
was prepared; that commit identifies the upstream PR, **not** the installed
package. The tuned service loaded a dispatch overlay through `LD_PRELOAD`;
the deployed `dispatch.so` SHA-256 was
`d1f8853fc445c5791c3b4c6231f3716a4df2d6f9dd23b9580e1a1ae12249a63a`.

| Configuration | Humming W4A8 baseline | Tuned FP8 recipe |
|---|---|---|
| Checkpoint | Original MXFP4 experts | Converted FP8 E4M3, 128×128 scales |
| Parallelism | TP8 | TP8 + EP8; 16 redundant EPLB experts |
| MoE backend | `humming` | `flashinfer_cutlass` plus dispatch overlay |
| Speculative decoding | DSpark, 2 tokens | DSpark, 2 tokens |
| Memory | GPU utilization 0.95; KV dtype auto; block size 64 | Same; Engram CPU offload |
| Serving limits | max model len 262,144; max seqs 256; max batched tokens 8,192 | Same; max queued requests 256 |
| Scheduling | Default | Async; chunked prefill; prefix caching; long-prefill threshold 6,144 |
| CUDA Graph | Image default | Capture limit 768 |
| Other tuned settings | — | FlashInfer attention; `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS` for fused MoE GEMM1/GEMM2 |

This is the current deployment configuration used for the three-run serving
sweeps. It was read from the live Pod commands and package imports; it does
not describe the separate source-built operator benchmark environment.

## Quality comparison

`quality_eval.mjs` samples 500 rows from the pinned `openai/gsm8k` main/test
split (revision `7cf1290ed87c28a31f867e0f47a7cb62a61d502e`) using seed
`20260926`. It sends each identical prompt to the Humming W4A8 baseline and
tuned FP8 OpenAI-compatible endpoints with temperature 0, top-p 1, seed 42,
and 2,048 max output tokens. It compares the final numeric answer with the
GSM8K `####` answer and writes per-item results and a summary. This is a small
accuracy sanity check, not a full evaluation or proof of parity.

With local port-forwards open to the two services:

```sh
BASELINE_URL=http://127.0.0.1:18080 \
TUNED_URL=http://127.0.0.1:18081 \
node quality_eval.mjs
```

The run produces `quality-results.jsonl` and `quality-summary.json` in the
current directory. It uses Node.js built-ins plus `curl` to honor the local
network proxy when downloading the pinned public dataset. Per-item output
contains only answer numbers, correctness, finish reason, and latency; it does
not retain model reasoning or full completions. Truncated outputs are excluded
from accuracy denominators.

## GuideLLM serving sweep

The serving sweep used GuideLLM 0.7.4 with two clients running at the same
time, one against each already-running OpenAI-compatible service endpoint.
The clients used `CUDA_VISIBLE_DEVICES=""` and `OMP_NUM_THREADS=2`. Each tier
was run three times with concurrent streams `[1, 8, 32, 64, 128, 256]` and
a 10-second warmup. The article shows only concurrency 1–64 for Prefill; the
full Prefill sweep also reached 256.

| Profile | Prompt / output | Duration | Sampling |
|---|---|---:|---|
| Decode | 4,096 unique short prompts per tier, exactly 1,024 output tokens | 90 s including 10 s warmup | temperature 0.7, seed 42, ignore EOS |
| Prefill | Exactly 1,024 server-side input tokens, exactly one output token | 60 s including 10 s warmup | temperature 0, unique prompts; prefix reuse checked |

The API backend used the `openai_http` kind, `/v1/completions`, streaming,
120-second request timeout, model alias `deepseek-v4-1-flash-tp8`; the Decode
request body set `min_tokens=1024`, while Prefill set `min_tokens=1`. The
server-side `generation_tokens_total` and `prompt_tokens_total` counters were
sampled every two seconds. Their increments were interpolated to GuideLLM's
measurement window boundaries, divided by the window duration, then averaged
across three runs; the reported spread is the sample standard deviation.
All tiers had zero request errors, zero prefix-cache hits, and no measured
preemptions. Per-run and aggregated throughput values are in
[`serving-results.csv`](serving-results.csv). Isolated operator measurements
and the tuned c128 rank-0 profile are in [`operator-results.csv`](operator-results.csv)
and [`profile-c128-rank0.csv`](profile-c128-rank0.csv).

GuideLLM accepts scenarios through `guidellm run --config <scenario.json>`.
The actual expanded scenarios, per-request prompts, GuideLLM JSON reports, and
two-second service metric samples remain in the two benchmark client Pods.
They are not included in this public bundle, so the CSV supports checking
the reported calculations rather than replaying identical prompt bytes.

## Conversion record

See [`conversion-notes.md`](conversion-notes.md) for tensor groups, scale
formats, tensor-level reconstruction errors, storage accounting, and the
limits of what is included. The private conversion utility and checkpoint
are intentionally not redistributed.

## Performance reproduction

The article describes the GuideLLM 0.7.4 Decode and fixed-1K Prefill sweeps,
their concurrency levels, and the captured profile summaries. Serving numbers
are whole-recipe comparisons against the Humming W4A8 baseline; they do not
isolate the MoE dispatch change.
The dispatch microbenchmark is a separate controlled operator ablation. Do
not multiply the two speedups or present them as one causal result.
[`operator-results.csv`](operator-results.csv) stores six-decimal latency
medians; speedup is global-M latency divided by per-expert-M latency, rounded
to two decimals.

Internal cluster names, addresses, credentials, and private filesystem paths
are intentionally excluded. Exact startup arguments and per-request raw
GuideLLM outputs are not in this public-safe bundle.
