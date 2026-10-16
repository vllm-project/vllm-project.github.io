# FP8-128 conversion notes

The tuned checkpoint was built offline from the published DeepSeek V4.1 Flash
checkpoint. This was a lossy second quantization to make block-scaled FP8
W8A8 paths eligible on the tested H20/vLLM/FlashInfer stack; it was not a
general recommendation to convert all model weights to FP8.

| Tensor group | Input | Output | Count / handling |
|---|---|---|---|
| Routed expert GEMMs, including draft experts | MXFP4 with E8M0 scales | FP8 E4M3, 128×128 blocks, FP32 scales | 47,232 weight tensors |
| Eligible non-expert GEMMs (attention, shared expert, indexer, Engram projections) | MXFP8 with 32×32 E8M0 scales | FP8 E4M3, 128×128 blocks, FP32 scales | 355 weight/scale pairs |
| Engram embedding tables | FP8 E4M3 with E8M0 lookup scales | Preserved | Offloaded to pinned host memory at serving time |
| Embeddings, router, norms, and small/sensitive tensors | Mostly BF16 | Preserved | Not converted |

The conversion checks reported 2.69% global relative L2 error for experts and
2.64% for converted non-experts. These are tensor-level reconstruction
metrics, not model accuracy. The 500-item GSM8K check of the Humming W4A8
baseline and tuned FP8 recipe is reported in the article and its summary.

Storage accounting: the original checkpoint was 475.24 GiB; the FP8 expert
overlay was 519.03 GiB; the assembled converted checkpoint was 718.60 GiB.
These are on-disk sizes. The 519.03 GiB expert overlay alone implies a
64.88 GiB/rank ideal split across eight ranks, before runtime state and
replicas. In the measured service startup, loaded weights were 70.94 GiB per
rank and only 3.57 GiB of KV-cache memory remained. Do not treat the overlay
size as GPU residency or add profiler counters from different accounting
points.

The private conversion implementation and source checkpoint are not included
in this public bundle. Reproduction requires the same source weights, tensor
mapping, block-scale convention, and vLLM/FlashInfer support; these aggregate
notes alone are not a drop-in conversion tool.
