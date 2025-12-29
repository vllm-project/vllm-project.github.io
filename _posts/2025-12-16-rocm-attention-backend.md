---
layout: post
title: "Performance boost on ROCm Backend with new Attention Design"
author: "AMD and Embedded LLM"
image: /assets/figures/ptpc/PTPC-tumbnail.png
math: true
---

## Introduction

### Attention backends on AMD ROCm backends 
Attention backend is one of the most complicated module in vLLM. Different models have different attention structure. Generally, the attention modules of common large language models can be categorized into two main types: Multi-Head Attention (MHA) and Multi-Latent Attention (MLA). Models like [Qwen](https://huggingface.co/Qwen/Qwen3-235B-A22B)/[Llama](https://huggingface.co/meta-llama/Meta-Llama-3-70B) are using MHA module, while [Deepseek](https://huggingface.co/deepseek-ai/DeepSeek-R1)/[Kimi](https://huggingface.co/moonshotai/Kimi-K2-Thinking) are using MLA module. Besides the model structure, different hardware platforms have different attention backend implementations. Even in the same hardware platform, there are some different attention backends to support different user scenarios and deliver different performance. In this post, we will introduce the AMD's ROCm attention backend in vLLM. We will elaborate the principles of the attention backend implementation, especially the ROCM_AITER_FA and ROCM_AITER_MLA, along with the corrsponding benchmark performance. 

In this post, we will elaborate the implementation principles on AMD ROCm Attention backends and provide a reference for vLLM users to select the best attention backends on AMD platforms according to different user scenarios. In vLLM, for AMD ROCm backend, there are four MHA implementations (`TRITON_ATTN`, `ROCM_AITER_UNIFIED_ATTN`, `ROCM_ATTN`, and the `ROCM_AITER_FA`) and three MLA implementations (`TRITON_MLA`, `ROCM_AITER_TRITON_MLA` and `ROCM_AITER_MLA`). vLLM provides the argument of `--attention-backend` to select the corresponding attention backends which can be summarized as in the following table for the attention selection on AMD's platform.

| Category | Backend | How to enable |
|:----------|:----------------|:--------------|
| MHA | TRITON_ATTN | --attention-backend TRITON_ATTN |
| MHA | ROCM_AITER_UNIFIED_ATTN | --attention-backend ROCM_AITER_UNIFIED_ATTN |
| MHA | ROCM_ATTN | --attention-backend ROCM_ATTN |
| MHA | ROCM_AITER_FA | --attention-backend ROCM_AITER_FA |
| MLA | TRITON_MLA | --attention-backend TRITON_MLA |
| MLA | ROCM_AITER_TRITON_MLA | --attention-backend ROCM_AITER_TRITON_MLA |
| MLA | ROCM_AITER_MLA | --attention-backend ROCM_AITER_MLA |

### Attention computation for a bucket of tokens

In each inference step for an online serving, a bucket of tokens possibly containing prefill/extend/decode are batched together for forward step computation. The extend frame happens in the chunked-prefill scenario which is usually results from the Prefix-Cache hit or Long-Context usage. Figure 1 illustrates an example of online serving contiguous batching with `--max-concurrency=5`, among which the `request3`/`request5` contain large Input Sequence Length (ISL) and trigger the chunked-prefill feature. In the following example, inference forward step 1~5 have different numbers of prefill/extend/decode tokens. In step4, the tokens in the bucket contains the prefill/extend/decode at the same time. Actually, the sequence of the prefill/extend/decode are randomly arranged in a bucket for real online serving, which is not be guaranteed in a fixed order. 

In each step, the attention backend will receives tokens features from prefill/extend/decode, and generate the corresponding attention computation results. Different attention backend have different implementation recipes for this computation, some of them compute the attention of prefill/extend/decode tokens in a single path and we usually call it as the `Unified Attention` backend like `TRITON_ATTN`, `ROCM_AITER_UNIFIED_ATTN`, `ROCM_ATTN`, while the others seperate the tokens and compute the prefill/extend/decode in different paths like `ROCM_AITER_FA`.

![Contiguous-Batching Example](/assets/figures/2025-12-16-rocm-attention-backend/contiguous-batching.png)

## Multi-Head Attention Backend

**Design Overview** 
![AITER FA Backend](/assets/figures/2025-12-16-rocm-attention-backend/vLLM-Attention.png)

### The Unified Attention backend 

In vLLM AMD ROCm backend, there are three unified attention implementations The [Triton Attention Backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py), The [ROCm AITER Unified Attention Backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_aiter_unified_attn.py) and The [ROCm Attention Backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py). They all compute the prefill/extend/decode in the same path while call different attention kernels. 

- The [Triton Attention Backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py) is the default attention backend which invokes the [vLLM triton source kernel](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py#L57) for computation.
- The [ROCm AITER Unified Attention Backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_aiter_unified_attn.py) uses the [AITER triton kernel](https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/unified_attention.py#L55) for computation.
- The [ROCm Attention Backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py) use the xxxx.


Figure 1(a) illustrates the detailed principle of the unified attention. The server receives many requests from the client and the schduler contiguously batches tokens from prefill/extend/decode toghether, among which the extended happens in chunked prefill scenario. The tokens are batched together and Query/Key/Value features are computed at first, which will be used in Attention backend. 

In `unified attention backend`, the key and value from all prefill/extend/decode in current step will be written into the KV cache buffer at first, then the query together with the KV Cache (containing both the current and history key/value) are used as inputs for attention computation with a single kernel. 

Below is the pseudo-code illustrating how unified attention works:
```python
def forward():
    # Stage 1: Save the Key/Value of all tokens in the bucket into KV-Cache
    reshape_and_cache_flush(new_key, new_value, ...)  
    # Stage 2: Read the KV-Cache for attention computation
    unified_attention_kernel(new_query, KV-Cache, ...)
```
The implementation recipe of `unified attention` is concise and general, which launches only one kernel for all prefill/extend/decode paths. However, prefill/extend and decode are computation and memory-access bounded tasks. It's not easy for a single kernel to handle two differnt forms of task.

### The ROCM AITER FA backend 

**Fundemental Design**

The `ROCM AITER FA` backend's implemention relies on two fundamental design compared with the `Unified Attention`.

- ***Request Routing***: It dynamically categorizes and processes the incoming request tokens into three distinct paths: Decode, Extend and Prefill. This allows for specialized and optimized kernels to be applied to each specific task.

- ***Hardware-Optimized KV Cache Layout***: The ROCm backend adopts the preshuffled the KV cache layout instead of the standard version to leverage the performant kernel in AITER. The preshuffled layout represents as:

    ```python
    k_cache: [num_blocks, num_heads, head_dim // x, block_size, x]
    v_cache: [num_blocks, num_heads, block_size // x, head_dim, x]
    ```

**Performance-Optimized Processing Paths**

Requests are reordered and processed in a decode:extend:prefill sequence, with each type taking a distinct, highly performant kernels with different Key/Value layout, which can help unlock significant AMD GPU performance. The detailed optimization recipes for different paths are elaborated as follow. 

- **Decode Path**: Leverages the shuffled KV cache layout directly. A custom `reshape_and_cache_flush` operator ensures the cache always in the shuffled layout, allowing the attention backend to call AITER performant paged_attention kernel with zero layout conversion overhead. Compared with the standard KV Cache layout, the shuffled layout can help boost **15-20% decode throughput improvement**.

- **Prefill Path**: The Qeury/Key/Value are in the standard `[num_tokens, num_heads, head_dim]` layout to align with the highly optimized AITER MHA kernel and avoid any extra memory copy operations.

- **Extend Path**: This is more challenging than Prefill/Decode paths. The new tokens should compute the attention with the context tokens, which is stored in KV-Cache buffer in shuffled layout as mentioned above. The shuffled layout is incompatible with AITER MHA kernel that we want to use for long-context computation at extend phase. Then we will insert an extra KV Cache fetching operator to fetch and convert the context Key/Value to the standard layout. In addition, considering the extra buffer used in KV Cache fetching, the long context will be chunked into serveral segments for computation.

    **Pseudo-Code for the Extend Path** :
    ```python
    def extend_forward():
        # Stage 1: Attention for new tokens
        flash_attn_varlen_func()  # calling AITER MHA

        # Stage 2: Context Chunk Loop Processing
        for chunk in context_chunks: 
            cp_mha_gather_cache()
            flash_attn_varlen_func()  # calling AITER MHA
            merge_attn_states()

        # Stage 3: Get the final result
        merge_attn_states()
    ```

## Multi Latent Attention Backend
For model with MLA structure will choose this attention backend. In vLLM, the general MLA backend is the `TRITON_MLA` backend and will be enabled by default for ROCm platform. Besides, we have defined a new backend as `ROCM_AITER_MLA` backend to leverage the performant kernel in AITER.

### TRITON MLA Backend
The implementation principle of the `TRITON_MLA` backend are similar to the `ROCM_ATIER_FA` backend. It seperates the decode path from the prefill/extend path and use different implementation recipe. For prefill/extend path, the non-absorbed recipe is adopt and it use the standard MHA for computation. However, for decode, it use the absorbed recipe and use the MLA kernel for computation. The `TRITON_MLA` backend use the vLLM default Triton kernel for MHA computation in prefill phase, while it doesn't provide the decode implementation and leave it to the `ROCM_AITER_MLA` backend.

**Pseudo-Code for the Prefill/Extend Path** :
```python
def _forward_prefill():
    # Stage 1: Attention for new tokens
    _run_prefill_new_tokens()  

    # Stage 2: for extend path, Context Chunk Loop Processing  
    for chunk in context_chunks:
        gather_and_maybe_dequant_cache()
        _run_prefill_context_chunk()  # _run_prefill_context_chunk_fa in rocm backend
        merge_attn_states()

    # Stage 3: Get the final result
    merge_attn_states()
```

### ROCM AITER MLA Backend

`AiterMLABackend` is a performance-optimized attention backend designed for models like DeepSeek that utilize MLA for their attention computation. This backend delivers significant inference speedups on AMD hardware through deeply optimized assembly kernels, `flash_attn_varlen_func` for the prefill phase, `mla_decode_fwd` for the decode phase

Beyond raw kernel performance, this backend inherits the full feature set of the FlashMLABackend, including PIECEWISE_AND_FULL CUDA graph support and MTP support.one other advantage is its near-identical performance across virtually any KV cache block size. In practice, this means you can treat every token as prefix cache without worrying about the performance penalties typically associated with fine-grained caching—enabling highly efficient long-context and multi-turn deployments.


## Performance Benchmark

### MHA attention performance comparision
### MLA attention performance comparision



## Get Started


