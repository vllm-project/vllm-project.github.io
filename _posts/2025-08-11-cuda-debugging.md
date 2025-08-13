---
layout: post
title: "CUDA Core Dump: An Effective Tool to Debug Memory Access Issues and Beyond"
author: "Kaichao You"
image: /assets/logos/vllm-logo-text-light.png
---

# Introduction

Have you ever felt you are developing attention kernel and your tests oftens runs into illegal memory access (IMA for short) and you have no idea how to debug? We definitely felt this pain again and again while working on vLLM, a high-performance inference engine for LLM models. 

If you are one of the developers who have faced this issue, this blog is for you! We will uncover some of advanced debugging techniques we use that can help users debug complicated issues in vLLM, such as IMA.

For example, here’s an error from PyTorch:

```text
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

The challenging bit here is: CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. In our experience the python stack traces for these types of exceptions are basically **always incorrect and pretty worthless**. To resolve this the error message suggests adding `CUDA_LAUNCH_BLOCKING=1` when running the code. However, there are still two problems:

1. Many people launch CUDA kernels using the `kernel<<<>>>` syntax without adding error checking for the kernel launch status, for example, this [code](https://github.com/pytorch/pytorch/blob/5e320eea665f773b78f6d3bfdbb1898b8e09e051/aten/src/ATen/native/cuda/SortStable.cu#L117). In such cases, even with `CUDA_LAUNCH_BLOCKING=1`, it’s still impossible to locate the faulty kernel.
2. If the illegal memory access occurs inside a kernel within a CUDA graph, then even with `CUDA_LAUNCH_BLOCKING=1`, we can only see that there’s an issue when launching the CUDA graph, but still cannot pinpoint the exact kernel that failed.

To accurately pinpoint this kind of problem, we need to react immediately when an illegal memory access occurs. Of course, this isn’t something users can do directly — it must be supported by the CUDA driver itself.

The [CUDA core dump functionality](https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-core-dump-support), is exactly designed for this purpose. It allows the CUDA driver to dump the GPU state when an illegal memory access occurs, so that users can analyze the GPU state later to find out which kernel caused the issue and what the illegal memory access was.

# What is a Core Dump?

A GPU is essentially a massively parallel processor, and many of its concepts can find counterparts in CPUs.

A [core dump](https://en.wikipedia.org/wiki/Core_dump) is a feature jointly provided by the CPU and the operating system. When a program crashes during execution, the operating system can record the program's memory data, runtime state, and other information for subsequent analysis and debugging. A program crash is a hardware-level concept. When the CPU encounters an error while executing certain instructions, it enters a `trap` state. At this point, the operating system takes over the program and executes the corresponding exception handling procedure (by default, this simply terminates the program, but options can be configured to generate a core dump for analysis. For example, `ulimit -c 1` can enable core dump generation, and `echo "core.%e.%p" > /proc/sys/kernel/core_pattern` can specify the path for the core dump file).

By analogy, the core dump functionality on GPUs requires collaboration between GPU hardware and GPU drivers. When a thread on the GPU crashes during execution, the GPU hardware needs to trigger an exception and pass it to the GPU driver, which then immediately handles the exception. However, according to [forum discussions](https://forums.developer.nvidia.com/t/difference-in-error-handling-between-driver-api-and-runtime-api/336389), the default behavior of the GPU driver when handling exceptions is to mark the current CUDA context as unusable, rather than terminating the program.

# How to Enable CUDA Core Dump

Enabling CUDA core dump is very straightforward; you just need to set the `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` environment variable. However, for a smoother experience, you should also set a few additional environment variables:

1. By default, the CUDA core dump saves the coredump file in the current directory without printing the file path. You can enable the `CUDA_COREDUMP_SHOW_PROGRESS=1` environment variable to display the progress and details of the coredump procedure. Most importantly, it shows the path of the coredump file after the procedure is complete, making it easier for subsequent debugging and analysis.
2. Many tasks run inside containers, and when a task fails, the container is destroyed, making it impossible to retain the coredump file. In such cases, you can use the `CUDA_COREDUMP_FILE` environment variable to specify a file path template for the coredump file. For example, you can store the coredump file in a persistent storage directory: `CUDA_COREDUMP_FILE="/persistent_dir/cuda_coredump_%h.%p.%t"`, where `%h` is the hostname, `%p` is the process ID, and `%t` is the timestamp of the coredump.
3. By default, the coredump procedure saves the entire GPU context. For programs like large model inference that occupy almost all GPU memory, a full coredump is impractical (hundreds of GiB of data). You can use the `CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory'` environment variable to skip saving GPU memory, shared memory, and local memory, thereby reducing the size of the coredump file.

The documentation also mentions that adding `skip_abort` to `CUDA_COREDUMP_GENERATION_FLAGS` prevents the CPU process from aborting after the coredump is complete. This allows the CPU process to add its own error trace, providing more debugging information. However, experiments have shown that this feature has a significant [bug](https://forums.developer.nvidia.com/t/cuda-core-dump-with-skip-abort-will-ignore-an-illegal-memory-access-error/341802/3), which may cause illegal memory access errors on the GPU to be ignored. In such cases, subsequent code may continue to run normally, but the program's memory data might already be corrupted. This is unacceptable for training tasks and undesirable for inference tasks. Therefore, this feature is generally unreliable and not recommended.

Additionally, the documentation states that enabling `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` not only enables CUDA core dump but also generates a CPU coredump by default. However, in practice, we find that the CPU coredump contains little useful information and is difficult to analyze.

If you want live data for debugging, you can also enable `CUDA_DEVICE_WAITS_ON_EXCEPTION=1` environment variable, which does not use CUDA core dump, but stops GPU execution immediately when an exception occurs, and hangs there, waiting for users to attach a debugger (like cuda-gdb) to inspect the GPU state, where the full GPU memory is still intact. However, this approach is less automatic and requires more manual intervention.

In summary, when using the CUDA core dump feature, it is recommended to use the following combination of environment variables:

`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory' CUDA_COREDUMP_FILE="/persistent_dir/cuda_coredump_%h.%p.%t"`

# Example of Using CUDA Core Dump

Let's use some code to verify the effectiveness of CUDA core dump.

## Debugging Improper Kernel Launch

```cpp
// test.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking macro
#define cuda_check(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel with illegal memory access - accesses memory beyond allocated bounds
__global__ void illegalMemoryAccessKernel(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // This will cause illegal memory access - accessing beyond allocated memory
    // We allocate 'size' elements but access up to size * 2
    if (idx < size * 2) {  // Access twice the allocated size
        data[idx - 1000000000] = idx;   // This will cause illegal access for idx == 0
    }
}

// Kernel with illegal memory access - accesses memory beyond allocated bounds
__global__ void normalKernel(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // This will cause illegal memory access - accessing beyond allocated memory
    // We allocate 'size' elements but access up to size * 2
    if (idx < size) {  // Access twice the allocated size
        data[idx] = idx;   // 
    }
}

int main() {
    printf("CUDA Illegal Memory Access Test\n");
    printf("===============================\n\n");
    
    int size = 100;
    int* h_data = (int*)malloc(size * sizeof(int));
    int* d_data;
    
    // Initialize host memory
    for (int i = 0; i < size; i++) {
        h_data[i] = 0;
    }
    
    // Allocate device memory
    cuda_check(cudaMalloc(&d_data, (unsigned long long)(size) * sizeof(int)));
    cuda_check(cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel with illegal memory access
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    printf("Launching kernel with out-of-bounds access...\n");
    illegalMemoryAccessKernel<<<numBlocks, blockSize>>>(d_data, size);

    normalKernel<<<numBlocks, blockSize>>>(d_data, size);

    cuda_check(cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 5; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n");
    
    // Synchronize to catch any runtime errors
    cuda_check(cudaDeviceSynchronize());
    
    printf("Test completed.\n");
    
    // Cleanup
    cuda_check(cudaFree(d_data));
    free(h_data);
    
    return 0;
}
```

This code launches two kernels consecutively (`illegalMemoryAccessKernel` and `normalKernel`). During normal execution, you would encounter an error message: `CUDA Error at test.cu:62 - cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost): an illegal memory access was encountered`, and the error would only be detected in the return value of `cudaMemcpy`. Even with `CUDA_LAUNCH_BLOCKING=1`, it is still impossible to identify the specific kernel that caused the error.

By adding the CUDA core dump-related environment variables, we can observe:

```text
[00:40:46.606413] coredump: SM 123/132 has finished state collection
[00:40:46.606419] coredump: SM 124/132 has finished state collection
[00:40:46.611453] coredump: Detected an exception of type CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS (14)
[00:40:46.611458] coredump:   - Device: 0
[00:40:46.611460] coredump:   - SM: 124
[00:40:46.611462] coredump:   - Warp: exception was detected after the warp has exited
[00:40:46.611465] coredump:   - PC 0x7f31abb9f6d0
[00:40:46.611467] coredump: SM 125/132 has finished state collection

[00:40:46.806153] coredump: Writing ELF file to /tmp/cuda_coredump_xxx.1799919.1754898045

[1]    1799919 IOT instruction (core dumped)  CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1 = = ./test3
```

After a GPU thread triggers an illegal memory access, the CPU immediately generates a coredump file and then triggers a CPU exception, directly terminating the program. At this point, we obtain a coredump file `/tmp/cuda_coredump_xxx.1799919.1754898045`. We can open it using `cuda-gdb` (command: `target cudacore /path/to/coredump_file`, where `cudacore` refers to the coredump on CUDA):

```bash
$ cuda-gdb
(cuda-gdb) target cudacore /tmp/cuda_coredump_xxx.1799919.1754898045
Opening GPU coredump: /tmp/cuda_coredump_xxx.1799919.1754898045

CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7f31abb9f6d0  illegalMemoryAccessKernel(int*, int)
[Current focus set to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 124, warp 0, lane 0]
#0  0x00007f31abb9f6e0 in illegalMemoryAccessKernel(int*, int)<<<(1,1,1),(256,1,1)>>> ()
```

We can clearly see that the exception is caused by `illegalMemoryAccessKernel` at `kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 124, warp 0, lane 0`.

## Debugging Kernel Exceptions in CUDA Graphs

Here’s a more complex example where an illegal memory access kernel is inserted into a CUDA graph:

```python
# core_dump.py
import torch
import torch.nn as nn

from dataclasses import dataclass

@dataclass
class CupyWrapper:
    data_ptr: int
    size_in_bytes: int

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.size_in_bytes,),
            "typestr": '|u1',
            "data": (self.data_ptr, False),
            "version": 3,
        }

def from_buffer(data_ptr: int, size_in_bytes: int) -> torch.Tensor:
    out = torch.as_tensor(CupyWrapper(data_ptr, size_in_bytes))
    assert data_ptr == out.data_ptr(), "not zero-copy convert, something must be wrong!"
    return out


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # First layer: [B, 10] -> [B, 20] with ReLU activation
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        # Second layer: [B, 20] -> [B, 30]
        self.layer2 = nn.Linear(20, 30)
        self.num_called = 0

    def forward(self, x):
        # Input shape: [B, 10]
        x = self.layer1(x)  # [B, 20]
        x = self.relu(x)    # [B, 20] with ReLU activation
        self.num_called += 1
        if self.num_called > 1:
            y = from_buffer(x.data_ptr(), x.numel() * 1024 * 1024)
            # will trigger illegal memory access
            y.fill_(1)
        x = self.layer2(x)  # [B, 30]
        return x


# Example usage
if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create the model and move to CUDA
    model = NeuralNetwork().to(device)
    
    # Create sample input with batch size B=4 and move to CUDA
    batch_size = 4
    input_tensor = torch.randn(batch_size, 10).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input device: {input_tensor.device}")
    
    # Forward pass
    with torch.no_grad():
        # warmup
        output = model(input_tensor)
        # capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = model(input_tensor)
        # replay graph
        g.replay()
    
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    
    print(f"Output: {output.sum()}")

    # Print model summary
    print("\nModel architecture:")
    print(model)
    
    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    
    # Verify model is on CUDA
    print(f"Model device: {next(model.parameters()).device}")
```

Direct execution results in the following error:

```text
Using device: cuda
Input shape: torch.Size([4, 10])
Input device: cuda:0
Output shape: torch.Size([4, 30])
Output device: cuda:0
Traceback (most recent call last):
  File "core_dump.py", line 76, in <module>
    print(f"Output: {output.sum()}")
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

The error is not printed until the `output.sum()` triggers a device synchronization and reveals the illegal memory access. However, we don't know which kernel caused the illegal memory access since cuda kernels are executed asynchronously.

After adding `CUDA_LAUNCH_BLOCKING=1`, the error message changed to:

```text
Using device: cuda
Input shape: torch.Size([4, 10])
Input device: cuda:0
Traceback (most recent call last):
  File "core_dump.py", line 71, in <module>
    g.replay()
  File "/uv_envs/py310/lib/python3.10/site-packages/torch/cuda/graphs.py", line 88, in replay
    super().replay()
RuntimeError: CUDA error: an illegal memory access was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

It can be inferred that an exception occurs in a kernel within the CUDA graph. However, conventional methods can only provide information up to this point.

By adding the environment variables `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory' CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t"`, we can clearly identify the kernel that caused the error:

```text
(cuda-gdb) target cudacore /tmp/cuda_coredump_flow-matic.1929094.1754901120
Opening GPU coredump: /tmp/cuda_coredump_flow-matic.1929094.1754901120

CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7fc2afba5e30  void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul>)
[Current focus set to CUDA kernel 0, grid 9, block (17454,0,0), thread (0,0,0), device 0, sm 0, warp 1, lane 0]
#0  0x00007fc2afba5e70 in void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul>)<<<(40960,1,1),(128,1,1)>>> ()
```

Clearly, this is a `fill` function, and the grid size of `40960` is very large. With this information, we can easily pinpoint that the lines `y = from_buffer(x.data_ptr(), x.numel() * 1024 * 1024); y.fill_(1);` forcibly expand the length of `x` by a million times and then fill it entirely with 1s, thereby triggering the `illegal memory access` exception.

On some GPUs, this line might cause `invalid argument` error instead of `illegal memory access`, because the grid size exceeds the maximum limit. In such cases, the CUDA core dump feature cannot be triggered, and you need to turn down the expansion factor `1024 * 1024` a little bit to avoid exceeding the grid size limit.

# Limitations and Considerations

1. In theory, CUDA core dump should be able to capture various exceptions caused by a specific thread on the GPU. However, in practice, on certain GPU and driver versions, exceptions like `operation not supported on global/shared address space` may fail to trigger a CUDA core dump. Fortunately, `illegal memory access` can generally trigger CUDA core dumps reliably, which satisfies most debugging needs.
2. For hardware-related errors, such as `Invalid access of peer GPU memory over nvlink or a hardware error`, these are not caused by a specific thread and cannot be attributed to a particular GPU thread. As a result, CUDA core dumps will not be triggered for such issues.
3. Errors caused by improper use of the driver API are considered [non-sticky errors](https://forums.developer.nvidia.com/t/difference-in-error-handling-between-driver-api-and-runtime-api/336389) and are unrelated to the GPU itself. These errors are reported at the driver API level and do not trigger CUDA core dumps. A common example is an out-of-memory error during `cudaMalloc`, which will not result in a CUDA core dump.
4. For distributed programs involving multi-GPU communication, memory mapping is often used to map the memory of other GPUs to the current GPU. If the program on another GPU exits, the mapped memory becomes invalid, and accessing it will trigger an `illegal memory access`. However, this does not fall under the typical `illegal memory access` issues. Such problems are common during the shutdown process of distributed programs. If GPUs are communicating during shutdown, the order of shutdown may cause some GPUs to report `illegal memory access`. When using CUDA core dump for such programs, it is important to distinguish these false positives.
5. Enabling CUDA core dump does have some performance impact on CUDA kernels (since it needs to check for errors and attribute them when GPU threads exit). Therefore, it is not advisable to enable CUDA core dump in production environments. It is recommended to enable CUDA core dump only after errors like `illegal memory access` can be reliably reproduced for debugging purposes.

# Conclusion

This blogpost analyzed the principles and use cases of CUDA core dump. This debugging method is effective for issues like improper kernel launches and kernel exceptions within CUDA graphs, making it a powerful tool for debugging `illegal memory access` issues and beyond.

As an example, we recently use this technique to debug a complex `illegal memory access` issue in vLLM, see [this PR](https://github.com/vllm-project/vllm/pull/22593) for more details. Basically, we add a [triton kernel](https://github.com/vllm-project/vllm/pull/22375) for MRope, but that kernel has an implicit assumption that `head_size==rotary_dim` (i.e. it's a full Rope). When `head_size!=rotary_dim` (i.e. it's a partial Rope), the kernel will trigger an `illegal memory access`, which is the case for the new [GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V) model. Without CUDA core dump, the error is reported as `Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:453 'an illegal memory access was encountered'`, which is very misleading. With CUDA core dump, we can easily pinpoint the error to the MRope kernel, and then fix it.

The vLLM project aims to provide easy, fast, and cheap LLM serving for everyone, and easy debugging is also an important aspect. We will continue to share more debugging tips and techniques in the future, to build a strong LLM inference ecosystem together. To share your story or usage with vLLM, please submit a PR at [the blogpost repository](https://github.com/vllm-project/vllm-project.github.io).

# Acknowledgement

We would like to thank Ze Long, Vikram Sharma Mailthody, Jeremy Iverson, and Sandarbh Jain from NVIDIA for their helpful discussions.
