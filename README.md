# Systems-Level Transformer Optimization

This repository documents a systematic approach to analyzing, profiling, and optimizing a LlaMA-style Transformer model. The project spans from low-level kernel implementation to high-level distributed training strategies, aiming to identify and resolve performance bottlenecks inherent in large-scale language models.

---

## Setup

This project uses `uv` for dependency management. To install the required packages, simply run any `uv run` command, which will automatically create a virtual environment and install the dependencies specified in `pyproject.toml`.

The repository is organized as follows:

-   `./transformer_implementation`: Contains the baseline, unoptimized implementation of a LlaMA-style language model, serving as a reference for correctness and performance comparison.
-   `./systems`: Contains the core optimization work, structured into distinct modules:
    -   `benchmarking_profiling`: Scripts and utilities for performance and memory profiling.
    -   `flashattention2_triton`: A from-scratch implementation of FlashAttention2 using Triton.
    -   `ddp_training`: Implementations of various Distributed Data Parallel (DDP) strategies.
    -   `optimizer_sharding`: A from-scratch implementation of optimizer state sharding (ZeRO Stage 1).

```sh
.
├── transformer_implementation
│   ├── __init__.py
│   └── ... (unoptimized transformer implementation)
├── systems
│   ├── __init__.py
│   ├── benchmarking_profiling
│   ├── flashattention2_triton
│   ├── ddp_training
│   └── optimizer_sharding
├── README.md
├── pyproject.toml
└── ...
```

---

## Testing

All custom implementations, including FlashAttention2 and distributed training modules, 
are rigorously validated against a comprehensive test suite adapted from the CS336 course at Stanford. 
This ensures correctness and parity with standard, well-established implementations.

---

## Technical Analysis and Optimizations

The optimization process was a systematic, 
multi-layered investigation designed to diagnose performance issues from first principles 
and apply targeted solutions across the entire software and hardware stack.

### Stage 1: Systematic Profiling and Bottleneck Identification

A rigorous profiling methodology was established to move beyond high-level timings and gather actionable, low-level performance data.

-   **Methodology**:
    -   **End-to-End Benchmarking:** Established baseline performance and validated scaling properties.
    -   **NVIDIA Nsight Systems Profiling:** Utilized `nvtx` ranges to conduct in-depth analysis of the CPU-GPU interaction, kernel launch overhead, and hardware utilization.
    -   **Memory Profiling:** Employed line-by-line memory profilers to identify and quantify memory consumption hotspots.

-   **Key Findings**:
    -   **CPU Dispatch Bottleneck:** Nsight analysis revealed that the GPU was frequently idle, waiting for the CPU. Micro-operations were heavily bound by CPU dispatch latency; a single attention head's execution wall-clock time was **~50% CPU dispatch** and **~50% actual GPU compute**.
    -   **Kernel Proliferation:** A single forward pass triggered over **2,500 distinct kernel launches**, primarily from unfused element-wise operations. This created a significant bottleneck at the CUDA API level, as each launch carries non-trivial overhead.
    -   **Memory Scaling ($O(L^2)$):** Memory profiling confirmed that the theoretical $O(L^2)$ complexity of the attention mechanism was the primary inhibitor to training on long contexts. The explicit materialization of the attention score matrix was identified as the main memory hotspot.

### Stage 2: Low-Level Optimization with Custom Kernels

The profiling results clearly indicated that the standard attention implementation was fundamentally inefficient. 
To address this, a from-scratch implementation of **FlashAttention2** was developed using **Triton**. 
The goal was to fuse the entire attention computation into a single, efficient kernel.

-   **Architectural Principles Applied**:
    -   **Tiled Computation & I/O Awareness:** The kernel avoids materializing the large attention matrix by loading blocks of Q, K, and V into fast GPU SRAM. This maximizes the compute-to-memory-access ratio, directly addressing the memory bandwidth bottleneck identified in profiling.
    -   **Memory Coalescing:** The Triton code was carefully written to ensure that threads within a warp access contiguous blocks of memory, a critical technique for maximizing DRAM bandwidth utilization.
    -   **Numerical Stability via Online Softmax:** The numerically stable "online softmax" algorithm was implemented to maintain precision without requiring the full attention matrix.

-   **Performance Impact**:
    -   **Forward Pass Speedup: ~2x.** By implementing intelligent causal masking that skips fully masked tiles, the custom Triton kernel achieved a ~2x speedup over a naive PyTorch implementation.
    -   **Backward Pass Speedup: ~1.5x.** The backward pass was also optimized by splitting it into two separate kernels (one for dQ, one for dK/dV) to avoid atomic operations on global memory, resulting in a ~1.5x speedup.

### Stage 3: Scaling Out with Distributed Systems Techniques

With the on-chip performance optimized, the focus shifted to scaling training across multiple GPUs.

-   **Optimizing Distributed Data Parallelism (DDP)**:
    An analysis of standard DDP showed that the `all_reduce` operation for gradient synchronization was the new bottleneck, consuming **over 35% of the total step time**. A series of progressively more sophisticated solutions were implemented:
    1.  **Baseline (Naive DDP):** Established the initial performance ceiling.
    2.  **Latency Hiding (Overlapped DDP):** Implemented backward hooks to fire asynchronous `all_reduce` calls as soon as parameter gradients became available. This strategy of overlapping communication with computation successfully hid the communication latency. **Result: 1.3x overall speedup.**
    3.  **Bandwidth Management (Bucketed DDP):** Explored bucketed communication to balance the number of `all_reduce` calls with the potential for overlap, confirming the system was in a bandwidth-bound regime.

-   **Tackling Memory Redundancy with Optimizer State Sharding (ZeRO-1)**:
    Even with optimized DDP, model size is limited by the fact that each GPU stores a full copy of the optimizer states. To overcome this, a **ZeRO Stage 1-style sharding mechanism was implemented from scratch**.
    -   **Mechanism:** The optimizer states were partitioned across all devices, with each rank managing only its assigned shard. During the optimizer step, an `all_gather` operation synchronizes the updated parameters.
    -   **Result:** This reduced the optimizer memory footprint by **~20% on 2 GPUs**, validating the strategy. It also illuminated the fundamental **memory-vs-communication trade-off**, as the memory savings came at the cost of increased communication overhead from the `all_gather` collective.


## Writeup:

I Included detailed technical writeups that document and provide context for all the experiments and their results.

- **Profiling and Benchmarking writeup:** I provide tables, plots, Nsight Systems screenshots that document each part of 
the benchmarking methodology with analysis and notes, you can find it under: `writeup_profiling_benchmarking.md`
- **FlashAttention2 writeup:** Includes my understanding of the algorithm with implementation details and benchmarking of baseline implementation and different 
optimizations, you can find it under `writeup_flash_attention.md`.
- **DDP training writeup:** I detail the implementation of DDP in different stages, benchmarking and analysing each step, I finish by communication accounting 
using standard formulas `writeup_ddp_training.md``.
- **Optimizer Sharding writeup:** Includes implementation details of the OptimizerStateSharding class and the different design choices, with benchmarking of training speed 
and memory profiling, `writeup_optimizer_sharding.md`.

---

## Development Log

<details>
<summary>Click to expand</summary>

### 2025-07-10
- Set up the initial repository, dependencies, and `uv` environment.
- Developed a flexible model benchmarking utility with `argparse` and dynamic YAML configuration.
- Established a reproducible benchmarking process by controlling random seeds.

### 2025-07-13
- Completed end-to-end benchmarking runs for multiple model sizes.
- Integrated `nvtx` ranges into the code for deep profiling with NVIDIA Nsight Systems.
- Analyzed Nsight traces for both forward and backward passes, identifying kernel launch overhead as a key bottleneck.
- Developed a memory profiling script to analyze usage under different context lengths and precisions.

### 2025-07-14
- Visualized memory allocation patterns using `memoryviz` to understand memory flow during training.
- Benchmarked raw PyTorch attention vs. its `torch.compile()` version, confirming that context length is the dominant cost factor.
- Researched the FlashAttention v1 and v2 papers and implemented the algorithm in pure PyTorch using `torch.autograd.Function`.

### 2025-07-15
- Began learning Triton from the official documentation.
- Implemented the FlashAttention2 forward pass as a custom Triton kernel.
- Debugged the Triton implementation, ensuring correct pointer arithmetic, broadcasting, and causal masking logic. The forward pass now passes all numerical correctness tests.

### 2025-07-16
- Implemented the FlashAttention2 backward pass in Triton, including support for causal masking.
- Began researching advanced Triton optimization techniques, such as autotuning, avoiding atomic operations, and intelligent masking.

### 2025-07-17
- Implemented and benchmarked several advanced optimizations for the FlashAttention2 kernel:
    - Split the backward pass into two kernels (for dQ, and for dK/dV) to eliminate the need for atomic additions.
    - Implemented "smart masking" to entirely skip computation for fully masked blocks and use more efficient logic for diagonal blocks.
- Extensively benchmarked each optimization to quantify its impact on performance.

### 2025-07-20
- Began studying distributed training paradigms (DP, TP, PP, FSDP).
- Implemented a benchmarking script to isolate and measure the overhead of the `all_reduce` communication collective.

### 2025-07-21
- Implemented and tested three variations of Distributed Data Parallelism (DDP):
    1.  Naive DDP (synchronize after backward pass).
    2.  DDP with flattened gradient communication.
    3.  DDP with computation-communication overlap on a per-parameter basis.
    4.  DDP with bucketed communication to balance overlap and communication efficiency.

### 2025-07-22
- Completed benchmarking and profiling of all DDP implementations.
- Analyzed results, confirming that overlapping communication is the most effective strategy.
- Researched the ZeRO paper and began implementing ZeRO Stage 1 (Optimizer State Sharding).

### 2025-07-23
- Completed and tested the from-scratch implementation of optimizer state sharding.
- Benchmarked its performance, measuring the trade-off between memory savings and increased communication overhead.

### 2025-07-24
- Finalized analysis and documentation for all implemented systems optimizations.

</details>
