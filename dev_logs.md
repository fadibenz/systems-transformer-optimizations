# 2025-07-10 (6h)
- Setup of the initial repository and dependencies using `uv`
- Added transformer model benchmarking utility
  - **`benchmark_operation` function**: Measures forward and optional backward pass execution time, including warm-up iterations and averaging for accuracy.
  - **`argparse` integration**: Allows specifying model size, context length, iterations, and whether to run a full (forward + backward) pass.
  - **Dynamic configuration**: Loads model parameters from a YAML configuration based on selected model size.
  - **Reproducibility**: Sets seeds for consistent benchmark results.
- Tested the benchmarking script on CPU using local machine.
- Pushed work to GitHub to start benchmarking using Kaggle.

# 2025-07-13 (8h)
- Finished benchmarking all runs, results and analysis in `writeup.md`.
- Added profiling with Nsight systems by adding different nvtx ranges.
- Refactored code and seperated benchmarking / profiling scripts.
- Analyzed the forward pass and went into the different detail, results in `writeup.md`
- Added annotated scaled-dot product for a deeper look at the different attention operations.
- Analyzed the backward pass and went into the attention profile in detail, results in `writeup.md`.
- Added a script for profiling memory.
- Run memory profiling on different context lengths, inference vs. training and FP vs. MP.

# 2025-07-14 (8h)
- Used `memoryviz` to visualize and understand memory allocation.
- Finished writeup section on memory profiling and added Figma diagrams.
- Wrote code to benchmark pytorch attention and memory allocation.
- Added options for lower precision and for compiled version.
- Added writeup section and analysis of raw vs. compiled implementation of attention.
- Read and understood FlashAttention 1 and 2, wrote summary in `writeup.md`.
- Implemented FlashAttention in pytorch using the torch.autograd.Function signature

# 2025-07-15
- Debugged the pure pytorch version of attention and pushed a stable version.
- Learned triton from official docs, I will add a section of my understanding in the weekend.
- Implemented FlashAttention2 forward pass in Triton.
- Implemented PyTorch wrapper for triton kernel.
- Debugged Triton implementation and pushed a stable version including:
  - Added porper broadcasting for elementwise-operations.
  - Added an explicit device and dtype declaration in output tensors.
  - Replaced the explicit `tl.transpose` call with pointer arithmetic to load keys directly transposed.
- Added causal masking to Triton Implementation
- Froward pass passes all tests. 

# 2025-07-16
- Implemented Backward pass in triton
- Added causal masking in the backward pass.
- Implemented script to benchmark FlashAttention2 against pure PyTorch
- Started reading about ways to make my implementation even faster. 
  - I will tune the tile sizes using `triton.autotune`
  - I will replace the backward pass with two passes one for dQ and another for dK and dV to avoid atomics or synchronization between blocks.
  - I will try to Stop program instances early when doing causal masking, skipping all tiles that are always all zero
  - I will separate the non-masked tiles from the tile diagonals, computing the first without ever comparing indices, and the second with a single comparison

# 2025-07-17
+ Implemented all the previously discussed optimization tricks:
  + Replaced the backward pass with two passes, one for dQ and another for dK and dV to avoid atomics or synchronization between blocks.
  + Implemented smart causal masking:
    + replace the backward pass with two passes, one for dQ and another for dK and dV to avoid atomics or synchronization between blocks.
      - Stop program instances early when doing causal masking, skipping all tiles that are always all zero
      - Separated the non-masked tiles from the tile diagonals, computing the first without ever comparing indices, and the second with a single comparison
+ Finished forward and backward sections in the writeup. 
+ Benchmarked extensively each new optimization trick I added and compared all of them in the writeup.

# 2025-07-20
+ Started learning about distributed training (DP, FSDP, TP, PP, EP), I will focus on DP.
+ Implemented a benchmarking script to investigate the overhead of distributed communication operations. 
+ Finished writeup on different benchmarking settings with tables and plots.

# 2025-07-21
+ Implemented the naive version of DDP and the benchmarking script.
+ Implemented DDP with flattened gradients communication and benchmarking scripts.
+ Implemented DDP with overlapped parameters communication.
+ Implemented DDP with overlapped and bucketed communication.
+ All implementations pass the tests.

# 2025-07-22
+ Added benchmarking for overlapped parameters communication.
+ Added benchmarking and profiling for overlapped and bucketed communication. 
+ Added writeup section on all the previous techniques with tables, plots, and comparisons.
+ Read the Ultra-scaling book sections on DP, and ZeRO (1, 2, 3).
+ Added writeup section on communication accounting.
