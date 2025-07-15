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
- Implemented FlashAttention2 forward pass in Triton.
- Implemented PyTorch wrapper for triton kernel.
- Debugged Triton implementation and pushed a stable version including:
  - Added porper broadcasting for elementwise-operations.
  - Added an explicit device and dtype declaration in output tensors.
  - Replaced the explicit `tl.transpose` call with pointer arithmetic to load keys directly transposed.
