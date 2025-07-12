# 2025-07-10
- Setup of the initial repository and dependencies using `uv`
- Added transformer model benchmarking utility
  - **`benchmark_operation` function**: Measures forward and optional backward pass execution time, including warm-up iterations and averaging for accuracy.
  - **`argparse` integration**: Allows specifying model size, context length, iterations, and whether to run a full (forward + backward) pass.
  - **Dynamic configuration**: Loads model parameters from a YAML configuration based on selected model size.
  - **Reproducibility**: Sets seeds for consistent benchmark results.
- Tested the benchmarking script on CPU using local machine.
- Pushed work to GitHub to start benchmarking using Kaggle.

#