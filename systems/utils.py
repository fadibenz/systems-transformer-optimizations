import yaml
import random
import numpy as np
import torch

def load_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed_everything(seed: int = 2025):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

git commit -m "feat: Add transformer model benchmarking utility

Implements a new utility to benchmark transformer model performance. Key features include:

- **`benchmark_operation` function**: Measures forward and optional backward pass execution time, including warm-up iterations and averaging for accuracy.
- **`argparse` integration**: Allows specifying model size, context length, iterations, and whether to run a full (forward + backward) pass.
- **Dynamic configuration**: Loads model parameters from a YAML configuration based on selected model size.
- **Reproducibility**: Sets seeds for consistent benchmark results.

This utility is essential for evaluating performance across different transformer model configurations."
