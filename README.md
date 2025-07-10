# Systems (Inspired from CS336 assignment 2)

## Setup

This directory is organized as follows:

- [`./transformer_implementation`](./transformer_implementation): directory containing a module
  `transformer_implementation` and its associated `pyproject.toml`. This module contains my 
   **unoptimized** implementation of a LlaMA-style language model from scratch.
- [`./systems`](./systems): The module where I will implement my optimized Transformer language model. 
  it contains the following submodules: 
  -  [`1_benchmarking_profiling`]()
  - [`2_flashattention2_triton`]()
  - [`3_ddp_training`]()
  - [`4_optimizer_sharding`]()

Visually, it should look something like:

``` sh
├── transformer_implementation 
│├── __init__.py
│ └── ... other files that implement a transformer from scratch ...
├── systems   
│ ├── __init__.py
│ └── 1_benchmarking_profiling
│ └── 2_flashattention2_triton
│ └── 3_ddp_training
│ └── 4_optimizer_sharding
├── README.md
├── pyproject.toml
└── ... other files or folders ...
```
I use `uv` to manage dependencies.

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Testing

All tests are adapted from CS336 assignment 2 public repository; they provide a rigorous way 
to check implementations against standard ones.