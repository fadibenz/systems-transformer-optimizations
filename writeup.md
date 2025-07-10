## Profiling and Benchmarking:

To make sure that what I optimize accounts for a good chunk of resources (time and memory),

I will start by implementing three performance evaluation paths:

1. A simple, end-to-end benchmarking using the Python standard library to time our forward and backward passes.
2. Profile compute with the NVIDIA Nsight Systems tool to understand how that time is distributed across operations on both the CPU and GPU:
   - Since I don't have GPU on my local machine, I will use a hybrid approach: Collect performance data in Kaggle using **CLI tools** and then analyze with GUI locally.
3. Profile memory usage.

We will use a vocab size of 10,000 and batch size of 4 in all experiments.

The model sizes are as follows:

| Size   | d_model | d_ff  | num_layers | num_heads |
|--------|---------|-------|------------|-----------|
| small  | 768     | 3072  | 12         | 12        |
| medium | 1024    | 4096  | 24         | 16        |
| large  | 1280    | 5120  | 36         | 20        |
| xl     | 1600    | 6400  | 48         | 25        |
| 2.7B   | 2560    | 10240 | 32         | 32        |

**Table 1: Specifications of different model sizes**


### End-to-End Benchmarking:

We benchmarked the forward and backward passes for the transformer models described in the table below. 


Each run used **5 warmup iterations** followed by **10 timed iterations**,
and we report both the **mean** and **standard deviation** of the runtimes.
I used a context length of 128, and I could fit up to XL model;
the 2.7B was too big for my Tesla P100 (Kaggle free GPU).

#### Forward Pass Timings

| Model Size | Avg Time (s) | Std Dev (s) |
|------------|--------------|-------------|
| Small      | 0.035        | 0.005       |
| Medium     | 0.070        | 0.010       |
| Large      | 0.103        | 0.010       |
| XL         | 0.178        | 0.001       |
| 2.7B       | OOM          | OOM         |


#### Backward Pass Timings

| Model Size | Avg Time (s) | Std Dev (s) |
|------------|--------------|-------------|
| Small      | 0.095        | 0.068       |
| Medium     | 0.190        | 0.056       |
| Large      | 0.360        | 0.052       |
| XL         | OOM          | OOM         |
| 2.7B       | OOM          | OOM         |


> **Notes:**
> + we can see an almost linear scaling with model size for the forward pass,
> it increases with around 0.035 s between model sizes.  
> + The full pass (froward + backward) exhibits similar linear scaling (increases with 0.1 s with model size)
> + The backward pass takes around two times the forward pass (Aligns with scaling heuristics)
> + Runtime variability was minimal, with standard deviations remaining low,
> suggesting stable system performance once the model reaches a steady state (after warmup).

---

#### Effect of Omitting Warmup Steps

We repeated the benchmark **without war mup steps**, 
and observed significantly higher runtimes and greater variance.
For example, with no warmup iterations, 
model size medium goes from 0.19 s for the full run with warm up to 0.33 s without it,
and we notice very high variance of around 0.4 s!

This is mainly due to the fact that the first few iterations incur one-time costs such as CUDA kernel compilation,
memory allocations, and cache setup.

Even when using only **1 or 2 warmup steps**, the timings were still inconsistent. 
Reliable benchmarking requires enough warmup iterations 
to let the system reach a deterministic and cache-optimized execution state.


#### Mixed-precision training:
In mixed-precision training, we might use either FP32 or BF16 (Nobody seriously uses FP16).

The use of BF16 leads to less memory use (obvious) and to faster training (less obvious) 
because it leads to higher memory throughput, better cache utilization and register efficiency (among other more technical reasons).

My heuristical understanding on where to use FP32/BF16 is the following:

+ Weights should always be stored in FP32.  
  - In mixed‑precision training (e.g., NVIDIA AMP), a master copy of the weights lives in FP32.  
  - FP16/BF16 “shadows” are used in matmuls, but updates are applied to the FP32 master to avoid drift.

+ Mixed‑precision is used in both forward and backward passes, but noisy gradients don’t accumulate unchecked.  
  - Forward *and* backward matrix‑multiply kernels run in FP16/BF16.  
  - Gradients are immediately up‑cast (and loss‐scaled if using FP16) to FP32 before the weight update.

+ Noisy activations (from low precision) act like a tiny regularizer, but the main goal is speed and memory savings.  
  - Quantization noise is structured by mantissa rounding, not pure Gaussian.  
  - We benefit from it only because matmul errors average out over many terms.

+ We only tolerate noise where it can be averaged out by gradient sums.  
  - Independent errors in multi‑term reductions (e.g. dot‑products) cancel out.  
  - Single‑term or few‑term ops (norms, losses) can’t average out error and must stay in FP32.

+ In simpler terms:  
  + We can use BF16 in matrix multiplications because each output is a sum over many products, rounding errors average out.  
  + But for normalization or loss calculations, a single rounding error would bias the entire result, use FP32.

+ Systematic gradient corruption (single‑point failure) vs. random gradient noise (independent ops)  
  - This difference is what determines precision requirements for training stability.

### Nsight Systems Profiler:
