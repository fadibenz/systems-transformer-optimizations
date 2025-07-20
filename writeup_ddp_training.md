# Distributed Data Parallel Training

## Benchmarking Distributed Applications:

To better understand the overhead of communication in distributed training.
I started by benchmarking the all-reduce operation, which is arguably the most used communication operation.

I followed best practices for benchmarking distributed applications including:
1. All benchmarks are run on the same machine for controlled comparisons.
2. I perform several warm-up steps, especially important for NCCL communications.
3. I use `torch.cuda.synchronize()` to wait for CUDA operations to complete when benchmarking on GPUs.
4. I aggregate measurements across ranks to improve estimates.

I started by first debugging locally with Gloo on CPU and then benchmarked on Kaggle with NCCL on GPU.

I used the following settings:
+ Backend + device type: Gloo + CPU, NCCL + GPU.  
+ Data sizes: float32 data tensors ranging over 1MB, 10MB, 100MB, 1GB.  
+ Number of processes: 2, 4, or 6 processes.

On GPU, I only had access to two T4 GPUs on Kaggle.

Results were as follows: 



>Notes:
> 
> 

Caveats and engineering details:
+ I couldn't use TCP to initialize on my machine (Windows), I used a local file instead.
+ 