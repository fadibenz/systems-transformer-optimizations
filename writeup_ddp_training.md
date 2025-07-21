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

| Backend      | Processes | Size   | Avg Time (ms) | Effective Bandwidth (GB/s) |
|--------------|-----------|--------|---------------|----------------------------|
| gloo (CPU)   | 2         | 1MB    | 1.0172        | 1.92                       |
| gloo         | 2         | 10MB   | 6.3603        | 3.07                       |
| gloo         | 2         | 100MB  | 69.6140       | 2.81                       |
| gloo         | 2         | 1024MB | 657.1070      | 3.04                       |
| gloo         | 4         | 1MB    | 2.8166        | 0.69                       |
| gloo         | 4         | 10MB   | 13.3875       | 1.46                       |
| gloo         | 4         | 100MB  | 139.9087      | 1.40                       |
| gloo         | 4         | 1024MB | 1377.8717     | 1.45                       |
| nccl   (GPU) | 2         | 10MB   | 2.6964        | 7.24                       |
| nccl         | 2         | 100MB  | 26.1726       | 7.46                       |
| nccl         | 2         | 1024MB | 268.5016      | 7.45                       |


To visualize the benchmarking data, I created two charts: one for Average Time (ms) and one for Effective Bandwidth (GB/s), 
both grouped by Backend and Processes, with Size as the x-axis.

###### **Average Time (ms) Chart:**

![](writeup_assets/average_time_ddp.png)

###### **Effective Bandwidth (GB/s) Chart:**

![](writeup_assets/bandwidtw_ddp.png)


>Notes:
> + Average time scales linearly both when we increase the data size and the number of processes.
> + For example, we can see that when we go from two processes to four on the CPU, the total time doubles.
> + Same for the data size,within the same number of processes (factor of 10)
> + I don't have GPU resources to check if this trend persists, but I assume it does.
> + The effective bandwidth stays consistent for the same number of processes as we increase the data size (except for 1MB), this means we're at the peak throughput.
> + For a different number of processes, the effective bandwidth decreases as we increase the number (at least for the CPU)
> + For the same settings, GPU is consistently faster than CPU, highlighting NCCL superior performance 

Caveats and engineering details:
+ I couldn't use TCP to initialize processes on my machine (Windows), I used a local file instead for local testing, I switched to TCP for actual benchmarking in Kaggle.
+ `dist.all_gather_object` causes deadlocks, I still don't know why, but I used `dist.gather` instead (I wasted a lot of time trying to make it to work)
