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
+ `dist.all_gather_object` causes deadlocks, I still don't know why, but I used `dist.gather` instead (I wasted a lot of time trying to make it work)


## Data Parallel Training:

This is one of the four methods of parallelism (DP, FSDP/TP, PP, EP), and arguably the simplest one.
It allows for bigger batch sizes, something important for stable transformer training. 

> Premise: Batches of data are split across multiple devices, and each device computes gradients for their own batch. 
> These gradients are then averaged across devices.

### Naive DDP:
For naïvely doing distributed data parallel training, we communicate gradients to average after we finish calculating 
the full backward pass, and we communicate gradients one parameter at a time.

You can find implementation under `systems/ddp_training/naive_ddp`.

To benchmark this solution,
I used the medium model size configuration (biggest I could fit) from `systems/configs/model_sizing.YAML`
with batch size 32 and context length 64, I ran experiments on Kaggle using 2 * T4 GPUS.

Results for benchmarking this implementation were as follows:

```
Started benchmarking naive DDP
------------------------------------------------------------
  -> Avg Time for full training step: 1406.7442 ms | Avg for reduce operation: 498.0934 ms (35.4%)
------------------------------------------------------------
 Finished benchmarking naive DDP
```

We can notice two things a large amount of time is spent in communication, which adds significant overhead;
this overhead would be even larger for bigger models (more gradients to communicate).

There are two solutions to improve this naive approach:
+ Reduce the number of communication calls by flattening gradients from all parameters and doing one reduce operation.
+ Overlap communication of parameters with backward pass computations (all-reduce parameter gradients as soon as they’re ready)

We will implement each optimization strategy at a time and see what it yields.

## DDP with flattened tensors:

This is a straightforward implementation, `systems/ddp_training/minimal_ddp_flat_benchmarking`. 

The results for benchmarking using the same setup as before were as follows:

```
 Started benchmarking DDP with flat tensors
------------------------------------------------------------
  -> Avg Time for full training step: 1365.1573 ms | Avg for reduce operation: 500.2333 ms (36.6%)
------------------------------------------------------------

 Finished benchmarking  DDP with flat tensors
```

These results were surprising as I expected to see improvement.

> However, this might be because our model size is small
> and the overhead from flattening/unflattening and then copying back parameters is greater than  
> the overhead associated with issuing a large number of small all-reduce operations.
> I expect to see a slight edge for the version with flat tensors in large models. 

## DDP with overlap: 

We will communicate gradients as soon as they’re computed by the backward pass,
this way we can overlap computation with communication.

Implementation is in `systems/ddp_training/ddp_overlap_individual_parameters`. 
It contains a wrapper that asynchronously all-reduces individual parameter tensors. 

We use two concepts Backward hooks and Asynchronous communication.

The results for benchmarking using the same setup as before were as follows:

```
 Started benchmarking DDP with overlap and individual parameter communication
------------------------------------------------------------
  -> Avg Time for full training step: 1075.7339 ms | 
------------------------------------------------------------

 Finished benchmarking  DDP with overlap and individual parameter communication
 ```

Great, we can see an improvement; our training step now is 28% faster!


## DDP with bucketed overlap:

To get the best of two worlds, we will implement overlap with bucketed communication.
We will organize our parameters into buckets (reducing the number of total communication calls)
and all-reducing buckets when each of their constituent tensors is ready
(enabling us to overlap communication with computation).

Implementation is in `systems/ddp_training/ddp_overlap_bucketed`.

We will benchmark using the same experimental setup as before varying 
the maximum bucket size (1, 10, 100, 1000 MB).


Results were as follows: 


Engineering details: 
+ This was a very fun class to implement, I used some cool concepts like closures and nonlocal variables.
+ In my current implementation, there's an overhead of allocating/deallocating tensors for flattening and unflattening;
a better option would be to allocate persistent buffers for each bucket.
