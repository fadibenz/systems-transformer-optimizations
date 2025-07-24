# Optimizer State Sharding:

To reduce redundancy in data-parallel training, we can partition he (1) optimizer state, (2) gradients, 
and (3) parameters across ranks, communicating them between workers as necessary.

I implemented a simplified version of optimizer state sharding 
Rather than keeping the optimizer states for all parameters, each rank’s 
optimizer instance will only handle a subset of the parameters (approximately 1 / world_size). 
When each rank’s optimizer takes an optimizer step, 
it’ll only update the subset of model parameters in its shard. 
Then, each rank will broadcast its updated parameters to the other ranks to ensure that the model parameters 
remain synchronized after each optimizer step.

## Implementation details:

The OptimizerStateSharding class wraps an arbitrary input PyTorch optim.Optimizer
and takes care of updating parameters after each optimizer step. You can find implementation inside `systems/optimizer_sharding/optimizer_state_sharding.py`

There are three main parts: 
+ 
+ When calling the wrapper we pass `params, optimizer_cls: Type[Optimizer], **kwargs: Any`, this
allows for proper initialization. Inside `__init__` we keep a reference to a local optimizer for this rank and 
call the superclass constructor.
+ The superclass constructor calls the `add_param_group` with each param_group we specified in the `params` argument,
we override this method with an implementation that takes care of sharding by dividing each parameter group to a subset of 
approximately `1 / world_size`, we calculate the subset using the rank of the process and add each subset to the process's
local optimizer.
+ The `step` method takes care of the communication part, we start by calling each rank's `local_optimizer.step()`
to update all subsets, we proceed by `all-reduce` of the loss that might be returned by the closure.
we proceed by communicating each subset of the parameters. 



### Optimizer step:

##### naive version:
This is the part that contains most of the code; I started by implementing it using a naive solution (see git history):
+ I iterate over each param group and extract the params data.
+ I iterate after all the ranks, extract the local shard, flatten it, and then do a `broadcast` from the current rank 
in the loop to all other ranks. 
+ After the broadcast, I unflatten the tensor and copy back to the original parameters.

##### Optimized version
This is a very inefficient communication pattern specially in large models 
with many nodes; a better pattern would be to use the `all-gather` collective.

In my final implementation, I followed this pattern:
+ I iterate over each param group and extract the params data.
+ Extract each local shard and flatten it, calculate the length of each flattened 
tensor and `all-gather` sizes.
+ Use the max size to pad all local shards.
+ Communicate all the padded local shards using `all-gather`.
+ Finish by iterating over the gathered list, extract the original size, unflatten the tensor and 
copy back the data.

## Benchmarking:
#### Training Speed Comparison:
I benchmarked the time taken per iteration with and without state sharding using
the standard configuration (one node, 2 * T4 GPUs, medium model size).
I also included the time for the naive and optimized versions. 

| Optimizer version                       | Avg Time (ms) | 
|-----------------------------------------|---------------|
| DDP Without Optimizer Sharding          | 966.32        |
| DDP With Optimizer Sharding (Naive)     | 1111.37       |
| DDP With Optimizer Sharding (Optimized) | 1203.45       |

---

##### Notes:
+ 
+ Using optimizer sharding leads to slower training speeds because
we add the overhead of communication.
+ The optimized version didn't lead to a faster training step, probably we can see
the difference if we use a larger number of GPUs and a larger model.

#### Memory Usage Comparison: Standard DDP vs. Sharded Optimizer:

I also profiled memory usage with and without optimizer sharding:

| **Metric**                                         | **Standard DDP (No Sharding)** | **Sharded Optimizer** |
|----------------------------------------------------|--------------------------------|-----------------------|
| Peak after model initialization                    | 3230.54 MB                     | 3230.54 MB            |
| Peak after optimizer initialization                | 3230.54 MB                     | 3230.54 MB            |
| Memory added by fwd/bwd pass (grads & activations) | 4873.88 MB                     | 4873.88 MB            |
| Memory added by optimizer step                     | 8097.26 MB                     | 6479.94 MB            |

---

##### Notes:

- **Sharded Optimizer** reduced memory usage during the optimizer step by **~20%**.
- The memory used by the states of the optimizer is around **~4GB**, by sharding the optimizer
we cut this memory footprint by half. 
- As expected, the memory usage of the grads and activations is the same, to shard these also we would want 
to implement ZeRO-2 and ZeRO-3.
