import os
import timeit

import torch
import torch.multiprocessing as mp
import torch.distributed as  dist
import numpy as np


def setup(rank, world_size, backend):

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    init_method = "tcp://127.0.0.1:29500"

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )

def distributed_benchmark(rank, world_size, backend, tensor_size):
    setup(rank, world_size, backend)

    cuda = backend == "nccl"

    device = f"cuda:{rank}" if cuda else "cpu"
    data = torch.randint(0, 100, (tensor_size, ), dtype=torch.float32).to(device)

    for _ in range(5):
        dist.all_reduce(data, async_op=False)

    if cuda:
        torch.cuda.synchronize()

    start_time = timeit.default_timer()
    dist.all_reduce(data, async_op=False)

    if cuda:
        torch.cuda.synchronize()

    end_time = timeit.default_timer() - start_time
    time_list = [None] * world_size
    dist.all_gather_object(time_list, end_time)

    if rank == 0:
        print(f"              Avg time, all-reduce:{np.mean(time_list)}")

    dist.destroy_process_group()

if __name__ == "__main__":
    backend_list = ["gloo", "nccl"]
    # will enforce Gloo + CPU and NCCL + GPU
    data_size_list = [262144, 2621440, 26214400, 268435456]
    number_processes = [2, 4, 6]
    print("\nStarted benchmarking")
    for backend in backend_list:
        if backend == "nccl" and not torch.cuda.is_available():
            print("Must have GPU to run nccl backend option")
            continue
        print(f"  Backend + device type: {backend} + {"GPU" if backend=="nccl" else "CPU"}")

        for data_size in data_size_list:
            print(f"      all-reduce data size: ~{data_size * 4 / 1048576:.1f} MB")

            for world_size in number_processes:
                try:
                    print(f"          Number of processes: {world_size}")
                    mp.spawn(fn=distributed_benchmark, args=(world_size, backend, data_size), nprocs=world_size, join=True)
                except (RuntimeError, ValueError) as e:
                    if "nprocs" in str(e).lower() or "processes" in str(e).lower():
                        print(f"Not enough processes for world_size={world_size}, Skipping...")
                        print(f"Error: {e}")
                    else:
                        raise
    print("\nFinished benchmarking")