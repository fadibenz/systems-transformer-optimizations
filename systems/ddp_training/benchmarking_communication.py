import os
import timeit

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from systems.ddp_training.utils import  setup, construct_config

MB = 1024 * 1024
GB = 1024 * MB


def distributed_benchmark(rank: int,
                          world_size: int,
                          backend: str,
                          device_type: str,
                          tensor_size_bytes: int,
                          num_iterations: int = 20,
                          warmup_iterations: int = 5):
    setup(rank, world_size, backend)

    if device_type == "gpu":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    num_elements = tensor_size_bytes // 4
    data = torch.randint(0, 100, (num_elements,), dtype=torch.float32).to(device)

    for _ in range(warmup_iterations):
        dist.all_reduce(data, async_op=False)

    if device_type == "gpu":
        torch.cuda.synchronize()
    dist.barrier()

    start_time = timeit.default_timer()
    for _ in range(num_iterations):
        dist.all_reduce(data, async_op=False)

    if device_type == "gpu":
        torch.cuda.synchronize()

    end_time = timeit.default_timer() - start_time

    local_time_tensor = torch.tensor([end_time], dtype=torch.float32, device=device)

    if rank == 0:
        gathered_times = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
    else:
        gathered_times = None

    dist.gather(tensor=local_time_tensor, gather_list=gathered_times, dst=0)

    if rank == 0:
        time_list = [t.item() for t in gathered_times]
        avg_total_time = np.mean(time_list)
        avg_time_s = avg_total_time / num_iterations

        bus_bandwidth_gbps = (tensor_size_bytes * 2) / avg_time_s / GB

        print(
            f"  -> Avg Time: {avg_time_s * 1000:.4f} ms | "
            f"Effective Bandwidth: {bus_bandwidth_gbps:.2f} GB/s"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    CONFIGS_TO_RUN = construct_config("systems/configs/distirbuted_benchmarking.YAML")

    num_gpus = torch.cuda.device_count()
    num_cores = os.cpu_count()

    print("\nStarted benchmarking")

    for config in CONFIGS_TO_RUN:
        world_size = config['world_size']
        if config['backend'] == 'nccl':
            if not torch.cuda.is_available():
                print("Must run nccl backend option in GPU environment")
                continue

            if num_gpus < world_size:
                print(
                    f"Skipping config ({world_size} procs): "
                    f"Requires {world_size} GPUs, but only {num_gpus} are available."
                )
                continue

        if config['backend'] == 'gloo' and num_cores < world_size:
            print(
                f"Skipping config ({config['backend']}, {world_size} procs): "
                f"Requires {world_size} CPU cores, but only {num_cores} are available."
            )
            continue

        print(f"Config: Backend={config['backend']}, "
              f"Processes={world_size}, Size={config['tensor_size_bytes'] / MB:.0f}MB")

        mp.spawn(
            fn=distributed_benchmark,
            args=(
                world_size,
                config['backend'],
                config['device'],
                config['tensor_size_bytes'],
            ),
            nprocs=world_size,
            join=True
        )
        print("-" * 60)

    print("\nFinished benchmarking")