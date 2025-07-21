import os
import torch.distributed as dist
import socket
from systems.utils import load_config
from itertools import product
import torch
import numpy as np
import torch.nn.functional as F

MB = 1024 * 1024
GB = 1024 * MB

def setup(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = "29500"
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]

def construct_config(file_path):
    config_data = load_config(file_path)
    CONFIGS_TO_RUN = []
    for benchmark in config_data['benchmarks']:
        backend = benchmark['backend']
        device = benchmark['device']
        for world_size, tensor_size_mb in product(benchmark['world_sizes'], benchmark['tensor_sizes_mb']):
            CONFIGS_TO_RUN.append({
                'backend': backend,
                'device': device,
                'world_size': world_size,
                'tensor_size_bytes': tensor_size_mb * MB
            })
    return CONFIGS_TO_RUN


def print_logs(end_time_e2e, end_time_reduce, device, rank, num_iterations, world_size):
    local_time_e2e = torch.tensor([end_time_e2e], dtype=torch.float32, device=device)
    local_time_reduce = torch.tensor([end_time_reduce], dtype=torch.float32, device=device)

    if rank == 0:
        gathered_e2e_times = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
        gathered_reduce_times = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
    else:
        gathered_e2e_times = None
        gathered_reduce_times = None

    dist.gather(local_time_e2e, gathered_e2e_times, dst=0)
    dist.gather(local_time_reduce, gathered_reduce_times, dst=0)

    if rank == 0:
        time_list_e2e = [t.item() for t in gathered_e2e_times]
        avg_time_e2e_s = np.mean(time_list_e2e) / num_iterations

        time_list_reduce = [t.item() for t in gathered_reduce_times]
        avg_time_reduce_s = np.mean(time_list_reduce) / num_iterations

        print(
            f"  -> Avg Time for full training step: {avg_time_e2e_s * 1000:.4f} ms | "
            f"Avg for reduce operation: {avg_time_reduce_s * 1000:.4f} ms ({(avg_time_reduce_s / avg_time_e2e_s) * 100:.1f}%)"
        )

def forward_backward_step(model, data, targets, optimizer):
    optimizer.zero_grad(set_to_none=True)
    logits = model(data)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    loss.backward()


def prepare_local_data(x, y, rank, device, world_size):
    batch_size = x.size(0)
    local_batch_size = int(batch_size / world_size)
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size

    data = x[start_index:end_index].pin_memory().to(device=device, non_blocking=True)
    targets = y[start_index: end_index].pin_memory().to(device=device, non_blocking=True)

    return data, targets