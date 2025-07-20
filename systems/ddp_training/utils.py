import os
import torch.distributed as dist
import socket
from systems.utils import load_config
from itertools import product

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

