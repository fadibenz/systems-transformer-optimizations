import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from systems.ddp_training.utils import setup
from systems.utils import load_config, set_seed_everything
import sys
import timeit
from transformer_implementation.data import get_batch
from transformer_implementation.model import BasicsTransformerLM
from transformer_implementation.optimizer import AdamW
import argparse

def train_model(rank:int,
                world_size:int,
                config: dict,
                model_name: str,
                x: torch.Tensor,
                y: torch.Tensor,
                warmup_iterations,
                num_iterations):

    setup(rank, world_size, "nccl")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    batch_size = x.size(0)
    local_batch_size = int(batch_size / world_size)
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size

    data = x[start_index:end_index].pin_memory().to(device=device, non_blocking=True)
    targets = y[start_index: end_index].pin_memory().to(device=device, non_blocking=True)

    model_config = config[model_name]

    model = BasicsTransformerLM(
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],

        context_length=config["context_length"],
        vocab_size=config["vocab_size"],
        rope_theta=config["rope_theta"]
    )
    model.to(device)

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    optimizer = AdamW(model.parameters())

    for _ in range(warmup_iterations):
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        optimizer.step()

    torch.cuda.synchronize()
    end_time_reduce = 0
    start_time_e2e = timeit.default_timer()

    for _ in range(num_iterations):
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss.backward()

        start_time_reduce = timeit.default_timer()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        torch.cuda.synchronize()

        end_time_reduce += timeit.default_timer() - start_time_reduce
        optimizer.step()

    torch.cuda.synchronize()
    end_time_e2e = timeit.default_timer() - start_time_e2e

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
            f"Avg for reduce operation: {avg_time_reduce_s * 1000:.4f} ms ({(avg_time_reduce_s / avg_time_e2e_s) * 100}%)"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument(
        "--model",
        type=str,
        choices=["small", "medium", "large", "xl", "2.7B"],
        help="Model size to benchmark"
    )
    parser.add_argument(
        '--context_length',
        type=int,
        default=512)

    # Profiling
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--warmup_iters', type=int, default=3)

    # Oter
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Must run this code with GPU")
        sys.exit(1)

    set_seed_everything(args.seed)
    world_size = torch.cuda.device_count()

    config = load_config("systems/configs/model_sizing.YAML")

    random_id_tokens = np.random.randint(0, config["vocab_size"], config["batch_size"] * args.context_length)

    x, y = get_batch(random_id_tokens, config["batch_size"], args.context_length, device="cpu")

    print("\n Started benchmarking naive DDP")
    print(60 * "-")
    mp.spawn(
        fn=train_model,
        args=(world_size,
              config,
              args.model,
              x, y,
              args.warmup_iters,
              args.iters),
        nprocs=world_size,
        join=True
    )
    print(60 * "-")
    print("\n Finished benchmarking naive DDP")
