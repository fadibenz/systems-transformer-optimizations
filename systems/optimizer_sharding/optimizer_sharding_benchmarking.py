import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

from systems.benchmarking_profiling.utils import get_args
from systems.ddp_training.ddp_overlap_bucketed import DDPBucketed
from systems.ddp_training.utils import setup, forward_backward_step, prepare_local_data
from systems.optimizer_sharding.optimizer_state_sharding import OptimizerStateSharding
from systems.utils import load_config, set_seed_everything
import sys
import timeit
from transformer_implementation.data import get_batch
from transformer_implementation.model import BasicsTransformerLM

MB = 1024 ** 2

def train_model(rank: int,
                world_size: int,
                config: dict,
                args,
                x: torch.Tensor,
                y: torch.Tensor,
                bucket_size: int = 1
                ):
    setup(rank, world_size, "nccl")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    torch.cuda.reset_peak_memory_stats(device)

    data, targets = prepare_local_data(x, y, rank, device, world_size)

    model_config = config[args.model]

    model = BasicsTransformerLM(
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        context_length=args.context_length,
        vocab_size=config["vocab_size"],
        rope_theta=config["rope_theta"]
    )

    model.to(device)
    model = DDPBucketed(model, bucket_size)

    peak_memory_initialization = torch.cuda.max_memory_allocated(device) / MB  # Convert to MB

    # Optimizer
    optimizer = OptimizerStateSharding(
        model.parameters(),
        optimizer_cls=torch.optim.AdamW,
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    peak_memory_with_optimizer = torch.cuda.max_memory_allocated(device) / MB

    for _ in range(args.warmup_iters):
        forward_backward_step(model, data, targets, optimizer)
        model.finish_gradient_synchronization()
        optimizer.step()

    # Reset memory stats before timing iterations
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    start_time_e2e = timeit.default_timer()

    for i in range(args.iters):
        forward_backward_step(model, data, targets, optimizer)
        model.finish_gradient_synchronization()
        optimizer.step()

    torch.cuda.synchronize()
    end_time_e2e = timeit.default_timer() - start_time_e2e

    # Measure peak memory before and after optimizer step
    torch.cuda.reset_peak_memory_stats(device)

    # Forward and backward pass
    forward_backward_step(model, data, targets, optimizer)
    model.finish_gradient_synchronization()

    # Memory before optimizer step (after forward/backward)
    peak_memory_before_optimizer = torch.cuda.max_memory_allocated(device) / MB

    # Reset peak memory stats to measure only optimizer step impact
    torch.cuda.reset_peak_memory_stats(device)

    # Optimizer step
    optimizer.step()

    # Memory after optimizer step
    peak_memory_after_optimizer = torch.cuda.max_memory_allocated(device) / MB

    # Gather timing results
    local_time_e2e = torch.tensor([end_time_e2e], dtype=torch.float32, device=device)
    local_memory_init = torch.tensor([peak_memory_initialization], dtype=torch.float32, device=device)
    local_memory_optimizer_init = torch.tensor([peak_memory_with_optimizer], dtype=torch.float32, device=device)
    local_memory_before_opt = torch.tensor([peak_memory_before_optimizer], dtype=torch.float32, device=device)
    local_memory_after_opt = torch.tensor([peak_memory_after_optimizer], dtype=torch.float32, device=device)

    if rank == 0:
        gathered_e2e_times = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
        gathered_memory_init = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
        gathered_memory_optimizer_init = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
        gathered_memory_before_opt = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
        gathered_memory_after_opt = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
    else:
        gathered_e2e_times = None
        gathered_memory_init = None
        gathered_memory_optimizer_init = None
        gathered_memory_before_opt = None
        gathered_memory_after_opt = None

    # Gather all metrics
    dist.gather(local_time_e2e, gathered_e2e_times, dst=0)
    dist.gather(local_memory_init, gathered_memory_init, dst=0)
    dist.gather(local_memory_optimizer_init, gathered_memory_optimizer_init, dst=0)
    dist.gather(local_memory_before_opt, gathered_memory_before_opt, dst=0)
    dist.gather(local_memory_after_opt, gathered_memory_after_opt, dst=0)

    if rank == 0:
        time_list_e2e = [t.item() for t in gathered_e2e_times]
        avg_time_e2e_s = np.mean(time_list_e2e) / args.iters

        memory_init_list = [m.item() for m in gathered_memory_init]
        memory_optimizer_init_list = [m.item() for m in gathered_memory_optimizer_init]
        memory_before_opt_list = [m.item() for m in gathered_memory_before_opt]
        memory_after_opt_list = [m.item() for m in gathered_memory_after_opt]

        memory_optimizer_overhead = np.mean(memory_after_opt_list) - np.mean(memory_before_opt_list)

        print(f"\n=== Benchmarking Results ===")
        print(f"Average time per training step: {avg_time_e2e_s * 1000:.4f} ms")
        print(f"\n=== Memory Usage (MB) ===")
        print(f"Peak memory after model initialization: {np.mean(memory_init_list):.2f} MB")
        print(f"Peak memory after optimizer initialization: {np.mean(memory_optimizer_init_list):.2f} MB")
        print(f"Peak memory BEFORE optimizer step: {np.mean(memory_before_opt_list):.2f} MB")
        print(f"Peak memory AFTER optimizer step: {np.mean(memory_after_opt_list):.2f} MB")
        print(f"Memory overhead from optimizer step: {memory_optimizer_overhead:.2f} MB")
        print(f"\n=== Per-GPU Breakdown ===")
        for i in range(world_size):
            opt_overhead = memory_after_opt_list[i] - memory_before_opt_list[i]
            print(f"GPU {i}: Before optimizer={memory_before_opt_list[i]:.2f}MB, "
                  f"After optimizer={memory_after_opt_list[i]:.2f}MB, "
                  f"Overhead={opt_overhead:.2f}MB")

    dist.destroy_process_group()


if __name__ == "__main__":
    args = get_args()
    if not torch.cuda.is_available():
        print("Must run this code with GPU")
        sys.exit(1)

    set_seed_everything(args.seed)
    world_size = torch.cuda.device_count()

    config = load_config("systems/configs/model_sizing.YAML")

    random_id_tokens = np.random.randint(0, config["vocab_size"], args.batch_size * args.context_length)
    x, y = get_batch(random_id_tokens, args.batch_size, args.context_length, device="cpu")

    print(f"\nStarted benchmarking bucketed DDP with {world_size} GPUs")
    print(f"Model: {args.model}, Batch size: {args.batch_size}, Context length: {args.context_length}")
    print(60 * "-")

    mp.spawn(
        fn=train_model,
        args=(world_size, config, args, x, y, 1),
        nprocs=world_size,
        join=True
    )

    print(60 * "-")
    print("Finished benchmarking bucketed DDP\n")