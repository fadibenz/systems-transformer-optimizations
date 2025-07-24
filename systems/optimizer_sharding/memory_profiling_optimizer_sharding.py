import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import sys

from systems.benchmarking_profiling.utils import get_args
from systems.ddp_training.ddp_overlap_bucketed import DDPBucketed
from systems.ddp_training.utils import setup, forward_backward_step, prepare_local_data
from systems.optimizer_sharding.optimizer_state_sharding import OptimizerStateSharding
from systems.utils import load_config, set_seed_everything
from transformer_implementation.data import get_batch
from transformer_implementation.model import BasicsTransformerLM

MB = 1024 ** 2

def profile_memory(rank: int,
                                 world_size: int,
                                 config: dict,
                                 args,
                                 x: torch.Tensor,
                                 y: torch.Tensor,
                                 use_sharding: bool,
                                 bucket_size: int = 1):
    setup(rank, world_size, "nccl")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)  # Start with clean stats

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
    peak_memory_model_init = torch.cuda.max_memory_allocated(device) / MB

    if use_sharding:
        optimizer = OptimizerStateSharding(
            model.parameters(),
            optimizer_cls=torch.optim.AdamW,
            lr=0.1,
            weight_decay=0.1,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)

    peak_memory_optimizer_init = torch.cuda.max_memory_allocated(device) / MB

    torch.cuda.reset_peak_memory_stats(device)
    forward_backward_step(model, data, targets, optimizer)
    model.finish_gradient_synchronization()
    peak_memory_before_step = torch.cuda.max_memory_allocated(device) / MB
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.step()
    peak_memory_during_step = torch.cuda.max_memory_allocated(device) / MB

    metrics = torch.tensor([
        peak_memory_model_init,
        peak_memory_optimizer_init,
        peak_memory_before_step,
        peak_memory_during_step
    ], dtype=torch.float32, device=device)

    if rank == 0:
        gathered_metrics = [torch.zeros_like(metrics) for _ in range(world_size)]
    else:
        gathered_metrics = None

    dist.gather(metrics, gathered_metrics, dst=0)

    if rank == 0:
        all_metrics = torch.stack(gathered_metrics)
        avg_metrics = all_metrics.mean(dim=0).tolist()

        avg_model_init, avg_optimizer_init, avg_before_step, avg_during_step = avg_metrics

        optimizer_type = "Sharded Optimizer" if use_sharding else "Standard DDP (No Sharding)"
        print(f"\n=== Memory Profile: {optimizer_type} ===")
        print(f"Peak memory after model initialization: {avg_model_init:.2f} MB")
        print(f"Peak memory after optimizer initialization: {avg_optimizer_init:.2f} MB")
        print(f"Peak memory added by fwd/bwd pass (grads & activations): {avg_before_step:.2f} MB")
        print(f"Peak memory added by optimizer step: {avg_during_step:.2f} MB")
        total_peak = avg_optimizer_init + avg_before_step
        print(f"Total peak memory before optimizer step: {total_peak:.2f} MB")

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

    print(f"\nStarted memory profiling with {world_size} GPUs")
    print(f"Model: {args.model}, Batch size: {args.batch_size}, Context length: {args.context_length}")
    print(60 * "-")

    # --- Run WITHOUT sharding ---
    mp.spawn(
        fn=profile_memory,
        args=(world_size, config, args, x, y, False),
        nprocs=world_size,
        join=True
    )

    # --- Run WITH sharding ---
    mp.spawn(
        fn=profile_memory,
        args=(world_size, config, args, x, y, True),
        nprocs=world_size,
        join=True
    )

    print(60 * "-")
    print("Finished memory profiling.\n")