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
                use_sharding: bool = False,
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

    if use_sharding:
        optimizer = OptimizerStateSharding(
            model.parameters(),
            optimizer_cls=torch.optim.AdamW,
            lr=0.1,
            weight_decay=0.1,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)


    for _ in range(args.warmup_iters):
        forward_backward_step(model, data, targets, optimizer)
        model.finish_gradient_synchronization()
        optimizer.step()

    torch.cuda.synchronize()
    start_time_e2e = timeit.default_timer()

    for i in range(args.iters):
        forward_backward_step(model, data, targets, optimizer)
        model.finish_gradient_synchronization()
        optimizer.step()

    torch.cuda.synchronize()
    end_time_e2e = timeit.default_timer() - start_time_e2e


    local_time_e2e = torch.tensor([end_time_e2e], dtype=torch.float32, device=device)

    if rank == 0:
        gathered_e2e_times = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]

    else:
        gathered_e2e_times = None

    dist.gather(local_time_e2e, gathered_e2e_times, dst=0)

    if rank == 0:
        time_list_e2e = [t.item() for t in gathered_e2e_times]
        avg_time_e2e_s = np.mean(time_list_e2e) / args.iters
        optimizer_type = "Sharded Optimizer" if use_sharding else "Standard DDP (No Sharding)"
        print(f"\n=== Benchmarking Results {optimizer_type} ===")
        print(f"Average time per training step: {avg_time_e2e_s * 1000:.4f} ms")

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

    print(f"\nStarted benchmarking optimizer sharding")
    print(f"Model: {args.model}, Batch size: {args.batch_size}, Context length: {args.context_length}")
    print(60 * "-")

    # --- Run WITHOUT sharding ---
    mp.spawn(
        fn=train_model,
        args=(world_size, config, args, x, y, False),
        nprocs=world_size,
        join=True
    )

    # --- Run WITH sharding ---
    mp.spawn(
        fn=train_model,
        args=(world_size, config, args, x, y, True),
        nprocs=world_size,
        join=True
    )

    print(60 * "-")
    print("Finished benchmarking optimizer sharding\n")