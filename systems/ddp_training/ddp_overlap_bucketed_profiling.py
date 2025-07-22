import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

from systems.benchmarking_profiling.utils import get_args
from systems.ddp_training.ddp_overlap_bucketed import DDPBucketed
from systems.ddp_training.utils import setup, forward_backward_step, prepare_local_data
from systems.utils import load_config, set_seed_everything
import sys
from transformer_implementation.data import get_batch
from transformer_implementation.model import BasicsTransformerLM
from transformer_implementation.optimizer import AdamW

def train_model(rank: int,
                world_size: int,
                config: dict,
                args,
                x: torch.Tensor,
                y: torch.Tensor,
                bucket_size: int
                ):
    setup(rank, world_size, "nccl")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # Data
    data, targets = prepare_local_data(x, y, rank, device, world_size)
    # Model
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
    optimizer = AdamW(model.parameters())
    total_steps = args.warmup_iters + args.iters + 1

    def trace_handler(prof):
        if rank == 0:
            prof.export_chrome_trace(f"trace_bucket_{bucket_size}mb_step_{prof.step_num}.json")
            print(f"\n=== Profiler Results - Bucket {bucket_size}MB, Step {prof.step_num} ===")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=args.warmup_iters,
            active=args.iters,
            repeat=1
        ),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(total_steps):
            forward_backward_step(model, data, targets, optimizer)
            model.finish_gradient_synchronization()
            optimizer.step()
            torch.cuda.synchronize()
            prof.step()

    dist.destroy_process_group()


if __name__ == "__main__":

    args = get_args()
    if not torch.cuda.is_available():
        print("Must run this code with GPU")
        sys.exit(1)

    set_seed_everything(args.seed)
    world_size = torch.cuda.device_count()

    config = load_config("systems/configs/model_sizing.YAML")
    bucket_sizes_mb = [1, 10, 100, 1000]

    random_id_tokens = np.random.randint(0, config["vocab_size"], args.batch_size * args.context_length)

    x, y = get_batch(random_id_tokens, args.batch_size, args.context_length, device="cpu")

    print("\n Started profiling Bucketed DDP")
    for bucket_size_mb in bucket_sizes_mb:
        print(60 * "-")
        print(f"Bucket size: {bucket_size_mb} MB")
        mp.spawn(
            fn=train_model,
            args=(world_size,
                  config,
                  args,
                  x, y,
                  bucket_size_mb),
            nprocs=world_size,
            join=True
        )
        print(60 * "-")
    print("\n Finished Started profiling Bucketed DDP")
