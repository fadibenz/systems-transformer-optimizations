import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from systems.benchmarking_profiling.utils import get_args
from systems.ddp_training.utils import setup, forward_backward_step, print_logs, prepare_local_data
from systems.utils import load_config, set_seed_everything
import sys
import timeit
from transformer_implementation.data import get_batch
from transformer_implementation.model import BasicsTransformerLM
from transformer_implementation.optimizer import AdamW

def train_model(rank:int,
                world_size:int,
                config: dict,
                args,
                x: torch.Tensor,
                y: torch.Tensor,
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

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    optimizer = AdamW(model.parameters())

    for _ in range(args.warmup_iters):
        forward_backward_step(model, data, targets, optimizer)
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        optimizer.step()

    torch.cuda.synchronize()
    end_time_reduce = 0
    start_time_e2e = timeit.default_timer()

    for _ in range(args.iters):
        forward_backward_step(model, data, targets, optimizer)

        start_time_reduce = timeit.default_timer()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        torch.cuda.synchronize()

        end_time_reduce += timeit.default_timer() - start_time_reduce
        optimizer.step()

    torch.cuda.synchronize()
    end_time_e2e = timeit.default_timer() - start_time_e2e

    print_logs(end_time_e2e, end_time_reduce, device, rank, args.iters, world_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    args = get_args()

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
              args,
              x, y),
        nprocs=world_size,
        join=True
    )
    print(60 * "-")
    print("\n Finished benchmarking naive DDP")
