import torch
import torch.nn as nn
import timeit
import numpy as np
import sys
from contextlib import nullcontext
import torch.cuda.nvtx as nvtx
from torch.amp.grad_scaler import GradScaler
from utils import get_args, initialize_model_data


def profile(model:nn.Module,
                        data: torch.Tensor,
                        num_iterations:int,
                        warmup_iterations: int,
                        full_run: bool = False,
                        mixed_precision: bool = False) -> [float, float]:

    """
    A function designed to be used with Nsight systems, that profiles model run
    based on the following params:
        model: The model to benchmark
        data: Data to run benchmarking on.
        num_iterations: number of iterations to run model, allowing for more honest estimate.
        warmup_iteration: number of iterations to warm up the model and allow for compilation
        full_run: whether to run the backward pass or not.
    """

    if not torch.cuda.is_available():
        print("Must run this code with GPU")
        sys.exit(1)

    if full_run:
        scaler = GradScaler() if mixed_precision else None

    context_manager = torch.autocast(device_type="cuda", dtype=torch.float16) if mixed_precision else nullcontext()

    nvtx.range_push("warmup phase")
    for _ in range(warmup_iterations):
        with context_manager:
            output = model(data)
            if full_run:
                loss = output.mean()
        if full_run:
            loss.backward()
    nvtx.range_pop()

    torch.cuda.synchronize()

    time_list: list[float] = []


    nvtx.range_push("benchmark")
    for _ in range(num_iterations):
        nvtx.range_push("iteration")

        start_time = timeit.default_timer()

        nvtx.range_push("forward_pass")
        with context_manager:
            output = model(data)
            if full_run:
                loss = output.mean()
        nvtx.range_pop()

        if full_run:
            if mixed_precision and scaler is not None:
                nvtx.range_push("backward_pass")
                scaler.scale(loss).backward()
                nvtx.range_pop()

            else:
                nvtx.range_push("backward_pass")
                loss.backward()
                nvtx.range_pop()
        torch.cuda.synchronize()
        run_time = timeit.default_timer() - start_time
        time_list.append(run_time)
        nvtx.range_pop()
    nvtx.range_pop()

    mean_time = np.mean(time_list)
    std_time = np.std(time_list)

    return mean_time, std_time

if __name__ == "__main__":
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, random_input_data = initialize_model_data(args, device, True)
    print("\n ----Started profiling----")

    print(f"Running on device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    mean_time, std_time = profile(
        model=model,
        data=random_input_data,
        num_iterations=args.iters,
        warmup_iterations=args.warmup_iters,
        full_run=args.full_run,
        mixed_precision=args.mixed_precision
    )

    print("\n ----Finished profiling----")
    print(f"\n it took {mean_time}s to run {'forward and backward pass' if args.full_run else 'forward pass'} on model of size {args.model} with standard deviation {std_time}s")