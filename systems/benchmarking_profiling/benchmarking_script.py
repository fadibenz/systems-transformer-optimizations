import torch
import torch.nn as nn
import timeit
import numpy as np
import sys
from contextlib import nullcontext

from torch import no_grad

from utils import get_args, initialize_model_data

from torch.amp.grad_scaler import GradScaler

def benchmark_operation(model:nn.Module,
                        data: torch.Tensor,
                        num_iterations:int,
                        warmup_iterations: int,
                        full_run: bool = False,
                        mixed_precision: bool = False) -> [float, float]:

    """
    A function that benchmarks model run based on following params:
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
    no_grad = torch.no_grad() if not full_run else nullcontext

    for _ in range(warmup_iterations):
        with context_manager:
            output = model(data)
            if full_run:
                loss = output.mean()
        if full_run:
            loss.backward()

    torch.cuda.synchronize()

    time_list: list[float] = []


    for _ in range(num_iterations):

        start_time = timeit.default_timer()
        with context_manager:
            with no_grad:
                output = model(data)
                if full_run:
                    loss = output.mean()

        if full_run:
            if mixed_precision and scaler is not None:
                scaler.scale(loss).backward()
                scaler.update()
            else:
                loss.backward()

        torch.cuda.synchronize()

        run_time = timeit.default_timer() - start_time
        time_list.append(run_time)

    mean_time = np.mean(time_list)
    std_time = np.std(time_list)

    return mean_time, std_time


if __name__ == "__main__":
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, random_input_data = initialize_model_data(args, device)

    print("\n ----Started benchmarking----")

    print(f"Running on device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    mean_time, std_time = benchmark_operation(
        model=model,
        data=random_input_data,
        num_iterations=args.iters,
        warmup_iterations=args.warmup_iters,
        full_run=args.full_run,
        mixed_precision=args.mixed_precision
    )

    print("\n ----Finished benchmarking----")
    print(f"\n it took {mean_time}s to run {'forward and backward pass' if args.full_run else 'forward pass'} on model of size {args.model} with standard deviation {std_time}s")