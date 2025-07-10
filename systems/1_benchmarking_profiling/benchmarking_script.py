import argparse

import torch
import torch.nn as nn
from transformer_implementation.model import BasicsTransformerLM
from systems.utils import load_config, set_seed_everything
import timeit
import numpy as np


def benchmark_operation(model:nn.Module,
                        data: torch.Tensor,
                        num_iterations:int,
                        warmup_iterations: int,
                        full_run: bool) -> [float, float]:

    """
    A function that benchmarks model run based on following params:
        model: The model to benchmark
        data: Data to run benchmarking on.
        num_iterations: number of iterations to run model, allowing for more honest estimate.
        warmup_iteration: number of iterations to warm up the model and allow for compilation
        full_run: whether to run the backward pass or not.
    """

    for _ in range(warmup_iterations):
        model(data)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    time_list: list[float] = []

    for _ in range(num_iterations):
        start_time = timeit.default_timer()

        output = model(data)

        if full_run:
            loss = output.mean()
            loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        run_time = timeit.default_timer() - start_time
        time_list.append(run_time)
    mean_time = np.mean(time_list)
    std_time = np.std(time_list)

    return mean_time, std_time

def get_args():
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
    parser.add_argument('--full_run', action='store_true', help="Run both forward and backward pass")

    # Oter
    parser.add_argument('--seed', type=int, default=2025)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    set_seed_everything(args.seed)

    config = load_config("systems/configs/model_sizing.YAML")

    vocab_size = config["vocab_size"]
    batch_size = config["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = config[args.model]

    model = BasicsTransformerLM(
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff= model_config["d_ff"],
        context_length=args.context_length,
        vocab_size=vocab_size,
        rope_theta= config["rope_theta"]
    )
    model.to(device)

    random_input_data = torch.randint(0, vocab_size,
                                      (batch_size, args.context_length))

    if torch.cuda.is_available():
        random_input_data = random_input_data.pin_memory().to(device, non_blocking=True)
    else:
        random_input_data = random_input_data.to(device)

    print("\n ----Started benchmarking----")

    print(f"Running on device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    mean_time, std_time = benchmark_operation(
        model=model,
        data=random_input_data,
        num_iterations=args.iters,
        warmup_iterations=args.warmup_iters,
        full_run=args.full_run
    )

    print("\n ----Finished benchmarking----")
    print(f"\n it took {mean_time}s to run {'forward and backward pass' if args.full_run else 'forward pass'} on model of size {args.model} with standard deviation {std_time}s")
