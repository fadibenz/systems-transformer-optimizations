import torch
import torch.nn as nn
import sys
from contextlib import nullcontext
from utils import get_args, initialize_model_data
from transformer_implementation.optimizer import AdamW
from torch.amp.grad_scaler import GradScaler

def profile_memory(model:nn.Module,
                        data: torch.Tensor,
                        num_iterations:int,
                        warmup_iterations: int,
                        full_run: bool = False,
                        mixed_precision: bool = False):

    """
    A function that profiles memory used in  model run
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
        optimizer = AdamW(model.parameters())
        scaler = GradScaler() if mixed_precision else None

    context_manager = torch.autocast(device_type="cuda", dtype=torch.float16) if mixed_precision else nullcontext()

    for _ in range(warmup_iterations):
        with context_manager:
            model(data)


    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    for _ in range(num_iterations):
        with context_manager:
            output = model(data)
            if full_run:
                loss = output.mean()

        if full_run:
            if mixed_precision and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        torch.cuda.synchronize()

    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, random_input_data = initialize_model_data(args, device)

    print("\n ----Started memory profiling----")

    print(f"Running on device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    profile_memory(
        model=model,
        data=random_input_data,
        num_iterations=args.iters,
        warmup_iterations=args.warmup_iters,
        full_run=args.full_run,
        mixed_precision=args.mixed_precision
    )

    print("\n ----Finished memory profiling----")
