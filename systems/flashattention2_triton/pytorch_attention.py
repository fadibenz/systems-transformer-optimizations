import sys
import torch
from systems.utils import set_seed_everything, load_config
# noinspection PyUnresolvedReferences
from transformer_implementation.model import scaled_dot_product_attention
import itertools
import argparse
import timeit
import numpy as np

def benchmark_attention(
        iters: int,
        warmup_iters: int,
        d_model: int,
        seq_len: int,
        batch_size: int,
        use_compiled: bool = False,
        half_precision: bool = False
        ) -> tuple[list[float], list[float], float]:

    if not torch.cuda.is_available():
        print("Must run this code with GPU")
        sys.exit(1)

    dtype = torch.float32 if not half_precision else torch.float16
    attention = scaled_dot_product_attention if not use_compiled else torch.compile(scaled_dot_product_attention)

    Q = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=dtype, requires_grad=True)
    K = torch.randn_like(Q, requires_grad=True)
    V = torch.randn_like(Q, requires_grad=True)

    flipped_mask = torch.triu(torch.ones(seq_len, seq_len, device="cuda"), 1).bool()
    mask = ~flipped_mask


    for _ in range(warmup_iters):
        attention(Q, K, V, mask)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    forward_times: list[float] = []

    for _ in range(iters):
        start_time = timeit.default_timer()
        attention(Q, K, V, mask)
        torch.cuda.synchronize()
        forward_times.append(timeit.default_timer() - start_time)

    peak_memory = torch.cuda.max_memory_allocated() / 1e6

    backward_times: list[float] = []
    for _ in range(iters):
        Q.grad = None
        K.grad = None
        V.grad = None

        output = attention(Q, K, V, mask)
        output = output.sum()

        start_time = timeit.default_timer()
        output.backward()
        torch.cuda.synchronize()

        backward_times.append(timeit.default_timer() - start_time)

    return forward_times, backward_times, peak_memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--use_compiled", action="store_true", help="Run with JIT compiled attention")
    parser.add_argument("--half_precision", action="store_true", help="use half-precision for attention" )

    args = parser.parse_args()
    set_seed_everything(args.seed)

    config = load_config("systems/configs/attention.YAML")

    print("\n--Started Benchmarking--")

    for d_model, seq_len in itertools.product(config["d_model"], config["seq_len"]):
        print(f"\nBenchmarking attention for d_model={d_model}, seq_len={seq_len}")

        print(f"\n Using: "
              f"{'      full precision' if not args.half_precision else 'half precision'}" 
              f"{'      JIT compiled attention' if args.use_compiled else ''}"
              f"        seed: {args.seed} ")

        try:
            forward_times, backward_times, allocated_memory = benchmark_attention(args.iters,
                                                                args.warmup_iters,
                                                                d_model,
                                                                seq_len,
                                                                config["batch_size"],
                                                                args.use_compiled,
                                                                args.half_precision)
            print("\n forward pass stats: "
                  f"     avg_time: {np.mean(forward_times):.3f}"
                  f"     std: {np.std(forward_times):.3f}")

            print(f"\n Allocated memory before backward pass: {allocated_memory:.2f} MB")

            print("\n backward pass stats: "
                  f"     avg_time: {np.mean(backward_times):.3f}"
                  f"     std: {np.std(backward_times):.3f}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at d_model={d_model}, seq_len={seq_len}. Skipping...")
                torch.cuda.empty_cache()
            else:
                raise e