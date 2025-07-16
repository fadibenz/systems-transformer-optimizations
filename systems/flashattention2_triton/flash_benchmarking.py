import sys
import torch
from systems.utils import set_seed_everything, load_config
# noinspection PyUnresolvedReferences
from transformer_implementation.model import scaled_dot_product_attention
from systems.flashattention2_triton.flashAttention2_triton_wrapper import FlashAttention2Triton
import itertools
import argparse
import triton


def benchmark_attention(
        d_model: int,
        seq_len: int,
        batch_size: int,
        use_flash: bool = False,
        half_precision: bool = False
        ):

    if not torch.cuda.is_available():
        print("Must run this code with GPU")
        sys.exit(1)

    dtype = torch.float32 if not half_precision else torch.float16
    attention = scaled_dot_product_attention if not use_flash else FlashAttention2Triton.apply

    Q, K, V = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=dtype, requires_grad=True)

    if not use_flash:
        flipped_mask = torch.triu(torch.ones(seq_len, seq_len, device="cuda"), 1).bool()
        mask = ~flipped_mask
    else:
        mask = True

    # Forward pass
    results_forward = triton.testing.do_bench(attention(Q, K, V, mask), rep=10000, warmup=500)

    # Backward pass
    O = attention(Q, K, V, mask)

    def backward(O):
        loss = O.sum()
        loss.backward()

    results_backward = triton.testing.do_bench(backward(O), rep=10000, warmup=500)

    # End-To-End pass
    def forward_backward():
        O = attention(Q, K, V, mask)
        loss = O.sum()
        loss.backward()

    results_end_to_end = triton.testing.do_bench(forward_backward, rep=10000, warmup=500)

    return results_forward, results_backward, results_end_to_end

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--use_flash", action="store_true", help="Run with JIT compiled attention")
    parser.add_argument("--half_precision", action="store_true", help="use half-precision for attention" )

    args = parser.parse_args()
    set_seed_everything(args.seed)

    config = load_config("systems/configs/attention.YAML")

    print("\n--Started Benchmarking--")

    for d_model, seq_len in itertools.product(config["d_model"], config["seq_len"]):
        print(f"\nBenchmarking attention for d_model={d_model}, seq_len={seq_len}")

        print(f"\n Using: "
              f"{'      full precision' if not args.half_precision else 'half precision'}" 
              f"{'      FlashAttention2' if args.use_flash else 'PyTorch Attention'}"
              f"        seed: {args.seed} ")

        try:
            results_forward, results_backward, results_end_to_end =  benchmark_attention(d_model,
                                                                                         seq_len,
                                                                                         config["batch_size"],
                                                                                         args.use_flash,
                                                                                         args.half_precision)
            print("\n Forward pass Results: "
                  f"\n {results_forward}")
            print("\n Backward pass Results: "
                  f"\n {results_backward}")
            print("\n End-To-End Results: "
                  f"\n {results_end_to_end}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at d_model={d_model}, seq_len={seq_len}. Skipping...")
                torch.cuda.empty_cache()
            else:
                raise e