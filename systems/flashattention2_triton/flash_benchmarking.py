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

    dtype = torch.float16 if half_precision else torch.float32
    attention = FlashAttention2Triton.apply if use_flash else scaled_dot_product_attention

    Q = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=dtype, requires_grad=True)
    K = torch.randn_like(Q, requires_grad=True)
    V = torch.randn_like(Q, requires_grad=True)

    if not use_flash:
        flipped_mask = torch.triu(torch.ones(seq_len, seq_len, device="cuda"), 1).bool()
        mask = ~flipped_mask
    else:
        mask = True

    # Forward pass
    results_forward_pass = triton.testing.do_bench(lambda: attention(Q, K, V, mask), rep=5000, warmup=500, quantiles=[0.2, 0.5, 0.8])

    # Backward pass
    O = attention(Q, K, V, mask)
    grad_O = torch.randn_like(O)

    results_backward_pass = triton.testing.do_bench(lambda : O.backward(grad_O, retain_graph=True), rep=5000, warmup=500, quantiles=[0.2, 0.5, 0.8])

    # End-To-End pass
    grad_O_e2e = torch.randn_like(O)
    def forward_backward_e2e():
       if Q.grad is not None:
           Q.grad = None
           K.grad = None
           V.grad = None
       output = attention(Q, K, V, mask)
       output.backward(grad_O_e2e, retain_graph= True)


    results_e2e = triton.testing.do_bench(forward_backward_e2e, rep=5000, warmup=500, quantiles=[0.2, 0.5, 0.8])

    return results_forward_pass, results_backward_pass, results_e2e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
              f"\n{'      full precision' if not args.half_precision else 'half precision'}" 
              f"\n{'      FlashAttention2' if args.use_flash else 'PyTorch Attention'}"
              f"\n        seed: {args.seed} ")

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