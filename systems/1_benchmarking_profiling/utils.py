import argparse
from argparse import Namespace

import torch
from systems.utils import load_config, set_seed_everything
from transformer_implementation.model import BasicsTransformerLM
import torch.nn as nn
from annotated_scaled_dot_product_attention import annotated_scaled_dot_product_attention

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
    parser.add_argument('--mixed_precision', action='store_true', help="Run with mixed-precision")

    # Oter
    parser.add_argument('--seed', type=int, default=2025)

    return parser.parse_args()

def initialize_model_data(args: Namespace,
                          device: str,
                          profile: bool = False) ->[nn.Module, torch.Tensor]:
    set_seed_everything(args.seed)

    config = load_config("systems/configs/model_sizing.YAML")

    vocab_size = config["vocab_size"]
    batch_size = config["batch_size"]

    model_config = config[args.model]

    if profile:
        transformer_implementation.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    model = BasicsTransformerLM(
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        context_length=args.context_length,
        vocab_size=vocab_size,
        rope_theta=config["rope_theta"]
    )

    model.to(device)

    random_input_data = torch.randint(0, vocab_size,
                                      (batch_size, args.context_length))

    if torch.cuda.is_available():
        random_input_data = random_input_data.pin_memory().to(device, non_blocking=True)
    else:
        random_input_data = random_input_data.to(device)

    return model, random_input_data

