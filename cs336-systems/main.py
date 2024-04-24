from cs336_systems.config import Systems_Config
from cs336_systems.benchmarking import benchmark_transformer
from cs336_systems.profiling import profile_transformer
import torch

import argparse

def main():
    parser = argparse.ArgumentParser(description='Train a model with float16 parameters.')
    parser.add_argument('--version', type=str, default=None, help='Version Number of Model')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark Transformer')
    parser.add_argument('--profile', action='store_true', help='Profile Transformer')
    parser.add_argument('--num_warmup', type=int, default=1, help='Num Warmups')
    parser.add_argument('--num_exp', type=int, default=5, help='Num Warmups')
    parser.add_argument('--forward_only', action='store_true', help='Only time forward pass')
    parser.add_argument('--mixed', action='store_true', help='Used Mixed Precision')
    args = parser.parse_args()

    # config = Systems_Config(version=args.version)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.profile:
        profile_transformer(args.version, device, args.num_warmup, args.num_exp, args.forward_only, use_mixed_precision=args.mixed)

    if args.benchmark:
        benchmark_transformer(args.version, device, args.num_warmup, args.num_exp, args.forward_only, use_mixed_precision=args.mixed)


if __name__ == "__main__":

    main()