from cs336_systems.config import Systems_Config
from cs336_systems.benchmarking import benchmark_transformer
from cs336_systems.profiling import profile_transformer
from cs336_systems.memory import benchmark_memory
import torch

import argparse

def main():
    parser = argparse.ArgumentParser(description='Train a model with float16 parameters.')
    parser.add_argument('--version', type=str, default=None, help='Version Number of Model')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark Transformer')
    parser.add_argument('--memory', action='store_true', help='Benchmark Memory of Transformer')
    parser.add_argument('--profile', action='store_true', help='Profile Transformer')
    parser.add_argument('--num_warmup', type=int, default=1, help='Num Warmups')
    parser.add_argument('--num_exp', type=int, default=5, help='Num Warmups')
    parser.add_argument('--forward_only', action='store_true', help='Only time forward pass')
    parser.add_argument('--mixed', action='store_true', help='Used Mixed Precision')
    parser.add_argument('--norm_type', type=str, default="rms", help='"rms", "rms_triton", "layer"')
    parser.add_argument('--compiled', action="store_true", help='Compile Transformer Model')
    args = parser.parse_args()

    # config = Systems_Config(version=args.version)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    versions = ["s", "m", "l", "xl", "2.7"]

    if args.version != "sweep":
        versions = [args.version]

    for version in versions:

        if args.profile:
            print("Profiling version: " + version)
            profile_transformer(version, device, args.num_warmup, args.num_exp, args.forward_only, use_mixed_precision=args.mixed)

        if args.benchmark:
            print("Benchmarking version: " + version)
            benchmark_transformer(version, device, args.num_warmup, args.num_exp, args.forward_only, use_mixed_precision=args.mixed, norm_type=args.norm_type, compiled=args.compiled)

        if args.memory:
            print("Benchmarking Memory for version: " + version)
            benchmark_memory(version, device, args.num_exp, args.forward_only, use_mixed_precision=args.mixed, norm_type=args.norm_type, compiled=args.compiled)


if __name__ == "__main__":

    main()