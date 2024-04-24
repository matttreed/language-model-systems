from cs336_basics.model.transformer import Transformer
from cs336_systems.config import Systems_Config
from cs336_systems.util import get_random_batch
from torch.cuda.amp import autocast, GradScaler
import torch
import time
import timeit
from contextlib import nullcontext

def benchmark_transformer(version, device, num_warmup: int, num_exp: int, forward_only: bool, use_mixed_precision: bool = False, use_layer_norm: bool = False):

    config = Systems_Config(version)
    
    model = Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        attn_pdrop=config.attn_pdrop,
        residual_pdrop=config.residual_pdrop,
        context_length=config.context_length,
        use_layer_norm=use_layer_norm
    ).to(device)

    context = torch.cuda.amp.autocast if use_mixed_precision else nullcontext

    for _ in range(num_warmup):
        x, _ = get_random_batch(config, device)
        with context():
            y = model(x)

        if "cuda" in device.type:
            torch.cuda.synchronize()

    forward_times = []
    backward_times = []
    total_times = []

    for _ in range(num_exp):
        x, _ = get_random_batch(config, device)
        start = timeit.default_timer()

        with context():
            y = model(x)


        if "cuda" in device.type:
            torch.cuda.synchronize()
        
        end = timeit.default_timer()
        forward_times.append(end - start)

        if not forward_only:
            start = end
            with context():
                y.sum().backward()

            if "cuda" in device.type:
                torch.cuda.synchronize()

            end = timeit.default_timer()
            backward_times.append(end - start)

            total_times.append(forward_times[-1] + backward_times[-1])
        else:
            total_times.append(forward_times[-1])

    # print("Total Times:", total_times)
    print(f"Average time: {sum(total_times) / len(total_times)}")
    print(f"Standard deviation: {sum((t - sum(total_times) / len(total_times)) ** 2 for t in total_times) / len(total_times)}")

    # print("Forward Pass Times:", forward_times)
    print(f"Average time: {sum(forward_times) / len(forward_times)}")
    print(f"Standard deviation: {sum((t - sum(forward_times) / len(forward_times)) ** 2 for t in forward_times) / len(forward_times)}")

    # print("Backward Pass Times:", backward_times)
    print(f"Average time: {sum(backward_times) / len(backward_times)}")
    print(f"Standard deviation: {sum((t - sum(backward_times) / len(backward_times)) ** 2 for t in backward_times) / len(backward_times)}")