from cs336_basics.model.transformer import Transformer
from cs336_systems.config import Systems_Config
from cs336_systems.util import get_random_batch
from torch.cuda.amp import autocast, GradScaler
import torch
import time
import timeit

def benchmark_transformer_mixed(version, device, num_warmup: int, num_exp: int, forward_only: bool):

    config = Systems_Config(version)
    
    model = Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        attn_pdrop=config.attn_pdrop,
        residual_pdrop=config.residual_pdrop,
        context_length=config.context_length
    ).to(device)

    for _ in range(num_warmup):
        x, _ = get_random_batch(config, device)
        with autocast():
            y = model(x)

        if "cuda" in device.type:
            torch.cuda.synchronize()

    times = []

    for _ in range(num_exp):
        x, _ = get_random_batch(config, device)
        start = time.time()
        with autocast():
            y = model(x)

            if not forward_only:
                y.sum().backward()

            if "cuda" in device.type:
                torch.cuda.synchronize()

        end = time.time()
        times.append(end - start)

    print("Times:", times)
    print(f"Average time: {sum(times) / len(times)}")
    print(f"Standard deviation: {sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)}")

def benchmark_transformer(version, device, num_warmup: int, num_exp: int, forward_only: bool):

    config = Systems_Config(version)
    
    model = Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        attn_pdrop=config.attn_pdrop,
        residual_pdrop=config.residual_pdrop,
        context_length=config.context_length
    ).to(device)

    for _ in range(num_warmup):
        x, _ = get_random_batch(config, device)
        y = model(x)

        if "cuda" in device.type:
            torch.cuda.synchronize()

    times = []

    for _ in range(num_exp):
        x, _ = get_random_batch(config, device)
        start = timeit.default_timer()
        y = model(x)

        if not forward_only:
            y.sum().backward()

        if "cuda" in device.type:
            torch.cuda.synchronize()

        end = timeit.default_timer()
        times.append(end - start)

    print("Times:", times)
    print(f"Average time: {sum(times) / len(times)}")
    print(f"Standard deviation: {sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)}")