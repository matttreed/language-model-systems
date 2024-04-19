from cs336_basics.model.transformer import Transformer
from cs336_systems.configs.config import Systems_Config
import torch
import time

def benchmark_transformer(config: Systems_Config, device, num_warmup: int, num_exp: int):
    def get_batch():
        return torch.randint(0, config.vocab_size, (config.batch_size, config.context_length)).to(device)
    
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
        x = get_batch()
        y = model(x)

        if "cuda" in device.type:
            torch.cuda.synchronize()

    times = []

    for _ in range(num_exp):
        x = get_batch()
        start = time.time()
        y = model(x)

        if "cuda" in device.type:
            torch.cuda.synchronize()

        end = time.time()
        times.append(end - start)

    print("Times:", times)
    print(f"Average time: {sum(times) / len(times)}")
    print(f"Standard deviation: {sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)}")