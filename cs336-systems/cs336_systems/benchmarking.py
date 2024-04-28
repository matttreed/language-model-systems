from cs336_basics.model.transformer import Transformer
from cs336_basics.model.layers import RMSNorm
from cs336_basics.model.util import crossEntropyLoss
from cs336_basics.training.optimizer import AdamW
from cs336_systems.config import Systems_Config
from cs336_systems.layers import RMSNormTriton
from cs336_systems.util import get_random_batch
from torch.cuda.amp import autocast, GradScaler
import torch
import time
import timeit
from contextlib import nullcontext

def benchmark_transformer(version, device, num_warmup: int, num_exp: int, forward_only: bool, use_mixed_precision: bool = False, norm_type: str = "rms", compiled=False):

    config = Systems_Config(version)

    norm_function = None

    if norm_type == "rms":
        norm_function = RMSNorm
    elif norm_type == "layer":
        norm_function = torch.nn.LayerNorm
    elif norm_type == "rms_triton":
        norm_function = RMSNormTriton
    
    model = Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        attn_pdrop=config.attn_pdrop,
        residual_pdrop=config.residual_pdrop,
        context_length=config.context_length,
        norm_function=norm_function
    ).to(device)

    if compiled:
        print("Compiling Model")
        model = torch.compile(model)
        print("Done Compiling")

    optimizer = AdamW(
        params=model.parameters(),
        lr=0.01
    )

    context = torch.cuda.amp.autocast if use_mixed_precision else nullcontext

    times = []

    with context():

        for i in range(num_exp + num_warmup):

            optimizer.zero_grad()

            x, y = get_random_batch(config, device)
            start = timeit.default_timer()

            y_hat = model(x)

            if not forward_only:
                loss = crossEntropyLoss(y, y_hat).mean()
                loss.backward()
                optimizer.step()

                if "cuda" in device.type:
                    torch.cuda.synchronize()

            end = timeit.default_timer()

            if i >= num_warmup:
                times.append(end-start)


    print(f"Average time: {sum(times) / len(times)}")
    print(f"Standard deviation: {sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)}")
    print("TIMES: ", times)
