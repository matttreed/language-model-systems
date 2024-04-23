from cs336_basics.model.transformer import Transformer
from cs336_basics.training.optimizer import AdamW
from cs336_basics.model.util import crossEntropyLoss
from cs336_systems.config import Systems_Config
from torch.profiler import profile, record_function, ProfilerActivity
from cs336_systems.util import get_random_batch
import torch
import time
import os

def profile_transformer(version, device, num_warmup: int, num_exp: int, forward_only: bool):

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

    optimizer = AdamW(
        model.parameters(), 
        betas=config.betas,
        weight_decay=config.weight_decay,
        eps=config.eps,
        lr=config.lr)

    for _ in range(num_warmup):
        x, y = get_random_batch(config, device)
        y_hat = model(x)
        loss = crossEntropyLoss(y, y_hat).mean()

        loss.backward()

        if "cuda" in device.type:
            torch.cuda.synchronize()

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ], experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        for _ in range(num_exp):
            optimizer.zero_grad()

            x, y = get_random_batch(config, device)
            with record_function('forward_pass'):
                y_hat = model(x)

            if not forward_only:

                with record_function('backward_pass'):
                    loss = crossEntropyLoss(y, y_hat).mean()
                    loss.backward()

                with record_function('optimizer'):
                    optimizer.step()      

            if "cuda" in device.type:
                torch.cuda.synchronize()
            
            prof.step()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    export_path = os.path.join(current_dir, f'lm_profiler_stacks_{version}.txt')

    prof.export_stacks(export_path, "self_cuda_time_total")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
