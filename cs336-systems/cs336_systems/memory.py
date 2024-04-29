from torch.profiler import profile, record_function
from cs336_basics.model.transformer import Transformer
from cs336_basics.model.layers import RMSNorm
from cs336_basics.model.util import crossEntropyLoss
from cs336_basics.training.optimizer import AdamW
from cs336_systems.config import Systems_Config
from cs336_systems.layers import RMSNormTriton
from cs336_systems.util import get_random_batch
from torch.cuda.amp import autocast, GradScaler
import torch
# import matplotlib
import time
import timeit
from contextlib import nullcontext
import torch
import os

def benchmark_memory(version, device, num_exp: int, forward_only: bool, use_mixed_precision: bool = False, norm_type: str = "rms", compiled=False):

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

    if compiled:
        print("Compiling Model")
        model = torch.compile(model)
        print("Done Compiling")

    optimizer = AdamW(
        params=model.parameters(),
        lr=0.01
    )

    context = torch.cuda.amp.autocast if use_mixed_precision else nullcontext

    extra_info = ""
    if forward_only:
        extra_info += "_forward"
    if use_mixed_precision:
        extra_info += "_mixed"
    extra_info += "_" + version

    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_export_path = os.path.join(current_dir, f'benchmarking/timeline{extra_info}.html')
    pickle_export_path = os.path.join(current_dir, f'benchmarking/memory_snapshot{extra_info}.pickle')

    torch.cuda.memory._record_memory_history(max_entries=1000000)

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=num_exp),
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with context():
            for _ in range(num_exp):
                optimizer.zero_grad()

                x, y = get_random_batch(config, device)

                y_hat = model(x)

                if not forward_only:
                    loss = crossEntropyLoss(y, y_hat).mean()
                    loss.backward()
                    optimizer.step()

                    # if "cuda" in device.type:
                    #     torch.cuda.synchronize()
                prof.step()
        # Save a graphical timeline of memory usage.
    prof.export_memory_timeline(html_export_path, device=device)

    # Save a pickle file to be loaded by PyTorch's online tool.
    torch.cuda.memory._dump_snapshot(pickle_export_path)
    # Stop recording history.
    torch.cuda.memory._record_memory_history(enabled=None)