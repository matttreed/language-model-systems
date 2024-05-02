from cs336_basics.configs.config import Config
from cs336_basics.model.transformer import Transformer
from cs336_basics.training.optimizer import AdamW
from cs336_basics.model.util import crossEntropyLoss, load_model, save_model, get_batch, log_validation_loss, log, build_model
from cs336_systems.config import Systems_Config
from cs336_systems.ddp import DDP_Bucketed, DDP_Individual_Parameters, DDP_Naive
from torch.profiler import profile, record_function, ProfilerActivity
import torch
import numpy as np
import os
from datetime import timedelta
import torch.distributed as dist
import timeit
import argparse

SEED = 32
BACKEND = "nccl"
DEVICE = "cuda"
NUM_EXP = 1

def setup():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])

    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]

    timeout = timedelta(seconds=60)
    dist.init_process_group(BACKEND, rank=rank, world_size=world_size, timeout=timeout)
    return rank, world_size, local_rank, local_world_size


def train_model(version: str, type: str, world_size):

    config = Systems_Config(version)
    config.batch_size = 12
    
    model = Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        attn_pdrop=config.attn_pdrop,
        residual_pdrop=config.residual_pdrop,
        context_length=config.context_length,
    ).to(DEVICE)

    model.train()

    optimizer = AdamW(
        params=model.parameters(),
        lr=0.01
    )

    if type == "naive":
        model = DDP_Naive(model)
    elif type == "indiv":
        model = DDP_Individual_Parameters(model)
    elif type == "bucket":
        model = DDP_Bucketed(model, bucket_size_mb=12)

    MINI_BATCH_SIZE = config.batch_size // world_size

    assert config.batch_size % world_size == 0, "batch_size must divide world_size"

    forward_time = []
    all_reduce_time = []
    optimizer_step_time = []

    for i in range(5):
        optimizer.zero_grad()
        x = torch.randint(0, config.vocab_size - 1, (MINI_BATCH_SIZE, config.context_length), device=DEVICE) # NOTE THIS IS NOT ACCURATE SINCE NOT SHARING DATA
        y = torch.randint(0, config.vocab_size - 1, (MINI_BATCH_SIZE, config.context_length), device=DEVICE)

        y_hat = model(x)
        loss = crossEntropyLoss(y, y_hat).mean()
        loss.backward()

        model.finish_gradient_synchronization()
        optimizer.step()

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ], experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as prof:

        for i in range(NUM_EXP):
            optimizer.zero_grad()
            x = torch.randint(0, config.vocab_size - 1, (MINI_BATCH_SIZE, config.context_length), device=DEVICE) # NOTE THIS IS NOT ACCURATE SINCE NOT SHARING DATA
            y = torch.randint(0, config.vocab_size - 1, (MINI_BATCH_SIZE, config.context_length), device=DEVICE)

            y_hat = model(x)
            loss = crossEntropyLoss(y, y_hat).mean()
            loss.backward()

            model.finish_gradient_synchronization()
            optimizer.step()

            prof.step()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # stacks_export_path = os.path.join(current_dir, f'stacks/lm_profiler_stacks_{version}.txt')
    rank = dist.get_rank()
    chrome_export_path = os.path.join(current_dir, f'traces/{type}/lm_profiler_chrome_{version}_{rank}.json')

    # prof.export_stacks(stacks_export_path, "self_cuda_time_total")
    prof.export_chrome_trace(chrome_export_path)


def main():
    parser = argparse.ArgumentParser(description='Train a model with float16 parameters.')
    parser.add_argument('--version', type=str, default=None, help='Version Number of Model')
    parser.add_argument('--type', type=str, default=None, help='Version Number of Model')
    args = parser.parse_args()

    rank, world_size, local_rank, local_world_size = setup()
    torch.cuda.set_device(local_rank % local_world_size)

    versions = ["s", "m", "l", "xl", "2.7"]

    if args.version != "sweep":
        versions = [args.version]

    for version in versions:
        train_model(version, args.type, world_size)

if __name__ == "__main__":
    main()