from cs336_basics.configs.config import Config
from cs336_basics.model.transformer import Transformer
from cs336_basics.training.optimizer import AdamW
from cs336_basics.model.util import crossEntropyLoss, load_model, save_model, get_batch, log_validation_loss, log, build_model
from cs336_systems.config import Systems_Config
from cs336_systems.ddp import DDP_Bucketed, DDP_Individual_Parameters, DDP_Naive
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
NUM_EXP = 20

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

    for i in range(NUM_EXP):
        optimizer.zero_grad()
        x = torch.randint(0, config.vocab_size - 1, (MINI_BATCH_SIZE, config.context_length), device=DEVICE) # NOTE THIS IS NOT ACCURATE SINCE NOT SHARING DATA
        y = torch.randint(0, config.vocab_size - 1, (MINI_BATCH_SIZE, config.context_length), device=DEVICE)

        t = timeit.default_timer()
        y_hat = model(x)
        loss = crossEntropyLoss(y, y_hat).mean()
        loss.backward()

        torch.cuda.synchronize()
        forward_time.append(timeit.default_timer()-t)

        t = timeit.default_timer()

        model.finish_gradient_synchronization()

        torch.cuda.synchronize()
        all_reduce_time.append(timeit.default_timer()-t)

        t = timeit.default_timer()
        optimizer.step()

        torch.cuda.synchronize()
        optimizer_step_time.append(timeit.default_timer()-t)

    if dist.get_rank() == 0:
        f = sum(forward_time)/len(forward_time)
        ar = sum(all_reduce_time)/len(all_reduce_time)
        b = sum(optimizer_step_time)/len(optimizer_step_time)
        t = f + ar + b
        # print("FORward:", sum(forward_time)/len(forward_time))
        # print("all reduce: ", sum(all_reduce_time)/len(all_reduce_time))
        # print("optimizer: ", sum(optimizer_step_time)/len(optimizer_step_time))
        print(type, version, t, f,ar,b)


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