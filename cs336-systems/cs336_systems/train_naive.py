from cs336_basics.configs.config import Config
from cs336_basics.model.transformer import Transformer
from cs336_basics.training.optimizer import AdamW
from cs336_basics.model.util import crossEntropyLoss, load_model, save_model, get_batch, log_validation_loss, log, build_model
from cs336_systems.config import Systems_Config
import torch
import numpy as np
import os
from datetime import timedelta
import torch.distributed as dist
import timeit

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

def flatten_gradients(model):
    grads = [p.grad.data.view(-1) for p in model.parameters() if p.grad is not None]
    return torch.cat(grads)  # Concatenate all gradients

def unflatten_gradients(flat_grad, model):
    grad_idx = 0
    for param in model.parameters():
        if param.grad is not None:
            num_param = param.numel()  # Number of elements in the parameter
            param.grad.data.copy_(flat_grad[grad_idx:grad_idx + num_param].view_as(param))
            grad_idx += num_param

def train_model(version: str):

    rank, world_size, local_rank, local_world_size = setup()
    torch.cuda.set_device(local_rank % local_world_size)

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
    ).to(DEVICE)

    model.train()

    optimizer = AdamW(
        params=model.parameters(),
        lr=0.01
    )

    # train_data_name = config.data.training_data
    # valid_data_name = config.data.validation_data
    # train_data = np.memmap(f"data/processed/{train_data_name}.npy", dtype=np.int16, mode="r", offset=16*8)
    # valid_data = np.memmap(f"data/processed/{valid_data_name}.npy", dtype=np.int16, mode="r", offset=16*8)

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
        # for param in model.parameters():
        #     dist.all_reduce(param.grad.data, async_op=False)
        #     param.grad.data /= world_size

        flat_grad = flatten_gradients(model)

        # Perform all-reduce on the flattened gradient tensor
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)

        # Average the gradients by the number of processes
        flat_grad /= world_size

        # Split the flattened gradients back into the original structure
        unflatten_gradients(flat_grad, model)

        torch.cuda.synchronize()
        all_reduce_time.append(timeit.default_timer()-t)

        t = timeit.default_timer()
        optimizer.step()

        torch.cuda.synchronize()
        optimizer_step_time.append(timeit.default_timer()-t)

    print("FORward:", sum(forward_time)/len(forward_time))
    print("all reduce: ", sum(all_reduce_time)/len(all_reduce_time))
    print("optimizer: ", sum(optimizer_step_time)/len(optimizer_step_time))


if __name__ == "__main__":
    train_model("2.7")