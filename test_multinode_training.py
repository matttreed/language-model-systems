import os
from datetime import timedelta
import torch
import torch.distributed as dist
from cs336_basics.configs.config import Config
from cs336_basics.model.transformer import Transformer
from cs336_basics.training.optimizer import AdamW
from cs336_basics.model.util import crossEntropyLoss, load_model, save_model, get_batch, log_validation_loss, log, build_model
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy

BACKEND = "gloo"
DEVICE = "cpu"
BATCH_SIZE = 24
NUM_BATCHES = 16
D_MODEL = 128
SEED = 12345

class ToyModel(nn.Module):
    def __init__(self, d_model):
        super(ToyModel, self).__init__()
        self.lin1 = nn.Linear(d_model, d_model, bias=False)
        self.lin2 = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):
        x = self.lin1(x)
        x = nn.functional.relu(x)
        x = self.lin2(x)
        return torch.squeeze(x, dim=-1)

def setup():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])

    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]

    timeout = timedelta(seconds=60)
    dist.init_process_group(BACKEND, rank=rank, world_size=world_size, timeout=timeout)
    return rank, world_size, local_rank, local_world_size

# def train_parallel():

#     data = torch.randn((tensor_len), device=DEVICE)

#     start_time = torch.cuda.Event(enable_timing=True)
#     end_time = torch.cuda.Event(enable_timing=True)
#     start_time.record()
#     dist.all_reduce(data, async_op=False)
#     # if device == "cuda":
#     #     torch.cuda.synchronize()
#     end_time.record()
#     end_time.synchronize()
#     duration = start_time.elapsed_time(end_time)
#     all_durations = [None] * world_size
#     dist.all_gather_object(all_durations, duration)

#     if rank == 0:
#         avg_time = sum(all_durations) / len(all_durations)
#         print(f"{avg_time}")


# def train_model(version: str):
#     config = Config(version)
#     torch.random.manual_seed(config.random_seed)
#     np.random.seed(config.random_seed)
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     model = build_model(config).to(device)

#     model.train()

#     optimizer = AdamW(
#         params=model.parameters(), 
#         weight_decay=config.training.weight_decay,
#         betas=config.training.betas,
#         eps=config.training.eps,
#         max_grad_norm=config.training.max_grad_norm,
#         alpha_min=config.training.alpha_min,
#         alpha_max=config.training.alpha_max,
#         T_warmup=config.training.T_warmup,
#         T_cosine=config.training.T_cosine
#     )

#     if from_checkpoint_k:
#         load_model(model, optimizer, version, from_checkpoint_k)

#     train_data_name = config.data.training_data
#     valid_data_name = config.data.validation_data
#     train_data = np.memmap(f"data/processed/{train_data_name}.npy", dtype=np.int16, mode="r", offset=16*8)
#     valid_data = np.memmap(f"data/processed/{valid_data_name}.npy", dtype=np.int16, mode="r", offset=16*8)

#     iteration = from_checkpoint_k * 1000 if from_checkpoint_k else 0

#     log(version, f"Training {version} from iteration {iteration}")
#     while iteration < config.training.total_iterations:
#         optimizer.zero_grad()
#         x, y = get_batch(data=train_data,
#                         batch_size=config.training.batch_size,
#                         context_length=config.transformer.context_length,
#                         device=device)
#         y_hat = model(x)
#         loss = crossEntropyLoss(y, y_hat).mean()
#         loss.backward()

#         optimizer.step()

#         if iteration % config.training.log_every == 0:
#             log_validation_loss(iteration, model, valid_data, version, config, device)

#         if iteration != 0 and iteration % config.training.checkpoint_every == 0:
#             save_model(model, optimizer, version, iteration, config)

#         iteration += 1

#     save_model(model, optimizer, version, iteration, config)



def main():
    rank, world_size, local_rank, local_world_size = setup()
    torch.cuda.set_device(local_rank % local_world_size)

    MINI_BATCH_SIZE = BATCH_SIZE // world_size

    assert BATCH_SIZE % world_size == 0, "batch_size must divide world_size"

    x = torch.randn((BATCH_SIZE * NUM_BATCHES, D_MODEL), device=DEVICE)
    y = torch.randn((BATCH_SIZE * NUM_BATCHES), device=DEVICE)

    model = ToyModel(D_MODEL).to(DEVICE)
    model_parallel = deepcopy(model)

    optimizer = AdamW(params=model.parameters(), lr=.01)
    optimizer_parallel = AdamW(params=model_parallel.parameters(), lr= 0.01)
    
    for i in range(NUM_BATCHES):
        if rank == 0:
            print(f"Batch {i}")
        x_slice = x[i * BATCH_SIZE:(i+1) * BATCH_SIZE]
        y_slice = y[i * BATCH_SIZE:(i+1) * BATCH_SIZE]

        optimizer.zero_grad()
        y_slice_hat = model(x_slice)
        loss = torch.nn.functional.mse_loss(y_slice_hat, y_slice)
        loss.backward()
        optimizer.step()


        x_parallel_slice = x_slice[rank * MINI_BATCH_SIZE: (rank+1) * MINI_BATCH_SIZE]
        y_parallel_slice = y_slice[rank * MINI_BATCH_SIZE: (rank+1) * MINI_BATCH_SIZE]

        optimizer_parallel.zero_grad()
        y_parallel_slice_hat = model_parallel(x_parallel_slice)
        loss_parallel = torch.nn.functional.mse_loss(y_parallel_slice_hat, y_parallel_slice)
        loss_parallel.backward()

        for param in model_parallel.parameters():
            dist.all_reduce(param.grad.data, async_op=False)
            param.grad.data /= world_size

        # for group in optimizer_parallel.param_groups:
        #     for param in group["params"]:
        #         dist.all_reduce(param.grad.data, async_op=False)
        #         param.grad.data /= world_size

        optimizer_parallel.step()

        # if rank == 0:
        #     print(model.lin1.weight.data, model_parallel.lin1.weight.data)
        # print(loss_parallel, x_parallel_slice, model_parallel.lin2.weight.data, model_parallel.lin2.weight.grad.data)
        # if rank == 0:
        #     print(loss, x_slice, model.lin2.weight.data, model.lin2.weight.grad.data)

    if rank == 0:
        print("Checking Correctness")
        for param, param_parallel in zip(model.parameters(), model_parallel.parameters()):
            diff = param.data - param_parallel.data
            l1_norm = torch.norm(diff, p=1)
            l2_norm = torch.norm(diff, p=2)
            max_diff = torch.max(torch.abs(diff))
            num_significant_diff = torch.sum(torch.abs(diff) > 1e-5)

            print("L1 Norm of Difference:", l1_norm.item())
            print("L2 Norm of Difference:", l2_norm.item())
            print("Max Difference:", max_diff.item())
            num_diff = num_significant_diff.item()
            total = param.data.numel()
            print(f"Number of Significant Differences: {num_diff}/{total}")
            assert torch.allclose(param, param_parallel, atol=1e-6), "Params not close"

        print("Success")


if __name__ == "__main__":
    main()