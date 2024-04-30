import os
from datetime import timedelta
import torch
import torch.distributed as dist

TENSOR_LENS = [512, 1080, 10800, 54000, 108000, 540000, 1166400]

def setup(backend):
    # These variables are set via srun
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    # MASTER_ADDR and MASTER_PORT should have been set in our sbatch script,
    # so we make sure that's the case.
    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]
    # Default timeout is 30 minutes. Reducing the timeout here, so the job fails quicker if there's
    # a communication problem between nodes.
    timeout = timedelta(seconds=60)
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout)
    return rank, world_size, local_rank, local_world_size

def multinode_distributed_demo():
    tensor_len = int(TENSOR_LENS[int(os.environ["LEN_TENSOR"])] * 1080 / 4)
    device = os.environ["DEVICE"]
    backend = os.environ["BACKEND"]

    rank, world_size, local_rank, local_world_size = setup(backend)

    torch.cuda.set_device(local_rank % local_world_size)
    data = torch.randn((tensor_len), device=device)

    for i in range(5):
        # print(i)
        dist.all_reduce(data, async_op=False)
        data = torch.randn((tensor_len), device=device)
    
    # if device == "cuda":
    #     torch.cuda.synchronize()

    data = torch.randn((tensor_len), device=device)

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    dist.all_reduce(data, async_op=False)
    # if device == "cuda":
    #     torch.cuda.synchronize()
    end_time.record()
    end_time.synchronize()
    duration = start_time.elapsed_time(end_time)
    all_durations = [None] * world_size
    dist.all_gather_object(all_durations, duration)
    if rank == 0:
        avg_time = sum(all_durations) / len(all_durations)
        print(f"{avg_time}")








if __name__ == "__main__":
    multinode_distributed_demo()