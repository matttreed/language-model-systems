import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import signal
import timeit
import argparse

NUM_GPU = 6

def cleanup():
    dist.destroy_process_group()

def signal_handler(sig, frame):
    print("Caught signal, cleaning up...")
    cleanup()
    exit(1)

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, device, backend, tensor_len, size_name):
    setup(rank, world_size, backend)
    torch.cuda.set_device(rank % NUM_GPU)
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
        print(f"{avg_time}, {world_size}, {size_name}")
    # dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--backend', type=str, default="nccl", help='device')
    args = parser.parse_args()

    backend = args.backend
    device = args.device
    print(f"Testing All Reduce Single node with device {device} and backend {backend}")
    for world_size in [2,4,6]:
        for size, size_name in [
            (512, "512KB"),
            (1080, "1MB"),
            (10800, "10MB"),
            (54000, "50MB"),
            (108000, "100MB"), 
            (540000, "500MB"),
            (1166400, "1GB")
            ]:
            tensor_len = int(size * 1080 / 4)

            mp.spawn(fn=distributed_demo, args=(world_size, device, backend, tensor_len, size_name), nprocs=world_size, join=True)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()