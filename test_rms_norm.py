import torch
import timeit
import cs336_basics.model.layers as layers

def main():
    num_rows = 50000
    last_dimensions = [1024, 2048, 4096, 8192]
    num_forward_passes = 1000

    for last_dim in last_dimensions:

        input_shape = (num_rows, last_dim)
        x = torch.randn(input_shape).to("cuda")
        w = torch.randn(last_dim).to("cuda")
        bias = torch.randn(last_dim).to("cuda")

        rmsnorm = layers.RMSNorm(last_dim).to("cuda")
        rmsnorm.weight.data = w
        layernorm = torch.nn.LayerNorm(last_dim).to("cuda")
        layernorm.weight.data = w
        layernorm.bias.data = bias


        # Benchmark RMSNorm
        start_time = timeit.default_timer()
        for _ in range(num_forward_passes):
            # Perform forward pass for RMSNorm
            # Replace the following line with your implementation of RMSNorm forward pass
            output = rmsnorm(x)
            torch.cuda.synchronize()

        rmsnorm_time = timeit.default_timer() - start_time

        # Benchmark LayerNorm
        start_time = timeit.default_timer()
        for _ in range(num_forward_passes):
            # Perform forward pass for LayerNorm
            # Replace the following line with your implementation of LayerNorm forward pass
            output = layernorm(x)
            torch.cuda.synchronize()

        layernorm_time = timeit.default_timer() - start_time

        print(f"Last Dimension: {last_dim}")
        print(f"RMSNorm Time: {rmsnorm_time} seconds")
        print(f"LayerNorm Time: {layernorm_time} seconds")
        print()

main()