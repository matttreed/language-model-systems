import torch
import timeit
import cs336_basics.model.layers as layers
import cs336_systems.layers as layers_systems
from cs336_systems.kernels import RMS_Norm_Func_Triton
# from torch.profiler import profile, record_function, ProfilerActivity
import torch.autograd.profiler as profiler

forward_only = False

def main():
    num_rows = 50000
    last_dimensions = [1024, 2048, 4096, 8192]
    # last_dimensions = [1024]
    num_forward_passes = 1000

    # with profiler.profile(use_cuda=True, record_shapes=True) as prof:

    for last_dim in last_dimensions:

        input_shape = (num_rows, last_dim)
        x = torch.randn(input_shape).to("cuda")
        w = torch.randn(last_dim).to("cuda")
        bias = torch.randn(last_dim).to("cuda")
        dy = torch.randn(input_shape).to("cuda")

        rmsnorm = layers.RMSNorm(last_dim).to("cuda")
        rmsnorm.weight.data = w
        rms_triton = layers_systems.RMSNormTriton(last_dim).to("cuda")
        rms_triton.weight.data = w
        layernorm = torch.nn.LayerNorm(last_dim).to("cuda")
        layernorm.weight.data = w
        layernorm.bias.data = bias
        compiled_rms_norm = torch.compile(rmsnorm).to("cuda")

    # with profiler.record_function("RMS"):
        # Benchmark RMSNorm
        start_time = timeit.default_timer()
        for i in range(num_forward_passes):
            # Perform forward pass for RMSNorm
            # Replace the following line with your implementation of RMSNorm forward pass
            output = rmsnorm(x)

            if not forward_only:
                output.backward(dy)
            torch.cuda.synchronize()
            rmsnorm.weight.grad = None
            x.grad = None
            output.grad = None
            dy.grad = None

        rmsnorm_time = timeit.default_timer() - start_time

    # with profiler.record_function("LayerNorm"):
        # Benchmark LayerNorm
        start_time = timeit.default_timer()
        for i in range(num_forward_passes):
            # Perform forward pass for LayerNorm
            # Replace the following line with your implementation of LayerNorm forward pass
            output = layernorm(x)
            # output_layernorm = torch.nn.functional.layer_norm(x, w.shape, w, bias)
            if not forward_only:
                output.backward(dy)
            torch.cuda.synchronize()
            layernorm.weight.grad = None
            layernorm.bias.grad = None
            x.grad = None
            output.grad = None
            dy.grad = None
        layernorm_time = timeit.default_timer() - start_time

    # with profiler.record_function("RMS_Triton"):
        # # Benchmark TritonNorm
        start_time = timeit.default_timer()
        for _ in range(num_forward_passes):
            # Perform forward pass for LayerNorm
            # Replace the following line with your implementation of LayerNorm forward pass
            # output = rms_triton(x)
            # output = RMS_Norm_Func_Triton.apply(x, w)
            output = rms_triton(x)
            if not forward_only:
                output.backward(dy)
            
            torch.cuda.synchronize()
            rms_triton.weight.grad = None
            x.grad = None
            output.grad = None
            dy.grad = None

        triton_time = timeit.default_timer() - start_time

    # with profiler.record_function("RMS_Compiled"):
        start_time = timeit.default_timer()
        for _ in range(num_forward_passes):

            output = compiled_rms_norm(x)

            if not forward_only:
                output.backward(dy)
            
            torch.cuda.synchronize()
            rms_triton.weight.grad = None
            x.grad = None
            output.grad = None
            dy.grad = None

        compiled_time = timeit.default_timer() - start_time

        print(f"Last Dimension: {last_dim}")
        print(f"RMSNorm Time: {rmsnorm_time} seconds")
        print(f"LayerNorm Time: {layernorm_time} seconds")
        print(f"Triton Time: {triton_time} seconds")
        print(f"Compiled Time: {compiled_time} seconds")
        print()
    
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=1000))

if __name__ == "__main__":
    main()