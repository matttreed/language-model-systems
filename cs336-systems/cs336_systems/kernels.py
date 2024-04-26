import triton
import triton.language as tl
import torch
import os

@triton.jit
def weighted_sum_fwd(x_ptr : tl.pointer_type,
                     weight_ptr : tl.pointer_type,
                     x_row_stride : tl.uint32,
                     output_ptr : tl.pointer_type,
                     H : tl.uint32,
                     BLOCK_SIZE: tl.constexpr):
    # Each instance will compute the weighted sum of a row of x.
    row_idx = tl.program_id(0)
    # Pointer to the first entry of the row this instance sums up.
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    # Pointers to the entries we'll sum up.
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets
    # Load the data from x given the pointers to its entries,
    # using a mask since BLOCK_SIZE may be > H.
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)
    output = tl.sum(row * weight)
    # Write back output (a single scalar per instance).
    output_ptr = output_ptr + row_idx
    tl.store(output_ptr, output)

@triton.jit
def weighted_sum_backward(grad_output_ptr : tl.pointer_type,
                          grad_x_ptr : tl.pointer_type,
                          partial_grad_weight_ptr : tl.pointer_type,
                          x_ptr : tl.pointer_type,
                          weight_ptr : tl.pointer_type,
                          x_row_stride : tl.uint32,
                          H : tl.uint32,
                          BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    grad_output_ptrs = weight_ptr + offsets
    mask = offsets < H
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0)
    grad_output = tl.load(grad_output_ptr + row_idx) # (scalar)
    grad_x_row = grad_output * weight # (See Eq 4)
    grad_x_ptr = grad_x_ptr + row_idx * x_row_stride
    tl.store(grad_x_ptr + offsets, grad_x_row, mask=mask)
    partial_grad_weight_ptr = partial_grad_weight_ptr + row_idx * x_row_stride + offsets
    row = tl.load(row_start_ptr + offsets, mask=mask, other=0)
    grad_weight_row = row * grad_output # (See Eq 3)
    tl.store(partial_grad_weight_ptr, grad_weight_row, mask=mask)

class WeightedSumFunc_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Remember x and weight for the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        ctx.save_for_backward(x, weight)

        H, output_dims = x.shape[-1], x.shape[:-1]

        assert len(weight.shape) == 1 and weight.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        y = torch.empty(output_dims, device=x.device)

        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(n_rows, )](
            x, weight, x.stride(0), y, H,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        N, H = x.shape
        # Allocate output tensors.
        partial_grad_weight = torch.empty_like(x)
        grad_x = torch.empty_like(x)
        weighted_sum_backward[(N, )](
            grad_out, grad_x, partial_grad_weight,
            x, weight, x.stride(0), H,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return grad_x, partial_grad_weight.sum(axis=0)
    



@triton.jit
def rms_norm_fwd(x_ptr : tl.pointer_type,
                     weight_ptr : tl.pointer_type,
                     x_row_stride : tl.uint32,
                     output_ptr : tl.pointer_type,
                     H : tl.uint32,
                     eps: tl.float32,
                     BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H

    # Load input row and gain
    x_row = tl.load(row_start_ptr + offsets, mask=mask, other=0)
    gain = tl.load(weight_ptr + offsets, mask=mask, other=1)

    # Compute RMS
    squared_row =  x_row * x_row
    squared_mean = tl.sum(squared_row) / H
    rms = tl.sqrt(squared_mean + eps)

    # Normalize and apply gain
    normalized_row = x_row / rms
    scaled_row = normalized_row * gain

    # Store the result in the output
    tl.store(output_ptr + row_idx * x_row_stride + offsets, scaled_row, mask=mask)

@triton.jit
def rms_norm_backward(grad_output_ptr : tl.pointer_type,
                          grad_x_ptr : tl.pointer_type,
                          partial_grad_weight_ptr : tl.pointer_type,
                          x_ptr : tl.pointer_type,
                          weight_ptr : tl.pointer_type,
                          x_row_stride : tl.uint32,
                          H : tl.uint32,
                          eps: tl.float32,
                          BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H

    grad_output_row = tl.load(grad_output_ptr + row_idx * x_row_stride + offsets, mask=mask, other=0)
    x_row = tl.load(x_ptr + row_idx * x_row_stride + offsets, mask=mask, other=0)
    gain_row = tl.load(weight_ptr + offsets, mask=mask, other=1)

    # Compute RMS
    squared_row =  x_row * x_row
    squared_mean = tl.sum(squared_row) / H
    rms = tl.sqrt(squared_mean + eps)

    # Gradient with respect to input (grad_x)
    normalized_row = x_row / rms
    # grad_x_row = grad_output_row * gain_row * (1 - normalized_row) / rms
    # tl.store(grad_x_ptr + row_idx * x_row_stride + offsets, grad_x_row, mask=mask)

    # Gradient with respect to gain (grad_gain)
    grad_gain_row = grad_output_row * normalized_row
    tl.store(partial_grad_weight_ptr + row_idx * x_row_stride + offsets, grad_gain_row, mask=mask)
    # row_idx = tl.program_id(0)
    # row_start_ptr = x_ptr + row_idx * x_row_stride
    # offsets = tl.arange(0, BLOCK_SIZE)
    # x_ptrs = row_start_ptr + offsets
    # grad_output_ptrs = weight_ptr + offsets
    # mask = offsets < H
    # weight = tl.load(weight_ptr + offsets, mask=mask, other=0)
    # grad_output = tl.load(grad_output_ptr + row_idx) # (scalar)
    # grad_x_row = grad_output * weight # (See Eq 4)
    # grad_x_ptr = grad_x_ptr + row_idx * x_row_stride
    # tl.store(grad_x_ptr + offsets, grad_x_row, mask=mask)
    # partial_grad_weight_ptr = partial_grad_weight_ptr + row_idx * x_row_stride + offsets
    # row = tl.load(row_start_ptr + offsets, mask=mask, other=0)
    # grad_weight_row = row * grad_output # (See Eq 3)
    # tl.store(partial_grad_weight_ptr, grad_weight_row, mask=mask)

class RMS_Norm_Func_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Remember x and weight for the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        ctx.save_for_backward(x, weight)

        H = x.shape[-1]
        n_rows = x.numel() // H  # Flatten other dimensions
        x_reshaped = x.reshape(n_rows, H)

        assert len(weight.shape) == 1 and weight.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        

        ctx.BLOCK_SIZE = triton.next_power_of_2(H)

        y_reshaped = torch.empty((n_rows, H), device=x.device)

        # Launch our kernel with n instances in our 1D grid.
        rms_norm_fwd[(n_rows, )](
            x, weight, x_reshaped.stride(0), y_reshaped, H, eps=1e-9,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        y = y_reshaped.view(x.shape)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors

        H = x.shape[-1]
        n_rows = x.numel() // H  # Flatten other dimensions
        x_reshaped = x.reshape(n_rows, H)

        partial_grad_weight = torch.empty_like(x_reshaped)
        grad_x = torch.empty_like(x_reshaped)
        rms_norm_backward[(n_rows, )](
            grad_out, grad_x, partial_grad_weight,
            x_reshaped, weight, x_reshaped.stride(0), H, 1e-5,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        # print(partial_grad_weight)
        return grad_x.view(x.shape), partial_grad_weight.sum(axis=0)

