import torch

def rmsnorm_jvp_g(dL_drms, x, g):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5)
    num_dims = x.dim()
    dims_to_sum = list(range(num_dims - 1))
    result = torch.sum(dL_drms * x / rms, dim=dims_to_sum)
    return result

def rmsnorm_jvp_x(dL_drms, x, g):
    eps = 1e-5
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)

    grad_x_1 = (dL_drms * g) / rms

    grad_x_2 = - x * torch.mean(dL_drms * x * g, dim=-1, keepdim=True) / (rms**3)

    return grad_x_1 + grad_x_2

class RMS_Norm_Func_Python(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        eps = 1e-5
        ctx.save_for_backward(x, weight)
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps) * weight
    
    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors

        return rmsnorm_jvp_x(grad_out, x, weight), rmsnorm_jvp_g(grad_out, x, weight)
