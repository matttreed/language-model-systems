import torch

def rmsnorm_jvp_g(dL_drms, x, g):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5)
    num_dims = x.dim()
    dims_to_sum = list(range(num_dims - 1))
    result = torch.sum(dL_drms * x / rms, dim=dims_to_sum)
    return result

def rmsnorm_jvp_x(dL_drms, x, g):
    eps = 1e-5
    d = x.shape[-1]

    # Compute RMS and normalized x
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)

    # Gradient with respect to x due to scaling by gain
    grad_x_scaling = (dL_drms * g) / rms

    # Additional gradient due to normalization's impact on RMS
    grad_x_additional = -(
        x * torch.mean(dL_drms * x * g, dim=-1, keepdim=True) / (rms**3)
    )

    print(x.shape, grad_x_scaling.shape, grad_x_additional.shape)

    # Total gradient with respect to x
    return grad_x_scaling + grad_x_additional

class RMS_Norm_Func_Python(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        eps = 1e-5
        ctx.save_for_backward(x, weight)
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps) * weight
    
    @staticmethod
    def backward(ctx, grad_out):
        eps = 1e-5
        x, weight = ctx.saved_tensors

        rms_norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        
        # Gradient with respect to x
        grad_x_norm = (grad_out * weight) / rms_norm
        mean_x2 = torch.mean(x**2, dim=-1, keepdim=True)
        
        # Additional terms for gradient with respect to x (due to normalization)
        grad_x_additional = -(x * torch.mean(grad_out * x * weight, dim=-1, keepdim=True)) / (rms_norm * mean_x2)
        
        # Total gradient with respect to x
        grad_x = grad_x_norm + grad_x_additional
        
        # Gradient with respect to weight
        grad_weight = torch.sum(grad_out * (x / rms_norm), dim=0)

        return rmsnorm_jvp_x(grad_out, x, weight), rmsnorm_jvp_g(grad_out, x, weight)
