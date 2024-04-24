import torch
from cs336_basics.model.layers import RMSNorm

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

        return grad_x, grad_weight
