import torch

class RMS_Norm_Func_Python(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError