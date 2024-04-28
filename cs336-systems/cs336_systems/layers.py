import torch
from cs336_systems.kernels import RMS_Norm_Func_Triton

class RMSNormTriton(torch.nn.Module):
    def __init__(self, d_model):
        super(RMSNormTriton, self).__init__()
        self.d_model = d_model
        self.weight = torch.nn.Parameter(torch.ones(d_model)) # gain

    def forward(self, x):
        return RMS_Norm_Func_Triton.apply(x, self.weight)