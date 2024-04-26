import torch
from cs336_systems.functions import RMS_Norm_Func_Python

class RMSNormTriton(torch.nn.Module):
    def __init__(self, d_model):
        super(RMSNormTriton, self).__init__()
        self.d_model = d_model
        self.weight = torch.nn.Parameter(torch.ones(d_model)) # gain

    def forward(self, x):
        return RMS_Norm_Func_Python.apply(x, self.weight)