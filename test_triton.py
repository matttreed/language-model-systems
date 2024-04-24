import torch
from cs336_systems.kernels import WeightedSumFunc_Triton, RMS_Norm_Func_Triton
from cs336_systems.functions import RMS_Norm_Func_Python


def main():
    x = torch.randn(4,2,3).cuda()
    w = torch.randn(3).cuda()
    weighted_sum_func = WeightedSumFunc_Triton.apply
    triton_y = weighted_sum_func(x, w)
    python_y = (w * x).sum(axis=-1)

    print("TRITON: ", triton_y)
    print("PYTHON: ", python_y)

    assert torch.allclose(triton_y, python_y), "Output mismatch"

if __name__ == "__main__":
    main()