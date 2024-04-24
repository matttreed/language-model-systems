import torch
from cs336_systems.rms_kernel import WeightedSumFunc


def main():
    x = torch.randn(8,32,64)
    w = torch.randn(64)
    weighted_sum_func = WeightedSumFunc.apply
    triton_y = weighted_sum_func(x, w)
    python_y = (w * x).sum(axis=-1)

    assert torch.allclose(triton_y, python_y), "Output mismatch"


main()