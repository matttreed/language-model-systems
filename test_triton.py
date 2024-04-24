import torch
from cs336_systems.kernels import WeightedSumFunc


def main():
    x = torch.randn(8,32,64).cuda()
    w = torch.randn(64).cuda()
    weighted_sum_func = WeightedSumFunc.apply
    triton_y = weighted_sum_func(x, w)
    python_y = (w * x).sum(axis=-1)

    assert torch.allclose(triton_y, python_y), "Output mismatch"

if __name__ == "__main__":
    main()