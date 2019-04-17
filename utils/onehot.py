import numpy as np
import torch


def to_one_hot(x, size, device='cpu'):
    assert type(x) is int, "input must be an integer. Currently input has type {}".format(type(x))
    assert x < size, "input x must be strictly smaller than size"

    v = torch.zeros(size, dtype=torch.float32, device=device)
    v[x] = 1

    return v
