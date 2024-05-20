# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import torch

from .hadamard_tensors import (
    get_had12,
    get_had20,
    get_had28,
    get_had36,
    get_had40,
    get_had52,
    get_had60,
    get_had108,
    get_had140,
    get_had156,
    get_had172,
)

try:
    from scipy.linalg import hadamard
except ImportError:
    hadamard = None
import math

import torch
import torch.nn.functional as F


def hadamard_transform_ref(x, scale=1.0):
    """
    x: (..., dim)
    out: (..., dim)
    """
    if hadamard is None:
        raise ImportError("Please install scipy")
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def factored_hadamard(X: torch.Tensor) -> torch.Tensor:
    """
    Default Hadamard transform implementation. This is a wrapper around `factored_hadamard` that uses the fast Hadamard transform if available.
    """
    return factored_fast_hadamard(X) if hadamard else factored_slow_hadamard(X)


def get_hadK(n: int) -> torch.Tensor:
    """
    Attempts to factorize N = KR where R is a power of 2. If so, returns a Hadamard matrix of size K.
    """
    if is_pow2(n):
        return torch.ones((1, 1))

    hadamard_functions = {
        172: get_had172,
        156: get_had156,
        140: get_had140,
        108: get_had108,
        60: get_had60,
        52: get_had52,
        40: get_had40,
        36: get_had36,
        28: get_had28,
        20: get_had20,
        12: get_had12,
    }

    for K, get_hadK in hadamard_functions.items():
        if n % K == 0 and is_pow2(n // K):
            return get_hadK()

    raise ValueError(f"Unsupported Hadamard dimension: {n}")


def factored_slow_hadamard(X: torch.Tensor) -> torch.Tensor:
    """
    This is a PyTorch version of `factored_fast_hadamard`. Performs a Hadamard transform on the last dimension of X. If the last dimension is not a power of 2,
    we factorize the dimension n as n = KR where K is the size of a known Hadamard matrix (not necessarily
      a Hadamard-Walsh Matrix). Then we complete a Hadamard transform efficiently over R, and using a matmul over K.

    """
    n = X.shape[-1]
    hadK = get_hadK(n)
    return _apply_slow_hadamard(X, hadK)


def _apply_slow_hadamard(X: torch.Tensor, hadK: torch.Tensor) -> torch.Tensor:
    """
    Performs a Hadamard transform on X given a Hadamard matrix hadK of size K.
    """
    n = X.shape[-1]
    K = hadK.shape[0]
    input = X.clone().contiguous().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def factored_fast_hadamard(X: torch.tensor) -> torch.tensor:
    """
    Performs a Hadamard transform on the last dimension of X. If the last dimension is not a power of 2,
    we factorize the dimension n as n = KR where K is the size of a known Hadamard matrix (not necessarily
      a Hadamard-Walsh Matrix). Then we complete a Hadamard transform efficiently over R, and using a matmul over K.
    """
    n = X.shape[-1]
    hadK = get_hadK(n)
    return _apply_slow_hadamard(X, hadK)


def apply_hadamard_headwise(module: torch.nn.Linear, head_dim: int):
    """
    Apply a Hadamard transform to each head (chunk of columns) of the weight matrix of a module.
    We do this by reshaping the weight matrix into (in_features, num_heads, head_dim) and applying the Hadamard transform along the last dimension, then reshaping.

    Args:
    - module: the torch.nn.Linear instance to modify.
    - head_dim: the head dimension. Must be a power of 2.
    """
    assert is_pow2(head_dim), "Head dimension must be a power of 2!"

    W_ = module.weight.data.t()
    dtype = W_.dtype
    dev = W_.device
    W_ = W_.float().cuda()

    scale = 1 / math.sqrt(head_dim)
    num_heads = module.out_features // head_dim
    W_ = hadamard_transform_ref(W_.reshape(module.in_features, num_heads, head_dim), scale=scale)
    W_ = W_.reshape((module.in_features, module.out_features)).t()

    module.weight.data = W_.to(device=dev, dtype=dtype)


def apply_hadamard(module: torch.nn.Linear) -> None:
    """
    Modifies the weights contained in a torch.nn.Linear instance. If the weights are W,
    this turns them into HW, where H is a Hadamard matrix.

    Args:
    - module: the torch.nn.Linear instance to modify.
    """
    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    W_ = W_.float().cuda()

    W_ = factored_fast_hadamard(W_) if hadamard else factored_slow_hadamard(W_)
    module.weight.data = W_.to(device=dev, dtype=dtype)


def random_hadamard_matrix(size: int, seed: int = 0) -> torch.Tensor:
    """
    Generates a random Hadamard matrix of size `size`. See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation" for details.
    """
    torch.manual_seed(seed)
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return factored_slow_hadamard(Q)


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)
