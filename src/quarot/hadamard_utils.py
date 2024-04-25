# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import torch
from fast_hadamard_transform import hadamard_transform

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


def get_hadK(n: int) -> tuple[torch.Tensor, int]:
    if is_pow2(n):
        return None, 1

    Ks = [172, 156, 140, 108, 60, 52, 40, 36, 28, 20, 12]
    for K in Ks:
        if n % K == 0:
            assert is_pow2(n // K)
            return get_hadamard_tensor(K), K

    raise ValueError(f"Unsupported Hadamard dimension: {n}")


def get_hadamard_tensor(dim: int) -> torch.Tensor:
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
    return hadamard_functions[dim]()


def matmul_hadU(X: torch.Tensor) -> torch.Tensor:
    """
    Performs a Hadamard transform on the last dimension of X.
    """
    n = X.shape[-1]
    hadK, K = get_hadK(n)
    input = X.clone().view(-1, n, 1)
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
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def random_hadamard_matrix(size: int, seed: int = 0) -> torch.Tensor:
    """
    Generates a random Hadamard matrix of size `size`. See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation" for details.
    """
    torch.manual_seed(seed)
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q)


def matmul_hadU_cuda(X: torch.tensor, hadK: torch.tensor, K: int) -> torch.tensor:
    """
    Performs a Hadamard transform on the last dimension of X using an efficient CUDA implementation.
    """
    n = X.shape[-1]
    if K == 1:
        return hadamard_transform(X.contiguous(), 1.0 / torch.tensor(n).sqrt())

    input = X.view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), 1.0 / torch.tensor(n).sqrt())
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.reshape(X.shape)


def apply_hadamard(module: torch.nn.Linear, had_dim: int = -1, output: bool = False):
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    W_ = W_.float().cuda()

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.t(), had_K, K).t()
        else:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K)
    else:
        # Apply Hadamard to the last had_dim chunks of the weights
        if output:
            W_ = W_.t()
            transposed_shape = W_.shape
            W_ = (
                hadamard_transform(
                    W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim), scale=1 / math.sqrt(had_dim)
                )
                .reshape(transposed_shape)
                .t()
            )
        else:
            raise NotImplementedError("Not implemented (or tested) yet!")
    module.weight.data = W_.to(device=dev, dtype=dtype)


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)
