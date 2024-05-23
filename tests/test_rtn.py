# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from quarot.rtn import calculate_scales, quantize_weight_rtn

torch.set_default_dtype(torch.float16)


@pytest.mark.quarot
@pytest.mark.gpu
@pytest.mark.parametrize(
    "bits, weight, expected_scales",
    [
        (3, torch.tensor([[-1.0, 1.0, 2.0, 3.0], [-0.5, 1.0, 2.0, -3.0]]), torch.tensor([[1.0], [1.0]])),
        (3, torch.tensor([[-1.0, 1.0, 1.0, 1.5], [-0.5, 1.0, 1.0, -1.5]]), torch.tensor([[0.5], [0.5]])),
        (4, torch.tensor([[-1.0, 1.0, 2.0, 3.5], [-0.5, 1.0, 2.0, -7.0]]), torch.tensor([[0.5], [1.0]])),
        (
            8,
            torch.tensor([[-1.0, 1.0, 2.0, 31.75], [-0.5, 1.0, 2.0, -63.5], [-0.5, 1.0, 2.0, -127.0]]),
            torch.tensor([[0.25], [0.5], [1.0]]),
        ),
    ],
)
def test_calculate_scales_symmetric(bits, weight, expected_scales):
    symmetric = True
    scales, offsets = calculate_scales(weight, bits, symmetric=symmetric, clip_weights=False, vectorized=False)
    assert torch.allclose(scales, expected_scales)
    assert offsets is None

    scales_nonvec, offsets_nonvec = calculate_scales(
        weight, bits, symmetric=symmetric, clip_weights=True, vectorized=False
    )
    scales_vec, offsets_vec = calculate_scales(weight, bits, symmetric=symmetric, clip_weights=True, vectorized=True)
    assert torch.allclose(scales_vec, scales_nonvec)
    assert offsets_nonvec is None
    assert offsets_vec is None


@pytest.mark.quarot
@pytest.mark.gpu
@pytest.mark.parametrize(
    "bits, weight, expected_quantized_weight",
    [
        (3, torch.tensor([[1.1, -2.3, 3.4], [-1.5, 2.6, -2.9]]), torch.tensor([[1.0, -2.0, 3.0], [-2.0, 3.0, -3.0]])),
        (4, torch.tensor([[1.1, -2.3, 3.5], [-1.5, 2.6, -7.0]]), torch.tensor([[2.0, -5.0, 7.0], [-2.0, 3.0, -7.0]])),
        (
            8,
            torch.tensor([[1.1, -2.3, 31.75], [-1.5, 2.6, -63.5], [-1.5, 2.6, -127.0]]),
            torch.tensor([[4.0, -9.0, 127.0], [-3.0, 5.0, -127.0], [-2.0, 3.0, -127.0]]),
        ),
    ],
)
def test_weight_rtn_symmetric(bits, weight, expected_quantized_weight):
    symmetric = True
    scales, offsets = calculate_scales(weight, bits, symmetric=symmetric, clip_weights=False)
    quantized_weight = quantize_weight_rtn(weight, scales, offsets, bits)
    assert torch.allclose(quantized_weight, expected_quantized_weight)

    scales_nonvec, offsets_nonvec = calculate_scales(
        weight, bits, symmetric=symmetric, clip_weights=True, vectorized=False
    )
    scales_vec, offsets_vec = calculate_scales(weight, bits, symmetric=symmetric, clip_weights=True, vectorized=True)
    quantized_weight_nonvec = quantize_weight_rtn(weight, scales_nonvec, offsets_nonvec, bits)
    quantized_weight_vec = quantize_weight_rtn(weight, scales_vec, offsets_vec, bits)
    assert torch.allclose(quantized_weight_vec, quantized_weight_nonvec)


@pytest.mark.quarot
@pytest.mark.gpu
@pytest.mark.parametrize(
    "bits, weight, expected_scales, expected_offsets",
    [
        (3, torch.tensor([[-4.0, 1.0, 2.0, 3.0]]), torch.tensor([[1.0]]), torch.tensor([4.0])),
        (3, torch.tensor([[-1.0, 4.0, 5.0, 6.0]]), torch.tensor([[1.0]]), torch.tensor([1.0])),
        (3, torch.tensor([[-0.5, 2.0, 2.5, 3.0]]), torch.tensor([[0.5]]), torch.tensor([1.0])),
        (4, torch.tensor([[-2.0, 1.2, 2.3, 13.0]]), torch.tensor([[1.0]]), torch.tensor([2.0])),
        (4, torch.tensor([[2.0, 5.2, 6.6, 7.5]]), torch.tensor([[0.5]]), torch.tensor([0.0])),
    ],
)
def test_calculate_scales_asymmetric(bits, weight, expected_scales, expected_offsets):
    symmetric = False
    scales, offsets = calculate_scales(weight, bits, symmetric=symmetric, clip_weights=False)
    assert torch.allclose(scales, expected_scales)
    assert torch.allclose(offsets, expected_offsets)

    scales_nonvec, offsets_nonvec = calculate_scales(
        weight, bits, symmetric=symmetric, clip_weights=True, vectorized=False
    )
    scales_vec, offsets_vec = calculate_scales(weight, bits, symmetric=symmetric, clip_weights=True, vectorized=True)
    assert torch.allclose(scales_vec, scales_nonvec)
    assert torch.allclose(offsets_nonvec, offsets_vec)


@pytest.mark.quarot
@pytest.mark.gpu
@pytest.mark.parametrize(
    "bits, weight, expected_quantized_weight",
    [
        (3, torch.tensor([[-4.0, 1.0, 2.0, 3.0]]), torch.tensor([[0.0, 5.0, 6.0, 7.0]])),
        (3, torch.tensor([[-1.0, 4.0, 5.0, 6.0]]), torch.tensor([[0.0, 5.0, 6.0, 7.0]])),
        (4, torch.tensor([[-2.0, 1.2, 2.3, 13.0]]), torch.tensor([[0.0, 3.0, 4.0, 15.0]])),
        (4, torch.tensor([[2.0, 5.2, 6.6, 7.5]]), torch.tensor([[4.0, 10.0, 13.0, 15.0]])),
    ],
)
def test_weight_rtn_asymmetric(bits, weight, expected_quantized_weight):
    symmetric = False
    scale, offset = calculate_scales(weight, bits, symmetric=symmetric, clip_weights=False)
    quantized_weight = quantize_weight_rtn(weight, scale, offset, bits)
    assert torch.allclose(quantized_weight, expected_quantized_weight)

    scales_nonvec, offsets_nonvec = calculate_scales(
        weight, bits, symmetric=symmetric, clip_weights=True, vectorized=False
    )
    scales_vec, offsets_vec = calculate_scales(weight, bits, symmetric=symmetric, clip_weights=True, vectorized=True)
    quantized_weight_nonvec = quantize_weight_rtn(weight, scales_nonvec, offsets_nonvec, bits)
    quantized_weight_vec = quantize_weight_rtn(weight, scales_vec, offsets_vec, bits)
    assert torch.allclose(quantized_weight_vec, quantized_weight_nonvec)


@pytest.mark.quarot
@pytest.mark.gpu
@pytest.mark.parametrize(
    "bits, groupsize, weight, expected_scales, expected_offsets",
    [
        (3, 3, torch.tensor([[-4.0, 2.0, 3.0]]), torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([4.0, 4.0, 4.0])),
        (
            3,
            3,
            torch.tensor([[-4.0, 2.0, 3.0], [-2.0, 1.1, 1.5]]),
            torch.tensor([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]),
            torch.tensor([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]),
        ),
        (
            4,
            2,
            torch.tensor([[-5.0, 2.5, 3.0, 15.0]]),
            torch.tensor([[0.5, 0.5, 1.0, 1.0]]),
            torch.tensor([10.0, 10.0, 0.0, 0.0]),
        ),
    ],
)
def test_calculate_scales_asymmetric_groupwise(bits, groupsize, weight, expected_scales, expected_offsets):
    scales, offsets = calculate_scales(weight, bits, symmetric=False, clip_weights=False, groupsize=groupsize)
    assert torch.allclose(scales, expected_scales)
    assert torch.allclose(offsets, expected_offsets)


@pytest.mark.quarot
@pytest.mark.gpu
@pytest.mark.parametrize(
    "bits, groupsize, weight, expected_quantized_weight",
    [
        (3, 3, torch.tensor([[-4.0, 2.0, 3.0]]), torch.tensor([[0.0, 6.0, 7.0]])),
        (3, 3, torch.tensor([[-4.0, 2.0, 3.0], [-2.0, 1.1, 1.5]]), torch.tensor([[0.0, 6.0, 7.0], [0.0, 6.0, 7.0]])),
        (4, 2, torch.tensor([[-5.0, 2.5, 3.0, 15.0]]), torch.tensor([[0.0, 15.0, 3.0, 15.0]])),
    ],
)
def test_weight_rtn_asymmetric_groupwise(bits, groupsize, weight, expected_quantized_weight):
    expected_quantized_weight = expected_quantized_weight.to(dtype=torch.float16)
    scale, offset = calculate_scales(weight, bits, symmetric=False, clip_weights=False, groupsize=groupsize)
    quantized_weight = quantize_weight_rtn(weight, scale, offset, bits)
    assert torch.allclose(quantized_weight, expected_quantized_weight)
