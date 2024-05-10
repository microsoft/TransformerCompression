# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from quarot.rtn import calculate_scales_asymmetric, calculate_scales_symmetric, quantize_weight_rtn


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
    scales = calculate_scales_symmetric(weight, bits, perchannel=True, clip_weights=False)
    assert torch.allclose(scales, expected_scales)


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
    scale = calculate_scales_symmetric(weight, bits, perchannel=True, clip_weights=False)
    quantized_weight = quantize_weight_rtn(weight, scale, None, bits, symmetric=True)
    assert torch.allclose(quantized_weight, expected_quantized_weight)


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
    scales, offsets = calculate_scales_asymmetric(weight, bits, perchannel=True)
    assert torch.allclose(scales, expected_scales)
    assert torch.allclose(offsets, expected_offsets)


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
    scale, offset = calculate_scales_asymmetric(weight, bits, perchannel=True)
    quantized_weight = quantize_weight_rtn(weight, scale, offset, bits, symmetric=False)
    assert torch.allclose(quantized_weight, expected_quantized_weight)
