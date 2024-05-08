# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from quarot.rtn import calculate_scales, quantize_weight_rtn


@pytest.mark.quarot
@pytest.mark.gpu
@pytest.mark.parametrize(
    "bits, weight, expected_scales",
    [
        (3, torch.tensor([[-1.0, 1.0, 2.0, 3.0], [-0.5, 1.0, 2.0, -3.0]]), torch.tensor([[1.0], [1.0]])),
        (3, torch.tensor([[-1.0, 1.0, 1.0, 1.5], [-0.5, 1.0, 1.0, -1.5]]), torch.tensor([[0.5], [0.5]])),
        (4, torch.tensor([[-1.0, 1.0, 2.0, 3.5], [-0.5, 1.0, 2.0, -7.0]]), torch.tensor([[0.5], [1.0]])),
    ],
)
def test_calculate_scales(bits, weight, expected_scales):
    scales = calculate_scales(weight, bits, symmetric=True, perchannel=True, clip_weights=False)
    assert torch.allclose(scales, expected_scales)


@pytest.mark.quarot
@pytest.mark.gpu
@pytest.mark.parametrize(
    "bits, weight, expected_quantized_weight",
    [
        (3, torch.tensor([[1.1, -2.3, 3.4], [-1.5, 2.6, -2.9]]), torch.tensor([[1.0, -2.0, 3.0], [-2.0, 3.0, -3.0]])),
        (4, torch.tensor([[1.1, -2.3, 3.5], [-1.5, 2.6, -7.0]]), torch.tensor([[2.0, -5.0, 7.0], [-2.0, 3.0, -7.0]])),
    ],
)
def test_quantize_weight_rtn(bits, weight, expected_quantized_weight):
    scale = calculate_scales(weight, bits, symmetric=True, perchannel=True, clip_weights=False)
    quantized_weight = quantize_weight_rtn(weight, scale, bits, symmetric=True)
    assert torch.allclose(quantized_weight, expected_quantized_weight)
