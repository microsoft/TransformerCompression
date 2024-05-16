# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from quarot.gptq import quantize_weight_gptq


@pytest.mark.quarot
@pytest.mark.gpu
@pytest.mark.parametrize(
    "bits, weight, hessian, expected_quantized_weight, expected_scale",
    [
        (
            4,
            torch.tensor([[1.1, -2.3, 3.5], [-1.5, 2.6, -7.0]]),
            torch.eye(3, 3),
            torch.tensor([[2.0, -5.0, 7.0], [-2.0, 3.0, -7.0]]),
            torch.tensor([[0.5], [1.0]]),
        ),
    ],
)
def test_weight_gptq(bits, weight, hessian, expected_quantized_weight, expected_scale):
    quantized_weight, scale = quantize_weight_gptq(weight, hessian, bits, percdamp=0.0, clip_weights=False)
    assert torch.allclose(scale, expected_scale)
    assert torch.allclose(quantized_weight, expected_quantized_weight)
