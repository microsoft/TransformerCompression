# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from quarot.nn.quantizer import DummyActQuantizer


def test_dummy_quantizer():
    """Sanity checks for emulated quantization."""

    quantizer = DummyActQuantizer()

    x = torch.tensor([1.02, 0.056, -3.2, 4.999, 5.0])

    packed_tensor = quantizer(x)

    quantized_x = packed_tensor.quantized_x
    scales_x = packed_tensor.scales_x

    dequantized_x = quantized_x * scales_x

    assert torch.allclose(dequantized_x, x, atol=1e-3, rtol=1e-3)
