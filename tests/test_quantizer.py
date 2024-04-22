# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from quarot.nn.linear import QuarotFP16Linear
from quarot.nn.quantizer import FP16ActQuantizer


def test_quantizer():
    """Sanity checks for emulated quantization."""

    quantizer = FP16ActQuantizer()

    x = torch.tensor([1.02, 0.056, -3.2, 4.999, 5.0])

    packed_tensor = quantizer(x)

    quantized_x = packed_tensor.quantized_x
    scales_x = packed_tensor.scales_x

    dequantized_x = quantized_x * scales_x

    assert torch.allclose(dequantized_x, x, atol=1e-3, rtol=1e-3)


def test_fp16linear():

    # x = torch.randn(4, 4)
    linear = torch.nn.Linear(4, 4)
    x = torch.tensor([1.02, 2.056, -3.2, 4.999])
    y = linear(x)

    quarot_linear = QuarotFP16Linear.like(linear)
    quarot_linear.weight = linear.weight
    quantizer = FP16ActQuantizer()
    quantized_x = quantizer(x)
    y_hat = quarot_linear(quantized_x)

    assert torch.allclose(y, y_hat, atol=1e-3, rtol=1e-3)
