# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from quarot.nn.quantizer import ActQuantizer, DummyActQuantizer


@pytest.mark.quarot
def test_dummy_quantizer():
    quantizer = DummyActQuantizer()

    act = torch.tensor([1.02, 0.056, -3.2, 4.999, 5.0])

    packed_quantized_act = quantizer(act)
    quantized_x, scales_x = packed_quantized_act.quantized_x, packed_quantized_act.scales_x

    # Check quantization error
    dequantized_x = quantized_x * scales_x
    assert torch.allclose(act, dequantized_x)


@pytest.mark.quarot
@pytest.mark.gpu
def test_act_quantizer():
    quantizer = ActQuantizer(bits=8)

    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    act = torch.randn(batch_size, seq_len, hidden_dim)

    packed_quantized_act = quantizer(act)
    quantized_act, act_scales = packed_quantized_act.quantized_x, packed_quantized_act.scales_x

    # Check shapes
    assert quantized_act.shape == (batch_size, seq_len, hidden_dim)
    assert act_scales.shape == (batch_size, seq_len, 1)

    # Check quantization error
    dequantized_act = quantized_act * act_scales
    assert torch.allclose(act, dequantized_act, rtol=1e-2, atol=1e-2)
