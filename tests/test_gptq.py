# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch


@pytest.mark.quarot
@pytest.mark.gpu
@pytest.mark.parametrize(
    "weight",
    [torch.tensor([[1.1, -2.3, 3.5], [-1.5, 2.6, -7.0]])],
)
@pytest.mark.parametrize("bits", [3, 4, 8])
@pytest.mark.parametrize("symmetric", [True, False])
def test_gptq_eye_hessian(weight, bits, symmetric):
    # imports which require a GPU build machine
    from quarot.gptq import quantize_weight_gptq
    from quarot.rtn import calculate_scales, quantize_weight_rtn

    hessian = torch.eye(weight.shape[1], dtype=torch.float32)

    gptq_quantized_weight, gptq_scale, gptq_offset = quantize_weight_gptq(
        weight, hessian, bits, symmetric=symmetric, clip_weights=False
    )

    rtn_scale, rtn_offset = calculate_scales(weight, bits, symmetric=symmetric, search=False)
    rtn_quantized_weight = quantize_weight_rtn(weight, rtn_scale, rtn_offset, bits)

    assert torch.allclose(gptq_quantized_weight, rtn_quantized_weight)
    assert torch.allclose(gptq_scale, rtn_scale)
    if symmetric:
        assert gptq_offset is None
        assert rtn_offset is None
    else:
        assert torch.allclose(gptq_offset, rtn_offset)
