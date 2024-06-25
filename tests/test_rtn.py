# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

torch.set_default_dtype(torch.float16)


@pytest.mark.quarot
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
    from quarot.rtn import calculate_scales, quantize_weight_rtn, dequantize

    symmetric = True
    
    # check that the scales are as expected when scaling by max(abs(weight))
    scales, offsets = calculate_scales(weight, bits, symmetric=symmetric, search=False, device='cpu')
    assert torch.allclose(scales, expected_scales)
    assert offsets is None
    
@pytest.mark.quarot
@pytest.mark.parametrize('leading_dims', [(), (2,), (2, 3)])
@pytest.mark.parametrize('quant_dim', [64, 256])
@pytest.mark.parametrize('groupsize', [16, 32])
@pytest.mark.parametrize('bits', [2, 3, 4, 8])
@pytest.mark.parametrize('symmetric', [True, False])
@pytest.mark.parametrize('seed', [0])
def test_search_improves_error(leading_dims, quant_dim, groupsize, bits, symmetric, seed):
    from quarot.rtn import calculate_scales, quantize_weight_rtn, dequantize
    # test that searching for the optimal scale gives the same or better result
    # as using the max scaling factor
    torch.manual_seed(seed)
    W = torch.randn(leading_dims + (quant_dim,), dtype=torch.float16)
    
    scales_nosearch, offsets_nosearch = calculate_scales(W, bits, symmetric=symmetric, search=False, device='cpu')
    scales_search, offsets_search = calculate_scales(W, bits, symmetric=symmetric, search=True, device='cpu')
    
    W_int_nosearch = quantize_weight_rtn(W, scales_nosearch, offsets_nosearch, bits)
    W_int_search = quantize_weight_rtn(W, scales_search, offsets_search, bits)
    
    W_recon_nosearch = dequantize(W_int_nosearch, scales_nosearch, offsets_nosearch)
    W_recon_search = dequantize(W_int_search, scales_search, offsets_search)
    
    err_nosearch = torch.norm(W - W_recon_nosearch, p=2.4) / torch.norm(W, p=2.4)
    err_search = torch.norm(W - W_recon_search, p=2.4) / torch.norm(W, p=2.4)
    
    assert err_search <= err_nosearch
    
                                    


@pytest.mark.quarot
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
    from quarot.rtn import calculate_scales, quantize_weight_rtn

    symmetric = True
    scales, offsets = calculate_scales(weight, bits, symmetric=symmetric, search=False, device='cpu')
    quantized_weight = quantize_weight_rtn(weight, scales, offsets, bits)
    assert torch.allclose(quantized_weight, expected_quantized_weight)


@pytest.mark.quarot
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
    from quarot.rtn import calculate_scales

    symmetric = False
    scales, offsets = calculate_scales(weight, bits, symmetric=symmetric, search=False, device='cpu')
    assert torch.allclose(scales, expected_scales)
    assert torch.allclose(offsets, expected_offsets)


@pytest.mark.quarot
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
    from quarot.rtn import calculate_scales, quantize_weight_rtn

    symmetric = False
    scale, offset = calculate_scales(weight, bits, symmetric=symmetric, search=False, device='cpu')
    quantized_weight = quantize_weight_rtn(weight, scale, offset, bits)
    assert torch.allclose(quantized_weight, expected_quantized_weight)


@pytest.mark.quarot
@pytest.mark.parametrize(
    "bits, groupsize, weight, expected_scales, expected_offsets",
    [
        (3, 3, torch.tensor([[-4.0, 2.0, 3.0]]), torch.tensor([[1.0]]), torch.tensor([4.0])),
        (
            3,
            3,
            torch.tensor([[-4.0, 2.0, 3.0], [-2.0, 1.1, 1.5]]),
            torch.tensor([[1.0], [0.5]]),
            torch.tensor([[4.0], [4.0]]),
        ),
        (
            4,
            2,
            torch.tensor([[-5.0, 2.5, 3.0, 15.0]]),
            torch.tensor([[0.5, 1.0]]),
            torch.tensor([10.0, 0.0]),
        ),
    ],
)
def test_calculate_scales_asymmetric_groupwise(bits, groupsize, weight, expected_scales, expected_offsets):
    from quarot.rtn import calculate_scales

    scales, offsets = calculate_scales(weight, bits, symmetric=False, search=False, groupsize=groupsize, device='cpu')
    assert torch.allclose(scales, expected_scales)
    assert torch.allclose(offsets, expected_offsets)


@pytest.mark.quarot
@pytest.mark.parametrize(
    "bits, groupsize, weight, expected_quantized_weight",
    [
        (3, 3, torch.tensor([[-4.0, 2.0, 3.0]]), torch.tensor([[0.0, 6.0, 7.0]])),
        (3, 3, torch.tensor([[-4.0, 2.0, 3.0], [-2.0, 1.1, 1.5]]), torch.tensor([[0.0, 6.0, 7.0], [0.0, 6.0, 7.0]])),
        (4, 2, torch.tensor([[-5.0, 2.5, 3.0, 15.0]]), torch.tensor([[0.0, 15.0, 3.0, 15.0]])),
    ],
)
def test_weight_rtn_asymmetric_groupwise(bits, groupsize, weight, expected_quantized_weight):
    from quarot.rtn import calculate_scales, quantize_weight_rtn

    expected_quantized_weight = expected_quantized_weight.to(dtype=torch.float16)
    scale, offset = calculate_scales(weight, bits, symmetric=False, search=False, groupsize=groupsize, device='cpu')
    quantized_weight = quantize_weight_rtn(weight, scale, offset, bits)
    assert torch.allclose(quantized_weight, expected_quantized_weight)
