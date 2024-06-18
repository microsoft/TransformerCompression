import pytest
import torch

from quarot import rtn
from quarot.quant_utils import from_int4, to_int4, PackedQuantizedTensor
from quarot.nn import QuarotFP16Linear


def rand_posdef(n):
    A = torch.randn(n, n)
    return A @ A.T

@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("quant_dim", [16, 32])
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("groupsize", [4, 8])
@pytest.mark.parametrize("out_features", [37, 14])
def test_int4_packing(symmetric, quant_dim, bits, groupsize, out_features):
    W = torch.randn(out_features, quant_dim)
    scales, offset = rtn.calculate_scales(W, bits=bits, symmetric=symmetric, search=True, groupsize=groupsize, device='cpu')
    W_int = rtn.quantize_weight_rtn(W, scales, offset, bits=bits)
    W_int4 = to_int4(W_int, signed=symmetric)
    assert W_int4.shape[-1] == quant_dim // 2
    assert W_int4.dtype == torch.uint8
    assert torch.all(W_int == from_int4(W_int4, signed=symmetric))

@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("quant_dim", [1024, 4096])
@pytest.mark.parametrize("bits", [3, 4, 8])
@pytest.mark.parametrize("groupsize", [16, 32, 64, None])
@pytest.mark.parametrize("out_features", [37, 14])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("seed", [0, 42, 123, 7])
def test_compress(symmetric, quant_dim, bits, groupsize, out_features, dtype, seed):
    # set random seed
    torch.manual_seed(seed)
    
    # create a module with quantized weights 
    m = QuarotFP16Linear(quant_dim, out_features, bits=bits, offset= not symmetric, group_size=groupsize, dtype=dtype)
    W = torch.randn(out_features, quant_dim, dtype=dtype)
    scales, offset = rtn.calculate_scales(W, bits=bits, symmetric=symmetric, search=True, groupsize=groupsize, device='cpu')
    W_int = rtn.quantize_weight_rtn(W, scales, offset, bits=bits)
    rtn.set_tensors(m, W_int, scales, offset)
        
    x = PackedQuantizedTensor(torch.randn(3, quant_dim, dtype=dtype), torch.randn(3, scales.shape[-1], dtype=dtype))
    y1 = m(x)
    m.compress()
    y2 = m(x)
    assert torch.allclose(y1, y2)
    
    
