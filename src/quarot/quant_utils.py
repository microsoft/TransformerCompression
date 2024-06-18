# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch


class PackedQuantizedTensor:
    '''Object to store a quantized tensor and its scale.'''

    def __init__(self, quantized_x: torch.Tensor, scales_x: torch.Tensor, offset: torch.Tensor | None = None):
        self.quantized_x = quantized_x
        self.scales_x = scales_x
        self.offset = offset

    def size(self) -> torch.Size:
        return self.quantized_x.size()

    @property
    def device(self) -> torch.device:
        return self.quantized_x.device

    @property
    def dtype(self) -> torch.dtype:
        return self.quantized_x.dtype


def dequantize(W_ints: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor | None):
    """
    Reconstruct the (approximate) weight tensor from the quantized weights, scales, and offsets.

    Here, repeat_interleave is used apply the scale and offset accross each group.

    The shape of W_ints is (..., out_features, in_features)
    The shape of scale is (..., out_features, in_features // group_size)
    The shape of offset is (..., out_features, in_features // group_size) (optional)
    """
    *shape, in_features = W_ints.shape
    num_groups = scale.shape[-1]
    groupsize = in_features // num_groups
    
    W_ints = W_ints.reshape(*shape, num_groups, groupsize)
    if offset is None:
        W = W_ints * scale[..., None] # broadcast scale across groupsize
    else:
        W = (W_ints - offset[..., None]) * scale[..., None]
    W = W.reshape(*shape, in_features)
    
    return W


def calculate_min_max_int(bits: int, symmetric: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the maximum representable integer value given the number of bits and the quantization scheme.
    """
    if symmetric:
        max_int = torch.tensor(2 ** (bits - 1) - 1)
        min_int = -(max_int + 1)
    else:
        max_int = torch.tensor(2**bits - 1)
        min_int = torch.zeros(1)
    return min_int, max_int


def to_int4(w: torch.Tensor, signed: bool = False):
    """
    Convert a tensor of ints to a tensor of "int4" values, where each value is represented by 4 bits.
    
    Because pytroch has no native int4 type, we use uint8 to store the 4-bit values, which means that
    the dimension of the resulting tensor is half the size of the input tensor.
    
    The inputs must be within the bounds of the int4 representation. For unsigned int4, the range is [0, 15].
    For signed int4, the range is [-8, 7].
    """
    assert w.shape[-1] % 2 == 0, 'last dim must be even'
    minint, maxint = calculate_min_max_int(4, signed)
    assert torch.all(w <= maxint) and torch.all(w >= minint), 'input must be within bounds'
    w = w.to(torch.uint8)
    even_cols = w[..., ::2]
    odd_cols = w[..., 1::2]
    if signed:
        # Mask to keep only lower 4 bits and handle negative values correctly
        even_cols = even_cols & 0xF
        odd_cols = odd_cols & 0xF

    return even_cols | (odd_cols << 4)

def from_int4(w, signed=False, dtype=torch.float32):
    """
    Convert a tensor of "int4" values to a tensor of ints.
    """
    # Extract lower and upper 4 bits
    lower = w & 0xF
    upper = (w >> 4) & 0xF
    if signed:
        # Convert from unsigned to signed int4
        lower = (lower ^ 0x8) - 0x8
        upper = (upper ^ 0x8) - 0x8
        
    # Concatenate (interweave) the unpacked tensors
    out = torch.zeros(w.shape[:-1] + (w.shape[-1] * 2,), dtype=torch.uint8, device=w.device)
    out[..., ::2] = lower
    out[..., 1::2] = upper
    return out.to(torch.int8).to(dtype)