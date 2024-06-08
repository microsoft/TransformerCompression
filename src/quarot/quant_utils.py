# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch


class PackedQuantizedTensor:
    '''Object to store a quantized tensor and its scale.'''

    def __init__(self, quantized_x: torch.Tensor, scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

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

    The shape of W_ints is (out_features, in_features)
    The shape of scale is (out_features, in_features // group_size)
    The shape of offset is (out_features, in_features // group_size) (optional)
    """
    if offset is None:
        offset = 0
    else:
        offset = torch.repeat_interleave(offset, W_ints.shape[-1] // offset.shape[-1], dim=1)
    W = (W_ints - offset) * torch.repeat_interleave(scale, W_ints.shape[-1] // scale.shape[-1], dim=1)
    return W
