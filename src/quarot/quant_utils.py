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
