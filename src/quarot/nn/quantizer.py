import torch

from ..quant_utils import PackedQuantizedTensor, dequantize
from ..rtn import calculate_scales, quantize_weight_rtn


class DummyActQuantizer(torch.nn.Module):
    '''Dummy quantizer: returns x unchanged and with scales of 1s.'''

    def forward(self, x: torch.Tensor) -> PackedQuantizedTensor:
        # take all the shape of x up to last dim
        shape = x.shape[:-1] + (1,)
        scales_x = torch.ones(shape, device=x.device, dtype=x.dtype)
        return PackedQuantizedTensor(x, scales_x)


class ActQuantizer(torch.nn.Module):
    '''Quantizer for activations. Applies round-to-nearest quantization tensor-wise across the seqlen and hidden_dim dimensions.'''

    def __init__(
        self, bits: int, symmetric: bool = True, clip_ratio: float = 1.0, groupsize: int | None = None
    ) -> None:
        super().__init__()
        self.bits = bits
        assert symmetric, "Activation quantization should be symmetric."
        self.clip_ratio = clip_ratio
        self.groupsize = groupsize

    def forward(self, x: torch.Tensor) -> PackedQuantizedTensor:
        x_scales, offset = calculate_scales(
            x, self.bits, symmetric=True, clip_weights=False, clip_ratio=self.clip_ratio, groupsize=self.groupsize
        )
        quantized_x = quantize_weight_rtn(
            weight=x, scale=x_scales, offset=offset, bits=self.bits
        )  # note that offset is always none becasue symmteric is True.
        return PackedQuantizedTensor(quantized_x, x_scales)


class KVQuantizerDequantizer(torch.nn.Module):
    '''Quantizer for quantizing and immediately dequantizing K and V. Applies round-to-nearest quantization head-wise.'''

    def __init__(
        self, bits: int, symmetric: bool = False, clip_ratio: float = 1.0, groupsize: int | None = None
    ) -> None:
        super().__init__()
        self.bits = bits
        assert not symmetric, "KV quantization should be asymmetric."
        self.clip_ratio = clip_ratio
        self.groupsize = groupsize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scales, x_offsets = calculate_scales(
            x, self.bits, symmetric=False, clip_weights=False, clip_ratio=self.clip_ratio, groupsize=self.groupsize
        )
        quantized_x = quantize_weight_rtn(x, x_scales, x_offsets, self.bits)
        dequantized_x = dequantize(quantized_x, x_scales, x_offsets)
        return dequantized_x
