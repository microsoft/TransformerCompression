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
        self,
        bits: int,
        symmetric: bool = True,
        clip_ratio: float | None = None,
        groupsize: int | None = None,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.clip_ratio = clip_ratio
        self.groupsize = groupsize
        self.symmetric = symmetric

    def forward(self, x: torch.Tensor) -> PackedQuantizedTensor:
        scale, offset = calculate_scales(
            x,
            self.bits,
            symmetric=self.symmetric,
            search=False,
            clip_ratio=self.clip_ratio,
            groupsize=self.groupsize,
        )
        x_int = quantize_weight_rtn(weight=x, scale=scale, offset=offset, bits=self.bits)
        return PackedQuantizedTensor(x_int, scale, offset)


class KVQuantizerDequantizer(torch.nn.Module):
    '''Quantizer for quantizing and immediately dequantizing K and V. Applies round-to-nearest quantization head-wise.'''

    def __init__(
        self,
        bits: int,
        symmetric: bool = False,
        clip_ratio: float | None = None,
        groupsize: int | None = None,
    ) -> None:
        super().__init__()
        self.bits = bits
        assert not symmetric, "KV quantization should be asymmetric."
        self.clip_ratio = clip_ratio
        self.groupsize = groupsize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scales, x_offsets = calculate_scales(
            x,
            self.bits,
            symmetric=False,
            search=False,
            clip_ratio=self.clip_ratio,
            groupsize=self.groupsize,
        )
        quantized_x = quantize_weight_rtn(x, x_scales, x_offsets, self.bits)
        dequantized_x = dequantize(quantized_x, x_scales, x_offsets)
        return dequantized_x
