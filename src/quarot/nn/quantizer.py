import torch

from quarot.quant_utils import PackedQuantizedTensor
from quarot.rtn import calculate_scales, quantize_weight_rtn


class DummyActQuantizer(torch.nn.Module):
    '''Dummy quantizer: returns x unchanged and with scales of 1s.'''

    def forward(self, x: torch.Tensor) -> PackedQuantizedTensor:
        # take all the shape of x up to last dim
        shape = x.shape[:-1] + (1,)
        scales_x = torch.ones(shape, device=x.device, dtype=x.dtype)
        return PackedQuantizedTensor(x, scales_x)


class ActQuantizer(torch.nn.Module):
    '''Quantizer for activations. Applies round-to-nearest quantization tensor-wise across the seqlen and hidden_dim dimensions.'''

    def __init__(self, bits: int, symmetric: bool = True, clip_ratio: float = 1.0) -> None:
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.clip_ratio = clip_ratio

    def forward(self, x: torch.Tensor) -> PackedQuantizedTensor:
        x_scales = (
            calculate_scales(x, self.bits, symmetric=self.symmetric, perchannel=True, clip_weights=False)
            * self.clip_ratio
        )
        quantized_x = quantize_weight_rtn(x, x_scales, self.bits, self.symmetric)
        return PackedQuantizedTensor(quantized_x, x_scales)
