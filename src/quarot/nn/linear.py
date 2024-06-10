import torch

from ..quant_utils import PackedQuantizedTensor, dequantize


class QuarotFP16Linear(torch.nn.Module):
    """
    Linear module for emulating quantized weights and activations. All tensors are stored in FP16.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        offset: bool = False,
        group_size: int = None,
        dtype=torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # figure out group size
        if group_size is None:
            group_size = in_features  # tensor-level scaling, one effective group
        assert in_features % group_size == 0, "implied number of groups must be an integer!"
        self.group_size = group_size

        # register necessary buffers
        self.register_buffer('weight_scales', torch.ones((self.out_features, in_features // group_size), dtype=dtype))
        self.register_buffer('weight', torch.zeros((self.out_features, self.in_features), dtype=dtype))
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
        if offset:
            self.register_buffer('offset', torch.zeros((self.out_features, in_features // group_size), dtype=dtype))
        else:
            self.offset = None

    def forward(self, x: PackedQuantizedTensor) -> torch.tensor:
        # de-quantize the activations
        x, scales_x = x.quantized_x, x.scales_x
        x = x * scales_x

        # de-quantize the weights
        W = dequantize(self.weight, self.weight_scales, self.offset)

        #  run standard linear on dequantized weights and activations
        return torch.functional.F.linear(x, W, self.bias)

    @classmethod
    def like(cls: type, module: torch.nn.Linear, groupsize: int = None, offset: bool = False) -> 'QuarotFP16Linear':
        '''
        Generate a new QuarotFP16Linear module with the same shapes & bias flag as a given Linear module.
        '''
        return cls(
            module.in_features, module.out_features, bias=module.bias is not None, group_size=groupsize, offset=offset
        )
