import torch

from quarot.quant_utils import PackedQuantizedTensor


class QuarotFP16Linear(torch.nn.Module):
    """
    Linear module for emulating quantized weights and activations. All tensors are stored in FP16.
    """

    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales', torch.ones((self.out_features, 1), dtype=dtype))
        self.register_buffer('weight', torch.zeros((self.out_features, self.in_features), dtype=dtype))
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: PackedQuantizedTensor) -> torch.tensor:
        x, scales_x = x.quantized_x, x.scales_x

        # perform matmul with the emulated quantized tensors
        x = x @ self.weight.T

        # symmetric dequantization
        x = (x * scales_x) * self.weight_scales.T

        if self.bias is not None:
            x = x + self.bias

        return x

    @classmethod
    def like(
        cls: type,
        module: torch.nn.Linear,
    ) -> 'QuarotFP16Linear':
        '''
        Generate a new QuarotFP16Linear module with the same shapes & bias flag as a given Linear module.
        '''
        return cls(module.in_features, module.out_features, bias=module.bias is not None)
