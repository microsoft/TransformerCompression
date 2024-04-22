import torch

from quarot.quant_utils import PackedQuantizedTensor


class QuarotFP16Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales', torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', torch.zeros((self.out_features, self.in_features), dtype=dtype))
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):
        assert isinstance(x, PackedQuantizedTensor)  # Quantized input is given
        x, scales_x = x.quantized_x, x.scales_x

        # perform matmul with the emulated quantized tensors
        x = x @ self.weight.T

        # symmetric dequantization
        x = (x * scales_x) * self.weight_scales.T

        if self.bias is not None:
            x = x + self.bias

        return x

    @staticmethod
    def from_float(
        module: torch.nn.Linear,
        weight_scales=None,
    ) -> 'QuarotFP16Linear':
        '''
        Generate a new QuarotFP16Linear module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data
        quarot_fp16_module = QuarotFP16Linear(
            module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype
        ).to(weight_matrix.dtype)

        if weight_scales is not None:
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            quarot_fp16_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix / weight_scales.cuda()).round()
            quarot_fp16_module.weight.copy_(int_rounded_weight.cpu())

            if module.bias is not None:
                quarot_fp16_module.bias.copy_(module.bias)

        return quarot_fp16_module


# class Linear4bit(torch.nn.Module):
#     def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
#         '''
#         Symmetric 4-bit Linear Layer.
#         '''
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.register_buffer('weight_scales',
#                              torch.zeros((self.out_features, 1), requires_grad=False))
#         self.register_buffer('weight', (torch.randint(1, 7, (self.out_features, self.in_features // 2),
#                                                              # SubByte weight
#                                                              dtype=torch.uint8, requires_grad=False)))
#         if bias:
#             self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
#         else:
#             self.bias = None

#     def forward(self, x):
#         #if torch.cuda.current_device() != x.device:
#         #    torch.cuda.set_device(x.device)

#         assert type(x) == quarot.PackedQuantizedTensor #Quantized input is given
#         x, scales_x = x.quantized_x, x.scales_x
#         #shape_handler = ShapeHandler(quantized_x)
#         #quantized_x = shape_handler.flatten(quantized_x)
#         x = quarot.matmul(x, self.weight)
#         #out = shape_handler.unflatten(
#         #    quarot.sym_dequant(in_result, scales_x, self.weight_scales))
#         if self.bias is not None:
#             return quarot.sym_dequant(x, scales_x, self.weight_scales) + self.bias
#         else:
#             return quarot.sym_dequant(x, scales_x, self.weight_scales)
