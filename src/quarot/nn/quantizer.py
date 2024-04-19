import torch

from quarot.quant_utils import PackedQuantizedTensor, get_minq_maxq

# class Quantizer(torch.nn.Module):
#     def __init__(self, input_clip_ratio=1.0):
#         super().__init__()
#         self.input_clip_ratio = input_clip_ratio

#     def forward(self, x):
#         scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1)/7).to(torch.float16) * self.input_clip_ratio
#         quantized_x = quarot.sym_quant(x, scales_x)
#         packed_tensor = quarot.PackedQuantizedTensor(quantized_x, scales_x)
#         return packed_tensor


class FP16ActQuantizer(torch.nn.Module):
    # TODO: sort out device management here
    def __init__(self, a_bits: int = 16, input_clip_ratio=1.0):
        super().__init__()
        self.min_q, self.max_q = get_minq_maxq(a_bits, True)
        self.input_clip_ratio = input_clip_ratio

    def forward(self, x: torch.Tensor) -> PackedQuantizedTensor:
        scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(-1) / self.max_q.to(x.device)).to(
            torch.float16
        ) * self.input_clip_ratio

        quantized_x = torch.round(x / scales_x)
        quantized_x = torch.clip(quantized_x, self.min_q.to(x.device), self.max_q.to(x.device))
        return PackedQuantizedTensor(quantized_x, scales_x)
