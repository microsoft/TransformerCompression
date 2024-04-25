import torch

from quarot.quant_utils import PackedQuantizedTensor


class DummyActQuantizer(torch.nn.Module):
    '''Dummy quantizer: returns x unchanged and with scales of 1s.'''

    def forward(self, x: torch.Tensor) -> PackedQuantizedTensor:
        # take all the shape of x up to last dim
        shape = x.shape[:-1] + (1,)
        scales_x = torch.ones(shape, device=x.device)
        return PackedQuantizedTensor(x, scales_x)
