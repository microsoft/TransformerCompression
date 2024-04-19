# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(
        self,
        shape: int = 1,
        bits: int = 16,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        max_shrink: float = 0.8,
    ) -> None:
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

        self.bits = bits
        self.per_channel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.max_shrink = max_shrink
        if sym:
            self.max_quantized_value = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.max_quantized_value = torch.tensor(2**bits - 1)

    def calc_params(self, w: torch.tensor) -> None:
        '''
        Calculates the quantization parameters max_quantized_value, scale and zero from the weight tensor w.
        '''
        # if self.bits == 16:
        #     return

        dev = w.device
        self.max_quantized_value = self.max_quantized_value.to(dev)

        shape = w.shape
        if self.per_channel:
            w = w.flatten(1)
        else:
            w = w.flatten().unsqueeze(0)

        tmp = torch.zeros(w.shape[0], device=dev)
        w_min = torch.minimum(w.min(1)[0], tmp)
        w_max = torch.maximum(w.max(1)[0], tmp)

        if self.sym:
            w_max = torch.maximum(torch.abs(w_min), w_max).clamp(min=1e-5)
            self.scale = w_max / self.max_quantized_value
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (w_min == 0) & (w_max == 0)
            w_min[tmp] = -1
            w_max[tmp] = +1
            self.scale = (w_max - w_min).clamp(min=1e-5) / self.max_quantized_value
            self.zero = torch.round(-w_min / self.scale)

        if self.mse:
            best = torch.full([w.shape[0]], float('inf'), device=dev)
            for i in range(int(self.max_shrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * w_min
                xmax1 = p * w_max

                if self.sym:
                    scale1 = xmax1 / self.max_quantized_value
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(w, scale1.unsqueeze(1), self.max_quantized_value)
                else:
                    scale1 = (xmax1 - xmin1) / self.max_quantized_value
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(w, scale1.unsqueeze(1), zero1.unsqueeze(1), self.max_quantized_value)

                q -= w
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.per_channel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.max_quantized_value).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.max_quantized_value).to(x_dtype)
        return x

    def enabled(self):
        return self.max_quantized_value > 0

    def ready(self):
        return torch.all(self.scale != 0)


def get_linear_modules(module: torch.nn.Module, name: str = '') -> dict[str, torch.nn.Linear]:
    """
    Get all the torch.nn.Linear modules in the model as a dictionary of name: module pairs.
    """
    res = {}
    if isinstance(module, torch.nn.Linear):
        res[name] = module
    else:
        for child_name, child in module.named_children():
            res.update(get_linear_modules(child, name=f'{name}.{child_name}' if name else child_name))
    return res


class PackedQuantizedTensor:
    def __init__(self, quantized_x: torch.Tensor, scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

    def size(self):
        return self.quantized_x.size()

    @property
    def device(self):
        return self.quantized_x.device

    @property
    def dtype(self):
        return self.quantized_x.dtype
