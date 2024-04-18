# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math

import fast_hadamard_transform
import torch

from slicegpt import utils

from .hadamard_utils import matmul_hadU_cuda


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


class ActQuantizer(torch.nn.Module):

    '''
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16

    def free(self):
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(self, bits, groupsize=-1, sym=False, clip_ratio=1.0):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            utils.cleanup_memory()
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)


class ActQuantWrapper(torch.nn.Module):
    '''
    Wrapper for torch.nn.Linear blocks to emulate activation quantization. It emulates the quantization
    of the input and output activations of the linear layer. In addition, it can apply an online Hadamard
    transform (partial or full) to the input activations.
    '''

    def __init__(self, module: torch.nn.Linear):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.input_quantizer = ActQuantizer()
        self.output_quantizer = ActQuantizer()
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.input_quantizer.bits}'
        if self.input_quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.input_quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.output_quantizer.bits}'
        if self.output_quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.output_quantizer.sym else f' (Symmetric Per-Token)'

        return str_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:

            if self.fp32_had:  # Full Hadamard in FP32
                x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:  # Full Hadamard in FP16
                x = matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(
                    x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2),
                    scale=1 / math.sqrt(init_shape[-1] // self.had_dim),
                ).transpose(1, 2)
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)) / math.sqrt(
                    init_shape[-1] // self.had_dim
                )

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.input_quantizer.bits < 16:  # Quantize, if needed
            self.input_quantizer.find_params(x)
            x = self.input_quantizer(x).to(x_dtype)
            self.input_quantizer.free()

        # Forward pass through the linear layer
        x = self.module(x).to(x_dtype)

        if self.output_quantizer.bits < 16:  # Quantize the output, if needed
            self.output_quantizer.find_params(x)
            x = self.output_quantizer(x).to(x_dtype)
            self.output_quantizer.free()

        return x


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
        if self.bits == 16:
            return

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


def wrap_linears_with_actquantwrapper(module: torch.nn.Module, name: str = '') -> None:
    """
    Wraps the linear blocks in the model with ActQuantWrapper.
    """
    if isinstance(module, ActQuantWrapper):
        return

    linear_type = torch.nn.Linear

    for attribute in dir(module):
        module_attribute = getattr(module, attribute)
        if isinstance(module_attribute, linear_type):
            setattr(module, attribute, ActQuantWrapper(module_attribute))
        elif isinstance(module_attribute, (torch.nn.Sequential, torch.nn.ModuleList)):
            replaced = [
                ActQuantWrapper(child) if isinstance(child, linear_type) else child
                for child in module_attribute.children()
            ]
            setattr(module, attribute, type(module_attribute)(replaced))

    for child_name, child in module.named_children():
        wrap_linears_with_actquantwrapper(child, f'{name}.{child_name}' if name else child_name)


def get_quantizeable_modules(module: torch.nn.Module, name: str = '') -> dict[str, ActQuantWrapper]:
    """
    Get all the ActQuantWrapper modules in the model as a dictionary of name: module pairs.
    """
    res = {}
    if isinstance(module, ActQuantWrapper):
        res[name] = module
    else:
        for child_name, child in module.named_children():
            res.update(get_quantizeable_modules(child, name=f'{name}.{child_name}' if name else child_name))
    return res


def llama_down_proj_groupsize(model, groupsize):

    assert groupsize > 1, 'groupsize should be greater than 1!'

    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsiz = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert groupsize * group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert down_proj_groupsize * group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize
