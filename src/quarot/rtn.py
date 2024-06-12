# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from tqdm import tqdm

from .nn.linear import QuarotFP16Linear
from .quant_utils import dequantize


def calculate_min_max_int(bits: int, symmetric: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the maximum representable integer value given the number of bits and the quantization scheme.
    """
    if symmetric:
        max_int = torch.tensor(2 ** (bits - 1) - 1)
        min_int = -(max_int + 1)
    else:
        max_int = torch.tensor(2**bits - 1)
        min_int = torch.zeros(1)
    return min_int, max_int


def calculate_min_max_weight(weight: torch.Tensor, symmetric: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the minimum and maximum weights in a weight tensor.
    If symmetric, the max weight is
    the larger of max weight or the absolute value of min weight.
    """
    max_weight = torch.max(weight, torch.zeros_like(weight)).max(dim=-1, keepdim=True).values
    min_weight = torch.min(weight, torch.zeros_like(weight)).min(dim=-1, keepdim=True).values

    if symmetric:
        max_weight = torch.maximum(max_weight, torch.abs(min_weight)).clamp(min=1e-5)

    return min_weight, max_weight


def calculate_scales_search(weight, bits, symmetric):
    """
    Perform a grid search to find the best weight scales according to quantization error
    We loop over a series of candiate scales and offsets, quantize the weights and reconstruct
    them, then choose the scale and offset that minimizes the quantization error.
    """
    max_shrink_factor = 0.80
    n_steps = int(100 * (max_shrink_factor)) + 1
    error_norm = 2.4
    shrink_factors = torch.linspace(1.0, 1 - max_shrink_factor, n_steps).to(weight.device)

    best_quantization_error = torch.full(
        weight.shape[:-1], float("inf"), device=weight.device, dtype=weight.dtype
    )
    
    min_weight, max_weight = calculate_min_max_weight(weight, symmetric=symmetric)
    _, max_int = calculate_min_max_int(bits, symmetric=symmetric)
    
    # initialize scale and offset
    scale = max_weight / max_int
    offset = None if symmetric else torch.round(-min_weight / scale)
    
    for shrink_factor in shrink_factors:
        # find the scale & offset for this shrink factor, quantize and reconstruct weights
        candidate_max_weight = shrink_factor * max_weight
        if symmetric:
            candidate_scale = candidate_max_weight / max_int
            candidate_quantized_weight = quantize_weight_rtn(weight, candidate_scale, offset=None, bits=bits)
            reconstructed_weight = candidate_quantized_weight * candidate_scale
        else:
            candidate_min_weight = shrink_factor * min_weight
            candidate_scale = (candidate_max_weight - candidate_min_weight) / max_int
            candidate_offset = torch.round(-candidate_min_weight / candidate_scale)
            candidate_quantized_weight = quantize_weight_rtn(weight, candidate_scale, candidate_offset, bits=bits)
            reconstructed_weight = (candidate_quantized_weight - candidate_offset) * candidate_scale

        # Compute quantization error and find the best scale (and offset if asymmetric)
        quantization_error = torch.sum(torch.abs(reconstructed_weight - weight).pow_(error_norm), -1)
        if torch.any(quantization_error < best_quantization_error):
            improved_idx = torch.where(quantization_error < best_quantization_error)
            scale[improved_idx] = candidate_scale[improved_idx]
            if not symmetric:
                offset[improved_idx] = candidate_offset[improved_idx]

            best_quantization_error[improved_idx] = quantization_error[improved_idx]
            
    return scale, offset


def calculate_scales_quantile(weight, bits, symmetric, quantile):
    """
    Calculate the scales (and offsets if asymmetric) for quantizing a weight tensor to INT<bits>
    Scale by a factor of the quantile of the data."""
    _, max_int = calculate_min_max_int(bits, symmetric=symmetric)
    if symmetric:
        scale = torch.quantile(torch.abs(weight), quantile, dim=-1, keepdim=True) / max_int
        offset = None
    else:
        upper = torch.quantile(weight, quantile, dim=-1, keepdim=True)
        lower = torch.quantile(weight, 1 - quantile, dim=-1, keepdim=True)
        scale = (upper - lower) / max_int
        offset = torch.round(-lower / scale)
    return scale, offset


def calculate_scales_clip(weight, bits, symmetric, clip_ratio):
    """
    Calculate the scales (and offsets if asymmetric) for quantizing a weight tensor to INT<bits>
    Scale by a factor of clip_ratio times the max weight.
    """
    
    min_weight, max_weight = calculate_min_max_weight(weight, symmetric=symmetric)
    min_weight *= clip_ratio
    max_weight *= clip_ratio

    _, max_int = calculate_min_max_int(bits, symmetric=symmetric)
    max_int = max_int.to(device=weight.device)
    if symmetric:
        scale = max_weight / max_int
        offset = None
    else:
        scale = (max_weight - min_weight) / max_int
        offset = torch.round(-min_weight / scale)
    return scale, offset
    

def calculate_scales(
    weight: torch.Tensor,
    bits: int,
    symmetric: bool,
    search: bool = False,
    quantile: float|None = None,
    clip_ratio: float=1.0,
    groupsize: int|None = None,
    device='cuda',
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Calculate the scales (and offsets if asymmetric) for quantizing a weight tensor to INT<bits>
    
    If search is true, we run a grid search for the best scalubg factor
    otherwise, we use the quantile of the data to scale
    if quantile is None we use a 'clip ratio' time the max weight.
    
    Quantile is ignored if doing a search. Clip ratio is ignored if quantile is not None.
    
    This method account for grouping: we reshape the data into num_groups x group_size, and then squeeze out at the end.
    """
    orig_device = weight.device
    weight = weight.to(device=device)

    if groupsize:
        # reshape the last dimension into num_groups x group_size
        new_shape = list(weight.shape[:-1]) + [weight.shape[-1] // groupsize, groupsize]
        weight = weight.reshape(new_shape)

    # Calculate scales and offsets
    if search:
        scale, offset = calculate_scales_search(weight, bits, symmetric)
    elif quantile is not None:
        scale, offset = calculate_scales_quantile(weight, bits, symmetric, quantile)
    else:
        scale, offset = calculate_scales_clip(weight, bits, symmetric, clip_ratio)

    if groupsize:
        scale = scale.squeeze(-1)
        if offset is not None:
            offset = offset.squeeze(-1)

    weight = weight.to(orig_device)
    scale = scale.to(orig_device)
    offset = offset.to(orig_device) if offset is not None else None

    return scale, offset


def quantize_weight_rtn(weight: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor | None, bits: int):
    """
    Quantize a weight tensor to INT<bits> using the given scale and offset.
    """
    scale = torch.repeat_interleave(scale, weight.shape[-1] // scale.shape[-1], dim=-1)

    if offset is None:
        _offset = 0
    else:
        _offset = torch.repeat_interleave(offset, weight.shape[-1] // offset.shape[-1], dim=-1)

    min_int, max_int = calculate_min_max_int(bits, symmetric=offset is None)
    min_int = min_int.to(weight.device, weight.dtype)
    max_int = max_int.to(weight.device, weight.dtype)
    weight_ints = torch.round(weight / scale) + _offset

    quantized_weight = torch.clamp(weight_ints, min_int, max_int)
    return quantized_weight


def quantize_module_rtn(
    module: QuarotFP16Linear,
    bits: int,
    symmetric: bool = True,
    clip_weights: bool = True,
    vectorized: bool = True,
    groupsize: int | None = None,
) -> None:
    """
    Quantize the weights of a QuarotFP16Linear module to INT<bits> using the Round-to-Nearest scheme, storing the weights in torch.float16. The weight
    scales are stored in the module's weight_scales buffer, and are also stored in torch.float16.
    """
    weight = module.weight

    scale, offset = calculate_scales(weight, bits, symmetric, clip_weights, vectorized=vectorized, groupsize=groupsize)
    quantized_weight = quantize_weight_rtn(weight, scale, offset, bits)

    if isinstance(module, QuarotFP16Linear):
        module.weight.data = quantized_weight
        module.weight_scales.data = scale
        if not symmetric:
            module.offset.data = offset
    elif isinstance(module, torch.nn.Linear):
        module.weight.data = dequantize(quantized_weight, scale, offset)
    else:
        raise NotImplementedError


def quantize_model_rtn(
    model,
    bits: int,
    symmetric: bool = True,
    clip_weights: bool = True,
    vectorized: bool = True,
    groupsize: int | None = None,
) -> None:
    """
    Quantize the weights (in QuarotFP16Linear modules) of a QuaRot model using the Round-to-Nearest scheme.

    Args:
        model: the model to quantize
        bits: the number of bits to quantize the weights to
        symmetric: whether to use symmetric quantization
        clip_weights: whether to perform a search for the best clip ratio for weight clipping
        vectorized: whether to use a vectorized implementation for weight clipping
        groupsize: the groupsize for quantization. If None, quantize each channel in full.
    """
    for layer in tqdm(model.model.layers, unit="layer", desc="Quantizing layer"):
        for _, module in layer.named_modules():
            if isinstance(module, QuarotFP16Linear):
                quantize_module_rtn(module, bits, symmetric, clip_weights, vectorized, groupsize)
