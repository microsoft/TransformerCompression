# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from tqdm import tqdm

from .nn.linear import QuarotFP16Linear
from .quant_utils import dequantize, calculate_min_max_int


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


def calculate_scales_search(weight, bits, symmetric, search_range=None, num_grid=8, depth=3):
    """
    Compute the optimal scales (and offsets) for quantizing a weight tensor. 
    
    Args:
        weight: torch.Tensor, shape [..., quant_dim]
        bits: int, number of bits to quantize to
        symmetric: bool, whether to quantize symmetrically
        search_range: tuple of two tensors, the range to search in. If None, the full range is searched
        num_grid: int, number of grid points to search in the search range
        
    This function works recursively, first using a coarse grid to find the best region (specified by the range tensors), then
    zooming in on that region, until depth is depleted. 
    """
    # handle leading shapes, so that  W is shape [num_features, quant_dim]
    leading_shape = weight.shape[:-1]
    weight = weight.view(-1, weight.shape[-1])
    num_features, _ = weight.shape
    
    # get max weight and int
    min_weight, max_weight = calculate_min_max_weight(weight, symmetric=symmetric)
    min_int, max_int = calculate_min_max_int(bits, symmetric=symmetric)
    min_int, max_int = min_int.to(weight.device, weight.dtype), max_int.to(weight.device, weight.dtype)

    # set search range. In the 'outermost' zoom, first call, we set the range to be the full range
    if search_range is None:
        if symmetric:
            upper = max_weight  / max_int  #  [num_features, 1]
        else:
            upper = (max_weight - min_weight) / max_int
        lower = upper * 0.2
    else:
        lower, upper = search_range
    
    # create candidate scales
    alpha = torch.linspace(0, 1, num_grid, device=weight.device, dtype=weight.dtype)
    candidate_scales = lower * alpha + upper * (1-alpha) # [num_features, num_grid]
    
    # quantize and reconstruct, compute error
    if symmetric:
        candidate_offsets = torch.zeros_like(candidate_scales)
    else:
        candidate_offsets = torch.round(-min_weight / candidate_scales)
    weight_ints = torch.round(weight[:, :, None] / candidate_scales[:, None, :]) + candidate_offsets[:, None, :]# [num_features, quant_dim, num_grid]
    quantized_weight = torch.clamp(weight_ints, min_int, max_int)
    reconstructed_weight = (quantized_weight - candidate_offsets[:, None, :])* candidate_scales[:, None, :] # [num_features, quant_dim, num_grid]
    errors = torch.sum(torch.abs(reconstructed_weight - weight[:, :, None]).pow_(2.4), dim=1, keepdim=False)  # [num_features, num_grid]
    
    # find the index of the best scale
    best_idx = torch.min(errors, dim=-1, keepdim=False).indices
    
    # if we're done zooming in, grab the best scales and offsets
    if depth==0:
        best_scales = candidate_scales[torch.arange(num_features), best_idx]
        if symmetric:
            best_offsets = None
        else:
            best_offsets = candidate_offsets[torch.arange(num_features), best_idx]
    else:
        # if we're not done, zoom in
        lower_idx = torch.clamp(best_idx-1, 0, num_grid)
        upper_idx = torch.clamp(best_idx+1, 0, num_grid-1)
        lower = candidate_scales[torch.arange(num_features), lower_idx][:, None]
        upper = candidate_scales[torch.arange(num_features), upper_idx][:, None]
        best_scales, best_offsets =  calculate_scales_search(weight, bits, symmetric, search_range=(lower, upper), num_grid=num_grid, depth=depth-1)
    
    # reshape back to original shape and return
    best_scales = best_scales.view(*leading_shape, 1)
    if not symmetric:
        best_offsets = best_offsets.view(*leading_shape, 1)
    return best_scales, best_offsets


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
    quantile: float | None = None,
    clip_ratio: float = 1.0,
    groupsize: int | None = None,
    device='cuda',
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Calculate the scales (and offsets if asymmetric) for quantizing a weight tensor to INT<bits>

    If search is true, we run a grid search for the best scaling factor
    otherwise, we use the quantile of the data to scale
    if quantile is None we use a 'clip ratio' times the max weight.

    Quantile is ignored if doing a search. Clip ratio is ignored if quantile is not None.

    This method accounts for grouping: we reshape the data into num_groups x group_size, and then squeeze out at the end.
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


def set_tensors(
    module: torch.nn.Linear | QuarotFP16Linear,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor | None = None,
) -> None:
    """
    Set the quantized weight, scale, and offset into a module. If the module is a torch.nn.Linear, the weight is dequantized using the scale and offset.
    Otherwise if it is a QuarotFP16Linear, the weight buffer is set to be equal to quantized_weight - offset, and the weight scale buffer is equal to the given scale.
    """
    out_features, in_features = module.weight.data.shape
    assert quantized_weight.shape == (out_features, in_features)
    assert scale.shape[0] == out_features
    if offset is not None:
        assert offset.shape == scale.shape

    if isinstance(module, QuarotFP16Linear):
        module.weight.data = quantized_weight  # out_features x in_features
        module.weight_scales.data = scale  # out_features x num_groups
        if offset is not None:
            module.offset.data = offset
    elif isinstance(module, torch.nn.Linear):
        module.weight.data = dequantize(quantized_weight, scale, offset)
    else:
        raise ValueError(f"Unsupported module type {type(module)}")


def quantize_model_rtn(
    model,
    bits: int,
    symmetric: bool = True,
    groupsize: int | None = None,
) -> None:
    """
    Quantize the weights (in QuarotFP16Linear modules) of a QuaRot model using the Round-to-Nearest scheme.

    Args:
        model: the model to quantize
        bits: the number of bits to quantize the weights to
        symmetric: whether to use symmetric quantization
        clip_weights: whether to perform a search for the best clip ratio for weight clipping
        groupsize: the groupsize for quantization. If None, quantize each channel in full.
    """
    for layer in tqdm(model.model.layers, unit="layer", desc="Quantizing layer"):
        for _, module in layer.named_modules():
            if isinstance(module, QuarotFP16Linear):
                scale, offset = calculate_scales(module.weight, bits, symmetric=symmetric, groupsize=groupsize)
                quantized_weight = quantize_weight_rtn(module.weight, scale, offset, bits)
                set_tensors(module, quantized_weight, scale, offset)
