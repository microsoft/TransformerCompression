# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from tqdm import tqdm

from quarot.nn.linear import QuarotFP16Linear


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


def calculate_min_max_weight(
    weight: torch.Tensor, symmetric: bool = True, perchannel: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the minimum and maximum weights in a weight tensor. If perchannel, these are calculated per-row. If symmetric, the max weight is
    the larger of max weight or the absolute value of min weight.
    """
    if perchannel:
        max_weight = torch.max(weight, torch.zeros_like(weight)).max(dim=-1, keepdim=True).values
        min_weight = torch.min(weight, torch.zeros_like(weight)).min(dim=-1, keepdim=True).values
    else:
        max_weight = torch.max(weight, torch.zeros_like(weight)).max()
        min_weight = torch.min(weight, torch.zeros_like(weight)).min()

    if symmetric:
        max_weight = torch.maximum(max_weight, torch.abs(min_weight)).clamp(min=1e-5)

    return min_weight, max_weight


def calculate_scales(
    weight: torch.Tensor,
    bits: int,
    symmetric: bool,
    perchannel: bool = True,
    clip_weights: bool = False,
    vectorized: bool = True,
    clip_ratio: float = 1.0,
    groupsize: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Calculate the scales (and offsets if asymmetric) for quantizing a weight tensor to INT<bits> using the Round-to-Nearest scheme.
    """
    device = weight.device
    weight = weight.cuda()

    if groupsize:
        init_shape = weight.shape
        weight = weight.reshape(-1, weight.shape[-2], weight.shape[-1] // groupsize, groupsize)

    min_weight, max_weight = calculate_min_max_weight(weight, symmetric=symmetric, perchannel=perchannel)
    min_weight *= clip_ratio
    max_weight *= clip_ratio

    _, max_int = calculate_min_max_int(bits, symmetric=symmetric)
    max_int = max_int.to(device=weight.device)

    # Calculate scales and offsets
    if symmetric:
        scale = max_weight / max_int
        offset = None
    else:
        scale = (max_weight - min_weight) / max_int
        offset = torch.round(-min_weight / scale)

    if clip_weights:
        # Perform a grid search to find the best weight scales according to quantization error.
        max_shrink_factor = 0.20
        n_steps = int(10 * (max_shrink_factor)) + 1
        error_norm = 2.4
        shrink_factors = torch.linspace(1.0, 1 - max_shrink_factor, n_steps).to(weight.device)

        if vectorized:
            # This vectorized implementation gives OOM for Llama-2 70B.
            candidate_max_weights = shrink_factors * max_weight
            candidate_min_weights = shrink_factors * min_weight

            if symmetric:
                candidate_scales = candidate_max_weights / max_int
                candidate_offset = None
            else:
                candidate_scales = (candidate_max_weights - candidate_min_weights) / max_int
                candidate_offset = torch.round(-candidate_min_weights / candidate_scales)

            candidate_scales = candidate_scales.unsqueeze(1)
            if not symmetric:
                candidate_offset = candidate_offset.unsqueeze(1)

            weight = weight.unsqueeze(-1)

            # Quantize weights
            candidate_quantized_weights = quantize_weight_rtn(
                weight, candidate_scales, candidate_offset, bits, symmetric=symmetric
            )

            # Dequantize weights
            if symmetric:
                reconstructed_weights = candidate_quantized_weights * candidate_scales
            else:
                reconstructed_weights = (candidate_quantized_weights - offset) * candidate_scales

            # Compute quantization error and find the best scale (and offset if asymmetric) for each weight
            quantization_errors = torch.sum(torch.abs(reconstructed_weights - weight).pow_(error_norm), 1)
            best_idx = torch.argmin(quantization_errors, dim=-1)
            scale = torch.gather(candidate_scales.squeeze(1), 1, best_idx.unsqueeze(1))
            if not symmetric:
                offset = torch.gather(candidate_offset.squeeze(1), 1, best_idx.unsqueeze(1))

        else:
            best_quantization_error = torch.full(
                (weight.shape[0],), float("inf"), device=weight.device, dtype=torch.float16
            )
            for i in range(n_steps):
                shrink_factor = shrink_factors[i]
                candidate_max_weight = shrink_factor * max_weight
                candidate_min_weight = shrink_factor * min_weight

                if symmetric:
                    candidate_scale = candidate_max_weight / max_int
                    candidate_offset = None
                else:
                    candidate_scale = (candidate_max_weight - candidate_min_weight) / max_int
                    candidate_offset = torch.round(-candidate_min_weight / candidate_scale)

                # Quantize weights
                candidate_quantized_weight = quantize_weight_rtn(
                    weight, candidate_scale, candidate_offset, bits, symmetric=symmetric
                )

                # Dequantize weights
                if symmetric:
                    reconstructed_weight = candidate_quantized_weight * candidate_scale
                else:
                    reconstructed_weight = (candidate_quantized_weight - candidate_offset) * candidate_scale

                # Compute quantization error and find the best scale (and offset if asymmetric)
                quantization_error = torch.sum(torch.abs(reconstructed_weight - weight).pow_(error_norm), 1)
                if i == 0 or torch.any(quantization_error < best_quantization_error):
                    improved_idx = torch.where(quantization_error < best_quantization_error)
                    scale[improved_idx] = candidate_scale[improved_idx]
                    if not symmetric:
                        offset[improved_idx] = candidate_offset[improved_idx]

                    best_quantization_error[improved_idx] = quantization_error[improved_idx]

    if groupsize:
        scale = scale.repeat(1, 1, 1, groupsize).reshape(init_shape)
        offset = offset.repeat(1, 1, 1, groupsize).reshape(init_shape)

    weight = weight.to(device)
    scale = scale.to(device)
    offset = offset.to(device) if offset is not None else None

    return scale, offset


def quantize_weight_rtn(
    weight: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor | None, bits: int, symmetric: bool = True
) -> torch.Tensor:
    """
    Quantize a weight tensor to INT<bits> using the given scale and offset.
    """
    device = weight.device
    weight = weight.cuda()
    scale = scale.cuda()
    offset = offset.cuda() if offset is not None else None

    min_int, max_int = calculate_min_max_int(bits, symmetric)
    min_int = min_int.to(device=weight.device)
    max_int = max_int.to(device=weight.device)
    if symmetric:
        quantized_weight = torch.clamp(torch.round(weight / scale), min_int, max_int)
    else:
        offset = offset.to(device=weight.device)
        quantized_weight = torch.clamp(torch.round(weight / scale) + offset, min_int, max_int)

    quantized_weight = quantized_weight.to(device)
    return quantized_weight


def quantize_module_rtn(
    module: QuarotFP16Linear,
    bits: int,
    symmetric: bool = True,
    clip_weights: bool = True,
    groupsize: int | None = None,
) -> None:
    """
    Quantize the weights of a QuarotFP16Linear module to INT<bits> using the Round-to-Nearest scheme, storing the weights in torch.float16. The weight
    scales are stored in the module's weight_scales buffer, and are also stored in torch.float16.
    """
    weight = module.weight

    scale, offset = calculate_scales(weight, bits, symmetric, clip_weights, vectorized=True, groupsize=groupsize)
    quantized_weight = quantize_weight_rtn(weight, scale, offset, bits, symmetric)

    module.weight.data = quantized_weight
    module.weight_scales = scale


def quantize_model_rtn(
    model,
    bits: int,
    symmetric: bool = True,
    clip_weights: bool = True,
    groupsize: int | None = None,
) -> None:
    """
    Quantize the weights of a model using the Round-to-Nearest scheme.

    Args:
        model: the model to quantize
        bits: the number of bits to quantize the weights to
        symmetric: whether to use symmetric quantization
        clip_weights: whether to clip the weights to the maximum representable value
    """
    layers = model.model.layers
    for layer in tqdm(layers, desc="Quantizing layers", unit="layer"):
        quantize_module_rtn(layer.mlp.up_proj, bits, symmetric, clip_weights, groupsize)
        quantize_module_rtn(layer.mlp.gate_proj, bits, symmetric, clip_weights, groupsize)
        quantize_module_rtn(layer.mlp.down_proj, bits, symmetric, clip_weights, groupsize)
        quantize_module_rtn(layer.self_attn.q_proj, bits, symmetric, clip_weights, groupsize)
        quantize_module_rtn(layer.self_attn.k_proj, bits, symmetric, clip_weights, groupsize)
        quantize_module_rtn(layer.self_attn.v_proj, bits, symmetric, clip_weights, groupsize)
        quantize_module_rtn(layer.self_attn.o_proj, bits, symmetric, clip_weights, groupsize)
