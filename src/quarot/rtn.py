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


def calculate_scales_symmetric(
    weight: torch.Tensor, bits: int, perchannel: bool = True, clip_weights=True
) -> torch.Tensor:
    """
    Calculate the scales for symmetric quantization of a weight tensor to INT<bits> using the Round-to-Nearest scheme. If clip_weights is True,
    the scales are found using a grid search to minimize the quantization error.
    """
    device = weight.device
    weight = weight.cuda()
    _, max_weight = calculate_min_max_weight(weight, symmetric=True, perchannel=perchannel)
    _, max_int = calculate_min_max_int(bits, symmetric=True)

    # Calculate the scales
    max_int = max_int.to(device=weight.device)
    scale = max_weight / max_int

    if clip_weights:
        # Perform a grid search to find the best weight scales according to quantization error.
        # TODO: This vectorized implementation gives OOM for Llama-2 70B.
        max_shrink_factor = 0.80
        n_steps = int(100 * (max_shrink_factor)) + 1
        error_norm = 2.4

        shrink_factors = torch.linspace(1.0, 1 - max_shrink_factor, n_steps)
        candidate_max_weights = shrink_factors.to(weight.device) * max_weight

        candidate_scales = candidate_max_weights / max_int
        candidate_scales = candidate_scales.unsqueeze(1)
        weight = weight.unsqueeze(-1)

        # Quantize weights
        candidate_quantized_weights = quantize_weight_rtn(weight, candidate_scales, None, bits, symmetric=True)

        # Dequantize weights
        reconstructed_weights = candidate_quantized_weights * candidate_scales

        # Compute quantization error and find the best scale for each weight
        quantization_errors = torch.sum(torch.abs(reconstructed_weights - weight).pow_(error_norm), 1)
        best_scale_indices = torch.argmin(quantization_errors, dim=-1)
        scale = torch.gather(candidate_scales.squeeze(1), 1, best_scale_indices.unsqueeze(1))

    weight = weight.to(device)
    scale = scale.to(device)

    return scale


def calculate_scales_asymmetric(
    weight: torch.Tensor, bits: int, perchannel: bool = True, clip_ratio: float = 1.0, groupsize: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the scales and offsets for asymmetric quantization a weight tensor to INT<bits> using the Round-to-Nearest scheme.
    """
    device = weight.device
    weight = weight.cuda()

    if groupsize:
        init_shape = weight.shape
        weight = weight.reshape(-1, weight.shape[-2], weight.shape[-1] // groupsize, groupsize)

    min_weight, max_weight = calculate_min_max_weight(weight, symmetric=False, perchannel=perchannel)
    min_weight *= clip_ratio
    max_weight *= clip_ratio

    _, max_int = calculate_min_max_int(bits, symmetric=False)

    # Calculate scales and offsets
    scale = (max_weight - min_weight) / max_int
    offset = torch.round(-min_weight / scale)

    if groupsize:
        scale = scale.repeat(1, 1, 1, groupsize).reshape(init_shape)
        offset = offset.repeat(1, 1, 1, groupsize).reshape(init_shape)

    weight = weight.to(device)
    scale = scale.to(device)
    offset = offset.to(device)
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
    perchannel: bool = True,
    clip_weights: bool = True,
    groupsize: int | None = None,
) -> None:
    """
    Quantize the weights of a QuarotFP16Linear module to INT<bits> using the Round-to-Nearest scheme, storing the weights in torch.float16. The weight
    scales are stored in the module's weight_scales buffer, and are also stored in torch.float16.
    """
    weight = module.weight
    offset = None
    if symmetric:
        scale = calculate_scales_symmetric(weight, bits, perchannel, clip_weights)
    else:
        scale, offset = calculate_scales_asymmetric(weight, bits, perchannel, groupsize)

    quantized_weight = quantize_weight_rtn(weight, scale, offset, bits, symmetric)

    module.weight.data = quantized_weight
    module.weight_scales = scale


def quantize_model_rtn(
    model,
    bits: int,
    symmetric: bool = True,
    perchannel: bool = True,
    clip_weights: bool = True,
    groupsize: int | None = None,
) -> None:
    """
    Quantize the weights of a model using the Round-to-Nearest scheme.

    Args:
        model: the model to quantize
        bits: the number of bits to quantize the weights to
        symmetric: whether to use symmetric quantization
        perchannel: whether to use per-channel quantization
        clip_weights: whether to clip the weights to the maximum representable value
    """
    layers = model.model.layers

    # Should work, but evaluation perplexity takes too long, not sure why
    # since the instance is exactly the same module == layer.ml.up_proj etc
    # so having to use if-else over model names and explicit layer names instead :(
    # for layer in tqdm(layers, desc="Quantizing layers", unit="layer"):
    #     for name, module in layer.named_modules():
    #         if isinstance(module, QuarotFP16Linear):
    #             quantize_module_rtn(module, bits, symmetric, perchannel, clip_weights, groupsize)

    if model.config.model_type == "llama":
        for layer in tqdm(layers, desc="Quantizing layers", unit="layer"):
            quantize_module_rtn(layer.mlp.up_proj, bits, symmetric, perchannel, clip_weights, groupsize)
            quantize_module_rtn(layer.mlp.gate_proj, bits, symmetric, perchannel, clip_weights, groupsize)
            quantize_module_rtn(layer.mlp.down_proj, bits, symmetric, perchannel, clip_weights, groupsize)
            quantize_module_rtn(layer.self_attn.q_proj, bits, symmetric, perchannel, clip_weights, groupsize)
            quantize_module_rtn(layer.self_attn.k_proj, bits, symmetric, perchannel, clip_weights, groupsize)
            quantize_module_rtn(layer.self_attn.v_proj, bits, symmetric, perchannel, clip_weights, groupsize)
            quantize_module_rtn(layer.self_attn.o_proj, bits, symmetric, perchannel, clip_weights, groupsize)
    elif model.config.model_type == "phi3":
        for layer in tqdm(layers, desc="Quantizing layers", unit="layer"):
            quantize_module_rtn(layer.mlp.gate_up_proj, bits, symmetric, perchannel, clip_weights, groupsize)
            quantize_module_rtn(layer.mlp.down_proj, bits, symmetric, perchannel, clip_weights, groupsize)
            quantize_module_rtn(layer.self_attn.qkv_proj, bits, symmetric, perchannel, clip_weights, groupsize)
            quantize_module_rtn(layer.self_attn.o_proj, bits, symmetric, perchannel, clip_weights, groupsize)
    else:
        raise NotImplementedError("Model type not supported")

