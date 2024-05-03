import torch
from tqdm import tqdm

from quarot.nn.linear import QuarotFP16Linear


def quantize_module_rtn(
    module: QuarotFP16Linear, bits: int, symmetric: bool = True, perchannel: bool = True, clip_weights=True
) -> None:
    """
    Quantize the weights of a QuarotFP16Linear module to INT<bits> using the Round-to-Nearest scheme, storing the weights in torch.float16. The weight
    scales are stored in the module's weight_scales buffer, and are also stored in torch.float16.
    """
    weight = module.weight.data
    device = weight.device
    weight = weight.cuda()

    if perchannel:
        max_weight = torch.max(weight, torch.zeros_like(weight)).max(dim=1, keepdim=True).values
        min_weight = torch.min(weight, torch.zeros_like(weight)).min(dim=1, keepdim=True).values
    else:
        raise NotImplementedError("Tensor-wise quantization not implemented yet.")

    if symmetric:
        max_weight = torch.maximum(max_weight, -min_weight).clamp(min=1e-5)
        max_int = torch.tensor(2 ** (bits - 1) - 1, device=weight.device)
        weight_scales = max_weight / max_int
    else:
        raise NotImplementedError("Asymmetric quantization not implemented yet.")

    if clip_weights:
        # Perform a grid search to find the best weight scales according to the error between the original weights and the
        # quantized-then-dequantized weights.
        max_shrink_factor = 0.8
        grid_size = 100
        error_norm = 2.4

        shrink_factors = 1 - torch.arange(int(max_shrink_factor * grid_size)) / grid_size
        candidate_max_weights = shrink_factors.to(weight.device) * max_weight

        if symmetric:
            candidate_scales = candidate_max_weights / max_int
            candidate_scales = candidate_scales.unsqueeze(1)
            weight = weight.unsqueeze(-1)

            # Quantize weights
            candidate_quantized_weights = torch.clamp(torch.round(weight / candidate_scales), -(max_int + 1), max_int)

            # Dequantize weights
            reconstructed_weights = candidate_quantized_weights * candidate_scales
        else:
            raise NotImplementedError("Asymmetric quantization not implemented yet.")

        # Compute quantization error and find the best scale for each weight
        quantization_errors = torch.sum(torch.abs(reconstructed_weights - weight).pow_(error_norm), 1)
        best_scale_indices = torch.argmin(quantization_errors, dim=-1)
        weight_scales = torch.gather(candidate_scales.squeeze(1), 1, best_scale_indices.unsqueeze(1))

    # Quantize the weights
    if symmetric:
        weight = weight.squeeze(-1)
        W_quantized = torch.clamp(torch.round(weight / weight_scales), -(max_int + 1), max_int)
    else:
        raise NotImplementedError("Asymmetric quantization not implemented yet.")

    # Set the int-quantized weights and the scales
    module.weight.data = W_quantized
    module.weight_scales = weight_scales

    # Move the weights back to the original device
    module.weight.data = module.weight.data.to(device)
    module.weight_scales = module.weight_scales.to(device)


def quantize_model_rtn(
    model, bits: int, symmetric: bool = True, perchannel: bool = True, clip_weights: bool = True
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
    for layer in tqdm(layers, desc="Quantizing layers", unit="layer"):
        quantize_module_rtn(layer.mlp.up_proj, bits, symmetric, perchannel, clip_weights)
        quantize_module_rtn(layer.mlp.gate_proj, bits, symmetric, perchannel, clip_weights)
        quantize_module_rtn(layer.mlp.down_proj, bits, symmetric, perchannel, clip_weights)
        quantize_module_rtn(layer.self_attn.q_proj, bits, symmetric, perchannel, clip_weights)
        quantize_module_rtn(layer.self_attn.k_proj, bits, symmetric, perchannel, clip_weights)
        quantize_module_rtn(layer.self_attn.v_proj, bits, symmetric, perchannel, clip_weights)
        quantize_module_rtn(layer.self_attn.o_proj, bits, symmetric, perchannel, clip_weights)
