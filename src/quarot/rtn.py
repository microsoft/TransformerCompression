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
        max_shrink = 0.8
        n_grid = 100
        error_norm = 2.4
        best_quantization_error = torch.full([weight.shape[0]], float('inf'), device=weight.device)
        for i in range(int(max_shrink * n_grid)):
            shrink_factor = 1 - i / n_grid
            candidate_max_weight = shrink_factor * max_weight

            if symmetric:
                candidate_scale = candidate_max_weight / max_int

                # quantize the candidate weights
                candidate_quantized_W = torch.clamp(torch.round(weight / candidate_scale), -max_int, max_int)

                # dequantize the candidate weights
                reconstructed_weight = candidate_quantized_W * candidate_scale
            else:
                raise NotImplementedError("Asymmetric quantization not implemented yet.")

            # calculate in-place the `error_norm`-norm error between the original and reconstructed weights
            quantization_error = torch.sum(reconstructed_weight.sub_(weight).abs_().pow_(error_norm), 1)

            best_scale_indices = quantization_error < best_quantization_error
            if torch.any(best_scale_indices):
                best_quantization_error[best_scale_indices] = quantization_error[best_scale_indices]
                weight_scales[best_scale_indices] = candidate_scale[best_scale_indices]

    # Quantize the weights
    W_quantized = torch.clamp(torch.round(weight / weight_scales), -max_int, max_int)

    # Set the int-quantized weights and the scales
    module.weight.data = W_quantized
    module.weight_scales = weight_scales


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
