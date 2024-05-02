import torch

from quarot.nn.linear import QuarotFP16Linear


def quantize_module_rtn(module: QuarotFP16Linear, bits: int, sym: bool = True, perchannel: bool = True) -> None:
    """
    Quantize the weights of a QuarotFP16Linear module to INT<bits> using the Round-to-Nearest scheme, storing the weights in torch.float16. The weight
    scales are stored in the module's weight_scales buffer, and are also stored in torch.float16.
    """
    W = module.weight.data

    if perchannel:
        max_weight = torch.max(W, torch.zeros_like(W)).max(dim=1, keepdim=True).values
        min_weight = torch.min(W, torch.zeros_like(W)).min(dim=1, keepdim=True).values
    else:
        raise NotImplementedError("Tensor-wise quantization not implemented yet.")

    if sym:
        max_weight = torch.maximum(max_weight, -min_weight).clamp(min=1e-5)
        max_integer_value = torch.tensor(2 ** (bits - 1) - 1, device=W.device)
        weight_scales = max_weight / max_integer_value
    else:
        raise NotImplementedError("Asymmetric quantization not implemented yet.")

    # Quantize the weights
    W_quantized = torch.round(W / weight_scales).clamp(-max_integer_value, max_integer_value)

    # Set the int-quantized weights and the scales
    module.weight.data = W_quantized
    module.weight_scales = weight_scales


def quantize_model_rtn(model, bits: int, sym: bool = True, perchannel: bool = True) -> None:
    """
    Quantize the weights of a model using the Round-to-Nearest scheme.
    """
    layers = model.model.layers
    for layer in layers:
        quantize_module_rtn(layer.mlp.up_proj, bits, sym, perchannel)
        quantize_module_rtn(layer.mlp.gate_proj, bits, sym, perchannel)
        quantize_module_rtn(layer.mlp.down_proj, bits, sym, perchannel)
        quantize_module_rtn(layer.self_attn.q_proj, bits, sym, perchannel)
        quantize_module_rtn(layer.self_attn.k_proj, bits, sym, perchannel)
        quantize_module_rtn(layer.self_attn.v_proj, bits, sym, perchannel)
        quantize_module_rtn(layer.self_attn.o_proj, bits, sym, perchannel)
