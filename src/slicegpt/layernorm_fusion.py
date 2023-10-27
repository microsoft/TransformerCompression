# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaPreTrainedModel, LlamaRMSNorm
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from .model_utils import (
    get_attention_inputs,
    get_attention_output,
    get_embeddings,
    get_first_layernorm,
    get_layers,
    get_lm_head,
    get_mlp_inputs,
    get_mlp_output,
    get_pre_head_layernorm,
    get_second_layernorm,
)
from .modules import RMSN, CompressedLlamaDecoderLayer, CompressedOPTDecoderLayer


def replace_modules(model, config, verbose=True):
    """
    Replace
       OPTDecoder with CompressedOPTDecoderLayer,
       LLAMADecoderLayer with CompressedLlamaDecoderLayer
    This adds a 'shortcut operation' to each block.
    This function should be called before fusing the modules!
    """
    if verbose:
        print("Replacing modules...", end=" ", flush=True)

    if isinstance(model, LlamaPreTrainedModel):
        model = model.model

    for name, module in model.named_children():
        new_module = None

        if isinstance(module, OPTDecoderLayer):
            new_module = CompressedOPTDecoderLayer(config).to(config.torch_dtype)
        elif isinstance(module, LlamaDecoderLayer):
            new_module = CompressedLlamaDecoderLayer(config).to(config.torch_dtype)
        elif len(list(module.children())) > 0:
            replace_modules(module, config, verbose=False)

        if new_module is not None:
            new_module.load_state_dict(module.state_dict(), strict=True)
            new_module.to(original_module_device)
            setattr(model, name, new_module)

    if verbose:
        print("Done.")


def replace_layernorms(model, config):
    """
    Replace
       nn.LayerNorm with slicegpt.modules.RMSN
    """
    if isinstance(model, LlamaPreTrainedModel):
        model = model.model

    for name, module in model.named_children():
        new_module = None
        if isinstance(module, (torch.nn.LayerNorm, LlamaRMSNorm)):
            new_module = RMSN(config.hidden_size)
        elif len(list(module.children())) > 0:
            replace_layernorms(module, config)

        if new_module is not None:
            setattr(model, name, new_module)
            getattr(model, name).to(module.weight.device)


def fuse_modules(model):
    """
    This function fuses the linear and layernorm into each other inplace.
    After this function is called, the model should outputs the same results as before.

    args:
        model: the model to be fused
    """

    print("Fusing layernorm modules...", end=" ", flush=True)

    # make a copy of the weights in the lm head, which are shared with embeddings...
    head = get_lm_head(model)
    head.weight = torch.nn.Parameter(head.weight.clone())

    # We add the mean subtraction to the first embeddings
    for W in get_embeddings(model):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = get_layers(model)

    # First we modify the layernorms to fold their weights
    for layer in layers:
        fuse_ln_linear(get_first_layernorm(layer), get_attention_inputs(layer))
        fuse_ln_linear(get_second_layernorm(layer), get_mlp_inputs(layer))

        # Then we bake the mean substitution into the previous linear layers
        bake_mean_into_linear(get_attention_output(layer))
        bake_mean_into_linear(get_mlp_output(layer))

    fuse_ln_linear(get_pre_head_layernorm(model), [get_lm_head(model)])

    replace_layernorms(model, model.config)
    print("Done.")


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_ln_linear(layernorm: torch.nn.LayerNorm, linear_layers: list):
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
