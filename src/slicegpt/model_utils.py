# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TypeAlias

import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.opt.modeling_opt import OPTConfig, OPTDecoderLayer, OPTForCausalLM

from . import utils
from .config import config

OPT_MODEL = OPTForCausalLM
OPT_LAYER = OPTDecoderLayer
LLAMA_MODEL = LlamaForCausalLM
LLAMA_LAYER = LlamaDecoderLayer

MODEL: TypeAlias = OPTForCausalLM | LlamaForCausalLM
LAYER: TypeAlias = OPTDecoderLayer | LlamaDecoderLayer
MODEL_CONFIG: TypeAlias = OPTConfig | LlamaConfig


def get_embeddings(model: MODEL) -> list[torch.nn.Module]:
    if isinstance(model, OPT_MODEL):
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    if isinstance(model, LLAMA_MODEL):
        return [model.model.embed_tokens]

    raise NotImplementedError


def get_layers(model: MODEL) -> list[torch.nn.Module]:
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.layers
    if isinstance(model, LLAMA_MODEL):
        return model.model.layers

    raise NotImplementedError


def get_first_layernorm(layer: LAYER) -> torch.nn.Module:
    if isinstance(layer, OPT_LAYER):
        return layer.self_attn_layer_norm
    if isinstance(layer, LLAMA_LAYER):
        return layer.input_layernorm

    raise NotImplementedError


def get_second_layernorm(layer: LAYER) -> torch.nn.Module:
    if isinstance(layer, OPT_LAYER):
        return layer.final_layer_norm
    if isinstance(layer, LLAMA_LAYER):
        return layer.post_attention_layernorm

    raise NotImplementedError


def get_pre_head_layernorm(model: MODEL) -> torch.nn.Module:
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.final_layer_norm
    if isinstance(model, LLAMA_MODEL):
        return model.model.norm

    raise NotImplementedError


def get_attention_inputs(layer: LAYER) -> list[torch.nn.Linear]:
    if isinstance(layer, (OPT_LAYER, LLAMA_LAYER)):
        return [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]

    raise NotImplementedError


def get_attention_output(layer: LAYER) -> torch.nn.Linear:
    if isinstance(layer, OPT_LAYER):
        return layer.self_attn.out_proj
    if isinstance(layer, LLAMA_LAYER):
        return layer.self_attn.o_proj

    raise NotImplementedError


def get_mlp_inputs(layer: LAYER) -> list[torch.nn.Linear]:
    if isinstance(layer, OPT_LAYER):
        return [layer.fc1]
    if isinstance(layer, LLAMA_LAYER):
        return [layer.mlp.gate_proj, layer.mlp.up_proj]

    raise NotImplementedError


def get_mlp_output(layer: LAYER) -> torch.nn.Linear:
    if isinstance(layer, OPT_LAYER):
        return layer.fc2
    if isinstance(layer, LLAMA_LAYER):
        return layer.mlp.down_proj

    raise NotImplementedError


def get_lm_head(model: MODEL) -> torch.nn.Linear:
    if isinstance(model, (OPT_MODEL, LLAMA_MODEL)):
        return model.lm_head

    raise NotImplementedError


def get_layer0_inputs(model: MODEL, batch: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Returns the inputs to the first layer of the model (after embeddings).

    Also returns the additional args and kwargs that are passed to
    the first layer (such as the attention mask, or caches K/V values).

    This relies on the layer taking the hidden states as the first argument,
    and all arguments to subsequent layers being the same.

    NB: this won't work from OPT 350m.
    """
    # Move embeddings to device.
    for W in get_embeddings(model):
        W.weight = torch.nn.Parameter(W.weight.to(config.device))

    layers = get_layers(model)

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            self.saved_inps = inp
            self.saved_args = args
            self.saved_kwargs = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])

    try:
        model(batch.to(config.device))
    except ValueError:
        pass

    # grab the inputs and caught arguments
    inps = layers[0].saved_inps
    args = layers[0].saved_args
    kwargs = layers[0].saved_kwargs

    # put the caught stuff on cpu
    inps = utils.map_tensors(inps, device='cpu')
    args = utils.map_tensors(args, device='cpu')
    kwargs = utils.map_tensors(kwargs, device='cpu')

    # put the layer back to normal
    layers[0] = layers[0].module

    # Move embeddings back to cpu, and clear GPU cache.
    for W in get_embeddings(model):
        W.weight = torch.nn.Parameter(W.weight.to('cpu'))

    # Run GC and cleanup GPU memory
    utils.cleanup_memory()

    return inps, args, kwargs


def get_signals(
    layer: LAYER, inputs: list[torch.Tensor], layer_args, layer_kwargs
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the layer (not layernormed) and the input to the MLP (pre-layernorm).
    """
    mlp_ln_inputs = []
    outputs = []

    layer = layer.to(config.device)
    seqlen = inputs[0].shape[-2]

    def hook_fn(_, inp, _output):
        if isinstance(inp, tuple):
            inp = inp[0]

        # The mlp operates on (batch_size * seqlen, hidden_size) tensors, so recover batch dimension.
        mlp_ln_inputs.append(inp.cpu().reshape(-1, seqlen, inp.shape[-1]))

    hook = get_second_layernorm(layer).register_forward_hook(hook_fn)
    for inp, layer_args_batch, layer_kwargs_batch in zip(inputs, layer_args, layer_kwargs):
        inp, layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [inp, layer_args_batch, layer_kwargs_batch], device=config.device
        )
        out = layer(inp, *layer_args_batch, **layer_kwargs_batch)[0].cpu()
        outputs.append(out)

    hook.remove()

    return mlp_ln_inputs, outputs
