# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import transformers

from . import utils
from .modules import CompressedOPTDecoderLayer

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = CompressedOPTDecoderLayer
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_embeddings(model):
    if isinstance(model, OPT_MODEL):
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    if isinstance(model, LLAMA_MODEL):
        return [model.model.embed_tokens]

    raise NotImplementedError


def get_layers(model):
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.layers
    if isinstance(model, LLAMA_MODEL):
        return model.model.layers

    raise NotImplementedError


def get_first_layernorm(layer):
    if isinstance(layer, OPT_LAYER):
        return layer.self_attn_layer_norm
    if isinstance(layer, LLAMA_LAYER):
        return layer.input_layernorm

    raise NotImplementedError


def get_second_layernorm(layer):
    if isinstance(layer, OPT_LAYER):
        return layer.final_layer_norm
    if isinstance(layer, LLAMA_LAYER):
        return layer.post_attention_layernorm

    raise NotImplementedError


def get_pre_head_layernorm(model):
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.final_layer_norm
    if isinstance(model, LLAMA_MODEL):
        return model.model.norm

    raise NotImplementedError


def get_attention_inputs(layer):
    if isinstance(layer, (OPT_LAYER, LLAMA_LAYER)):
        return [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]

    raise NotImplementedError


def get_attention_output(layer):
    if isinstance(layer, OPT_LAYER):
        return layer.self_attn.out_proj
    if isinstance(layer, LLAMA_LAYER):
        return layer.self_attn.o_proj

    raise NotImplementedError


def get_mlp_inputs(layer):
    if isinstance(layer, OPT_LAYER):
        return [layer.fc1]
    if isinstance(layer, LLAMA_LAYER):
        return [layer.mlp.gate_proj, layer.mlp.up_proj]

    raise NotImplementedError


def get_mlp_output(layer):
    if isinstance(layer, OPT_LAYER):
        return layer.fc2
    if isinstance(layer, LLAMA_LAYER):
        return layer.mlp.down_proj

    raise NotImplementedError


def get_lm_head(model):
    if isinstance(model, (OPT_MODEL, LLAMA_MODEL)):
        return model.lm_head

    raise NotImplementedError


def get_layer0_inputs(model, batch):
    """
    Returns the inputs to the first layer of the model (after embeddings).
    NB: this won't work from OPT 350m.
    """
    # Move embeddings to device.
    for W in get_embeddings(model):
        W.weight = torch.nn.Parameter(W.weight.to(DEV))

    layers = get_layers(model)

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.saved_inps = None
            self.saved_attention_mask = None

        def forward(self, inp, **kwargs):
            self.saved_inps = inp
            self.saved_attention_masks = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    try:
        model(batch.to(DEV))
    except ValueError:
        pass

    inps = layers[0].saved_inps
    attention_mask = layers[0].saved_attention_masks[-1].unsqueeze(0)

    layers[0] = layers[0].module

    # Move embeddings back to cpu, and clear GPU cache.
    for W in get_embeddings(model):
        W.weight = torch.nn.Parameter(W.weight.to('cpu'))

    # Run GC and cleanup GPU memory
    utils.cleanup_memory()

    return inps, attention_mask


def get_signals(layer, inputs, attention_mask):
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the layer (not layernormed) and the input to the MLP (pre-layernorm).
    """
    mlp_ln_inputs = []
    layer = layer.to(DEV)

    def hook_fn(_, inp, _output):
        if isinstance(inp, tuple):
            inp = inp[0]
        mlp_ln_inputs.append(inp.cpu())

    hook = get_second_layernorm(layer).register_forward_hook(hook_fn)
    outs = [layer(inp.unsqueeze(0), attention_mask=attention_mask)[0] for inp in inputs]
    hook.remove()

    return torch.cat(mlp_ln_inputs), torch.cat(outs)
