# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM

from .layernorm_fusion import fuse_modules, replace_modules
from .model_utils import get_layers
from .rotate import slice_rotated_model


class UninitializedOPTForCausalLM(OPTForCausalLM):
    def _init_weights(self, module):
        # Prevent weight initialization
        pass


class UninitializedLlamaForCausalLM(LlamaForCausalLM):
    def _init_weights(self, module):
        # Prevent weight initialization
        pass


def get_model(model_path: str, uninitialized: bool = False, dtype: torch.dtype = torch.float16, token=None):
    """Loads the model and the tokenizer from the given path."""
    if uninitialized:
        model_type = "uninitialized"
    else:
        model_type = "pretrained"

    logging.info(f"Loading {model_type} {model_path} model")

    if "facebook/opt" in model_path:
        if uninitialized:
            config = OPTConfig.from_pretrained(model_path)
            model = UninitializedOPTForCausalLM(config)
            model = model.to(dtype=dtype)
        else:
            model = OPTForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    elif "meta-llama" in model_path:
        if uninitialized:
            config = LlamaConfig.from_pretrained(model_path, token=token)
            model = UninitializedLlamaForCausalLM(config)
            model = model.to(dtype=dtype)
        else:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, token=token)
    else:
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)

    model.seqlen = model.config.max_position_embeddings
    model.eval()  # This switches off dropout.
    model.config.use_cache = False

    logging.info("Loading model done")

    return model, tokenizer


def load_sliced_model(model_name: str, model_path: str, sparsity: float, token: str, device: torch.device) -> tuple:
    """Loads the sliced model and the tokenizer from the given path."""
    model, tokenizer = get_model(model_name, uninitialized=True, token=token)
    replace_modules(model, model.config)
    fuse_modules(model)
    new_embedding_dimension = int((1 - sparsity) * model.config.hidden_size)

    for layer in get_layers(model):
        mlp_shortcut_Q = torch.zeros(model.config.hidden_size, model.config.hidden_size).to(dtype=torch.float16)
        attn_shortcut_Q = torch.zeros(model.config.hidden_size, model.config.hidden_size).to(dtype=torch.float16)
        layer.register_buffer("mlp_shortcut_Q", mlp_shortcut_Q)
        layer.register_buffer("attn_shortcut_Q", attn_shortcut_Q)

    slice_rotated_model(model, new_embedding_dimension)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, tokenizer
