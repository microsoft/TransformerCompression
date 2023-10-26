# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import torch
import transformers
from slicegpt import rotate, layernorm_fusion, model_utils


def skip(*args, **kwargs):
    pass


def do_not_initialize(func):
    """
    A decorator that prevents initalization of torch.nn modules.
    """

    def wrapper(*args, **kwargs):
        kiming_fn = torch.nn.init.kaiming_uniform_
        uniform_fn = torch.nn.init.uniform_
        normal_fn = torch.nn.init.normal_

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        result = func(*args, **kwargs)

        torch.nn.init.kaiming_uniform_ = kiming_fn
        torch.nn.init.uniform_ = uniform_fn
        torch.nn.init.normal_ = normal_fn

        return result

    return wrapper


@do_not_initialize
def get_model(model_path, hf_token=None):
    print("Loading model from {} ...".format(model_path))

    if "facebook/opt" in model_path:
        model = transformers.OPTForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    elif "meta-llama" in model_path:
        model = transformers.LlamaForCausalLM.from_pretrained(model_path, torch_dtype='auto', use_auth_token=hf_token)
    else:
        raise NotImplementedError

    if hf_token == None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False, use_auth_token=hf_token)

    model.seqlen = model.config.max_position_embeddings
    model.eval()  # This switches off dropout.
    model.config.use_cache = False  # Do not cache attention key values.

    return model, tokenizer

def load_sliced_model(model_name, hf_token, model_path, sparsity, device):
    """ Loads the sliced model and the tokenzer from the given path. """
    model, tokenizer = get_model(model_name, hf_token)
    layernorm_fusion.replace_modules(model, model.config)
    layernorm_fusion.fuse_modules(model)
    new_embedding_dimension = int((1 - sparsity) * model.config.hidden_size)

    for layer in model_utils.get_layers(model):
        mlp_shortcut_Q = torch.zeros(model.config.hidden_size, model.config.hidden_size).to(dtype=torch.float16)
        attn_shortcut_Q = torch.zeros(model.config.hidden_size, model.config.hidden_size).to(dtype=torch.float16)
        layer.register_buffer("mlp_shortcut_Q", mlp_shortcut_Q)
        layer.register_buffer("attn_shortcut_Q", attn_shortcut_Q)

    rotate.slice_rotated_model(model, new_embedding_dimension)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, tokenizer
