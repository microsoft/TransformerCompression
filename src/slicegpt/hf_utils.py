# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import torch
import transformers


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
