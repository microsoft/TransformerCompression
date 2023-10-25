# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from slicegpt import hf_utils, layernorm_fusion


def test_layernorm_fusion():
    model_name = "facebook/opt-125m"
    model, tokenizer = hf_utils.get_model(model_name)

    layernorm_fusion.replace_modules(model, model.config)
    orig_layers = get_layers(model)

    assert orig_layers != get_layers(model)

    layernorm_fusion.fuse_modules(model)
    assert orig_layers != get_layers(model)

def get_layers(model):
    return [name for name, _ in model.named_parameters()]