# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from slicegpt import hf_utils, layernorm_fusion
from slicegpt.adapters.opt_adapter import OPTModelAdapter


def test_layernorm_fusion_replaces_modules() -> None:
    """Checks that module parameters are changes after applying layernorm fusion"""
    model_name = "facebook/opt-125m"
    model, _ = hf_utils.get_model(model_name)
    assert isinstance(model, OPTModelAdapter)

    orig_modules = get_module_names(model.raw_model)

    layernorm_fusion.replace_layers(model)
    layernorm_fusion.fuse_modules(model)

    assert orig_modules != get_module_names(model.raw_model)


def get_module_names(model) -> list[str]:
    return [name for name, _ in model.named_parameters()]
