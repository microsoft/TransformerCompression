# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from slicegpt import hf_utils, layernorm_fusion
from slicegpt.adapters.opt_adapter import OPTModelAdapter


def test_layernorm_fusion_replaces_modules() -> None:
    """Checks that module parameters are changed after applying layernorm fusion"""
    model_name = "facebook/opt-125m"
    model_adapter, _ = hf_utils.get_model_and_tokenizer(model_name)
    assert isinstance(model_adapter, OPTModelAdapter)

    orig_modules = get_module_names(model_adapter.model)

    layernorm_fusion.replace_layers(model_adapter)
    layernorm_fusion.fuse_modules(model_adapter)

    assert orig_modules != get_module_names(model_adapter.model)


def get_module_names(model) -> list[str]:
    return [name for name, _ in model.named_parameters()]
