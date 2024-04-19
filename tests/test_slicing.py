# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers.models.phi.modeling_phi import PhiConfig

from slicegpt import hf_utils, layernorm_fusion, rotate
from slicegpt.adapters.hf_compatible_phi import SlicedPhiForCausalLM
from slicegpt.adapters.opt_adapter import OPTModelAdapter
from slicegpt.slicing_scheduler import ConstSlicingScheduler


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


def test_HF_model():
    """Check that the HF model weights are equivalent to the sliced model weights after layernorm fusion"""
    model_name = "microsoft/phi-2"
    model_adapter, _ = hf_utils.get_model_and_tokenizer(model_name)

    layernorm_fusion.replace_layers(model_adapter)
    layernorm_fusion.fuse_modules(model_adapter)

    config = PhiConfig.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
    )

    sliced_model = SlicedPhiForCausalLM(config).to(torch.float16)
    sliced_model.load_state_dict(model_adapter.model.state_dict(), strict=True, assign=True)

    assert compare_weights(model_adapter.model, sliced_model.model)


def test_save_and_load_HF_model():
    """Check that the HF model weights are equivalent to the sliced model weights after layernorm fusion"""
    config = PhiConfig.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
    )

    sliced_model = SlicedPhiForCausalLM(config).to(torch.float16)
    sliced_model.save_pretrained("sliced_model")
    sliced_model = SlicedPhiForCausalLM.from_pretrained("sliced_model", None, "microsoft/phi-2")


def compare_weights(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(p1.data, p2.data):
            return False
    return True


if __name__ == "__main__":
    test_HF_model()
