# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers.models.phi.modeling_phi import PhiConfig

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate
from slicegpt.adapters.opt_adapter import OPTModelAdapter
from slicegpt.adapters.sliced_phi import SlicedPhiForCausalLM
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
    """Check that the HF model weights are equivalent to the sliced model weights"""
    model_name = "microsoft/phi-2"
    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(model_name)
    sparsity = 0.1

    layernorm_fusion.replace_layers(model_adapter)
    layernorm_fusion.fuse_modules(model_adapter)

    config = PhiConfig.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )

    new_embedding_dimension = int((1 - sparsity) * model_adapter.hidden_size)
    new_embedding_dimension -= new_embedding_dimension % 8

    sliced_model = SlicedPhiForCausalLM(config).to(torch.float16)
    sliced_model.load_state_dict(model_adapter.model.state_dict(), strict=True, assign=True)

    # The sliced model weights should be identical to the HF model weights after layer norm fusion
    assert compare_weights(model_adapter.model, sliced_model.model)

    dataset = data_utils.get_dataset("wikitext2")
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    train_loader = data_utils.prepare_dataloader(dataset=train_dataset, tokenizer=tokenizer)

    test_loader = data_utils.prepare_test_dataloader(dataset=test_dataset, tokenizer=tokenizer)

    scheduler = ConstSlicingScheduler(new_embedding_dimension)
    rotate.rotate_and_slice(model_adapter, train_loader, scheduler, final_orientation="random")

    sliced_ppl = gpu_utils.evaluate_ppl(model_adapter.model.to("cuda"), tokenizer.pad_token_id, test_loader)

    sliced_model = SlicedPhiForCausalLM(config, scheduler).to(torch.float16)
    sliced_model = sliced_model.to(torch.float16)
    sliced_model.load_state_dict(model_adapter.model.state_dict(), strict=True, assign=True)

    new_model_ppl = gpu_utils.evaluate_ppl(sliced_model.to("cuda"), tokenizer.pad_token_id, test_loader)

    # The perplexity of the sliced model should be the same as the HF model
    assert sliced_ppl == new_model_ppl


def test_save_and_load_HF_model():
    """Test HF model saving and loading"""
    base_model_name = "microsoft/phi-2"
    config = PhiConfig.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
    )

    sliced_model = SlicedPhiForCausalLM(config).to(torch.float16)
    sliced_model.save_pretrained("sliced_model")
    sliced_model = SlicedPhiForCausalLM.from_pretrained("sliced_model", None, base_model_name)

    assert isinstance(sliced_model, SlicedPhiForCausalLM)
    assert sliced_model.model.config == config


def compare_weights(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(p1.data, p2.data):
            return False
    return True


if __name__ == "__main__":
    test_HF_model()
