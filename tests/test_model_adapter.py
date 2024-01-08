# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from abc import ABC, abstractmethod
from inspect import get_annotations
from typing import Any, Protocol, runtime_checkable

import pytest
from pyreporoot import project_root
from torch import Tensor
from torch.nn import Module, Parameter
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from transformers.models.opt.modeling_opt import OPTConfig, OPTForCausalLM

sys.path.append(project_root(__file__, root_files="pyproject.toml"))
from phi2_hf.configuration_phi import PhiConfig
from phi2_hf.modeling_phi import InferenceParams, ParallelBlock, PhiForCausalLM
from slicegpt.adapters.llama_adapter import LlamaModelAdapter
from slicegpt.adapters.opt_adapter import OPTModelAdapter
from slicegpt.adapters.phi2hf_adapter import Phi2HFModelAdapter
from slicegpt.model_adapter import ModelAdapter


@runtime_checkable
class HasShortcuts(Protocol):
    mlp_shortcut_Q: Tensor | None
    attn_shortcut_Q: Tensor | None


@runtime_checkable
class HasWeight(Protocol):
    weight: Parameter


def _validate_protocol_attr(instance: Any, protocol: type, err_message: str) -> None:
    errors: list[str] = [err_message]
    for (a, t) in get_annotations(protocol).items():
        if not hasattr(instance, a):
            errors.append(f"Missing attribute '{a}")
        elif not isinstance(getattr(instance, a), t):
            errors.append(f"Attribute '{a}' is not an instance of {t}")
    if not isinstance(instance, protocol):
        errors.append(f"Does not implement {protocol}")
    success = len(errors) == 1
    assert success, "\n".join(errors)


# Name of the abstract test class can not start with "Test", because pytest tries
# to instantiate all such classes while collecting tests.
class ModelAdapterTestBase(ABC):
    @abstractmethod
    def create_adapter(self) -> ModelAdapter:
        raise NotImplementedError

    @pytest.fixture
    def model_adapter(self) -> ModelAdapter:
        return self.create_adapter()

    def test_convert_layer_to_compressible(self, model_adapter: ModelAdapter) -> None:
        for i, layer_adapter in enumerate(model_adapter.get_layers()):
            compressed_layer = model_adapter.convert_layer_to_compressible(layer_adapter.layer)
            assert isinstance(compressed_layer, Module), f"Converted compressible layer {i} is not a torch module"
            compressed_layer = model_adapter.convert_layer_to_compressible_and_register_buffers(layer_adapter.layer)
            _validate_protocol_attr(compressed_layer, HasShortcuts, f"Converted compressible layer {i} is invalid")
            # TODO: test actual forward pass dependency on Q

    def test_layernorms_have_weight(self, model_adapter: ModelAdapter) -> None:
        pre_head_layernorm = model_adapter.get_pre_head_layernorm()
        assert isinstance(pre_head_layernorm, Module), "Pre-head layernorm is not a torch module"
        _validate_protocol_attr(pre_head_layernorm, HasWeight, "Pre-head layernorm is invalid")
        for i, layer_adapter in enumerate(model_adapter.get_layers()):
            first_layernorm = layer_adapter.get_first_layernorm()
            assert isinstance(first_layernorm, Module), f"First layernorm of layer {i} is not a torch module"
            _validate_protocol_attr(first_layernorm, HasWeight, f"First layernorm of layer {i} is invalid")
            second_layernorm = layer_adapter.get_second_layernorm()
            assert isinstance(second_layernorm, Module), f"Second layernorm of layer {i} is not a torch module"
            _validate_protocol_attr(second_layernorm, HasWeight, f"Second layernorm of layer {i} is invalid")

    def test_embeddings_have_weight(self, model_adapter: ModelAdapter) -> None:
        for i, emb in enumerate(model_adapter.get_embeddings()):
            assert isinstance(emb, Module), f"Embeddings element {i} is not a torch module"
            _validate_protocol_attr(emb, HasWeight, f"Embeddings element {i} is invalid")

    def test_can_set_use_cache(self, model_adapter: ModelAdapter) -> None:
        old_use_cache = model_adapter.use_cache
        model_adapter.use_cache = not old_use_cache
        assert model_adapter.use_cache != old_use_cache, "use_cache.setter does not work"


class TestOPTAdapter(ModelAdapterTestBase):
    def create_adapter(self) -> OPTModelAdapter:
        config = OPTConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=2,
            ffn_dim=32,
            max_position_embeddings=16,
            num_attention_heads=2,
        )
        model = OPTForCausalLM(config)
        return OPTModelAdapter(model)


class TestLlamaAdapter(ModelAdapterTestBase):
    def create_adapter(self) -> LlamaModelAdapter:
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=16,
        )
        model = LlamaForCausalLM(config)
        return LlamaModelAdapter(model)


class TestPhi2HFAdapter(ModelAdapterTestBase):
    def create_adapter(self) -> Phi2HFModelAdapter:
        config = PhiConfig()
        model = PhiForCausalLM(config)
        return Phi2HFModelAdapter(model)
