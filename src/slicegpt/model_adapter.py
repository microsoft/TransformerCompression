# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Sequence
from inspect import get_annotations
from typing import Any, Protocol, cast, final, runtime_checkable

from torch import FloatTensor, Tensor
from torch.nn import Linear, Module, Parameter


@runtime_checkable
class HasShortcuts(Protocol):
    mlp_shortcut_Q: Tensor | None
    attn_shortcut_Q: Tensor | None


@runtime_checkable
class HasWeight(Protocol):
    weight: Parameter


def _validate_protocol_attr(instance: Any, protocol: type, err_message: str | None = None) -> None:
    errors: list[str] = []
    for (a, t) in get_annotations(protocol).items():
        if not hasattr(instance, a):
            errors.append(f"Missing attribute '{a}")
        elif not isinstance(getattr(instance, a), t):
            errors.append(f"Attribute '{a}' is not an instance of {t}")
    if not isinstance(instance, protocol):
        errors.append(f"Does not implement {protocol}")
    if len(errors) != 0:
        raise TypeError(err_message, errors, instance) if err_message is not None else TypeError(errors, instance)


class LayerAdapter(ABC):
    @property
    @abstractmethod
    def raw_layer(self) -> Module:
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_states_args_position(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_states_output_position(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_first_layernorm(self) -> Module:
        raise NotImplementedError

    @abstractmethod
    def get_second_layernorm(self) -> Module:
        raise NotImplementedError

    @abstractmethod
    def get_attention_inputs(self) -> Sequence[Linear]:
        raise NotImplementedError

    @abstractmethod
    def get_attention_output(self) -> Linear:
        raise NotImplementedError

    @abstractmethod
    def get_mlp_inputs(self) -> Sequence[Linear]:
        raise NotImplementedError

    @abstractmethod
    def get_mlp_output(self) -> Linear:
        raise NotImplementedError

    @final
    def get_validated_first_layernorm(self) -> HasWeight:
        layer_norm = self.get_first_layernorm()
        _validate_protocol_attr(layer_norm, HasWeight, "Layer has invalid first layer norm")
        return cast(HasWeight, layer_norm)

    @final
    def get_validated_second_layernorm(self) -> HasWeight:
        layer_norm = self.get_second_layernorm()
        _validate_protocol_attr(layer_norm, HasWeight, "Layer has invalid second layer norm")
        return cast(HasWeight, layer_norm)

    def get_args_with_updated_hidden_states(self, hidden_states: Any, args: tuple) -> tuple:
        return (
            args[: self.hidden_states_args_position] + (hidden_states,) + args[self.hidden_states_args_position + 1 :]
        )


class ModelAdapter(ABC):
    @property
    @abstractmethod
    def raw_model(self) -> Module:
        raise NotImplementedError

    @property
    @abstractmethod
    def no_split_module_classes(self) -> list[str] | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def seqlen(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def should_bake_mean_into_linear(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def original_layer_type(self) -> type:
        raise NotImplementedError

    @property
    @abstractmethod
    def original_layer_norm_type(self) -> type:
        raise NotImplementedError

    @abstractmethod
    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        raise NotImplementedError

    @abstractmethod
    def convert_layer_to_compressible(self, layer: Module) -> Module:
        raise NotImplementedError

    @abstractmethod
    def get_layers(self) -> Sequence[LayerAdapter]:
        raise NotImplementedError

    @abstractmethod
    def get_raw_layer_at(self, index: int) -> Module:
        raise NotImplementedError

    @abstractmethod
    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self) -> list[Module]:
        raise NotImplementedError

    @abstractmethod
    def get_pre_head_layernorm(self) -> Module:
        raise NotImplementedError

    @abstractmethod
    def get_lm_head(self) -> Linear:
        raise NotImplementedError

    @final
    def convert_layer_to_compressible_and_validate(self, layer: Module) -> Module:
        compressed_layer = self.convert_layer_to_compressible(layer)
        if not isinstance(compressed_layer, Module):
            raise TypeError("Converted compressible layer is not a torch module")
        _validate_protocol_attr(compressed_layer, HasShortcuts, "Converted compressible layer is invalid")
        return compressed_layer

    @final
    def get_validated_embeddings(self) -> list[HasWeight]:
        embeddings = self.get_embeddings()
        for emb in embeddings:
            _validate_protocol_attr(emb, HasWeight, "Model has invalid embeddings")
        return cast(list[HasWeight], embeddings)
