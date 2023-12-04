# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, final

from torch import FloatTensor, Tensor
from torch.nn import Linear, Module


class LayerAdapter(ABC):
    @property
    @abstractmethod
    def layer(self) -> Module:
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

    def get_args_with_updated_hidden_states(self, hidden_states: Any, args: tuple) -> tuple:
        return (
            args[: self.hidden_states_args_position] + (hidden_states,) + args[self.hidden_states_args_position + 1 :]
        )


class ModelAdapter(ABC):
    @property
    @abstractmethod
    def model(self) -> Module:
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
    def _get_use_cache(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _set_use_cache(self, value: bool) -> None:
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

    @property
    def use_cache(self) -> bool:
        return self._get_use_cache()

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self._set_use_cache(value)

    @final
    def convert_layer_to_compressible_and_register_buffers(self, layer: Module) -> Module:
        compressed_layer = self.convert_layer_to_compressible(layer)
        compressed_layer.register_buffer('mlp_shortcut_Q', None)
        compressed_layer.register_buffer('attn_shortcut_Q', None)
        return compressed_layer
