# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, Protocol, TypeVar, final, runtime_checkable

from torch import FloatTensor, Tensor
from torch.nn import Linear, Module


@runtime_checkable
class HasShortcuts(Protocol):
    mlp_shortcut_Q: Tensor
    attn_shortcut_Q: Tensor


class _ModuleWithShortcutsMeta(ABCMeta):
    def __call__(self, *args, **kwargs) -> Any:
        cls = ABCMeta.__call__(self, *args, **kwargs)

        cls.register_buffer("mlp_shortcut_Q", None)
        cls.register_buffer("attn_shortcut_Q", None)

        return cls


class ModuleWithShortcuts(ABC, metaclass=_ModuleWithShortcutsMeta):
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass


TLayer = TypeVar("TLayer", bound=Module)
TCompressableLayer = TypeVar("TCompressableLayer", bound=Module)


class LayerAdapter(ABC, Generic[TLayer, TCompressableLayer]):
    @property
    @abstractmethod
    def raw_layer(self) -> TLayer | TCompressableLayer:
        pass

    @abstractmethod
    def get_first_layernorm(self) -> Module:
        pass

    @abstractmethod
    def get_second_layernorm(self) -> Module:
        pass

    @abstractmethod
    def get_attention_inputs(self) -> Sequence[Linear]:
        pass

    @abstractmethod
    def get_attention_output(self) -> Linear:
        pass

    @abstractmethod
    def get_mlp_inputs(self) -> Sequence[Linear]:
        pass

    @abstractmethod
    def get_mlp_output(self) -> Linear:
        pass


TLayerAdapter = TypeVar("TLayerAdapter", bound=LayerAdapter)


class ModelAdapter(ABC, Generic[TLayer, TCompressableLayer, TLayerAdapter]):
    @property
    @abstractmethod
    def raw_model(self) -> Module:
        pass

    @property
    @abstractmethod
    def no_split_module_classes(self) -> list[str] | None:
        pass

    @property
    @abstractmethod
    def seqlen(self) -> int:
        pass

    @property
    @abstractmethod
    def should_bake_mean_into_linear(self) -> bool:
        pass

    @property
    @abstractmethod
    def original_layer_type(self) -> type[TLayer]:
        pass

    @property
    @abstractmethod
    def compressable_layer_type(self) -> type[TCompressableLayer]:
        pass

    @abstractmethod
    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        pass

    @abstractmethod
    def convert_layer_to_compressable(self, layer: TLayer) -> TCompressableLayer:
        pass

    @final
    def convert_layer_to_compressable_and_validate(self, layer: TLayer) -> TCompressableLayer:
        compressed_layer = self.convert_layer_to_compressable(layer)
        assert isinstance(compressed_layer, HasShortcuts)
        return compressed_layer

    @abstractmethod
    def get_layers(self) -> Sequence[TLayerAdapter]:
        pass

    @abstractmethod
    def get_embeddings(self) -> list[Module]:
        pass

    @abstractmethod
    def get_pre_head_layernorm(self) -> Module:
        pass

    @abstractmethod
    def get_lm_head(self) -> Linear:
        pass
