# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, Protocol, TypeVar, cast, final, runtime_checkable

from torch import FloatTensor, Tensor
from torch.nn import Linear, Module


@runtime_checkable
class HasShortcuts(Protocol):
    mlp_shortcut_Q: Tensor
    attn_shortcut_Q: Tensor


class _ModuleWithShortcutsMeta(ABCMeta):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        cls = ABCMeta.__call__(cls, *args, **kwargs)
        if not isinstance(cls, Module):
            raise TypeError("This metaclass can be applied only to descendants of torch.nn.Module")

        cls.register_buffer("mlp_shortcut_Q", None)
        cls.register_buffer("attn_shortcut_Q", None)

        return cls


class ModuleWithShortcuts(ABC, metaclass=_ModuleWithShortcutsMeta):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass


AnyLayer = TypeVar("AnyLayer", bound=Module)
AnyCompressableLayer = TypeVar("AnyCompressableLayer", bound=Module)
AnyLayerNorm = TypeVar("AnyLayerNorm", bound=Module)


class LayerAdapter(ABC, Generic[AnyLayer, AnyCompressableLayer, AnyLayerNorm]):
    @property
    @abstractmethod
    def raw_layer(self) -> AnyLayer | AnyCompressableLayer:
        pass

    @abstractmethod
    def get_first_layernorm(self) -> AnyLayerNorm:
        pass

    @abstractmethod
    def get_second_layernorm(self) -> AnyLayerNorm:
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


AnyLayerAdapter = TypeVar("AnyLayerAdapter", bound=LayerAdapter)


class ModelAdapter(ABC, Generic[AnyLayer, AnyCompressableLayer, AnyLayerNorm, AnyLayerAdapter]):
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
    def hidden_size(self) -> int:
        pass

    @property
    @abstractmethod
    def should_bake_mean_into_linear(self) -> bool:
        pass

    @property
    @abstractmethod
    def original_layer_type(self) -> type[AnyLayer]:
        pass

    @property
    @abstractmethod
    def compressable_layer_type(self) -> type[AnyCompressableLayer]:
        pass

    @property
    @abstractmethod
    def layer_norm_type(self) -> type[AnyLayerNorm]:
        pass

    @abstractmethod
    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        pass

    @abstractmethod
    def convert_layer_to_compressible(self, layer: AnyLayer) -> AnyCompressableLayer:
        pass

    @final
    def convert_layer_to_compressible_and_validate(self, layer: AnyLayer) -> AnyCompressableLayer:
        compressed_layer = self.convert_layer_to_compressible(layer)
        assert isinstance(compressed_layer, HasShortcuts)
        return cast(AnyCompressableLayer, compressed_layer)

    @abstractmethod
    def get_layers(self) -> Sequence[AnyLayerAdapter]:
        pass

    @abstractmethod
    def get_embeddings(self) -> list[Module]:
        pass

    @abstractmethod
    def get_pre_head_layernorm(self) -> AnyLayerNorm:
        pass

    @abstractmethod
    def get_lm_head(self) -> Linear:
        pass
