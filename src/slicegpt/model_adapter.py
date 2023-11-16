# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from torch.nn import Linear, Module

from .layer_adapter import LayerAdapter

TLayerAdapter = TypeVar("TLayerAdapter", bound=LayerAdapter)


class ModelAdapter(ABC, Generic[TLayerAdapter]):
    @property
    @abstractmethod
    def raw_model(self) -> Module:
        pass

    @property
    @abstractmethod
    def no_split_module_classes(self) -> list[str] | None:
        pass

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
