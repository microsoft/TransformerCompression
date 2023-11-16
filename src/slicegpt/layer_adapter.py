# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch.nn import Linear, Module


class LayerAdapter(ABC):
    @property
    @abstractmethod
    def raw_layer(self) -> Module:
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
