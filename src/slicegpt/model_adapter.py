# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, final

from torch import FloatTensor, Tensor
from torch.nn import Linear, Module

"""
To add support for a new model, you need to create a new adapter class that inherits from ModelAdapter, and a new 
adapter class that inherits from LayerAdapter. The ModelAdapter class tells sliceGPT how to interact with the model, 
an instance of which is stored at self.model. For example, how to access each of the layers of the model. Similarly, 
the LayerAdapter class tells sliceGPT how to interact with each layer of the model. For example, how to access the 
attention and MLP components of the layer, and how to update the arguments to the layer's forward method.
See src/slicegpt/adapters/llama_adapter.py for an example of how to implement these classes.
"""


class LayerAdapter(ABC):
    """
    To implement a new layer adapter, implement the interface defined in this class
    """

    @property
    @abstractmethod
    def layer(self) -> Module:
        """
        Instance of the transformer layer to be wrapped. This contains the forward() method of the original model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_states_args_position(self) -> int:
        """
        Returns the position of the hidden_states argument in the layer's forward method.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_states_output_position(self) -> int:
        """
        Returns the position of the hidden_states in the output of the layer's forward method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_first_layernorm(self) -> Module:
        """
        Returns the first layer norm in the layer, usually the one before the attention component.
        """
        raise NotImplementedError

    @abstractmethod
    def get_second_layernorm(self) -> Module:
        """
        Returns the second layer norm in the layer, usually the one before the MLP component.
        In the case where the layer has only one LayerNorm, should raise an exception.
        """
        raise NotImplementedError

    @abstractmethod
    def get_attention_inputs(self) -> Sequence[Linear]:
        """
        Returns a list of the Linear layers (nn.modules) that are inputs to the attention component.
        """
        raise NotImplementedError

    @abstractmethod
    def get_attention_output(self) -> Linear:
        """
        Returns the Linear layer (nn.module) that is the output of the attention component.
        """
        raise NotImplementedError

    @abstractmethod
    def get_mlp_inputs(self) -> Sequence[Linear]:
        """
        Returns a list of the Linear layers (nn.modules) that are inputs to the MLP component.
        For simple mlps, this will be a list of length 1. For gated mlps (as in the Llama models) there will be two.
        """
        raise NotImplementedError

    @abstractmethod
    def get_mlp_output(self) -> Linear:
        """
        Returns the Linear layer (nn.module) that is the output of the MLP component (usually fc2 or down_proj)
        """
        raise NotImplementedError

    def get_updated_args(self, hidden_states: Any, args: tuple) -> tuple:
        """
        `args` is a tuple of the arguments to the layer's forward method. hidden_states is the new value for the
        hidden_states argument. This method returns a new tuple of arguments with the hidden_states argument updated.
        """
        return (
            args[: self.hidden_states_args_position] + (hidden_states,) + args[self.hidden_states_args_position + 1 :]
        )


class ModelAdapter(ABC):
    """
    To implement a new model adapter, implement the interface defined in this class
    """

    def __init__(self):
        self.slicing_conf: SlicingConfig | None = None

    @property
    @abstractmethod
    def model(self) -> Module:
        """
        The original model that slicegpt interacts with.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def config(self) -> object:
        """
        The model config
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def config_type(self) -> type:
        """
        Type of the config class
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parallel_blocks(self) -> bool:
        """
        Whether the model has parallel attention and mlp blocks (True in phi2) or sequential (False in llama).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def seqlen(self) -> int:
        """
        The (maximum) sequence length of the model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """
        The hidden size of the model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def should_bake_mean_into_linear(self) -> bool:
        """
        Whether the model's normalization layers (e.g. LayerNorm) contain a mean-subtraction
        operation that needs to be absorbed into previous linear layers.
        For LayerNorm, this is True. For RMSNorm, this is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def original_layer_type(self) -> type:
        """
        Type of the transformer layer containing forward() method of the original model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def original_layer_norm_type(self) -> type:
        """
        The class of the LayerNorm (or equivalent) in the original model, so that we can replace it with RMSNorm
        (needed for computational invariance).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def layer_adapter_type(self) -> type:
        """
        Type of the class implementing the sliceGPT.LayerAdapter interface
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def compressed_layer_type(self) -> type:
        """
        Type of the compressed transformer layer defined by the user;
        subclasses the transformer layer class;
        contains the adapted forward() method for the compressed model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def use_cache(self) -> bool:
        """Must define a setter"""
        raise NotImplementedError

    @use_cache.setter
    @abstractmethod
    def use_cache(self, value: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        """
        Returns the logits for the model on the given input_ids. For example, this might look like:
            `self._model(input_ids=input_ids).logits`
        """
        raise NotImplementedError

    @abstractmethod
    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        """
        Replace the given layer with a compressed version of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def get_layers(self) -> Sequence[LayerAdapter]:
        """
        Returns a list of LayerAdapters, one for each layer in the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_raw_layer_at(self, index: int) -> Module:
        """
        Returns the raw layer (no adapter) at the given index.
        """
        raise NotImplementedError

    @abstractmethod
    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        """
        Assigns the given layer to the model at the given index.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self) -> list[Module]:
        """
        Returns a list of the embedding modules in the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_pre_head_layernorm(self) -> Module:
        """
        Returns the layer norm (or equivalent) before the head of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_lm_head(self) -> Linear:
        """
        Returns the linear layer at the head of the model (usually of size hidden-size x vocab-size)
        """
        raise NotImplementedError

    @property
    def no_split_module_classes(self) -> list[str] | None:
        """
        A list of strings specifying the class names of modules that should not be split.
        See https://huggingface.co/docs/accelerate/concept_guides/big_model_inference for more details.
        """
        return [self.original_layer_type.__name__, self.compressed_layer_type.__name__]

    @final
    def convert_layer_to_compressed_and_register_buffers(self, layer: Module, layer_idx: int | None) -> Module:
        """
        Replace the given layer with a compressed version of the layer. Also register the shortcut_Q matrices
        to be used in Compressed transformer layer's forward() method to be updated during slicing.
        """
        compressed_layer = self.convert_layer_to_compressed(layer, layer_idx)
        if not self.parallel_blocks:
            compressed_layer.register_buffer('mlp_shortcut_Q', None)
        compressed_layer.register_buffer('attn_shortcut_Q', None)
        return compressed_layer


@dataclass
class SlicingConfig:
    """Slicing configuration such as individual layer dimensions and whether to slice head."""

    hidden_size: int = 0
    layers_num: int = 0
    do_slice_head: bool = False
    parallel_blocks: bool = False

    # use dict[int, int] instead of list[int] to allow for arbitrary order updates and default dicts
    embedding_dimensions: dict[int, int] = field(default_factory=dict)

    attention_input_dimensions: dict[int, int] = field(default_factory=dict)
    attention_output_dimensions: dict[int, int] = field(default_factory=dict)

    mlp_input_dimensions: dict[int, int] = field(default_factory=dict)
    mlp_output_dimensions: dict[int, int] = field(default_factory=dict)

    head_dimension: int | None = None

    const_dimension: int | None = None  # to be able to load models without config, sliced with const sparsity

    @staticmethod
    def from_dict(d: dict) -> 'SlicingConfig':
        """Return a SliceConfig object constructed from the provided dictionary."""

        def convert_dict_keys_to_int(d: Any) -> Any:
            # recursively convert all numeric string keys to int
            if not isinstance(d, dict):
                return d

            if all(isinstance(k, str) and k.isnumeric() for k in d.keys()):
                d = {int(k): v for k, v in d.items()}
            else:
                d = {k: convert_dict_keys_to_int(v) for k, v in d.items()}

            return d

        return SlicingConfig(**convert_dict_keys_to_int(d))

    @staticmethod
    def from_json_string(json_str: str) -> 'SlicingConfig':
        """Return a SliceConfig object constructed from the provided JSON string."""
        return SlicingConfig.from_dict(json.loads(json_str))

    def to_dict(self) -> dict:
        """Return a dictionary representation of this object."""
        # workaround until 'dataclasses.asdict support defaultdict fields #32056' is in the Python release used
        self.embedding_dimensions = {k: v for k, v in self.embedding_dimensions.items()}

        return asdict(self)

    def to_json_string(self) -> str:
        """Return a JSON representation of this object."""
        return json.dumps(self.to_dict())

    def clone(self) -> 'SlicingConfig':
        """Return a clone of this object."""
        return copy.deepcopy(self)
