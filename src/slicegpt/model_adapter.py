# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, final

from torch import FloatTensor, Tensor
from torch.nn import Linear, Module

"""
To add support for a new model, you need to create a new adapter class that inherits from ModelAdapter, and a new adapter class that inherits from LayerAdapter.

The ModelAdapter class tells sliceGPT how to interact with the model, an instance of which is stored at self.model. For example, how to access each of the layers of the model. 

Similarly, the LayerAdapter class tells sliceGPT how to interact with each layer of the model. For example, how to access the attention and MLP components of the layer, and how to update the arguments to the layer's forward method.

See src/slicegpt/adapters/llama_adapter.py for an example of how to implement these classes.
"""


class LayerAdapter(ABC):
    @property
    @abstractmethod
    def layer(self) -> Module:
        """
        Returns the layer that this adapter wraps.
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
        `args` is a tuple of the arguments to the layer's forward method. hidden_states is the new value for the hidden_states argument.
        This method returns a new tuple of arguments with the hidden_states argument updated.
        """
        return (
            args[: self.hidden_states_args_position] + (hidden_states,) + args[self.hidden_states_args_position + 1 :]
        )


class ModelAdapter(ABC):
    @property
    @abstractmethod
    def parallel_blocks(self) -> bool:
        """
        Whether the model has parallel attention and mlp blocks (True in phi2) or sequential (False in llama).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> Module:
        """
        The base model that slicegpt interacts with.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def no_split_module_classes(self) -> list[str] | None:
        """
        A list of string specifying the names of modules that should not be split.

        See https://huggingface.co/docs/accelerate/concept_guides/big_model_inference for more details.
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
        The class of the compressible layer (so that we can replace it with a compressed version)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def original_layer_norm_type(self) -> type:
        """
        The class of the LayerNorm (or equivalent) in the original model, so that we can replace it with RMSNorm (needed for computational invariance).
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
    def convert_layer_to_compressible(self, layer: Module) -> Module:
        """
        Replace the given layer with a compressible version of the layer.
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

    @final
    def convert_layer_to_compressible_and_register_buffers(self, layer: Module) -> Module:
        compressed_layer = self.convert_layer_to_compressible(layer)
        compressed_layer.register_buffer('mlp_shortcut_Q', None)
        compressed_layer.register_buffer('attn_shortcut_Q', None)
        return compressed_layer
