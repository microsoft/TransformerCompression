# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Callable, Iterable, TypeVar

import torch
from torch.nn import Linear, Module, Parameter

from .model_adapter import ModelAdapter
from .modules import RMSN


def replace_layers(model: ModelAdapter, verbose: bool = True) -> None:
    """Replace layers with compressible versions.

    This adds a 'shortcut operation' to each block.
    This function should be called before fusing the modules!
    """
    if verbose:
        logging.info("Replacing modules")

    _replace_modules(
        model.raw_model,
        model.original_layer_type,
        model.convert_layer_to_compressible_and_validate,
    )

    if verbose:
        logging.info("Replacing modules done")


_AnyModule = TypeVar("_AnyModule", bound=Module)


def _replace_modules(
    root: Module, type_to_replace: type[_AnyModule], new_module_factory: Callable[[_AnyModule], Module]
) -> None:
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            _replace_modules(module, type_to_replace, new_module_factory)

        if new_module is not None:
            setattr(root, name, new_module)


def fuse_modules(model: ModelAdapter) -> None:
    """
    This function fuses the linear and layernorm into each other inplace.
    After this function is called, the model should outputs the same results as before.

    args:
        model: the model to be fused
    """

    logging.info("Fusing layernorm modules")

    # make a copy of the weights in the lm head, which are shared with embeddings...
    head = model.get_lm_head()
    head.weight = Parameter(head.weight.clone())

    # We add the mean subtraction to the first embeddings
    for W in model.get_validated_embeddings():
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = model.get_layers()

    # First we modify the layernorms to fold their weights
    for layer in layers:
        fuse_ln_linear(layer.get_first_layernorm(), layer.get_attention_inputs())
        fuse_ln_linear(layer.get_second_layernorm(), layer.get_mlp_inputs())

        if model.should_bake_mean_into_linear:
            # Then we bake the mean substitution into the previous linear layers
            bake_mean_into_linear(layer.get_attention_output())
            bake_mean_into_linear(layer.get_mlp_output())

    fuse_ln_linear(model.get_pre_head_layernorm(), [model.get_lm_head()])

    _replace_modules(model.raw_model, model.original_layer_norm_type, lambda _: RMSN(model.hidden_size))
    logging.info("Fusing layernorm modules done")


def bake_mean_into_linear(linear: Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_ln_linear(layernorm: Module, linear_layers: Iterable[Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    if not hasattr(layernorm, 'weight'):
        raise TypeError("Layer norm does not define weight")

    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
