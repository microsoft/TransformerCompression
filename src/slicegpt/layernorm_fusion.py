# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Callable, Iterable, Optional, TypeVar

import torch
from torch.nn import Linear, Module, Parameter

from .model_adapter import ModelAdapter
from .modules import RMSN


def replace_layers(model_adapter: ModelAdapter, verbose: bool = True) -> None:
    """Replace layers with compressed versions.

    This adds a 'shortcut operation' to each block.
    This function should be called before fusing the modules!
    """
    if verbose:
        logging.info("Replacing layers")

    replace_modules(
        model_adapter.model,
        model_adapter.original_layer_type,
        model_adapter.convert_layer_to_compressed_and_register_buffers,
        replace_layers=True,
    )

    if verbose:
        logging.info("Replacing layers done")


AnyModule = TypeVar("AnyModule", bound=Module)


def replace_modules(
    root: Module,
    type_to_replace: type[AnyModule],
    new_module_factory: Callable[
        [AnyModule, Optional[int]],
        Module,
    ],
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


def fuse_modules(model_adapter: ModelAdapter) -> None:
    """
    This function fuses the linear and layernorm into each other inplace.
    After this function is called, the model should outputs the same results as before.

    args:
        model_adapter: A ModelAdapter for the model to be fused
    """

    logging.info("Fusing layernorm modules")

    # make a copy of the weights in the lm head, which are shared with embeddings...
    head = model_adapter.get_lm_head()
    head.weight = Parameter(head.weight.clone())

    # We add the mean subtraction to the first embeddings
    for W in model_adapter.get_embeddings():
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = model_adapter.get_layers()

    # First we modify the layernorms to fold their weights
    for layer_adapter in layers:
        if model_adapter.parallel_blocks:
            fuse_ln_linear(
                layer_adapter.get_first_layernorm(),
                layer_adapter.get_attention_inputs() + layer_adapter.get_mlp_inputs(),
            )
        else:
            fuse_ln_linear(layer_adapter.get_first_layernorm(), layer_adapter.get_attention_inputs())
            fuse_ln_linear(layer_adapter.get_second_layernorm(), layer_adapter.get_mlp_inputs())

        if model_adapter.should_bake_mean_into_linear:
            # Then we bake the mean substitution into the previous linear layers
            bake_mean_into_linear(layer_adapter.get_attention_output())
            bake_mean_into_linear(layer_adapter.get_mlp_output())

    fuse_ln_linear(model_adapter.get_pre_head_layernorm(), [model_adapter.get_lm_head()])

    replace_modules(
        model_adapter.model,
        model_adapter.original_layer_norm_type,
        lambda _: RMSN(model_adapter.hidden_size),
        replace_layers=False,
    )
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
