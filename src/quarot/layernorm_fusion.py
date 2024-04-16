# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import TypeVar

from torch.nn import Module, Parameter

from slicegpt.layernorm_fusion import bake_mean_into_linear, fuse_ln_linear, replace_modules
from slicegpt.modules import RMSN

from .model_adapter import ModelAdapter


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
        model_adapter.convert_layer_to_quarot,
        replace_layers=True,
    )

    if verbose:
        logging.info("Replacing layers done")


AnyModule = TypeVar("AnyModule", bound=Module)


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
