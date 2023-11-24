# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import cast

import torch
from torch import Tensor

from . import utils
from .config import config
from .model_adapter import LayerAdapter, ModelAdapter


def get_layer0_inputs(model: ModelAdapter, batch: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    """
    Returns the inputs to the first layer of the model (after embeddings).

    Also returns the additional args and kwargs that are passed to
    the first layer (such as the attention mask, or caches K/V values).

    This relies on the layer taking the hidden states as the first argument,
    and all arguments to subsequent layers being the same.

    NB: this won't work from OPT 350m.
    """
    device = cast(torch.device, config.device)
    # Move embeddings to device.
    for W in model.get_validated_embeddings():
        W.weight = torch.nn.Parameter(W.weight.to(device))

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            self.saved_inps = inp
            self.saved_args = args
            self.saved_kwargs = kwargs
            raise ValueError

    layer0_catcher = Catcher(model.get_raw_layer_at(0))
    model.set_raw_layer_at(0, layer0_catcher)

    try:
        model.raw_model(batch.to(device))
    except ValueError:
        pass

    # grab the inputs and caught arguments
    inps = layer0_catcher.saved_inps
    args = layer0_catcher.saved_args
    kwargs = layer0_catcher.saved_kwargs

    # put the caught stuff on cpu
    inps = utils.map_tensors(inps, device='cpu')
    args = utils.map_tensors(args, device='cpu')
    kwargs = utils.map_tensors(kwargs, device='cpu')

    # put the layer back to normal
    model.set_raw_layer_at(0, layer0_catcher.module)

    # Move embeddings back to cpu, and clear GPU cache.
    for W in model.get_validated_embeddings():
        W.weight = torch.nn.Parameter(W.weight.to('cpu'))

    # Run GC and cleanup GPU memory
    utils.cleanup_memory()

    return inps, args, kwargs


def get_signals(
    layer: LayerAdapter, inputs: list[torch.Tensor], layer_args, layer_kwargs
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the layer (not layernormed) and the input to the MLP (pre-layernorm).
    """
    mlp_ln_inputs = []
    outputs = []

    layer.raw_layer.to(config.device)
    seqlen = inputs[0].shape[-2]

    def hook_fn(_, inp, _output):
        if isinstance(inp, tuple):
            inp = inp[0]

        # The mlp operates on (batch_size * seqlen, hidden_size) tensors, so recover batch dimension.
        mlp_ln_inputs.append(inp.cpu().reshape(-1, seqlen, inp.shape[-1]))

    hook = layer.get_second_layernorm().register_forward_hook(hook_fn)
    for inp, layer_args_batch, layer_kwargs_batch in zip(inputs, layer_args, layer_kwargs):
        inp, layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [inp, layer_args_batch, layer_kwargs_batch], device=config.device
        )
        out = layer.raw_layer(inp, *layer_args_batch, **layer_kwargs_batch)[0].cpu()
        outputs.append(out)

    hook.remove()

    return mlp_ln_inputs, outputs
