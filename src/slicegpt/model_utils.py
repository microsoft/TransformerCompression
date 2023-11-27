# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

import torch
from torch import Tensor

from slicegpt.modules import RMSN

from . import utils
from .config import config
from .model_adapter import LayerAdapter, ModelAdapter


def get_layer0_inputs(model_adapter: ModelAdapter, batch: Tensor) -> tuple[Tensor, tuple, dict[str, Any]]:
    """
    Returns the inputs to the first layer of the model (after embeddings).

    Also returns the additional args and kwargs that are passed to
    the first layer (such as the attention mask, or caches K/V values).

    This relies on all arguments to subsequent layers being the same.

    NB: this won't work from OPT 350m.
    """
    # Move embeddings to device.
    for W in model_adapter.get_embeddings():
        W.weight = torch.nn.Parameter(W.weight.to(config.device))

    class Catcher(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *args, **kwargs):
            self.saved_args = args
            self.saved_kwargs = kwargs
            raise ValueError

    layer0_adapter = model_adapter.get_layers()[0]
    layer0_catcher = Catcher()
    model_adapter.set_raw_layer_at(0, layer0_catcher)

    try:
        batch = utils.map_tensors(batch, device=config.device)
        model_adapter.model(batch)
    except ValueError:
        pass

    # grab the inputs and caught arguments
    args = layer0_catcher.saved_args
    kwargs = layer0_catcher.saved_kwargs

    # put the caught stuff on cpu
    args = utils.map_tensors(args, device='cpu')
    kwargs = utils.map_tensors(kwargs, device='cpu')

    # put the layer back to normal
    model_adapter.set_raw_layer_at(0, layer0_adapter.layer)

    # Move embeddings back to cpu, and clear GPU cache.
    for W in model_adapter.get_embeddings():
        W.weight = torch.nn.Parameter(W.weight.to('cpu'))

    # Run GC and cleanup GPU memory
    utils.cleanup_memory()

    return args[layer0_adapter.hidden_states_args_position], args, kwargs


def get_signals(
    layer_adapter: LayerAdapter, seqlen: int, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the layer (not layernormed) and the input to the MLP (pre-layernorm).
    """
    mlp_ln_inputs = []
    outputs = []

    layer_adapter.layer.to(config.device)
    batch_size = inputs[0].shape[-3]

    def hook_fn(_, inp, _output):
        if isinstance(inp, tuple):
            inp = inp[0]

    def hook_fn(_, args: tuple, _output: Any) -> None:
        inp = args[0]  # Position in RMSN.forward args
        # The mlp operates on (batch_size * seqlen, hidden_size) tensors, so recover batch dimension.
        mlp_ln_inputs.append(inp.cpu().reshape(batch_size, -1, inp.shape[-1]))

    second_layernorm = layer_adapter.get_second_layernorm()
    assert isinstance(second_layernorm, RMSN)
    hook = second_layernorm.register_forward_hook(hook_fn)
    for layer_args_batch, layer_kwargs_batch in zip(layer_args, layer_kwargs):
        layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        if isinstance(out, tuple):
            out = out[layer_adapter.hidden_states_output_position]
        out = out.cpu()
        outputs.append(out)

    hook.remove()

    return mlp_ln_inputs, outputs
