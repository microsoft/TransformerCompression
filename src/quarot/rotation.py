# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import tqdm

from slicegpt import utils
from slicegpt.rotate import (
    config,
    rotate_attention_inputs,
    rotate_attention_output,
    rotate_embeddings,
    rotate_head,
    rotate_mlp_input,
    rotate_mlp_output,
)

from .hadamard_utils import apply_hadamard, apply_hadamard_headwise, random_hadamard_matrix
from .model_adapter import ModelAdapter


@torch.inference_mode()
def rotate_model(model_adapter: ModelAdapter, seed: int = 0) -> None:
    '''
    Rotate the model using the QuaRot method.
    '''
    model = model_adapter.model

    # Generate a random Hadamard matrix.
    Q = random_hadamard_matrix(model.config.hidden_size, seed=seed)
    Q = Q.to(config.device)

    # Work out head_dim, needed for applying Hadamards to o_proj and v_proj in attention.
    head_dim = model_adapter.config.hidden_size // model_adapter.config.num_attention_heads

    rotate_embeddings(model_adapter, Q)
    rotate_head(model_adapter, Q)

    layer_adapters = model_adapter.get_layers()
    for layer_adapter in tqdm.tqdm(layer_adapters, unit="layer", desc="Rotating"):
        rotate_attention_inputs(layer_adapter, Q)
        rotate_attention_output(layer_adapter, Q)
        rotate_mlp_input(layer_adapter, Q)
        rotate_mlp_output(layer_adapter, Q)
        apply_hadamard(layer_adapter.get_mlp_output())
        apply_hadamard_headwise(layer_adapter.get_v_proj(), head_dim=head_dim)
        apply_hadamard(layer_adapter.get_attention_output())

    utils.cleanup_memory()
