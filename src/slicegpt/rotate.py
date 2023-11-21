# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from functools import partial
from typing import Callable

import torch
from tqdm import tqdm

from .model_utils import (
    LAYER,
    MODEL,
    get_attention_inputs,
    get_attention_output,
    get_embeddings,
    get_first_layernorm,
    get_layer0_inputs,
    get_layers,
    get_lm_head,
    get_mlp_inputs,
    get_mlp_output,
    get_pre_head_layernorm,
    get_second_layernorm,
    get_signals,
)
from .utils import cleanup_memory

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SparsityProvider = Callable[[torch.Tensor], float]


def rotate_attention_inputs(layer: LAYER, Q: torch.Tensor) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in get_attention_inputs(layer):
        dtype = W.weight.dtype
        W_ = W.weight.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_attention_inputs(layer: LAYER, new_embedding_dimension: int) -> None:
    # Slice the  WQ, WK and WV matrices of the self-attention layer.
    for W in get_attention_inputs(layer):
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

    layer.attn_shortcut_Q = layer.attn_shortcut_Q[:new_embedding_dimension, :]

    get_first_layernorm(layer).normalized_shape = (new_embedding_dimension,)


def rotate_attention_output(layer: LAYER, Q: torch.Tensor) -> None:
    # Rotate output matrix of the self-attention layer.
    W = get_attention_output(layer)

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_attention_output(layer: LAYER, new_embedding_dimension: int) -> None:
    # Slice output matrix of the self-attention layer.
    W = get_attention_output(layer)
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension

    layer.attn_shortcut_Q = layer.attn_shortcut_Q[:, :new_embedding_dimension]


def rotate_mlp_input(layer: LAYER, Q: torch.Tensor) -> None:
    # Rotate the MLP input weights.
    for W in get_mlp_inputs(layer):
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_mlp_input(layer: LAYER, new_embedding_dimension: int) -> None:
    # Slice the MLP input weights.
    for W in get_mlp_inputs(layer):
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

    # slice shortcut
    layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:new_embedding_dimension, :]

    # modify layernorm
    get_second_layernorm(layer).normalized_shape = (new_embedding_dimension,)


def rotate_mlp_output(layer: LAYER, Q: torch.Tensor) -> None:
    # Rotate the MLP output weights and bias.
    W = get_mlp_output(layer)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_mlp_output(layer: LAYER, new_embedding_dimension: int) -> None:
    # Slice the MLP output weights and bias.
    W = get_mlp_output(layer)
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension

    layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:, :new_embedding_dimension]


def rotate_embeddings(model: MODEL, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in get_embeddings(model):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    # Run GC and cleanup GPU memory
    cleanup_memory()


def slice_embeddings(model: MODEL, new_embedding_dimension: int) -> None:
    # Slice the embeddings.
    for W in get_embeddings(model):
        W.weight.data = W.weight.data[:, :new_embedding_dimension]


def rotate_head(model: MODEL, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = get_lm_head(model)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_head(model: MODEL, new_embedding_dimension: int) -> None:
    # Slice the head.
    model.lm_head.weight.data = model.lm_head.weight.data[:, :new_embedding_dimension]
    model.lm_head.in_features = new_embedding_dimension


@torch.no_grad()
def rotate_and_slice(
    model: MODEL,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    new_embedding_dimension: int,
    sparsity_provider: SparsityProvider = None,
    do_slice_head: bool = False,
) -> None:
    """
    Rotate and slice the model, with interleaved slicing and PCA calculations.
    """

    cached_dimensions: dict[tuple[int, str], int] = {}  # cached new dimensions for each block

    # layer-wise new dimension provider based on const or varying sparsity
    get_sliced_dim = (
        partial(get_sliced_dimension, sparsity_provider=sparsity_provider, cached_dimensions=cached_dimensions)
        if sparsity_provider
        else lambda *_: new_embedding_dimension
    )

    model.eval()
    dtype = next(iter(model.parameters())).dtype

    inps, attn_masks = zip(
        *[(inp.cpu(), attn_mask.cpu()) for inp, attn_mask in (get_layer0_inputs(model, batch) for batch in dataloader)]
    )

    eig_val_emb, Q = pca_calc(inps)
    Q = Q.to(device=DEV)

    rotate_embeddings(model, Q)
    slice_embeddings(model, get_sliced_dim(-1, 'emb', eig_val_emb))

    # rotate and slice inputs
    inps = [
        torch.matmul(inp.to(device=DEV), Q.to(dtype=dtype))[:, :, : get_sliced_dim(-1, 'emb')].cpu() for inp in inps
    ]

    logging.info("Rotate and slice layers")
    layers = get_layers(model)
    for i, layer in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer.attn_shortcut_Q = Q.T.clone().to(dtype=dtype)

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer, Q)
        slice_attention_inputs(layer, get_sliced_dim(i - 1, 'emb'))

        # get signal between attention and mlp, rotate and slice
        mlp_ln_inputs, _ = get_signals(layer, inps, attn_masks)
        eig_val_attn, Q = pca_calc(mlp_ln_inputs)
        Q = Q.to(device=DEV, dtype=torch.float64)

        layer.attn_shortcut_Q = torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype))
        rotate_attention_output(layer, Q)
        slice_attention_output(layer, get_sliced_dim(i, 'attn', eig_val_attn))

        layer.mlp_shortcut_Q = Q.T.clone().to(dtype=dtype)
        rotate_mlp_input(layer, Q)
        slice_mlp_input(layer, get_sliced_dim(i, 'attn'))

        # run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the layer with slicing between Attention and mlp.
        _, outs = get_signals(layer, inps, attn_masks)
        eig_val_output, Q = pca_calc(outs)

        layer.mlp_shortcut_Q = torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype))

        # optionally slice the mlp/head connection in the last layer
        dim = get_sliced_dim(i, 'emb', eig_val_output)
        if layer is layers[-1]:
            if not do_slice_head:
                dim = model.config.hidden_size

        rotate_mlp_output(layer, Q)
        slice_mlp_output(layer, dim)

        inps = [torch.matmul(out.to(device=DEV), Q.to(dtype=dtype))[:, :, :dim].cpu() for out in outs]

        layer = layer.to('cpu')

        # run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model, Q)
    if do_slice_head:
        slice_head(model, new_embedding_dimension)

    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate(model: MODEL, dataloader: torch.utils.data.DataLoader[torch.Tensor]) -> None:
    """
    Rotate a model.
    TODO: Make this gpu memory efficient.
    """
    model.eval()
    dtype = next(iter(model.parameters())).dtype  # Get the dtype of the model.

    # List of layers to rotate.
    layers = get_layers(model)

    # Get the input of the first layer norm and calculate the Q_1
    inps, attn_masks = get_layer0_inputs(model, dataloader)
    _, Q_1 = pca_calc(inps.reshape(-1, model.config.hidden_size))
    Q_1 = Q_1.to(device=DEV)

    # Rotate the embeddings.
    rotate_embeddings(model, Q_1)

    # Rotate the rest of the model.
    logging.info("Rotate layers")
    for layer in tqdm(layers, unit="layer", desc="Rotating"):
        # Extract the inputs and outputs of the second layernorm input and calculate the Q_3
        mlp_ln_inputs, outs = get_signals(layer, inps, attn_masks)
        _, Q_3 = pca_calc(mlp_ln_inputs.reshape(-1, mlp_ln_inputs.shape[-1]))
        Q_3 = Q_3.to(device=DEV)
        _, Q_5 = pca_calc(outs.reshape(-1, outs.shape[-1]))
        Q_5 = Q_5.to(device=DEV)

        # Rotate the Q, K and V matrices of the self-attention layer.
        rotate_attention_inputs(layer, Q_1)

        # Set the shortcut rotation matrix of the self-attention layer.
        layer.attn_shortcut_Q = torch.matmul(Q_1.clone().T, Q_3.clone()).to(device="cpu", dtype=dtype)

        # Rotate the Attention output matrix
        rotate_attention_output(layer, Q_3)

        # Rotate the MLP input
        rotate_mlp_input(layer, Q_3)

        # Set the shortcut rotation matrix of the MLP.
        layer.mlp_shortcut_Q = torch.matmul(Q_3.clone().T, Q_5.clone()).to(device="cpu", dtype=dtype)

        # Rotate MLP output
        rotate_mlp_output(layer, Q_5)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        inps = outs  # The inputs to the next layer are the outputs from this one!
        Q_1 = Q_5  # first rotation in the next layer is the last one in this...

    rotate_head(model, Q_5)
    logging.info("Rotate layers done")


def slice_rotated_model(model: MODEL, new_embedding_dimension: int, do_slice_head: bool = False) -> None:
    """
    TODO: Make this gpu memory efficient.
    """
    model.eval()

    # slice embeddings
    slice_embeddings(model, new_embedding_dimension)

    # List of layers to sice.
    layers = get_layers(model)

    for layer in layers:
        slice_attention_inputs(layer, new_embedding_dimension)
        slice_attention_output(layer, new_embedding_dimension)

        # Slice attention shortcut matrix
        layer.attn_shortcut_Q = layer.attn_shortcut_Q[:new_embedding_dimension, :new_embedding_dimension]

        slice_mlp_input(layer, new_embedding_dimension)

        # optionally slice the mlp/head connection in the last layer
        dim = new_embedding_dimension
        if layer is layers[-1]:
            if not do_slice_head:
                dim = model.config.hidden_size

        slice_mlp_output(layer, dim)
        layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:new_embedding_dimension, :dim]

    if do_slice_head:
        get_pre_head_layernorm(model).normalized_shape = (new_embedding_dimension,)
        slice_head(model, new_embedding_dimension)


@torch.no_grad()
def pca_calc(X: list[torch.Tensor], damp_factor: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA on a list of batched data. Returns the eigenvalues and eigenvectors.
    """
    # Run GC and cleanup GPU memory
    cleanup_memory()

    H = None
    for X_batch in X:
        X_batch = X_batch.double().to(device=DEV)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch

    damp = damp_factor * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=DEV)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    return eig_val, eigen_vec


def get_sliced_dimension(
    layer_index: int,
    location: str,
    eig_values: torch.Tensor = None,
    sparsity_provider: SparsityProvider = None,
    cached_dimensions: dict[tuple[int, str], int] = None,
) -> int:
    """
    Get the new dimension (after slicing) to for the given layer and location.

    Args:
        layer_index: The index of the layer.
        location: The location of the slice.
        eig_values: The eigenvalues of the covariance matrix.
        sparsity_provider: The sparsity provider.
        cached_dimensions: If provided, will be used to cache previously computed dimensions.

    Returns:
        The new dimension.
    """
    k = (layer_index, location)
    if cached_dimensions is not None and k in cached_dimensions:
        sliced_dim = cached_dimensions[k]
        return sliced_dim

    sparsity = sparsity_provider(eig_values)
    sliced_dim = int(eig_values.shape[0] * (1.0 - sparsity))

    assert 0 < sliced_dim <= eig_values.shape[0]
    logging.debug(f'Dimension for layer {layer_index}:{location}: {sliced_dim}')

    if cached_dimensions is not None:
        cached_dimensions[k] = sliced_dim

    return sliced_dim


def compute_cev_sparsity(eig_values: torch.Tensor, threshold: float) -> float:
    """
    Compute sparsity based on the explained variance of the PCA.

    Args:
        eig_values: The eigenvalues of the covariance matrix.
        threshold: The threshold for the remaining unexplained variance.

    Returns:
        The calculated sparsity.
    """
    dim = eig_values.shape[0]
    assert dim > 0

    total_var = torch.sum(eig_values)
    explained_var = (eig_values / total_var).cumsum(dim=0)

    # find the index where the remaining unexplained variance drops below the threshold
    unexplained_var = 1.0 - explained_var
    idx = torch.where(unexplained_var < threshold)

    if len(idx[0]) == 0:  # unexplained variance never drops below the threshold
        return 0.0

    idx = idx[0][0].item()

    sparsity = (dim - idx) / dim
    return sparsity


def get_sparsity_provider(
    schedule: str = 'const',
    sparsity: float = 0.0,
    cev_threshold: float = 0.0,
) -> SparsityProvider:
    """
    Get the layer-wise sparsity provider for the specified schedule.

    Args:
        schedule: The sparsity schedule - const or varying.
        sparsity: The sparsity to use when the schedule is 'const'.
        cev_threshold: The threshold to use when the schedule is 'cev'.

    Returns:
        The sparsity provider.
    """

    match schedule:
        case 'const':
            return lambda *_: sparsity
        case 'cev':
            return partial(compute_cev_sparsity, threshold=cev_threshold)
        case _:
            raise ValueError(f'Unknown sparsity schedule: {schedule}')
