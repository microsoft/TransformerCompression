# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals
from .utils import cleanup_memory


def rotate_attention_inputs(layer: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in layer.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_attention_inputs(layer: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the  WQ, WK and WV matrices of the self-attention layer.
    for W in layer.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

    layer.raw_layer.attn_shortcut_Q = layer.raw_layer.attn_shortcut_Q[:new_embedding_dimension, :]

    layer.get_first_layernorm().normalized_shape = (new_embedding_dimension,)  # TODO: remove?


def rotate_attention_output(layer: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.get_attention_output()

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_attention_output(layer: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice output matrix of the self-attention layer.
    W = layer.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension

    layer.raw_layer.attn_shortcut_Q = layer.raw_layer.attn_shortcut_Q[:, :new_embedding_dimension]


def rotate_mlp_input(layer: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP input weights.
    for W in layer.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_mlp_input(layer: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP input weights.
    for W in layer.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

    # slice shortcut
    layer.raw_layer.mlp_shortcut_Q = layer.raw_layer.mlp_shortcut_Q[:new_embedding_dimension, :]

    # modify layernorm
    layer.get_second_layernorm().normalized_shape = (new_embedding_dimension,)  # TODO: is this needed?


def rotate_mlp_output(layer: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP output weights and bias.
    W = layer.get_mlp_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_mlp_output(layer: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP output weights and bias.
    W = layer.get_mlp_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension

    layer.raw_layer.mlp_shortcut_Q = layer.raw_layer.mlp_shortcut_Q[:, :new_embedding_dimension]


def rotate_embeddings(model: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in model.get_embeddings():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    # Run GC and cleanup GPU memory
    cleanup_memory()


def slice_embeddings(model: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the embeddings.
    for W in model.get_embeddings():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]


def rotate_head(model: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_head(model: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the head.
    lm_head = model.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension]
    lm_head.in_features = new_embedding_dimension


@torch.no_grad()
def rotate_and_slice(
    model: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    new_embedding_dimension: int,
    do_slice_head: bool = False,
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations
    """
    model.raw_model.eval()
    dtype = next(iter(model.raw_model.parameters())).dtype

    inps, args, kwargs = [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)

    _, Q = pca_calc(inps)
    Q = Q.to(device=config.device)

    rotate_embeddings(model, Q)
    slice_embeddings(model, new_embedding_dimension)

    logging.info("Rotate and slice layers")
    layers = model.get_layers()
    for layer in tqdm(layers, unit="layer", desc="Rotating and slicing"):
        layer.raw_layer.attn_shortcut_Q = Q.T.clone().to(dtype=dtype)

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer, Q)
        slice_attention_inputs(layer, new_embedding_dimension)

        # get signal between attention and mlp, rotate and slice
        for i, inp in enumerate(inps):
            args[i] = layer.get_args_with_updated_hidden_states(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[:, :, :new_embedding_dimension].cpu(),
                args[i],
            )

        mlp_ln_inputs, _ = get_signals(layer, model.seqlen, args, kwargs)
        _, Q = pca_calc(mlp_ln_inputs)
        Q = Q.to(device=config.device, dtype=torch.float64)

        layer.raw_layer.attn_shortcut_Q = torch.matmul(layer.raw_layer.attn_shortcut_Q, Q.to(dtype=dtype))
        rotate_attention_output(layer, Q)
        slice_attention_output(layer, new_embedding_dimension)

        layer.raw_layer.mlp_shortcut_Q = Q.T.clone().to(dtype=dtype)
        rotate_mlp_input(layer, Q)
        slice_mlp_input(layer, new_embedding_dimension)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        _, inps = get_signals(layer, model.seqlen, args, kwargs)
        _, Q = pca_calc(inps)

        layer.raw_layer.mlp_shortcut_Q = torch.matmul(layer.raw_layer.mlp_shortcut_Q, Q.to(dtype=dtype))

        # optionally slice the mlp/head connection in the last layer
        dim = new_embedding_dimension
        if layer is layers[-1]:
            if not do_slice_head:
                dim = model.hidden_size

        rotate_mlp_output(layer, Q)
        slice_mlp_output(layer, dim)

        layer.raw_layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model, Q)
    if do_slice_head:
        slice_head(model, new_embedding_dimension)

    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate(model: ModelAdapter, dataloader: torch.utils.data.DataLoader[torch.Tensor]) -> None:
    """
    Rotate a model.
    TODO: Make this gpu memory efficient.
    """
    model.raw_model.eval()
    dtype = next(iter(model.raw_model.parameters())).dtype  # Get the dtype of the model.

    # List of layers to rotate.
    layers = model.get_layers()

    # Get the input of the first layer norm and calculate the Q_1
    inps, attn_masks = get_layer0_inputs(model, dataloader)
    _, Q_1 = pca_calc(inps.reshape(-1, model.config.hidden_size))
    Q_1 = Q_1.to(device=config.device)

    # Rotate the embeddings.
    rotate_embeddings(model, Q_1)

    # Rotate the rest of the model.
    logging.info("Rotate layers")
    for layer in tqdm(layers, unit="layer", desc="Rotating"):
        # Extract the inputs and outputs of the second layernorm input and calculate the Q_3
        mlp_ln_inputs, outs = get_signals(layer, inps, attn_masks)
        _, Q_3 = pca_calc(mlp_ln_inputs.reshape(-1, mlp_ln_inputs.shape[-1]))
        Q_3 = Q_3.to(device=config.device)
        _, Q_5 = pca_calc(outs.reshape(-1, outs.shape[-1]))
        Q_5 = Q_5.to(device=config.device)

        # Rotate the Q, K and V matrices of the self-attention layer.
        rotate_attention_inputs(layer, Q_1)

        # Set the shortcut rotation matrix of the self-attention layer.
        layer.raw_layer.attn_shortcut_Q = torch.matmul(Q_1.clone().T, Q_3.clone()).to(device="cpu", dtype=dtype)

        # Rotate the Attention output matrix
        rotate_attention_output(layer, Q_3)

        # Rotate the MLP input
        rotate_mlp_input(layer, Q_3)

        # Set the shortcut rotation matrix of the MLP.
        layer.raw_layer.mlp_shortcut_Q = torch.matmul(Q_3.clone().T, Q_5.clone()).to(device="cpu", dtype=dtype)

        # Rotate MLP output
        rotate_mlp_output(layer, Q_5)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        inps = outs  # The inputs to the next layer are the outputs from this one!
        Q_1 = Q_5  # first rotation in the next layer is the last one in this...

    rotate_head(model, Q_5)
    logging.info("Rotate layers done")


def slice_rotated_model(model: ModelAdapter, new_embedding_dimension: int, do_slice_head: bool = False) -> None:
    """
    TODO: Make this gpu memory efficient.
    """
    model.raw_model.eval()

    # slice embeddings
    slice_embeddings(model, new_embedding_dimension)

    # List of layers to sice.
    layers = model.get_layers()

    for layer in layers:
        slice_attention_inputs(layer, new_embedding_dimension)
        slice_attention_output(layer, new_embedding_dimension)

        # Slice attention shortcut matrix
        layer.raw_layer.attn_shortcut_Q = layer.raw_layer.attn_shortcut_Q[
            :new_embedding_dimension, :new_embedding_dimension
        ]

        slice_mlp_input(layer, new_embedding_dimension)

        # optionally slice the mlp/head connection in the last layer
        dim = new_embedding_dimension
        if layer is layers[-1]:
            if not do_slice_head:
                dim = model.hidden_size

        slice_mlp_output(layer, dim)
        layer.raw_layer.mlp_shortcut_Q = layer.raw_layer.mlp_shortcut_Q[:new_embedding_dimension, :dim]

    if do_slice_head:
        model.get_pre_head_layernorm().normalized_shape = (new_embedding_dimension,)  # TODO: remove?
        slice_head(model, new_embedding_dimension)


@torch.no_grad()
def pca_calc(X: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA on a list of batched data. Returns the eigenvalues and eigenvectors.
    """
    # Run GC and cleanup GPU memory
    cleanup_memory()

    H = None
    for X_batch in X:
        X_batch = X_batch.double().to(device=config.device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    return eig_val, eigen_vec
