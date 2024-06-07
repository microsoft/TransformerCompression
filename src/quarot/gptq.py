import logging
from typing import Any

import torch
from tqdm import tqdm

from quarot.model_adapter import LayerAdapter, ModelAdapter
from quarot.nn.linear import QuarotFP16Linear
from quarot.quant_utils import PackedQuantizedTensor
from quarot.rtn import calculate_scales, quantize_weight_rtn
from slicegpt.rotate import get_layer0_inputs
from slicegpt.utils import cleanup_memory, map_tensors


def gptq_quantize_column(i, col_idx, block_end_idx, W, Q, Err_block, L_inv_transpose, scale, offset, bits, symmetric):
    """
    Quantize one column of the weight matrix, W[:, col_idx]

    i indexes the current position in the block.
    """
    # store the int-quantized weight column
    Q[:, col_idx] = quantize_weight_rtn(W[:, col_idx : col_idx + 1], scale, offset, bits, symmetric=symmetric).flatten()

    # calculate the dequantized weight
    if symmetric:
        W_dequantized = Q[:, col_idx] * scale.flatten()
    else:
        W_dequantized = (Q[:, col_idx] - offset.flatten()) * scale.flatten()

    # calculate quantization error (between original and dequantized weight column)
    Err_block[:, i] = (W[:, col_idx] - W_dequantized) / L_inv_transpose[col_idx, col_idx]

    # update the rest of the weights in the block
    W[:, col_idx:block_end_idx] -= (
        Err_block[:, i : i + 1] * L_inv_transpose[col_idx : col_idx + 1, col_idx:block_end_idx]
    )


@torch.no_grad()
def quantize_weight_gptq(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    symmetric: bool = True,
    max_blocksize: int = 128,
    percdamp: float = 0.01,
    groupsize: int | None = None,
    clip_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Quantize a weight tensor to INT<bits> using the GPTQ scheme.

    Args:
        W: The weight tensor to quantize.
        H: The Hessian of the activations, must have dtype==torch.float32.
        bits: The number of bits to quantize to.
        symmetric: Whether to use symmetric quantization.
        max_blocksize: The maximum block size for the GPTQ algorithm.
        percdamp: The damping factor for the Hessian.
        groupsize: The number of weights to group together for quantization.
        clip_weights: Whether to clip the weights (by searching for good clipping ratios) to the range of the quantization.
    """
    num_features, num_columns = W.shape
    orig_dev = W.device
    orig_dtype = W.dtype
    W = W.to('cuda', dtype=torch.float32)
    H = H.to(W.device)

    # deal with group quantization.
    # If groupsize is None, we quantize the entire tensor at once (there's a single scale and offset).
    # otherwise, we collect the scale/offset for each group in a list
    if groupsize is None:
        scale, offset = calculate_scales(W, bits, symmetric=symmetric, clip_weights=clip_weights, vectorized=False)
        scale = scale.float()
        offset = offset.float() if offset is not None else None
    else:
        group_scales = []
        group_offsets = []

    # find dead weights and set them to zero, and their Hess value to 1
    dead = torch.diag(H) == 0.0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    # add damping to the diagonal
    H.diagonal().add_(percdamp * torch.mean(torch.diag(H)))

    # calculate the inverse layer Hessian (Cholesky form).
    # NB this is different to the original implementation. We require the (transpose) of the choleskyu of the _inverse_ of H.
    # oringially this was done by chol(inv(H)). Here we do chol(H)^-1, with some re-ordering to ensure we get an equivalent solution.
    # chol(inv(H), upper=True) = flip(inv(chol(flip(H), upper=False)))
    # This version requires fewer flops and should be numerically superior.
    PHP = torch.flip(H, (0, 1))
    L = torch.linalg.cholesky(PHP)
    L_inv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
    L_inv_transpose = torch.flip(L_inv, (0, 1))

    Q = torch.zeros_like(W)
    for block_start_idx in range(0, num_columns, max_blocksize):
        block_end_idx = min(block_start_idx + max_blocksize, num_columns)
        blocksize = block_end_idx - block_start_idx

        Err_block = torch.zeros((W.shape[0], blocksize), dtype=W.dtype, device=W.device)

        for i in range(blocksize):  # i = which col to quantize (wrt block)
            col_idx = block_start_idx + i  # which ciolumn to quantize (wrt original matrix)

            # if this is a new group, calculate the scales for the group
            if groupsize is not None and col_idx % groupsize == 0:
                scale, offset = calculate_scales(
                    W[:, col_idx : col_idx + groupsize],
                    bits,
                    symmetric=symmetric,
                    clip_weights=clip_weights,
                    vectorized=False,
                )
                scale = scale.float()
                offset = offset.float() if offset is not None else None
                group_scales.append(scale)
                group_offsets.append(offset)

            gptq_quantize_column(
                i, col_idx, block_end_idx, W, Q, Err_block, L_inv_transpose, scale, offset, bits, symmetric
            )

        # update the rest of the weights in the tensor
        W[:, block_end_idx:] -= Err_block @ L_inv_transpose[block_start_idx:block_end_idx, block_end_idx:]

    # stack the scales for grouped quantization
    if groupsize is not None:
        scale = torch.cat(group_scales, dim=1)
        offset = torch.cat(group_offsets, dim=1) if group_offsets[0] is not None else None

    Q = Q.to(orig_dev, dtype=orig_dtype)
    scale = scale.to(orig_dev, dtype=orig_dtype)
    if not symmetric:
        offset = offset.to(orig_dev, dtype=orig_dtype)

    return Q, scale, offset


def construct_hessian(
    X: list[torch.Tensor | PackedQuantizedTensor], ignore_masks: list[torch.Tensor] | None = None
) -> torch.Tensor:
    """
    Construct the Hessian matrix for a given set of activations.
    """
    # Run GC and cleanup GPU memory
    cleanup_memory()

    if isinstance(X[0], PackedQuantizedTensor):
        X = [x.quantized_x for x in X]

    H = None
    num_samples = 0
    hidden_dim = X[0].shape[-1]
    H = torch.zeros(hidden_dim, hidden_dim, device='cuda')
    for idx, X_batch in enumerate(X):
        batch_size, seq_len = X_batch.shape[:2]
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0
            num_elements = ignore_masks[idx].sum()
        else:
            num_elements = batch_size * seq_len

        X_batch = X_batch.to('cuda', dtype=torch.float32) * torch.rsqrt(torch.tensor(num_elements, device='cuda'))
        H_batch = torch.einsum('bld,blc->dc', X_batch, X_batch)
        H = H * (num_samples / (num_samples + num_elements)) + H_batch * (num_elements / (num_samples + num_elements))
        num_samples += num_elements

    return H


@torch.no_grad()
def get_signals(
    layer_adapter: LayerAdapter, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]]
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the layer and the inputs to the attention inputs & output, and to the mlp inputs & output.
    """
    qkv_inputs, o_proj_inputs, upgate_inputs, down_proj_inputs = [], [], [], []
    outputs = []
    device = 'cuda'
    layer_adapter.layer.to(device)

    def make_hook(storage_list):
        def hook_fn(_, args: tuple, _output: Any) -> None:
            storage_list.append(args[0])

        return hook_fn

    hooks = [make_hook(qkv_inputs), make_hook(o_proj_inputs), make_hook(upgate_inputs), make_hook(down_proj_inputs)]
    modules = [
        layer_adapter.get_attention_inputs()[0],
        layer_adapter.get_attention_output(),
        layer_adapter.get_mlp_inputs()[0],
        layer_adapter.get_mlp_output(),
    ]
    hooks = [m.register_forward_hook(h) for m, h in zip(modules, hooks)]

    for layer_args_batch, layer_kwargs_batch in zip(layer_args, layer_kwargs):
        layer_args_batch, layer_kwargs_batch = map_tensors([layer_args_batch, layer_kwargs_batch], device=device)
        out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        if isinstance(out, tuple):
            out = out[layer_adapter.hidden_states_output_position]
        outputs.append(out)

    for h in hooks:
        h.remove()

    return qkv_inputs, o_proj_inputs, upgate_inputs, down_proj_inputs, outputs


def set_tensors(
    module: torch.nn.Linear | QuarotFP16Linear,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor | None = None,
) -> None:
    """
    Set the quantized weight, scale, and offset into a module. If the module is a torch.nn.Linear, the weight is dequantized using the scale and offset.
    Otherwise if it is a QuarotFP16Linear, the weight buffer is set to be equal to quantized_weight - offset, and the weight scale buffer is equal to the given scale.
    """
    out_features, in_features = module.weight.data.shape
    assert quantized_weight.shape == (out_features, in_features)
    assert scale.shape == (out_features, 1)
    if offset is not None:
        assert offset.shape == (out_features, 1)

    if isinstance(module, QuarotFP16Linear):
        module.weight.data = quantized_weight  # out_features x in_features
        module.weight_scales.data = scale  # out_features x 1
        if offset is not None:
            module.weight.data -= offset
    elif isinstance(module, torch.nn.Linear):
        # Here we dequantize the weights and set them back into the module
        module.weight.data = quantized_weight * scale
        if offset is not None:
            module.weight.data -= offset * scale
    else:
        raise ValueError(f"Unsupported module type {type(module)}")


def quantize_model_gptq(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    bits: int,
    symmetric: bool = True,
    apply_mask: bool = False,
    damping: float = 0.01,
) -> None:
    """
    Quantize the model in-place using the GPTQ scheme, using the dataloader calibration data. All weights are stored in FP16.
    If the model is a QuaRot model, the weights-minus-offsets and scales are stored in the QuarotFP16Linear modules. If the model is not a QuaRot model,
    the weights are dequantized and stored in the torch.nn.Linear modules.
    """
    model_adapter.model.eval()

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    for layer_adapter in tqdm(model_adapter.get_layers(), unit="layer", desc="Quantizing layer"):
        layer_adapter.layer.to('cuda')

        # get all activations for the current layer (4 sets)
        qkv_inputs, o_proj_inputs, upgate_inputs, down_proj_inputs, outputs = get_signals(layer_adapter, args, kwargs)

        # compute the 4 Hessians
        H_qkv, H_o_proj, H_upgate, H_down_proj = [
            construct_hessian(X, ignore_masks) for X in [qkv_inputs, o_proj_inputs, upgate_inputs, down_proj_inputs]
        ]

        # get 4 weight matrices (concat as needed)
        W_qkv = torch.cat([w.weight.data for w in layer_adapter.get_attention_inputs()], dim=0)
        W_o_proj = layer_adapter.get_attention_output().weight.data
        W_upgate = torch.cat([w.weight.data for w in layer_adapter.get_mlp_inputs()], dim=0)
        W_down_proj = layer_adapter.get_mlp_output().weight.data

        # 4 calls to quantizer
        Q_qkv, scale_qkv, offset_qkv = quantize_weight_gptq(W_qkv, H_qkv, bits, symmetric=symmetric, percdamp=damping)
        Q_o_proj, scale_o_proj, offset_o_proj = quantize_weight_gptq(
            W_o_proj, H_o_proj, bits, symmetric=symmetric, percdamp=damping
        )
        Q_upgate, scale_upgate, offset_upgate = quantize_weight_gptq(
            W_upgate, H_upgate, bits, symmetric=symmetric, percdamp=damping
        )
        Q_down_proj, scale_down_proj, offset_down_proj = quantize_weight_gptq(
            W_down_proj, H_down_proj, bits, symmetric=symmetric, percdamp=damping
        )

        # set the quantized weights and scales of the attention inputs
        attn_inputs = layer_adapter.get_attention_inputs()
        for module, quantized_weight, scale, offset in zip(
            attn_inputs,
            torch.chunk(Q_qkv, len(attn_inputs), dim=0),
            torch.chunk(scale_qkv, len(attn_inputs), dim=0),
            torch.chunk(offset_qkv, len(attn_inputs), dim=0) if offset_qkv is not None else [None] * len(attn_inputs),
        ):
            set_tensors(module, quantized_weight, scale, offset)

        # set the quantized weights and scales of the attention output
        set_tensors(layer_adapter.get_attention_output(), Q_o_proj, scale_o_proj, offset_o_proj)

        # set the quantized weights and scales of the MLP inputs
        mlp_inputs = layer_adapter.get_mlp_inputs()
        for module, quantized_weight, scale, offset in zip(
            mlp_inputs,
            torch.chunk(Q_upgate, len(mlp_inputs), dim=0),
            torch.chunk(scale_upgate, len(mlp_inputs), dim=0),
            torch.chunk(offset_upgate, len(mlp_inputs), dim=0)
            if offset_upgate is not None
            else [None] * len(mlp_inputs),
        ):
            set_tensors(module, quantized_weight, scale, offset)

        # set the quantized weights and scales of the MLP output
        set_tensors(layer_adapter.get_mlp_output(), Q_down_proj, scale_down_proj, offset_down_proj)

        # outputs of this layer are inputs of the next
        args = [layer_adapter.get_updated_args(output_i, args_i) for output_i, args_i in zip(outputs, args)]

        layer_adapter.layer.to('cpu')
