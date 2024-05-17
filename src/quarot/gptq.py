import logging
from typing import Any

import torch
from tqdm import tqdm

from quarot.model_adapter import LayerAdapter, ModelAdapter
from quarot.rtn import calculate_scales, quantize_weight_rtn
from slicegpt.rotate import get_layer0_inputs
from slicegpt.utils import cleanup_memory, map_tensors


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
    TODO.
    """
    num_features, num_columns = W.shape
    orig_dev = W.device
    W = W.cuda()
    H = H.cuda()

    if groupsize is None:
        scale, offset = calculate_scales(W, bits, symmetric=symmetric, clip_weights=clip_weights, vectorized=False)

    # find dead weights and set them to zero, and their Hess value to 1
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # add damping to the diagonal
    H.diagonal().add_(percdamp * torch.mean(torch.diag(H)))

    # calculate the inverse layer Hessian (Cholesky form)
    L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L)
    L_inv_transpose = torch.linalg.cholesky(H_inv, upper=True).to(torch.float16)

    Q = torch.zeros_like(W)
    for block_start_idx in range(0, num_columns, max_blocksize):
        block_end_idx = min(block_start_idx + max_blocksize, num_columns)
        blocksize = block_end_idx - block_start_idx

        Err_block = torch.zeros((W.shape[0], blocksize), dtype=torch.float16).cuda()

        for i in range(blocksize):
            cur_idx = block_start_idx + i

            if groupsize is not None and cur_idx % groupsize == 0:
                scale, offset = calculate_scales(
                    W[:, cur_idx : cur_idx + groupsize],
                    bits,
                    symmetric=symmetric,
                    clip_weights=clip_weights,
                    vectorized=False,
                )

            # store the int-quantized weight column
            Q[:, cur_idx] = quantize_weight_rtn(W[:, cur_idx : cur_idx + 1], scale, offset, bits).flatten()

            # calculate quantization error (between original and dequantized weight column)
            Err_block[:, i] = (W[:, cur_idx] - Q[:, cur_idx] * scale.flatten()) / L_inv_transpose[cur_idx, cur_idx]

            # update the rest of the weights in the block
            W[:, cur_idx:block_end_idx] -= (
                Err_block[:, i : i + 1] * L_inv_transpose[cur_idx : cur_idx + 1, cur_idx:block_end_idx]
            )

        # update the rest of the weights in the tensor
        W[:, block_end_idx:] -= Err_block.matmul(L_inv_transpose[block_start_idx:block_end_idx, block_end_idx:])

    Q = Q.to(orig_dev)
    scale = scale.to(orig_dev)
    offset = offset.to(orig_dev) if offset is not None else None
    return Q, scale, offset


@torch.no_grad()
def construct_hessian(X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None) -> torch.Tensor:
    """
    TODO.
    """
    # Run GC and cleanup GPU memory
    cleanup_memory()
    device = 'cuda'

    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0

        X_batch = X_batch.double().to(device=device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch

    return H


def get_signals(
    layer_adapter: LayerAdapter, layer_args: list[tuple], layer_kwargs: list[dict[str, Any]]
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the layer (not layernormed) and the input to the MLP (pre-layernorm).
    """
    qkv_inputs, o_proj_inputs, upgate_inputs, down_proj_inputs = [], [], [], []
    outputs = []
    device = 'cuda'
    layer_adapter.layer.to(device)

    def make_hook(storage_list):
        def hook_fn(_, args: tuple, _output: Any) -> None:
            storage_list.append(args[0].cpu())

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
        out = out.cpu()
        outputs.append(out)

    [h.remove() for h in hooks]
    return qkv_inputs, o_proj_inputs, upgate_inputs, down_proj_inputs, outputs


@torch.no_grad()
def quantize_model_gptq(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    bits: int,
    symmetric: bool = True,
    apply_mask: bool = True,
) -> None:
    """
    TODO
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

    layers = model_adapter.get_layers()

    logging.info("Quantizing model with GPTQ")
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Quantizing layer")):
        # get all activations for the current layer (4 sets)
        qkv_inputs, o_proj_inputs, upgate_inputs, down_proj_inputs, outputs = get_signals(layer_adapter, args, kwargs)

        # compute the 4 Hessians [construct_hessian(inps, ignore_masks)  # TODO: rescale?]
        H_qkv, H_o_proj, H_upgate, H_down_proj = [
            construct_hessian(X, ignore_masks) for X in [qkv_inputs, o_proj_inputs, upgate_inputs, down_proj_inputs]
        ]

        # get 4 weight matrices (concat as needed)
        W_qkv = torch.cat([w.weight.data for w in layer_adapter.get_attention_inputs()], dim=0)
        W_o_proj = layer_adapter.get_attention_output().weight.data
        W_upgate = torch.cat([w.weight.data for w in layer_adapter.get_mlp_inputs()], dim=0)
        W_down_proj = layer_adapter.get_mlp_output().weight.data

        # 4 calls to quantizer
        Q_qkv, scale_qkv, offset_qkv = quantize_weight_gptq(W_qkv, H_qkv, bits, symmetric=symmetric)
        Q_o_proj, scale_o_proj, offset_o_proj = quantize_weight_gptq(W_o_proj, H_o_proj, bits, symmetric=symmetric)
        Q_upgate, scale_upgate = quantize_weight_gptq(W_upgate, H_upgate, bits, symmetric=symmetric)
        Q_down_proj, scale_down_proj = quantize_weight_gptq(W_down_proj, H_down_proj, bits, symmetric=symmetric)

        # set the quantized qkv weights and scales
        for module in layer_adapter.get_attention_inputs():
            out_features = module.weight.data.shape[0]
            if hasattr(module, "weight_scales"):
                module.weight.data = Q_qkv[:out_features]
                module.weight_scales = scale_qkv[:out_features]
                # TODO: offsets!
            else:
                if symmetric:
                    module.weight.data = Q_qkv[:out_features] * scale_qkv[:out_features]
                else:
                    module.weight.data = (Q_qkv[:out_features] - offset_qkv[:out_features]) * scale_qkv[:out_features]

            Q_qkv = Q_qkv[out_features:]
            scale_qkv = scale_qkv[out_features:]
            if not symmetric:
                offset_qkv = offset_qkv[out_features:]

        # set the quantized o_proj weights and scales
        if hasattr(layer_adapter.get_attention_output(), "weight_scales"):
            layer_adapter.get_attention_output().weight.data = Q_o_proj
            layer_adapter.get_attention_output().weight_scales = scale_o_proj
            # TODO: offsets!
        else:
            if symmetric:
                layer_adapter.get_attention_output().weight.data *= scale_o_proj
            else:
                layer_adapter.get_attention_output().weight.data = (Q_o_proj - offset_o_proj) * scale_o_proj

        # set the quantized upgate weights and scales
        for module in layer_adapter.get_mlp_inputs():
            out_features = module.weight.data.shape[0]
            module.weight.data = Q_upgate[:out_features]
            if hasattr(module, "weight_scales"):
                module.weight_scales = scale_upgate[:out_features]
            else:
                module.weight.data *= scale_upgate[:out_features]

            Q_upgate = Q_upgate[out_features:]
            scale_upgate = scale_upgate[out_features:]

        # set the quantized down_proj weights and scales
        layer_adapter.get_mlp_output().weight.data = Q_down_proj
        if hasattr(layer_adapter.get_mlp_output(), "weight_scales"):
            layer_adapter.get_mlp_output().weight_scales = scale_down_proj
        else:
            layer_adapter.get_mlp_output().weight.data *= scale_down_proj

        # inps = outs
        # cleanup_memory()
        # break

    logging.info("Quantizing model with GPTQ done.")
