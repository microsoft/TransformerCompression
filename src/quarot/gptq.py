from typing import Any

import torch
from tqdm import tqdm

from slicegpt.rotate import get_layer0_inputs
from slicegpt.utils import cleanup_memory, map_tensors

from .model_adapter import LayerAdapter, ModelAdapter
from .nn.linear import QuarotFP16Linear
from .quant_utils import PackedQuantizedTensor, dequantize
from .rtn import calculate_scales, quantize_weight_rtn


def gptq_quantize_column(i, col_idx, block_end_idx, W, Q, Err_block, L_inv_transpose, scale, offset, bits, symmetric):
    """
    Quantize one column of the weight matrix, W[:, col_idx]

    i indexes the current position in the block.
    """
    # store the int-quantized weight column
    Q[:, col_idx] = quantize_weight_rtn(W[:, col_idx : col_idx + 1], scale, offset, bits).flatten()

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
    optimize_scales: bool = True,
    device='cuda',
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
    W = W.to(device=device, dtype=torch.float32)
    H = H.to(W.device, dtype=torch.float32)

    # deal with group quantization.
    # If groupsize is None, we quantize the entire tensor at once (there's a single scale and offset).
    # otherwise, we collect the scale/offset for each group in a list
    if groupsize is None:
        scale, offset = calculate_scales(W, bits, symmetric=symmetric, search=clip_weights, device=device)
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
                    search=clip_weights,
                    device=device,
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

    # if optimize_scales is True, solve for the optimal scaling factors
    if optimize_scales:
        scale = solve_optimal_scales(W, Q, H, groupsize=groupsize, offset=offset)

    # move back to original device and dtype
    Q = Q.to(orig_dev, dtype=orig_dtype)
    scale = scale.to(orig_dev, dtype=orig_dtype)
    if not symmetric:
        offset = offset.to(orig_dev, dtype=orig_dtype)

    return Q, scale, offset


def solve_optimal_scales(W, W_int, H, groupsize=None, offset=None):
    """
    Given a weight matrix W, and its quantized version W_int, solve for the optimal scaling factors.

    Note that the _optimal_ scaling factors are not necessarily the same as the ones used in the quantization process. To derive this, consider the GPTQ loss function:

      W_recon = (W_int - offset) * scale
      L = trace( [W - W_recon] H [W - W_recon]^T )

    differentiate with respect to scale, set to zero, and solve for scale, being careful of the tiling used in reconstructing across groups.
    """
    assert W.shape == W_int.shape
    assert len(W.shape) == 2
    num_cols = W.shape[1]
    assert H.shape[0] == H.shape[1] == num_cols
    if groupsize is None:
        groupsize = num_cols
    assert num_cols % groupsize == 0
    num_groups = num_cols // groupsize
    scales = []
    if offset is not None:
        W_int = W_int - torch.repeat_interleave(offset, groupsize, dim=1)
    for wh_i, w_int_i in zip(W @ H, W_int):
        rhs = (w_int_i * wh_i).view(num_groups, groupsize).sum(dim=1)
        mat = H * w_int_i * w_int_i[:, None]
        mat = mat.view(num_cols, num_groups, groupsize).sum(dim=2).view(num_cols, num_groups)
        mat = mat.T.view(num_groups, num_groups, groupsize).sum(dim=2)
        scales.append(torch.linalg.solve(mat, rhs))
    scales = torch.stack(scales)
    return scales


class HessianHook:
    """
    A Hook function designed to colect the input to a layer, and compute the Hessian of the layer online.

    Onnce the hook is applied to a module, each batch gets stored in the hook. Between batches, update_hessian()
    is called to compute the Hessian of the layer.
    """

    def __init__(self):
        self.H = None
        self.num_samples = 0
        self.X = None

    def __call__(self, _, args: tuple, _output: Any) -> None:
        if self.X is not None:
            raise ValueError("HessianHook called multiple times without update")
        self.X = args[0]

    def update_hessian(self, ignore_mask: torch.Tensor | None = None) -> None:
        if self.X is None:
            raise ValueError("No input to compute Hessian")

        # remove reference to this match
        X = self.X
        self.X = None

        # maske as needed, count elements
        batch_size, seq_len = X.shape[:2]
        if ignore_mask is not None:
            X[ignore_mask == 0] = 0
            num_elements = ignore_mask.sum()
        else:
            num_elements = batch_size * seq_len

        # compute the Hessian update
        X = X * torch.rsqrt(torch.tensor(num_elements, device=X.device, dtype=X.dtype))
        H_batch = torch.einsum('bld,blc->dc', X, X)
        if self.H is None:
            self.H = H_batch
        else:
            H = H * (self.num_samples / (self.num_samples + num_elements)) + H_batch * (
                num_elements / (self.num_samples + num_elements)
            )
        self.num_samples += num_elements


@torch.no_grad()
def get_hessians_and_output(
    layer_adapter: LayerAdapter,
    layer_args: list[tuple],
    layer_kwargs: list[dict[str, Any]],
    layer_masks: list[torch.Tensor] | None = None,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.

    for each of the 4 components of the layer (QKV, O_proj, upgate, down_proj), we captuire the signal at this point and compute the Hessian.

    Return
    """
    layer_adapter.layer.to(device)
    modules = [
        layer_adapter.get_attention_inputs()[0],
        layer_adapter.get_attention_output(),
        layer_adapter.get_mlp_inputs()[0],  # TODO for MOE, we need a hook on every mlp input and output
        layer_adapter.get_mlp_output(),
    ]
    hooks = [HessianHook() for _ in modules]
    hooks = [m.register_forward_hook(h) for m, h in zip(modules, hooks)]

    # loop over the batches
    outputs = []
    if layer_masks is None:
        layer_masks = [None] * len(layer_args)
    for layer_args_batch, layer_kwargs_batch, mask_batch in zip(layer_args, layer_kwargs, layer_masks):
        layer_args_batch, layer_kwargs_batch = map_tensors([layer_args_batch, layer_kwargs_batch], device=device)
        out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        if isinstance(out, tuple):
            out = out[layer_adapter.hidden_states_output_position]
        outputs.append(out)
        [h.update_hessian(mask_batch) for h in hooks]

    for h in hooks:
        h.remove()

    H_qkv, H_o_proj, H_upgate, H_down_proj = [h.H for h in hooks]
    return H_qkv, H_o_proj, H_upgate, H_down_proj, outputs


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
    assert scale.shape[0] == out_features
    if offset is not None:
        assert offset.shape == scale.shape

    if isinstance(module, QuarotFP16Linear):
        module.weight.data = quantized_weight  # out_features x in_features
        module.weight_scales.data = scale  # out_features x num_groups
        if offset is not None:
            module.offset.data = offset
    elif isinstance(module, torch.nn.Linear):
        module.weight.data = dequantize(quantized_weight, scale, offset)
    else:
        raise ValueError(f"Unsupported module type {type(module)}")


def quantize_model_gptq(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    bits: int,
    symmetric: bool = True,
    apply_mask: bool = False,
    damping: float = 0.01,
    groupsize: int = None,
    optimize_scales: bool = True,
) -> None:
    """
    Quantize the model in-place using the GPTQ scheme, using the dataloader calibration data. All weights are stored in FP16.
    If the model is a QuaRot model, the weights-minus-offsets and scales are stored in the QuarotFP16Linear modules. If the model is not a QuaRot model,
    the weights are dequantized and stored in the torch.nn.Linear modules.

    Arguments:
    - model_adapter: the model adapter to quantize.
    - dataloader: the dataloader to use for calibration.
    - bits: the number of bits to quantize to.
    - symmetric: whether to use symmetric quantization.
    - apply_mask: whether to apply the attention mask to the activations.
    - damping: the damping factor for the Hessian.
    - groupsize: the group size for quantization.
    - optimize_scales: whether to optimize the scaling factors after quantization.
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

        # get all Hessians for the current layer, and layer outputs
        H_qkv, H_o_proj, H_upgate, H_down_proj, outputs = get_hessians_and_output(
            layer_adapter, args, kwargs, ignore_masks
        )

        # get 4 weight matrices (concat as needed)
        W_qkv = torch.cat([w.weight.data for w in layer_adapter.get_attention_inputs()], dim=0)
        W_o_proj = layer_adapter.get_attention_output().weight.data
        W_upgate = torch.cat([w.weight.data for w in layer_adapter.get_mlp_inputs()], dim=0)
        W_down_proj = layer_adapter.get_mlp_output().weight.data

        # 4 calls to quantizer
        kwargs = dict(
            bits=bits, symmetric=symmetric, percdamp=damping, groupsize=groupsize, optimize_scales=optimize_scales
        )
        Q_qkv, scale_qkv, offset_qkv = quantize_weight_gptq(W_qkv, H_qkv, **kwargs)
        Q_o_proj, scale_o_proj, offset_o_proj = quantize_weight_gptq(W_o_proj, H_o_proj, **kwargs)
        Q_upgate, scale_upgate, offset_upgate = quantize_weight_gptq(W_upgate, H_upgate, bits, **kwargs)
        Q_down_proj, scale_down_proj, offset_down_proj = quantize_weight_gptq(W_down_proj, H_down_proj, **kwargs)

        # set the quantized weights and scales of the attention inputs
        # now that we deconcateenate here, with start/end accounting for the different sizes
        # in the case of grouped keys.
        qkv_sizes = [w.weight.shape[0] for w in layer_adapter.get_attention_inputs()]
        qkv_starts = [0] + list(torch.cumsum(torch.tensor(qkv_sizes), 0).numpy())
        qkv_ends = qkv_starts[1:]
        attn_inputs = layer_adapter.get_attention_inputs()
        for module, quantized_weight, scale, offset in zip(
            attn_inputs,
            [Q_qkv[start:end] for start, end in zip(qkv_starts, qkv_ends)],
            [scale_qkv[start:end] for start, end in zip(qkv_starts, qkv_ends)],
            [offset_qkv[start:end] for start, end in zip(qkv_starts, qkv_ends)]
            if offset_qkv is not None
            else [None] * len(attn_inputs),
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
