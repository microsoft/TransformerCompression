import torch
import os
import numpy as np
import random


# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def symmetric_quantize(x, scale, bits):
    if bits >= 16:
        return x
    elif bits == 4:
        q = torch.clamp(torch.round(x / scale), -8, 7)
    elif bits == 8:
        q = torch.clamp(torch.round(x / scale), -128, 127)
    elif bits == 2:
        q = torch.clamp(torch.round(x / scale), -4, 3)
    elif bits == 3:
        q = torch.clamp(torch.round(x / scale), -2, 1)
    elif bits == 1:
        q = torch.clamp(torch.round(x / scale), -1, 0)
    else:
        raise NotImplementedError
    return scale * q


def asymmetric_quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def layer_quantizer(
    layer: torch.nn.Linear,
    quant_ratio: float,
    bit: int,
    mode="input",
    symmetric=False,
    perchannel=True,
):
    """
    Quantize the last few columns (or rows) of the weight matrix of a linear layer

    :param layer: layer to be quantized
    :param quant_ratio: quantization ratio
    :param bit: number of bits
    :param mode: input or output
    :param symmetric: symmetric or asymmetric quantization
    :param perchannel: per channel or per layer quantization
    :return: quantized layer
    """
    if quant_ratio == 0 or bit >= 16:
        return layer

    col_to_quant = int(quant_ratio * layer.weight.shape[-1])
    rows_to_quant = int(quant_ratio * layer.weight.shape[0])

    dev = layer.weight.device
    if quant_ratio == 0 or bit >= 16:
        return layer
    if mode == "input":
        quant_w = layer.weight.data[:, -col_to_quant:]
        int8_w = layer.weight.data[:, :-col_to_quant]
    elif mode == "output":
        quant_w = layer.weight.data[-rows_to_quant:, :]
        int8_w = layer.weight.data[:-rows_to_quant, :]

    if col_to_quant != rows_to_quant:
        quant_w = quant_w.T

    if not (perchannel) and symmetric:
        scale = torch.max(torch.abs(quant_w))
        quant_w = symmetric_quantize(quant_w, scale, bit)
    elif perchannel and symmetric:
        maxq = (2 ** (bit - 1)) - 1
        xmax = quant_w.abs().max(1)[0]
        scale = xmax / maxq
        shape = [-1] + [1] * (len(quant_w.shape) - 1)
        scale = scale.reshape(shape)
        quant_w = symmetric_quantize(quant_w, scale, bit)
    elif perchannel and not (symmetric):

        # quant_w part (using low-precision)
        maxq = (2**bit) - 1
        tmp = torch.zeros(quant_w.shape[0], device=dev)
        xmin = torch.minimum(quant_w.min(1)[0], tmp)
        xmax = torch.maximum(quant_w.max(1)[0], tmp)
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / scale)
        shape = [-1] + [1] * (len(quant_w.shape) - 1)
        scale = scale.reshape(shape)
        zero = zero.reshape(shape)
        quant_w = asymmetric_quantize(quant_w, scale, zero, maxq)

        # int8_w part (using INT-8)
        maxq = (2**8) - 1
        tmp = torch.zeros(int8_w.shape[0], device=dev)
        xmin = torch.minimum(int8_w.min(1)[0], tmp)
        xmax = torch.maximum(int8_w.max(1)[0], tmp)
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / scale)
        shape = [-1] + [1] * (len(int8_w.shape) - 1)
        scale = scale.reshape(shape)
        zero = zero.reshape(shape)
        int8_w = asymmetric_quantize(int8_w, scale, zero, maxq)

    else:
        raise NotImplementedError

    if col_to_quant != rows_to_quant:
        quant_w = quant_w.T

    if mode == "input":
        layer.weight.data[:, -col_to_quant:] = quant_w
        layer.weight.data[:, :-col_to_quant] = int8_w

    elif mode == "output":
        layer.weight.data[-rows_to_quant:, :] = quant_w
        layer.weight.data[:-rows_to_quant, :] = int8_w

    return layer


@torch.no_grad()
def pca_calc(X):
    torch.cuda.empty_cache()
    try:
        X = X.double().cuda()
        H = X.T @ X
    except:
        print("Out of memory, trying to calculate PCA on CPU!")
        X = X.cpu().double()
        H = X.T @ X
        H = H.cuda()
    del X
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).cuda()
    H[diag, diag] += damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    return eig_val, eigen_vec


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.register_buffer("mean_dim", torch.Tensor([mean_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


@torch.no_grad()
def magnitude_pruner(layer: torch.nn.Linear, sparsity_ratio: float, mode="input"):
    """
    Prune the last few columns (or rows) of the weight matrix of a linear layer

    :param layer: layer to be sparsified
    :param sparsity_ratio: sparsity ratio
    :param mode: input or output
    """

    if sparsity_ratio == 0:
        return layer

    col_to_prune = int(sparsity_ratio * layer.weight.shape[-1])
    rows_to_prune = int(sparsity_ratio * layer.weight.shape[0])

    if mode == "input":
        column_norms = torch.norm(layer.weight.data, p=2, dim=0)
        sorted_indices = torch.argsort(column_norms)
        selected_indices = sorted_indices[:col_to_prune]
        layer.weight.data[:, selected_indices] *= 0

        # layer.weight.data[:, -col_to_prune:] *= 0
    elif mode == "output":
        rows_norms = torch.norm(layer.weight.data, p=2, dim=1)
        sorted_indices = torch.argsort(rows_norms)
        selected_indices = sorted_indices[:rows_to_prune]
        layer.weight.data[selected_indices, :] *= 0
        if layer.bias is not None:
            layer.bias.data[selected_indices] *= 0
        # layer.weight.data[-rows_to_prune:, :] *= 0
        # if layer.bias is not None:
        #     layer.bias.data[-rows_to_prune:] *= 0
    else:
        raise NotImplementedError
    return layer


def deeplearn2_cache_dir():
    os.environ["TRANSFORMERS_CACHE"] = "/storage/experiments/saleh"
    os.environ["HF_DATASETS_CACHE"] = "/storage/experiments/saleh"
    os.environ["HF_HOME"] = "/storage/experiments/saleh"
