# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import math

import torch
import tqdm
from fast_hadamard_transform import hadamard_transform

from slicegpt import utils
from slicegpt.config import config
from slicegpt.rotate import rotate_attention_inputs as rotate_attention_inputs_slicegpt
from slicegpt.rotate import rotate_attention_output as rotate_attention_output_slicegpt
from slicegpt.rotate import rotate_embeddings as rotate_embeddings_slicegpt
from slicegpt.rotate import rotate_head as rotate_head_slicegpt
from slicegpt.rotate import rotate_mlp_input as rotate_mlp_input_slicegpt
from slicegpt.rotate import rotate_mlp_output as rotate_mlp_output_slicegpt

from .hadamard_utils import apply_exact_had_to_linear, is_pow2, random_hadamard_matrix
from .model_adapter import LayerAdapter, ModelAdapter
from .monkeypatch import add_wrapper_after_function_call_in_method
from .quant_utils import ActQuantizer


def random_orthogonal_matrix(size, device, seed=17):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device=config.device, seed: int = 17):
    if mode == 'random':
        return random_orthogonal_matrix(size, device, seed=seed)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device, seed=seed)
    else:
        raise ValueError(f'Unknown mode {mode}')


def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation.
    It reshapes X and applies Walsh-Hadamard transform to the last dimension.
    Then, it will multiply the retult by another hadamard matrix.
    '''
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1 / math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.to(X.device).to(X.dtype).reshape(X.shape)


@torch.inference_mode()
def rotate_model(model_adapter: ModelAdapter, rotate_mode: str = "hadamard", seed: int = 17) -> None:
    '''
    Rotate the model using the QuaRot method.
    '''
    model = model_adapter.model
    Q = get_orthogonal_matrix(model.config.hidden_size, rotate_mode, seed=seed)  # Generate Q

    # Work out head_dim, needed for applying Hadamards to o_proj and v_proj in attention.
    head_dim = model_adapter.config.hidden_size // model_adapter.config.num_attention_heads

    rotate_embeddings_slicegpt(model_adapter, Q)  # Rotate embeddings
    rotate_head_slicegpt(model_adapter, Q)  # Rotate head
    utils.cleanup_memory()
    layer_adapters = model_adapter.get_layers()
    for layer_adapter in tqdm.tqdm(layer_adapters, unit="layer", desc="Rotating (slicegpt)"):
        rotate_attention_inputs_slicegpt(layer_adapter, Q)
        rotate_attention_output_slicegpt(layer_adapter, Q)
        rotate_mlp_input_slicegpt(layer_adapter, Q)
        rotate_mlp_output_slicegpt(layer_adapter, Q)
        apply_hadamard_to_mlp_output(layer_adapter)
        apply_hadamards_to_ov_proj(layer_adapter, head_dim)


def apply_hadamard_to_mlp_output(layer_adapter: LayerAdapter) -> None:
    # Apply Hadamard to the input of the MLP's output.
    W = layer_adapter.get_mlp_output()
    apply_exact_had_to_linear(
        W, had_dim=-1, output=False
    )  # apply exact (inverse) hadamard on the weights of mlp output


def apply_hadamards_to_ov_proj(layer_adapter: LayerAdapter, head_dim: int) -> None:
    # Apply Hadamard to the output projection of the self-attention layer.
    v_proj = layer_adapter.get_v_proj()
    o_proj = layer_adapter.get_o_proj()
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [
                -1,
                head_dim,
            ], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1 / math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1 / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = add_wrapper_after_function_call_in_method(
        module, "forward", function_name, functools.partial(QKRotationWrapper, *args, **kwargs)
    )
    setattr(module, attr_name, wrapper)
