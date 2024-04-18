# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import math

import torch
import tqdm
from fast_hadamard_transform import hadamard_transform

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

from .hadamard_utils import apply_exact_had_to_linear, get_hadK, is_pow2, random_hadamard_matrix
from .model_adapter import ModelAdapter
from .monkeypatch import add_wrapper_after_function_call_in_method
from .quant_utils import ActQuantizer, get_quantizeable_modules


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
        apply_exact_had_to_linear(layer_adapter.get_mlp_output(), had_dim=-1, output=False)
        apply_exact_had_to_linear(layer_adapter.get_v_proj(), had_dim=head_dim, output=True)
        apply_exact_had_to_linear(layer_adapter.get_attention_output(), had_dim=-1, output=False)

    utils.cleanup_memory()


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


def add_online_hadamards(model, fp32_had: bool = False) -> None:
    '''
    Set the online Hadamard matrices for the MLP down-projection and attention out-projection.
    '''
    quantizeable_modules = get_quantizeable_modules(model)
    for name, module in quantizeable_modules.items():
        if 'down_proj' in name:
            had_K, K = get_hadK(model.config.intermediate_size)
            module.online_full_had = True
            module.had_K = had_K
            module.K = K
            module.fp32_had = fp32_had
        elif 'o_proj' in name:
            had_K, K = get_hadK(model.config.num_attention_heads)
            module.online_partial_had = True
            module.had_K = had_K
            module.K = K
            module.had_dim = model.config.hidden_size // model.config.num_attention_heads
            module.fp32_had = fp32_had
