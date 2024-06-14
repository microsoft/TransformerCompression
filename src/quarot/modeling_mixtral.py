# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
# https://www.apache.org/licenses/LICENSE-2.0

from typing import Optional, Tuple

import torch
from torch import nn
from transformers import Cache, MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
    MixtralFlashAttention2,
    MixtralForCausalLM,
    MixtralSparseMoeBlock,
    apply_rotary_pos_emb,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from slicegpt.modules import RMSN

from .nn import OnlineHadamard, QuarotFP16Linear
from .nn.quantizer import ActQuantizer, DummyActQuantizer, KVQuantizerDequantizer

try:
    from transformers.models.mistral.modeling_mistral import _flash_supports_window_size
except ImportError:
    _flash_supports_window_size = False


class QuarotMixtralConfig(MixtralConfig):
    model_type = "mixtral_quarot"
    groupsize = None
    offset = False


class QuarotMixtralBlockSparseTop2MLP(MixtralBlockSparseTop2MLP):
    def __init__(
        self,
        config: QuarotMixtralConfig,
        act_bits: int = 16,
        act_clip_ratio: float | None = None,
        act_quantile: float | None = None,
        act_groupsize: int | None = None,
        online_had: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.online_had = online_had
        self.w1 = QuarotFP16Linear.like(self.w1, groupsize=config.groupsize, offset=config.offset)
        self.w2 = QuarotFP16Linear.like(self.w2, groupsize=config.groupsize, offset=config.offset)
        self.w3 = QuarotFP16Linear.like(self.w3, groupsize=config.groupsize, offset=config.offset)
        self.online_down_proj_hadamard = OnlineHadamard(self.ffn_dim)
        if act_bits < 16:
            self.input_quantizer = ActQuantizer(
                act_bits, symmetric=True, clip_ratio=act_clip_ratio, quantile=act_quantile, groupsize=act_groupsize
            )
            self.down_proj_input_quantizer = ActQuantizer(
                act_bits, symmetric=True, clip_ratio=act_clip_ratio, quantile=act_quantile, groupsize=act_groupsize
            )
        else:
            self.input_quantizer = DummyActQuantizer()
            self.down_proj_input_quantizer = DummyActQuantizer()

    def forward(self, hidden_states):
        # Quantize inputs to mlp
        hidden_states = self.input_quantizer(hidden_states)

        # Calculate activations
        hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)

        # Apply online hadamard if needed
        if self.online_had:
            hidden_states = self.online_down_proj_hadamard(hidden_states)

        # Quantize inputs to down_proj
        hidden_states = self.down_proj_input_quantizer(hidden_states)

        # Return final activations
        return self.w2(hidden_states)


class QuarotMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(
        self,
        config: QuarotMixtralConfig,
        act_bits: int = 16,
        act_clip_ratio: float = 1.0,
        act_groupsize: int | None = None,
        online_had: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)

        # No change to gate, but experts must be replaced
        self.experts = nn.ModuleList(
            [
                QuarotMixtralBlockSparseTop2MLP(config, act_bits, act_clip_ratio, act_groupsize, online_had)
                for _ in range(config.num_local_experts)
            ]
        )


class QuarotMixtralFlashAttention2(MixtralFlashAttention2):
    def __init__(
        self,
        config: QuarotMixtralConfig,
        act_bits: int = 16,
        act_clip_ratio: float | None = None,
        act_quantile: int | None = None,
        act_groupsize: int | None = None,
        k_bits: int = 16,
        k_clip_ratio: float | None = None,
        k_quantile: float | None = None,
        k_groupsize: int | None = None,
        v_bits: int = 16,
        v_clip_ratio: float | None = None,
        v_quantile: float | None = None,
        v_groupsize: int | None = None,
        online_had=False,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.online_had = online_had
        self.q_proj = QuarotFP16Linear.like(self.q_proj, groupsize=config.groupsize, offset=config.offset)
        self.k_proj = QuarotFP16Linear.like(self.k_proj, groupsize=config.groupsize, offset=config.offset)
        self.v_proj = QuarotFP16Linear.like(self.v_proj, groupsize=config.groupsize, offset=config.offset)
        self.o_proj = QuarotFP16Linear.like(self.o_proj, groupsize=config.groupsize, offset=config.offset)
        self.online_o_proj_hadamard = OnlineHadamard(self.num_heads)
        self.online_k_hadamard = OnlineHadamard(self.head_dim)
        self.online_q_hadamard = OnlineHadamard(self.head_dim)

        if act_bits < 16:
            self.input_quantizer = ActQuantizer(
                act_bits, symmetric=True, clip_ratio=act_clip_ratio, quantile=act_quantile, groupsize=act_groupsize
            )
            self.o_proj_input_quantizer = ActQuantizer(
                act_bits, symmetric=True, clip_ratio=act_clip_ratio, quantile=act_quantile, groupsize=act_groupsize
            )
        else:
            self.input_quantizer = DummyActQuantizer()
            self.o_proj_input_quantizer = DummyActQuantizer()

        if k_bits < 16:
            self.k_quantizer = KVQuantizerDequantizer(
                k_bits, symmetric=False, clip_ratio=k_clip_ratio, quantile=k_quantile, groupsize=k_groupsize
            )
        else:
            self.k_quantizer = lambda x: x

        if v_bits < 16:
            self.v_quantizer = KVQuantizerDequantizer(
                v_bits, symmetric=False, clip_ratio=v_clip_ratio, quantile=v_quantile, groupsize=v_groupsize
            )
        else:
            self.v_quantizer = lambda x: x

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        # QuaRot: quantize hidden states at input of attention
        hidden_states = self.input_quantizer(hidden_states)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)  # QuaRot: remove transpose
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)  # QuaRot: remove transpose
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )  # QuaRot: remove transpose

        kv_seq_len = key_states.shape[-3]  # QuaRot: get sequence length

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, unsqueeze_dim=2
        )  # QuaRot: requires unsqueeze

        # QuaRot: apply online hadamard to queries and keys
        if self.online_had:
            query_states = self.online_q_hadamard(query_states)
            key_states = self.online_k_hadamard(key_states)

        # QuaRot: quantize and dequantize keys and values
        key_states = self.k_quantizer(key_states)
        value_states = self.v_quantizer(value_states)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        torch.repeat_interleave(key_states, dim=2, repeats=self.num_key_value_groups)
        torch.repeat_interleave(value_states, dim=2, repeats=self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        # QuaRot: apply online hadamard if needed
        if self.online_had:
            attn_output = self.online_o_proj_hadamard(attn_output.transpose(-1, -2)).transpose(-1, -2)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        # QuaRot: quantize inputs of output projection
        attn_output = self.o_proj_input_quantizer(attn_output)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QuarotMixtralForCausalLM(MixtralForCausalLM):
    config_class = QuarotMixtralConfig

    def __init__(
        self,
        config: QuarotMixtralConfig = None,
        online_had_mlp: bool = False,
        online_had_attn: bool = False,
        rms_norm: bool = False,
        act_bits: int = 16,
        act_clip_ratio: float | None = None,
        act_quantile: float | None = None,
        act_groupsize: int | None = None,
        k_bits: int = 16,
        k_clip_ratio: float | None = None,
        k_quantile: float | None = None,
        k_groupsize: int | None = None,
        v_bits: int = 16,
        v_clip_ratio: float = 1.0,
        v_quantile: float | None = None,
        v_groupsize: int | None = None,
    ) -> None:
        """
        Args:
            online_had_mlp: Whether to use an online Hadamard at the input of down_proj in the MLP, required if the model has been rotated with QuaRot.
            online_had_attn: Whether to use an online Hadamard at the input of out_proj in attention, required if the model has been rotated with QuaRot.
            rms_norm: Whether the model has rms_norm (instead of layernorm) normalizations. This is True if the base model's layernorms have been fused.
            config: The model config.
        """
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        if rms_norm:
            self.model.norm = RMSN(config.hidden_size, eps=config.rms_norm_eps)

        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = QuarotMixtralFlashAttention2(
                config=config,
                act_bits=act_bits,
                act_clip_ratio=act_clip_ratio,
                act_quantile=act_quantile,
                act_groupsize=act_groupsize,
                k_bits=k_bits,
                k_clip_ratio=k_clip_ratio,
                k_quantile=k_quantile,
                k_groupsize=k_groupsize,
                v_bits=v_bits,
                v_clip_ratio=v_clip_ratio,
                v_quantile=v_quantile,
                v_groupsize=v_groupsize,
                online_had=online_had_attn,
                layer_idx=layer_idx,
            )

            layer.block_sparse_moe = QuarotMixtralSparseMoeBlock(
                config=config,
                act_bits=act_bits,
                act_clip_ratio=act_clip_ratio,
                act_quantile=act_quantile,
                act_groupsize=act_groupsize,
                online_had=online_had_mlp,
            )
            if rms_norm:
                layer.input_layernorm = RMSN(config.hidden_size, eps=config.rms_norm_eps)
                layer.post_attention_layernorm = RMSN(config.hidden_size, eps=config.rms_norm_eps)
