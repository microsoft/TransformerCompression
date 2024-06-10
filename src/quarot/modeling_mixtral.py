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


class QuarotMixtralConfig(MixtralConfig):
    model_type = "mixtral_quarot"
    groupsize = None
    offset = False


class QuarotMixtralBlockSparseTop2MLP(MixtralBlockSparseTop2MLP):
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
        self.online_had = online_had
        self.w1 = QuarotFP16Linear.like(self.w1, groupsize=config.groupsize, offset=config.offset)
        self.w2 = QuarotFP16Linear.like(self.w2, groupsize=config.groupsize, offset=config.offset)
        self.w3 = QuarotFP16Linear.like(self.w3, groupsize=config.groupsize, offset=config.offset)
        self.online_down_proj_hadamard = OnlineHadamard(self.ffn_dim)
        if act_bits < 16:
            self.input_quantizer = ActQuantizer(
                act_bits, symmetric=True, clip_ratio=act_clip_ratio, groupsize=act_groupsize
            )
            self.down_proj_input_quantizer = ActQuantizer(
                act_bits, symmetric=True, clip_ratio=act_clip_ratio, groupsize=act_groupsize
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
                for _ in range(config.num_experts)
            ]
        )


class QuarotMixtralFlashAttention2(MixtralFlashAttention2):
    def __init__(
        self,
        config: QuarotMixtralConfig,
        act_bits: int = 16,
        act_clip_ratio: float = 1.0,
        act_groupsize: int | None = None,
        k_bits: int = 16,
        k_clip_ratio: float = 1.0,
        k_groupsize: int | None = None,
        v_bits: int = 16,
        v_clip_ratio: float = 1.0,
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
                act_bits, symmetric=True, clip_ratio=act_clip_ratio, groupsize=act_groupsize
            )
            self.o_proj_input_quantizer = ActQuantizer(
                act_bits, symmetric=True, clip_ratio=act_clip_ratio, groupsize=act_groupsize
            )
        else:
            self.input_quantizer = DummyActQuantizer()
            self.o_proj_input_quantizer = DummyActQuantizer()

        if k_bits < 16:
            self.k_quantizer = KVQuantizerDequantizer(
                k_bits, symmetric=False, clip_ratio=k_clip_ratio, groupsize=k_groupsize
            )
        else:
            self.k_quantizer = lambda x: x

        if v_bits < 16:
            self.v_quantizer = KVQuantizerDequantizer(
                v_bits, symmetric=False, clip_ratio=v_clip_ratio, groupsize=v_groupsize
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

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)  # QuaRot: remove transpose
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)  # QuaRot: remove transpose
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )  # QuaRot: remove transpose

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)


class QuarotMixtralForCausalLM(MixtralForCausalLM):
    def __init__(
        self,
        online_had_mlp: bool = False,
        online_had_attn: bool = False,
        rms_norm: bool = False,
        act_bits: int = 16,
        act_clip_ratio: float = 1.0,
        act_groupsize: int | None = None,
        k_bits: int = 16,
        k_clip_ratio: float = 1.0,
        k_groupsize: int | None = None,
        v_bits: int = 16,
        v_clip_ratio: float = 1.0,
        v_groupsize: int | None = None,
        config: QuarotMixtralConfig = None,
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
            layer.block_sparse_moe = QuarotMixtralSparseMoeBlock(
                config=config,
                act_bits=act_bits,
                act_clip_ratio=act_clip_ratio,
                act_groupsize=act_groupsize,
                online_had=online_had_mlp,
            )
            if rms_norm:
                layer.input_layernorm = RMSN(config.hidden_size, eps=config.rms_norm_eps)
                layer.post_attention_layernorm = RMSN(config.hidden_size, eps=config.rms_norm_eps)
