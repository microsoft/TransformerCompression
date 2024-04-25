# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# https://www.apache.org/licenses/LICENSE-2.0

from typing import Optional, Tuple

import torch
from transformers import Cache, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaMLP,
    apply_rotary_pos_emb,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from quarot.nn import DummyActQuantizer, OnlineHadamard, QuarotFP16Linear
from slicegpt.modules import RMSN

ALL_LAYERNORM_LAYERS.append(RMSN)


class QuarotLlamaConfig(LlamaConfig):
    model_type = "llama_quarot"


class QuarotLlamaMLP(LlamaMLP):
    def __init__(self, config, act_quantization=False, online_had=False, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.act_quantization = act_quantization
        self.online_had = online_had
        self.input_quantizer = DummyActQuantizer()
        self.up_proj = QuarotFP16Linear.like(self.up_proj)
        self.gate_proj = QuarotFP16Linear.like(self.gate_proj)
        self.down_proj = QuarotFP16Linear.like(self.down_proj)
        self.online_down_proj_hadamard = OnlineHadamard(self.intermediate_size)
        self.down_proj_input_quantizer = DummyActQuantizer()

    def forward(self, x):
        # Quantize inputs to mlp
        x = self.input_quantizer(x)

        # Calculate activations
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        # Apply online hadamard if needed
        if self.online_had:
            x = self.online_down_proj_hadamard(x)

        # Quantize inputs to down_proj
        x = self.down_proj_input_quantizer(x)

        # Return final activations
        return self.down_proj(x)


class QuarotFP16LlamaFlashAttention2(LlamaFlashAttention2):
    def __init__(self, config, act_quantization=False, online_had=False, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.input_quantizer = DummyActQuantizer()
        self.act_quantization = act_quantization
        self.online_had = online_had
        self.q_proj = QuarotFP16Linear.like(self.q_proj)
        self.k_proj = QuarotFP16Linear.like(self.k_proj)
        self.v_proj = QuarotFP16Linear.like(self.v_proj)
        self.o_proj = QuarotFP16Linear.like(self.o_proj)
        self.online_o_proj_hadamard = OnlineHadamard(self.num_heads)
        self.o_proj_input_quantizer = DummyActQuantizer()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        # QuaRot: quantize hidden states at input of attention
        hidden_states = self.input_quantizer(hidden_states)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        kv_seq_len = key_states.shape[1]
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, unsqueeze_dim=2
        )

        past_key_value = getattr(self, "past_key_value", past_key_value)
        assert past_key_value is not None
        # sin and cos are specific to RoPE models; position_ids needed for the static cache

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "attention_mask": attention_mask}
        cache_out = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        dropout_rate = self.attention_dropout if self.training else 0.0

        assert self.is_causal

        if isinstance(cache_out, tuple):
            key_states, value_states = cache_out
            attn_output = self._flash_attention_forward(
                query_states, key_states, value_states, query_length=q_len, attention_mask=attention_mask
            )
        else:
            attn_output = cache_out(query_states)

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


class QuarotLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, online_had_mlp=False, online_had_attn=False, config=None):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        self.model.norm = RMSN(config.hidden_size, eps=config.rms_norm_eps)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = QuarotFP16LlamaFlashAttention2(
                online_had=online_had_attn, config=config, layer_idx=layer_idx
            )
            layer.input_layernorm = RMSN(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = RMSN(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = QuarotLlamaMLP(online_had=online_had_mlp, config=config)
