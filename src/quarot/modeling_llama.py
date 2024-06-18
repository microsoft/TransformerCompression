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

from slicegpt.modules import RMSN

from .nn import OnlineHadamard, QuarotFP16Linear
from .nn.quantizer import ActQuantizer, DummyActQuantizer, KVQuantizerDequantizer

ALL_LAYERNORM_LAYERS.append(RMSN)


class QuarotLlamaConfig(LlamaConfig):
    model_type = "llama_quarot"
    groupsize = None
    offset = False

    # weight quantization args
    w_bits: int = 16
    w_asym: bool = False
    w_groupsize: int | None = None

    # activation quantization args
    act_bits: int = 16
    act_asym: bool = False
    act_groupsize: int | None = None
    act_clip_ratio: float | None = None

    # key and value quantization args
    k_bits: int = 16
    k_clip_ratio: float | None = None
    k_groupsize: int | None = None
    v_bits: int = 16
    v_clip_ratio: float | None = None
    v_groupsize: int | None = None
    

class QuarotLlamaMLP(LlamaMLP):
    def __init__(
        self,
        config: QuarotLlamaConfig,
        online_had: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.online_had = online_had
        self.up_proj = QuarotFP16Linear.like(self.up_proj, bits=config.w_bits, groupsize=config.w_groupsize, offset=config.w_asym)
        self.gate_proj = QuarotFP16Linear.like(self.gate_proj, bits=config.w_bits, groupsize=config.w_groupsize, offset=config.w_asym)
        self.down_proj = QuarotFP16Linear.like(self.down_proj, bits=config.w_bits, groupsize=config.w_groupsize, offset=config.w_asym)
        self.online_down_proj_hadamard = OnlineHadamard(self.intermediate_size)
        if config.act_bits < 16:
            self.input_quantizer = ActQuantizer(
                config.act_bits,
                symmetric=not config.act_asym,
                clip_ratio=config.act_clip_ratio,
                groupsize=config.act_groupsize,
            )
            self.down_proj_input_quantizer = ActQuantizer(
                config.act_bits,
                symmetric=not config.act_asym,
                clip_ratio=config.act_clip_ratio,
                groupsize=config.act_groupsize,
            )
        else:
            self.input_quantizer = DummyActQuantizer()
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
    def __init__(
        self,
        config: QuarotLlamaConfig,
        online_had=False,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.online_had = online_had
        self.q_proj = QuarotFP16Linear.like(self.q_proj, bits=config.w_bits, groupsize=config.w_groupsize, offset=config.w_asym)
        self.k_proj = QuarotFP16Linear.like(self.k_proj, bits=config.w_bits, groupsize=config.w_groupsize, offset=config.w_asym)
        self.v_proj = QuarotFP16Linear.like(self.v_proj, bits=config.w_bits, groupsize=config.w_groupsize, offset=config.w_asym)
        self.o_proj = QuarotFP16Linear.like(self.o_proj, bits=config.w_bits, groupsize=config.w_groupsize, offset=config.w_asym)
        self.online_o_proj_hadamard = OnlineHadamard(self.num_heads)
        self.online_k_hadamard = OnlineHadamard(self.head_dim)
        self.online_q_hadamard = OnlineHadamard(self.head_dim)

        if config.act_bits < 16:
            self.input_quantizer = ActQuantizer(
                config.act_bits,
                symmetric=not config.act_asym,
                clip_ratio=config.act_clip_ratio,
                groupsize=config.act_groupsize,
            )
            self.o_proj_input_quantizer = ActQuantizer(
                config.act_bits,
                symmetric=not config.act_asym,
                clip_ratio=config.act_clip_ratio,
                groupsize=config.act_groupsize,
            )
        else:
            self.input_quantizer = DummyActQuantizer()
            self.o_proj_input_quantizer = DummyActQuantizer()

        if config.k_bits < 16:
            self.k_quantizer = KVQuantizerDequantizer(
                config.k_bits, symmetric=False, clip_ratio=config.k_clip_ratio, groupsize=config.k_groupsize
            )
        else:
            self.k_quantizer = lambda x: x

        if config.v_bits < 16:
            self.v_quantizer = KVQuantizerDequantizer(
                config.v_bits, symmetric=False, clip_ratio=config.v_clip_ratio, groupsize=config.v_groupsize
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
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)  # QuaRot: remove transpose
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)  # QuaRot: remove transpose
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )  # QuaRot: remove transpose

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )  # QuaRot: requires unsqueeze

        # QuaRot: apply online hadamard to queries and keys
        if self.online_had:
            query_states = self.online_q_hadamard(query_states)
            key_states = self.online_k_hadamard(key_states)

        # QuaRot: quantize and dequantize keys and values
        key_states = self.k_quantizer(key_states)
        value_states = self.v_quantizer(value_states)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "attention_mask": attention_mask}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, query_length=q_len, attention_mask=attention_mask
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


class QuarotLlamaForCausalLM(LlamaForCausalLM):
    def __init__(
        self,
        config: QuarotLlamaConfig = None,
        online_had_mlp: bool = False,
        online_had_attn: bool = False,
        rms_norm: bool = False,
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
            layer.self_attn = QuarotFP16LlamaFlashAttention2(
                config=config,
                online_had=online_had_attn,
                layer_idx=layer_idx,
            )
            layer.mlp = QuarotLlamaMLP(
                config=config,
                online_had=online_had_mlp,
            )
            if rms_norm:
                layer.input_layernorm = RMSN(config.hidden_size, eps=config.rms_norm_eps)
                layer.post_attention_layernorm = RMSN(config.hidden_size, eps=config.rms_norm_eps)
