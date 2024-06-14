# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" PyTorch Phi-3 model."""

from typing import Optional, Tuple

import torch
from transformers import Cache, Phi3Config
from transformers.models.phi3.modeling_phi3 import (
    Phi3FlashAttention2,
    Phi3ForCausalLM,
    Phi3MLP,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from slicegpt.modules import RMSN

from .nn import OnlineHadamard, QuarotFP16Linear
from .nn.quantizer import ActQuantizer, DummyActQuantizer, KVQuantizerDequantizer

ALL_LAYERNORM_LAYERS.append(RMSN)

try:
    from transformers.models.phi3.modeling_phi3 import _flash_supports_window_size
except ImportError:
    _flash_supports_window_size = False


class QuarotPhi3Config(Phi3Config):
    model_type = "phi3_quarot"
    groupsize = None
    offset = False


class QuarotPhi3MLP(Phi3MLP):
    def __init__(
        self,
        config: QuarotPhi3Config,
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
        self.gate_up_proj = QuarotFP16Linear.like(self.gate_up_proj, groupsize=config.groupsize, offset=config.offset)
        self.down_proj = QuarotFP16Linear.like(self.down_proj, groupsize=config.groupsize, offset=config.offset)
        self.online_down_proj_hadamard = OnlineHadamard(config.intermediate_size)
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

    def forward(self, x):
        # Quantize inputs to mlp
        x = self.input_quantizer(x)

        # Calculate activations
        x = self.gate_up_proj(x)
        gate, x = x.chunk(2, dim=-1)
        x = x * self.activation_fn(gate)

        # Apply online hadamard if needed
        if self.online_had:
            x = self.online_down_proj_hadamard(x)

        # Quantize inputs to down_proj
        x = self.down_proj_input_quantizer(x)

        # Return final activations
        return self.down_proj(x)


class QuarotFP16Phi3FlashAttention2(Phi3FlashAttention2):
    def __init__(
        self,
        config: QuarotPhi3Config,
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
        online_had=False,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.online_had = online_had
        self.qkv_proj = QuarotFP16Linear.like(self.qkv_proj, groupsize=config.groupsize, offset=config.offset)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Phi3FlashAttention2 attention does not support output_attentions

        if not _flash_supports_window_size:
            raise ValueError("The current flash attention version does not support sliding window attention.")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        # QuaRot: quantize hidden states at input of attention
        hidden_states = self.input_quantizer(hidden_states)

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        kv_seq_len = key_states.shape[1]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, unsqueeze_dim=2
        )

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

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_dropout = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32.

        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.qkv_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
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


class QuarotPhi3ForCausalLM(Phi3ForCausalLM):
    config_class = QuarotPhi3Config

    def __init__(
        self,
        config: QuarotPhi3Config = None,
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
            layer.self_attn = QuarotFP16Phi3FlashAttention2(
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
            layer.mlp = QuarotPhi3MLP(
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
