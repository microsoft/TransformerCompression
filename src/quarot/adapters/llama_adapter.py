# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# https://www.apache.org/licenses/LICENSE-2.0

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import FloatTensor, Tensor, nn
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from quarot.model_adapter import LayerAdapter, ModelAdapter


class QuaRotLlamaAttention(LlamaAttention):
    '''
    QuaRot Llama Attention layer which adds a Hadamard operation before apply_rotary_pos_emb. The rest of the code is the same as the original LlamaAttention.
    '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # TODO: apply qk rotation here for QuaRot

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QuaRotLlamaFlashAttention2(QuaRotLlamaAttention):
    '''
    QuaRot Llama FlashAttention2 layer which adds a Hadamard operation before apply_rotary_pos_emb. The rest of the code is the same as the original LlamaFlashAttention2.
    '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        raise NotImplementedError(
            "QuaRotFlashAttention2 is not implemented yet."
        )  # TODO: copy and modify forward method


class QuaRotLlamaSdpaAttention(QuaRotLlamaAttention):
    '''
    QuaRot Llama SdpaAttention layer which adds a Hadamard operation before apply_rotary_pos_emb. The rest of the code is the same as the original LlamaSdpaAttention.
    '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        raise NotImplementedError("QuaRotSdpaAttention is not implemented yet.")  # TODO: copy and modify forward method


QUAROT_LLAMA_ATTENTION_CLASSES = {
    "eager": QuaRotLlamaAttention,
    "flash_attention_2": QuaRotLlamaFlashAttention2,
    "sdpa": QuaRotLlamaSdpaAttention,
}


class QuaRotLlamaDecoderLayer(LlamaDecoderLayer):
    '''
    QuaRot Llama Decoder Layer which replaces the original LlamaAttention with QuaRotLlamaAttention.
    '''

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__(config)
        self.self_attn = QUAROT_LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)


class LlamaLayerAdapter(LayerAdapter):
    def __init__(self, layer: QuaRotLlamaDecoderLayer) -> None:
        super().__init__()
        self._layer: QuaRotLlamaDecoderLayer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> Module:
        return self.layer.input_layernorm

    def get_second_layernorm(self) -> Module:
        return self.layer.post_attention_layernorm

    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj

    # The below methods are QuaRot specific.
    def get_v_proj(self) -> Linear:
        return self.layer.self_attn.v_proj

    def get_rope_function_name(self) -> str:
        return "apply_rotary_pos_emb"

    def get_self_attn(self) -> Module:
        return self.layer.self_attn


class LlamaModelAdapter(ModelAdapter):
    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self._model: LlamaForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return LlamaConfig

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False

    @property
    def original_layer_type(self) -> type:
        return LlamaDecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return LlamaRMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return LlamaLayerAdapter

    @property
    def quarot_layer_type(self) -> type:
        return QuaRotLlamaDecoderLayer

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def convert_layer_to_quarot(self, layer: Module, layer_idx: int | None) -> Module:
        quarot_layer = self.quarot_layer_type(self.config, layer_idx).to(self.config.torch_dtype)
        quarot_layer.load_state_dict(layer.state_dict(), strict=True)
        return quarot_layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> type:
        pre_head_layernorm = self.model.model.norm
        assert isinstance(pre_head_layernorm, self.original_layer_norm_type)
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Llama-2 doesn't have a pad token by default
        tokenizer.pad_token = tokenizer.eos_token
        self.config.pad_token_id = tokenizer.pad_token_id

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not model_name.startswith("meta-llama/Llama-2"):
            return None

        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model.config.torch_dtype = dtype

        return LlamaModelAdapter(model)

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not model_name.startswith("meta-llama/Llama-2"):
            return None

        class UninitializedLlamaForCausalLM(LlamaForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        config = LlamaConfig.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model = UninitializedLlamaForCausalLM(config)
        model = model.to(dtype=dtype)

        return LlamaModelAdapter(model)
