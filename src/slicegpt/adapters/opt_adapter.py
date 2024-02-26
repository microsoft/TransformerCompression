# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
from typing import cast

import torch
from torch import FloatTensor, Tensor, matmul
from torch.nn import LayerNorm, Linear, Module
from torch.nn.functional import dropout
from transformers import PretrainedConfig
from transformers.models.opt.modeling_opt import OPTConfig, OPTDecoderLayer, OPTForCausalLM

from slicegpt.model_adapter import LayerAdapter, ModelAdapter


class CompressedOPTDecoderLayer(OPTDecoderLayer):
    """
    This class simulates the OPTDecoderLayer class from transformers
    but with the addition of a shortcut_Q attributes.
    We also support the input rotation and mean subtraction in this class (if needed).
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        layer_head_mask: Tensor | None = None,
        past_key_value: tuple[Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
    ) -> tuple:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)
        if self.attn_shortcut_Q is not None:
            rotated_shortcut = matmul(residual, self.attn_shortcut_Q)
            hidden_states = rotated_shortcut + hidden_states
        else:
            hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            raise NotImplementedError("Layer norm after attention is not implemented yet!")

        # Fully Connected
        hidden_states_shape = list(hidden_states.shape)
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states_shape[-1] = self.fc2.out_features  # to make sure the shape is correct

        if self.mlp_shortcut_Q is not None:
            rotated_shortcut = matmul(residual, self.mlp_shortcut_Q)
            hidden_states = rotated_shortcut.view(hidden_states_shape) + hidden_states.view(hidden_states_shape)
        else:
            hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class OPTLayerAdapter(LayerAdapter):
    def __init__(self, layer: OPTDecoderLayer) -> None:
        super().__init__()
        self._layer: OPTDecoderLayer = layer

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
        return self.layer.self_attn_layer_norm

    def get_second_layernorm(self) -> Module:
        return self.layer.final_layer_norm

    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.out_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.fc1]

    def get_mlp_output(self) -> Linear:
        return self.layer.fc2


class OPTModelAdapter(ModelAdapter):
    def __init__(self, model: OPTForCausalLM) -> None:
        super().__init__()
        self._model: OPTForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return OPTConfig

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
        return True

    @property
    def original_layer_type(self) -> type:
        return OPTDecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return LayerNorm

    @property
    def layer_adapter_type(self) -> type:
        return OPTLayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedOPTDecoderLayer

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        compressed_layer = self.compressed_layer_type(cast(self.config_type, self.config)).to(self.config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.decoder.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.decoder.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.decoder.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.decoder.embed_tokens, self.model.model.decoder.embed_positions]

    def get_pre_head_layernorm(self) -> type:
        pre_head_layernorm = self.model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

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
        if not model_name.startswith("facebook/opt"):
            return None

        model = OPTForCausalLM.from_pretrained(model_path, torch_dtype=dtype, local_files_only=local_files_only)
        model.config.torch_dtype = dtype

        return OPTModelAdapter(model)

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
        if not model_name.startswith("facebook/opt"):
            return None

        class UninitializedOPTForCausalLM(OPTForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        config = OPTConfig.from_pretrained(model_path, torch_dtype=dtype, local_files_only=local_files_only)
        model = UninitializedOPTForCausalLM(config)
        model = model.to(dtype=dtype)

        return OPTModelAdapter(model)
