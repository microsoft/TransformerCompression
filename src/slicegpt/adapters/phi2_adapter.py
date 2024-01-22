# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi/modeling_phi.py
# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# License updated to MIT license since 7e10f3e in https://huggingface.co/microsoft/phi-2/blob/main/LICENSE

from typing import cast

from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import LayerNorm, Linear, Module
from transformers import PretrainedConfig
from transformers.models.phi.modeling_phi import PhiConfig, PhiDecoderLayer, PhiForCausalLM

from slicegpt.model_adapter import LayerAdapter, ModelAdapter


class CompressiblePhiDecoderLayer(PhiDecoderLayer):
    """
    This class simulates the PhiDecoderlayer class from PhiModel (PhiForCausalLM)
    https://huggingface.co/microsoft/phi-2/blob/main/modeling_phi.py
    but with the addition of a shortcut_Q attribute. This attribute is used to rotate the residual tensors.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        past_key_value: tuple[Tensor] | None = None,
    ) -> tuple:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))

        if self.attn_shortcut_Q is not None:
            rotated_residual = matmul(residual, self.attn_shortcut_Q)
            hidden_states = attn_outputs + feed_forward_hidden_states + rotated_residual
        else:
            hidden_states = attn_outputs + feed_forward_hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Phi2LayerAdapter(LayerAdapter):
    def __init__(self, layer: PhiDecoderLayer) -> None:
        super().__init__()
        self._layer: PhiDecoderLayer = layer

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
        return self._layer.input_layernorm

    def get_second_layernorm(self) -> Module:
        return None

    def get_attention_inputs(self) -> list[Linear]:
        return [self._layer.self_attn.q_proj, self._layer.self_attn.k_proj, self._layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self._layer.self_attn.dense

    def get_mlp_inputs(self) -> list[Linear]:
        return [self._layer.mlp.fc1]

    def get_mlp_output(self) -> Linear:
        return self._layer.mlp.fc2


class Phi2ModelAdapter(ModelAdapter):
    def __init__(self, model: PhiForCausalLM) -> None:
        super().__init__()
        self._model: PhiForCausalLM = model
        self._config_type: 'type' = PhiConfig
        self._layer_adapter_type: 'type' = Phi2LayerAdapter
        self._layer_type: 'type' = PhiDecoderLayer
        self._compressible_layer_type: 'type' = CompressiblePhiDecoderLayer
        self._layer_norm_type: 'type' = LayerNorm

    @property
    def parallel_blocks(self) -> bool:
        return True

    @property
    def _config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def model(self) -> Module:
        return self._model

    @property
    def no_split_module_classes(self) -> list[str]:
        return [self._layer_type.__name__, self._compressible_layer_type.__name__]

    @property
    def seqlen(self) -> int:
        return self._config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self._config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return True

    @property
    def original_layer_type(self) -> 'type':
        return self._layer_type

    @property
    def original_layer_norm_type(self) -> 'type':
        return self._layer_norm_type

    @property
    def use_cache(self) -> bool:
        return self._config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self._config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self._model(input_ids=input_ids).logits

    def convert_layer_to_compressible(self, layer: Module, layer_idx: int | None) -> Module:
        compressed_layer = self._compressible_layer_type(cast(self._config_type, self._config), layer_idx).to(
            self._config.torch_dtype
        )
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self._layer_adapter_type(layer) for layer in self._model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self._model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self._model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self._model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> 'type':
        pre_head_layernorm = self._model.model.final_layernorm
        assert pre_head_layernorm is not None
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self._model.lm_head