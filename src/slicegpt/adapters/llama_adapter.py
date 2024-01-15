# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# https://www.apache.org/licenses/LICENSE-2.0
from typing import cast

from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm

from slicegpt.model_adapter import LayerAdapter, ModelAdapter


class CompressibleLlamaDecoderLayer(LlamaDecoderLayer):
    """
    This class simulates the LlamaDecoderLayer class from transformers
    (https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L376)
    but with the addition of a shortcut_Q attribute. This attribute is used to rotate the residual tensors.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_value: tuple[Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        padding_mask: LongTensor | None = None,
    ) -> tuple:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        if self.attn_shortcut_Q is not None:
            rotated_residual = matmul(residual, self.attn_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.mlp_shortcut_Q is not None:
            rotated_residual = matmul(residual, self.mlp_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaLayerAdapter(LayerAdapter):
    def __init__(self, layer: LlamaDecoderLayer) -> None:
        super().__init__()
        self._layer: LlamaDecoderLayer = layer

    @property
    def layer(self) -> LlamaDecoderLayer:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> LlamaRMSNorm:
        return self._layer.input_layernorm

    def get_second_layernorm(self) -> LlamaRMSNorm:
        return self._layer.post_attention_layernorm

    def get_attention_inputs(self) -> list[Linear]:
        return [self._layer.self_attn.q_proj, self._layer.self_attn.k_proj, self._layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self._layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self._layer.mlp.gate_proj, self._layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self._layer.mlp.down_proj


class LlamaModelAdapter(ModelAdapter):
    @property
    def parallel_blocks(self) -> bool:
        return False

    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self._model: LlamaForCausalLM = model

    @property
    def _config(self) -> LlamaConfig:
        return cast(LlamaConfig, self._model.config)

    @property
    def model(self) -> Module:
        return self._model

    @property
    def no_split_module_classes(self) -> list[str]:
        return ["LlamaDecoderLayer", "CompressedLlamaDecoderLayer"]

    @property
    def seqlen(self) -> int:
        return self._config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self._config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False

    @property
    def original_layer_type(self) -> type[LlamaDecoderLayer]:
        return LlamaDecoderLayer

    @property
    def original_layer_norm_type(self) -> type[LlamaRMSNorm]:
        return LlamaRMSNorm

    @property
    def use_cache(self) -> bool:
        return self._config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self._config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self._model(input_ids=input_ids).logits

    def convert_layer_to_compressible(
        self, layer: LlamaDecoderLayer, layer_idx: int | None
    ) -> CompressibleLlamaDecoderLayer:
        compressed_layer = CompressibleLlamaDecoderLayer(self._config, layer_idx).to(self._config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[LlamaLayerAdapter]:
        return [LlamaLayerAdapter(cast(LlamaDecoderLayer, layer)) for layer in self._model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self._model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self._model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self._model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> LlamaRMSNorm:
        pre_head_layernorm = self._model.model.norm
        assert isinstance(pre_head_layernorm, LlamaRMSNorm)
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self._model.lm_head
