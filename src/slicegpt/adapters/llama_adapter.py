# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import cast

from torch import FloatTensor, Tensor
from torch.nn import Linear, Module
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm

from slicegpt.model_adapter import LayerAdapter, ModelAdapter
from slicegpt.modules import CompressedLlamaDecoderLayer


class LlamaLayerAdapter(LayerAdapter):
    _layer: LlamaDecoderLayer | CompressedLlamaDecoderLayer

    def __init__(self, layer: LlamaDecoderLayer | CompressedLlamaDecoderLayer) -> None:
        super().__init__()
        self._layer = layer

    @property
    def raw_layer(self) -> LlamaDecoderLayer | CompressedLlamaDecoderLayer:
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
    _model: LlamaForCausalLM
    _no_split_module_classes: list[str] = ["LlamaDecoderLayer", "CompressedLlamaDecoderLayer"]

    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self._model = model

    @property
    def raw_model(self) -> Module:
        return self._model

    @property
    def no_split_module_classes(self) -> list[str] | None:
        return LlamaModelAdapter._no_split_module_classes

    @property
    def seqlen(self) -> int:
        return cast(LlamaConfig, self._model.config).max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return cast(LlamaConfig, self._model.config).hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False

    @property
    def original_layer_type(self) -> type[LlamaDecoderLayer]:
        return LlamaDecoderLayer

    @property
    def original_layer_norm_type(self) -> type[LlamaRMSNorm]:
        return LlamaRMSNorm

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self._model(input_ids=input_ids).logits

    def convert_layer_to_compressible(self, layer: LlamaDecoderLayer) -> CompressedLlamaDecoderLayer:
        config = cast(LlamaConfig, self._model.config)
        compressed_layer = CompressedLlamaDecoderLayer(config).to(config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[LlamaLayerAdapter]:
        return [
            LlamaLayerAdapter(cast(LlamaDecoderLayer | CompressedLlamaDecoderLayer, layer))
            for layer in self._model.model.layers
        ]

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
