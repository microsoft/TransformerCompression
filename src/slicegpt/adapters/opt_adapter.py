# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import cast

from torch import FloatTensor, Tensor
from torch.nn import LayerNorm, Linear, Module
from transformers.models.opt.modeling_opt import OPTConfig, OPTDecoderLayer, OPTForCausalLM

from slicegpt.model_adapter import LayerAdapter, ModelAdapter
from slicegpt.modules import CompressedOPTDecoderLayer


class OPTLayerAdapter(LayerAdapter):
    _layer: OPTDecoderLayer | CompressedOPTDecoderLayer

    def __init__(self, layer: OPTDecoderLayer | CompressedOPTDecoderLayer) -> None:
        super().__init__()
        self._layer = layer

    @property
    def raw_layer(self) -> OPTDecoderLayer | CompressedOPTDecoderLayer:
        return self._layer

    def get_first_layernorm(self) -> LayerNorm:
        return self._layer.self_attn_layer_norm

    def get_second_layernorm(self) -> LayerNorm:
        return self._layer.final_layer_norm

    def get_attention_inputs(self) -> list[Linear]:
        return [self._layer.self_attn.q_proj, self._layer.self_attn.k_proj, self._layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self._layer.self_attn.out_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self._layer.fc1]

    def get_mlp_output(self) -> Linear:
        return self._layer.fc2


class OPTModelAdapter(ModelAdapter):
    _model: OPTForCausalLM
    _no_split_module_classes: list[str] = ["OPTDecoderLayer", "CompressedOPTDecoderLayer"]

    def __init__(self, model: OPTForCausalLM) -> None:
        super().__init__()
        self._model = model

    @property
    def raw_model(self) -> Module:
        return self._model

    @property
    def no_split_module_classes(self) -> list[str] | None:
        return OPTModelAdapter._no_split_module_classes

    @property
    def seqlen(self) -> int:
        return cast(OPTConfig, self._model.config).max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return cast(OPTConfig, self._model.config).hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return True

    @property
    def original_layer_type(self) -> type[OPTDecoderLayer]:
        return OPTDecoderLayer

    @property
    def layer_norm_type(self) -> type[LayerNorm]:
        return LayerNorm

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self._model(input_ids=input_ids).logits

    def convert_layer_to_compressible(self, layer: OPTDecoderLayer) -> CompressedOPTDecoderLayer:
        config = cast(OPTConfig, self._model.config)
        compressed_layer = CompressedOPTDecoderLayer(config).to(config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[OPTLayerAdapter]:
        return [
            OPTLayerAdapter(cast(OPTDecoderLayer | CompressedOPTDecoderLayer, layer))
            for layer in self._model.model.decoder.layers
        ]

    def get_raw_layer_at(self, index: int) -> Module:
        return self._model.model.decoder.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self._model.model.decoder.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self._model.model.decoder.embed_tokens, self._model.model.decoder.embed_positions]

    def get_pre_head_layernorm(self) -> LayerNorm:
        pre_head_layernorm = self._model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self._model.lm_head
