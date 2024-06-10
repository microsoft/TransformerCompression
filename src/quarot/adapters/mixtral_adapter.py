# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import FloatTensor, Tensor
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase

from transformers.models.mixtral.modeling_mixtral import MixtralConfig, MixtralDecoderLayer, MixtralForCausalLM, MixtralRMSNorm

from quarot.model_adapter import LayerAdapter, ModelAdapter


class MixtralLayerAdapter(LayerAdapter):
    is_moe = True
    def __init__(self, layer: MixtralDecoderLayer) -> None:
        super().__init__()
        self._layer = layer

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
        return [[expert.w1, expert.w3] for expert in self.layer.block_sparse_moe.experts]

    def get_mlp_output(self) -> Linear:
        return [expert.w2 for expert in self.layer.block_sparse_moe.experts]

    # QuaRot specific.
    def get_v_proj(self) -> Linear:
        return self.layer.self_attn.v_proj


class MixtralModelAdapter(ModelAdapter):
    def __init__(self, model: MixtralForCausalLM) -> None:
        super().__init__()
        self._model = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return MixtralConfig

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
        return MixtralDecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return MixtralRMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return MixtralLayerAdapter

    # QuaRot specific
    @property
    def quarot_layer_type(self) -> type:
        return MixtralLayerAdapter

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

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
        if not model_name.startswith("mistralai/Mixtral"):
            return None

        model = MixtralForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model.config.torch_dtype = dtype

        return MixtralModelAdapter(model)

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
        if not model_name.startswith("mistralai/Mixtral"):
            return None

        class UninitializedMixtralForCausalLM(MixtralForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        config = MixtralConfig.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model = UninitializedMixtralForCausalLM(config)
        model = model.to(dtype=dtype)

        return MixtralModelAdapter(model)
