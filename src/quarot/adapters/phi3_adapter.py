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

import torch
from torch import FloatTensor, Tensor
from torch.nn import Linear, Module
from transformers import PretrainedConfig
from transformers.models.phi3.modeling_phi3 import Phi3Config, Phi3DecoderLayer, Phi3ForCausalLM, Phi3RMSNorm

from quarot.model_adapter import LayerAdapter, ModelAdapter


class Phi3LayerAdapter(LayerAdapter):
    def __init__(self, layer: Phi3DecoderLayer) -> None:
        super().__init__()
        self._layer: Phi3DecoderLayer = layer

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
        return [self.layer.self_attn.qkv_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj

    # QuaRot specific. NB diff behaviour to Llama2/3!
    def get_v_proj(self) -> Linear:
        return self.layer.self_attn.qkv_proj


class Phi3ModelAdapter(ModelAdapter):
    def __init__(self, model: Phi3ForCausalLM) -> None:
        super().__init__()
        self._model: Phi3ForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return Phi3Config

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        # if no sliding window, max_position_embeddings is same as original_max_position_embeddings
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False

    @property
    def original_layer_type(self) -> type:
        return Phi3DecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return Phi3RMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return Phi3LayerAdapter

    # QuaRot specific
    @property
    def quarot_layer_type(self) -> type:
        return Phi3DecoderLayer

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

    def get_pre_head_layernorm(self) -> Module:
        pre_head_layernorm = self.model.model.norm
        assert isinstance(pre_head_layernorm, self.original_layer_norm_type)
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
        if not model_name.startswith("microsoft/Phi-3-mini-4k-instruct"):
            return None

        model = Phi3ForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model.config.torch_dtype = dtype

        return Phi3ModelAdapter(model)

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
        if not model_name.startswith("microsoft/Phi-3-mini-4k-instruct"):
            return None

        class UninitializedPhi3ForCausalLM(Phi3ForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        config = Phi3Config.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model = UninitializedPhi3ForCausalLM(config)
        model = model.to(dtype=dtype)

        return Phi3ModelAdapter(model)
