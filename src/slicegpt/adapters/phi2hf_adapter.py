import sys
from typing import Optional, Union, cast

from pyreporoot import project_root
from torch import BoolTensor, FloatTensor, Tensor, matmul
from torch.nn import LayerNorm, Linear, Module

sys.path.append(project_root(__file__, root_files="pyproject.toml"))
from phi2_hf.configuration_phi import PhiConfig
from phi2_hf.modeling_phi import InferenceParams, ParallelBlock, PhiForCausalLM
from slicegpt.model_adapter import LayerAdapter, ModelAdapter


class CompressibleParallelBlock(ParallelBlock):
    """
    This class simulates the ParallelBlock class from PhiModel (PhiForCausalLM)
    https://huggingface.co/microsoft/phi-2/blob/main/modeling_phi.py
    but with the addition of a shortcut_Q attribute. This attribute is used to rotate the residual tensors.
    """

    def forward(
        self,
        hidden_states: FloatTensor,
        past_key_values: Optional[Union[FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[BoolTensor] = None,
        **kwargs,
    ) -> FloatTensor:
        residual = hidden_states
        hidden_states = self.ln(hidden_states)  # ln = nn.LayerNorm

        attn_outputs = self.mixer(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        if isinstance(attn_outputs, tuple):
            attn_outputs = attn_outputs[0]

        attn_outputs = self.resid_dropout(attn_outputs)  # resid_dropout = nn.Dropout
        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))

        if self.attn_shortcut_Q is not None:
            rotated_residual = matmul(residual, self.attn_shortcut_Q)
            hidden_states = attn_outputs + feed_forward_hidden_states + rotated_residual
        else:
            hidden_states = attn_outputs + feed_forward_hidden_states + residual

        return hidden_states


class Phi2HFLayerAdapter(LayerAdapter):
    def __init__(self, layer: ParallelBlock) -> None:
        super().__init__()
        self._layer: ParallelBlock = layer

    @property
    def layer(self) -> ParallelBlock:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> LayerNorm:
        return self._layer.ln

    def get_second_layernorm(self) -> LayerNorm:
        raise NotImplementedError("Phi-2-HF does not have a post-attention layer norm")

    def get_attention_inputs(self) -> list[Linear]:
        return [self._layer.mixer.Wqkv]

    def get_attention_output(self) -> Linear:
        return self._layer.mixer.out_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self._layer.mlp.fc1]

    def get_mlp_output(self) -> Linear:
        return self._layer.mlp.fc2


class Phi2HFModelAdapter(ModelAdapter):
    parallel_blocks = True

    def __init__(self, model: PhiForCausalLM) -> None:
        super().__init__()
        self._model: PhiForCausalLM = model

    @property
    def _config(self) -> PhiConfig:
        return cast(PhiConfig, self._model.config)

    @property
    def model(self) -> Module:
        return self._model

    @property
    def no_split_module_classes(self) -> list[str]:
        return ["ParallelBlock", "CompressibleParallelBlock"]

    @property
    def seqlen(self) -> int:
        return self._config.n_positions

    @property
    def hidden_size(self) -> int:
        return self._config.n_embd

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return True

    @property
    def original_layer_type(self) -> type[ParallelBlock]:
        return ParallelBlock

    @property
    def original_layer_norm_type(self) -> type[LayerNorm]:
        return LayerNorm

    @property
    def use_cache(self) -> bool:
        return False  # managed internally in phi-2

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        pass  # raise NotImplementedError("cache managed internally in phi-2")

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self._model(input_ids=input_ids).logits

    def convert_layer_to_compressible(self, layer: ParallelBlock) -> CompressibleParallelBlock:
        compressed_layer = CompressibleParallelBlock(self._config).to(self._config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[Phi2HFLayerAdapter]:
        return [Phi2HFLayerAdapter(cast(ParallelBlock, layer)) for layer in self._model.transformer.h]

    def get_raw_layer_at(self, index: int) -> Module:
        return self._model.transformer.h[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self._model.transformer.h[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self._model.transformer.embd.wte]

    def get_pre_head_layernorm(self) -> LayerNorm:
        return self._model.lm_head.ln

    def get_lm_head(self) -> Linear:
        return self._model.lm_head.linear
