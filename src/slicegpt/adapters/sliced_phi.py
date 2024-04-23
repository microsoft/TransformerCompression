import torch
import torch.nn as nn
from transformers.models.phi.modeling_phi import PhiConfig, PhiForCausalLM, PhiModel

from slicegpt.adapters.phi2_adapter import CompressedPhiDecoderLayer, Phi2ModelAdapter
from slicegpt.modules import RMSN
from slicegpt.rotate import slice_rotated_model
from slicegpt.slicing_scheduler import SlicingScheduler


class SlicedPhi2Config(PhiConfig):
    model_type = "sliced_phi2"
    is_composition = True

    def __init__(self, sparsity=0.1, new_hidden_size=1024, **kwargs):
        self.sparsity = sparsity
        self.new_hidden_size = new_hidden_size
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, config_path, sparsity, new_hidden_size):
        return super().from_pretrained(config_path, sparsity, new_hidden_size)


class SlicedPhi(PhiModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [
                CompressedPhiDecoderLayer(config, layer_idx, replace_layernorm=True)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.final_layernorm = RMSN(config.hidden_size)


class SlicedPhiForCausalLM(PhiForCausalLM):
    def __init__(
        self,
        config,
        scheduler: SlicingScheduler | None = None,
        sparsity: float = 0.0,
        new_hidden_size: int = 1024,
        *model_args,
        **kwargs,
    ):
        super().__init__(config)
        self.model = SlicedPhi(config)
        self.model_adapter = Phi2ModelAdapter(self)

        if scheduler:
            self.update_dims(scheduler)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        scheduler: SlicingScheduler | None,
        sparsity: float,
        new_hidden_size: int,
        config_path: str,
        *model_args,
        **kwargs,
    ):
        """Overrides the from_pretrained method to accept the scheduler and returns the sliced model"""
        config = SlicedPhi2Config.from_pretrained(config_path, sparsity, new_hidden_size)
        model = super().from_pretrained(pretrained_model_name_or_path, config=config)
        model.load_state_dict(model.state_dict())
        return model

    def update_dims(self, scheduler: SlicingScheduler) -> None:
        layers = self.model_adapter.get_layers()

        hidden_size = self.model_adapter.hidden_size
        for layer_adapter in layers:
            if not self.model_adapter.parallel_blocks:
                layer_adapter.layer.mlp_shortcut_Q = torch.nn.Parameter(
                    torch.zeros(hidden_size, hidden_size).to(dtype=torch.float16).contiguous()
                )
            layer_adapter.layer.attn_shortcut_Q = torch.nn.Parameter(
                torch.zeros(hidden_size, hidden_size).to(dtype=torch.float16).contiguous()
            )

        slice_rotated_model(self.model_adapter, scheduler)
