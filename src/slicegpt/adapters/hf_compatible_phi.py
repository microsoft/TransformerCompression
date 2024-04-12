import pathlib
from slicegpt.rotate import slice_attention_inputs, slice_attention_output, slice_embeddings, slice_head, slice_mlp_input, slice_mlp_output
from transformers.models.phi.modeling_phi import PhiConfig, PhiForCausalLM, PhiModel
from slicegpt.slicing_scheduler import ConfigSlicingScheduler, SlicingScheduler
from slicegpt.model_adapter import SlicingConfig
from slicegpt.adapters.phi2_adapter import CompressedPhiDecoderLayer
from slicegpt.modules import RMSN
from slicegpt.adapters.phi2_adapter import Phi2ModelAdapter
import torch
import torch.nn as nn

class SlicedPhi2Config(PhiConfig):
    model_type = "sliced_phi2"
    is_composition = True

    def __init__(self, sparsity, hidden_size, new_hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.sparsity = sparsity
        self.hidden_size = hidden_size
        self.new_hidden_dim = new_hidden_dim
        
    def to_dict(self):
        output = super().to_dict()
        output.update({"sparsity": self.sparsity, "new_hidden_dim": self.new_hidden_dim})
        return output

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

class SlicedPhi(PhiModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [CompressedPhiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = RMSN(config.hidden_size)
        
class SlicedPhiForCausalLM(PhiForCausalLM):
    def __init__(self, config, scheduler: SlicingScheduler | None = None):
        super().__init__(config)
        self.model = SlicedPhi(config)
        self.model_adapter = Phi2ModelAdapter(self)
        
        if scheduler:
            self.slice(scheduler)
            
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Overrides the from_pretrained method to accept the scheduler and returns the sliced model"""
        scheduler = kwargs.pop("slicing_scheduler", None)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls(model.config, scheduler)
        model.load_state_dict(model.state_dict())
        return model

    def slice(self, scheduler: SlicingScheduler):
        slice_embeddings(self.model_adapter, scheduler.get_embedding_dimensions())
        
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
        
        for idx, layer_adapter in enumerate(layers):
            slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
            slice_mlp_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
            
            slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
            slice_attention_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
            
            layer_adapter.layer.attn_shortcut_Q = nn.Parameter(
                layer_adapter.layer.attn_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)].contiguous()
            )
    
        if slicing_scheduler.do_slice_head:
            slice_head(self.model_adapter, slicing_scheduler.get_head_dimension())

if __name__ == "__main__":
    sparsity = 0.1
    hidden_size = 2560
    num_hidden_layers= 32
    round_interval = 8
    config_path = ""
    
    new_embedding_dim = int((1 - sparsity) * hidden_size)
    new_embedding_dim -= new_embedding_dim % round_interval

    config_path = pathlib.Path(config_path)

    slicing_conf = SlicingConfig.from_json_string(config_path.read_text())
    
    slicing_scheduler = ConfigSlicingScheduler(slicing_conf)

    sliced_model = SlicedPhi2Config(sparsity=sparsity, hidden_size=hidden_size, new_hidden_dim=new_embedding_dim)
    sliced_model.save_pretrained("sliced_phi2")
    
    config = SlicedPhi2Config.from_pretrained("sliced_phi2")
    print(config)
    
    sliced_model = SlicedPhiForCausalLM(config, slicing_scheduler)
    print(sliced_model)
    
    sliced_model.save_pretrained("sliced_phi2_model")
    sliced_model.from_pretrained("sliced_phi2_model", slicing_scheduler)
    
    print(sliced_model)