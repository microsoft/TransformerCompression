from transformers import PreTrainedModel
from transformers.models.phi.modeling_phi import PhiConfig, PhiForCausalLM, PhiModel
from slicegpt.adapters.phi2_adapter import CompressedPhiDecoderLayer
from slicegpt.modules import RMSN
import torch.nn as nn

class SlicedPhi2Config(PhiConfig):
    model_type = "sliced_phi2"
    is_composition = True

    def __init__(self, sparsity, num_hidden_layers, hidden_size, new_hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.sparsity = sparsity
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.new_hidden_dim = new_hidden_dim
        
    def to_dict(self):
        output = super().to_dict()
        output.update({"sparsity": self.sparsity, "num_hidden_layers": self.num_hidden_layers, "new_hidden_dim": self.new_hidden_dim})
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
    def __init__(self, config):
        super().__init__(config)
        self.model = SlicedPhi(config)
    
        self.layers = nn.ModuleList(
            [CompressedPhiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = RMSN(config.hidden_size)
        
        # TODO: apply slicing here according to the new_hidden_dim (or incorporate it into the CompressedPhiDecoderLayer)

class SlicedPhi2(PreTrainedModel):
    """Wrapper class around SlicedPhiForCausalLM so it can be registered as a HF model"""
    config_class = SlicedPhi2Config
    base_model_prefix = "sliced_phi2"
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = SlicedPhi(config)
        
    def forward(self, input_ids, **kwargs):
        return input_ids
    
    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = SlicedPhi2.from_pretrained(pretrained_model_name_or_path)
        return cls(config, *model_args, **kwargs)
    
if __name__ == "__main__":
    sparsity = 0.1
    hidden_size = 2560
    num_hidden_layers= 31
    round_interval = 8
    
    new_embedding_dim = int((1 - sparsity) * hidden_size)
    new_embedding_dim -= new_embedding_dim % round_interval

    sliced_model = SlicedPhi2Config(sparsity=sparsity, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, new_hidden_dim=new_embedding_dim)
    sliced_model.save_pretrained("sliced_phi2")
    
    config = SlicedPhi2Config.from_pretrained("sliced_phi2")
    print(config)
    
    sliced_model = SlicedPhiForCausalLM(config)
    print(sliced_model)
    
    sliced_model.save_pretrained("sliced_phi2_model")
    sliced_model.from_pretrained("sliced_phi2_model")
    
    print(sliced_model)