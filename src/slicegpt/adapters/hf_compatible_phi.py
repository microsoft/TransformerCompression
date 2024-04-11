import pathlib
from transformers.models.phi.modeling_phi import PhiConfig, PhiForCausalLM, PhiModel
from slicegpt.slicing_scheduler import ConfigSlicingScheduler, SlicingScheduler
from slicegpt.model_adapter import SlicingConfig
from slicegpt.adapters.phi2_adapter import CompressedPhiDecoderLayer
from slicegpt.modules import RMSN
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
    
        self.layers = nn.ModuleList(
            [CompressedPhiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        self.final_layernorm = RMSN(config.hidden_size)
        
        # TODO: apply slicing here according to the new_hidden_dim (or incorporate it into the CompressedPhiDecoderLayer)
        if not scheduler:
            print("Slicing scheduler is not prodided. No slicing is applied")
        else:
            self.slice(scheduler)
            
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Overrides the from_pretrained method to accept the scheduler and return the sliced model"""
        scheduler = kwargs.pop("slicing_scheduler", None)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls(model.config, scheduler)
        model.load_state_dict(model.state_dict())
        return model
    
    def slice(self, scheduler: SlicingScheduler):
        self.slice_embeddings(scheduler.get_embedding_dimensions())
    
    def slice_embeddings(self, new_hidden_dim):
        for i, W in enumerate([self.model.get_input_embeddings()]):
            W.weight.data = W.weight.data[:, : new_hidden_dim[i]].contiguous()
            W.embedding_dim = new_hidden_dim[i]

if __name__ == "__main__":
    sparsity = 0.1
    hidden_size = 2560
    num_hidden_layers= 31
    round_interval = 8
    
    new_embedding_dim = int((1 - sparsity) * hidden_size)
    new_embedding_dim -= new_embedding_dim % round_interval

    config_path = pathlib.Path("/home/t-lmikaelyan/new_2/TransformerCompression/sliced_phi_0.1/phi-2_0.1.json")

    slicing_conf = SlicingConfig.from_json_string(config_path.read_text())
    
    slicing_scheduler = ConfigSlicingScheduler(slicing_conf)

    sliced_model = SlicedPhi2Config(sparsity=sparsity, hidden_size=hidden_size, new_hidden_dim=new_embedding_dim)
    sliced_model.save_pretrained("sliced_phi2")
    
    config = SlicedPhi2Config.from_pretrained("sliced_phi2")
    print(config)
    
    sliced_model = SlicedPhiForCausalLM(config, slicing_scheduler)
    print(sliced_model)
    
    sliced_model.save_pretrained("sliced_phi2_model")
    
    sliced_model = SlicedPhiForCausalLM(config, slicing_scheduler)
    sliced_model.from_pretrained("sliced_phi2_model", slicing_scheduler)
    
    print(sliced_model)