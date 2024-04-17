import argparse
import pathlib
from slicegpt import data_utils, gpu_utils, hf_utils
from slicegpt.rotate import slice_rotated_model
from transformers.models.phi.modeling_phi import PhiConfig, PhiForCausalLM, PhiModel
from slicegpt.slicing_scheduler import ConfigSlicingScheduler, SlicingScheduler
from slicegpt.model_adapter import SlicingConfig
from slicegpt.adapters.phi2_adapter import CompressedPhiDecoderLayer
from slicegpt.modules import RMSN
from slicegpt.adapters.phi2_adapter import Phi2ModelAdapter
import torch
import torch.nn as nn
import os

class SlicedPhi2Config(PhiConfig):
    model_type = "sliced_phi2"
    is_composition = True

    def __init__(self, sparsity, num_layers, hidden_size, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.sparsity = sparsity
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
    def to_dict(self):
        output = super().to_dict()
        output.update({"sparsity": self.sparsity})
        return output

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

class SlicedPhi(PhiModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [CompressedPhiDecoderLayer(config, layer_idx, replace_layernorm=True)for layer_idx in range(config.num_layers)]
        )
        self.final_layernorm = RMSN(config.hidden_size)
        
class SlicedPhiForCausalLM(PhiForCausalLM):
    def __init__(self, config, scheduler: SlicingScheduler | None = None, *model_args, **kwargs):
        super().__init__(config)
        self.model = SlicedPhi(config)
        self.model_adapter = Phi2ModelAdapter(self)
        
        if scheduler:
            self.update_dims(scheduler)
            
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, scheduler: SlicingScheduler, config_path, *model_args, **kwargs):
        """Overrides the from_pretrained method to accept the scheduler and returns the sliced model"""
        config = SlicedPhi2Config.from_pretrained(config_path)
        #model = cls(config, scheduler)
        model = super().from_pretrained(pretrained_model_name_or_path, scheduler, config)
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

def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="Model to load",
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        default=None,
        help="Path to save the final model to",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.1, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the sliced model to copy the weights from",
        default="",
    )
    parser.add_argument(
        "--sliced-model-config-path",
        type=str,
        help="Path to load the config of the sliced model from",
        default="",
    )
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    return parser.parse_args()

if __name__ == "__main__":
    
    args = arg_parser()
    
    config_path = pathlib.Path(args.sliced_model_config_path)

    slicing_conf = SlicingConfig.from_json_string(config_path.read_text())
    
    slicing_scheduler = ConfigSlicingScheduler(slicing_conf)

    sliced_model_conf = SlicedPhi2Config(sparsity=args.sparsity, hidden_size=slicing_conf.hidden_size, intermediate_size=slicing_conf.intermediate_size, num_layers=slicing_conf.layers_num)
    sliced_model_conf.save_pretrained("sliced_phi2")
    
    config = SlicedPhi2Config.from_pretrained("sliced_phi2")

    config.torch_dtype = torch.float16
    sliced_model = SlicedPhiForCausalLM(config, slicing_scheduler)
    
    # load the saved sliced model
    model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            sparsity=args.sparsity,
            round_interval=args.round_interval,
            token=args.hf_token,
        )
    
    dataset = data_utils.get_dataset("wikitext2")
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    
    test_loader = data_utils.prepare_test_dataloader(
        dataset=test_dataset, tokenizer=tokenizer, batch_size=8
    )
    
    # evaluate original perplexity
    dataset_ppl = gpu_utils.evaluate_ppl(model_adapter.model.to("cuda"), tokenizer.pad_token_id, test_loader)
    print(f'Loaded sliced model perplexity: {dataset_ppl}')
    

    sliced_model = sliced_model.to(torch.float16)
    sliced_model.load_state_dict(model_adapter.model.state_dict(), strict=True, assign=True)
    print("Model loaded successfully!")

    
    sliced_model.to("cuda")
    
    dataset_ppl = gpu_utils.evaluate_ppl(sliced_model, tokenizer.pad_token_id, test_loader)
    print(f'Loaded new sliced model perplexity: {dataset_ppl}')
    
    sliced_model.save_pretrained("new_sliced_phi2_model")
    sliced_model_new = sliced_model.from_pretrained("new_sliced_phi2_model", slicing_scheduler, "sliced_phi2")
    sliced_model_new = sliced_model_new.to(torch.float16)
    
    dataset_ppl = gpu_utils.evaluate_ppl(sliced_model_new.to("cuda"), tokenizer.pad_token_id, test_loader)
    print(f'Loaded new sliced model perplexity: {dataset_ppl}')