import torch
import transformers
from .opt_modules import CompressedOPTDecoderLayer

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = CompressedOPTDecoderLayer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embeddings(model):
    if model.__class__ is OPT_MODEL:
        return [model.model.decoder.embed_tokens,  model.model.decoder.embed_positions]
    else:
        raise NotImplementedError

def get_layers(model):
    if model.__class__ is OPT_MODEL:
        return model.model.decoder.layers
    else:
        raise NotImplementedError

def get_first_layernorm(layer):
    if layer.__class__ is OPT_LAYER:
        return layer.self_attn_layer_norm
    else:
        raise NotImplementedError

def get_second_layernorm(layer):
    if layer.__class__ is OPT_LAYER:
        return layer.final_layer_norm
    else:
        raise NotImplementedError
    
def get_pre_head_layernorm(model):
    if model.__class__ is OPT_MODEL:
        return model.model.decoder.final_layer_norm
    else:
        raise NotImplementedError
    
def get_attention_inputs(layer):
    if layer.__class__ is OPT_LAYER:
        return [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
    else:
        raise NotImplementedError
    

def get_attention_output(layer):
    if layer.__class__ is OPT_LAYER:
        return layer.self_attn.out_proj
    else:
        raise NotImplementedError

def get_mlp_inputs(layer):
    if layer.__class__ is OPT_LAYER:
        return [layer.fc1]  # this is a list because gated networks need a list.
    else:
        raise NotImplementedError

def get_mlp_output(layer):
    if layer.__class__ is OPT_LAYER:
        return layer.fc2
    else:
        raise NotImplementedError

def get_lm_head(model):
    if model.__class__ is OPT_MODEL:
        return model.lm_head
    else:
        raise NotImplementedError