# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM

from . import rotate, layernorm_fusion, model_utils

from .model_utils import get_layers
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory


class UninitializedOPTForCausalLM(OPTForCausalLM):
    def _init_weights(self, module):
        # Prevent weight initialization
        pass


class UninitializedLlamaForCausalLM(LlamaForCausalLM):
    def _init_weights(self, module):
        # Prevent weight initialization
        pass


def get_model(model_path, uninitialized=False, dtype=torch.float16, token=None):
    """Loads the model and the tokenizer from the given path."""
    if uninitialized:
        model_type = "uninitialized"
    else:
        model_type = "pretrained"

    print(f"Loading {model_type} {model_path} model...", end=" ")

    if "facebook/opt" in model_path:
        model = transformers.OPTForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    elif "meta-llama" in model_path:
        model = transformers.LlamaForCausalLM.from_pretrained(model_path, torch_dtype='auto', use_auth_token=hf_token)
    # dtype = torch.float16
    # with deepspeed.OnDevice(dtype=dtype, device="meta"):
    if "facebook/opt" in model_path:
        if uninitialized:
            config = OPTConfig.from_pretrained(model_path)
            model = UninitializedOPTForCausalLM(config)
            model = model.to(dtype=dtype)
        else:
            model = transformers.OPTForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    elif "meta-llama" in model_path:
        if uninitialized:
            config = LlamaConfig.from_pretrained(model_path)
            model = UninitializedLlamaForCausalLM(config)
            model = model.to(dtype=dtype)
        else:
            model = transformers.LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, token=token)
    else:
        raise NotImplementedError

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)

    model.seqlen = model.config.max_position_embeddings
    model.eval()  # This switches off dropout.

    print("Done.")
    return model, tokenizer


def load_sliced_model(model_name, model_path, sparsity, device):
    """Loads the sliced model and the tokenizer from the given path."""
    model, tokenizer = get_model(model_name, uninitialized=True)
    layernorm_fusion.replace_modules(model, model.config)
    layernorm_fusion.fuse_modules(model)
    new_embedding_dimension = int((1 - sparsity) * model.config.hidden_size)

    for layer in model_utils.get_layers(model):
        mlp_shortcut_Q = torch.zeros(model.config.hidden_size, model.config.hidden_size).to(dtype=torch.float16)
        attn_shortcut_Q = torch.zeros(model.config.hidden_size, model.config.hidden_size).to(dtype=torch.float16)
        layer.register_buffer("mlp_shortcut_Q", mlp_shortcut_Q)
        layer.register_buffer("attn_shortcut_Q", attn_shortcut_Q)

    rotate.slice_rotated_model(model, new_embedding_dimension)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, tokenizer

def infer_device_map(model):
    no_split_modules = ["OPTDecoderLayer", "CompressedOPTDecoderLayer"]
    max_memory = get_balanced_memory(
        model,
        max_memory=None,
        no_split_module_classes=no_split_modules,
    )

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split_modules
    )

    print(device_map)
    dispatch_model(model, device_map=device_map, offload_buffers=True, offload_dir="offload", state_dict=model.state_dict())
