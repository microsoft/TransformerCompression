# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import sys

import torch
from pyreporoot import project_root
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    PreTrainedTokenizerBase,
)

from .adapters.llama_adapter import LlamaModelAdapter
from .adapters.opt_adapter import OPTModelAdapter
from .adapters.phi2hf_adapter import Phi2HFModelAdapter
from .layernorm_fusion import fuse_modules, replace_layers
from .model_adapter import ModelAdapter
from .rotate import slice_rotated_model

sys.path.append(project_root(__file__, root_files="pyproject.toml"))
from phi2_hf.configuration_phi import PhiConfig
from phi2_hf.modeling_phi import PhiForCausalLM


class UninitializedOPTForCausalLM(OPTForCausalLM):
    def _init_weights(self, _) -> None:
        # Prevent weight initialization
        pass


class UninitializedLlamaForCausalLM(LlamaForCausalLM):
    def _init_weights(self, _) -> None:
        # Prevent weight initialization
        pass


class UninitializedPhiForCausalLM(PhiForCausalLM):
    def _init_weights(self, _) -> None:
        # Prevent weight initialization
        pass


def skip(*args, **kwargs) -> None:
    pass


def do_not_initialize(func):
    """
    A decorator that prevents initalization of torch.nn modules.
    """

    def wrapper(*args, **kwargs):
        kaiming_fn = torch.nn.init.kaiming_uniform_
        uniform_fn = torch.nn.init.uniform_
        normal_fn = torch.nn.init.normal_

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        result = func(*args, **kwargs)

        torch.nn.init.kaiming_uniform_ = kaiming_fn
        torch.nn.init.uniform_ = uniform_fn
        torch.nn.init.normal_ = normal_fn

        return result

    return wrapper


@do_not_initialize
def get_model_and_tokenizer(
    model_path: str, uninitialized: bool = False, dtype: torch.dtype = torch.float16, token: str | bool | None = None
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """Loads the model and the tokenizer from the given path."""
    if uninitialized:
        model_type = "uninitialized"
    else:
        model_type = "pretrained"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)

    logging.info(f"Loading {model_type} {model_path} model")

    if "facebook/opt" in model_path:
        if uninitialized:
            config = OPTConfig.from_pretrained(model_path)
            model = UninitializedOPTForCausalLM(config)
            model = model.to(dtype=dtype)
        else:
            model = OPTForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
            model.config.torch_dtype = dtype
        model_adapter = OPTModelAdapter(model)
    elif "meta-llama" in model_path:
        if uninitialized:
            config = LlamaConfig.from_pretrained(model_path, token=token)
            model = UninitializedLlamaForCausalLM(config)
            model = model.to(dtype=dtype)
        else:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, token=token)
            model.config.torch_dtype = dtype

        # TODO: change to <eos>
        tokenizer.add_special_tokens({"pad_token": "<pad>"})  # Llama-2 models don't have a pad token by default
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        model_adapter = LlamaModelAdapter(model)
    elif "microsoft/phi-2" in model_path:
        if uninitialized:
            config = PhiConfig.from_pretrained(model_path, token=token)
            model = UninitializedPhiForCausalLM(config)
            model = model.to(dtype=dtype)
        else:
            model = PhiForCausalLM.from_pretrained(model_path, torch_dtype=dtype, token=token)
            model.config.torch_dtype = dtype

        tokenizer.add_special_tokens({"pad_token": "<pad>"})  # Llama-2 models don't have a pad token by default
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        model_adapter = Phi2HFModelAdapter(model)
    else:
        raise NotImplementedError

    model.seqlen = model.config.max_position_embeddings
    model.eval()  # This switches off dropout.
    model_adapter.use_cache = False

    logging.info("Loading model done")

    return model_adapter, tokenizer


@do_not_initialize
def load_sliced_model(
    model_name: str, model_path: str, sparsity: float, token: str, lora_config: LoraConfig = None
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """Loads the sliced model and the tokenizer from the given path. If lora_config is supplied as an arg then this
    function will return a PEFT model (post-slicing finetuned model)."""
    model_adapter, tokenizer = get_model_and_tokenizer(model_name, uninitialized=True, token=token)
    replace_layers(model_adapter)
    fuse_modules(model_adapter)
    new_embedding_dimension = int((1 - sparsity) * model_adapter.hidden_size)

    for layer_adapter in model_adapter.get_layers():
        if not model_adapter.parallel_blocks:
            layer_adapter.layer.mlp_shortcut_Q = torch.zeros(model_adapter.hidden_size, model_adapter.hidden_size).to(
                dtype=torch.float16
            )

        layer_adapter.layer.attn_shortcut_Q = torch.zeros(model_adapter.hidden_size, model_adapter.hidden_size).to(
            dtype=torch.float16
        )

    slice_rotated_model(model_adapter, new_embedding_dimension)

    if lora_config:
        model_adapter.model = get_peft_model(model_adapter.model, lora_config)

    model_adapter.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model_adapter.model.eval()

    return model_adapter, tokenizer
