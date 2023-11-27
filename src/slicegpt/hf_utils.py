# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
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
from .layernorm_fusion import fuse_modules, replace_layers
from .model_adapter import ModelAdapter
from .rotate import slice_rotated_model


class UninitializedOPTForCausalLM(OPTForCausalLM):
    def _init_weights(self, _) -> None:
        # Prevent weight initialization
        pass


class UninitializedLlamaForCausalLM(LlamaForCausalLM):
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
def get_model(
    model_path: str, uninitialized: bool = False, dtype: torch.dtype = torch.float16, token: str | bool | None = None
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """Loads the model and the tokenizer from the given path."""
    if uninitialized:
        model_type = "uninitialized"
    else:
        model_type = "pretrained"

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
        model_adapter = LlamaModelAdapter(model)
    else:
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)

    model.eval()  # This switches off dropout.
    model_adapter.use_cache = False

    logging.info("Loading model done")

    return model_adapter, tokenizer


@do_not_initialize
def load_sliced_model(
    model_name: str, model_path: str, sparsity: float, token: str
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """Loads the sliced model and the tokenizer from the given path."""
    model_adapter, tokenizer = get_model(model_name, uninitialized=True, token=token)
    replace_layers(model_adapter)
    fuse_modules(model_adapter)
    new_embedding_dimension = int((1 - sparsity) * model_adapter.hidden_size)

    for layer_adapter in model_adapter.get_layers():
        layer_adapter.layer.mlp_shortcut_Q = torch.zeros(model_adapter.hidden_size, model_adapter.hidden_size).to(
            dtype=torch.float16
        )
        layer_adapter.layer.attn_shortcut_Q = torch.zeros(model_adapter.hidden_size, model_adapter.hidden_size).to(
            dtype=torch.float16
        )

    slice_rotated_model(model_adapter, new_embedding_dimension)

    model_adapter.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model_adapter.model.eval()

    return model_adapter, tokenizer
