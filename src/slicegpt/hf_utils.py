# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    PhiConfig,
    PhiForCausalLM,
    PreTrainedTokenizerBase,
)

from .adapters.llama_adapter import LlamaModelAdapter
from .adapters.opt_adapter import OPTModelAdapter
from .adapters.phi2_adapter import Phi2ModelAdapter
from .layernorm_fusion import fuse_modules, replace_layers
from .model_adapter import ModelAdapter, SlicingConfig
from .rotate import slice_rotated_model


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
    model_name: str,
    model_path: str | None = None,
    uninitialized: bool = False,
    dtype: torch.dtype = torch.float16,
    token: str | bool | None = None,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """Loads the model and the tokenizer from the given path."""
    if uninitialized:
        model_type = "uninitialized"
    else:
        model_type = "pretrained"

    if not model_path:
        # HF models can be downloaded using the name only, local models need to specify a path
        model_path = model_name

    logging.info(f"Loading {model_type} {model_name} model from {model_path}")

    if model_name.startswith("facebook/opt"):
        if uninitialized:
            config = OPTConfig.from_pretrained(model_path, torch_dtype=dtype)
            model = UninitializedOPTForCausalLM(config)
            model = model.to(dtype=dtype)
        else:
            model = OPTForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
            model.config.torch_dtype = dtype
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)
        model_adapter = OPTModelAdapter(model)
    elif model_name.startswith("meta-llama/Llama-2"):
        if uninitialized:
            config = LlamaConfig.from_pretrained(model_path, torch_dtype=dtype, token=token)
            model = UninitializedLlamaForCausalLM(config)
            model = model.to(dtype=dtype)
        else:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, token=token)
            model.config.torch_dtype = dtype
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)
        tokenizer.pad_token = tokenizer.eos_token  # Llama-2 models don't have a pad token by default
        model.config.pad_token_id = tokenizer.pad_token_id
        model_adapter = LlamaModelAdapter(model)
    elif model_name == "microsoft/phi-2":
        if uninitialized:
            config = PhiConfig.from_pretrained(model_path, torch_dtype=dtype, token=token)
            model = UninitializedPhiForCausalLM(config)
            model = model.to(dtype=dtype)
        else:
            model = PhiForCausalLM.from_pretrained(model_path, torch_dtype=dtype, token=token)
            model.config.torch_dtype = dtype
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)
        tokenizer.pad_token = tokenizer.eos_token  # Phi-2 models don't have a pad token by default
        model.config.pad_token_id = tokenizer.pad_token_id
        model_adapter = Phi2ModelAdapter(model)
    else:
        raise NotImplementedError

    model.seqlen = model.config.max_position_embeddings
    model.eval()  # This switches off dropout.
    model_adapter.use_cache = False

    logging.info("Loading model done")

    return model_adapter, tokenizer


@do_not_initialize
def load_sliced_model(
    model_name: str,
    model_path: str,
    *,
    token: str | None = None,
    lora_config: LoraConfig = None,
    sparsity: float | None = None,
    round_interval: int | None = 1,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """Loads the sliced model and the tokenizer from the given path. If lora_config is supplied as an arg then this
    function will return a PEFT model (post-slicing finetuned model)."""
    model_adapter, tokenizer = get_model_and_tokenizer(model_name, model_path, uninitialized=True, token=token)
    replace_layers(model_adapter)
    fuse_modules(model_adapter)

    for layer_adapter in model_adapter.get_layers():
        if not model_adapter.parallel_blocks:
            layer_adapter.layer.mlp_shortcut_Q = torch.zeros(model_adapter.hidden_size, model_adapter.hidden_size).to(
                dtype=torch.float16
            )

        layer_adapter.layer.attn_shortcut_Q = torch.zeros(model_adapter.hidden_size, model_adapter.hidden_size).to(
            dtype=torch.float16
        )

    model_path = pathlib.Path(model_path)
    config_path = model_path.with_suffix(".json")

    if config_path.exists():
        model_adapter.slicing_conf = SlicingConfig.from_json_string(config_path.read_text())

    if model_adapter.slicing_conf is None:
        # assume the model was sliced with the const sparsity specified in the arguments to this method
        new_embedding_dimension = int((1 - sparsity) * model_adapter.hidden_size)
        new_embedding_dimension -= new_embedding_dimension % round_interval
        config = SlicingConfig()
        config.const_dimension = new_embedding_dimension
        model_adapter.slicing_conf = config

    slice_rotated_model(model_adapter)

    if lora_config:
        model_adapter.model = get_peft_model(model_adapter.model, lora_config)

    model_adapter.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model_adapter.model.eval()

    return model_adapter, tokenizer
