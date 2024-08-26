# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib

import torch
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

from slicegpt.hf_utils import do_not_initialize

from .model_adapter import ModelAdapter
from .modeling_llama import QuarotLlamaConfig, QuarotLlamaForCausalLM
from .modeling_mixtral import QuarotMixtralConfig, QuarotMixtralForCausalLM
from .modeling_phi3 import QuarotPhi3Config, QuarotPhi3ForCausalLM
from .modeling_phi35 import QuarotPhi35Config, QuarotPhi35ForCausalLM


def quarot_model_config(
    model_name: str, model_path: str, dtype: torch.dtype, groupsize: int | None = None, offset: bool = False
):
    lm_configs = {
        'meta-llama/Llama-2-7b-hf': QuarotLlamaConfig,
        'meta-llama/Llama-2-13b-hf': QuarotLlamaConfig,
        'meta-llama/Meta-Llama-3-8B': QuarotLlamaConfig,
        'microsoft/Phi-3-mini-4k-instruct': QuarotPhi3Config,
        'Phi-3.5-mini-instruct': QuarotPhi35Config,
        'mistralai/Mixtral-8x7B-v0.1': QuarotMixtralConfig,
    }

    if model_name not in lm_configs:
        raise NotImplementedError("Model type not supported")

    model_config = lm_configs[model_name].from_pretrained(model_path, dtype=dtype, use_cache=False)
    model_config.groupsize = groupsize
    model_config.offset = offset
    return model_config


def get_quarot_model(
    model_name: str,
    rotate: bool,
    no_unfused_Had: bool,
    act_args: dict,
    key_args: dict,
    value_args: dict,
    model_config: PretrainedConfig,
):
    lm_class = resolve_quarot_model_class(model_name)

    online_had_mlp = True if rotate else False
    online_had_attn = True if rotate else False
    rms_norm = True if rotate else False

    if no_unfused_Had:
        online_had_mlp = False
        online_had_attn = False

    return lm_class(
        config=model_config,
        online_had_mlp=online_had_mlp,
        online_had_attn=online_had_attn,
        rms_norm=rms_norm,
        act_bits=act_args['a_bits'],
        act_clip_ratio=act_args['a_clip_ratio'],
        act_quantile=act_args['a_quantile'],
        act_groupsize=act_args['a_groupsize'],
        k_bits=key_args['k_bits'],
        k_clip_ratio=key_args['k_clip_ratio'],
        k_quantile=key_args['k_quantile'],
        k_groupsize=key_args['k_groupsize'],
        v_bits=value_args['v_bits'],
        v_clip_ratio=value_args['v_clip_ratio'],
        v_quantile=value_args['v_quantile'],
        v_groupsize=value_args['v_groupsize'],
    )


def resolve_quarot_model_class(model_name: str) -> PreTrainedModel:
    lm_classes = {
        'meta-llama/Llama-2-7b-hf': QuarotLlamaForCausalLM,
        'meta-llama/Llama-2-13b-hf': QuarotLlamaForCausalLM,
        'meta-llama/Meta-Llama-3-8B': QuarotLlamaForCausalLM,
        'microsoft/Phi-3-mini-4k-instruct': QuarotPhi3ForCausalLM,
        'Phi-3.5-mini-instruct': QuarotPhi35ForCausalLM,
        'mistralai/Mixtral-8x7B-v0.1': QuarotMixtralForCausalLM,
    }
    if model_name not in lm_classes:
        raise NotImplementedError("Model type not supported")

    return lm_classes[model_name]


@do_not_initialize
def get_model_and_tokenizer(
    model_name: str,
    model_path: str | None = None,
    *,
    uninitialized: bool = False,
    dtype: torch.dtype = torch.float16,
    token: str | bool | None = None,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """
    Load the model and the tokenizer from the given path.
    Set uninitialized to True when loading a pre-rotated and sliced model; in this case no weights are loaded
    in this method.
    The corresponding model adapter class must be imported before calling this method.
    Scenarios:
    - Rotate & slice HF model: model_name = name, model_path = empty, uninitialized = False
        -> Obtain the model config and weights from HF through path = name.
        -> Ignore model_path if provided.
    - Slice pre-rotated HF model: model_name = name, model_path = empty or local path, uninitialized = True
        -> Obtain the model config from HF via path = name and create uninitialized model.
        -> If the model_path is provided, confirm this use case by checking that config.json does not exist.
        -> There are no other uses of model_path in this case.
    - Rotate & slice local model: model_name = name, model_path = local path, uninitialized = False
        -> Obtain the model config through path, and the pretrained weights from the local path.
        -> Use the model name only to determine the correct model adapter to use.
    - Slice pre-rotated local model: model_name = name, model_path = local path, uninitialized = True
        -> Obtain the model config from the local path and create an uninitialized model.
        -> Use the model name only to determine the correct model adapter to use.
        -> Confirm this case by checking that config.json exists.
    """
    model_type = "uninitialized" if uninitialized else "pretrained"
    local_model = model_path is not None

    if local_model and uninitialized:
        local_model = (pathlib.Path(model_path) / "config.json").exists()

    # for HF models the path to use is the model name
    if not local_model:
        model_path = model_name

    logging.info(
        f"Loading %s config %s from %s",
        model_name,
        "and model weights" if not uninitialized else "",
        model_path if local_model else 'Hugging Face',
    )

    model_adapter = ModelAdapter.from_model(
        model_name,
        model_path=model_path,
        model_type=model_type,
        dtype=dtype,
        local_files_only=local_model,
        token=token,
    )

    model = model_adapter.model
    model.seqlen = model.config.max_position_embeddings
    model.eval()  # This switches off dropout.
    model_adapter.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token, local_files_only=local_model)

    model_adapter.post_init(tokenizer)
    logging.info("Loading model done")

    return model_adapter, tokenizer
