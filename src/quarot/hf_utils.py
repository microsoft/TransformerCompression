# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .model_adapter import ModelAdapter

from slicegpt.hf_utils import do_not_initialize

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
