# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from slicegpt import hf_utils


def test_download_model() -> None:
    print("Downloading model from HF.")
    model_name = "facebook/opt-125m"
    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(model_name)

    # Assert that the model is not null
    assert model_adapter is not None
    assert tokenizer is not None


def test_local_model() -> None:
    print("Downloading model from HF.")
    model_name = "facebook/opt-125m"

    _, _ = hf_utils.get_model_and_tokenizer(model_name)

    print("Loading model from local HF cache.")
    home_dir = os.getenv('USERPROFILE') or os.getenv('HOME')
    model_path = home_dir + "/.cache/huggingface/hub//models--facebook--opt-125m/snapshots/"
    dirs = [model_path + "/" + d for d in os.listdir(model_path) if os.path.isdir(model_path + "/" + d)]
    snapshot_dir = sorted(dirs, key=lambda x: os.path.getctime(x), reverse=True)[0]

    local_model_adapter, local_tokenizer = hf_utils.get_model_and_tokenizer(model_name, model_path=snapshot_dir)

    # Assert that the model is not null
    assert local_model_adapter is not None
    assert local_tokenizer is not None
