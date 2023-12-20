# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from slicegpt import hf_utils


def test_download_model() -> None:
    print("Downloading model from HF.")
    model_name = "facebook/opt-125m"
    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(model_name)

    # Assert that the model is not null
    assert model_adapter is not None
    assert tokenizer is not None
