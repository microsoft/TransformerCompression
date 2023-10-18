# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from slicegpt import hf_utils


def test_download_model():
    print("Downloading model from HF.")
    model_name = "facebook/opt-125m"
    model, tokenizer = hf_utils.get_model(model_name)

    # Assert that the model is not null
    assert model is not None
    assert tokenizer is not None
