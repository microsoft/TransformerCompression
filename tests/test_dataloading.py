import inspect

import numpy as np
import pytest

from slicegpt import data_utils, hf_utils


@pytest.mark.parametrize(
    "dataset_name",
    [
        "wikitext2",
        "ptb",
        "c4",
    ],
)
def test_get_dataset(dataset_name) -> None:
    train_dataset, test_dataset = data_utils.get_dataset(dataset_name=dataset_name)

    assert train_dataset is not None
    assert test_dataset is not None

    assert len(train_dataset) > 0
    assert len(test_dataset) > 0


@pytest.mark.parametrize(
    "dataset_name, max_seqlen, batch_size, nsamples",
    [
        ("wikitext2", None, None, None),
        ("wikitext2", 512, None, None),
        ("wikitext2", 512, 32, None),
        ("wikitext2", 512, 32, 1024),
        ("ptb", 256, 64, 512),
        ("c4", 128, 64, 512),
    ],
)
def test_get_loaders(dataset_name: str, max_seqlen: int, batch_size: int, nsamples: int) -> None:

    model_name = "facebook/opt-125m"
    _, tokenizer = hf_utils.get_model(model_name)

    dataset, _ = data_utils.get_dataset(dataset_name=dataset_name)

    def get_default_args(func):
        signature = inspect.signature(func)
        return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

    defaults = get_default_args(data_utils.get_loader_from_dataset)

    if not max_seqlen:
        max_seqlen = defaults["max_seqlen"]
    if not batch_size:
        batch_size = defaults["batch_size"]

    loader = data_utils.get_loader_from_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_seqlen=max_seqlen,
        batch_size=batch_size,
        nsamples=nsamples,
    )

    assert loader is not None

    if nsamples:
        n_batches = np.ceil(nsamples / batch_size)
        assert len(loader) == n_batches

    def check_shape_first_batch(loader):
        batch = next(iter(loader))
        for key in ["input_ids", "attention_mask"]:
            if len(loader) == 1:
                assert batch[key].shape[0] <= batch_size
            else:
                assert batch[key].shape[0] == batch_size

            assert batch[key].shape[1] <= max_seqlen

    check_shape_first_batch(loader)
