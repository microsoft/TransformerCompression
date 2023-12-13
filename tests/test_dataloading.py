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
        ("wikitext2", None, None, 8),
        ("wikitext2", 512, None, 8),
        ("wikitext2", 512, 4, 8),
        ("wikitext2", 512, 4, 8),
        ("ptb", 256, 2, 4),
        ("c4", 128, 2, 4),
    ],
)
def test_get_loaders(dataset_name: str, max_seqlen: int, batch_size: int, nsamples: int) -> None:

    model_name = "facebook/opt-125m"
    _, tokenizer = hf_utils.get_model_and_tokenizer(model_name)

    dataset, _ = data_utils.get_dataset(dataset_name=dataset_name)

    def get_default_args(func):
        signature = inspect.signature(func)
        return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

    defaults = get_default_args(data_utils.prepare_dataloader)

    if not max_seqlen:
        max_seqlen = defaults["max_seqlen"]
    if not batch_size:
        batch_size = defaults["batch_size"]
    if not nsamples:
        nsamples = defaults["nsamples"]

    loader_varied_seqlen = data_utils.prepare_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        max_seqlen=max_seqlen,
        batch_size=batch_size,
        nsamples=nsamples,
        varied_seqlen=True,
    )

    loader_fixed_seqlen = data_utils.prepare_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        max_seqlen=max_seqlen,
        batch_size=batch_size,
        nsamples=nsamples,
    )

    assert loader_varied_seqlen is not None
    assert loader_fixed_seqlen is not None

    n_batches = np.ceil(nsamples / batch_size)
    assert len(loader_varied_seqlen) == n_batches
    assert len(loader_fixed_seqlen) == n_batches

    def check_shape_first_batch(loader, fixed_length):
        batch = next(iter(loader))
        for key in ["input_ids", "attention_mask"]:
            if len(loader) == 1:
                assert batch[key].shape[0] <= batch_size
            else:
                assert batch[key].shape[0] == batch_size

            if fixed_length:
                assert batch[key].shape[1] == max_seqlen
            else:
                assert batch[key].shape[1] <= max_seqlen

    check_shape_first_batch(loader_varied_seqlen, False)
    check_shape_first_batch(loader_fixed_seqlen, True)
