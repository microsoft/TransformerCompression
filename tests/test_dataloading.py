import inspect

import numpy as np
import pytest

from slicegpt import data_utils, hf_utils


@pytest.mark.parametrize(
    "dataset_name, max_seqlen, batch_size, cal_nsamples",
    [
        ("wikitext2", None, None, None),
        ("wikitext2", 512, None, None),
        ("wikitext2", 512, 32, None),
        ("wikitext2", 512, 32, 1024),
        ("ptb", 256, 64, 512),
        ("c4", 128, 64, 512),
    ],
)
def test_get_loaders(dataset_name: str, max_seqlen: int, batch_size: int, cal_nsamples: int) -> None:

    model_name = "facebook/opt-125m"
    _, tokenizer = hf_utils.get_model(model_name)

    def get_default_args(func):
        signature = inspect.signature(func)
        return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

    defaults = get_default_args(data_utils.get_loaders)

    if not max_seqlen:
        max_seqlen = defaults["max_seqlen"]
    if not batch_size:
        batch_size = defaults["batch_size"]

    trainloader, testloader = data_utils.get_loaders(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_seqlen=max_seqlen,
        batch_size=batch_size,
        nsamples=cal_nsamples,
    )

    assert trainloader is not None
    assert testloader is not None

    if cal_nsamples:
        n_batches = np.ceil(cal_nsamples / batch_size)
        assert len(trainloader) == n_batches
        assert len(testloader) == n_batches

    def check_shape_first_batch(dataloader, only_batch=False):
        batch = next(iter(dataloader))
        for key in ["input_ids", "attention_mask"]:
            if only_batch:
                assert batch[key].shape[0] <= batch_size
            else:
                assert batch[key].shape[0] == batch_size

            assert batch[key].shape[1] <= max_seqlen

    if len(trainloader) == 1:
        check_shape_first_batch(trainloader, True)
    else:
        check_shape_first_batch(trainloader)

    if len(testloader) == 1:
        check_shape_first_batch(testloader, True)
    else:
        check_shape_first_batch(testloader)
