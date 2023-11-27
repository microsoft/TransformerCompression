from slicegpt import data_utils, hf_utils
import pytest
import inspect

@pytest.mark.parametrize("dataset_name, max_seqlen, batch_size, num_batches", [
    ("wikitext2", None, None, None),
    ("wikitext2", 512, None, None),
    ("wikitext2", 512, 32, None),
    ("wikitext2", 2048, 128, 16),
    ("ptb", 256, 64, 32),
    ("c4", 128, 64, 32)
])
def test_get_loaders(dataset_name, max_seqlen, batch_size, num_batches):

    model_name = "facebook/opt-125m"
    _, tokenizer = hf_utils.get_model(model_name)

    def get_default_args(func):
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

    defaults = get_default_args(data_utils.get_loaders)

    if not max_seqlen:
        max_seqlen = defaults["max_seqlen"]
    if not batch_size:
        batch_size = defaults["batch_size"]

    trainloader, testloader = data_utils.get_loaders(dataset_name=dataset_name, tokenizer=tokenizer, max_seqlen=max_seqlen, batch_size=batch_size, num_batches=num_batches)

    assert trainloader is not None
    assert testloader is not None

    if num_batches:
        assert len(trainloader) == num_batches
        assert len(testloader) == num_batches

    def check_shape_first_batch(dataloader):
        batch = next(iter(dataloader))
        for key in ["input_ids", "attention_mask"]:
            assert batch[key].shape[0] == batch_size
            assert batch[key].shape[1] <= max_seqlen

    check_shape_first_batch(trainloader)
    check_shape_first_batch(testloader)