from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate


def test_opt_125m(sparsity=0.2, ppl_upper_limit=36):
    """A complete test of OPT 125m, end-to-end. Includes:
    - loading the model
    - loading the calibration data
    - layernorm fusion & module replacement
    - slicing
    - ppl evaluation (ppl is verified to be close to published value)

    The intention of this test is that it should represent a complete run of SliceGPT,
    but on a model small enough to run on CPU in a few minutes.
    """
    # get model
    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer("facebook/opt-125m")
    model = model_adapter.model

    # prepare data
    train_dataset, test_dataset = data_utils.get_dataset("wikitext2")
    train_loader = data_utils.prepare_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_seqlen=model.seqlen,
        batch_size=1,
        nsamples=128,
        varied_seqlen=False,
        seed=42,
    )
    test_loader = data_utils.prepare_dataloader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        max_seqlen=model_adapter.seqlen,
        batch_size=1,
        varied_seqlen=False,
        seed=42,
    )

    # replace modules with compressible equivalents
    layernorm_fusion.replace_layers(model_adapter)

    # fuse layernorms and add rotations to skip connections
    layernorm_fusion.fuse_modules(model_adapter)

    # compute new embedding dimension given the desired sparsity level
    new_embedding_dimension = int((1 - sparsity) * model_adapter.hidden_size)

    # run slicing
    ignore_tokens = [tokenizer.pad_token_id]
    rotate.rotate_and_slice(model_adapter, train_loader, new_embedding_dimension, ignore_tokens=ignore_tokens)

    # get ppl
    dataset_ppl = gpu_utils.evaluate_ppl(model_adapter, test_loader)

    # check that ppl is close to published value
    assert dataset_ppl < ppl_upper_limit
