# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import datasets
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer


def get_dataset(dataset_name: str) -> tuple[datasets.Dataset, datasets.Dataset]:
    """
    Get the train and test dataset from the HuggingFace datasets library.

    Args:
        dataset_name: The name of the dataset to load. Must be one of "wikitext2", "ptb", or "c4".

    Returns:
        The train and test datasets.
    """
    logging.info(f"Loading dataset: {dataset_name}")

    train_data_files = None
    test_data_files = None
    test_split = "test"
    if dataset_name == "wikitext2":
        path = "wikitext"
        name = "wikitext-2-raw-v1"
    elif dataset_name == "ptb":
        path = "ptb_text_only"
        name = "penn_treebank"
    elif dataset_name == "c4":
        path = "allenai/c4"
        name = "allenai--c4"
        train_data_files = {"train": "en/c4-train.00000-of-01024.json.gz"}
        test_data_files = {"validation": "en/c4-validation.00000-of-00008.json.gz"}
        test_split = "validation"
    else:
        raise NotImplementedError("The provided dataset is not supported")

    train_dataset = datasets.load_dataset(path, name=name, data_files=train_data_files, split="train")
    test_dataset = datasets.load_dataset(path, name=name, data_files=test_data_files, split=test_split)

    logging.info("Loading dataset done")
    return train_dataset, test_dataset


def get_loader_from_dataset(
    dataset: datasets.Dataset,
    tokenizer: AutoTokenizer,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    nsamples: int = None,
    seed=42,
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a dataset.

    Args:
        dataset: The dataset to create a dataloader from load.
        tokenizer: The tokenizer to use.
        max_seqlen: The maximum sequence length, used for truncation of sequences in the dataset.
        batch_size: The batch size.
        nsamples: The number of samples to load.
        seed: The seed for sampling the dataset.

    Returns:
        A DataLoader.
    """

    data_name = list(dataset.features.keys())[0]

    def tokenize(data_batch):
        # tokenize then pad each batch according to longest sequence in the batch
        return tokenizer(
            data_batch[data_name], padding="longest", max_length=max_seqlen, truncation=True, return_tensors="pt"
        )

    # tokenize lazily
    dataset.set_transform(tokenize)

    if nsamples is None:
        nsamples = len(dataset)

    torch.manual_seed(seed)
    sampler = SubsetRandomSampler(torch.randperm(len(dataset))[:nsamples])

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
