# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import datasets
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler


def get_loaders(
    dataset_name: str, tokenizer, max_seqlen: int = 2048, batch_size: int = 1, num_batches: int = None, seed=42
) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    logging.info(f"Loading dataset: {dataset_name}")
    if dataset_name == "wikitext2":
        path = "wikitext"
        name = "wikitext-2-raw-v1"
        data_name = "text"
        train_data_files = None
        test_data_files = None
        test_split = "test"
    elif dataset_name == "ptb":
        path = "ptb_text_only"
        name = "penn_treebank"
        data_name = "sentence"
        train_data_files = None
        test_data_files = None
        test_split = "test"
    elif dataset_name == "c4":
        path = "allenai/c4"
        name = "allenai--c4"
        data_name = "text"
        train_data_files = {"train": "en/c4-train.00000-of-01024.json.gz"}
        test_data_files = {"validation": "en/c4-validation.00000-of-00008.json.gz"}
        test_split = "validation"
    else:
        raise NotImplementedError("The provided dataset is not supported")

    train_dataset = datasets.load_dataset(path, name=name, data_files=train_data_files, split="train")
    test_dataset = datasets.load_dataset(path, name=name, data_files=test_data_files, split=test_split)

    def tokenize(dataset):
        # tokenize then pad each batch according to longest sequence in the batch
        return tokenizer(
            dataset[data_name], padding="longest", max_length=max_seqlen, truncation=True, return_tensors="pt"
        )

    # tokenize lazily
    train_dataset.set_transform(tokenize)
    test_dataset.set_transform(tokenize)

    if num_batches is None:
        num_batches = len(train_dataset) // batch_size

    # sample the datasets to get the desired number of batches
    torch.manual_seed(seed)
    train_sampler = SubsetRandomSampler(torch.randperm(len(train_dataset))[: num_batches * batch_size])
    test_sampler = SubsetRandomSampler(torch.randperm(len(test_dataset))[: num_batches * batch_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    logging.info("Loading dataset done")
    return train_dataloader, test_dataloader
