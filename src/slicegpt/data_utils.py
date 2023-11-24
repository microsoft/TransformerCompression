# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import datasets
import torch
from torch.utils.data import DataLoader


def get_loaders(
    dataset_name: str, tokenizer, max_seqlen: int = 2048, batch_size: int = 1
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
        train_data_files = {"train": "en/c4-train.00000-of-01024.json.gz"}
        test_data_files = {"validation": "en/c4-validation.00000-of-00008.json.gz"}
        data_name = "text"
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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    logging.info("Loading dataset done")
    return train_dataloader, test_dataloader
