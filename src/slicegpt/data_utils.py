# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import logging

import datasets
from torch.utils.data import DataLoader


def get_loaders(dataset_name, tokenizer, nsamples=128, seed=0, seqlen=2048, batch_size=1):
    print(f"Loading dataset: {dataset_name}...", end=" ")
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

    traindata = datasets.load_dataset(path, name, data_files=train_data_files, split="train")
    testdata = datasets.load_dataset(path, name, data_files=test_data_files, split=test_split)

    random.seed(seed)
    if dataset_name == "c4":
        # Keep only a subset of c4's train & test datasets, because these are much larger than
        # wikitext and ptb datasets.
        train_indices = random.sample(range(len(traindata)), 5_000)
        test_indices = random.sample(range(len(testdata)), 500)

        traindata = traindata.select(train_indices)
        testdata = testdata.select(test_indices)

    trainenc = tokenizer("\n\n".join(traindata[data_name]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata[data_name]), return_tensors="pt")

    # sample the train set
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        input_ids = trainenc.input_ids[0, i:j]
        trainloader.append(input_ids)

    # test set
    n_test_samples = testenc.input_ids.numel() // seqlen
    testloader = testenc.input_ids[0, : n_test_samples * seqlen].reshape(n_test_samples, seqlen)

    # convert to torch dataloaders
    trainloader = DataLoader(trainloader, batch_size=batch_size)
    testloader = DataLoader(testloader, batch_size=batch_size)

    print("Done.")
    return trainloader, testloader
