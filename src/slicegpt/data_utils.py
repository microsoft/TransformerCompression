# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import datasets
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import PreTrainedTokenizerBase


def get_dataset(dataset_name: str) -> dict[str : datasets.Dataset]:
    """
    Get the train and test dataset from the HuggingFace datasets library.

    Args:
        dataset_name: The name of the dataset to load. Must be one of "wikitext2", "ptb", or "c4".

    Returns:
        The train and test datasets.
    """
    logging.info(f"Loading dataset: {dataset_name}")

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
    else:
        raise NotImplementedError("The provided dataset is not supported")

    if dataset_name == "c4":
        dataset = {
            split: datasets.load_dataset(path, name=name, data_files=data_files, split=split)
            for split, data_files in [("train", train_data_files), ("validation", test_data_files)]
        }
    else:
        dataset = {
            split: datasets.load_dataset(path, name=name, split=split) for split in ["train", "test", "validation"]
        }

    logging.info("Loading dataset done")
    return dataset


def prepare_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    min_seqlen: int = None,
    batch_size: int = 1,
    nsamples: int = 128,
    varied_seqlen: bool = False,
    seed=42,
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a dataset.

    Args:
        dataset: The dataset to create a dataloader from load.
        tokenizer: The tokenizer to use.
        max_seqlen: The maximum sequence length, used for truncation of sequences in the dataset.
        batch_size: The batch size.
        nsamples: The number of samples to produce.
        varied_seqlen: If False, concatenate multiple examples from the dataset into one example until max_seqlen is reached.
        seed: The seed for sampling the dataset.

    Returns:
        A DataLoader.
    """
    logging.info(f"Preparing dataloader")

    if not varied_seqlen and not nsamples:
        logging.warning(
            "varied_seqlen=False, but nsamples is not specified. This will lead to tokenization of the entire dataset, which will be slow."
        )

    data_name = dataset.column_names[0]
    dataset = dataset.filter(lambda x: len(x[data_name]) > 0)

    if not varied_seqlen:
        # create a new dataset where each example is a concatenation of multiple examples of total length = max_seqlen.
        data_list = dataset[data_name]
        new_data_list = []

        torch.manual_seed(seed)
        indices = list(range(len(data_list)))

        while len(new_data_list) < nsamples and len(indices) > 0:
            start_idx = torch.randint(0, len(indices), (1,)).item()
            idx = start_idx
            tokens = []
            while len(tokens) < max_seqlen and idx < len(indices):
                item = data_list[indices[idx]]
                sep = "" if not tokens else "\n\n"
                tokens += tokenizer.tokenize(sep + item)
                idx += 1

            # TODO(max): pls double check that we want non-overlapping examples
            indices = indices[:start_idx] + indices[idx:]

            if len(tokens) >= max_seqlen:
                tokens = tokens[:max_seqlen]  # truncate to max_seqlen
                new_data_list.append(tokenizer.convert_tokens_to_string(tokens))

        dataset = datasets.Dataset.from_dict({data_name: new_data_list})

    def tokenize(data_batch):
        # tokenize then pad each batch according to the longest sequence in the batch
        batch = tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # tokenize lazily
    dataset.set_transform(tokenize)

    torch.manual_seed(seed)
    sampler = SubsetRandomSampler(torch.randperm(len(dataset))[:nsamples])

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=1)
    logging.info(f"Preparing dataloader done")
    return loader
