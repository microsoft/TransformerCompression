import torch
import random
import datasets
import numpy as np
import transformers


def get_wikitext2(nsamples, seed, seqlen, model, hf_token):
    traindata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    if hf_token == None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model, use_fast=False, use_auth_token=hf_token
        )
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model, hf_token):
    traindata = datasets.load_dataset("ptb_text_only", "penn_treebank", split="train")
    valdata = datasets.load_dataset(
        "ptb_text_only", "penn_treebank", split="validation"
    )

    if hf_token == None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model, use_fast=False, use_auth_token=hf_token
        )
    trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model, hf_token):
    traindata = datasets.load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = datasets.load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    if hf_token == None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model, use_fast=False, use_auth_token=hf_token
        )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model, hf_token):

    traindata = datasets.load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = datasets.load_dataset("ptb_text_only", "penn_treebank", split="test")

    if hf_token == None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model, use_fast=False, use_auth_token=hf_token
        )
    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model, hf_token):
    traindata = datasets.load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = datasets.load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    if hf_token == None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model, use_fast=False, use_auth_token=hf_token
        )

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model="", hf_token=None):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token)
    if "ptb" in name:
        if "new" in name:
            return get_ptb_new(nsamples, seed, seqlen, model, hf_token)
        return get_ptb(nsamples, seed, seqlen, model, hf_token)
    if "c4" in name:
        if "new" in name:
            return get_c4_new(nsamples, seed, seqlen, model, hf_token)
        return get_c4(nsamples, seed, seqlen, model, hf_token)
