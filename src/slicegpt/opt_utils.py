import math
import os
import time

import numpy as np
import torch
import tqdm
import transformers

from .model_utils import (
    get_attention_inputs,
    get_attention_output,
    get_embeddings,
    get_first_layernorm,
    get_layer0_inputs,
    get_layers,
    get_lm_head,
    get_mlp_inputs,
    get_mlp_output,
    get_pre_head_layernorm,
    get_second_layernorm,
    get_signals,
)


@torch.no_grad()
def evaluate_perplexity(model, testloader, device):
    """
    evaluate the model's perplexity on the test set.
    This function loads each loayer onto the device one at a time,
    so that we can evaluate models that are too large to fit on a single GPU.
    """
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_layers(model)

    num_samples = len(testloader)
    X, mask = get_layer0_inputs(model, testloader)

    print("(Eval) Layers: ", end="", flush=True)
    for i, layer in enumerate(layers):
        print(f", {i}", end="", flush=True)
        layer = layer.to(device)
        outs = [layer(X[j].unsqueeze(0), attention_mask=mask)[0] for j in range(num_samples)]
        del layer
        torch.cuda.empty_cache()
        X = torch.cat(outs)
    print("")

    X = get_pre_head_layernorm(model).to(device)(X)

    lm_head = get_lm_head(model).to(device)
    nlls = []
    for i, sample in enumerate(testloader):
        x = X[i].unsqueeze(0)
        lm_logits = lm_head(x)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = sample[1:].to(device)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.squeeze(0), shift_labels.view(-1))
        neg_log_likelihood = loss.float()
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).mean())
    model.config.use_cache = use_cache
    return ppl.item()


def opt_multigpu(model, gpus):
    """
    Split the OPT model across multiple GPUs.
    """
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(gpus[0])
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(gpus[0])
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(gpus[0])
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(gpus[-1])
    if hasattr(model.model.decoder, "final_layer_norm") and model.model.decoder.final_layer_norm:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(gpus[-1])
    import copy

    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {"mask": None}

    class MoveModule(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache["mask"] is None or cache["mask"].device != self.dev:
                cache["mask"] = kwargs["attention_mask"].to(self.dev)
            kwargs["attention_mask"] = cache["mask"]
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.decoder.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus


def opt_benchmark(model, input_ids, dev, check=False):
    DEV = dev
    model.config.use_cache = True
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, "gpus") else dev)
    torch.cuda.synchronize()

    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    print("Benchmarking ...")

    if check:
        loss = torch.nn.CrossEntropyLoss()
        tot = 0.0

    def sync():
        if hasattr(model, "gpus"):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in tqdm.tqdm(range(input_ids.numel()), desc="Benchmarking", ncols=80):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape((1, -1)),
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, : (i + 1)].reshape((1, -1)),
            )
            sync()
            times.append(time.time() - tick)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache["past"] = list(out.past_key_values)
            del out
        sync()

        print("Median:", np.median(times))
        if check:
            print("PPL:", torch.exp(tot / (input_ids.numel() - 1)).item())
        return np.median(times)
