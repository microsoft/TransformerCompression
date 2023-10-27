import math

import numpy as np
import torch
import tqdm
import time
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory


@torch.no_grad()
def evaluate_ppl(model, testloader, device, model_distributed=False):
    """
    Evaluate the model's perplexity on the test set using batch processing.
    """
    start_time = time.time()
    
    model.eval()
    model_seqlen = model.seqlen
    
    if not model_distributed:
        model_orig_device = model.device
        model.to(device)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    for batch in testloader:
        input_ids = batch.to(device)

        logits = model(input_ids=input_ids).logits

        # Shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fct(logits.permute(0, 2, 1), shift_labels).float().sum(dim=1) / model_seqlen

        nlls.append(nll)

    ppl = torch.exp(nlls.sum() / nlls.numel())

    elapsed = time.time() - start_time 
    print("Time spent on evaluation: ", time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)))
    
    return ppl.item()


def infer_device_map(model):
    no_split_modules = ["OPTDecoderLayer", "CompressedOPTDecoderLayer"]
    max_memory = get_balanced_memory(
        model,
        max_memory=None,
        no_split_module_classes=no_split_modules,
    )

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split_modules
    )

    print(device_map)
    dispatch_model(model, device_map=device_map, offload_buffers=True, offload_dir="offload", state_dict=model.state_dict())


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
