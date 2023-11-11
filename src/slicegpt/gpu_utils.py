import logging
import time

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import transformers

from tqdm import tqdm
import numpy as np
from . import utils


@torch.no_grad()
def evaluate_ppl(model, testloader, device):
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """
    start_time = time.time()

    model.eval()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    for batch in testloader:
        input_ids = batch.to(device)
        logits = model(input_ids=input_ids).logits

        # Shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fct(logits.permute(0, 2, 1), shift_labels).float().sum(dim=1) / model.seqlen

        nlls.append(nll)

    nlls = torch.cat(nlls)
    ppl = torch.exp(nlls.sum() / nlls.numel())

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()


def distribute_model(model):
    # infer device map, make sure each layer is not split across multiple GPUs
    no_split_modules = [
        "OPTDecoderLayer",
        "CompressedOPTDecoderLayer",
        "LlamaDecoderLayer",
        "CompressedLlamaDecoderLayer",
    ]
    max_memory = get_balanced_memory(
        model,
        max_memory=None,
        no_split_module_classes=no_split_modules,
    )

    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_modules)

    dispatch_model(
        model, device_map=device_map, offload_buffers=True, offload_dir="offload", state_dict=model.state_dict()
    )

    # Run GC and cleanup GPU memory
    utils.cleanup_memory()


def benchmark(model, input_batch, device):
    model.config.use_cache = True

    batch_size = input_batch.shape[0]
    input_seqlen = input_batch.shape[1]
    input_batch = input_batch.to(device)
    torch.cuda.synchronize(device=device)

    cache = {"past": None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    if isinstance(model, transformers.LlamaForCausalLM):
        for i, layer in enumerate(model.model.layers):
            layer.register_forward_hook(clear_past(i))
    elif isinstance(model, transformers.OPTForCausalLM):
        for i, layer in enumerate(model.model.decoder.layers):
            layer.register_forward_hook(clear_past(i))
    else:
        raise NotImplementedError(f"Unsupported model type: {type(model)}")

    with torch.no_grad():
        attention_mask = torch.ones((batch_size, input_seqlen), device=device)
        times = []
        for i in tqdm(range(input_seqlen), desc="Benchmarking"):
            tick = time.time()
            out = model(
                input_batch[:, i].reshape((batch_size, 1)),
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, : (i + 1)]
            )

            torch.cuda.synchronize(device=device)
            times.append(time.time() - tick)
            cache["past"] = list(out.past_key_values)
            del out

        torch.cuda.synchronize(device=device)
        median_time = np.median(times)
        throughput = batch_size / median_time
        
        results = {"median_time": median_time, "latency": 1/throughput, "throughput": throughput}
        return results