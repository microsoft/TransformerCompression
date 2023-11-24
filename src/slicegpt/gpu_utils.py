import logging
import time
from typing import Callable, cast

import numpy as np
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import model_utils, utils
from .config import config
from .model_adapter import ModelAdapter


@torch.no_grad()
def evaluate_ppl(model: ModelAdapter, testloader: DataLoader[Tensor]) -> float:
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """
    sync_gpus()

    start_time = time.time()

    model.raw_model.eval()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    for batch in testloader:
        assert isinstance(batch, Tensor)
        input_ids = batch.to(config.device)  # type: ignore
        logits: Tensor = model.compute_output_logits(input_ids=input_ids)

        # Shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fct(logits.permute(0, 2, 1), shift_labels).float().sum(dim=1) / model.seqlen

        nlls.append(nll)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.sum() / nlls_tensor.numel())

    sync_gpus()

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()


def distribute_model(model: ModelAdapter) -> None:
    """Distribute the model across available GPUs."""
    max_memory = get_balanced_memory(
        model.raw_model,
        no_split_module_classes=model.no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model.raw_model, max_memory=max_memory, no_split_module_classes=model.no_split_module_classes
    )

    dispatch_model(
        model.raw_model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.raw_model.state_dict(),
    )

    # Run GC and cleanup GPU memory
    utils.cleanup_memory()


def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)


def benchmark(model, input_batch: torch.Tensor) -> dict:
    """Benchmark the model's latency and throughput on the given input batch."""
    model.config.use_cache = True

    cache = {"past": None}

    def clear_past_cache(layer_idx):
        def tmp(*_):
            if cache["past"]:
                cache["past"][layer_idx] = None

        return tmp

    layers = model_utils.get_layers(model)
    for idx, layer in enumerate(layers):
        # Clear past cache after each layer get called to get accurate timing of each forward pass.
        layer.register_forward_hook(clear_past_cache(idx))

    with torch.no_grad():
        batch_size, input_seq_len = input_batch.shape[:2]
        attention_mask = torch.ones((batch_size, input_seq_len))
        time_measurements = []

        for i in tqdm(range(input_seq_len), desc="Benchmarking"):
            input_batch_i = input_batch[:, i].reshape((batch_size, 1)).to(config.device)
            attention_mask_i = attention_mask[:, : (i + 1)].to(config.device)

            sync_gpus()
            start_time = time.time()
            output = model(input_batch_i, past_key_values=cache["past"], attention_mask=attention_mask_i)
            sync_gpus()
            time_measurements.append(time.time() - start_time)

            cache["past"] = list(output.past_key_values)
            del output

            input_batch_i, attention_mask_i = input_batch_i.to("cpu"), attention_mask_i.to("cpu")

        median_time = np.median(time_measurements)
        throughput = batch_size / median_time

        results = {"median_time": median_time, "latency": 1 / throughput, "throughput": throughput}
        return results
