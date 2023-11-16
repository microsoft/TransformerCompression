import logging
import time

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from torch.utils.data import DataLoader

from . import utils


@torch.no_grad()
def evaluate_ppl(model, testloader: DataLoader[torch.Tensor], device: torch.device) -> float:
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


def distribute_model(model) -> None:
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
