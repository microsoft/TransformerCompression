# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os

import torch

import wandb
from slicegpt import data_utils, gpu_utils, hf_utils, utils
from slicegpt.config import config

utils.configure_logging()

os.environ["WANDB__SERVICE_WAIT"] = "300"


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="OPT model to load; pass `facebook/opt-125m`.",
        choices=[
            # OPT models
            "facebook/opt-125m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
            "facebook/opt-13b",
            "facebook/opt-30b",
            "facebook/opt-66b",
            # LLAMA 2 Models
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-13b-hf',
            'meta-llama/Llama-2-70b-hf',
        ],
        default="facebook/opt-125m",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        help="Dataset to evaluate on.",
        choices=["wikitext2", "ptb", "c4"],
        default="wikitext2",
    )
    parser.add_argument(
        "--ntokens",
        type=int,
        help="Number of tokens to benchmark over.",
        default=128,
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for loading the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--load-model-path", type=str, default=None, help="Path to load the sliced model from.")

    parser.add_argument('--hf-token', type=str, default=None)

    args = parser.parse_args()

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    return args


def main():
    logging.info("Running benchmarking of a sliced model.")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    args = argparser()

    try:
        wandb.init(project="slicegpt-bench", config=args)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb.')
        wandb.init(project="slicegpt", mode='disabled')

    if args.load_model_path:
        # load the model from load_model_path to compute perplexity and skip rotation and slicing
        logging.info(f"Loading sliced {args.model} model from {args.load_model_path} with sparsity {args.sparsity}")
        model, tokenizer = hf_utils.load_sliced_model(args.model, args.load_model_path, args.sparsity, args.hf_token)
    else:
        # load one of the pre-trained models
        model, tokenizer = hf_utils.get_model(args.model, token=args.hf_token)

    if args.distribute_model:
        # distribute model across available GPUs
        gpu_utils.distribute_model(model)
    else:
        model = model.to(config.device)

    dataloader, _ = data_utils.get_loaders(
        dataset_name=args.eval_dataset,
        nsamples=args.batch_size,
        tokenizer=tokenizer,
        seqlen=args.ntokens,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    results = gpu_utils.benchmark(model, next(iter(dataloader)))
    logging.info(f"Median time per batch: {results['median_time']} s/batch.")
    logging.info(f"Throughput: {results['throughput']} token/s.")
    logging.info(f"Latency: {results['latency']} s/token.")
    wandb.log(results)


if __name__ == "__main__":
    main()
