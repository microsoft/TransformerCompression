# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os

import torch
import wandb

from slicegpt import data_utils, gpu_utils, hf_utils, utils
from slicegpt.config import config


def benchmarking_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )

    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
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

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    parser.add_argument('--wandb-project', type=str, default="slicegpt-bench", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    return parser.parse_args() if interactive else parser.parse_args('')


def process_benchmarking_args(args: argparse.Namespace):
    logging.debug(f'Parsed arguments:')
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")


def benchmarking_main(args: argparse.Namespace) -> None:
    logging.info("Running benchmarking of a sliced model.")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    if args.sliced_model_path:
        # load the model from sliced_model_path to compute perplexity and skip rotation and slicing
        logging.info(f"Loading sliced {args.model} model from {args.sliced_model_path} with sparsity {args.sparsity}")
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model, args.sliced_model_path, sparsity=args.sparsity, token=args.hf_token
        )
    else:
        # load one of the pre-trained models
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
            args.model, args.model_path, token=args.hf_token, dtype=config.dtype
        )

    if args.distribute_model:
        # distribute model across available GPUs
        gpu_utils.distribute_model(model_adapter)
    else:
        model_adapter.model.to(config.device)

    dataset = data_utils.get_dataset(args.eval_dataset)
    train_loader = data_utils.prepare_dataloader(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=model_adapter.seqlen,
        batch_size=args.batch_size,
        nsamples=args.ntokens,
        seed=args.seed,
    )

    results = gpu_utils.benchmark(model_adapter, next(iter(train_loader)))
    logging.info(f"Median time per batch: {results['median_time']} s/batch.")
    logging.info(f"Throughput: {results['throughput']} token/s.")
    logging.info(f"Latency: {results['latency']} s/token.")
    wandb.log(results)


if __name__ == "__main__":
    utils.configure_logging()
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    benchmarking_args = benchmarking_arg_parser()
    process_benchmarking_args(benchmarking_args)
    benchmarking_main(benchmarking_args)
