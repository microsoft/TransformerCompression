# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os

import torch
from transformers import LlamaForCausalLM, OPTForCausalLM

import wandb
from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.adapters import llama_adapter, opt_adapter
from slicegpt.config import config
from slicegpt.model_adapter import ModelAdapter

utils.configure_logging()

os.environ["WANDB__SERVICE_WAIT"] = "300"


def argparser() -> argparse.Namespace:
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
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate on.",
        choices=["wikitext2", "ptb", "c4"],
        default="wikitext2",
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples of the calibration data to load.",
        default=128,
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for loading the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument("--eval-baseline", action="store_true", help="Evaluate the baseline model.")
    parser.add_argument("--eval-fused-model", action="store_true", help="Evaluate the fused model.")
    parser.add_argument("--ppl-only", action="store_true", help="Evaluate the loaded model without doing compression.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--save-dir", type=str, default=None, help="Path to save the model.")
    parser.add_argument("--load-model-path", type=str, default=None, help="Path to load the sliced model from.")

    parser.add_argument('--hf-token', type=str, default=None)

    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    args = parser.parse_args()

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

    return args


def main() -> None:
    logging.info("Running SliceGPT perplexity experiment")

    args = argparser()

    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project="slicegpt", config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project="slicegpt", mode='disabled')

    if args.load_model_path:
        # load the model from load_model_path to compute perplexity and skip rotation and slicing
        logging.info(f"Loading sliced {args.model} model from {args.load_model_path} with sparsity {args.sparsity}")
        model, tokenizer = hf_utils.load_sliced_model(args.model, args.load_model_path, args.sparsity)
    else:
        # load one of the pre-trained models

        model, tokenizer = hf_utils.get_model(args.model, token=args.hf_token, dtype=config.dtype)

    if isinstance(model, LlamaForCausalLM):
        adapter: ModelAdapter = llama_adapter.LlamaModelAdapter(model)
    elif isinstance(model, OPTForCausalLM):
        adapter = opt_adapter.OPTModelAdapter(model)
    else:
        raise TypeError

    def reset_model_device() -> None:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(adapter)
        else:
            adapter.raw_model.to(config.device)

    dataloader, testloader = data_utils.get_loaders(
        dataset_name=args.cal_dataset,
        tokenizer=tokenizer,
        nsamples=args.cal_nsamples,
        seqlen=adapter.seqlen,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # evaluate perplexity and exit if sliced model is loaded or if ppl_only is set
    if args.load_model_path or args.ppl_only:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(adapter, testloader)
        logging.info(f'Loaded model perplexity: {dataset_ppl}')
        wandb.log({"original_ppl": dataset_ppl})
        return

    # original ppl
    if args.eval_baseline:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(adapter, testloader)
        logging.info(f'Original ppl: {dataset_ppl:.4f}')
        wandb.log({"original_ppl": dataset_ppl})
        model = model.cpu()
        utils.cleanup_memory()

    # replace modules with compressible equivalents
    layernorm_fusion.replace_modules(adapter)

    # fuse layernorms and add rotations to skip connections
    layernorm_fusion.fuse_modules(adapter)

    # don't run this on large and/or distributed models
    if args.eval_fused_model and not args.distribute_model:
        model = model.to(config.device)

        dataset_ppl = gpu_utils.evaluate_ppl(adapter, testloader)
        logging.info(f'Post-fusion: {dataset_ppl:.4f}')
        wandb.log({"post_fusion_ppl": dataset_ppl})

        model = model.cpu()

        # run GC and cleanup GPU memory
        utils.cleanup_memory()

    original_param_count = sum(int(p.nelement()) for p in model.parameters())
    logging.info(f'Original model parameters: {original_param_count:,d}')

    # compute new embedding dimension given the desired sparsity level
    new_embedding_dimension = int((1 - args.sparsity) * model.config.hidden_size)
    logging.info(f"New embedding dimension: {new_embedding_dimension} (sparsity {args.sparsity})")

    rotate.rotate_and_slice(model, dataloader, new_embedding_dimension)

    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        model_file = os.path.join(args.save_dir, os.path.basename(args.model) + "_" + str(args.sparsity) + ".pt")
        torch.save(model.state_dict(), model_file)
        logging.info(f"Saved sliced model to {args.save_dir}")

    reset_model_device()
    dataset_ppl = gpu_utils.evaluate_ppl(adapter, testloader)
    logging.info(f'After rotating and slicing {dataset_ppl:.4f}')
    wandb.log({"sliced_ppl": dataset_ppl})

    sliced_param_count = sum(int(p.nelement()) for p in model.parameters())
    sliced_fraction = 1.0 - sliced_param_count / original_param_count
    logging.info(f'Sliced model parameters: {sliced_param_count:,d} (sliced fraction {sliced_fraction:.4f})')


if __name__ == "__main__":
    main()
