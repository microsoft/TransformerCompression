# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import gc
import os
import logging

import torch

import wandb
from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils

utils.configure_logging()

os.environ["WANDB__SERVICE_WAIT"] = "300"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        "--cal_dataset",
        type=str,
        help="Dataset to calibrate on.",
        choices=["wikitext2", "ptb", "c4"],
        default="wikitext2",
    )
    parser.add_argument(
        "--cal_nsamples",
        type=int,
        help="Number of samples of the calibration data to load.",
        default=128,
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for loading the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1])"
    )
    parser.add_argument("--eval_baseline", action="store_true", help="Evaluate the baseline model.")
    parser.add_argument("--eval_fused_model", action="store_true", help="Evaluate the fused model.")
    parser.add_argument("--ppl_only", action="store_true", help="Evaluate the loaded model without doing compression.")
    parser.add_argument(
        "--distribute_model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--save_dir", type=str, default=None, help="Path to save the model.")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to load the sliced model from.")

    parser.add_argument('--hf_token', type=str, default=None)

    args = parser.parse_args()
    assert args.sparsity >= 0 and args.sparsity <= 1, "Sparsity should be in the range [0, 1]!"

    return args


def main():
    logging.info("Running SliceGPT perplexity experiment.")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    args = argparser()

    try:
        wandb.init(project="slicegpt", config=args)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb.')
        wandb.init(project="slicegpt", mode='disabled')

    if args.load_model_path:
        # load the model from load_model_path to compute perplexity and skip rotation and slicing
        logging.info(f"Loading sliced {args.model} model from {args.load_model_path} with sparsity {args.sparsity}")
        model, tokenizer = hf_utils.load_sliced_model(args.model, args.load_model_path, args.sparsity, DEV)
    else:
        # load one of the pre-trained models
        model, tokenizer = hf_utils.get_model(args.model, token=args.hf_token)

    dataloader, testloader = data_utils.get_loaders(
        dataset_name=args.cal_dataset,
        tokenizer=tokenizer,
        nsamples=args.cal_nsamples,
        seqlen=model.seqlen,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # evaluate perplexity and exit if sliced model is loaded or if ppl_only is set
    if args.load_model_path or args.ppl_only:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(model)
        else:
            model = model.to(DEV)
        
        dataset_ppl = gpu_utils.evaluate_ppl(model, testloader, DEV)
        logging.info(f'Loaded model perplexity: {dataset_ppl}')
        wandb.log({"original_ppl": dataset_ppl})
        return

    # original ppl
    if args.eval_baseline:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(model)
        else:
            model = model.to(DEV)
        dataset_ppl = gpu_utils.evaluate_ppl(model, testloader, DEV)
        logging.info(f'Original ppl: {dataset_ppl}')
        wandb.log({"original_ppl": dataset_ppl})
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    # fuse layernorms, add shorcuts, check perplexity
    layernorm_fusion.replace_modules(model, model.config)

    # gc.collect and empty cache are necessary to clean up GPU memory
    # if the model was distributed
    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    layernorm_fusion.fuse_modules(model)

    # don't run this on large and/or distributed models
    if args.eval_fused_model and not args.distribute_model:
        model = model.to(DEV)

        dataset_ppl = gpu_utils.evaluate_ppl(model, testloader, DEV)
        logging.info(f'Post-fusion: {dataset_ppl}')
        wandb.log({"post_fusion_ppl": dataset_ppl})

        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    # compute new embedding dimension given the slicegpt sparsity
    new_embedding_dimension = int((1 - args.sparsity) * model.config.hidden_size)
    logging.info(f"New embedding dimension: {new_embedding_dimension} (sparsity {args.sparsity})")

    rotate.rotate_and_slice(model, dataloader, new_embedding_dimension)

    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        model_file = os.path.join(args.save_dir, os.path.basename(args.model) + "_" + str(args.sparsity) + ".pt")
        torch.save(model.state_dict(), model_file)
        logging.info(f"Saved sliced model to {args.save_dir}")

    if args.distribute_model:
        gpu_utils.distribute_model(model)
    else:
        model = model.to(DEV)

    dataset_ppl = gpu_utils.evaluate_ppl(model, testloader, DEV)
    logging.info(f'After rotating and slicing {dataset_ppl}')
    wandb.log({"sliced_ppl": dataset_ppl})


if __name__ == "__main__":
    main()
