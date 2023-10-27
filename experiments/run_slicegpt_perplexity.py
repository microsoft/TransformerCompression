# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os 

import torch

import wandb
from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate

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

    parser.add_argument("--save_dir", type=str, default=None, help="Path to save the model.")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to load the sliced model from.")

    parser.add_argument('--hf_token', type=str, default=None)

    args = parser.parse_args()
    assert args.sparsity >= 0 and args.sparsity <= 1, "Sparsity should be in the range [0, 1]!"

    return args


def main():
    print("Running slicing experiment.")

    args = argparser()

    try:
        wandb.init(project="slicegpt", config=args)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        print(f'Failed to initialize wandb: {e}, continuing without wandb.')
        wandb.init(project="slicegpt", mode='disabled')

    if args.load_model_path:
        # load the model from load_model_path to compute perplexity and skip rotation and slicing
        print(f"Loading sliced {args.model} model from {args.load_model_path} with sparsity {args.sparsity}")
        model, tokenizer = hf_utils.load_sliced_model(args.model, args.load_model_path, args.sparsity, DEV)

        dataloader, testloader = data_utils.get_loaders(
            dataset_name=args.cal_dataset,
            tokenizer=tokenizer,
            nsamples=args.cal_nsamples,
            seqlen=model.seqlen,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        dataset_ppl = gpu_utils.evaluate_ppl(model, testloader, DEV)
        print('\nPerplexity after rotating and slicing', dataset_ppl)
        wandb.log({"sliced_ppl": dataset_ppl})

    else:
        # get model, data
        model, tokenizer = hf_utils.get_model(args.model)

        dataloader, testloader = data_utils.get_loaders(
            dataset_name=args.cal_dataset,
            tokenizer=tokenizer,
            nsamples=args.cal_nsamples,
            seqlen=model.seqlen,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        # original ppl
        if args.eval_baseline:
            dataset_ppl = gpu_utils.evaluate_ppl(model, testloader, DEV)
            print('Original ppl:', dataset_ppl)
            wandb.log({"original_ppl": dataset_ppl})
        # fuse layernorms, add shorcuts, check perplexity
        layernorm_fusion.replace_modules(model, model.config)

        model = model.cpu()
        layernorm_fusion.fuse_modules(model)

        if args.eval_fused_model:
            dataset_ppl = gpu_utils.evaluate_ppl(model, testloader, DEV)
            print('Post-fusion:', dataset_ppl)
            wandb.log({"post_fusion_ppl": dataset_ppl})

        # compute new embedding dimension given the slicegpt sparsity
        new_embedding_dimension = int((1 - args.sparsity) * model.config.hidden_size)
        print(f"New embedding dimension: {new_embedding_dimension} (sparsity {args.sparsity})")

        rotate.rotate_and_slice(model, dataloader, new_embedding_dimension)

        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            model_file = os.path.join(args.save_dir, os.path.basename(args.model) +  "_" + str(args.sparsity) + ".pt")
            torch.save(model.state_dict(), model_file)
            print("Saved sliced model to {}".format(args.save_dir))

        dataset_ppl = gpu_utils.evaluate_ppl(model, testloader, DEV)
        print('After rotating and slicing', dataset_ppl)
        wandb.log({"sliced_ppl": dataset_ppl})


if __name__ == "__main__":
    main()
