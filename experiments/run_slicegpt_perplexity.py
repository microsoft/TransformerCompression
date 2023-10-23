# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

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
        help="Dataset to calibrate.",
        choices=["wikitext2", "ptb", "c4"],
        default="wikitext2",
    )
    parser.add_argument(
        "--cal_nsamples",
        type=int,
        help="Number of samples to calibrate on.",
        default=128,
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        help="Dataset to evaluate.",
        choices=["wikitext2", "ptb", "c4"],
        default="wikitext2",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument("--sparsity", type=float, default=0.0, help="Sparsity of the calibration data.")
    parser.add_argument("--eval_baseline", action="store_true", help="Evaluate the baseline model.")
    parser.add_argument("--debug", action="store_true", help="Evaluate the fused model.")

    parser.add_argument("--save_dir", type=str, default=None, help="Path to save the model.")

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

    # get model, data
    model, tokenizer = hf_utils.get_model(args.model, args.hf_token)
    dataloader, testloader = data_utils.get_loaders(
        "wikitext2",
        seqlen=model.seqlen,
        batch_size=args.batch_size,
        seed=args.seed,
        tokenizer=tokenizer,
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

    if args.debug:
        dataset_ppl = gpu_utils.evaluate_ppl(model, testloader, DEV)
        print('Post-fusion:', dataset_ppl)
        wandb.log({"post_fusion_ppl": dataset_ppl})

    # run slicegpt sparsity
    new_embedding_dimension = int((1 - args.sparsity) * model.config.hidden_size)
    print(f"New embedding dimension: {new_embedding_dimension} (sparsity {args.sparsity})")

    rotate.rotate_and_slice_opt(model, dataloader, new_embedding_dimension)
    print()
    dataset_ppl = gpu_utils.evaluate_ppl(model, testloader, DEV)
    print('After rotating and slicing', dataset_ppl)
    wandb.log({"sliced_ppl": dataset_ppl})


"""
def old_main():
    args = opt_argparser()
    if args.dp2_cache:
        utils.deeplearn2_cache_dir()

    if args.wandb:
        wandb.init(
            project="llm_rotation",
            entity="saleh_ashkboos",
            tags=["FP16", "static_sparsification", "fp64_pca"],
        )
        wandb.config.update(args)

    utils.set_seed(args.seed)

    model = opt_utils.get_opt(args.model)

    if args.benchmark_baseline:
        gpus = [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]
        print("Using GPUs:", gpus)
        torch.cuda.empty_cache()
        dataloader, testloader = datautils.get_loaders(
            args.eval_dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        if len(gpus) > 1:
            opt_utils.opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        input_ids = next(iter(dataloader))[0][:, :128]  # benchmark over 128 tokens
        baseline_token_per_sec = opt_utils.opt_benchmark(
            model, input_ids, dev=DEV, check=True
        )
        print(
            f"\nBaseline Model ({args.eval_dataset.upper()}) (real) Sec/Token: {baseline_token_per_sec:.4f} ({len(gpus)} GPUs)"
        )
        if args.wandb:
            wandb.log(
                {
                    "token_per_sec_baseline/{}".format(
                        args.eval_dataset
                    ): baseline_token_per_sec
                }
            )
        exit(2)

    if args.eval_baseline:
        model.eval()
        dataloader, testloader = datautils.get_loaders(
            args.eval_dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        tick = time.time()
        dataset_ppl = opt_utils.opt_eval(model, testloader, DEV)
        tock = time.time()
        print(
            f"\nBaseline Model ({args.eval_dataset.upper()}) PPL: {dataset_ppl:.3f} \n (simulate) Time: {tock-tick:.4f}"
        )
        print(40 * "-")
        model = model.cpu()
        if args.wandb:
            wandb.log({"ppl_baseline/{}".format(args.eval_dataset): dataset_ppl})
            wandb.log(
                {"(simulate) time_baseline/{}".format(args.eval_dataset): tock - tick}
            )

    if args.sparsity > 0:
        # The order of calling these functions is important!
        opt_utils.replace_opt_modules(model, model.config)
        opt_utils.fuse_opt_modules(model)
        model = model.cpu()

    if args.benchmark and not args.ppl_check:
        model = slice_OPT_model(model, args)
        torch.cuda.empty_cache()
        dataloader, testloader = datautils.get_loaders(
            args.eval_dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        gpus = [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]
        print("Using GPUs:", gpus)

        if len(gpus) > 1:
            opt_utils.opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        input_ids = next(iter(dataloader))[0][:, :128]  # benchmark over 128 tokens
        baseline_token_per_sec = opt_utils.opt_benchmark(
            model, input_ids, dev=DEV, check=False
        )
        print(
            f"\nCompressed Model with {args.sparsity} ({args.eval_dataset.upper()}) (Compressed Real) Sec/Token: {baseline_token_per_sec:.4f} ({len(gpus)} GPUs)"
        )
        if args.wandb:
            wandb.log(
                {"sec_per_token/{}".format(args.eval_dataset): baseline_token_per_sec}
            )
        exit(2)

    if args.load_dir is None and args.sparsity > 0:
        dataloader, testloader = datautils.get_loaders(
            args.cal_dataset,
            nsamples=args.cal_nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )

        rotate_opt(model, dataloader, DEV)
        model = model.cpu()
        if args.save_dir is not None:
            save_rotated_model(model, args.model, args.save_dir)

    elif args.sparsity > 0:
        # load the model from load_dir
        print(f"Loading the model from {args.load_dir}...")
        model = slice_OPT_model(model, args) # this just makes an empty model!
        model.load_state_dict(
            torch.load(args.load_dir, map_location=torch.device("cpu"))\
        )

    if args.benchmark and args.ppl_check:
        gpus = [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]
        print("Using GPUs:", gpus)
        torch.cuda.empty_cache()

        if len(gpus) > 1:
            opt_utils.opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        
        input_ids = next(iter(dataloader))[0][:, :128]  # benchmark over 128 tokens
        baseline_token_per_sec = opt_utils.opt_benchmark(
            model, input_ids, dev=DEV, check=True
        )
        print(
            f"\nRotated Model with {args.sparsity} ({args.eval_dataset.upper()}) (Compressed Real) Sec/Token: {baseline_token_per_sec:.4f} ({len(gpus)} GPUs)"
        )
        
        if args.wandb:
            wandb.log(
                {
                    "sec_per_token_sparsified/{}".format(
                        args.eval_dataset
                    ): baseline_token_per_sec
                }
            )
        exit(2)

    tick = time.time()
    dataset_ppl = opt_utils.opt_eval(model, testloader, DEV)
    tock = time.time()
    print(
        f"\nRotated Model with {args.sparsity} ({args.eval_dataset.upper()}) PPL: {dataset_ppl:.3f} \n (simulate) Time: {tock-tick:.4f}"
    )
    print(40 * "-")
    if args.wandb:
        wandb.log({"ppl/{}".format(args.eval_dataset): dataset_ppl})
        wandb.log({"(simulate) time/{}".format(args.eval_dataset): tock - tick})
"""

if __name__ == "__main__":
    main()
