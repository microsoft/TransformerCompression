# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os

import lm_eval
import torch
import transformers
import wandb
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

from quarot import hf_utils, rotation_utils
from quarot.modeling_llama import QuarotLlamaConfig, QuarotLlamaForCausalLM
from slicegpt import data_utils, gpu_utils, layernorm_fusion, utils
from slicegpt.config import config


def quarot_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate and calculate perplexity on.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument(
        "--ppl-eval-nsamples", type=int, default=128, help="Number of samples to evaluate the perplexity on."
    )
    parser.add_argument("--eval-baseline", action="store_true", help="Evaluate the baseline model.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument('--wandb-project', type=str, default="quarot", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    # Rotation Arguments
    parser.add_argument(
        '--rotation-seed',
        type=int,
        default=0,
        help='Seed for generating random matrix. Use 0 to replicate paper results.',
    )
    parser.add_argument(
        '--fp32-had',
        action="store_true",
        default=False,
        help='Apply Hadamard rotation in FP32 (default: False means FP16)',
    )

    # LM Eval Arguments
    parser.add_argument("--lm-eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada"],
    )
    parser.add_argument(
        '--lm-eval-batch-size', type=int, default=128, help='Batch size for evaluating with lm eval harness.'
    )

    return parser.parse_args() if interactive else parser.parse_args('')


def process_quarot_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if args.device:
        config.device = torch.device(args.device)

    if args.distribute_model:
        raise NotImplementedError("Distributed evaluation is not supported yet.")

    config.dtype = torch.float16


def quarot_main(args: argparse.Namespace) -> None:
    logging.info("Running QuaRot experiment.")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    torch.set_default_dtype(config.dtype)

    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    # load one of the pre-trained models
    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
        args.model, args.model_path, token=args.hf_token, dtype=config.dtype
    )

    model = model_adapter.model

    dataset = data_utils.get_dataset(args.cal_dataset)
    test_dataset = dataset["test"]
    test_loader = data_utils.prepare_test_dataloader(
        dataset=test_dataset, tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
    )

    # original ppl
    if args.eval_baseline:
        model.to(config.device)
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Original ppl: {dataset_ppl:.4f}')
        wandb.log({"original_ppl": dataset_ppl})
        model.cpu()
        utils.cleanup_memory()

    # fuse layernorms
    layernorm_fusion.fuse_modules(model_adapter)  # TODO: fix expected adapter type

    # Rotate the model with fused Hadamard transformations.
    rotation_utils.rotate_model(model_adapter, args.rotation_seed)

    model_config = QuarotLlamaConfig.from_pretrained(args.model, dtype=config.dtype)
    model_config._attn_implementation = "flash_attention_2"
    with transformers.modeling_utils.no_init_weights():
        # initialise quarot model
        quarot_llama = QuarotLlamaForCausalLM(online_had_mlp=True, online_had_attn=True, config=model_config)

        # load the rotated weights into the quarot model
        quarot_llama.load_state_dict(model_adapter.model.state_dict(), strict=False)

    quarot_llama.to(config.device)
    dataset_ppl = gpu_utils.evaluate_ppl(quarot_llama, quarot_llama.config.pad_token_id, test_loader)
    logging.info(f'QuaRot ppl: {dataset_ppl:.4f}')
    wandb.log({"quarot_ppl": dataset_ppl})

    if not args.lm_eval:
        return

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    initialize_tasks()
    task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    results = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=args.lm_eval_batch_size)['results']

    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    logging.info(f"LM Eval results: {metric_vals}")
    wandb.log(metric_vals)


if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    quarot_args = quarot_arg_parser()
    process_quarot_args(quarot_args)
    quarot_main(quarot_args)
