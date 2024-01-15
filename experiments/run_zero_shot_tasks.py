# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os
import time

import lm_eval
import torch
import wandb
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

from slicegpt import hf_utils, utils
from slicegpt.gpu_utils import sync_gpus
from slicegpt.config import config

utils.configure_logging()

os.environ["WANDB__SERVICE_WAIT"] = "300"


def parse_args() -> argparse.Namespace:
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
            # Phi 2 Models
            'microsoft/phi-2',
        ],
        default="facebook/opt-125m",
    )
    parser.add_argument(
        "--load-model-path", type=str, default=None, help="Path to load the sliced model from.",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument('--hf-token', type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluating with lm eval harness.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument('--time', action="store_true", help="Time the evaluation.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"],
        choices=lm_eval_utils.MultiChoice(tasks.ALL_TASKS),
    )
    parser.add_argument('--num-fewshot', type=int, default=0, help="Number of fewshots for all tasks.")
    return parser.parse_args()


def main() -> None:
    logging.info("Running SliceGPT zeroshot tasks experiment.")

    initialize_tasks()
    args = parse_args()

    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project="slicegpt-lm-eval", config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb.')
        wandb.init(project="slicegpt-lm-eval", mode='disabled')

    if args.load_model_path:
        # load the sliced model
        logging.info(f"Loading sliced {args.model} model from {args.load_model_path} with sparsity {args.sparsity}")
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model, args.load_model_path, args.sparsity, token=args.hf_token
        )
    else:
        # load the original model
        logging.info(f"Loading {args.model} model")
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model, token=args.hf_token)

    # the lm eval harness ties the weights, but this should not be done for sliced models unless the lm_head was sliced
    model_adapter.model.tie_weights = lambda: None

    if args.distribute_model:
        # distribute model across available GPUs
        gpu_utils.distribute_model(model_adapter)
    else:
        model_adapter.model.to(config.device)

    ### LM Eval Harness ###
    hflm = HFLM(pretrained=model_adapter.model, tokenizer=tokenizer, batch_size=args.batch_size)

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)

    mmlu_task_num_questions = {}
    for task_name in task_names:
        if 'mmlu' in task_name:
            mmlu_task_num_questions[task_name] = lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows

    logging.info(f"Selected Tasks: {task_names}")

    if args.time:
        sync_gpus()
        start_time = time.time()

    results = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=args.num_fewshot, batch_size=args.batch_size)['results']

    if args.time:
        sync_gpus()
        elapsed = time.time() - start_time
        wandb.log({'time_eval': elapsed})
        logging.info(
            "Time spent on evaluation: %s",
            time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
        )

    wandb.log(results)
    logging.info(json.dumps(results, indent=2))

    # calculate the avg across the tasks
    n_tasks = len(task_names)
    acc_cumul = 0
    acc_mmlu = 0

    # Iterate over tasks and accumulate results
    for task, result in results.items():
        acc = result.get('acc_norm,none', result['acc,none'])
        if 'mmlu' in task:
            acc_mmlu += acc * mmlu_task_num_questions[task]
        else:
            acc_cumul += acc

    # Calculate average accuracy for mmlu tasks if any
    acc_mmlu_avg = acc_mmlu / sum(mmlu_task_num_questions.values()) if mmlu_task_num_questions else 0
    wandb.log({'acc_mmlu_avg': acc_mmlu_avg})

    # Calculate average accuracy
    acc_avg = (acc_cumul + acc_mmlu_avg) / n_tasks

    wandb.log({'acc_avg': acc_avg})
    logging.info(f"Average accuracy across tasks: {acc_avg}")


if __name__ == "__main__":
    main()
