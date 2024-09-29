# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os
import sys

import lm_eval
import torch
import wandb
from datasets import Dataset
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from quarot.modeling_phi35 import QuarotPhi35ForCausalLM
from slicegpt import data_utils, gpu_utils, utils
from slicegpt.config import config

TASK_METRIC_MAP = {
    "mmlu_abstract_algebra": "acc,none",
    "mmlu_business_ethics": "acc,none",
    "mmlu_college_computer_science": "acc,none",
    "mmlu_college_mathematics": "acc,none",
    "mmlu_conceptual_physics": "acc,none",
    "mmlu_formal_logic": "acc,none",
    "mmlu_machine_learning": "acc,none",
    "mmlu_miscellaneous": "acc,none",
    "mmlu_philosophy": "acc,none",
    "mmlu_global_facts": "acc,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
    "lambada_openai": "acc,none",
    "wikitext": "word_perplexity,none",
}


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
        "--quarot-model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )

    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate and calculate perplexity on.",
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
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples of the calibration data to load for GPTQ",
        default=128,
    )
    parser.add_argument(
        "--cal-batch-size",
        type=int,
        help="Batch size of the calibration data to load for GPTQ",
        default=4,
    )

    # LM Eval Arguments
    parser.add_argument("--lm-eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument(
        "--no-unfused-Had", action="store_true", help="Uses fused Hadamards only to allow export to onnx"
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"],
    )
    parser.add_argument(
        '--lm-eval-batch-size', type=int, default=128, help='Batch size for evaluating with lm eval harness.'
    )
    parser.add_argument("--save-dir", type=str, default=".", help="Path to save the model.")

    return parser.parse_args() if interactive else parser.parse_args('')


def process_quarot_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if args.device:
        config.device = torch.device(args.device)

    config.dtype = torch.float16

    os.makedirs(args.save_dir, exist_ok=True)
    with open(f"{args.save_dir}/args.json", 'w') as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)


def run_lm_eval(hflm: HFLM, task_list: list, fewshot: int, batch_size: int, fraction: float | None, output_file: str):
    if fraction and fraction < 1:
        limit = fraction
    else:
        limit = None
    results = lm_eval.simple_evaluate(hflm, tasks=task_list, num_fewshot=fewshot, batch_size=batch_size, limit=limit)[
        'results'
    ]
    metrics = {task: round(result.get(TASK_METRIC_MAP[task]), 4) for task, result in results.items()}
    metrics['acc_avg'] = round(sum(metrics.values()) / len(metrics.values()), 4)
    metrics['num_fewshot'] = fewshot
    metrics['limit'] = limit
    logging.info(f"{metrics}")
    with open(output_file, "w") as f:
        json.dump(metrics, f)


def quarot_main(args: argparse.Namespace) -> None:
    logging.info("Running QuaRot evals.")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    model = QuarotPhi35ForCausalLM.from_pretrained(args.quarot_model_path, local_files_only=True, torch_dtype="auto")
    model = model.to(config.device)

    tokenizer = AutoTokenizer.from_pretrained(args.quarot_model_path, local_files_only=True, trust_remote_code=True)

    if args.cal_dataset in data_utils.ds_properties:
        dataset = data_utils.get_dataset(args.cal_dataset)
        test_dataset = dataset["test"]
    elif os.path.exists(args.cal_dataset):
        train_texts, test_texts = data_utils.format_dataset_from_path(args.cal_dataset, tokenizer)
        test_dataset = Dataset.from_dict({"text": test_texts})
    else:
        raise NotImplementedError("The provided dataset is not supported")

    test_loader = data_utils.prepare_test_dataloader(
        dataset=test_dataset, tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size, seqlen=args.ppl_eval_seqlen
    )

    dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)

    logging.info(f'QuaRot ppl: {dataset_ppl:.4f}')
    wandb.log({"quarot_ppl": dataset_ppl})

    if not args.lm_eval:
        return

    hflm = HFLM(
        pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size, max_length=args.ppl_eval_seqlen
    )

    initialize_tasks()
    task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)

    run_lm_eval(
        hflm,
        task_list=task_names,
        fewshot=0,
        batch_size=args.lm_eval_batch_size,
        fraction=None,
        output_file=f"{args.save_dir}/lm_eval.json",
    )


if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    quarot_args = quarot_arg_parser()
    process_quarot_args(quarot_args)
    quarot_main(quarot_args)
