# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os
import sys

import lm_eval
import numpy as np
import torch
import transformers
import wandb
from datasets import Dataset
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from quarot.hadamard_utils import random_hadamard_matrix
from quarot import gptq, hf_utils, rotate, rtn
from quarot.adapters.llama_adapter import LlamaModelAdapter
from quarot.adapters.mixtral_adapter import MixtralModelAdapter
from quarot.adapters.phi3_adapter import Phi3ModelAdapter
from quarot.adapters.phi35_adapter import Phi35ModelAdapter
from quarot.hf_utils import get_quarot_model, quarot_model_config
from quarot.modeling_llama import QuarotLlamaForCausalLM
from quarot.modeling_mixtral import QuarotMixtralForCausalLM
from quarot.modeling_phi3 import QuarotPhi3ForCausalLM
from quarot.modeling_phi35 import QuarotPhi35ForCausalLM
from slicegpt import data_utils, gpu_utils, layernorm_fusion, utils
from slicegpt.config import config


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
        '--rotate',
        type=str2bool,
        default=False,
        help='Apply QuaRot/Hadamard rotation to the model.',
    )
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

    # Weight Quantization Arguments
    parser.add_argument(
        '--w-rtn',
        action="store_true",
        help='Quantize weights using RTN.',
    )
    parser.add_argument(
        '--w-gptq',
        action="store_true",
        help='Quantize weights using GPTQ.',
    )
    parser.add_argument(
        "--gptq-damping", type=float, default=0.01, help="Damping factor for GPTQ. (ignored for RTN quantization)"
    )
    parser.add_argument(
        "--gptq-opt-scales", action="store_true", help="Optimize scales for GPTQ (ignored for RTN quantization)"
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
    parser.add_argument(
        '--w-bits',
        type=int,
        default=16,
        help='Number of bits to quantize the weights to.',
    )
    parser.add_argument(
        '--w-asym',
        type=str2bool,
        default=False,
        help='Asymmetric weight quantization (else symmetric by default).',
    )
    parser.add_argument('--w-groupsize', type=int, default=None, help='Group size for groupwise weight quantization.')

    # Activation Quantization Arguments
    parser.add_argument(
        '--a-bits',
        type=int,
        default=16,
        help='Number of bits to quantize the weights to.',
    )
    parser.add_argument(
        '--a-clip-ratio',
        type=float,
        default=1.0,
        help='Clip ratio for activation quantization: new_max = max * clip_ratio.',
    )
    parser.add_argument(
        '--a-quantile',
        type=float,
        default=None,
        help='Quantile for activation quantization, default is None.',
    )
    parser.add_argument(
        '--a-groupsize',
        type=int,
        default=None,
        help='Group size for groupwise activation quantization, default is None.',
    )

    # KV Quantization Arguments
    parser.add_argument(
        '--k-bits',
        type=int,
        default=16,
        help='Number of bits to quantize the keys to.',
    )
    parser.add_argument(
        '--k-clip-ratio',
        type=float,
        default=1.0,
        help='Clip ratio for keys quantization: new_max = max * clip_ratio.',
    )
    parser.add_argument(
        '--k-quantile',
        type=float,
        default=None,
        help='Quantile for keys quantization, default is None.',
    )
    parser.add_argument(
        '--k-groupsize',
        type=int,
        default=None,
        help='Group size for groupwise keys quantization.',
    )
    parser.add_argument(
        '--v-bits',
        type=int,
        default=16,
        help='Number of bits to quantize the values to.',
    )
    parser.add_argument(
        '--v-clip-ratio',
        type=float,
        default=1.0,
        help='Clip ratio for values quantization: new_max = max * clip_ratio.',
    )
    parser.add_argument(
        '--v-quantile',
        type=float,
        default=None,
        help='Quantile for values quantization, default is None.',
    )
    parser.add_argument(
        '--v-groupsize',
        type=int,
        default=None,
        help='Group size for groupwise values quantization.',
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

    # if the user passed groupsize of zero, interpret this the same as None
    if args.a_groupsize == 0:
        args.a_groupsize = None
    if args.k_groupsize == 0:
        args.k_groupsize = None
    if args.v_groupsize == 0:
        args.v_groupsize = None
    if args.w_groupsize == 0:
        args.w_groupsize = None

    config.dtype = torch.float16

    if not args.model_path:
        args.model_path = args.model

    os.makedirs(args.save_dir, exist_ok=True)
    with open(f"{args.save_dir}/args.json", 'w') as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)


def run_lm_eval(
    hflm: HFLM, task_list: list, fewshot: int, batch_size: int, fraction: float | None, output_file: str, log_msg: str
):
    if fraction and fraction < 1:
        limit = fraction
    else:
        limit = None
    results = lm_eval.simple_evaluate(
        hflm, tasks=task_list, num_fewshot=fewshot, batch_size=batch_size, limit=limit
    )['results']
    metrics = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metrics['acc_avg'] = round(sum(metrics.values()) / len(metrics.values()), 4)
    metrics['num_fewshot'] = fewshot
    metrics['limit'] = limit
    logging.info(f"{log_msg} {metrics}")
    with open(output_file, "w") as f:
        json.dump(metrics, f)


def quarot_main(args: argparse.Namespace) -> None:
    logging.info("Running QuaRot experiment.")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

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

    if args.cal_dataset in data_utils.ds_properties:
        dataset = data_utils.get_dataset(args.cal_dataset)
        test_dataset = dataset["test"]
        train_dataset = dataset["train"]
    elif os.path.exists(args.cal_dataset):
        train_texts, test_texts = data_utils.format_dataset_from_path(args.cal_dataset, tokenizer)
        train_dataset = Dataset.from_dict({"text": train_texts})
        test_dataset = Dataset.from_dict({"text": test_texts})
    else:
        raise NotImplementedError("The provided dataset is not supported")

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

    if args.rotate:
        # fuse layernorms
        layernorm_fusion.fuse_modules(model_adapter)  # TODO: fix expected adapter type

        # Rotate the model with fused Hadamard transformations.
        rotate.rotate_model(model_adapter, args.no_unfused_Had, args.rotation_seed)

        # Save Q matrix
        Q = random_hadamard_matrix(model_adapter.model.config.hidden_size, seed=args.rotation_seed)
        torch.save(Q, f"{args.save_dir}/Q_mat.pt")
        np.save(f"{args.save_dir}/Q.npy", Q.numpy())

    model_config = quarot_model_config(
        args.model, args.model_path, dtype=config.dtype, groupsize=args.w_groupsize, offset=args.w_asym
    )

    with transformers.modeling_utils.no_init_weights():
        # initialize quarot model

        act_args = {
            'a_bits': args.a_bits,
            'a_clip_ratio': args.a_clip_ratio,
            'a_groupsize': args.a_groupsize,
            'a_quantile': args.a_quantile,
        }
        key_args = {
            'k_bits': args.k_bits,
            'k_clip_ratio': args.k_clip_ratio,
            'k_groupsize': args.k_groupsize,
            'k_quantile': args.k_quantile,
        }
        value_args = {
            'v_bits': args.v_bits,
            'v_clip_ratio': args.v_clip_ratio,
            'v_groupsize': args.v_groupsize,
            'v_quantile': args.v_quantile,
        }
        quarot_model = get_quarot_model(
            model_name=args.model,
            rotate=args.rotate,
            no_unfused_Had=args.no_unfused_Had,
            act_args=act_args,
            key_args=key_args,
            value_args=value_args,
            model_config=model_config,
        )

        quarot_model = quarot_model.to(config.dtype)

        # load the rotated weights into the quarot model
        quarot_model.load_state_dict(model_adapter.model.state_dict(), strict=False)

    # Wrap the quarot model in an adapter, required for GPTQ and for distributing the model across GPUs.
    if isinstance(quarot_model, QuarotLlamaForCausalLM):
        quarot_model_adapter = LlamaModelAdapter(quarot_model)
    elif isinstance(quarot_model, QuarotPhi3ForCausalLM):
        quarot_model_adapter = Phi3ModelAdapter(quarot_model)
    elif isinstance(quarot_model, QuarotPhi35ForCausalLM):
        quarot_model_adapter = Phi35ModelAdapter(quarot_model)
    elif isinstance(quarot_model, QuarotMixtralForCausalLM):
        quarot_model_adapter = MixtralModelAdapter(quarot_model)
    else:
        raise ValueError("Adapter for QuaRot model must be specified.")

    if args.w_rtn:
        logging.info(f"Quantizing weights to INT{args.w_bits} using RTN.")
        rtn.quantize_model_rtn(
            quarot_model, bits=args.w_bits, groupsize=args.w_groupsize, symmetric=False if args.w_asym else True
        )
        logging.info("Quantization complete.")
    elif args.w_gptq:
        logging.info(f"Quantizing weights to INT{args.w_bits} using GPTQ.")
        train_loader = data_utils.prepare_dataloader(
            dataset=train_dataset,
            tokenizer=tokenizer,
            batch_size=args.cal_batch_size,
            nsamples=args.cal_nsamples,
        )

        gptq.quantize_model_gptq(
            quarot_model_adapter,
            train_loader,
            bits=args.w_bits,
            symmetric=False if args.w_asym else True,
            damping=args.gptq_damping,
            groupsize=args.w_groupsize,
            optimize_scales=args.gptq_opt_scales,
        )
        logging.info("Quantization complete.")
    else:
        logging.info("No weight quantization performed")

    os.makedirs(args.save_dir, exist_ok=True)
    logging.info(f" Saving model to {args.save_dir}")
    quarot_model.to(config.device).eval()
    quarot_model.save_pretrained(args.save_dir)
    model_config.save_pretrained(args.save_dir)

    def reset_model_device() -> None:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(quarot_model_adapter)
        else:
            quarot_model.to(config.device)

    reset_model_device()
    dataset_ppl = gpu_utils.evaluate_ppl(quarot_model, quarot_model.config.pad_token_id, test_loader)

    if args.rotate:
        logging.info(f'QuaRot ppl: {dataset_ppl:.4f}')
        wandb.log({"quarot_ppl": dataset_ppl})
    else:
        logging.info(f'ppl: {dataset_ppl:.4f}')
        wandb.log({"ppl": dataset_ppl})

    if not args.lm_eval:
        return

    hflm = HFLM(pretrained=quarot_model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    initialize_tasks()
    task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)

    run_lm_eval(
        hflm,
        task_list=task_names,
        fewshot=0,
        batch_size=args.lm_eval_batch_size,
        fraction=None,
        output_file=f"{args.save_dir}/lm_eval.json",
        log_msg="LM Eval results (limit=1): ",
    )

    run_lm_eval(
        hflm,
        task_list=task_names,
        fewshot=0,
        batch_size=args.lm_eval_batch_size,
        fraction=0.15,
        output_file=f"{args.save_dir}/lm_eval_sub.json",
        log_msg="LM Eval results (limit=0.15): ",
    )

    run_lm_eval(
        hflm,
        task_list=['mmlu'],
        fewshot=5,
        batch_size=args.lm_eval_batch_size,
        fraction=None,
        output_file=f"{args.save_dir}/mmlu_eval.json",
        log_msg="MMLU Eval results (limit=1): ",
    )

    run_lm_eval(
        hflm,
        task_list=['mmlu'],
        fewshot=5,
        batch_size=args.lm_eval_batch_size,
        fraction=0.15,
        output_file=f"{args.save_dir}/mmlu_eval_sub.json",
        log_msg="MMLU Eval results (limit=0.15): ",
    )


if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    quarot_args = quarot_arg_parser()
    process_quarot_args(quarot_args)
    quarot_main(quarot_args)
