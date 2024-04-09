# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os

import torch
import wandb

from quarot import hadamard_utils, quant_utils, rtn_utils
from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, utils
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
    path_group.add_argument(
        "--quarot-model-path",
        type=str,
        help="Path to load the quarot model and tokenizer from",
        default=None,
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
    parser.add_argument("--eval-quantized-model", action="store_true", help="Evaluate the quantized model.")

    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )
    parser.add_argument("--save-dir", type=str, default=None, help="Path to save the model.")
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
        action="store_true",
        default=False,
        help='''Rotate the model. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys''',
    )
    parser.add_argument('--rotate-mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--rotation-seed', type=int, default=-1, help='Seed for generating random matrix.')
    parser.add_argument(
        '--fp32-had', action="store_true", default=False, help='Apply Hadamard rotation in FP32 (default: False)'
    )

    # Activation Quantization Arguments
    parser.add_argument(
        '--a-bits',
        type=int,
        default=16,
        help='Number of bits for inputs of the Linear layers. This will be for all the linear layers in the model (including down-projection and out-projection)',
    )
    parser.add_argument(
        '--a-groupsize',
        type=int,
        default=-1,
        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize',
    )
    parser.add_argument(
        '--a-asym', action="store_true", default=False, help='ASymmetric Activation quantization (default: False)'
    )
    parser.add_argument(
        '--a-clip_ratio',
        type=float,
        default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio',
    )

    # Weight Quantization Arguments
    parser.add_argument('--w-bits', type=int, default=16, help='Number of bits for weights of the Linear layers')
    parser.add_argument(
        '--w-groupsize',
        type=int,
        default=-1,
        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize',
    )
    parser.add_argument(
        '--w-asym', action="store_true", default=False, help='ASymmetric weight quantization (default: False)'
    )
    parser.add_argument(
        '--w-rtn',
        action="store_true",
        default=False,
        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ',
    )
    parser.add_argument(
        '--w-clip',
        action="store_true",
        default=False,
        help='Clipping the weight quantization. We do not support arguments for clipping and we find the best clip ratio during the weight quantization',
    )

    # General quantization args
    parser.add_argument(
        '--int8-down-proj',
        action="store_true",
        default=False,
        help='Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8',
    )

    return parser.parse_args() if interactive else parser.parse_args('')


def process_quarot_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if args.device:
        config.device = torch.device(args.device)

    config.dtype = torch.float16


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

    def reset_model_device() -> None:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(model_adapter)
        else:
            model.to(config.device)

    dataset = data_utils.get_dataset(args.cal_dataset)
    test_dataset = dataset["test"]
    test_loader = data_utils.prepare_test_dataloader(
        dataset=test_dataset, tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
    )

    # original ppl
    if args.eval_baseline:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Original ppl: {dataset_ppl:.4f}')
        wandb.log({"original_ppl": dataset_ppl})
        model.cpu()
        utils.cleanup_memory()

    if args.rotate:
        # replace modules with compressible equivalents
        layernorm_fusion.replace_layers(model_adapter)

        # fuse layernorms and add rotations to skip connections
        layernorm_fusion.fuse_modules(model_adapter)

        utils.cleanup_memory()

        # Prepare the model for quantization
        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if 'down_proj' in name:  # TODO : make this more general
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name:  # TODO : make this more general
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
        else:
            quant_utils.add_actquant(
                model
            )  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    # Quantize the model weights
    if args.w_bits < 16:
        save_dict = {}
        if args.quarot_model_path:  # Load Quantized Rotated Model
            raise NotImplementedError("Loading quantized rotated model is not yet implemented!")

        elif not args.w_rtn:  # GPTQ Weight Quantization
            raise NotImplementedError("GPTQ weight quantization is not yet implemented!")
        else:  # RTN Weight Quantization
            quantizers = rtn_utils.rtn_fwrd(model, config.device, args)
            save_dict["w_quantizers"] = quantizers

        if args.save_dir:
            raise NotImplementedError("Saving quantized model is not yet implemented!")

    reset_model_device()
    dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    logging.info(f'QuaRot ppl: {dataset_ppl:.4f}')
    wandb.log({"quarot_ppl": dataset_ppl})


if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    quarot_args = quarot_arg_parser()
    process_quarot_args(quarot_args)
    quarot_main(quarot_args)
