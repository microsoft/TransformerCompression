# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os

import torch
import wandb
from lm_eval import evaluator, tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.base import BaseLM

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config
from slicegpt.model_adapter import ModelAdapter

utils.configure_logging()

os.environ["WANDB__SERVICE_WAIT"] = "300"


class SlicedLM(BaseLM):
    """A wrapper for a model sliced with SliceGPT that allows it to be used with the LM Eval Harness."""

    def __init__(self, args):
        super().__init__()

        if args.load_model_path:
            model_adapter, tokenizer = hf_utils.load_sliced_model(
                args.model, args.load_model_path, args.sparsity, args.hf_token
            )
        else:
            model_adapter, tokenizer = hf_utils.get_model(args.model, token=args.hf_token)

        self.model_adapter = model_adapter
        self.model_adapter.model.config.sparsity = args.sparsity
        self.model_adapter.model.config.model_name = args.model
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.batch_size_per_gpu = args.batch_size
        self.seqlen = self.model_adapter.seqlen

        if (not args.load_model_path) and (not args.baseline):
            self.apply_slicegpt(self.model_adapter, tokenizer, args)

    def apply_slicegpt(self, model_adapter: ModelAdapter, tokenizer, args):
        layernorm_fusion.replace_layers(model_adapter)
        layernorm_fusion.fuse_modules(model_adapter)

        dataloader, _ = data_utils.get_loaders(
            dataset_name=args.cal_dataset,
            nsamples=args.cal_nsamples,
            batch_size=args.batch_size,
            seqlen=model_adapter.seqlen,
            tokenizer=tokenizer,
        )

        new_embedding_dimension = int((1 - args.sparsity) * model_adapter.hidden_size)
        logging.info(f"New embedding dimension: {new_embedding_dimension} (sparsity {args.sparsity})")

        rotate.rotate_and_slice(model_adapter, dataloader, new_embedding_dimension)

    @classmethod
    def create_from_arg_string(cls, args, kwargs):
        return cls(args, kwargs)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model_adapter.seqlen

    @property
    def max_gen_toks(self):
        logging.info('max_gen_toks fn')
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self.model_adapter.model.device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model_adapter.model(inps)[0][:, :, :50272]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model_adapter.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )


def parse_args():
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
    parser.add_argument(
        "--tasks",
        default="piqa,hellaswag,arc_easy,arc_challenge,winogrande",
        choices=lm_eval_utils.MultiChoice(tasks.ALL_TASKS),
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for loading the calibration data.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--load-model-path", type=str, default=None, help="Path to load the sliced model from.")

    parser.add_argument("--baseline", action="store_true", help="Evaluate the dense (un-sliced) model.")

    parser.add_argument('--hf-token', type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.info("Running SliceGPT zeroshot tasks experiment.")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project="slicegpt-zeroshot", config=args)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb.')
        wandb.init(project="slicegpt", mode='disabled')

    # Initialize the model for use in LM Eval Harness.
    model = SlicedLM(args)

    model.model_adapter.model.eval()
    if args.distribute_model:
        # distribute model across available GPUs
        gpu_utils.distribute_model(model.model_adapter)
    else:
        model.model_adapter.model.to(config.device)

    ### LM Eval Harness ###
    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = lm_eval_utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    logging.info(f"Selected Tasks: {task_names}")

    # Run the evaluation.
    results = evaluator.simple_evaluate(model=model, tasks=task_names, no_cache=True)
    wandb.log(results['results'])
    logging.info(json.dumps(results, indent=2))
    logging.info(evaluator.make_table(results))


if __name__ == "__main__":
    main()
