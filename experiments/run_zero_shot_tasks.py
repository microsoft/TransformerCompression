# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json

import torch
from lm_eval import evaluator, tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.base import BaseLM

import wandb
from slicegpt import data_utils, hf_utils, layernorm_fusion, rotate

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LMClass(BaseLM):
    def __init__(self, args):
        super().__init__()

        if args.load_dir:
            model, tokenizer = hf_utils.load_sliced_model(args.model, args.load_dir, args.sparsity, DEV)
        else:
            model, tokenizer = hf_utils.get_model(args.model)

        self.model = model
        self.model.config.sparsity = args.sparsity
        self.model.config.model_name = args.model
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.batch_size_per_gpu = args.batch_size
        self.seqlen = self.model.config.max_position_embeddings

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
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        print('max_gen_toks fn')
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self.model.device

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
            return self.model(inps)[0][:, :, :50272]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False)


def apply_slicegpt(model, eval_dataset='wikitext2', seed=42):
    layernorm_fusion.replace_modules(model.model, model.model.config)
    model.model = model.model.cpu()
    layernorm_fusion.fuse_modules(model.model)

    dataloader, _ = data_utils.get_loaders(
        dataset_name=eval_dataset,
        tokenizer=model.tokenizer,
        seed=seed,
    )

    new_embedding_dimension = int((1 - model.model.config.sparsity) * model.model.config.hidden_size)
    print(f"New embedding dimension: {new_embedding_dimension} (sparsity {model.model.config.sparsity})")

    rotate.rotate_and_slice(model.model, dataloader, new_embedding_dimension)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", default=None, choices=lm_eval_utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1])"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for loading the calibration data.")

    parser.add_argument("--load_dir", type=str, default=None, help="Path to load the sliced model from.")

    parser.add_argument('--hf_token', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    print("Running SliceGPT zeroshot tasks experiment.")

    try:
        wandb.init(project="slicegpt", config=args)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        print(f'Failed to initialize wandb: {e}, continuing without wandb.')
        wandb.init(project="slicegpt", mode='disabled')

    # Initialize the model for use in LM Eval Harness.
    model = LMClass(args)

    # Apply SliceGPT if the model is not a pre-sliced model.
    if not args.load_dir:
        apply_slicegpt(model)

    ### LM Eval Harness ###
    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = lm_eval_utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")
    model.model = model.model.to(DEV)
    model.model.eval()
    results = evaluator.simple_evaluate(model=model, tasks=task_names, no_cache=args.no_cache)
    wandb.log(results['results'])
    print(json.dumps(results, indent=2))
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
