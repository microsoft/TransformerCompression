# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os

import torch
import transformers
import wandb
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config

utils.configure_logging()

os.environ["WANDB__SERVICE_WAIT"] = "300"


def get_optimizer_and_scheduler(model, train_dataset):

    # Defaults vals from Liana's run.py. TODO: make this configurable from CLI
    class Config:
        learning_rate = 2e-4
        adam_beta1 = 0.9
        adam_beta2 = 0.95
        adam_epsilon = 1e-8
        weight_decay = 1e-2
        num_warmup_steps = 400
        epochs = 1
        batch_size = 1
        gradient_accumulation_steps = 4
        lr_scheduler_type = "linear"

    config = Config()

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    kwargs_lr_scheduler = {
        "optimizer": optimizer,
        "num_warmup_steps": config.num_warmup_steps,
        "num_training_steps": ((len(train_dataset) - 1) // (config.batch_size * config.gradient_accumulation_steps) + 1)
        * config.epochs,
    }
    if config.lr_scheduler_type in ("cosine", "cosine_with_warmup"):
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(**kwargs_lr_scheduler)
    elif config.lr_scheduler_type in ("linear", "linear_with_warmup"):
        lr_scheduler = transformers.get_linear_schedule_with_warmup(**kwargs_lr_scheduler)
    else:
        raise NotImplementedError

    return optimizer, lr_scheduler


class CustomTrainer(Trainer):
    def __init__(self, *args, train_loader=None, test_loader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.model.config.pad_token_id)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_train_dataloader(self) -> DataLoader:
        return self.train_loader

    def get_eval_dataloader(self, _) -> DataLoader:
        return self.test_loader


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
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate on.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--finetune-dataset",
        type=str,
        help="Dataset to calibrate on.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="alpaca",
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples of the calibration data to load.",
        default=128,
    )
    parser.add_argument(
        "--test-nsamples",
        type=int,
        help="Number of samples to load from the test set.",
        default=128,
    )
    parser.add_argument(
        "--train-nsamples",
        type=int,
        help="Number of samples to load from the train set.",
        default=128,
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for loading the calibration data.")
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--save-dir", type=str, default=None, help="Path to save the model.")
    parser.add_argument("--load-model-path", type=str, default=None, help="Path to load the sliced model from.")
    parser.add_argument('--hf-token', type=str, default=None)

    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    args = parser.parse_args()

    logging.debug(f'Parsed arguments:')
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")

    if args.batch_size > args.cal_nsamples:
        raise argparse.ArgumentTypeError(f"Batch size can not be greater than the number of calibration samples")

    return args


def main() -> None:
    logging.info("Running SliceGPT post-slicing finetuning experiment")

    args = argparser()

    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project="slicegpt-finetuning", config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project="slicegpt", mode='disabled')

    if args.load_model_path:
        # load the model from load_model_path to compute perplexity and skip rotation and slicing
        logging.info(f"Loading sliced {args.model} model from {args.load_model_path} with sparsity {args.sparsity}")
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model, args.load_model_path, args.sparsity, token=args.hf_token
        )
    else:
        # load one of the pre-trained models
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model, token=args.hf_token, dtype=config.dtype)

    wikitext_ds = data_utils.get_dataset(args.cal_dataset)
    calibration_loader = data_utils.prepare_dataloader(
        dataset=wikitext_ds["train"],
        tokenizer=tokenizer,
        max_seqlen=512,
        batch_size=args.batch_size,
        nsamples=args.cal_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )

    finetune_ds = data_utils.get_dataset(args.finetune_dataset)
    finetune_train_loader = data_utils.prepare_dataloader(
        dataset=finetune_ds["train"],
        tokenizer=tokenizer,
        max_seqlen=2048,
        batch_size=1,
        nsamples=128,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )

    finetune_test_loader = data_utils.prepare_dataloader(
        dataset=wikitext_ds["test"],
        tokenizer=tokenizer,
        max_seqlen=2048,
        batch_size=args.batch_size,
        nsamples=64,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )
    finetune_val_loader = data_utils.prepare_dataloader(
        dataset=wikitext_ds["validation"],
        tokenizer=tokenizer,
        max_seqlen=2048,
        batch_size=args.batch_size,
        nsamples=64,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )

    if not args.load_model_path:
        # replace modules with compressible equivalents
        layernorm_fusion.replace_layers(model_adapter)

        # fuse layernorms and add rotations to skip connections
        layernorm_fusion.fuse_modules(model_adapter)

        original_param_count = sum(int(p.nelement()) for p in model_adapter.model.parameters())
        logging.info(f'Original model parameters: {original_param_count:,d}')

        # compute new embedding dimension given the desired sparsity level
        new_embedding_dimension = int((1 - args.sparsity) * model_adapter.hidden_size)
        logging.info(f"New embedding dimension: {new_embedding_dimension} (sparsity {args.sparsity})")

        ignore_tokens = [tokenizer.pad_token_id]
        rotate.rotate_and_slice(model_adapter, calibration_loader, new_embedding_dimension, ignore_tokens=ignore_tokens)

        # save sliced model
        model_file = os.path.join("sliced_models", os.path.basename(args.model) + "_" + str(args.sparsity) + ".pt")
        torch.save(model_adapter.model.state_dict(), model_file)
        logging.info(f"Saved sliced model to sliced_models")

    if args.distribute_model:
        gpu_utils.distribute_model(model_adapter)
    else:
        model_adapter.model.to(config.device)

    dataset_ppl = gpu_utils.evaluate_ppl(model_adapter, finetune_val_loader)
    logging.info(f'PPL before finetuning: {dataset_ppl:.4f}')
    wandb.log({"pre_finetune_ppl": dataset_ppl})

    utils.cleanup_memory()

    # TODO: make this configurable from CLI? More general
    if "llama" in args.model:
        lora_target_modules = [
            "k_proj",
            "v_proj",
            "q_proj",
            # "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ]
    else:
        lora_target_modules = [
            "k_proj",
            "v_proj",
            "q_proj",
            "out_proj",
            "fc1",
            "fc2",
        ]

    # TODO: make this configurable from CLI
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_target_modules,
    )

    model = model_adapter.model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # create optimizer and scheduler
    optimizer, lr_scheduler = get_optimizer_and_scheduler(model, finetune_ds["train"])

    training_args = TrainingArguments(
        output_dir=f"./results_{os.path.basename(args.model)}_{args.sparsity}",  # output directory
        num_train_epochs=1,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        logging_steps=1,
        save_steps=16,
        save_total_limit=2,
        disable_tqdm=False,
        load_best_model_at_end=True,
        eval_steps=16,
        evaluation_strategy="steps",
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # lower eval_loss is better,
        gradient_checkpointing=True,
    )

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=finetune_train_loader,
        test_loader=finetune_val_loader,
        args=training_args,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # required to enable gradient_checkpointing
    model.enable_input_require_grads()

    model.train()
    trainer.train()

    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        model_file = os.path.join(args.save_dir, os.path.basename(args.model) + "_" + str(args.sparsity) + ".pt")

        # save peft model as a standard pt model
        merged_model = model.merge_and_unload()

        torch.save(merged_model.state_dict(), model_file)
        logging.info(f"Saved sliced and finetuned model to {args.save_dir}")

    utils.cleanup_memory()

    dataset_ppl = gpu_utils.evaluate_ppl(model, finetune_val_loader)
    logging.info(f'PPL after finetuning: {dataset_ppl:.4f}')
    wandb.log({"post_finetune_ppl": dataset_ppl})


if __name__ == "__main__":
    main()
