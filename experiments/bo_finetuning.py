import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from bo_options import lora_target_map
from syne_tune import StoppingCriterion, Tuner, num_gpu
from syne_tune.backend import LocalBackend
from syne_tune.config_space import choice, loguniform, randint, uniform
from syne_tune.optimizer.baselines import BayesianOptimization, RandomSearch

# Model-agnostic configuration space (or search space)
# May benefit from tweaking for specific model types including custom models
config_space = {
    "sparsity": 0.25,
    "learning-rate": loguniform(1e-4, 1e-2),
    "weight-decay": loguniform(1e-5, 1e-1),
    "adam-beta1": uniform(0.9, 0.99),
    "adam-beta2": uniform(0.9, 0.999),
    "adam-eps": loguniform(1e-9, 1e-6),
    "num-warmup-steps": randint(0, 10000),
    "lr-scheduler-type": choice(["linear", "cosine", "linear_with_warmup", "cosine_with_warmup"]),
    "lora-alpha": loguniform(4, 256),
    "lora-dropout": uniform(0, 0.5),
    "lora-r": randint(2, 64),
    "finetune-train-seqlen": randint(64, 1024),
    "finetune-test-seqlen": 2048,
    "finetune-train-nsamples": 8192,
    "finetune-train-batch-size": randint(1, 8),
    "wandb-project": "syne-tune-phi",
    "finetune-dataset": "alpaca",
    "ppl-eval-dataset": "alpaca",
}

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # Temporary fix to be able to use syne tune on AMD GPUs.
    # Can be removed once syne tune supports ROCm.
    num_gpu._num_gpus = torch.cuda.device_count()
    logging.info(f"Number of available cuda devices for syne tune: {num_gpu._num_gpus}")

    # [1]
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model to fine-tune",
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
            # Phi-2 model
            'microsoft/phi-2',
        ],
        required=True,
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to the model to fine-tune (sliced or dense)", required=True
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=(
            "RS",
            "BO",
        ),
        default="RS",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max-wallclock-time",
        type=int,
        default=3600,
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default="bo-finetune",
    )
    args, _ = parser.parse_known_args()

    train_file = "run_finetuning.py"
    entry_point = Path(__file__).parent / train_file
    mode = "min"
    metric = "ppl"

    # Local backend: Responsible for scheduling trials  [3]
    # The local backend runs trials as sub-processes on a single instance
    trial_backend = LocalBackend(entry_point=str(entry_point))

    # Common scheduler kwargs
    method_kwargs = dict(
        metric=metric,
        mode=mode,
        random_seed=args.random_seed,
        search_options={"num_init_random": args.n_workers + 2},
    )

    # Add model-specific config options such as model type, model path and layers to fine-tune
    config_space['model'] = args.model
    config_space['load-model-path'] = args.model_path
    config_space['lora-target-option'] = choice(list(lora_target_map(args.model).keys()))

    if args.method == "RS":
        scheduler = RandomSearch(config_space, **method_kwargs)
    elif args.method == "BO":
        scheduler = BayesianOptimization(config_space, **method_kwargs)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Stopping criterion: We stop after `args.max_wallclock_time` seconds
    stop_criterion = StoppingCriterion(max_wallclock_time=args.max_wallclock_time)

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=args.n_workers,
        tuner_name=args.experiment_tag,
        metadata={
            "seed": args.random_seed,
            "algorithm": args.method,
            "tag": args.experiment_tag,
        },
    )

    tuner.run()
