import logging
from argparse import ArgumentParser
from pathlib import Path

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import LocalBackend
from syne_tune.config_space import choice, loguniform, randint, uniform
from syne_tune.optimizer.baselines import BayesianOptimization, RandomSearch

# Configuration space (or search space)
# config_space = {
#     "model": "microsoft/phi-2",
#     "sparsity": 0.25,
#     "load-model-path": "sliced_models_alpaca/phi-2_0.25.pt",
#     "lora-target-modules": choice(["k_proj v_proj q_proj dense fc1 fc2", "k_proj v_proj q_proj dense"]),
#     "lora-alpha": loguniform(1e-2, 1e3),
#     "lora-dropout": uniform(0, 0.5),
#     "lora-r": randint(2, 64),
#     "finetune-train-seqlen": randint(64, 1024),
#     "finetune-test-seqlen": 2048,
#     "finetune-train-nsamples": 8192,
#     "finetune-train-batch-size": randint(1, 8),
#     "wandb-project": "syne-tune-phi",
#     "finetune-dataset": "alpaca",
#     "ppl-eval-dataset": "alpaca",
# }
config_space = {
    "model": "facebook/opt-13b",
    "sparsity": 0.25,
    "load-model-path": "sliced_models_alpaca/opt-13b_0.25.pt",
    "lora-target-modules": choice(["k_proj v_proj q_proj out_proj fc1 fc2", "k_proj v_proj q_proj out_proj"]),
    "lora-alpha": loguniform(1e1, 1e3),
    "lora-dropout": 0.05,
    "lora-r": randint(16, 256),
    "finetune-train-seqlen": randint(32, 2048),
    "finetune-test-seqlen": 2048,
    "finetune-train-nsamples": 8192,
    "finetune-train-batch-size": randint(1, 8),
    "wandb-project": "syne-tune-opt",
    "finetune-dataset": "alpaca",
    "ppl-eval-dataset": "alpaca",
    "hf-token": "***REMOVED***",
    "eval-steps": 64,
    "save-steps": 64,
    "early-stopping-patience": 5
}

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # [1]
    parser = ArgumentParser()
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
