import argparse
import json

from lm_eval import tasks, evaluator, utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--no_cache", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        no_cache=args.no_cache
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()