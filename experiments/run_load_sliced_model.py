import argparse
import torch
from slicegpt.hf_utils import load_sliced_model

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Base model to load; pass `facebook/opt-125m`.",
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
    parser.add_argument("--sparsity", type=float, default=0.1, help="A measure of how much on the layers are removed (in the range [0, 1])")
    parser.add_argument("--load_dir", type=str, default=None, help="Path to load the sliced model from.")
    parser.add_argument('--hf_token', type=str, default=None)

    return parser.parse_args()

def main():
    args = argparser()
    print(f"Loading sliced {args.model} model from {args.load_dir} with sparsity {args.sparsity}")
    model, tokenizer = load_sliced_model(args.model, args.hf_token, args.load_dir, args.sparsity, DEV)

    assert model is not None
    assert tokenizer is not None

if __name__ == "__main__":
    main()