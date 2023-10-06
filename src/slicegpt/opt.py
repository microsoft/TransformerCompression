import torch
import time
import utils
import argparse
import datautils
import opt_utils
import wandb
from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def opt_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="OPT model to load; pass `facebook/opt-125m`.",
        choices=[
            "facebook/opt-125m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
            "facebook/opt-13b",
            "facebook/opt-30b",
            "facebook/opt-66b",
        ],
        default="facebook/opt-125m",
    )
    parser.add_argument(
        "--cal_dataset",
        type=str,
        help="Dataset to calibrate.",
        choices=["wikitext2", "ptb", "c4"],
        default="wikitext2",
    )
    parser.add_argument(
        "--cal_nsamples",
        type=int,
        help="Number of samples to calibrate on.",
        default=128,
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        help="Dataset to evaluate.",
        choices=["wikitext2", "ptb", "c4"],
        default="wikitext2",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="Sparsity of the calibration data."
    )
    parser.add_argument(
        "--eval_baseline", action="store_true", help="Evaluate the baseline model."
    )

    parser.add_argument(
        "--debug", action="store_true", help="Evaluate the fused model."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark the compressed model (without ppl check).",
    )
    parser.add_argument(
        "--ppl_check", action="store_true", help="Benchmark the rotated model."
    )

    parser.add_argument(
        "--benchmark_baseline",
        action="store_true",
        help="Benchmark the Baseline model.",
    )

    parser.add_argument("--compress_head", action="store_true")

    parser.add_argument(
        "--save_dir", type=str, default=None, help="Path to save the model."
    )
    parser.add_argument(
        "--load_dir", type=str, default=None, help="Path to load the model."
    )

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dp2_cache", action="store_true")

    # SparseGPT args
    parser.add_argument(
        "--sparsegpt", action="store_true", help="Use SparseGPT on compressed model."
    )
    parser.add_argument(
        "--sparsegpt_sp", type=float, default=0.0, help="Sparsity on SparseGPT."
    )
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")

    args = parser.parse_args()
    assert (
        args.sparsity >= 0 and args.sparsity <= 1
    ), "Sparsity should be in the range [0, 1]!"
    if args.dp2_cache:
        utils.deeplearn2_cache_dir()

    if args.sparsegpt_sp > 0:
        args.sparsegpt = True

    if args.sparsegpt:
        args.nsamples = args.cal_nsamples
        args.minlayer = -1
        args.maxlayer = 1000
        args.percdamp = 0.01
        args.blocksize = 128
        args.gmp = False
        args.wbits = 16
        args.invert = False
        args.log_wandb = False
        args.prune_only = ""
        assert (
            args.sparsegpt_sp > 0 and args.sparsegpt_sp <= 1
        ), "Sparsity should be in the range (0, 1]!"

    return args


@torch.no_grad()
def rotate_opt(model, dataloader, dev, args):
    """
    TODO
    """
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    col_to_prune = int(args.sparsity * model.config.hidden_size)

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.cal_nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    inps = []
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    _, Q_1 = utils.pca_calc(torch.cat(inps).reshape(-1, model.config.hidden_size))
    if args.sparsity > 0:
        Q_1 = Q_1[:, :-col_to_prune].clone()

    model.model.decoder.embed_positions = opt_utils.opt_add_orth_pos_embedding(
        model.model.decoder.embed_positions, orth=Q_1.clone(), config=model.config
    )
    model.model.decoder.embed_tokens = opt_utils.opt_add_orth_token_embedding(
        model.model.decoder.embed_tokens, orth=Q_1.clone(), config=model.config
    )
    inps = []
    cache = {"i": 0, "attention_mask": None}
    layers[0] = layers[0].to(dev)
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    inps = torch.cat(inps)

    print(f"(Rotate) layers:", end=" ", flush=True)
    for i, layer in enumerate(layers):
        print(f" {i}", end="", flush=True)
        if i > 0:
            Q_1 = Q_5.clone()

        layer.self_attn.q_proj = opt_utils.opt_add_orth_linear_input(
            layer.self_attn.q_proj, Q_1.clone()
        )
        layer.self_attn.k_proj = opt_utils.opt_add_orth_linear_input(
            layer.self_attn.k_proj, Q_1.clone()
        )
        layer.self_attn.v_proj = opt_utils.opt_add_orth_linear_input(
            layer.self_attn.v_proj, Q_1.clone()
        )
        layer.attn_shortcut_Q = Q_1.clone().T.to(dtype)

        layer = layer.to(dev)  # Load the layer into GPU

        # Extract the input of the second layer norm input and calculate the Q_3
        mlp_ln_inputs = []

        def hook_fn(name):
            def hook(_, inp, output):
                if type(inp) == tuple:
                    inp = inp[0]
                if len(inp.shape) == 3:
                    inp = inp.reshape(-1, inp.shape[-1])
                mlp_ln_inputs.append(inp.cpu())

            return hook

        handles = []
        handles.append(
            layer.final_layer_norm.register_forward_hook(hook_fn("final_layer_norm"))
        )

        outs = []
        for j in range(args.cal_nsamples):
            out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            outs.append(out)
        for h in handles:
            h.remove()

        _, Q_3 = utils.pca_calc(torch.cat(mlp_ln_inputs, dim=0))
        _, Q_5 = utils.pca_calc(torch.cat(outs).reshape(-1, out.shape[-1]))

        if args.sparsity > 0:
            Q_3 = Q_3[:, :-col_to_prune].clone()
            if i < len(layers) - 1 or args.compress_head:
                Q_5 = Q_5[:, :-col_to_prune].clone()

        layer.attn_shortcut_Q = torch.matmul(Q_1.clone().T, Q_3.clone()).to(dtype)
        layer.mlp_shortcut_Q = torch.matmul(Q_3.clone().T, Q_5.clone()).to(dtype)

        model.model.decoder.layers[
            i
        ].self_attn.out_proj = opt_utils.opt_add_orth_linear_output(
            layer.self_attn.out_proj, Q_3.clone()
        )
        layer.fc1 = opt_utils.opt_add_orth_linear_input(layer.fc1, Q_3.clone())
        layer.fc2 = opt_utils.opt_add_orth_linear_output(layer.fc2, Q_5.clone())

        layer = layers[i].to(dev)
        # Now we can run the forward pass over this block
        outs = []
        for j in range(args.cal_nsamples):
            out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            outs.append(out)

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = torch.cat(outs), inps
    model.lm_head = opt_utils.opt_add_orth_linear_input(model.lm_head, Q_5.clone())
    print(" Done!")


@opt_utils.do_not_initialize
def compress_opt(model, args):

    # get the new embedding dimension
    col_to_prune = int(args.sparsity * model.config.hidden_size)
    if col_to_prune == 0:
        return
    new_embedding_dim = model.config.hidden_size - col_to_prune

    layers = model.model.decoder.layers
    dtype = next(iter(model.parameters())).dtype

    model.model.decoder.embed_positions = OPTLearnedPositionalEmbedding(
        model.config.max_position_embeddings, new_embedding_dim
    ).to(dtype)
    model.model.decoder.embed_tokens = torch.nn.Embedding(
        model.config.vocab_size, new_embedding_dim, model.config.pad_token_id
    ).to(dtype)

    for i, layer in enumerate(layers):

        layer.self_attn.q_proj = torch.nn.Linear(
            new_embedding_dim,
            layer.self_attn.q_proj.out_features,
            bias=layer.self_attn.q_proj.bias is not None,
        ).to(dtype)
        layer.self_attn.k_proj = torch.nn.Linear(
            new_embedding_dim,
            layer.self_attn.k_proj.out_features,
            bias=layer.self_attn.k_proj.bias is not None,
        ).to(dtype)
        layer.self_attn.v_proj = torch.nn.Linear(
            new_embedding_dim,
            layer.self_attn.v_proj.out_features,
            bias=layer.self_attn.v_proj.bias is not None,
        ).to(dtype)
        layer.self_attn.out_proj = torch.nn.Linear(
            layer.self_attn.out_proj.in_features,
            new_embedding_dim,
            bias=layer.self_attn.out_proj.bias is not None,
        ).to(dtype)
        layer.fc1 = torch.nn.Linear(
            new_embedding_dim,
            layer.fc1.out_features,
            bias=layer.fc1.bias is not None,
        ).to(dtype)
        layer.attn_shortcut_Q = torch.eye(new_embedding_dim).to(dtype)
        layer.mlp_shortcut_Q = torch.eye(new_embedding_dim).to(dtype)
        if i < len(layers) - 1 or args.compress_head:
            layer.fc2 = torch.nn.Linear(
                layer.fc2.in_features,
                new_embedding_dim,
                bias=layer.fc2.bias is not None,
            ).to(dtype)
        else:
            layer.mlp_shortcut_Q = torch.rand(
                new_embedding_dim, layer.fc2.out_features
            ).to(dtype)

    if args.compress_head:
        model.lm_head = torch.nn.Linear(
            new_embedding_dim,
            model.lm_head.out_features,
            bias=model.lm_head.bias is not None,
        ).to(dtype)

    return model


if __name__ == "__main__":
    args = opt_argparser()
    if args.dp2_cache:
        utils.deeplearn2_cache_dir()

    if args.wandb:
        wandb.init(
            project="llm_rotation",
            entity="saleh_ashkboos",
            tags=["FP16", "static_sparsification", "fp64_pca"],
        )
        wandb.config.update(args)

    utils.set_seed(args.seed)

    model = opt_utils.get_opt(args.model)

    if args.benchmark_baseline:
        gpus = [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]
        print("Using GPUs:", gpus)
        torch.cuda.empty_cache()
        dataloader, testloader = datautils.get_loaders(
            args.eval_dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        if len(gpus) > 1:
            opt_utils.opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        input_ids = next(iter(dataloader))[0][:, :128]  # benchmark over 128 tokens
        baseline_token_per_sec = opt_utils.opt_benchmark(
            model, input_ids, dev=DEV, check=True
        )
        print(
            f"\nBaseline Model ({args.eval_dataset.upper()}) (real) Sec/Token: {baseline_token_per_sec:.4f} ({len(gpus)} GPUs)"
        )
        if args.wandb:
            wandb.log(
                {
                    "token_per_sec_baseline/{}".format(
                        args.eval_dataset
                    ): baseline_token_per_sec
                }
            )
        exit(2)

    if args.eval_baseline:
        model.eval()
        dataloader, testloader = datautils.get_loaders(
            args.eval_dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        tick = time.time()
        dataset_ppl = opt_utils.opt_eval(model, testloader, DEV)
        tock = time.time()
        print(
            f"\nBaseline Model ({args.eval_dataset.upper()}) PPL: {dataset_ppl:.3f} \n (simulate) Time: {tock-tick:.4f}"
        )
        print(40 * "-")
        model = model.cpu()
        if args.wandb:
            wandb.log({"ppl_baseline/{}".format(args.eval_dataset): dataset_ppl})
            wandb.log(
                {"(simulate) time_baseline/{}".format(args.eval_dataset): tock - tick}
            )

    if args.sparsity > 0:
        # The order of calling these functions is important!
        opt_utils.replace_opt_modules(model, model.config)
        opt_utils.fuse_opt_modules(model)
        model = model.cpu()

    if args.benchmark and not args.ppl_check:
        model = compress_opt(model, args)
        torch.cuda.empty_cache()
        dataloader, testloader = datautils.get_loaders(
            args.eval_dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        gpus = [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]
        print("Using GPUs:", gpus)

        if len(gpus) > 1:
            opt_utils.opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        input_ids = next(iter(dataloader))[0][:, :128]  # benchmark over 128 tokens
        baseline_token_per_sec = opt_utils.opt_benchmark(
            model, input_ids, dev=DEV, check=False
        )
        print(
            f"\nCompressed Model with {args.sparsity} ({args.eval_dataset.upper()}) (Compressed Real) Sec/Token: {baseline_token_per_sec:.4f} ({len(gpus)} GPUs)"
        )
        if args.wandb:
            wandb.log(
                {"sec_per_token/{}".format(args.eval_dataset): baseline_token_per_sec}
            )
        exit(2)

    if args.load_dir is None and args.sparsity > 0:
        dataloader, testloader = datautils.get_loaders(
            args.cal_dataset,
            nsamples=args.cal_nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )

        rotate_opt(model, dataloader, DEV, args)
        model = model.cpu()
        if args.save_dir is not None:
            print(f"Saving the model to {args.save_dir}...")
            torch.save(model.state_dict(), args.save_dir)
    elif args.sparsity > 0:
        # load the model from load_dir
        print(f"Loading the model from {args.load_dir}...")
        model = compress_opt(model, args)
        model.load_state_dict(
            torch.load(args.load_dir, map_location=torch.device("cpu"))
        )

    if args.sparsegpt:
        import sys

        sys.path.append("./SparseGPT_Code")
        dataloader, testloader = datautils.get_loaders(
            args.cal_dataset,
            nsamples=args.cal_nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        import opt_sparsegpt

        sliceGPT_sparsity = args.sparsity
        args.sparsity = args.sparsegpt_sp
        opt_sparsegpt.opt_sequential(model, dataloader, DEV, args)
        args.sparsity = sliceGPT_sparsity

    dataloader, testloader = datautils.get_loaders(
        args.eval_dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.benchmark and args.ppl_check:
        gpus = [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]
        print("Using GPUs:", gpus)
        torch.cuda.empty_cache()

        if len(gpus) > 1:
            opt_utils.opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        input_ids = next(iter(dataloader))[0][:, :128]  # benchmark over 128 tokens
        baseline_token_per_sec = opt_utils.opt_benchmark(
            model, input_ids, dev=DEV, check=True
        )
        print(
            f"\nRotated Model with {args.sparsity} ({args.eval_dataset.upper()}) (Compressed Real) Sec/Token: {baseline_token_per_sec:.4f} ({len(gpus)} GPUs)"
        )
        if args.wandb:
            wandb.log(
                {
                    "sec_per_token_sparsified/{}".format(
                        args.eval_dataset
                    ): baseline_token_per_sec
                }
            )
        exit(2)

    tick = time.time()
    dataset_ppl = opt_utils.opt_eval(model, testloader, DEV)
    tock = time.time()
    print(
        f"\nRotated Model with {args.sparsity} ({args.eval_dataset.upper()}) PPL: {dataset_ppl:.3f} \n (simulate) Time: {tock-tick:.4f}"
    )
    print(40 * "-")
    if args.wandb:
        wandb.log({"ppl/{}".format(args.eval_dataset): dataset_ppl})
        wandb.log({"(simulate) time/{}".format(args.eval_dataset): tock - tick})
