import torch
import time
import utils
import argparse
import datautils
import llama_utils

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def llama_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str,
        help='LLAMA model to load; pass meta-llama/Llama-2-7b-hf.',
        choices=[
            # LLAMA 2 Models
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-13b-hf',
            'meta-llama/Llama-2-70b-hf',
            ],
        default='meta-llama/Llama-2-7b-hf'
    )
    parser.add_argument('--cal_dataset', type=str, help='Dataset to calibrate.', 
                        choices=['wikitext2', 'ptb', 'c4'], default='wikitext2')
    parser.add_argument('--cal_nsamples', type=int, help='Number of samples to calibrate on.', default=128)
    parser.add_argument('--eval_dataset', type=str, help='Dataset to evaluate.', 
                        choices=['wikitext2', 'ptb', 'c4'], default='wikitext2')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--sparsity', type=float, default=0.0, help='Sparsity of the calibration data.')
    parser.add_argument('--eval_baseline', action='store_true', help='Evaluate the baseline model.')

    parser.add_argument('--debug', action='store_true', help='Evaluate the fused model.')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark the compressed model (without ppl check).')
    parser.add_argument('--ppl_check', action='store_true', help='Benchmark the rotated model.')

    parser.add_argument('--benchmark_baseline', action='store_true', help='Benchmark the Baseline model.')

    parser.add_argument('--compress_head', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to save the model.')
    parser.add_argument('--load_dir', type=str, default=None, help='Path to load the model.')

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--dp2_cache', action='store_true')

    # SparseGPT args
    parser.add_argument('--sparsegpt', action='store_true', help='Use SparseGPT on compressed model.')
    parser.add_argument('--sparsegpt_sp', type=float, default=0.0, help='Sparsity on SparseGPT.')
    parser.add_argument( '--prunen', type=int, default=0, help='N for N:M pruning.')
    parser.add_argument( '--prunem', type=int, default=0, help='M for N:M pruning.')


    parser.add_argument(
        '--hf_token', type=str, default='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')
    args = parser.parse_args()

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
        args.prune_only = ''
        args.true_sequential = False
        assert args.sparsegpt_sp >0 and args.sparsegpt_sp <= 1, 'Sparsity should be in the range (0, 1]!'

    assert args.sparsity >=0 and args.sparsity <= 1, 'Sparsity should be in the range [0, 1]!'
    if args.dp2_cache:
        utils.deeplearn2_cache_dir()
    return args

@torch.no_grad()
def llama_rotate(model, dataloader, dev, args):
    '''
    TODO: Write a description of this function.
    '''
    model.eval()
    model.config.use_cache = False
    layers = model.model.layers
    
    col_to_prune = int(args.sparsity * model.config.hidden_size)

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            cache['i'] += 1
            inps.append(inp)
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
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
    
    model.model.embed_tokens = llama_utils.llama_add_orth_token_embedding(model.model.embed_tokens, Q_1.clone(), model.config)

    inps = []
    cache = {'i': 0, 'attention_mask': None}
    layers[0] = layers[0].to(dev)
    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    inps = torch.cat(inps)

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    
    print(f'(Rotate) layers:', end=' ', flush=True)

    for i in range(len(layers)):
        print(f' {i}', end='', flush=True)
        if i > 0:
            Q_1 = Q_5.clone()

        model.model.layers[i].self_attn.q_proj = llama_utils.llama_add_orth_linear_input(
            model.model.layers[i].self_attn.q_proj, Q_1.clone())
        model.model.layers[i].self_attn.k_proj = llama_utils.llama_add_orth_linear_input(
            model.model.layers[i].self_attn.k_proj, Q_1.clone())
        model.model.layers[i].self_attn.v_proj = llama_utils.llama_add_orth_linear_input(
            model.model.layers[i].self_attn.v_proj, Q_1.clone())
        model.model.layers[i].attn_shortcut_Q = Q_1.clone().T.to(dtype)

        layer = layers[i].to(dev) # Load the layer into GPU

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
        handles.append(layer.post_attention_layernorm.register_forward_hook(hook_fn('post_attention')))

        outs = []
        for j in range(args.cal_nsamples):
            out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            outs.append(out)
        for h in handles:
            h.remove()

        _, Q_3 = utils.pca_calc(torch.cat(mlp_ln_inputs, dim=0))
        _, Q_5 = utils.pca_calc(torch.cat(outs).reshape(-1, model.config.hidden_size))

        if args.sparsity > 0:
            Q_3 = Q_3[:, :-col_to_prune].clone()
            if i < len(layers) - 1 or args.compress_head:
                Q_5 = Q_5[:, :-col_to_prune].clone()
                
        model.model.layers[i].attn_shortcut_Q = torch.matmul(Q_1.clone().T, Q_3.clone()).to(dtype)
        model.model.layers[i].mlp_shortcut_Q = torch.matmul(Q_3.clone().T, Q_5.clone()).to(dtype)


        model.model.layers[i].self_attn.o_proj = llama_utils.llama_add_orth_linear_output(
            model.model.layers[i].self_attn.o_proj, Q_3)
        model.model.layers[i].mlp.gate_proj = llama_utils.llama_add_orth_linear_input(
            model.model.layers[i].mlp.gate_proj, Q_3)
        model.model.layers[i].mlp.up_proj = llama_utils.llama_add_orth_linear_input(
            model.model.layers[i].mlp.up_proj, Q_3)
        model.model.layers[i].mlp.down_proj = llama_utils.llama_add_orth_linear_output(
            model.model.layers[i].mlp.down_proj, Q_5)


        layer = layers[i].to(dev)
        # Now we can run the forward pass over this block
        outs = []
        for j in range(args.cal_nsamples):
            out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            outs.append(out)
            
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = torch.cat(outs), inps
    model.lm_head = llama_utils.llama_add_orth_linear_input(model.lm_head, Q_5)
    print(' Done!')



def compress_llama(model, args):
    col_to_prune = int(args.sparsity * model.config.hidden_size)
    config = model.config
    if col_to_prune == 0:
        return 
    new_d = model.config.hidden_size - col_to_prune
    
    layers = model.model.layers
    kiming_fn = torch.nn.init.kaiming_uniform_
    uniform_fn = torch.nn.init.uniform_
    normal_fn = torch.nn.init.normal_
    torch.nn.init.kaiming_uniform_ = llama_utils.skip
    torch.nn.init.uniform_ = llama_utils.skip
    torch.nn.init.normal_ = llama_utils.skip
    dtype = next(iter(model.parameters())).dtype

    for i in range(len(layers)):
        
        model.model.layers[i].self_attn.q_proj = torch.nn.Linear(new_d, 
                                                                         model.model.layers[i].self_attn.q_proj.out_features,
                                                                         bias=model.model.layers[i].self_attn.q_proj.bias is not None).to(dtype)
        model.model.layers[i].self_attn.k_proj = torch.nn.Linear(new_d,
                                                                        model.model.layers[i].self_attn.k_proj.out_features,
                                                                        bias=model.model.layers[i].self_attn.k_proj.bias is not None).to(dtype)
        model.model.layers[i].self_attn.v_proj = torch.nn.Linear(new_d,
                                                                        model.model.layers[i].self_attn.v_proj.out_features,
                                                                        bias=model.model.layers[i].self_attn.v_proj.bias is not None).to(dtype)
        model.model.layers[i].self_attn.o_proj = torch.nn.Linear(model.model.layers[i].self_attn.o_proj.in_features,
                                                                           new_d,
                                                                           bias=model.model.layers[i].self_attn.o_proj.bias is not None).to(dtype)
        model.model.layers[i].mlp.up_proj = torch.nn.Linear(new_d,
                                                            model.model.layers[i].mlp.up_proj.out_features,
                                                            bias=model.model.layers[i].mlp.up_proj.bias is not None).to(dtype)
        model.model.layers[i].mlp.gate_proj = torch.nn.Linear(new_d,
                                                            model.model.layers[i].mlp.gate_proj.out_features,
                                                            bias=model.model.layers[i].mlp.gate_proj.bias is not None).to(dtype)
        
        model.model.layers[i].attn_shortcut_Q = torch.eye(new_d).to(dtype)
        model.model.layers[i].mlp_shortcut_Q = torch.eye(new_d).to(dtype)
        if i < len(layers) - 1 or args.compress_head:
            model.model.layers[i].mlp.down_proj = torch.nn.Linear(model.model.layers[i].mlp.down_proj.in_features,
                                                            new_d,
                                                            bias=model.model.layers[i].mlp.down_proj.bias is not None).to(dtype)
        else:
            model.model.layers[i].mlp_shortcut_Q = torch.rand(new_d, model.model.layers[i].mlp.down_proj.out_features).to(dtype)
            
    model.model.embed_tokens = torch.nn.Embedding(config.vocab_size, new_d, config.pad_token_id).to(dtype)

    if args.compress_head:
        model.lm_head = torch.nn.Linear(new_d, model.lm_head.out_features, bias=False).to(dtype)

    torch.nn.init.kaiming_uniform_ = kiming_fn
    torch.nn.init.uniform_ = uniform_fn
    torch.nn.init.normal_ = normal_fn
    return model



if __name__ == '__main__':
    args = llama_argparser()
    if args.dp2_cache:
        utils.deeplearn2_cache_dir()

    if args.wandb:
        import wandb
        wandb.init(project="llm_rotation", entity='saleh_ashkboos', 
                   tags=['FP16', 'static_sparsification', 'fp64_pca'])
        wandb.config.update(args)
        
    utils.set_seed(args.seed)

    model = llama_utils.get_llama(args.model, args.hf_token)

    if args.benchmark_baseline:
        torch.cuda.empty_cache()
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        print('Using GPUs:', gpus)
        dataloader, testloader = datautils.get_loaders(
                    args.eval_dataset, seed=args.seed, model=args.model, 
                    seqlen=model.seqlen, hf_token=args.hf_token)
        if len(gpus) > 1:
            llama_utils.llama_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        input_ids = next(iter(dataloader))[0][:, :128] #benchmark over 128 tokens
        baseline_token_per_sec = llama_utils.llama_benchmark(model, input_ids, dev=DEV, check=True)
        print(f'\nBaseline Model ({args.eval_dataset.upper()}) (real) Sec/Token: {baseline_token_per_sec:.4f} ({len(gpus)} GPUs)')
        if args.wandb:
            wandb.log({'token_per_sec_baseline/{}'.format(args.eval_dataset): baseline_token_per_sec})
        exit(2)


    if args.eval_baseline:
        model.eval()
        dataloader, testloader = datautils.get_loaders(
                args.eval_dataset, seed=args.seed, model=args.model, 
                seqlen=model.seqlen, hf_token=args.hf_token)
        tick = time.time()
        dataset_ppl = llama_utils.llama_eval(model, testloader, DEV)
        tock = time.time()
        print(f'\nBaseline Model ({args.eval_dataset.upper()}) PPL: {dataset_ppl:.3f} \n (simulate) Time: {tock-tick:.4f}')
        print(40*'-')
        model = model.cpu()
        if args.wandb:
            wandb.log({'ppl_baseline/{}'.format(args.eval_dataset): dataset_ppl}) 
            wandb.log({'(simulate) time_baseline/{}'.format(args.eval_dataset): tock-tick})

    if args.sparsity > 0:
        llama_utils.replace_llama_modules(model, model.config)
        llama_utils.fuse_llama_modules(model)
        model = model.cpu()


    if args.benchmark and not args.ppl_check:
        model = compress_llama(model, args)
        torch.cuda.empty_cache()
        dataloader, testloader = datautils.get_loaders(
                args.eval_dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, hf_token=args.hf_token)
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        print('Using GPUs:', gpus)
        if len(gpus) > 1:
            llama_utils.llama_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        input_ids = next(iter(dataloader))[0][:, :128] #benchmark over 128 tokens
        baseline_token_per_sec = llama_utils.llama_benchmark(model, input_ids, dev=DEV, check=False)
        print(f'\nCompressed Model with {args.sparsity} ({args.eval_dataset.upper()}) (Compressed Real) Sec/Token: {baseline_token_per_sec:.4f} ({len(gpus)} GPUs)')
        if args.wandb:
            wandb.log({'token_per_sec_baseline/{}'.format(args.eval_dataset): baseline_token_per_sec})
        exit(2)



    if args.load_dir is None and args.sparsity > 0:
        dataloader, testloader = datautils.get_loaders(
        args.cal_dataset , nsamples=args.cal_nsamples, 
        seed=args.seed, model=args.model, seqlen=model.seqlen, 
        hf_token=args.hf_token
        )

        llama_rotate(model, dataloader, DEV, args)
        model = model.cpu()
        if args.save_dir is not None:
            print(f'Saving the model to {args.save_dir}...')
            torch.save(model.state_dict(), args.save_dir)
    elif args.sparsity > 0:
        #load the model from load_dir
        print(f'Loading the model from {args.load_dir}...')
        model = compress_llama(model, args)
        model.load_state_dict(torch.load(args.load_dir, map_location=torch.device('cpu')))


    if args.sparsegpt:
        import sys
        sys.path.append('./SparseGPT_Code')
        dataloader, testloader = datautils.get_loaders(
        args.cal_dataset , nsamples=args.cal_nsamples, 
        seed=args.seed, model=args.model, seqlen=model.seqlen, 
        hf_token=args.hf_token
        )
        import llama_sparsegpt
        sliceGPT_sparsity = args.sparsity
        args.sparsity = args.sparsegpt_sp
        llama_sparsegpt.llama_sequential(model, dataloader, DEV, args)
        args.sparsity = sliceGPT_sparsity


    dataloader, testloader = datautils.get_loaders(
                args.eval_dataset, seed=args.seed, model=args.model, 
                seqlen=model.seqlen, hf_token=args.hf_token)
    
    if args.benchmark and args.ppl_check:
        torch.cuda.empty_cache()
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        print('Using GPUs:', gpus)

        if len(gpus) > 1:
            llama_utils.llama_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        input_ids = next(iter(dataloader))[0][:, :128] #benchmark over 128 tokens
        baseline_token_per_sec = llama_utils.llama_benchmark(model, input_ids, dev=DEV, check=True)
        print(f'\nRotated Model with {args.sparsity} ({args.eval_dataset.upper()}) (Compressed Real) Sec/Token: {baseline_token_per_sec:.4f} ({len(gpus)} GPUs)')
        if args.wandb:
            wandb.log({'token_per_sec_sparsified/{}'.format(args.eval_dataset): baseline_token_per_sec})
        exit(2)

    tick = time.time()
    dataset_ppl = llama_utils.llama_eval(model, testloader, DEV)
    tock = time.time()
    print(f'\nRotated Model with {args.sparsity} ({args.eval_dataset.upper()}) PPL: {dataset_ppl:.3f} \n (simulate) Time: {tock-tick:.4f}')
    print(40*'-')
    if args.wandb:
        wandb.log({'ppl/{}'.format(args.eval_dataset): dataset_ppl})
        wandb.log({'(simulate) time/{}'.format(args.eval_dataset): tock-tick})