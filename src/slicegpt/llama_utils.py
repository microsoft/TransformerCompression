import torch
import utils
import transformers
import llama_modules
import time
import os
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaModel, LlamaDecoderLayer
import math

def skip(*args, **kwargs):
    pass

def get_llama(model, hf_token):
    kiming_fn = torch.nn.init.kaiming_uniform_
    uniform_fn = torch.nn.init.uniform_
    normal_fn = torch.nn.init.normal_
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    print('Loading {} Model...'.format(model))
    cache_dir = os.getenv("TRANSFORMERS_CACHE")
    if cache_dir is not None:
        print('----> Using cache dir: {}'.format(cache_dir))
        model = transformers.LlamaForCausalLM.from_pretrained(model, torch_dtype='auto', use_auth_token=hf_token, cache_dir=cache_dir)
    else:
        print('----> Using default cache dir.')
        model = transformers.LlamaForCausalLM.from_pretrained(model, torch_dtype='auto', use_auth_token=hf_token)
    model.seqlen = 2048
    torch.nn.init.kaiming_uniform_ = kiming_fn
    torch.nn.init.uniform_ = uniform_fn
    torch.nn.init.normal_ = normal_fn
    return model


@torch.no_grad()
def llama_eval(model, testenc, dev):

    '''
    This function is used to evaluate the LLAMA model.
    It loads the blocks one by one into the GPU and evaluates them.
    '''
    model.eval()

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype

    inps = []

    cache = {'i': 0, 'attention_mask': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp)
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    inps = torch.cat(inps)
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        if i == 0:
            print('(Eval) Layers: 0', end='', flush=True)
        else:
            print(f', {i}', end='', flush=True)
        layer = layers[i].to(dev)
        outs = []
        for j in range(nsamples):
            out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            outs.append(out)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = torch.cat(outs), inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    

    model.config.use_cache = use_cache
    
    return ppl.item()


def llama_multigpu(model, gpus):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])
    if hasattr(model.model, 'norm') and model.model.norm is not None:
        model.model.norm = model.model.norm.to(gpus[-1])

    cache = {'mask': None, 'positions': None}

    class MoveModule(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['positions'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
                cache['positions'] = kwargs['position_ids'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            kwargs['position_ids'] = cache['positions']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus



def llama_benchmark(model, input_ids, dev, check=False):
    DEV = dev
    model.config.use_cache = True
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else dev)
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = torch.nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in tqdm(range(input_ids.numel()), desc='Benchmarking', ncols=80):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape((1,-1)),
                past_key_values=cache['past'],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
            )
            sync()
            times.append(time.time() - tick)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        import numpy as np
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
        return np.median(times)


def replace_llama_modules(model, config):
    '''
    Replace LLAMADecoder with CompressedLlamaDecoderLayer.
    '''
    if isinstance(model, transformers.models.llama.modeling_llama.LlamaPreTrainedModel):
        model = model.model

    for name, mod in model.named_children():
        new_mod = None
        if isinstance(mod, LlamaDecoderLayer):
            new_mod = llama_modules.CompressedLlamaDecoderLayer(config).to(config.torch_dtype)
        elif len(list(mod.children())) > 0:
            replace_llama_modules(mod, config)            
      
        if new_mod is not None:
            new_mod.load_state_dict(mod.state_dict(), strict=True)
            setattr(model, name, new_mod)


def fold_llama_layernorm_linear(layernorm: LlamaRMSNorm, linear_layers: list) -> torch.nn.Linear:
    """This function takes a LLAMA layer norm and fold it's weights into
    the list of linear layers in layers. The linear layers are assumed to be
    consecutive.
    """
    ln_dtype = layernorm.weight.dtype
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        new_weight = linear.weight.data.double() * layernorm.weight.double()
        linear.weight.data = new_weight.to(linear_dtype)
    layernorm.weight.data = torch.ones_like(layernorm.weight).to(ln_dtype)

    return linear



def fuse_llama_modules(model: LlamaModel) -> LlamaModel:
    
    '''
    Modifies a llama model to fuse the layernorms. 
    Note that the LLAMA model does not have mean subtraction and bias in the layernorms.
    '''

    print('Fusing the LLAMA-2 modules...')


    layers = model.model.layers
    number_transformer_blocks = model.config.num_hidden_layers

    # First we modify the layernorms to fold their weights
    for i in range(number_transformer_blocks):
        fold_llama_layernorm_linear(
            layers[i].input_layernorm,
            [layers[i].self_attn.q_proj,
            layers[i].self_attn.k_proj,
            layers[i].self_attn.v_proj]
        )
        fold_llama_layernorm_linear(
            layers[i].post_attention_layernorm,
            [layers[i].mlp.gate_proj, layers[i].mlp.up_proj]
        )
        model.model.layers[i].input_layernorm = utils.RMSN(model.config.hidden_size)
        model.model.layers[i].post_attention_layernorm = utils.RMSN(model.config.hidden_size)

    # Fold the final layernorm and the lm_head
    fold_llama_layernorm_linear(
        model.model.norm,
        [model.lm_head]
    )
    model.model.norm = utils.RMSN(model.config.hidden_size)

    return model


def llama_add_orth_linear_input(linear: torch.nn.Linear, orth: torch.Tensor) -> torch.nn.Linear:
    layer_device = linear.weight.device
    layer_dtype = linear.weight.dtype
    new_weight = torch.matmul(orth.T.unsqueeze(0).double().cuda(), 
                                      linear.weight.unsqueeze(2).double().cuda()).squeeze()
    new_linear = torch.nn.Linear(orth.shape[-1], linear.out_features, bias=False)
    state_dict = {'weight': new_weight}
    new_linear.load_state_dict(state_dict)
    
    return new_linear.to(layer_dtype).to(layer_device)

def llama_add_orth_linear_output(linear: torch.nn.Linear, orth: torch.Tensor) -> torch.nn.Linear:
    layer_device = linear.weight.device
    layer_dtype = linear.weight.dtype
    new_weight= torch.matmul(linear.weight.T.double().cuda(),
                                     orth.double().cuda()).T.to(layer_dtype).to(layer_device)
    
    new_linear = torch.nn.Linear(linear.in_features, orth.shape[-1], bias=False)
    state_dict = {'weight': new_weight}
    new_linear.load_state_dict(state_dict)

    return new_linear.to(layer_dtype).to(layer_device)

def llama_add_orth_token_embedding(token_embedding, orth, config):
    layer_device = token_embedding.weight.device
    layer_dtype = token_embedding.weight.dtype
    new_embed_tokens = torch.nn.Embedding(config.vocab_size, orth.shape[-1], config.pad_token_id)
    new_weight =  new_weight= torch.matmul(token_embedding.weight.data.double().cuda(),
                                      orth.double().cuda())
    state_dict = {'weight': new_weight}
    new_embed_tokens.load_state_dict(state_dict)
    return new_embed_tokens.to(layer_device).to(layer_dtype)

