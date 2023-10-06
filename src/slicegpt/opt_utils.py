import torch
import numpy as np
import math
import transformers
import opt_modules
import tqdm
import utils
import os
import time
from transformers.models.opt.modeling_opt import (
    OPTDecoderLayer,
    OPTLearnedPositionalEmbedding,
)


def skip(*args, **kwargs):
    pass


def do_not_initialize(func):
    """
    A decorator that prevents initalization of torch.nn modules.
    """

    def wrapper(*args, **kwargs):
        kiming_fn = torch.nn.init.kaiming_uniform_
        uniform_fn = torch.nn.init.uniform_
        normal_fn = torch.nn.init.normal_

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        result = func(*args, **kwargs)

        torch.nn.init.kaiming_uniform_ = kiming_fn
        torch.nn.init.uniform_ = uniform_fn
        torch.nn.init.normal_ = normal_fn
        
        return result

    return wrapper


@do_not_initialize
def get_opt(model):
    print("Loading {} Model...".format(model))

    cache_dir = os.getenv("TRANSFORMERS_CACHE")
    if cache_dir is not None:
        print("----> Using cache dir: {}".format(cache_dir))
        model = transformers.OPTForCausalLM.from_pretrained(
            model, torch_dtype="auto", cache_dir=cache_dir
        )
    else:
        print("----> Using default cache dir.")
        model = transformers.OPTForCausalLM.from_pretrained(model, torch_dtype="auto")

    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_eval(model, testenc, dev):
    """
    evaluate the OPT model's perplexity on the test set.
    This function loads each loayer onto the device one at a time,
    so that we can evaluate models that are too large to fit on a single GPU.
    """
    model.eval()
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
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
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    inps = torch.cat(inps)
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        if i == 0:
            print("(Eval) Layers: 0", end="", flush=True)
        else:
            print(f", {i}", end="", flush=True)
        layer = layers[i].to(dev)

        outs = []
        for j in range(nsamples):
            out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            outs.append(out)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = torch.cat(outs), inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(
            dev
        )
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    model.config.use_cache = use_cache
    return ppl.item()


def opt_multigpu(model, gpus):
    """
    Split the OPT model across multiple GPUs.
    """
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(gpus[0])
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
        gpus[0]
    )
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(gpus[0])
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(gpus[-1])
    if (
        hasattr(model.model.decoder, "final_layer_norm")
        and model.model.decoder.final_layer_norm
    ):
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(
            gpus[-1]
        )
    import copy

    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {"mask": None}

    class MoveModule(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache["mask"] is None or cache["mask"].device != self.dev:
                cache["mask"] = kwargs["attention_mask"].to(self.dev)
            kwargs["attention_mask"] = cache["mask"]
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.decoder.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus


def opt_benchmark(model, input_ids, dev, check=False):
    DEV = dev
    model.config.use_cache = True
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, "gpus") else dev)
    torch.cuda.synchronize()

    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    print("Benchmarking ...")

    if check:
        loss = torch.nn.CrossEntropyLoss()
        tot = 0.0

    def sync():
        if hasattr(model, "gpus"):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in tqdm.tqdm(range(input_ids.numel()), desc="Benchmarking", ncols=80):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape((1, -1)),
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, : (i + 1)].reshape((1, -1)),
            )
            sync()
            times.append(time.time() - tick)
            if check and i != input_ids.numel() - 1:
                tot += loss(
                    out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)
                ).float()
            cache["past"] = list(out.past_key_values)
            del out
        sync()

        print("Median:", np.median(times))
        if check:
            print("PPL:", torch.exp(tot / (input_ids.numel() - 1)).item())
        return np.median(times)


def bake_mean_into_embedding(embedding: torch.nn.Embedding) -> None:
    """
    This function takes an embedding layer and subtracts the means from the
    weights. This will result in the embedding layer performing
    the mean substitution.
    """
    embedding_dtype = embedding.weight.dtype
    embedding.weight.data = embedding.weight.double() - embedding.weight.double().mean(
        (-1), keepdim=True
    )
    embedding.weight.data = embedding.weight.data.to(embedding_dtype)


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution.
    """
    linear_dtype = linear.weight.dtype
    linear.weight.data = linear.weight.double() - linear.weight.double().mean(
        (-2), keepdim=True
    )
    linear.bias.data = linear.bias.double() - linear.bias.double().mean(
        (-1), keepdim=True
    )
    linear.weight.data = linear.weight.data.to(linear_dtype)
    linear.bias.data = linear.bias.data.to(linear_dtype)


def fold_opt_ln_linear(
    layernorm: torch.nn.LayerNorm, linear_layers: list
) -> torch.nn.Linear:
    """
    Modifies a OPT model to fuse the layernorms.
    """
    ln_dtype = layernorm.weight.dtype
    ln_device = layernorm.weight.device

    for linear in linear_layers:

        # Check if linear layer has a bias, and if it doesn't, we have to add one

        linear_dtype = linear.weight.dtype
        linear_device = linear.weight.device

        if linear.bias is None:
            new_linear = torch.nn.Linear(linear.in_features, linear.out_features)
            new_linear.weight.data = linear.weight.data
            new_linear.bias.data = torch.zeros(linear.out_features)
            linear = new_linear.to(linear_dtype)

        # Calculating new weight and bias
        new_weight = linear.weight.double() * layernorm.weight.double()
        new_bias = (
            torch.matmul(linear.weight.cuda().double(), layernorm.bias.cuda().double())
            + linear.bias.cuda().double()
        )

        linear.weight.data = new_weight.to(linear_device).to(linear_dtype)
        linear.bias.data = new_bias.to(linear_device).to(linear_dtype)

    # Substituting new values
    layernorm.weight.data = (
        torch.ones_like(layernorm.weight.data).to(ln_device).to(ln_dtype)
    )
    layernorm.bias.data = (
        torch.zeros_like(layernorm.bias.data).to(ln_device).to(ln_dtype)
    )

    return linear


def fuse_opt_modules(model):
    """
    This function fuses the linear and layernorm into each other inplace.
    After this function is called, the model should outputs the same results as before.

    args:
        model: the model to be fused
    """

    print("Fusing the OPT modules...")

    # We add the mean subtraction to the first layer

    number_transformer_blocks = model.config.num_hidden_layers

    # First we modify the layernorms to fold their weights
    for i in range(number_transformer_blocks):
        fold_opt_ln_linear(
            model.model.decoder.layers[i].self_attn_layer_norm,
            [
                model.model.decoder.layers[i].self_attn.q_proj,
                model.model.decoder.layers[i].self_attn.k_proj,
                model.model.decoder.layers[i].self_attn.v_proj,
            ],
        )
        fold_opt_ln_linear(
            model.model.decoder.layers[i].final_layer_norm,
            [model.model.decoder.layers[i].fc1],
        )

    # Then we bake the mean substitution into the previous linear layers
    # after this we can also substitute the layernorms for RMSN
    for i in range(number_transformer_blocks):
        bake_mean_into_linear(model.model.decoder.layers[i].self_attn.out_proj)
        bake_mean_into_linear(model.model.decoder.layers[i].fc2)

        model.model.decoder.layers[i].self_attn_layer_norm = utils.RMSN(
            model.config.hidden_size
        )
        model.model.decoder.layers[i].final_layer_norm = utils.RMSN(
            model.config.hidden_size
        )

    model.lm_head = fold_opt_ln_linear(
        model.model.decoder.final_layer_norm, [model.lm_head]
    )
    model.model.decoder.final_layer_norm = utils.RMSN(model.config.hidden_size)

    bake_mean_into_embedding(model.model.decoder.embed_tokens)
    bake_mean_into_embedding(model.model.decoder.embed_positions)


def replace_opt_modules(model, config):
    """
    Replace OPTDecoder with CompressedOPTDecoderLayer.
    This function should be called before fusing the modules!
    """

    for name, mod in model.named_children():
        new_mod = None
        if isinstance(mod, OPTDecoderLayer):
            new_mod = opt_modules.CompressedOPTDecoderLayer(config).to(
                config.torch_dtype
            )
        elif len(list(mod.children())) > 0:
            replace_opt_modules(mod, config)

        if new_mod is not None:
            new_mod.load_state_dict(mod.state_dict(), strict=True)
            setattr(model, name, new_mod)


@torch.no_grad()
def opt_add_orth_linear_input(
    linear: torch.nn.Linear, orth: torch.Tensor
) -> torch.nn.Linear:
    layer_device = linear.weight.device
    layer_dtype = linear.weight.dtype
    new_weight = torch.matmul(
        orth.T.unsqueeze(0).cuda().double(),
        linear.weight.data.unsqueeze(2).double().cuda(),
    ).squeeze()

    new_linear = torch.nn.Linear(orth.shape[-1], linear.out_features, bias=True)

    state_dict = {"weight": new_weight, "bias": linear.bias.data}
    new_linear.load_state_dict(state_dict)

    return new_linear.to(layer_dtype).to(layer_device)


@torch.no_grad()
def opt_add_orth_linear_output(
    linear: torch.nn.Linear, orth: torch.Tensor
) -> torch.nn.Linear:
    layer_device = linear.weight.device
    layer_dtype = linear.weight.dtype
    new_weight = torch.matmul(linear.weight.T.double().cuda(), orth.double().cuda()).T
    new_bias = torch.matmul(orth.T.double().cuda(), linear.bias.double().cuda())

    new_linear = torch.nn.Linear(linear.in_features, orth.shape[-1], bias=True)

    state_dict = {"weight": new_weight, "bias": new_bias}
    new_linear.load_state_dict(state_dict)

    return new_linear.to(layer_dtype).to(layer_device).eval()


def opt_add_orth_pos_embedding(pos_embedding, orth, config):

    layer_device = pos_embedding.weight.device
    layer_dtype = pos_embedding.weight.dtype
    new_weight = torch.matmul(
        pos_embedding.weight.data.double().cuda(), orth.double().cuda()
    )
    new_pos_embedding = OPTLearnedPositionalEmbedding(
        config.max_position_embeddings, orth.shape[-1]
    )
    state_dict = {"weight": new_weight}
    new_pos_embedding.load_state_dict(state_dict)
    return new_pos_embedding.to(layer_device).to(layer_dtype)


def opt_add_orth_token_embedding(token_embedding, orth, config):
    layer_device = token_embedding.weight.device
    layer_dtype = token_embedding.weight.dtype
    new_embed_tokens = torch.nn.Embedding(
        config.vocab_size, orth.shape[-1], config.pad_token_id
    )
    new_weight = new_weight = torch.matmul(
        token_embedding.weight.data.double().cuda(), orth.double().cuda()
    )
    state_dict = {"weight": new_weight}
    new_embed_tokens.load_state_dict(state_dict)
    return new_embed_tokens.to(layer_device).to(layer_dtype)
