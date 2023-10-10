import torch
from .opt_modules import CompressedOPTDecoderLayer
from .model_utils import (
    get_attention_inputs,
    get_attention_output,
    get_mlp_inputs,
    get_mlp_output,
    get_first_layernorm,
    get_second_layernorm,
    get_embeddings,
    get_lm_head,
    get_layers,
    get_pre_head_layernorm
)
from . import utils

from transformers.models.opt.modeling_opt import (
    OPTDecoderLayer
)


def replace_opt_modules(model, config):
    """
    Replace OPTDecoder with CompressedOPTDecoderLayer. This adds a 'shortcut operation' to each block.
    This function should be called before fusing the modules!
    """
    for name, mod in model.named_children():
        new_mod = None
        if isinstance(mod, OPTDecoderLayer):
            new_mod = CompressedOPTDecoderLayer(config).to(
                config.torch_dtype
            )
        elif len(list(mod.children())) > 0:
            replace_opt_modules(mod, config)

        if new_mod is not None:
            new_mod.load_state_dict(mod.state_dict(), strict=True)
            setattr(model, name, new_mod)

 
def fuse_opt_modules(model):
    """
    This function fuses the linear and layernorm into each other inplace.
    After this function is called, the model should outputs the same results as before.

    args:
        model: the model to be fused
    """

    print("Fusing the OPT modules...")

    # We add the mean subtraction to the first embeddings
    for W in get_embeddings(model):
        W.weight.data = W.weight.data - w.weight.data.mean(dim=-1, keepdim=True)

    layers = get_layers(model)

    # First we modify the layernorms to fold their weights
    for layer in layers:
        fold_opt_ln_linear(get_first_layernorm(layer), get_attention_inputs(layer))
        fold_opt_ln_linear(get_second_layernorm(layer), get_mlp_inputs(layer))

    # Then we bake the mean substitution into the previous linear layers
    # after this we can also substitute the layernorms for RMSN
    for layer in layers:
        bake_mean_into_linear(get_attention_output(layer))
        bake_mean_into_linear(get_mlp_output(layer))

        layers.self_attn_layer_norm = utils.RMSN(
            model.config.hidden_size
        )
        layers.final_layer_norm = utils.RMSN(
            model.config.hidden_size
        )

    fold_opt_ln_linear(get_pre_head_layernorm(model), [get_lm_head(model)])

    model.model.decoder.final_layer_norm = utils.RMSN(model.config.hidden_size)

    bake_mean_into_embedding(model.model.decoder.embed_tokens)
    bake_mean_into_embedding(model.model.decoder.embed_positions)


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

    # Substituting new values - zero offset (bias) and unit scaling
    layernorm.weight.data = (
        torch.ones_like(layernorm.weight.data).to(ln_device).to(ln_dtype)
    )
    layernorm.bias.data = (
        torch.zeros_like(layernorm.bias.data).to(ln_device).to(ln_dtype)
    )






