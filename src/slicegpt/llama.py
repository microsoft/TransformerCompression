import torch
import utils
from opt import (
    get_layer0_inputs,
    rotate_head,
    slice_head,
    rotate_attention_input,
    slice_attention_input
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_signals(layer, inputs, attention_mask):
    """
    Take the input signals ("activations") for a layer, run the layer forward. 
    Return the output of the layer (not layernormed) and the input to the MLP (pre-layernorm also). 
    
    TODO this is the same as the OPT model, except "layer.final_layernorm" -> "layer.post_attention_layernorm".
    
    """
    mlp_ln_inputs = []
    layer = layer.to(DEV)

    def hook_fn(_, inp, _output):
        if type(inp) == tuple:
            inp = inp[0]
        mlp_ln_inputs.append(inp.cpu())

    hook = layer.post_attention_layernorm.register_forward_hook(hook_fn)  
    outs = [layer(input.unsqueeze(0), attention_mask=attention_mask)[0] for input in inputs]
    hook.remove()
        
    return torch.cat(mlp_ln_inputs), torch.cat(outs)


def rotate_attention_output(layer, Q):
    # Rotate output matrix of the self-attention layer.
    dtype = layer.self_attn.o_proj.weight.data.dtype
    W = layer.self_attn.o_proj.weight.data.to(device=DEV, dtype=torch.float64)
    layer.self_attn.o_proj.weight.data = torch.matmul(Q.T, W).to(device="cpu", dtype=dtype)
    b = layer.self_attn.o_proj.bias.data.to(device=DEV, dtype=torch.float64)
    layer.self_attn.o_proj.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def slice_attention_output(layer, new_embedding_dimension):
    # Slice output matrix of the self-attention layer.
    layer.self_attn.o_proj.weight.data = layer.self_attn.out_proj.weight.data[:new_embedding_dimension, :]
    layer.self_attn.o_proj.bias.data = layer.self_attn.out_proj.bias.data[:new_embedding_dimension]
    layer.self_attn.o_proj.out_features = new_embedding_dimension

    layer.attn_shortcut_Q = layer.attn_shortcut_Q[:, :new_embedding_dimension]

def rotate_mlp_input(layer, Q):
    # Rotate the MLP input weights.
    dtype = layer.gate_proj.weight.data.dtype
    W = layer.gate_proj.weight.data.to(device=DEV, dtype=torch.float64)
    layer.gate_proj.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)

def slice_mlp_input(layer, new_embedding_dimension):
    # Slice the MLP input weights.
    layer.post_attention_layernorm.normalized_shape = (new_embedding_dimension,)
    
    layer.gate_proj.weight.data = layer.gate_proj.weight.data[:, :new_embedding_dimension]
    layer.gate_proj.in_features = new_embedding_dimension
    layer.up_proj.weight.data = layer.up_proj.weight.data[:, :new_embedding_dimension]
    layer.up_proj.in_features = new_embedding_dimension
    
    layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:new_embedding_dimension, :]

def rotate_mlp_output(layer, Q):
    dtype = layer.down_proj.weight.data.dtype
    
    W = layer.down_proj.weight.data.to(device=DEV, dtype=torch.float64)
    layer.down_proj.weight.data = torch.matmul(Q.T, W).to(device="cpu", dtype=dtype)
    b = layer.down_proj.bias.data.to(device=DEV, dtype=torch.float64)
    layer.down_proj.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def slice_mlp_output(layer, new_embedding_dimension):
    # Slice the MLP output weights and bias.
    layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:, :new_embedding_dimension]
    layer.down_proj.weight.data = layer.down_proj.weight.data[:new_embedding_dimension, :]
    layer.down_proj.bias.data = layer.down_proj.bias.data[:new_embedding_dimension]
    layer.down_proj.out_features = new_embedding_dimension

def rotate_embeddings(model, Q):
    # Rotate the embeddings.
    dtype = model.model.embed_tokens.weight.data.dtype
    W = model.model.embed_tokens.weight.data.to(device=DEV, dtype=torch.float64)
    model.model.embed_tokens.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)
    
    W = model.model.decoder.embed_positions.weight.data.to(device=DEV, dtype=torch.float64)
    model.model.decoder.embed_positions.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)

def slice_embeddings(model, new_embedding_dimension):
    # Slice the embeddings.
    model.model.embed_tokens.weight.data = model.model.decoder.embed_tokens.weight.data[:, :new_embedding_dimension]
    model.model.embed_positions.weight.data = model.model.decoder.embed_positions.weight.data[:, :new_embedding_dimension]
    
    