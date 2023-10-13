import torch
from . import utils

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    get_pre_head_layernorm,
    get_layer0_inputs,
    get_signals
)

def rotate_attention_inputs(layer, Q):
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in get_attention_inputs(layer):
        dtype = W.weight.dtype
        W_ = W.weight.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        
def slice_attention_inputs(layer, new_embedding_dimension):
    # Slice the  WQ, WK and WV matrices of the self-attention layer.
    for W in get_attention_inputs(layer):
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

    layer.attn_shortcut_Q = layer.attn_shortcut_Q[:new_embedding_dimension, :]

    get_first_layernorm(layer).normalized_shape = (new_embedding_dimension,)

def rotate_attention_output(layer, Q):
    # Rotate output matrix of the self-attention layer.
    W = get_attention_output(layer)
    
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def slice_attention_output(layer, new_embedding_dimension):
    # Slice output matrix of the self-attention layer.
    W = get_attention_output(layer)
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension

    layer.attn_shortcut_Q = layer.attn_shortcut_Q[:, :new_embedding_dimension]

def rotate_mlp_input(layer, Q):
    # Rotate the MLP input weights.
    for W in get_mlp_inputs(layer):
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def slice_mlp_input(layer, new_embedding_dimension):
    # Slice the MLP input weights.
    for W in get_mlp_inputs(layer):
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension
        
    # slice shortcut
    layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:new_embedding_dimension, :]
    
    # modify layernorm
    get_second_layernorm(layer).normalized_shape = (new_embedding_dimension,)

def rotate_mlp_output(layer, Q):
    # Rotate the MLP output weights and bias.
    W = get_mlp_output(layer)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def slice_mlp_output(layer, new_embedding_dimension):
    # Slice the MLP output weights and bias.
    W = get_mlp_output(layer)
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension
    
    layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:, :new_embedding_dimension]

def rotate_embeddings(model, Q):
    # Rotate the embeddings.
    for W in get_embeddings(model):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    torch.cuda.empty_cache()

def slice_embeddings(model, new_embedding_dimension):
    # Slice the embeddings.
    for W in get_embeddings(model):
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
    
def rotate_head(model, Q):
    # Rotate the head.
    W = get_lm_head(model)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    
def slice_head(model, new_embedding_dimension):
    # Slice the head.
    model.lm_head.weight.data = model.lm_head.weight.data[:, :new_embedding_dimension]
    model.lm_head.in_features = new_embedding_dimension
    

@torch.no_grad()
def rotate_and_slice_opt(model, dataloader, new_embedding_dimension, do_slice_head=False):
    """
    Rotate and slice an OPT model, with interleaved slicing and PCA calculations
    """
    dtype = next(iter(model.parameters())).dtype # Get the dtype of the model.

    # Get the input of the first layer norm and calculate the Q_1
    inps, attention_mask = get_layer0_inputs(model, dataloader)
    _, Q = utils.pca_calc(inps.reshape(-1, model.config.hidden_size))
    Q = Q.to(device=DEV)

    rotate_embeddings(model, Q)
    slice_embeddings(model, new_embedding_dimension)

    # rotate and slice inputs
    inps = torch.matmul(inps, Q.to(dtype=dtype))[:, :, :new_embedding_dimension]

    print("(Rotate and slice) layers:", end=" ", flush=True)
    layers = get_layers(model)
    for i, layer in enumerate(layers):
        print(f" {i}", end="", flush=True)
        
        layer.attn_shortcut_Q = Q.T.clone().to(dtype=dtype)
        
        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer, Q)
        slice_attention_inputs(layer, new_embedding_dimension)

        # get signal between attention and mlp, rotate and slice
        mlp_ln_inputs, _ = get_signals(layer, inps, attention_mask)
        _, Q = utils.pca_calc(mlp_ln_inputs.reshape(-1, mlp_ln_inputs.shape[-1]))
        Q = Q.to(device=DEV, dtype=torch.float64)
        
        layer.attn_shortcut_Q = torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype))
        rotate_attention_output(layer, Q)
        slice_attention_output(layer, new_embedding_dimension)
        
        layer.mlp_shortcut_Q = Q.T.clone().to(dtype=dtype)
        rotate_mlp_input(layer, Q)
        slice_mlp_input(layer, new_embedding_dimension)
        
        # Clear GPU cache.
        torch.cuda.empty_cache()
        
        #now compute the outputs of the layer with slicing between Attention and mlp.
        _, outputs = get_signals(layer, inps, attention_mask)
        _, Q = utils.pca_calc(outputs.reshape(-1, outputs.shape[-1]))
        
        layer.mlp_shortcut_Q = torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype))

        #optionally slice the mlp/head connection in the last layer
        dim = new_embedding_dimension
        if layer is layers[-1]:
            if not do_slice_head:
                dim = model.config.hidden_size
        
        rotate_mlp_output(layer, Q)
        slice_mlp_output(layer, dim)

        inps = torch.matmul(outputs, Q.to(dtype=dtype))[:, :, :dim]
        
        layer = layer.to('cpu')
        
        # Clear GPU cache.
        torch.cuda.empty_cache()
    
    # rotate and slice head
    rotate_head(model, Q)
    if do_slice_head:
        slice_head(model, new_embedding_dimension)


@torch.no_grad()
def rotate_opt(model, dataloader):
    """
    Rotate an OPT model.
    """
    dtype = next(iter(model.parameters())).dtype # Get the dtype of the model.

    # List of layers to rotate.
    layers = get_layers(model)

    # Get the input of the first layer norm and calculate the Q_1
    inps, attention_mask = get_layer0_inputs(model, dataloader)
    _, Q_1 = utils.pca_calc(inps.reshape(-1, model.config.hidden_size))
    Q_1 = Q_1.to(device=DEV)

    # Rotate the embeddings.
    rotate_embeddings(model, Q_1)

    # Rotate the rest of the model.
    print("(Rotate) layers:", end=" ", flush=True)
    for i, layer in enumerate(layers):
        print(f" {i}", end="", flush=True)

        # Extract the inputs and outputs of the second layernorm input and calculate the Q_3
        mlp_ln_inputs, outs = get_signals(layer, inps, attention_mask)
        _, Q_3 = utils.pca_calc(mlp_ln_inputs.reshape(-1, mlp_ln_inputs.shape[-1]))
        Q_3 = Q_3.to(device=DEV)
        _, Q_5 = utils.pca_calc(outs.reshape(-1, outs.shape[-1]))
        Q_5 = Q_5.to(device=DEV)

        # Rotate the Q, K and V matrices of the self-attention layer.
        rotate_attention_inputs(layer, Q_1)

        # Set the shortcut rotation matrix of the self-attention layer.
        layer.attn_shortcut_Q = torch.matmul(Q_1.clone().T, Q_3.clone()).to(device="cpu", dtype=dtype)

        # Rotate the Attention output matrix
        rotate_attention_output(layer, Q_3)

        # Rotate the MLP input
        rotate_mlp_input(layer, Q_3)

        # Set the shortcut rotation matrix of the MLP.
        layer.mlp_shortcut_Q = torch.matmul(Q_3.clone().T, Q_5.clone()).to(device="cpu", dtype=dtype)

        # Rotate MLP output
        rotate_mlp_output(layer, Q_5)
        
        # Clear GPU cache.
        torch.cuda.empty_cache()

        inps = outs  # The inputs to the next layer are the outputs from this one!
        Q_1 = Q_5  # first rotation in the next layer is the last one in this...
        
    rotate_head(model, Q_5)
    
    print(" Done rotating!")
    

def slice_rotated_OPT_model(model, new_embedding_dimension, do_slice_head=False):
    
    # slice embeddings
    slice_embeddings(model, new_embedding_dimension)
    
    # List of layers to sice.
    layers = get_layers(model)
    
    for layer in layers:
        
        slice_attention_inputs(layer, new_embedding_dimension)
        slice_attention_output(layer, new_embedding_dimension)
                
        # Slice attention shortcut matrix
        layer.attn_shortcut_Q =  layer.attn_shortcut_Q[:new_embedding_dimension, :new_embedding_dimension]
        
        slice_mlp_input(layer, new_embedding_dimension)
        
        #optionally slice the mlp/head connection in the last layer
        dim = new_embedding_dimension
        if layer is layers[-1]:
            if not slice_head:
                dim = model.config.hidden_size
                
        slice_mlp_output(layer, dim)
        layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:new_embedding_dimension, :dim]
    
    if do_slice_head:
        get_pre_head_layernorm(model).normalized_shape = (new_embedding_dimension,)
        slice_head(model, new_embedding_dimension)