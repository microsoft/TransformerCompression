import torch
import time
import utils
import argparse
import datautils
import opt_utils
import wandb
from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_layer0_inputs(model, dataloader):
    """
    Returns the inputs to the first layer of the model (after embeddings).
    """
    # Move the relevant parts of the model to device. NB: this won't work from OPT 350m. 
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(DEV)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(DEV)

    layers = model.model.decoder.layers

    inps = []
    attention_masks = []

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_masks.append(kwargs["attention_mask"])
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(DEV))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # Move relevant parts of the model back to cpu (if not already), and clear GPU cache.
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    torch.cuda.empty_cache()

    return torch.cat(inps), attention_masks[-1]

def get_signals(layer, inputs, attention_mask):
    """
    Take the input signals ("activations") for a layer, run the layer forward. 
    Return the output of the layer (not layernormed) and the input to the MLP (pre-layernorm also). 
    """
    mlp_ln_inputs = []
    layer = layer.to(DEV)

    def hook_fn(_, inp, _output):
        if type(inp) == tuple:
            inp = inp[0]
        mlp_ln_inputs.append(inp.cpu())

    hook = layer.final_layer_norm.register_forward_hook(hook_fn)  
    outs = [layer(input.unsqueeze(0), attention_mask=attention_mask)[0] for input in inputs]
    hook.remove()
        
    return torch.cat(mlp_ln_inputs), torch.cat(outs)

def rotate_attention_inputs(layer, Q):
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    dtype = layer.self_attn.q_proj.weight.data.dtype
    W = layer.self_attn.q_proj.weight.data.to(device=DEV, dtype=torch.float64)
    layer.self_attn.q_proj.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)

    W = layer.self_attn.k_proj.weight.data.to(device=DEV, dtype=torch.float64)
    layer.self_attn.k_proj.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)

    W = layer.self_attn.v_proj.weight.data.to(device=DEV, dtype=torch.float64)
    layer.self_attn.v_proj.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)

def slice_attention_inputs(layer, new_embedding_dimension):
    # Slice the  WQ, WK and WV matrices of the self-attention layer.
    layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[:, :new_embedding_dimension]
    layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[:, :new_embedding_dimension]
    layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[:, :new_embedding_dimension]

    layer.attn_shortcut_Q =  layer.attn_shortcut_Q[:new_embedding_dimension, :]

    layer.self_attn_layer_norm.normalized_shape = (new_embedding_dimension,)
    layer.self_attn.q_proj.in_features = new_embedding_dimension
    layer.self_attn.k_proj.in_features = new_embedding_dimension
    layer.self_attn.v_proj.in_features = new_embedding_dimension

def rotate_attention_output(layer, Q):
    # Rotate output matrix of the self-attention layer.
    dtype = layer.self_attn.q_proj.weight.data.dtype
    W = layer.self_attn.out_proj.weight.data.to(device=DEV, dtype=torch.float64)
    layer.self_attn.out_proj.weight.data = torch.matmul(Q.T, W).to(device="cpu", dtype=dtype)
    b = layer.self_attn.out_proj.bias.data.to(device=DEV, dtype=torch.float64)
    layer.self_attn.out_proj.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def slice_attention_output(layer, new_embedding_dimension):
    # Slice output matrix of the self-attention layer.
    layer.self_attn.out_proj.weight.data = layer.self_attn.out_proj.weight.data[:new_embedding_dimension, :]
    layer.self_attn.out_proj.bias.data = layer.self_attn.out_proj.bias.data[:new_embedding_dimension]
    layer.self_attn.out_proj.out_features = new_embedding_dimension

    layer.attn_shortcut_Q = layer.attn_shortcut_Q[:, :new_embedding_dimension]

def rotate_mlp_input(layer, Q):
    # Rotate the MLP input weights.
    dtype = layer.fc1.weight.data.dtype
    W = layer.fc1.weight.data.to(device=DEV, dtype=torch.float64)
    layer.fc1.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)

def slice_mlp_input(layer, new_embedding_dimension):
    # Slice the MLP input weights.
    layer.final_layer_norm.normalized_shape = (new_embedding_dimension,)
    layer.fc1.weight.data = layer.fc1.weight.data[:, :new_embedding_dimension]
    layer.fc1.in_features = new_embedding_dimension
    layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:new_embedding_dimension, :]

def rotate_mlp_output(layer, Q):
    # Rotate the MLP output weights and bias.
    dtype = layer.fc2.weight.data.dtype
    W = layer.fc2.weight.data.to(device=DEV, dtype=torch.float64)
    layer.fc2.weight.data = torch.matmul(Q.T, W).to(device="cpu", dtype=dtype)
    b = layer.fc2.bias.data.to(device=DEV, dtype=torch.float64)
    layer.fc2.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def slice_mlp_output(layer, new_embedding_dimension):
    # Slice the MLP output weights and bias.
    layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:, :new_embedding_dimension]
    layer.fc2.weight.data = layer.fc2.weight.data[:new_embedding_dimension, :]
    layer.fc2.bias.data = layer.fc2.bias.data[:new_embedding_dimension]
    layer.fc2.out_features = new_embedding_dimension

def rotate_embeddings(model, Q):
    # Rotate the embeddings.
    dtype = model.model.decoder.embed_tokens.weight.data.dtype
    W = model.model.decoder.embed_tokens.weight.data.to(device=DEV, dtype=torch.float64)
    model.model.decoder.embed_tokens.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)
    
    W = model.model.decoder.embed_positions.weight.data.to(device=DEV, dtype=torch.float64)
    model.model.decoder.embed_positions.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)

def slice_embeddings(model, new_embedding_dimension):
    # Slice the embeddings.
    model.model.decoder.embed_tokens.weight.data = model.model.decoder.embed_tokens.weight.data[:, :new_embedding_dimension]
    model.model.decoder.embed_positions.weight.data = model.model.decoder.embed_positions.weight.data[:, :new_embedding_dimension]
    
def rotate_head(model, Q):
    # Rotate the head.
    dtype = model.lm_head.weight.data.dtype
    W = model.lm_head.weight.data.to(device=DEV, dtype=torch.float64)
    model.lm_head.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)
    
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
    # List of layers to rotate.
    layers = model.model.decoder.layers

    # Get the input of the first layer norm and calculate the Q_1
    inps, attention_mask = get_layer0_inputs(model, dataloader)
    _, Q = utils.pca_calc(inps.reshape(-1, model.config.hidden_size))
    Q = Q.to(device=DEV)

    rotate_embeddings(model, Q)
    slice_embeddings(model, new_embedding_dimension)

    # rotate and slice inputs
    inps = torch.matmul(inps, Q.to(dtype=dtype))[:, :, :new_embedding_dimension]

    print("(Rotate and slice) layers:", end=" ", flush=True)
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
    layers = model.model.decoder.layers

    # Get the input of the first layer norm and calculate the Q_1
    inps, attention_mask = get_layer0_inputs(model, dataloader)
    _, Q_1 = utils.pca_calc(inps.reshape(-1, model.config.hidden_size))
    Q_1 = Q_1.to(device=DEV)

    # Rotate the embeddings.
    rotate_embeddings(model, Q_1)

    # Clear GPU cache.
    torch.cuda.empty_cache()

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
    layers = model.model.decoder.layers
    
    for i, layer in enumerate(layers):
        
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
             
    model.model.decoder.final_layer_norm.normalized_shape = (new_embedding_dimension,)
    
    if do_slice_head:
        slice_head(model, new_embedding_dimension)


def save_rotated_model(model, model_name, save_dir):
    """
    Saves the rotated model to the specified directory.

    Args:
        model: the rotated model
        model_name: the name of the model
        save_dir: the directory to save the model to
    """
    model = model.cpu()
    model_name = model_name.replace("/", "_")
    save_path = save_dir + model_name + ".pt"
    print(f"Saving the rotated model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Model saved.")

def load_rotated_model(model_name, load_dir):
    """
    TODO
    """
    model = opt_utils.get_opt(model_name)
    model = model.cpu() # necessary?
    model_name = model_name.replace("/", "_")
    load_path = load_dir + model_name + ".pt"
    print(f"Loading the rotated model from {load_path}...")
    model.load_state_dict(torch.load(load_path, map_location=torch.device("cpu")))
    print("Model loaded.")
    return model

if __name__ == "__main__":
    model_name = "facebook/opt-125m"
    model = opt_utils.get_opt(model_name)

    dataloader, testloader = datautils.get_loaders(
            "wikitext2", seed=42, model=model_name, seqlen=model.seqlen
        )
    
    opt_utils.replace_opt_modules(model, model.config)
    opt_utils.fuse_opt_modules(model)
    print()
    model = model.cpu()
    
    dataset_ppl = opt_utils.opt_eval(model, testloader, DEV)
    print('orig', dataset_ppl)
    
    rotate_and_slice_opt(model, dataloader, int(0.8 * model.config.hidden_size))
    dataset_ppl = opt_utils.opt_eval(model, testloader, DEV)
    print('rotate and slice', dataset_ppl)

    """
    rotate_opt(model, dataloader)
    dataset_ppl = opt_utils.opt_eval(model, testloader, DEV)
    print('rotate', dataset_ppl)
    """