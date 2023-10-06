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
        # TODO do we really want to do the reshaeping here?
        if type(inp) == tuple:
            inp = inp[0]
        mlp_ln_inputs.append(inp.cpu())

    hook = layer.final_layer_norm.register_forward_hook(hook_fn)  
    outs = [layer(input.unsqueeze(0), attention_mask=attention_mask)[0] for input in inputs]
    hook.remove()
        
    return torch.cat(mlp_ln_inputs), torch.cat(outs)


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

    # Rotate the embeddings.
    W = model.model.decoder.embed_tokens.weight.data.to(device=DEV, dtype=torch.float64)
    Q_1 = Q_1.to(device=DEV, dtype=torch.float64)
    model.model.decoder.embed_tokens.weight.data = torch.matmul(W, Q_1).to(device="cpu", dtype=dtype)
    
    W = model.model.decoder.embed_positions.weight.data.to(device=DEV, dtype=torch.float64)
    model.model.decoder.embed_positions.weight.data = torch.matmul(W, Q_1).to(device="cpu", dtype=dtype)

    # Clear GPU cache.
    torch.cuda.empty_cache()

    # Rotate the rest of the model.
    print(f"(Rotate) layers:", end=" ", flush=True)
    for i, layer in enumerate(layers):
        print(f" {i}", end="", flush=True)
        if i > 0:
            Q_1 = Q_5.clone()

        # Rotate the Q, K and V matrices of the self-attention layer.
        W = layer.self_attn.q_proj.weight.data.to(device=DEV, dtype=torch.float64)
        layer.self_attn.q_proj.weight.data = torch.matmul(W, Q_1).to(dtype)

        W = layer.self_attn.k_proj.weight.data.to(device=DEV, dtype=torch.float64)
        layer.self_attn.k_proj.weight.data = torch.matmul(W, Q_1).to(dtype)

        W = layer.self_attn.v_proj.weight.data.to(device=DEV, dtype=torch.float64)
        layer.self_attn.v_proj.weight.data = torch.matmul(W, Q_1).to(dtype)

        # Extract the inputs and outputs of the second layernorm input and calculate the Q_3
        mlp_ln_inputs, outs = get_signals(layer, inps, attention_mask)
        _, Q_3 = utils.pca_calc(mlp_ln_inputs.reshape(-1, mlp_ln_inputs.shape[-1]))
        _, Q_5 = utils.pca_calc(outs.reshape(-1, outs.shape[-1]))

        # Set the shortcut rotation matrix of the self-attention layer.
        layer.attn_shortcut_Q = torch.matmul(Q_1.clone().T, Q_3.clone()).to(dtype)
        layer.mlp_shortcut_Q = torch.matmul(Q_3.clone().T, Q_5.clone()).to(dtype)

        # Rotate the Attention output matrix
        W = layer.self_attn.out_proj.weight.data.to(device=DEV, dtype=torch.float64)
        layer.self_attn.out_proj.weight.data = torch.matmul(Q_3.T, W).to(device="cpu", dtype=dtype)
        b = layer.self_attn.out_proj.bias.data.to(device=DEV, dtype=torch.float64)
        layer.self_attn.out_proj.bias.data = torch.matmul(Q_3.T, b).to(device="cpu", dtype=dtype)

        # Rotate the MLP input
        W = layer.fc1.weight.data.to(device=DEV, dtype=torch.float64)
        layer.fc1.weight.data = torch.matmul(W, Q_3).to(device="cpu", dtype=dtype)

        # Rotate MLP output
        W = layer.fc2.weight.data.to(device=DEV, dtype=torch.float64)
        layer.fc2.weight.data = torch.matmul(Q_5.T, W).to(device="cpu", dtype=dtype)
        b = layer.fc2.bias.data.to(device=DEV, dtype=torch.float64)
        layer.fc2.bias.data = torch.matmul(Q_5.T, b).to(device="cpu", dtype=dtype)
        
        # Clear GPU cache.
        torch.cuda.empty_cache()

        # The inputs to the next layer are the outputs from this one!
        inps = outs
        
    # rotate head
    W = model.lm_head.weight.data.to(device=DEV, dtype=torch.float64)
    model.lm_head.weight.data = torch.matmul(W, Q_5).to(device="cpu", dtype=dtype)
    
    print(" Done rotating!")
    

def slice_rotated_OPT_model(model, new_embedding_dimension, slice_head=False):
    
    # slice embeddings
    model.model.decoder.embed_tokens.weight.data = model.model.decoder.embed_tokens.weight.data[:, :new_embedding_dimension]
    model.model.decoder.embed_positions.weight.data = model.model.decoder.embed_positions.weight.data[:, :new_embedding_dimension]
    
    # List of layers to sice.
    layers = model.model.decoder.layers
    for i, layer in enumerate(layers):
        
        # slice attention
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[:, :new_embedding_dimension]
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[:, :new_embedding_dimension]
        layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[:, :new_embedding_dimension]
        layer.self_attn.out_proj.weight.data = layer.self_attn.out_proj.weight.data[:new_embedding_dimension, :]
        layer.self_attn.out_proj.bias.data = layer.self_attn.out_proj.bias.data[:new_embedding_dimension]
        
        # set attention shapes
        layer.self_attn_layer_norm.normalized_shape = (new_embedding_dimension,)
        layer.self_attn.q_proj.in_features = new_embedding_dimension
        layer.self_attn.k_proj.in_features = new_embedding_dimension
        layer.self_attn.v_proj.in_features = new_embedding_dimension
        layer.self_attn.out_proj.out_features = new_embedding_dimension
                
        # Slice attention shortcut matrix
        layer.attn_shortcut_Q =  layer.attn_shortcut_Q[:new_embedding_dimension, :new_embedding_dimension]
        
        # slice the MLP
        layer.final_layer_norm.normalized_shape = (new_embedding_dimension,)
        layer.fc1.weight.data = layer.fc1.weight.data[:, :new_embedding_dimension]
        layer.fc1.in_features = new_embedding_dimension
        
        # slice output weights of mlp (but perhaps not in the last layer)
        if layer is layers[-1] and (not slice_head):
            layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:new_embedding_dimension, :]
        else:
            layer.fc2.weight.data = layer.fc2.weight.data[:new_embedding_dimension, :]
            layer.fc2.bias.data = layer.fc2.bias.data[:new_embedding_dimension]
            layer.fc2.out_features = new_embedding_dimension
            layer.mlp_shortcut_Q = layer.mlp_shortcut_Q[:new_embedding_dimension, :new_embedding_dimension]
             
    model.model.decoder.final_layer_norm.normalized_shape = (new_embedding_dimension,)
    
    if slice_head:
        model.lm_head.weight.data = model.lm_head.weight.data[:, :new_embedding_dimension]
        model.lm_head.in_features = new_embedding_dimension
        
    


@opt_utils.do_not_initialize
def slice_OPT_model(model, args):
    """
    TODO
    """
    
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


def save_rotated_model(model, model_name, save_dir):
    model = model.cpu()
    model_name = model_name.replace("/", "_")
    save_path = save_dir + model_name + ".pt"
    print(f"Saving the rotated model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Model saved.")

def load_rotated_model(model_name, load_dir):
    model = opt_utils.get_opt(model_name)
    model = model.cpu() # necessary?
    model_name = model_name.replace("/", "_")
    load_path = load_dir + model_name + ".pt"
    print(f"Loading the rotated model from {load_path}...")
    model.load_state_dict(torch.load(load_path, map_location=torch.device("cpu")))
    print("Model loaded.")
    return model

if __name__ == "__main__":
    model = opt_utils.get_opt("facebook/opt-125m")

    
    dataloader, testloader = datautils.get_loaders(
            "wikitext2", seed=42, model="facebook/opt-125m", seqlen=model.seqlen
        )
    
    opt_utils.replace_opt_modules(model, model.config)
    opt_utils.fuse_opt_modules(model)
    print()
    model = model.cpu()
    
    dataset_ppl = opt_utils.opt_eval(model, testloader, DEV)
    print('orig', dataset_ppl)
    
    rotate_opt(model, dataloader)
    dataset_ppl = opt_utils.opt_eval(model, testloader, DEV)
    print('rotate', dataset_ppl)
    
    slice_rotated_OPT_model(model, int(0.8 * model.config.hidden_size))
    dataset_ppl = opt_utils.opt_eval(model, testloader, DEV)
    print('rotate and slice', dataset_ppl)