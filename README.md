# Transformer Compression with SliceGPT

This repository contains the code for the paper [SliceGPT](https://arxiv.org/abs/2401.15024) (ICLR'24). Also discussed on [Hugging Face](https://huggingface.co/papers/2401.15024). 

SliceGPT is a new post-training sparsification scheme that makes transformer networks (including LLMs) smaller by 
first applying orthogonal transformations to each transformer layer that leave the model unchanged, and then slicing off the 
least-significant rows and columns (chosen by the eigenvalue decay) of the weight matrices. The model structure is 
left unchanged, but each weight matrix is replaced by a smaller (dense) weight matrix, reducing the embedding dimension 
of the model. This results in speedups (without any additional code optimization) and a reduced memory footprint.  

The code is arranged as a package `slicegpt` in `/src`, and scripts to replicate experiments from the paper are in 
`/experiments`. To install the `slicegpt` package, we recommend

```
    pip install -e .[experiment]
```

## Running SliceGPT

To run SliceGPT on `microsoft/phi-2`, from the `experiments` folder, run 
```
    python run_slicegpt.py \
           --model microsoft/phi-2 \
           --save-dir dir/to/save/sliced_model/in \
           --sparsity 0.25 \
           --device cuda:0 \
           --eval-baseline \
           --no-wandb
```

This will compress the `microsoft/phi-2` model and save the compressed model to the specified directory. Please consult 
the script for the full set of options.

_Note:_ For models that require Hugging Face authentication, set the `--hf-token` argument 
manually or using a key vault. Alternatively, set the environment variable `HF_TOKEN`.

### Recovery fine-tuning

To install additional dependencies required for post-slicing recovery fine-tuning (RFT):

```
    pip install -e .[experiment,finetune]
```

The following replicates the experiments in the paper (LoRA hyperparams valid for all Llama-2 and Phi-2 models): 
```
    python run_finetuning.py \
           --model microsoft/phi-2 \
           --sliced-model-path path/to/sliced \
           --save-dir dir/to/save/finetuned_model/in \
           --sparsity 0.25 \
           --device cuda:0 \
           --ppl-eval-dataset alpaca \
           --finetune-dataset alpaca \
           --finetune-train-nsamples 8000 \
           --finetune-train-seqlen 1024 \
           --finetune-train-batch-size 3 \
           --lora-alpha 10 \
           --lora-r 32 \
           --lora-dropout 0.05 \
           --lora-target-option attn_head_and_mlp \
           --eval-steps 16 \
           --save-steps 16 \
           --no-wandb
```

Notes: 
- The script [`bo_finetuning.py`](./experiments/bo_finetuning.py) can be used to run Bayesian optimization over the RFT hyperparameters.
- To run finetuning on the original model, specify `--model-path` instead of `--sliced-model-path`. 
- `sparsity` must be specified when specifying `sliced-model-path` to avoid default sparsity being used

### Evaluation using the [LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness) 
```
    python run_lm_eval.py \
           --model microsoft/phi-2 \
           --sliced-model-path path/to/sliced \
           --sparsity 0.25 \
           --tasks piqa \
           --no-wandb
```

Notes: 
- To run lm-eval on the original model, specify `--model-path` instead of `--sliced-model-path`. 
- `sparsity` must be specified when specifying `sliced-model-path` to avoid default sparsity being used

## Supported models

The following models from Hugging Face hub are currently supported
- [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b)
- [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b)
- [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b)
- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
- [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
- [facebook/opt-125m](https://huggingface.co/facebook/opt-125m)
- [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b)
- [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b)
- [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b)
- [facebook/opt-13b](https://huggingface.co/facebook/opt-13b)
- [facebook/opt-30b](https://huggingface.co/facebook/opt-30b)
- [facebook/opt-66b](https://huggingface.co/facebook/opt-66b)

## Extending support to a new model type

The model you wish to support must be in Hugging Face Hub format. The model files can be downloaded from 
Hugging Face Hub by supplying `--model` argument, or accessed from local storage by using the `--model` and 
`--model-path` argument. To add SliceGPT support for a new model, one needs to implement a new model adapter 
and update `hf_utils.get_model_and_tokenizer` before slicing the new model.

### Implementing a new model adapter
- Implement the [ModelAdapter](./src/slicegpt/model_adapter.py) interface for the new model. The ModelAdapter class tells SliceGPT 
  how to interact with the model, an instance of which is stored at `self.model`. For example, 
  how to access each of the layers of the model.
- Implement the [LayerAdapter](./src/slicegpt/model_adapter.py) interface for the transformer layers. 
  The LayerAdapter class tells SliceGPT how to interact 
  with each transformer layer of the model, an instance of which is stored at `self.layer`. 
  For example, how to access the attention and MLP components of the transformer layer, and 
  how to update the arguments to the transformer layer's forward method.
- Implement a compressed transformer layer class that subclasses the transformer layer. 
  This class should also  provide an adapted `forward()` method to work with the compressed model. 
  This method should specify how the skip connection orthogonal matrices are used, depending on 
  whether MLP and attention blocks are sequential ([OPT](./src/slicegpt/adapters/opt_adapter.py), 
  [Llama-2/Llama-3](./src/slicegpt/adapters/llama_adapter.py)) or parallel 
  ([Phi-2](./src/slicegpt/adapters/phi2_adapter.py)). The `self.*_shortcut_Q` matrices are attached to the modules during
  slicing and are available in `forward()`. If the skip connection does not need modification, these matrices will be None, 
  and the `forward()` method can follow the original workflow. For more details on this, 
  please read Section 3 in [the paper](https://arxiv.org/abs/2401.15024).

Example: [llama_adapter.py](./src/slicegpt/adapters/llama_adapter.py)

### Using a new model adapter to slice a model
Once a model adapter is implemented, compressing the model involves three conceptual steps:
  - Replace modules with compressed equivalents (via `slicegpt.layernorm_fusion.replace_layers`)
  - Fuse layer norms and add rotations to skip connections (via `slicegpt.layernorm_fusion.fuse_modules`)
  - Rotate the inputs and slice the layers (via `slicegpt.rotate.rotate_and_slice`)

Example: [run_slicegpt.py](./experiments/run_slicegpt.py)

_Note:_ If the model you wish to support is not available in Hugging Face, you will also need to implement 
custom model loading and initialization functionality.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply 
Microsoft sponsorship.

Any use of third-party trademarks or logos are subject to those third-party's policies.

