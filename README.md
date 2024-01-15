# Transformer Compression with SliceGPT

This repository contains the code for the paper [SliceGPT](link made available on publication). 

SliceGPT is a new post-training sparsification scheme that makes transformer networks (including LLMs) smaller by first applying orthogonal transformations to each layer that leave the model unchanged, and then slicing off the least-significant rows and columns (chosen by the eigenvalue decay) of the weight matrices. The model structure is left unchanged, but each weight matrix is replaced by a smaller (dense) weight matrix, reducing the embedding dimension of the model. This results in speedups (without any additional code optimization) and a reduced memory footprint.  

The code is arranged as a package 'slicegpt' in /src, and script to replicate experiments from the paper are in /experiments. To install the SliceGPT package, we recommend

`pip install -e .`


### Running SliceGPT

To run SliceGPT on `microsoft/phi-2`, from the `experiments` folder, run 
```
    python run_slicegpt_perplexity.py \
           --model microsoft/phi-2 \
           --save-dir dir/to/save/sliced_model/in \
           --sparsity 0.25 \
           --no-wandb \
           --device cuda:0 \
           --eval-baseline
```
This will compress the `microsoft/phi-2` model and save the compressed model to the specified directory. Please consult the script for the full set of options.

The experiments folder also contains scripts for 
- [finetuning](./experiments/run_finetuning.py) the compressed model to recover most of the quality lost during compression
- [zero-shot task evaluation](./experiments/run_zero_shot_tasks.py) on a given version of the model (dense, compressed or finetuned)

### Supported models

The following models from Huggingface hub are currently supported
- [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) (commit 834565c)
- [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b)
- [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b)
- [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b)
- [facebook/opt-125m](https://huggingface.co/facebook/opt-125m)
- [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b)
- [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b)
- [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b)
- [facebook/opt-13b](https://huggingface.co/facebook/opt-13b)
- [facebook/opt-30b](https://huggingface.co/facebook/opt-30b)
- [facebook/opt-66b](https://huggingface.co/facebook/opt-66b)

### Extending support to a new model type

The model you wish to support must be available in HuggingFace. To add SliceGPT support for a new model, one needs to: 
- Implement the [ModelAdapter](./src/slicegpt/model_adapter.py) interface for the new model. The ModelAdapter class tells SliceGPT how to interact with the model, an instance of which is stored at self.model. For example, how to access each of the layers of the model.
- Implement the [LayerAdapter](./src/slicegpt/model_adapter.py) interface for the layer. The LayerAdapter class tells SliceGPT how to interact with each layer of the model. For example, how to access the attention and MLP components of the layer, and how to update the arguments to the layer's forward method.
- Implement a compressible layer class that subclasses the decoder layer and provides an adapted `forward()` method to work with the compressed model. The `forward()` method should specify how the skip connection orthogonal matrices are used, depending on whether MLP and attention blocks are sequential (OPT, Llama-2) or parallel (Phi-2). For more details on this, please read [Section 3 in the paper](link made available on publication).
- See [llama_adapter.py](./src/slicegpt/adapters/llama_adapter.py), [opt_adapter.py](./src/slicegpt/adapters/opt_adapter.py) or [phi2_adapter.py](./src/slicegpt/adapters/phi2_adapter.py) for a examples of how to implement these classes.

_Note:_ If the model you wish to support is not available in HuggingFace, you will also need to implement custom model loading and initialization functionality.

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
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
