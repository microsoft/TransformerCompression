# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Based on the implementation in https://github.com/spcl/QuaRot/

import torch
import tqdm

from slicegpt import utils
from slicegpt.config import config

from .quant_utils import WeightQuantizer, get_quantizeable_modules

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def apply_weight_rtn_quantization(
    model, bits: int = 16, int8_down_proj: bool = False, asymmetric: bool = True, clip: bool = True
) -> dict[str, WeightQuantizer]:
    '''
    Apply emulation of round-to-nearest quantization of a model's weights. The weights are quantized into
    INT<bits> but are stored in torch.float16.
    '''
    quantizers = {}
    layers = model.model.layers
    for layer_idx, layer in tqdm.tqdm(enumerate(layers), desc="(RtN Quant.) Layers"):
        layer = layer.to(config.device)
        layer_modules = get_quantizeable_modules(layer)
        for name in layer_modules:
            if 'lm_head' in name:
                bits = 16
                continue
            elif int8_down_proj and 'down_proj' in name:
                bits = 8

            quantizer = WeightQuantizer(bits=bits, perchannel=True, sym=not asymmetric, mse=clip)
            W = layer_modules[name].weight.data
            quantizer.calc_params(W)
            layer_modules[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)
            quantizers['model.layers.%d.%s' % (layer_idx, name)] = quantizer.cpu()

        layer = layer.cpu()
        layers[layer_idx] = layer
        utils.cleanup_memory()

    return quantizers
