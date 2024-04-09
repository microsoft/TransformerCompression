# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Based on the implementation in https://github.com/spcl/QuaRot/

import torch
import tqdm

from slicegpt import utils

from .quant_utils import WeightQuantizer, find_qlayers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def rtn_fwrd(model, dev, args):
    '''
    Round-to-nearest quantization of the model weights.
    '''
    assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            quantizer = WeightQuantizer()
            quantizer.configure(layer_weight_bits, perchannel=True, sym=not (args.w_asym), mse=args.w_clip)
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory()
    return quantizers
